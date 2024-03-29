"""
Script to evaluate importance of the input variables.

It relies on adding random normal noise to the input variables and evaluating
the model performance [1]_. The idea is that if large degradation of
performance is occurred when noise is added to a given input then that input
was important.

References
----------
.. [1] J.D. Olden et al. / Ecological Modeling 178 (2004) 389-397
"""

import argparse
import logging
import os

import pandas as pd

from vlndata.dataset.transform import NoiseTransform
from vlndata.dataset           import DatasetTransform

from vlne.eval.eval           import eval_model
from vlne.plot.profile        import plot_profile
from vlne.presets             import PRESETS_EVAL
from vlne.utils               import setup_logging
from vlne.utils.eval          import standard_eval_prologue, parse_binning
from vlne.utils.parsers       import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from vlne.data.data_generator.keras_sequence import KerasSequence
from vlne.plot.plot_spec import PlotSpec

def make_hist_specs(cmdargs, preset):
    binning = parse_binning(cmdargs, suffix = '_x')

    return {
        label : PlotSpec(**binning)
            for (label, name) in preset.name_map.items()
    }

def add_energy_resolution_parser(parser):
    parser.add_argument(
        '--fit_margin',
        help    = 'Fraction of resolution peak to fit gaussian',
        default = 0.5,
        dest    = 'fit_margin',
        type    = float,
    )

    add_hist_binning_parser(
        parser,
        default_range_lo = -1,
        default_range_hi = 1,
        default_bins     = 100,
    )

def parse_cmdargs():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        "Make input importance(based on output sensitivity to random"
        " input perturbations) profiles"
    )

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    parser.add_argument(
        '--smear',
        help    = 'Smearing Value',
        default = 0.5,
        dest    = 'smear',
        type    = float,
    )

    parser.add_argument(
        '-a', '--annotate',
        action = 'store_true',
        dest   = 'annotate',
        help   = 'Show y-values for each point'
    )

    add_energy_resolution_parser(parser)

    return parser.parse_args()

def prologue(cmdargs):
    """Load dataset, model and initialize output directory"""
    dgen, args, model, outdir, plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    plotdir = os.path.join(
        outdir, f'input_importance_perturb_{cmdargs.smear:.3e}'
    )
    os.makedirs(plotdir, exist_ok = True)

    return (args, model, dgen, plotdir, preset)

def get_stats_for_energy(stat_list, energy):
    """Extract stats for a given energy type"""
    return pd.DataFrame([ x[energy] for x in stat_list ])

def plot_vars_profile(
    var_list, stat_list, label_x, annotate, preset, plotdir, ext,
    stats_to_plot = ( 'rms', 'sigma' ),
):
    for energy in stat_list[0].keys():
        for stat in stats_to_plot:

            label_y = '%s(%s [%s])' % (
                stat, preset.name_map[energy], preset.units_map[energy]
            )

            fname  = f"{stat}_{energy}_vs_{label_x}"
            fname += "_ann" if annotate else ''
            fname  = os.path.join(plotdir, fname)

            stats = get_stats_for_energy(stat_list, energy)

            plot_profile(
                var_list, stats, stat,
                base_stats  = None,
                label_x     = label_x,
                label_y     = label_y,
                sort_type   = 'y',
                annotate    = annotate,
                categorical = True,
                fname       = fname,
                ext         = ext
            )

def scalar_perturb_generator(args, dgen, group_name, smear):
    # pylint: disable=protected-access
    dset = dgen.dataset

    for vname in args.data.input_groups_scalar[group_name]:
        smearning    = NoiseTransform(
            { 'name' : 'gaussian', 'mu' : 0, 'sigma' : smear },
            correlated    = False,
            relative      = True,
            scalar_groups = { group_name : [ vname, ] },
            vlarr_groups  = None,
        )

        dgen._dgen._dataset = DatasetTransform(dset, [ smearning, ])
        yield (vname, dgen)

    dgen._dgen._dataset = dset

def vlarr_perturb_generator(args, dgen, group_name, smear):
    # pylint: disable=protected-access
    dset = dgen.dataset

    for vname in args.data.input_groups_vlarr[group_name]:
        smearning    = NoiseTransform(
            { 'name' : 'gaussian', 'mu' : 0, 'sigma' : smear },
            correlated    = False,
            relative      = True,
            scalar_groups = None,
            vlarr_groups  = { group_name : [ vname, ] },
        )
        dgen._dgen._dataset = DatasetTransform(dset, [ smearning, ])
        yield (vname, dgen)

    dgen._dgen._dataset = dset

def save_stats(var_list, stat_list, label, plotdir):
    """Save input importance stats vs input variable"""
    result = []

    for idx,var in enumerate(var_list):
        for k,v in stat_list[idx].items():
            result.append({ 'var' : var, 'energy' : k, **v })

    df = pd.DataFrame.from_records(result, index = ('var', 'energy'))
    df.to_csv('%s/stats_%s.csv' % (plotdir, label))

def make_perturb_profile(
    smeared_var_generator, var_list, stat_list, args, model, hist_specs,
    preset, plotdir, label, cmdargs
):
    """
    Evaluate performance for generators yielded by `smeared_var_generator`
    """
    if smeared_var_generator is None:
        return

    var_list   = var_list[:]
    stat_list  = stat_list[:]

    for (vname, dgen) in smeared_var_generator:
        logging.info("Evaluating '%s' var...", vname)

        stats, _ = eval_model(
            args, KerasSequence(dgen), model, hist_specs, cmdargs.fit_margin
        )

        var_list .append(f"{vname} : {cmdargs.smear}")
        stat_list.append(stats)

    plot_vars_profile(
        var_list, stat_list, label, cmdargs.annotate, preset, plotdir,
        cmdargs.ext
    )

    save_stats(var_list, stat_list, label, plotdir)

def main():
    setup_logging()

    cmdargs = parse_cmdargs()
    args, model, dgen, plotdir, preset = prologue(cmdargs)

    hist_specs = make_hist_specs(cmdargs, preset)
    var_list   = [ 'none' ]
    stat_list  = [ eval_model(args, dgen, model, hist_specs)[0] ]

    scalar_groups = args.data.input_groups_scalar or []
    vlarr_groups  = args.data.input_groups_vlarr or []

    for group_name in scalar_groups:
        make_perturb_profile(
            scalar_perturb_generator(args, dgen, group_name, cmdargs.smear),
            var_list, stat_list, args, model, hist_specs, preset, plotdir,
            group_name, cmdargs
        )

    for group_name in vlarr_groups:
        make_perturb_profile(
            vlarr_perturb_generator(args, dgen, group_name, cmdargs.smear),
            var_list, stat_list, args, model, hist_specs, preset, plotdir,
            group_name, cmdargs
        )

if __name__ == '__main__':
    main()

