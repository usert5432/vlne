"""Make Relative Energy Resolution Histograms"""

import argparse
import os

from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.plot.aux      import plot_rel_res_vs_true
from lstm_ee.utils.eval    import standard_eval_prologue, parse_binning
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from lstm_ee.eval.predict  import (
    get_true_energies, get_base_energies, predict_energies
)
from lstm_ee.plot.plot_spec import PlotSpec

def make_plot_spec(name, unit, **kwargs):
    return PlotSpec(
        title   = None,
        label_x = f'True {name} Energy [{unit}]',
        label_y = f'(Reco - True) / True {name} Energy',
        minor = True,
        grid  = True,
        **kwargs
    )

def make_plot_specs(cmdargs, preset):
    binning_x = parse_binning(cmdargs, suffix = '_x')
    binning_y = {
        'bins_y'  : binning_x['bins_x'],
        'range_y' : (-cmdargs.ymargin, cmdargs.ymargin),
    }

    return {
        label : make_plot_spec(
            name, preset.units_map[label], **binning_x, **binning_y
        ) for (label, name) in preset.name_map.items()
    }

def parse_cmdargs():
    parser = argparse.ArgumentParser("Make Auxiliary plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    add_hist_binning_parser(
        parser,
        default_range_lo = 0,
        default_range_hi = 5,
        default_bins     = 50,
    )

    parser.add_argument(
        '--ymargin',
        help    = 'y-range margin. range_y = (-ymargin, +ymargin)',
        default = 0.2,
        dest    = 'ymargin',
        type    = float,
    )

    return parser.parse_args()

def make_relative_resolution_plots(
    pred_model_dict, pred_base_dict, true_dict, weights, relres_specs,
    plotdir, ext
):
    """Make plots of 2D histograms of relative energy resolution vs TrueE"""
    plot_rel_res_vs_true(
        pred_model_dict, true_dict, weights,
        relres_specs, os.path.join(plotdir, "plot_model_rel_res_vs_true"), ext
    )

    plot_rel_res_vs_true(
        pred_base_dict, true_dict, weights,
        relres_specs, os.path.join(plotdir, "plot_base_rel_res_vs_true"), ext
    )

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    relres_specs = make_plot_specs(cmdargs, preset)

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict       = get_true_energies(dgen)
    pred_base_dict  = get_base_energies(dgen, preset.base_map)

    make_relative_resolution_plots(
        pred_model_dict, pred_base_dict, true_dict, dgen.weights,
        relres_specs, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

