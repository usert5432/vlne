"""
Script to plot true energy distributions.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot  import (
    plot_rhist1d, plot_rhist1d_error, save_fig, remove_bottom_margin
)
from cafplot.rhist import RHist1D

from vlne.eval.predict   import get_true_energies
from vlne.eval.funcs     import get_weights
from vlne.presets        import PRESETS_EVAL
from vlne.utils.eval     import standard_eval_prologue, parse_binning
from vlne.utils.log      import setup_logging
from vlne.utils.parsers  import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from vlne.plot.plot_spec import PlotSpec

def make_hist_specs(cmdargs, preset):
    binning = parse_binning(cmdargs, suffix = '_x')

    return {
        label : PlotSpec(
            title   = None,
            label_x = f'True {name} Energy [{preset.units_map[label]}]',
            label_y = 'Events',
            **binning
        )
        for (label, name) in preset.name_map.items()
    }

def parse_cmdargs():
    parser = argparse.ArgumentParser("Make True Energy Hist plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    add_hist_binning_parser(
        parser,
        default_range_lo = 0,
        default_range_hi = 5,
        default_bins     = 100,
    )

    return parser.parse_args()

def plot_single_energy_hist(data, weights, name, spec, log_scale = False):
    """Plot single energy distribution."""

    f, ax = plt.subplots()
    if log_scale:
        ax.set_yscale('log')

    rhist = RHist1D.from_data(data, spec.bins_x, weights)

    plot_rhist1d(
        ax, rhist,
        histtype = 'step',
        linewidth = 2,
        color     = 'C0',
        label     = "True %s. Mean: %.2e" % (
            name, np.average(data, weights = weights)
        ),
    )
    plot_rhist1d_error(
        ax, rhist, err_type = 'bar', color = 'C0', linewidth = 2
    )

    spec.decorate(ax)
    ax.legend()
    remove_bottom_margin(ax)

    return f, ax, rhist

def plot_energy_hists(
    true_dict, weights_dict, preset, hist_specs, plotdir, ext
):
    for target, values in true_dict.items():
        for log_scale in [ True, False ]:
            weights = get_weights(weights_dict, target, true_dict)

            f, _ax, rhist = plot_single_energy_hist(
                values, weights, preset.name_map[target], hist_specs[target],
                log_scale
            )

            suffix = 'log' if log_scale else 'linear'
            save_fig(f, os.path.join(plotdir, f'{target}_{suffix}'), ext)

        np.savetxt(os.path.join(plotdir, f"{target}_hist.txt"), rhist.hist)
        np.savetxt(os.path.join(plotdir, f"{target}_bins.txt"), rhist.bins_x)

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, _args, _model, outdir, _plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    hist_specs = make_hist_specs(cmdargs, preset)
    plotdir    = os.path.join(outdir, 'targets')
    os.makedirs(plotdir, exist_ok = True)

    true_dict    = get_true_energies(dgen)
    weights_dict = dgen.weights

    plot_energy_hists(
        true_dict, weights_dict, preset, hist_specs, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

