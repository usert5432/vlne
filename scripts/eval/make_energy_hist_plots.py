import argparse
import os

from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.plot.hist     import plot_energy_hists
from lstm_ee.utils.eval    import standard_eval_prologue, parse_binning
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from lstm_ee.eval.predict  import (
    get_true_energies, get_base_energies, predict_energies
)
from lstm_ee.plot.plot_spec import PlotSpec

def make_hist_specs(cmdargs, preset):
    binning = parse_binning(cmdargs, suffix = '_x')

    return {
        label : PlotSpec(
            title   = None,
            label_x = f'{name} Energy [{preset.units_map[label]}]',
            label_y = 'Events',
            **binning
        )
        for (label, name) in preset.name_map.items()
    }

def parse_cmdargs():
    parser = argparse.ArgumentParser("Make Energy Hist plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    add_hist_binning_parser(
        parser,
        default_range_lo = 0,
        default_range_hi = 5,
        default_bins     = 100,
    )

    return parser.parse_args()

def make_hist_plots(
    pred_model_dict, pred_base_dict, true_dict, weights, hist_specs,
    plotdir, ext
):
    """Make plots of energy histograms"""
    plot_energy_hists(
        [
            (true_dict,       weights, 'True',  'C3'),
            (pred_base_dict,  weights, 'Base',  'C0'),
            (pred_model_dict, weights, 'Model', 'C1'),
        ],
        hist_specs, os.path.join(plotdir, "plot_hist"), ext
    )

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    hist_specs = make_hist_specs(cmdargs, preset)

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict       = get_true_energies(dgen)
    pred_base_dict  = get_base_energies(dgen, preset.base_map)

    make_hist_plots(
        pred_model_dict, pred_base_dict, true_dict, dgen.weights,
        hist_specs, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

