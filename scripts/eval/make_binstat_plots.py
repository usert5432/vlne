import argparse
import os

from lstm_ee.presets         import PRESETS_EVAL
from lstm_ee.plot.binstat    import plot_binstats
from lstm_ee.utils.eval      import standard_eval_prologue, parse_binning
from lstm_ee.utils.log       import setup_logging
from lstm_ee.utils.parsers   import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from lstm_ee.eval.predict    import (
    get_true_energies, get_base_energies, predict_energies
)
from lstm_ee.plot.plot_spec import PlotSpec

def make_plot_spec(name, unit, relative = False, **kwargs):

    if relative:
        label_y = f'(Reco - True) / True {name} Energy'
    else:
        label_y = f'(Reco - True) {name} Energy [{unit}]'

    return PlotSpec(
        title   = None,
        label_x = f'True {name} Energy [{unit}]',
        label_y = label_y,
        minor   = True,
        grid    = True,
        **kwargs
    )

def make_plot_specs(cmdargs, preset, relative):
    binning = parse_binning(cmdargs, suffix = '_x')

    return {
        label : make_plot_spec(
            name, preset.units_map[label], relative, **binning
        )
        for (label, name) in preset.name_map.items()
    }

def parse_cmdargs():
    parser = argparse.ArgumentParser("Make Binstat plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    add_hist_binning_parser(
        parser,
        default_range_lo = 0,
        default_range_hi = 5,
        default_bins     = 50,
    )

    return parser.parse_args()

def make_binstat_plots(
    pred_model_dict, pred_base_dict, true_dict, weights, plot_spec_abs,
    plot_spec_rel, plotdir, ext
):
    root = os.path.join(plotdir, "binstat")
    os.makedirs(root)

    plot_binstats(
        [
            (pred_base_dict,  true_dict, weights, 'Baseline', 'C0'),
            (pred_model_dict, true_dict, weights, 'Model',    'C1'),
        ],
        plot_spec_abs, plot_spec_rel, os.path.join(root, "plot_binstat"), ext
    )

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict       = get_true_energies(dgen)
    pred_base_dict  = get_base_energies(dgen, preset.base_map)

    plot_spec_abs = make_plot_specs(cmdargs, preset, relative = False)
    plot_spec_rel = make_plot_specs(cmdargs, preset, relative = True)

    make_binstat_plots(
        pred_model_dict, pred_base_dict, true_dict, dgen.weights,
        plot_spec_abs, plot_spec_rel, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

