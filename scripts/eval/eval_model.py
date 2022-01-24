import argparse
import os

from vlne.presets       import PRESETS_EVAL
from vlne.utils.log     import setup_logging
from vlne.utils.parsers import (
    add_basic_eval_args, add_concurrency_parser, add_hist_binning_parser
)
from vlne.utils.eval     import standard_eval_prologue, parse_binning
from vlne.eval.eval      import evaluate
from vlne.plot.fom       import plot_fom
from vlne.plot.plot_spec import PlotSpec

def make_hist_specs(cmdargs, preset):
    binning = parse_binning(cmdargs, suffix = '_x')

    return {
        label : PlotSpec(
            title   = None,
            label_x = f'(Reco - True) / True {name} Energy',
            label_y = 'Events',
            **binning
        )
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
    parser = argparse.ArgumentParser("Evaluate Energy Resolution")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)
    add_energy_resolution_parser(parser)

    return parser.parse_args()

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, outdir, plotdir, preset = \
        standard_eval_prologue(cmdargs, PRESETS_EVAL)

    hist_specs = make_hist_specs(cmdargs, preset)

    (stats_model_dict, hCont_model_dict), (stats_base_dict, hCont_base_dict) \
        = evaluate(
            args, dgen, model, preset.base_map, hist_specs, cmdargs.fit_margin,
            outdir
        )

    plot_fom(
        [
            (hCont_base_dict,  stats_base_dict,  'Base ', 'left',  'C0'),
            (hCont_model_dict, stats_model_dict, 'Model', 'right', 'C1'),
        ],
        hist_specs,
        fname   = os.path.join(plotdir, 'resolution'),
        ext     = cmdargs.ext
    )

if __name__ == '__main__':
    main()

