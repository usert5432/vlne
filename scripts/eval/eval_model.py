"""Evaluate model energy resolutions."""

import argparse
import os

from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import add_basic_eval_args, add_concurrency_parser
from lstm_ee.utils.eval    import standard_eval_prologue
from lstm_ee.eval.eval     import evaluate
from lstm_ee.plot.fom      import plot_fom
from lstm_ee.plot.plot_spec import PlotSpec

# pylint: disable=redefined-builtin
def make_hist_spec_resolution(energy, bins = 200, range = (-1, 1), **kwargs):
    """Create `PlotSpec` for the energy resolution plot."""
    return PlotSpec(
        title   = None,
        label_x = '(Reco - True) / True %s Energy' % (energy),
        label_y = 'Events',
        bins_x  = bins,
        range_x = range,
        **kwargs
    )

def make_hist_specs(cmdargs, preset):
    binning = parse_binning(cmdargs)

    return {
        k : make_hist_spec_resolution(v, **binning)
            for (k,v) in preset.name_map.items()
    }

def add_energy_resolution_parser(parser):
    parser.add_argument(
        '--fit_margin',
        help    = 'Fraction of resolution peak to fit gaussian',
        default = 0.5,
        dest    = 'fit_margin',
        type    = float,
    )

    parser.add_argument(
        '--range-lo',
        help    = 'Plot left boundary',
        default = -1,
        dest    = 'range_lo',
        type    = float,
    )

    parser.add_argument(
        '--range-hi',
        help    = 'Plot right boundary',
        default = 1,
        dest    = 'range_hi',
        type    = float,
    )

    parser.add_argument(
        '--bins',
        help    = 'Number of bins',
        default = 200,
        dest    = 'bins',
        type    = int,
    )

    parser.add_argument(
        '--bin_edges',
        help    = 'Edges of bins. Overwrites bins and range settings',
        default = None,
        dest    = 'bin_edges',
        type    = float,
        nargs   = '+',
    )

def parse_binning(cmdargs):
    result = {
        'range' : (cmdargs.range_lo, cmdargs.range_hi),
    }

    if cmdargs.bin_edges is not None:
        result['bins'] = cmdargs.bin_edges
    else:
        result['bins'] = cmdargs.bins

    return result

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Evaluate Performance of the Model")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)
    add_energy_resolution_parser(parser)

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
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

