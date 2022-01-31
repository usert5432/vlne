import argparse

from lstm_ee.presets       import PRESETS_EVAL
from lstm_ee.plot.hist     import plot_energy_hists
from lstm_ee.utils.eval    import standard_eval_prologue
from lstm_ee.utils.log     import setup_logging
from lstm_ee.utils.parsers import add_basic_eval_args, add_concurrency_parser
from lstm_ee.eval.predict  import (
    get_true_energies, get_base_energies, predict_energies
)

def parse_cmdargs():
    parser = argparse.ArgumentParser("Make Energy Hist plots")
    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)
    return parser.parse_args()

def make_aux_hist_plots(
    pred_model_dict, pred_base_dict, true_dict, weights, eval_specs,
    plotdir, ext
):
    """Make plots of energy histograms"""
    plot_energy_hists(
        [
            (true_dict,       weights, 'True',  'C3'),
            (pred_base_dict,  weights, 'Base',  'C0'),
            (pred_model_dict, weights, 'Model', 'C1'),
        ],
        eval_specs['hist'],
        "%s/plot_aux_hist" % (plotdir), ext
    )

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, eval_specs = standard_eval_prologue(
        cmdargs, PRESETS_EVAL
    )

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict       = get_true_energies(dgen)
    pred_base_dict  = get_base_energies(dgen, eval_specs['base_map'])

    make_aux_hist_plots(
        pred_model_dict, pred_base_dict, true_dict, dgen.weights,
        eval_specs, plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

