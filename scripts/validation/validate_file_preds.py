import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot          import save_fig
from vlne.consts        import LABEL_TOTAL, LABEL_PRIMARY
from vlne.data          import load_data
from vlne.eval.predict  import predict_energies
from vlne.utils.eval    import modify_concurrency_args
from vlne.utils.io      import load_model
from vlne.utils.log     import setup_logging
from vlne.utils.parsers import add_concurrency_parser

LABELS = [ LABEL_PRIMARY, LABEL_TOTAL ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        "Compare model prediction to the ones stored in the file"
    )

    parser.add_argument(
        'outdir',
        help    = 'directory with saved model',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-d', '--data',
        help    = 'evaluation dataset',
        default = None,
        dest    = 'data',
        type    = str,
    )

    parser.add_argument(
        '-p', '--plots',
        help    = 'plot directory',
        default = 'plots',
        dest    = 'plotdir',
        type    = str,
    )

    parser.add_argument(
        '-e', '--ext',
        help    = 'plot file extension. Example: pdf',
        default = [ 'png' ],
        dest    = 'ext',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '--preds',
        help    = 'names of branches with predictions in the input file',
        default = [ 'vlne.primaryE', 'vlne.totalE' ],
        dest    = 'file_preds',
        nargs   = 2,
        type    = str,
    )

    add_concurrency_parser(parser)

    return parser.parse_args()

def make_comparison_plot(test_preds, null_preds, label):
    f, ax = plt.subplots()
    ax.set_yscale('log')

    diff = 2 * (test_preds - null_preds) / (test_preds + null_preds)

    diff[np.isnan(test_preds)] =  1
    diff[np.isnan(null_preds)] = -1
    diff[np.isnan(null_preds) & np.isnan(test_preds)] = 0

    ax.hist(100 * diff, bins = 100)

    ax.set_xlabel('(Test - Null) / Null [%]')
    ax.set_ylabel('Events')
    ax.set_title(label)

    return f

def main():
    setup_logging()
    cmdargs = parse_cmdargs()

    args, model = load_model(cmdargs.outdir, compile = False)
    modify_concurrency_args(args, cmdargs)

    args.root_datadir     = ''
    args.config.dataset   = cmdargs.data
    args.config.test_size = None
    args.config.weights   = None

    args.config.var_target_primary = cmdargs.file_preds[0]
    args.config.var_target_total   = cmdargs.file_preds[1]

    dgen            = load_data(args)[0]
    test_preds_dict = predict_energies(args, dgen, model)

    for idx,label in enumerate(LABELS):
        null_preds = dgen.data_loader.get(cmdargs.file_preds[idx])
        test_preds = test_preds_dict[label]

        f = make_comparison_plot(test_preds, null_preds, label)
        save_fig(f, os.path.join(cmdargs.plotdir, label), cmdargs.ext)

if __name__ == '__main__':
    main()

