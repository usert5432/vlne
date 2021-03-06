"""
Functions to create standard `vlne` command line argument parsers.
"""

import argparse

def add_basic_eval_args(parser, presets_eval):
    """Create cmdargs parser of the standard evaluation options"""

    parser.add_argument(
        'outdir',
        help    = 'directory with saved model',
        metavar = 'OUTDIR',
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
        '-w', '--weights',
        help    = 'weights to use during evaluation.',
        default = 'weight',
        dest    = 'weights',
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
        '-n', '--noise',
        help    = 'noise configuration to use during evaluation.',
        default = 'none',
        dest    = 'noise',
        type    = str,
    )

    parser.add_argument(
        '-P', '--preset',
        help     = 'evaluation preset',
        choices  = list(presets_eval.keys()),
        default  = None,
        dest     = 'preset',
        type     = str,
        required = True,
    )

    parser.add_argument(
        '-s', '--prong-sorter',
        help     = 'prong ordering configuration.',
        default  = None,
        dest     = 'prong_sorter',
        type     = str,
    )

    parser.add_argument(
        '--seed',
        help    =
            'seed to initialize PRG for data split into train/test parts',
        default = 'same',
        dest    = 'seed',
        type    = str,
    )

    parser.add_argument(
        '-t', '--test-size',
        help    = 'subset size of the dataset to use for evaluation',
        default = 'same',
        dest    = 'test_size',
        type    = str,
    )

    parser.add_argument(
        '--label',
        help    = 'evaluation label',
        default = None,
        dest    = 'label',
        type    = str,
    )

def add_concurrency_parser(parser):
    """Create cmdargs parser of the concurrency/caching options"""

    parser.add_argument(
        '--cache',
        help    = 'use RAM cache',
        action  = 'store_true',
        dest    = 'cache',
    )

    parser.add_argument(
        '--disk-cache',
        help    = 'use disk based cache',
        action  = 'store_true',
        dest    = 'disk_cache',
    )

    parser.add_argument(
        '--workers',
        help    = 'number of concurrent workers',
        dest    = 'workers',
        default = None,
        type    = int,
    )

    parser.add_argument(
        '--concurrency',
        help    = 'type of parallelization',
        dest    = 'concurrency',
        choices = [ 'thread', 'process' ],
        default = None,
    )

def add_hist_binning_parser(
    parser,
    default_range_lo = None,
    default_range_hi = None,
    default_bins     = 10,
):

    parser.add_argument(
        '--range-lo',
        help    = 'plot left boundary',
        default = default_range_lo,
        dest    = 'range_lo',
        type    = float,
    )

    parser.add_argument(
        '--range-hi',
        help    = 'plot right boundary',
        default = default_range_hi,
        dest    = 'range_hi',
        type    = float,
    )

    parser.add_argument(
        '--bins',
        help    = 'number of bins',
        default = default_bins,
        dest    = 'bins',
        type    = int,
    )

    parser.add_argument(
        '--bin-edges',
        help    = 'bin edges. Overrides bins and range settings',
        default = None,
        dest    = 'bin_edges',
        type    = float,
        nargs   = '+',
    )

def parse_concurrency_cmdargs(config_dict, title = "Train"):
    """Parse command line concurrency options into `config_dict`"""
    parser = argparse.ArgumentParser(title)
    add_concurrency_parser(parser)

    cmdargs = parser.parse_args()
    config_dict['concurrency'] = cmdargs.concurrency
    config_dict['cache']       = cmdargs.cache
    config_dict['disk_cache']  = cmdargs.disk_cache
    config_dict['workers']     = cmdargs.workers

