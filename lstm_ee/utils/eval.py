"""
A collection of evaluation routines.
"""

import copy
import os

from cafplot.plot import make_plotdir
from lstm_ee.data import load_data

from .eval_config import EvalConfig
from .io          import load_model

def make_eval_outdir(outdir, eval_config):
    """Create evaluation subdir unique for `eval_config`"""
    result = os.path.join(outdir, 'evals')
    result = eval_config.get_evaldir(result)

    os.makedirs(result, exist_ok = True)

    return result

def modify_concurrency_args(args, cmdargs):
    """Modify concurrency arguments of `args` from `argparse.Namespace`"""
    args.concurrency = cmdargs.concurrency
    args.cache       = cmdargs.cache
    args.workers     = cmdargs.workers

def modify_specs(specs, func):
    """Map `func` over a dict of `PlotSpec`"""
    return { k : func(copy.deepcopy(v)) for k,v in specs.items() }

def modify_title(spec, value):
    """Modify title of `PlotSpec`"""
    if spec.title is None:
        spec.title = value
    else:
        spec.title = "%s : %s" % (spec.title, value)

    return spec

def parse_binning(cmdargs, suffix = ''):
    result = {
        'range' + suffix : (cmdargs.range_lo, cmdargs.range_hi),
    }

    if cmdargs.bin_edges is not None:
        result['bins' + suffix] = cmdargs.bin_edges
    else:
        result['bins' + suffix] = cmdargs.bins

    return result

def standard_eval_prologue(cmdargs, presets_eval):
    """Standard evaluation prologue"""
    args, model = load_model(cmdargs.outdir, compile = False)
    eval_config = EvalConfig.from_cmdargs(cmdargs)

    eval_config.modify_eval_args(args)
    modify_concurrency_args(args, cmdargs)

    _, dgen = load_data(args)
    outdir  = make_eval_outdir(cmdargs.outdir, eval_config)
    plotdir = make_plotdir(outdir)
    preset  = presets_eval[cmdargs.preset]

    return (dgen, args, model, outdir, plotdir, preset)

