"""
Functions to make auxiliary (i.e. mostly useless) evaluation plots.
"""

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from cafplot.plot      import save_fig

from vlne.eval.funcs import get_weights

def plot_rel_res_vs_true_base(pred, true, weights, spec, logNorm = False):
    """Plot 2D histogram of relative energy resolution vs true energy"""

    f, ax = plt.subplots()

    res = (pred - true) / true

    if logNorm:
        kwargs = { 'norm' : LogNorm() }
    else:
        kwargs = { }

    _hist2d, _, _, image = ax.hist2d(
        true, res,
        bins    = [spec.bins_x, spec.bins_y],
        weights = weights,
        cmin = 1e-5,
        **kwargs
    )

    ax.axhline(0, 0, 1, color = 'C2', linestyle = 'dashed')

    spec.decorate(ax)
    f.colorbar(image)

    return f, ax

def plot_rel_res_vs_true(
    pred_dict, true_dict, weights_dict, plot_specs, fname, ext
):
    """
    Make and save 2D hist plots of relative energy resolution vs true energy.
    """

    for target in pred_dict.keys():
        pred    = pred_dict[target]
        true    = true_dict[target]
        weights = get_weights(weights_dict, target, pred_dict)
        spec    = plot_specs[target]

        for logNorm in [True, False]:
            try:
                f, _ = plot_rel_res_vs_true_base(
                    pred, true, weights, spec, logNorm
                )
            except ValueError as e:
                print("Failed to make plot: %s" % (str(e)))
                continue

            path = "%s_%s_log(%s)" % (fname, target, logNorm)

            save_fig(f, path, ext)
            plt.close(f)

