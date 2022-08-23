"""
Functions to make binstat plots.
"""

import matplotlib.pyplot as plt
from cafplot.plot import plot_nphist1d_base, save_fig

from vlne.eval.binned_stats import calc_binned_stats, calc_stat
from vlne.eval.funcs        import get_weights

def plot_binstat_single(ax, x, y, weights, label, color, spec, stat):
    """Add a plot of single binstat to axes `ax`"""

    binstats = calc_binned_stats(x, y, weights, spec.bins_x, stat)
    fullstat = calc_stat(y, weights, stat)

    plot_nphist1d_base(
        ax, binstats, spec.bins_x,
        color     = color,
        label     = '%s : %.3e' % (label, fullstat),
        linewidth = 2
    )

    return binstats

def plot_binstat_base(plot_data_list, target, spec, stat, is_rel = False):
    """Plot binstats of relative energy resolution vs true energy."""
    spec = spec.copy()

    if spec.title is None:
        spec.title = stat.upper()
    else:
        spec.title = "%s( %s )" % (stat.upper(), spec.title)

    f, ax = plt.subplots()

    for plot_data in plot_data_list:
        x = plot_data.true[target]
        y = (plot_data.pred[target] - plot_data.true[target])
        w = get_weights(plot_data.weights, target, plot_data.pred)

        if is_rel:
            y = y / x

        plot_binstat_single(
            ax, x, y, w, plot_data.label, plot_data.color, spec, stat
        )

    ax.axhline(0, 0, 1, color = 'C2', linestyle = 'dashed')
    spec.decorate(ax)

    ax.legend()

    return f, ax

def plot_binstats(
    plot_data_list, plot_specs_abs, plot_specs_rel, fname, ext,
    stat_list = [ 'mean', 'rms' ]
):
    """Make and save binstat plots of energy resolution vs true energy.

    Parameters
    ----------
    plot_data_list : list of PlotData
        List of graph data.
    plot_specs_abs : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the absolute energy resolution plots.
    plot_specs_rel : dict
        Dictionary where keys are energy labels and values are `PlotSpec` that
        specify axes and bins of the relative energy resolution plots.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    stat_list : list, optional
        List of statistic properties for which binstat plots will be made.
        Default: [ 'mean', 'rms' ]
    """
    # pylint: disable=dangerous-default-value

    targets = plot_specs_abs.keys()

    for (is_rel, spec, rel_label) in zip(
        [ True,           False ],
        [ plot_specs_rel, plot_specs_abs ],
        [ 'rel',          'abs' ]
    ):
        for target in targets:
            for stat in stat_list:
                f, _ = plot_binstat_base(
                    plot_data_list, target, spec[target], stat, is_rel
                )

                fullname = "%s_%s_%s_%s" % (fname, target, stat, rel_label)
                save_fig(f, fullname, ext)
                plt.close(f)

