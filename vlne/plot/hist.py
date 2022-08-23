"""
Functions to make plots of energy histograms.
"""

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot  import (
    make_figure_with_ratio,
    plot_rhist1d, plot_rhist1d_error, plot_rhist1d_ratios,
    save_fig, remove_bottom_margin
)
from cafplot.rhist import RHist1D

from vlne.eval.funcs import get_weights

def plot_hist_base(
    hist_data_list, target, spec, ratio_plot_type, stat_err, log = False
):
    """Plot multiple energy histograms"""

    if ratio_plot_type is not None:
        f, ax, axr = make_figure_with_ratio()
    else:
        f, ax = plt.subplots()

    if log:
        ax.set_yscale('log')

    list_of_rhist_color = []

    for hist_data in hist_data_list:
        values  = hist_data.values[target]
        weights = get_weights(hist_data.weights, target, hist_data.values)
        rhist   = RHist1D.from_data(values, spec.bins_x, weights)

        centers = (rhist.bins_x[1:] + rhist.bins_x[:-1]) / 2
        mean    = np.average(centers, weights = rhist.hist)

        plot_rhist1d(
            ax, rhist,
            histtype = 'step',
            marker    = None,
            linestyle = '-',
            linewidth = 2,
            label     = "%s. MEAN = %.3e" % (hist_data.label, mean),
            color     = hist_data.color,
        )

        if stat_err:
            plot_rhist1d_error(
                ax, rhist, err_type = 'bar', color = hist_data.color,
                linewidth = 2, alpha = 0.8
            )

        list_of_rhist_color.append((rhist, hist_data.color))

    spec.decorate(ax, ratio_plot_type)

    if not log:
        remove_bottom_margin(ax)

    ax.legend()

    if ratio_plot_type is not None:
        plot_rhist1d_ratios(
            axr,
            [rhist_color[0] for rhist_color in list_of_rhist_color],
            [rhist_color[1] for rhist_color in list_of_rhist_color],
            err_kwargs = { 'err_type' : 'bar' if stat_err else None },
        )
        spec.decorate_ratio(axr, ratio_plot_type)

    return f, ax

def plot_energy_hists(
    hist_data_list, plot_specs, fname, ext, log = False
):
    """Make and save plots of energy histograms.

    Parameters
    ----------
    hist_data_list : list of HistData
        List of hist data containers to plot.
    plot_specs : dict
        Dictionary where targets are energy labels and values are `PlotSpec` that
        specify axes and bins of the energy plots.
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    log : bool
        If True then the vertical axis will have logarithmic scale.
        Default: False.
    """

    for target in plot_specs.keys():
        for ratio_plot_type in [ None, 'auto', 'fixed' ]:
            for stat_err in [ True, False ]:
                f, _ = plot_hist_base(
                    hist_data_list, target, plot_specs[target],
                    ratio_plot_type, stat_err, log
                )

                fullname = "%s_%s_ratio-%s_staterr-%s" % (
                    fname, target, ratio_plot_type, stat_err
                )
                save_fig(f, fullname, ext)
                plt.close(f)

