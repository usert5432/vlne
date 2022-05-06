"""
Functions for calculating data weights.
"""

import numpy as np

def calc_flat_whist(
    df, var = 'trueE', bins = 50, range = (0, 5), clip = None
):
    """Calculate normalized inverse of the `var` histogram.

    This function calculates histogram of the variable `var` whose values
    stored in `data_loader` and returned inverse of this histogram, normalized
    to 1.

    Parameters
    ----------
    df : DataFrameBase
        Data Frame that contains `var` column.
    var : str
        Variable name in `IDataLoader` which histogram is to be calculated.
    bins : int or ndarray
        Number of bins in a histogram, or a list of bin edges.
    range : (float, float) or None, optional
        Range of a histogram (lower, upper). If None, range will be determined
        according to the `np.histogram` rules. Default: None.
    clip : float or None, optional
        If `clip` is not None, then the maximum value of the inverse histogram
        of the `var` will be clipped by `clip`. Default: None.

    Returns
    -------
    values : ndarray, shape (len(data_loader),)
        Values of the `var` variable extracted from `data_loader`
    whist : ndarray
        Inverse of the `values` histogram.
    bins : ndarray
        List of bin edges.
    """
    # pylint: disable=redefined-builtin
    wvalues    = df[var]
    hist, bins = np.histogram(wvalues, bins = bins, range = range)

    # Regularization
    hist += 1
    whist = 1 / hist

    if clip is not None:
        min_w = min(whist)
        max_w = clip * min_w

        whist[whist > max_w] = max_w

    whist = whist / sum(whist)

    return (wvalues, whist, bins)

def flat_weights(
    df, var = 'trueE', bins = 50, range = (0, 5), clip = None
):
    """Calculate weights that will make weighted histogram of `var` flat.

    Parameters
    ----------
    df : DataFrameBase
        Data Frame that contains `var` column.
    var : str
        Variable name in `IDataLoader` which histogram should be flattened.
    bins : int or ndarray
        Number of bins in a histogram, or a list of bin edges.
    range : (float, float) or None, optional
        Range of a histogram (lower, upper). If None, range will be determined
        according to the `np.histogram` rules. Default: None.
    clip : float or None, optional
        If `clip` is not None, then it will limit the maximum value the weight
        can achieve by `clip`. Default: None.

    Returns
    -------
    ndarray, shape (len(data_loader),)
        An array of weights, that will make weighted histogram of `var` with
        bin edges defined by `bins` and `range` flat (up to clipping).
    """

    # pylint: disable=redefined-builtin
    (wvalues, whist, bins) = calc_flat_whist(df, var, bins, range, clip)

    wpos = np.digitize(wvalues, bins)

    # [0, len(bins)] are overflow bins
    wpos[wpos == 0] = 1
    wpos[wpos == len(bins)] = len(bins) - 1

    weights = whist[wpos - 1]
    weights = weights / sum(weights) * len(df)

    return weights

