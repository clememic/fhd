"""
Histogram distances module.
"""

import numpy as np

metrics = {'L1': ['L1', 'manhattan', 'man'],
           'L2': ['L2', 'euclidean', 'euc'],
           'CHI2': ['CHI2', 'chi2'],
           'CEMD': ['CEMD', 'cemd']}


def distance(a, b, metric='L2'):
    """
    Compute the distance between two 1D histograms.

    Several metrics are available: L1, L2, CHI2 and CEMD.

    Parameters
    ----------
    a, b : (N,) ndarray
        Two histograms between which the distance is computed.
    metric : str, optional
        The metric used to compute the distance. Default value is 'L2'.

    Returns
    -------
    distance : float
        The distance between `a` and `b`.

    Raises
    ------
    ValueError
        If `a` and `b` are empty, not 1D or don't have the same size.
        If `metric` isn't valid.

    """
    a, b = np.atleast_1d(a, b)
    if a.ndim > 1 or b.ndim > 1:
        raise ValueError("a and b should be 1D arrays.")
    if a.size == 0 or b.size == 0:
        raise ValueError("a and b should not be empty.")
    if a.size != b.size:
        raise ValueError("a and b should have the same size.")

    if metric in metrics['L1']:
        return np.abs(a - b).sum()

    elif metric in metrics['L2']:
        return np.linalg.norm(a - b)

    elif metric in metrics['CHI2']:
        # "0/0 = 0" convention is used
        olderr = np.seterr(invalid='ignore')
        d = np.nansum(((a - b) ** 2) / (a + b))
        np.seterr(**olderr)
        return d

    elif metric in metrics['CEMD']:
        raise NotImplementedError('Coming soon!')  # TODO

    else:
        raise ValueError('Not a valid metric.')
