#
# This module contains functions to extract useful features of the EFI
# measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


def baseline(raw, method=1):
    """
    # Get baseline of the EFI measurement.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `method`: Compute method, 1: lowest 50th percentile; 2: lowest of all
    #
    # Return
    # =====
    # (float) The baseline value.
    """
    with np.errstate(all='ignore'):
        minimum = np.nanmin(raw, axis=1)
    if method == 1:
        # The lowest 50th percentile of each readout
        n = len(minimum)
        minimum = np.sort(minimum)[:(n // 2)]
        return np.mean(minimum)
    elif method == 2:
        return np.nanmin(minimum)


def powerlaw(x, a, b):
    """
    # Function: y = a * x ^ (-b)
    """
    return a * x ** (-b)


class PowerLawBaseline(object):
    """
    # Function: y = a * x ^ (-b) + baseline
    """
    def __init__(self, baseline=0):
        self.set_baseline(baseline)

    def set_baseline(self, baseline):
        # Set baseline value
        self.baseline = baseline

    def baseline(self):
        # Return current baseline value
        return self.baseline

    def __call__(self, x, a, b):
        # Function: y = a * x ^ (-b) + baseline
        return powerlaw(x, a, b) + self.baseline


def remove_nan(x, y):
    """
    # Remove any Nan value in `y` from both `x` and `y`.
    """
    ni = np.isnan(y)
    return x[~ni], y[~ni]


def curve_fit(raw, func, loc=None):
    """
    # Get the curvature of the EFI measurement, with given a parameteric form.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `func`: Function to fit to each curve, with `n_parameters`, giving the
    #         parameters as the gradients.
    # `loc`: EFI read out locations (in mm), if None, uses readout index.
    #
    # Return
    # =====
    # (dict) A dictionary with the stimulation electrode number as the key,
    # and parameters as the value which may contain 1 or 2 sets depending on
    # the stimulation electrode number.
    """
    from scipy.optimize import curve_fit
    n_readout, n_stimuli = raw.shape

    debug = False

    x_fit_right = range(0, 4)
    x_fit_both = range(4, 12)
    x_fit_left = range(12, 16)

    out = {}

    for i in range(n_stimuli):
        # If all nan, then skip it...
        if all(~np.isfinite(raw[:, i])):
            out[i] = [None, None]
            continue

        # Do the fits
        if i in x_fit_right:
            x1 = np.arange(1, n_readout - i)
            if loc is not None:
                x1 = loc[i + x1] - loc[i]
            y1 = raw[i + 1:, i]
            x1, y1 = remove_nan(x1, y1)  # remove any nan
            popt1, _ = curve_fit(func, x1, y1)
            if debug:
                import matplotlib.pyplot as plt
                plt.plot(x1, func(x1, *popt1)); plt.scatter(x1, y1, color='r')
                plt.show()
            out[i] = [popt1, None]
        elif i in x_fit_both:
            # Right
            x1 = np.arange(1, n_readout - i)
            if loc is not None:
                x1 = loc[i + x1] - loc[i]
            y1 = raw[i + 1:, i]
            x1, y1 = remove_nan(x1, y1)  # remove any nan
            popt1, _ = curve_fit(func, x1, y1)
            if debug:
                plt.plot(x1, func(x1, *popt1)); plt.scatter(x1, y1, color='r')
                plt.show()
            # Left
            x2 = np.arange(1, i)
            if loc is not None:
                x2 = loc[i] - loc[x2 - 1][::-1]
            y2 = raw[:i - 1:, i][::-1]
            x2, y2 = remove_nan(x2, y2)  # remove any nan
            popt2, _ = curve_fit(func, x2, y2)
            if debug:
                plt.plot(x2, func(x2, *popt2)); plt.scatter(x2, y2, color='r')
                plt.show()
            out[i] = [popt1, popt2]
        elif i in x_fit_left:
            x2 = np.arange(1, i)
            if loc is not None:
                x2 = loc[i] - loc[x2 - 1][::-1]
            y2 = raw[:i - 1:, i][::-1]
            x2, y2 = remove_nan(x2, y2)  # remove any nan
            popt2, _ = curve_fit(func, x2, y2)
            if debug:
                plt.plot(x2, func(x2, *popt2)); plt.scatter(x2, y2, color='r')
                plt.show()
            out[i] = [None, popt2]
        else:
            raise ValueError('No! We don\'t have that many readouts!')

    return out


def gradients(raw):
    """
    # Get the 4-point gradients of the EFI measurement.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    #
    # Return
    # =====
    # (array) A 2-D array of the 4-point gradients of each stimulus, with
    # shape (4, `n_stimuli`).
    """
    raise NotImplementedError


def peaks(raw):
    """
    # Get the peak values of the EFI measurement.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    #
    # Return
    # =====
    # (array) A 1-D array of the peak values of each stimulus.
    """
    with np.errstate(all='ignore'):
        return np.nanmax(raw, axis=1)
