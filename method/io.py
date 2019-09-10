#
# This module contains I/O helper classes for the Cochlea EFI project.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


def load(filename):
    """
    # Load the raw data of the EFI measurement.
    #
    # Input
    # =====
    # `filename`: File name of the EFI measurement.
    #
    # Return
    # =====
    # (array) A 2-D array of the raw EFI measurement, with shape
    # (`n_readout`, `n_stimuli`).
    """
    return np.loadtxt(filename)


def save(filename, data):
    """
    # Save the raw data of the EFI measurement to a file.
    #
    # Input
    # =====
    # `filename`: File name of the EFI measurement to save to.
    # `data`: The data to save, expect shape (`n_readout`, `n_stimuli`).
    """
    raise NotImplementedError


def mask(raw, x=[]):
    """
    # Remove broken electrodes.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `x`: electrode index to be removed; note that index starts from 1.
    #
    # Return
    # =====
    # Modified data with broken electrodes removed.
    """
    out = np.copy(raw)

    for i in x:
        out[:,i - 1] = np.nan # Remove column i
        out[i - 1] = np.nan # Remove row i

    np.fill_diagonal(out, np.nan) # Remove diagonal

    return out


def load_input(filename):
    """
    # Load the input/printing parameters of the EFI measurement.
    #
    # Input
    # =====
    # `filename`: File name of the input parameters.
    #
    # Return
    # =====
    # (array) A 1-D array of the input parameters, with shape
    # (`n_inputs`, ).
    """
    return np.loadtxt(filename)


def save_peaks(filename, data):
    """
    # Save the peak feature of the EFI measurement to a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    # `data`: The data to save, expect shape (`n_readout`, ).
    """
    np.savetxt(filename + '-peaks.csv', data, delimiter=',', comments='',
            header='\"EFI_peaks\"')


def load_peaks(filename):
    """
    # Load the peak feature of the EFI measurement from a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    #
    # Return
    # =====
    # (array) A 1-D array of the peak feature of the EFI measurement, with
    # shape (`n_readout`, ).
    """
    return np.loadtxt(filename + '-peaks.csv', delimiter=',', skiprows=1)


def save_baseline(filename, data):
    """
    # Save the baseline feature of the EFI measurement to a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    # `data`: The data to save, expect a scalar.
    """
    np.savetxt(filename + '-baseline.csv', np.asarray([data]), delimiter=',',
            comments='', header='\"EFI_baseline\"')


def load_baseline(filename):
    """
    # Load the baseline feature of the EFI measurement from a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    #
    # Return
    # =====
    # (float) The baseline feature of the EFI measurement.
    """
    return np.loadtxt(filename + '-baseline.csv', delimiter=',', skiprows=1)


def save_curve_parameters(filename, data, n_parameters):
    """
    # Save the curve fit parameters of the EFI measurement to a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    # `data`: The data to save, expect a dictionary with the stimulation
    #         electrode number as the key, and parameters as the value which
    #         contain 2 sets of parameters, one for right and one for the left.
    """
    n_electrodes = len(data)
    out = np.full((n_electrodes, 2 * n_parameters), np.NaN)

    for i in range(n_electrodes):
        if data[i][0] is not None:
            out[i, :n_parameters] = data[i][0][:]
        if data[i][1] is not None:
            out[i, n_parameters:] = data[i][1][:]

    left_p = ''
    right_p = ''
    for i in range(n_parameters):
        idx = str(i) + '\"' if (i + 1) == n_parameters else str(i) + '\",'
        left_p += '\"left.p' + idx
        right_p += '\"right.p' + idx
    header = right_p + ',' + left_p

    np.savetxt(filename + '-curve_fit.csv', out, delimiter=',', comments='',
            header=header)


def load_curve_parameters(filename):
    """
    # Load the feature the EFI measurement from a file.
    #
    # Input
    # =====
    # `filename`: File name/path without extension.
    #
    # Return
    # =====
    # (dict) A dictionary with the stimulation electrode number as the key,
    # and parameters as the value which may contain 1 or 2 sets depending on
    # the stimulation electrode number.
    """
    p = np.loadtxt(filename + '-curve_fit.csv', delimiter=',', skiprows=1)

    n_electrodes, n_parameters = p.shape
    n_parameters = int(n_parameters / 2)
    out = {}

    for i in range(n_electrodes):
        if all(np.isfinite(p[i, :n_parameters])):
            r = p[i, :n_parameters]
        else:
            r = None

        if all(np.isfinite(p[i, n_parameters:])):
            l = p[i, n_parameters:]
        else:
            l = None

        out[i] = [r, l]

    return out


def split(input_dict, index):
    """
    # Split the keys of the `input_dict` into groups such that except `index`
    # all other elements in the are the same.
    """
    out = []
    same = []

    for k in input_dict.keys():
        mask = np.ones(len(input_dict[k]), dtype=bool)
        mask[index] = False
        v = input_dict[k][mask]
        is_repeat = False
        for i, s in enumerate(same):
            if all(v == s):
                out[i].append(k)
                is_repeat = True
                continue
        if not is_repeat:
            out.append([k])
            same.append(v)

    return out, same

