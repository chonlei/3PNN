#
# This module contains I/O helper classes for the Cochlea EFI project.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


def load(filename):
    """
    # Load the raw data the EFI measurement.
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
    # Save the raw data the EFI measurement to a file.
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
    # `x`: 
    #
    # Return
    # =====
    # Modified data with broken electrodes removed.
    """
    out = np.copy(raw)

    for i in x:
        out[:,i - 1] = np.nan # Remove column 12
        out[i - 1] = np.nan # Remove row 12

    np.fill_diagonal(out, np.nan) # Remove diagonal

    return out

