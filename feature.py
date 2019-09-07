#!/usr/bin/env python3
import numpy as np

"""
This module contains functions to extract useful features of the EFI
measurements.
"""


def baseline(raw):
    """
    # Get baseline of the EFI measurement.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    #
    # Return
    # =====
    # (float) The baseline value.
    """
    # The lowest 50th percentile of each readout
    with np.errstate(all='ignore'):
        minimum = np.nanmin(raw, axis=1)
    n = len(minimum)
    minimum = np.sort(minimum)[:(n // 2)]
    return np.mean(minimum)


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

