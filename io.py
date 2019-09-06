#!/usr/bin/env python3
import numpy as np

"""
This module contains I/O functions to read and write the raw EFI measurements.
"""


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
    raise NotImplementedError


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

