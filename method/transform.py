#
# This module contains classes for parameter transformation.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import StandardScaler


class StandardScalingTransform(StandardScaler):
    """
    Apply transformation ``z = (x - u) / s`` where `z` is the transformed
    variable, `x` a sample, `u` the mean, `s` the standard deviation.
    """


class NaturalLogarithmicTransform(object):
    """
    Apply transformation ``z = ln(x + 1)`` where `z` is the transformed
    variable, `x` a sample, `ln(.)` natural log-function.
    """
    def __init__(self, copy=True):
        """
        Parameters:
            copy: boolean, optional, default True. If False, try to avoid a
                  copy and do inplace scaling instead.
        """
        self._copy = copy

    def transform(self, X, copy=None):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               trasnformed.
            copy: bool, optional (default: None). Copy the input X or not.
        """
        if copy is not None:
            copy = copy
        else:
            copy = self._copy

        if copy:
            X_copy = np.copy(X)
            return np.log(X_copy + 1)
        else:
            np.log(X + 1)

    def inverse_transform(self, X, copy=None):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               inverse-trasnformed.
            copy: bool, optional (default: None). Copy the input X or not.
        """
        if copy is not None:
            copy = copy
        else:
            copy = self._copy

        if copy:
            X_copy = np.copy(X)
            return np.exp(X_copy) - 1
        else:
            np.exp(X) - 1
