#
# This module contains functions of Gaussian process to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def kernel(c_value=1.0, c_bound=(1e-3, 1e3), rbf_scale=10.,
        rbf_bound=(1e-2, 1e2)):
    """
    Create a scikit-learn constant-Radial-basis function kernel instance for
    Gaussian process.
    """
    return C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))


def gaussian_process(kernel, alpha=1e-10, n_restarts_optimiser=10,
        random_state=None):
    """
    Create a scikit-learn Gaussian process instance.

    Example
    =======
    >>> k = kernal()
    >>> gp = gaussian_process(k)
    >>> gp.fit(x_data, y_data)  # input data for training
    >>> gp.score(x_training_data, y_training_data)  # to check the fitted score
    >>> y_predicted = gp.predict(x_validation_data, return_std=True)
    """
    return GaussianProcessRegressor(kernel=kernel,
                                    alpha=alpha,
                                    n_restarts_optimizer=n_restarts_optimiser,
                                    random_state=random_state)

