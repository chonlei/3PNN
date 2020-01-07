#
# This module contains functions of Gaussian process to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel as W


def kernel_constant_rbf(c_value=1.0, c_bound=(1e-8, 1e3), rbf_scale=10.,
        rbf_bound=(1e-2, 1e2), w_level=1e-3, w_bound=(1e-6, 1e-1)):
    """
    Create a scikit-learn constant-Radial-basis function kernel instance for
    Gaussian process.
    """
    return C(c_value, c_bound) * RBF(rbf_scale, rbf_bound) \
        + W(w_level, w_bound)


def kernel_constant_matern_white(c_value=1.0, c_bound=(1e-8, 1e3),
        matern_scale=10., matern_bound=(1e-2, 1e2), matern_nu=1.5,
        w_level=1e-3, w_bound=(1e-6, 1e3)):
    """
    Create a scikit-learn constant-Matern-white kernel instance for
    Gaussian process.
    """
    return C(c_value, c_bound) + Matern(matern_scale, matern_bound, matern_nu)\
        + W(w_level, w_bound)


def kernel():
    """
    Create a default kernel instance for Gaussian process.
    """
    return kernel_constant_rbf()


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


#
# PINTS Model
#
class GPModelForPints(object):
    """
    A wrapper for the scikit-learn GP model for PINTS inverse problem.

    Expect the GP model was fitted individually/independently to each stimulus.
    """
    def __init__(self, gp_model, stim_idx, stim_pos, shape, transform_x=None,
                 transform=None):
        """
        Input
        =====
        `gp_model`: (dict) A dictionary with key = j-stimulus and
                    value = scikit-learn GP model fitted independently to the
                    j-stimulus.
        `stim_idx`: (array) Stimulus indices, usually [0, 1, 2, ..., 15].
        `stim_pos`: (array) Stimulus position, corresponding to the measurement
                    positions.
        `shape`: (tuple) Shape of simulate() output matrix.

        Optional input
        =====
        `transform_x`: (dict) Transformation of GP input from search space,
                       with key = j-stimulus and value = transformation
                       function for the j-stimulus.
        `transform`: Transformation of GP output to EFI output.
        """
        super(GPModelForPints, self).__init__()
        self.gpr = gp_model
        self._np = self.gpr[stim_idx[0] + 1].X_train_.shape[1]
        self._np -= 1  # first one is stim pos.
        self._stim_idx = stim_idx
        self._stim_pos = stim_pos
        self._shape = shape
        if transform is not None:
            self._transform = transform
        else:
            self._transform = lambda x: x
        if transform_x is not None:
            self._transform_x = transform_x
        else:
            self._transform_x = {}
            for i in self._stim_idx:
                self._transform_x[i + 1] = lambda x: x

    def n_parameters(self):
        """
        Return number of parameters.
        """
        return self._np

    def simulate(self, x, return_std=False):
        """
        Return a simulated EFI given the parameters `x`.
        """
        out = np.zeros(self._shape)
        if return_std:
            std = np.zeros(self._shape)

        for j_stim in self._stim_idx:
            gpr_j = self.gpr[j_stim + 1]
            transform_x_j = self._transform_x[j_stim + 1]

            predict_x = [np.append(i, x) for i in self._stim_pos]
            predict_x = transform_x_j(predict_x)
            y = gpr_j.predict(predict_x, return_std=return_std)

            if return_std:
                out[self._stim_idx, j_stim] = self._transform(y[0])
                std[self._stim_idx, j_stim] = self._transform(y[1])
            else:
                out[self._stim_idx, j_stim] = self._transform(y)

        if return_std:
            return out, std
        else:
            return out


#
# Log-likelihood for inverse problem
#
class GaussianLogLikelihood(pints.LogPDF):
    """
    Define a log-likelihood for the problem for PINTS [1].

    .. math::
        \log{L(\theta, \sigma|\boldsymbol{x})} =
            -\frac{n_t}{2} \log{2\pi}
            -n_t \log{\sigma}
            -\frac{1}{2\sigma^2}\sum_{j=1}^{n_t}{(x_j - f_j(\theta))^2}

    where ``n_t`` is the number of measurement points, ``x_j`` is the
    sampled data at ``j`` and ``f_j`` is the simulated data at ``j``.

    [1] Clerx M, et al., 2019, JORS.
    """
    def __init__(self, model, values, mask=None, fix=None, transform=None):
        """
        Input
        =====
        `model`: A GP model, following the ForwardModel requirements in
                 PINTS.
        `values`: The data that match the output of the `model` simulation.

        Optional input
        =====
        `mask`: A function that takes in the data and replace undesired
                entries with `nan`.
        `fix`: A list containing
               (1) a function that takes the input parameters and return a full
               set of parameters with the fixed parameters; and
               (2) number of fitting parameters.
        `transform`: Transformation of EFI output (data) to GP output; such
                     that likelihood is Gaussian.
        """
        super(GaussianLogLikelihood, self).__init__()

        self._model = model
        self._values = values
        self._mask = mask
        if transform is not None:
            self._transform = transform
        else:
            self._transform = lambda x: x
        self._trans_values = self._transform(self._values)
        if fix is not None:
            self._fix = fix[0]
            self._np = fix[1]
        else:
            self._fix = lambda x: x
            self._np = self._model.n_parameters()

        # Store counts
        self._nt = np.nansum(self._mask(np.ones(self._values.shape)))

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def n_parameters(self):
        """
        Return number of parameters.
        """
        return self._np

    def __call__(self, x):
        """
        Return the computed Gaussian log-likelihood for the given parameters
        `x`.
        """
        x = self._fix(x)  # get fixed parameters
        mean, sigma = self._model.simulate(x, return_std=True)
        # Compare transformed values so that sigma makes sense in Gaussian LL
        error = self._trans_values - mean

        if self._mask is not None:
            error = self._mask(error)  # error may contain nan.
            sigma = self._mask(sigma)  # sigma may contain nan.

        # NOTE: This can be larger than zero as this is PDF, not probability
        return - (self._logn + np.nansum(np.log(sigma)) \
                      + 0.5 * np.nansum((error / sigma) ** 2))

