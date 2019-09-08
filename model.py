#!/usr/bin/env python3
import numpy as np
import pints


#
# Cochlea EFI model
#
class FirstOrderLeakyTransmissionLineNetwork(object):
    """
    This is a cochlea EFI model using the 1st order leaky transmission line
    network [1].

    [1] Vanpoucke FJ, et al., 2004, IEEE Trans Biomed Eng.
    """

    def __init__(self, n_electrodes=16):
        """
        Optional input
        =====
        `n_electrodes`: Number of electrodes, default = 16.
        """
        self.n_e = n_electrodes

        # Set up arithmetic matrix
        self.A = np.matrix(np.zeros((self.n_e, self.n_e + self.n_e - 1)))
        # For R_T
        self.A[[range(self.n_e), range(self.n_e)]] = 1.
        # For R_L
        self.A[[range(self.n_e - 1), self.n_e + np.arange(self.n_e - 1)]] = 1.
        self.A[[range(1, self.n_e), self.n_e + np.arange(self.n_e - 1)]] = -1.

    def n_parameters(self):
        """
        Return number of parameters of the model.
        """
        return self.n_e + self.n_e - 1

    def simulate(self, parameters):
        """
        Return a simulated EFI given the parameters.

        Input
        =====
        `parameters`: [R_T1, R_T2, ..., R_L1, R_L2, ...], where `R_Ti` is the
                      transversal resistance along the i^th electrodes, and 
                      `R_Li` is the longitudinal resistance between the i^th
                      and the (i+1)^th electrodes.
        """
        rt = np.asarray(parameters[:self.n_e], dtype=np.float)
        rl = np.asarray(parameters[self.n_e:], dtype=np.float)

        # Compute conductance matrix
        G = np.matrix(np.zeros((self.n_e + self.n_e - 1,
            self.n_e + self.n_e - 1)))
        # For R_T
        G[[range(self.n_e), range(self.n_e)]] = 1. / rt
        # For R_L
        G[[self.n_e + np.arange(self.n_e - 1),
            self.n_e + np.arange(self.n_e - 1)]] = 1. / rl

        return np.linalg.inv(self.A * G * self.A.T)


#
# Log-likelihood
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
    def __init__(self, model, values, mask=None):
        super(GaussianLogLikelihood, self).__init__()

        self._model = model
        self._values = values
        seff._mask = mask

        # Store counts
        self._np = model.n_parameters() + 1  # assume all has the same sigma
        self._nt = np.nansum(self._mask(np.ones(self._values.shape)))

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def n_parameters(self):
        return self._np

    def __call__(self, x):
        error = self._values - self._model.simulate(x, self._times)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

    def __call__(self, x):
        sigma = x[-1]
        error = self._values - self._model.simulate(x[:-1])
        if self._mask is not None:
            error = self._mask(error)  # error may contain nan.
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.nansum(error ** 2) / (2 * sigma ** 2))

