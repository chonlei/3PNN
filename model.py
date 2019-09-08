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

    def __init__(self, n_electrodes=16, transform=None):
        """
        Optional input
        =====
        `n_electrodes`: Number of electrodes, default = 16.
        `transform`: Transformation of search space parameters to model
                     parameters.
        """
        self.n_e = n_electrodes
        self.transform = transform

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
                      and the (i+1)^th electrodes. Parameter units are all in
                      kiloOhms.
        """
        if self.transform is not None:
            parameters = self.transform(parameters)
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

        # try: ? 
        return np.linalg.inv(self.A * G * self.A.T)


#
# Parameter transformation
#
def log_transform_from_model_param(param):
    # Apply natural log transformation to model parameters
    out = np.copy(param)
    out[:] = np.log(out[:])
    return out


def log_transform_to_model_param(param):
    # Inverse of log_transform_from_model_param()
    # Apply natural exp transformation to model parameters
    out = np.copy(param)
    out[:] = np.exp(out[:])
    return out


def donothing(param):
    out = np.copy(param)
    return out


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
        """
        Input
        =====
        `model`: The cochlea model, following the ForwardModel requirements in
                 PINTS.
        `values`: The data that match the output of the `model` simulation.
        `mask`: A function that takes in the data and replace undesired 
                entries with `nan`.
        """
        super(GaussianLogLikelihood, self).__init__()

        self._model = model
        self._values = values
        self._mask = mask

        # Store counts
        self._np = model.n_parameters() + 1  # assume all has the same sigma
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
        sigma = x[-1]
        error = self._values - self._model.simulate(x[:-1])
        if self._mask is not None:
            error = self._mask(error)  # error may contain nan.
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.nansum(error ** 2) / (2 * sigma ** 2))


#
# Log-prior
#
class UniformLogPrior(pints.LogPrior):
    """
    Unnormalised uniform prior for the cochlea model for PINTS [1].

    [1] Clerx M, et al., 2019, JORS.
    """
    def __init__(self, n_electrodes=16, transform=donothing,
            inv_transform=donothing):
        """
        Optional input
        =====
        `n_electrodes`: Number of electrodes, default = 16.
        `transform`: Parameter transformation to model parameters.
        `inv_transform`: Parameter transformation from model parameters.
        """
        super(UniformLogPrior, self).__init__()

        self.n_e = n_electrodes

        self.lower = np.array(
                [5.0] * (self.n_e - 1)  # R_T, kOhms
                + [1.0]  # R_basel, kOhms
                + [0.01] * (self.n_e - 1)  # R_L, kOhms
                )

        self.upper = np.array(
                [50.0] * (self.n_e - 1)  # R_T, kOhms
                + [10.0]  # R_basel, kOhms
                + [1.6] * (self.n_e - 1)  # R_L, kOhms
                )

        self.minf = -float('inf')

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        """
        Return number of parameters.
        """
        return self.n_e + self.n_e - 1

    def __call__(self, parameters):
        """
        Return the computed unnormalised log-prior for the given `parameters`.
        """
        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Return
        return 0

    def sample(self, n=1):
        """
        Return `n` parameter samples, sampled from this log-prior.
        """
        out = np.zeros((n, self.n_parameters()))

        for i in range(n):
            p = np.zeros(self.n_parameters())

            p[:] = self.transform(np.random.uniform(
                self.inv_transform(self.lower[:-1]),
                self.inv_transform(self.upper[:-1])))

            out[i, :] = self.inv_transform(p)

        # Return
        return out

