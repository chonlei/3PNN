#!/usr/bin/env python3
import numpy as np

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

