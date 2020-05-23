#
# This module contains functions of neural network to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
import tensorflow as tf


def build_regression_model(input_neurons=10, input_dim=1, num_layers=1,
        architecture=[32], act_func="relu"):
    """
    Builds a densely connected neural network model.

    Arguments
        input_neurons: Number of input neurons.
        input_dim: Dimension of the input vector.
        num_layers: Number of hidden layers.
        architecture: Architecture of the hidden layers (densely connected).
        act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.
    Returns
        A neural net (Keras) model for regression.
    """
    if act_func == "relu":
        activation = tf.nn.relu
    elif act_func == "sigmoid":
        activation = tf.nn.sigmoid
    elif act_func == "tanh":
        activation = tf.nn.tanh

    layers = [tf.keras.layers.Dense(input_neurons, input_dim=input_dim,
            activation=activation)]
    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(architecture[i],
                activation=activation))
    layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.models.Sequential(layers)
    return model


def compile_train_regression_model(model, x_train, y_train, callbacks=None,
        learning_rate=0.001, batch_size=1, epochs=10, verbose=0):
    """
    Compiles and trains a given Keras ``model`` with the given data
    (``x_train``, ``y_train``) for regression. Assumes Adam optimizer for this
    implementation. Assumes mean-squared-error loss.
      
    Arguments
        learning_rate: Learning rate for the optimizer Adam.
        batch_size: Batch size for the mini-batch operation.
        epochs: Number of epochs to train.
        verbose: Verbosity of the training process.
      
    Returns
        A copy of the trained model.
    """

    model_copy = model
    model_copy.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss="mean_squared_error",
            metrics=["accuracy"],
        )
    if callbacks != None:
        model_copy.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[callbacks],
                verbose=verbose,
            )
    else:
        model_copy.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
    return model_copy


#
# PINTS Model
#
class NNModelForPints(object):
    """
    A wrapper for the scikit-learn NN model for PINTS inverse problem.

    Expect the NN model was fitted individually/independently to each stimulus.
    """
    def __init__(self, nn_model, stim_idx, stim_pos, shape, transform_x=None,
                 transform=None):
        """
        Input
        =====
        `nn_model`: (dict) A dictionary with key = j-stimulus and
                    value = scikit-learn NN model fitted independently to the
                    j-stimulus.
        `stim_idx`: (array) Stimulus indices, usually [0, 1, 2, ..., 15].
        `stim_pos`: (array) Stimulus position, corresponding to the measurement
                    positions.
        `shape`: (tuple) Shape of simulate() output matrix.

        Optional input
        =====
        `transform_x`: (dict) Transformation of NN input from search space,
                       with key = j-stimulus and value = transformation
                       function for the j-stimulus.
        `transform`: Transformation of NN output to EFI output.
        """
        super(NNModelForPints, self).__init__()
        self.nn = nn_model
        #self._np = self.nn[stim_idx[0] + 1].X_train_.shape[1]
        self._np = 6   # TODO
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

    def simulate(self, x):
        """
        Return a simulated EFI given the parameters `x`.
        """
        out = np.zeros(self._shape)

        for j_stim in self._stim_idx:
            nn_j = self.nn[j_stim + 1]
            transform_x_j = self._transform_x[j_stim + 1]

            predict_x = [np.append(i, x) for i in self._stim_pos]
            predict_x = transform_x_j(predict_x)
            y = nn_j.predict(predict_x)

            out[self._stim_idx, j_stim] = self._transform(y).reshape(len(y))

        return out


#
# Summary statistics for ABC
#
class RootMeanSquaredError(pints.ErrorMeasure):
    """
    Define a RMSE summary statistics for the problem for PINTS [1].

    .. math::
        RMSE(\theta, \boldsymbol{x}) =
            \sum_{j=1}^{n_t}{(x_j - f_j(\theta))^2}

    where ``n_t`` is the number of measurement points, ``x_j`` is the
    sampled data at ``j`` and ``f_j`` is the simulated data at ``j``.

    [1] Clerx M, et al., 2019, JORS.
    """
    def __init__(self, model, values, mask=None, fix=None, transform=None):
        """
        Input
        =====
        `model`: A NN model, following the ForwardModel requirements in
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
        `transform`: Transformation of EFI output (data) to NN output; if
                     the comparison would like to be performed in NN output.
        """
        super(RootMeanSquaredError, self).__init__()

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
        self._c = 1. / self._nt

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
        mean = self._model.simulate(x)
        # Compare transformed values so that sigma makes sense in Gaussian LL
        error = self._trans_values - mean

        if self._mask is not None:
            error = self._mask(error)  # error may contain nan.

        e = np.sqrt(np.nansum(error ** 2) * self._c)

        if np.isfinite(e):
            return e
        else:
            return 1e10

