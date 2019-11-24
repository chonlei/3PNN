#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

import method.io
import method.transform as transform
import method.nn as nn
import method.plot as plot

"""
Run regression for the cochlea EFI data with a Gaussian process.
"""

input_ids = ['101315', '170042', '160132', '155808', '144838',
             '141658', '134001', '111641', '135701', '151206']  # TODO
path2data = 'data'
path2input = 'input'

savedir = './out-nn'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = 'test'  # TODO

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)

broken_electrodes = [1, 12, 16]
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()
scaletransform_x = transform.StandardScalingTransform()
scaletransform_y = transform.StandardScalingTransform()

# Load EFI data
input_values = []
filtered_data = []
for i, input_id in enumerate(input_ids):
    fi = path2input + '/' + input_id + '.txt'
    fd = path2data + '/' + input_id + '.txt'

    # Load input values
    input_values.append(method.io.load_input(fi))

    # Load data
    raw_data = method.io.load(fd)
    d = method.io.mask(raw_data, x=broken_electrodes)
    filtered_data.append(d)
    if i == 0:
        n_readout, n_stimuli = d.shape
    else:
        assert((n_readout, n_stimuli) == d.shape)

j_stim = 2  # TODO

X_jstim = []
y_jstim = []
for i in range(len(input_ids)):  # TODO
    for j in range(n_readout):
        if (j + 1) not in broken_electrodes + [j_stim + 1]:
            X_j = logtransform_x.transform(input_values[i])
            # NOTE: Here readout index j might want to input position x_j
            X_j = np.append(j, X_j)
            X_jstim.append(X_j)
            y_j = logtransform_y.transform(filtered_data[i][j, j_stim])
            y_jstim.append(y_j)
X_jstim = np.asarray(X_jstim)
y_jstim = np.asarray(y_jstim)

# Turn into 2D array
y_jstim = y_jstim.reshape(y_jstim.size, 1)

# Scale data
X_jstim_scaled = scaletransform_x.fit_transform(X_jstim)
y_jstim_scaled = scaletransform_y.fit_transform(y_jstim)

# TODO: maybe split training and testing data.

# TODO: get a better estimate of the noise level
noise_level = 1e-4  # SD of the transimpedence measurements

# Neural network architecture TODO
architecture = [32]
input_neurons = 32
num_layers = 1
activation = 'relu'
input_dim = X_jstim_scaled.shape[1]

# Neural network epochs and batch size
batch_size = 4
epochs = 250

# NN fit
nn_model = nn.build_regression_model(
        input_neurons=input_neurons,
        num_layers=num_layers,
        architecture=architecture,
        input_dim=input_dim,
        act_func=activation)
nn_model.summary()
print('Training the neural network...')
trained_nn_model = nn.compile_train_regression_model(
        nn_model,
        X_jstim_scaled, # TODO maybe training data only
        y_jstim_scaled,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0)

# Inspect loss function
plt.figure()
plt.semilogy(trained_nn_model.history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Log loss (mean-squared error) of scaled y")
plt.savefig('%s/nn-%s-loss' % (savedir, saveas))

# Save trained NN TODO
trained_nn_model.save('%s/nn-%s.h5' % (savedir, saveas))
# NOTE, to load:
# >>> import tensorflow as tf
# >>> trained_nn_model = tf.keras.models.load_model(
# ...                    '%s/nn-%s.h5' % (savedir, saveas))

# Simple check
predict_k = 8
predict_k_x = [np.append(i, logtransform_x.transform(input_values[predict_k]))
        for i in np.linspace(1, 15, 100)]
predict_k_y_scaled = trained_nn_model.predict(
        scaletransform_x.transform(predict_k_x))
predict_k_y = scaletransform_y.inverse_transform(predict_k_y_scaled)
predict_k_y_mean = logtransform_y.inverse_transform(predict_k_y)
# Turn it into 1D array
predict_k_y_mean = predict_k_y_mean[:, 0]
# Here only index 0 is the readout index
predict_k_x_i = np.asarray(predict_k_x)[:, 0]
data_k_x_i = range(n_readout)
data_k_y = filtered_data[predict_k][:, j_stim]

plt.figure()
plt.plot(predict_k_x_i, predict_k_y_mean, c='C0')
plt.plot(data_k_x_i, data_k_y, 'x', c='C1')
plt.xlabel('Electrode #')
plt.ylabel(r'Transimpedence (k$\Omega$)')
plt.savefig('%s/nn-%s-simple-check' % (savedir, saveas))
plt.close()

