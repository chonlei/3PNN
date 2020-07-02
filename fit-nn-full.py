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
Run regression for the cochlea EFI data with a neural network.
"""

try:
    input_file = sys.argv[1]
except IndexError:
    print('Usage: python %s [str:input_file]' % os.path.basename(__file__))
    sys.exit()
path2data = 'data'
path2input = 'input'

input_ids = []
with open(input_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

savedir = './out-nn-full'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas_pre = os.path.splitext(os.path.basename(input_file))[0]

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)

all_broken_electrodes = method.io.load_broken_electrodes(
        'data/available-electrodes.csv')
main_broken_electrodes = [12, 16]
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()

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
    d = method.io.mask(raw_data, x=list(all_broken_electrodes[input_id]))
    filtered_data.append(d)
    if i == 0:
        n_readout, n_stimuli = d.shape
    else:
        assert((n_readout, n_stimuli) == d.shape)

# Store the input ID list
with open('%s/nn-%s-training-id.txt' % (savedir, saveas_pre), 'w') as f:
    for i, ii in enumerate(input_ids):
        if i < len(input_ids) - 1:
            f.write(ii + '\n')
        else:
            f.write(ii)

stim_nodes = range(16)
stim_dict = np.loadtxt('./input/stimulation_positions.csv', skiprows=1,
        delimiter=',')
stim_positions = {}
for i, x in stim_dict:
    stim_positions[i] = x
del(stim_dict)

X_jstim = []
y_jstim = []
saveas = saveas_pre + '-stim_all'
for j_stim in stim_nodes:
    if (j_stim + 1) in main_broken_electrodes:
        continue  # TODO: just ignore it?

    for i, input_id in enumerate(input_ids):
        broken_electrodes = list(all_broken_electrodes[input_id])
        for j in range(n_readout):
            if ((j + 1) not in (broken_electrodes + [j_stim + 1])) and \
                    ((j_stim + 1) not in broken_electrodes):
                X_j = logtransform_x.transform(input_values[i])
                stim_j_pos = stim_positions[j + 1]  # convert to phy. position
                X_j = np.append([stim_j_pos, j_stim], X_j)
                X_jstim.append(X_j)
                y_j = logtransform_y.transform(filtered_data[i][j, j_stim])
                y_jstim.append(y_j)
X_jstim = np.asarray(X_jstim)
y_jstim = np.asarray(y_jstim)

# TODO: maybe split training and testing data.
    
# Neural network architecture TODO: need to try other architecture
architecture = [32]
input_neurons = 32
num_layers = 1
activation = 'relu'
input_dim = X_jstim.shape[1]

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
print('Training the neural network for all stimuli...')
trained_nn_model = nn.compile_train_regression_model(
        nn_model,
        X_jstim, # TODO: maybe training data only
        y_jstim,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0)

# Inspect loss function
plt.figure()
plt.semilogy(trained_nn_model.history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Log loss (mean-squared error) of scaled y")
plt.savefig('%s/nn-%s-loss' % (savedir, saveas))

# Save trained NN
trained_nn_model.save('%s/nn-%s.h5' % (savedir, saveas))
# NOTE, to load:
# >>> import tensorflow as tf
# >>> trained_nn_model = tf.keras.models.load_model(
# ...                    '%s/nn-%s.h5' % (savedir, saveas))

# Load saved model
import tensorflow as tf
trained_nn_model_new = tf.keras.models.load_model(
        '%s/nn-%s.h5' % (savedir, saveas))


# Simple plots
for j_stim in stim_nodes:
    if (j_stim + 1) in main_broken_electrodes:
        continue  # TODO: just ignore it?
    
    saveas_j = saveas_pre + '-stim_%s' % (j_stim + 1)
    # Simple check
    predict_k = 2
    predict_k_x = [np.append([i, j_stim], logtransform_x.transform(
                input_values[predict_k])) for i in np.linspace(2, 18.5, 100)]
    predict_k_x = np.asarray(predict_k_x).reshape(len(predict_k_x), -1)
    predict_k_y = trained_nn_model.predict(predict_k_x)
    predict_k_y_mean = logtransform_y.inverse_transform(predict_k_y)
    # Turn it into 1D array
    predict_k_y_mean = predict_k_y_mean[:, 0]
    # Here only index 0 is the readout index
    predict_k_x_i = np.asarray(predict_k_x)[:, 0]
    data_k_i = range(1, n_readout + 1)
    data_k_x_i = [stim_positions[i] for i in data_k_i]  # convert to phy. pos.
    data_k_y = filtered_data[predict_k][:, j_stim]

    plt.figure()
    plt.plot(predict_k_x_i, predict_k_y_mean, c='C0')
    plt.plot(data_k_x_i, data_k_y, 'x', c='C1')
    plt.xlabel('Distance from round window (mm)')
    plt.ylabel(r'Transimpedence (k$\Omega$)')
    plt.savefig('%s/nn-%s-simple-check' % (savedir, saveas_j))
    plt.close()

    # Test saved model
    predict_k_y_new = trained_nn_model_new.predict(predict_k_x)
    predict_k_y_mean_new = logtransform_y.inverse_transform(predict_k_y_new)
    # Turn it into 1D array
    predict_k_y_mean_new = predict_k_y_mean_new[:, 0]

    assert(np.sum(np.abs(predict_k_y_mean - predict_k_y_mean_new)) < 1e-6)

    plt.figure()
    plt.plot(predict_k_x_i, predict_k_y_mean, c='C0')
    plt.plot(predict_k_x_i, predict_k_y_mean_new, c='C2')
    plt.plot(data_k_x_i, data_k_y, 'x', c='C1')
    plt.xlabel('Distance from round window (mm)')
    plt.ylabel(r'Transimpedence (k$\Omega$)')
    plt.savefig('%s/nn-%s-test-saved-model' % (savedir, saveas_j))
    plt.close()
