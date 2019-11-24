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
import method.gp as gp
import method.plot as plot

"""
Run regression for the cochlea EFI data with a Gaussian process.
"""

input_ids = ['101315', '170042', '160132', '155808', '144838',
             '141658', '134001', '111641', '135701', '151206']  # TODO
path2data = 'data'
path2input = 'input'

savedir = './out-gp'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = 'test'  # TODO

broken_electrodes = [1, 12, 16]
logtransform = transform.NaturalLogarithmicTransform()
scaletransform = transform.StandardScalingTransform()

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
            X_j = logtransform.transform(input_values[i])
            # NOTE: Here readout index j might want to input position x_j
            X_j = np.append(j, X_j)
            X_jstim.append(X_j)
            y_jstim.append(filtered_data[i][j, j_stim])
X_jstim = np.asarray(X_jstim)
y_jstim = np.asarray(y_jstim)

# TODO: maybe split training and testing data.

# GP fit
k = gp.kernel()
gpr = gp.gaussian_process(k,
        alpha=1e-10,
        n_restarts_optimiser=10,
        random_state=0)
print('Fitting a Gaussian process...')
gpr.fit(X_jstim, y_jstim)
print('Fitted score: ', gpr.score(X_jstim, y_jstim))

# Save fitted GP
joblib.dump(gpr, '%s/gpr-%s.pkl' % (savedir, saveas), compress=3)
# NOTE: gpr = joblib.load('%s/gpr-%s.pkl' % (savedir, saveas))

# Simple check
predict_k = 8
predict_k_x = [np.append(i, logtransform.transform(input_values[predict_k]))
        for i in np.linspace(1, 15, 100)]
predict_k_y = gpr.predict(predict_k_x, return_std=True)
plt.figure()
# Here only index 0 is the readout index
predict_k_x_i = np.asarray(predict_k_x)[:, 0]
plt.plot(predict_k_x_i, predict_k_y[0], c='C0')
plt.plot(predict_k_x_i, predict_k_y[0] + 2 * predict_k_y[1], 'b--')
plt.plot(predict_k_x_i, predict_k_y[0] - 2 * predict_k_y[1], 'b--')
data_k_x_i = range(n_readout)
data_k_y = filtered_data[predict_k][:, j_stim]
plt.plot(data_k_x_i, data_k_y, 'x', c='C1')
plt.savefig('%s/gpr-%s-simple-check' % (savedir, saveas))
plt.close()

