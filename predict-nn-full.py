#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import method.io
import method.transform as transform
import method.nn as nn
import method.plot as plot

"""
Predicting the cochlea EFI data with a trained neural network.
"""

try:
    loadas_pre = sys.argv[1]  # trained nn name
    input_file = sys.argv[2]  # file path containing input ID to predict
except IndexError:
    print('Usage: python %s [str:nn_name]' % os.path.basename(__file__)
            + ' [str:input_file(predict)]')
    sys.exit()
path2data = 'data'
path2input = 'input'

input_ids = []
with open(input_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

loaddir = './out-nn-full'

savedir = './out-nn-full/%s-predict/' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)

broken_electrodes = [12, 16]
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()

stim_nodes = range(16)
stim_dict = np.loadtxt('./input/stimulation_positions.csv', skiprows=1,
        delimiter=',')
stim_positions = {}
for i, x in stim_dict:
    stim_positions[i] = x
del(stim_dict)

# Load trained nn model
import tensorflow as tf
print('Loading trained Neural Network models...')
loadas = loadas_pre + '-stim_all'
trained_nn_model = tf.keras.models.load_model(
            '%s/nn-%s.h5' % (loaddir, loadas))


# Go through each input in the input file
for i, input_id in enumerate(input_ids):
    print('Predicting ' + input_id + ' ...')
    saveas = 'id_' + input_id
    fi = path2input + '/' + input_id + '.txt'
    fd = path2data + '/' + input_id + '.txt'

    # Load input values
    input_value = method.io.load_input(fi)

    # Load data
    # NOTE: We might want to predict new conditions without measurements
    try:
        raw_data = method.io.load(fd)
        filtered_data = method.io.mask(raw_data, x=broken_electrodes)
        n_readout, n_stimuli = filtered_data.shape
        has_data = True
        print('Running validation...')
    except FileNotFoundError:
        has_data = False
        print('Predicting new conditions...')

    # Predict
    predict_stims = []
    predict_xs = []
    predict_y_means = []
    data_xs = []
    data_ys = []
    
    for j_stim in stim_nodes:
        if (j_stim + 1) in broken_electrodes:
            continue  # TODO: just ignore it?
        
        predict_x = [np.append([i, j_stim], logtransform_x.transform(
                    input_value)) for i in np.linspace(2, 18.5, 100)]
        predict_y = [trained_nn_model.predict(x.reshape(1, -1))
            for x in predict_x]
        
        predict_y_mean = logtransform_y.inverse_transform(predict_y)
        # Turn it into 1D array
        predict_y_means.append(predict_y_mean[:, 0])
        # Here only index 0 is the readout index
        predict_xs.append(np.asarray(predict_x)[:, 0])
        predict_stims.append(j_stim + 1)
        

        
        if has_data:
            data_i = range(1, n_readout + 1)
            # Convert to physical positions
            data_xs.append([stim_positions[i] for i in data_i])
            data_ys.append(filtered_data[:, j_stim])
    predict_xs = np.asarray(predict_xs)

    predict_y_means = np.asarray(predict_y_means)[:, :, 0]


    # Plot
    fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10, 8))
    offset = 10
    c = sns.color_palette('Blues', n_readout + offset)
    cd = sns.color_palette('Oranges', n_readout + offset)
    for i, j_stim in enumerate(predict_stims):
        ax = axes[j_stim // 4, j_stim % 4]
        ci = j_stim + offset
        ax.set_title('Stim %s' % (j_stim + 1))
        ax.plot(predict_xs[i], predict_y_means[i], c=c[ci])
        if has_data:
            ax.plot(data_xs[i], data_ys[i], 'x', c=cd[ci])
    fig.text(-0.4, 1.2, r'Transimpedence (k$\Omega$)', va='center',
            ha='center', rotation=90, transform=axes[2, 0].transAxes, clip_on=False)
    fig.text(1.1, -0.35, 'Distance from round window (mm)',
            va='center', ha='center', transform=axes[-1, 1].transAxes, clip_on=False)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig('%s/%s-simple-plot' % (savedir, saveas), dpi=300)
    plt.close()
    # Save predictions
    save_header = ','.join(['\"Stim_%s\"' % i for i in predict_stims])
    np.savetxt('%s/%s-x.csv' % (savedir, saveas), predict_xs.T, delimiter=',',
            comments='', header=save_header)
    np.savetxt('%s/%s-efi.csv' % (savedir, saveas), predict_y_means.T,
            delimiter=',', comments='', header=save_header)

