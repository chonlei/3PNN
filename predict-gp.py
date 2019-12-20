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
import method.gp as gp
import method.plot as plot

"""
Predicting the cochlea EFI data with a trained Gaussian process.
"""

try:
    loadas_pre = sys.argv[1]  # trained gp name
    input_file = sys.argv[2]  # file path containing input ID to predict
except IndexError:
    print('Usage: python %s [str:gp_name]' % os.path.basename(__file__)
            + ' [str:input_file(predict)]')
    sys.exit()
path2data = 'data'
path2input = 'input'

input_ids = []
with open(input_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

loaddir = './out-gp'

savedir = './out-gp/%s-predict' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

broken_electrodes = [1, 12, 16]
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()

stim_nodes = range(16)
stim_dict = np.loadtxt('./input/stimulation_positions.csv', skiprows=1,
        delimiter=',')
stim_positions = {}
for i, x in stim_dict:
    stim_positions[i] = x
del(stim_dict)

# Load trained gp model
trained_gp_models = {}
print('Loading trained Gaussian process models...')
for j_stim in stim_nodes:
    if (j_stim + 1) in broken_electrodes:
        continue  # TODO: just ignore it?
    loadas = loadas_pre + '-stim_%s' % (j_stim + 1)
    trained_gp_models[j_stim + 1] = joblib.load('%s/gpr-%s.pkl' % \
            (loaddir, loadas))

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
    predict_y_uppers = []
    predict_y_lowers = []
    data_xs = []
    data_ys = []
    for j_stim in stim_nodes:
        if (j_stim + 1) in broken_electrodes:
            continue  # TODO: just ignore it?

        gpr = trained_gp_models[j_stim + 1]

        predict_x = [np.append(i, logtransform_x.transform(input_value))
                for i in np.linspace(2, 18.5, 100)]
        predict_y = gpr.predict(predict_x, return_std=True)
        predict_y_mean = logtransform_y.inverse_transform(predict_y[0])
        predict_y_upper = logtransform_y.inverse_transform(predict_y[0]
                + 2 * predict_y[1])
        predict_y_lower = logtransform_y.inverse_transform(predict_y[0]
                - 2 * predict_y[1])  # assymetric bound
        # Store
        predict_y_means.append(predict_y_mean)
        predict_y_uppers.append(predict_y_upper)
        predict_y_lowers.append(predict_y_lower)
        # Here only index 0 is the readout index
        predict_xs.append(np.asarray(predict_x)[:, 0])
        predict_stims.append(j_stim + 1)

        if has_data:
            data_i = range(1, n_readout + 1)
            # Convert to physical positions
            data_xs.append([stim_positions[i] for i in data_i])
            data_ys.append(filtered_data[:, j_stim])
    predict_xs = np.asarray(predict_xs)
    predict_y_means = np.asarray(predict_y_means)
    predict_y_uppers = np.asarray(predict_y_uppers)
    predict_y_lowers = np.asarray(predict_y_lowers)

    # Plot
    plt.figure()
    c = sns.color_palette('Blues', n_readout)
    cd = sns.color_palette('Oranges', n_readout)
    for i, j_stim in enumerate(predict_stims):
        plt.plot(predict_xs[i], predict_y_means[i], c=c[j_stim])
        plt.plot(predict_xs[i], predict_y_uppers[i], c=c[j_stim], ls='--')
        plt.plot(predict_xs[i], predict_y_lowers[i], c=c[j_stim], ls='--')
        if has_data:
            plt.plot(data_xs[i], data_ys[i], 'x', c=cd[j_stim])
    plt.xlabel('Distance from round window (mm)')
    plt.ylabel(r'Transimpedence (k$\Omega$)')
    plt.savefig('%s/%s-simple-plot' % (savedir, saveas), dpi=200)
    plt.close()

    # Save predictions
    save_header = ','.join(['\"Stim_%s\"' % i for i in predict_stims])
    np.savetxt('%s/%s-x.csv' % (savedir, saveas), predict_xs.T, delimiter=',',
            comments='', header=save_header)
    np.savetxt('%s/%s-efi.csv' % (savedir, saveas), predict_y_means.T,
            delimiter=',', comments='', header=save_header)
    np.savetxt('%s/%s-efi-upper.csv' % (savedir, saveas), predict_y_uppers.T,
            delimiter=',', comments='', header=save_header)
    np.savetxt('%s/%s-efi-lower.csv' % (savedir, saveas), predict_y_lowers.T,
            delimiter=',', comments='', header=save_header)

