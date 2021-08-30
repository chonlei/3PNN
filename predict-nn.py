#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import method.io
import method.transform as transform
import method.nn as nn
import method.plot as plot

"""
Forward prediction - predicting the EFI profile with a trained neural network.

To run this:
    1. Specify the unavailable electrodes in line 77. 
    
    2. Specify the electrode array position information in lines 80-83. By default, 
    predicton of EFI of 1J electrode array is made. (np.linspace(3, 22.5, 16) 
    should be used for predictions of slimJ electrode array).

    3. Run predict-nn.py, with argument [str:nn_name] and [str:predict_ids.txt].
    
    'nn_name' is the name of the trained NN model, which is the name of the txt 
    file contains the list of training file IDs fitted in fit-nn.py. Note that the 
    first argument is without '.txt'.
    
    'predict_ids.txt' contains a list of file IDs for prediction; Their input
    parameters are stored in the 'input' folder.
    

Output: All outputs will be saved in './out-nn/[str:nn_name]-predict' folder
‘id_predict_id-efi.csv’ : predicted EFI profile
‘id_predict_id-x.csv’ : the position along the cochlear lumen associated with 
                        each transimpednace magnitude entry in the predicted EFI 
                        profile
‘id_predict_id-simple-plot.png’ : compares the predicted EFI and the experimental 
                                  EFI if the experimental EFI is available.
"""

try:
    loadas_pre = sys.argv[1]  # trained NN name
    input_file_ids = sys.argv[2] # A list containing input IDs to predict
except IndexError:
    print('Usage: python %s [str:nn_name]' % os.path.basename(__file__)
          + ' [str:predict_ids.txt]')
    sys.exit()

path2data = 'data' # folder name of the EFI experimental data
path2input = 'input' # folder name where the input parameter infomration of the training data are stored

# Load the IDs to predict
input_ids = []
with open(input_file_ids, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

loaddir = './out-nn/' # directory of the trained model

# Save directory 
savedir = './out-nn/%s-predict/' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)


# Load electrode information
main_unavailable_electrodes = [12,16] # the electrode number not to be included in prediction
# Positions of the electrodes in prediction - if 1J, np.linspace(2, 18.5, 16);
# if slimJ, np.linspace(3, 22.5, 16).
electrode_pos_pred = np.linspace(2, 18.5, 16)
# Positions of the electrodes in trained model. 1J is used in this study.
electrode_pos_train = np.linspace(2, 18.5, 16)
stim_nodes = range(16) # Number of electrodes
# Positions of electrodes in prediction relative to the positions of electrodes in trained model
stim_relative_position = [(electrode_pos_train[-1] - pred_i) / (electrode_pos_train[1] - electrode_pos_train[0])
                          for pred_i in electrode_pos_pred[::-1]]


# Load transformation fn. z = ln(x + 1). Note that the model takes log-transformed parameters.
logtransform_x = transform.NaturalLogarithmicTransform() # x = inputs
logtransform_y = transform.NaturalLogarithmicTransform() # y = transimpedance magnitude,|z|


# Create a dictionary of {electrode number:position} in prediction
stim_positions = {}
for i, x in zip(stim_nodes[::-1], electrode_pos_pred):
    stim_positions[i+1] = x


# Load trained NN model
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

    # Load input parameters
    input_value = method.io.load_input(fi)

    # Load experimental data if available
    # NOTE: We might want to predict new conditions without measurements
    try:
        # if exp. data is available
        raw_data = method.io.load(fd) 
        filtered_data = method.io.mask(raw_data, x=main_unavailable_electrodes)
        n_readout, n_stimuli = filtered_data.shape
        has_data = True
        print('Running validation...')
    except (OSError, OSError) as e:
        # if exp. data is unavailable
        has_data = False
        print('Predicting new conditions...')

    # Create predict output
    predict_stims = []
    predict_xs = []
    predict_y_means = []
    data_xs = []
    data_ys = [] 

    for j_stim, j_stim_pos in zip(stim_nodes, stim_relative_position):
        # j_stim = stimulated electrode number.
        # j_stim_pos = relative pos. of the stimulated electrode.

        if (j_stim + 1) in main_unavailable_electrodes:
            continue

        # laod input parameters and transform
        predict_x = [np.append([i, j_stim_pos], logtransform_x.transform(input_value))
                     for i in electrode_pos_pred]
        predict_x = np.asarray(predict_x).reshape(len(predict_x), -1)
        # Predict transimpedance magnitude using the trained NN model
        predict_y = trained_nn_model.predict(predict_x)
        # Inverse transform prediction and turn it into 1D array.
        predict_y_mean = logtransform_y.inverse_transform(predict_y)
        predict_y_means.append(predict_y_mean[:, 0])
        # Positions of electrodes. Here only index 0 is the readout index
        predict_xs.append(np.asarray(predict_x)[:, 0])
        predict_stims.append(j_stim + 1)

        # Append experimental data for plotting
        if has_data:
            data_i = range(1, n_readout + 1) # number of electrodes
            # Convert electrode number to physical positions
            data_xs.append([stim_positions[i] for i in data_i])
            data_ys.append(filtered_data[:, j_stim])

        if has_data == False:
            n_readout = 16

    predict_xs = np.asarray(predict_xs) # position of electrodes
    predict_y_means = np.asarray(predict_y_means) # predicted EFI profile

    # Plotting experimental data and predicted results
    fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10, 8))
    offset = 10
    c = sns.color_palette('Blues', n_readout + offset)
    cd = sns.color_palette('Oranges', n_readout + offset)

    for stim_e, j_stim_pos in zip(stim_nodes,stim_relative_position):
        ax = axes[(stim_e) // 4, (stim_e) % 4]
        ax.set_title('Stim. %s' % (stim_e + 1))
        ci = stim_e + offset
        # Only plot prediction results at position of 2 - 18.6 mm
        if stim_e+1 in predict_stims and -0.1 <= j_stim_pos <= 16:
            i = predict_stims.index(stim_e + 1)
            ax.plot(predict_xs[i], predict_y_means[i], c=c[ci], label='Prediction')

            if has_data:
                ax.plot(data_xs[i], data_ys[i], 'x', c=cd[ci], label='Experimental data')

        ax.set_xlim([2, 18.6])
        ax.set_xticks(np.linspace(2, 18, 4))

    axes[0,3].legend()
    fig.text(-0.4, 1.2, r'Transimpedence (k$\Omega$)', va='center',
             ha='center', rotation=90, transform=axes[2, 0].transAxes, clip_on=False)
    fig.text(1.1, -0.35, 'Distance from cochlear lumen (mm)',
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
