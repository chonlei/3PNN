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
Run regression for the EFI data obtained from 3D printed models with a neural network.

The EFI data of the training dataset should be stored in './data' folder, and 
the input parameters of the EFI data should be stored in './input' folder.

To run this:
    1. Specify the unavailable electrodes of each experimental EFI measurement 
    in 'data/available-electrodes.csv' and in line 68.
    
    2. Specify the position information of the electrode array used to acquire
    the experimental EFI measurements in line 74. By default, 1J electrode array
    is used.
    
    3. Run fit-nn.py, with argument [str:input_file_ids.txt] containing a list of 
    file IDs of the training data. 
    
Output: a NN model file (nn-[str:input_file_ids.txt]-stim_all.h5) and a graph showing
        the loss function vs number of epiochs will be saved in 'out-nn' folder.
"""

try:
    input_file_ids = sys.argv[1] # A list of file IDs of the training data
except IndexError:
    print('Usage: python %s [str:input_file_ids.txt]' % os.path.basename(__file__))
    sys.exit()
    
path2data = 'data' # folder name of the EFI experimental data
path2input = 'input' # folder name where the input parameter information of the training data are stored

# Load the IDs of the training data
input_ids = []
with open(input_file_ids, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

# Save directory 
savedir = './out-nn' 
if not os.path.isdir(savedir):
    os.makedirs(savedir)
    
saveas_pre = os.path.splitext(os.path.basename(input_file_ids))[0]

# Control fitting seed
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)

# Load electrode information
# Load unavailable electrodes. 
all_unavailable_electrodes = method.io.load_unavailable_electrodes(
        'data/available-electrodes.csv') 
main_unavailable_electrodes = [12, 16] 

# Positions (along the cochlear lumen) of the electrode array used to acquire 
# the training data. The training data of this study was measured by Advanced
# Bionics 1J electrode array, which has 16 electrodes with electrode position 
# 2-18.5mm.
electrode_pos_train = np.linspace(2, 18.5, 16) 

# Load transformation fn. z = ln(x + 1). Note that the model takes log-transformed parameters.
logtransform_x = transform.NaturalLogarithmicTransform() # x = inputs
logtransform_y = transform.NaturalLogarithmicTransform() # y = transimpedance magnitude,|z|

# Load EFI training data & input parameters
input_values = []
filtered_data = []
for i, input_id in enumerate(input_ids):
    fi = path2input + '/' + input_id + '.txt'
    fd = path2data + '/' + input_id + '.txt'

    # Load input parameters of the EFI training data
    input_values.append(method.io.load_input(fi))

    # Load data
    raw_data = method.io.load(fd)
    d = method.io.mask(raw_data, x=list(all_unavailable_electrodes[input_id]))
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

stim_nodes = range(16) # number of electrodes 
# Create a dictionary of {electrode number:position}

stim_positions = {}
for i, x in zip(stim_nodes[::-1], electrode_pos_train):
    stim_positions[i+1] = x

# Apply transformation fn. to input parameters and EFIs of the training data
X_jstim = []
y_jstim = []
saveas = saveas_pre + '-stim_all'
for j_stim in stim_nodes:
    if (j_stim + 1) in main_unavailable_electrodes:
        continue  

    for i, input_id in enumerate(input_ids):
        unavailable_electrodes = list(all_unavailable_electrodes[input_id])
        for j in range(n_readout):
            if ((j + 1) not in (unavailable_electrodes + [j_stim + 1])) and \
                    ((j_stim + 1) not in unavailable_electrodes):
                X_j = logtransform_x.transform(input_values[i])
                stim_j_pos = stim_positions[j + 1]  # convert to phy. position
                X_j = np.append([stim_j_pos, j_stim], X_j)
                X_jstim.append(X_j)
                y_j = logtransform_y.transform(filtered_data[i][j, j_stim])
                y_jstim.append(y_j)         
         
X_jstim = np.asarray(X_jstim)
y_jstim = np.asarray(y_jstim)

# Neural network architecture (optimised in 10-fold cross-validation). 
num_layers = 1
input_neurons = 32
architecture = [input_neurons]*num_layers

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

# Trained NN model
trained_nn_model = nn.compile_train_regression_model(
        nn_model,
        X_jstim, 
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
 
# Save trained NN model
trained_nn_model.save('%s/nn-%s.h5' % (savedir, saveas))

print('Done. NN model is saved.')

