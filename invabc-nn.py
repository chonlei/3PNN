#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pints
import pints.io
import pints.plot
import method.io
import method.transform as transform
import method.nn as nn
import method.plot as plot

from fix_param import fix_input # Load fix_param.py

"""
Inverse prediction - to predict a posterior distribution of the input parameters 
('model descriptors') that best fits a given EFI data using approximate Bayesian
computation (ABC).


To run this:
    1. Specify the parameters that are fixed in 'fix_param.py'.
    
    2. Specify the unavailable electrodes of the EFI measurements in 
    'data/available-electrodes.csv' and in line 96.
    
    2. Specify the electrode array position information in lines 100, 105 & 107.
    By default, predicton of slimj electrode array is made. (np.linspace(2, 18.5, 16) 
    should be used in line 106 for predictions of 1J electrode array).
    
    3. Specify the 'MAPE_threshold_array' in line 240, which defines the MAPE threshold
    of the intermediate distributions and the final approximate posterior 
    distribution. By default, a final MAPE threshold of 7% is set.
    
    4. Specify the number of samples drawn from the final posterior distribution
    in line 243. By default, 1000 samples are drawn.
    
    5. Run invabc-nn.py, with argument [str:nn_name] and [str:predict_ids.txt].
    
    'nn_name' is the name of the trained NN model, which is the name of the txt 
    file contains the list of training file IDs fitted in fit-nn.py. Note that the 
    first argument is without '.txt'.
    
    'predict_ids.txt' contains a list of file IDs for prediction. Their EFI data
     are stored in the 'input' folder.

Output: All outputs will be saved in './out-nn/[str:nn_name]-inv-predict' folder.
'id_predict_id-samples.csv': 1000 samples of the predicted parameters. (p0: basal 
                            lumen diameter(mm), p1: infill density(%), p2: taper ratio(mm), 
                            p3: cochlear width(mm), p4: cochlear height(mm) and 
                            resistivity(kohm.cm) which converted from p1)
    
"""

try:
    loadas_pre = sys.argv[1] # trained NN name
    input_file = sys.argv[2] # A list containing input IDs to predict
except IndexError:
    print('Usage: python %s [str:nn_name]' % os.path.basename(__file__) \
            + ' [str:predict_ids.txt]')
    sys.exit()
    
path2data = 'data' # folder name of the EFI experimental data
path2input = 'input' # folder name where the input parameter infomration of the training data are stored


# Load the IDs to predict
input_ids = []
with open(input_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

loaddir = './out-nn' # directory of the trained model

# Save directory 
savedir = './out-nn/%s-inv-predict' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

# Load electrode information
# Load unavailable electrodes. 
all_unavailable_electrodes = method.io.load_unavailable_electrodes(
        'data/available-electrodes.csv')
main_unavailable_electrodes = []
main_unavailable_electrodes_idx = np.array(main_unavailable_electrodes) - 1

stim_nodes_all = range(16) # number of electrodes in the electrode array
# Mask unavilable electrodes for prediction
stim_nodes = list(set(stim_nodes_all) - set(main_unavailable_electrodes_idx)) 
# Positions of the electrodes in prediction 
# if 1J, np.linspace(2, 18.5, 16); if slimJ, np.linspace(3, 22.5, 16).
electrode_pos_pred = np.linspace(3, 22.5, 16)  
# Positions of the electrodes in trained model. 1J is used in this study. 
electrode_pos_train = np.linspace(2, 18.5, 16) 
# Positions of electrodes in prediction relative to the positions of electrodes in trained model
stim_relative_position = [(electrode_pos_train[-1] - pred_i)/(electrode_pos_train[1]-electrode_pos_train[0])
                             for pred_i in electrode_pos_pred[::-1]] 


# Create a dictionary of {electrode number:position} in prediction
stim_positions = {}
for i, x in zip(stim_nodes_all[::-1], electrode_pos_pred):
    stim_positions[i+1] = x
stim_pos = [stim_positions[i + 1] for i in stim_nodes]
    
shape = (16, 16) # Shape of EFI profile. 1J EFI profile is 16x16.

# Load transformation fn. z = ln(x + 1). Note that the model takes log-transformed parameters.
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()


# Load trained NN model
print('Loading trained Neural Network models...')
import tensorflow as tf
loadas = loadas_pre + '-stim_all'
trained_nn_model = tf.keras.models.load_model(
            '%s/nn-%s.h5' % (loaddir, loadas))

# Handle fixed parameters
def fix_param(x, fix=fix_input):
    o = np.zeros(len(fix))
    j = 0
    for i in range(len(o)):
        if fix[i] is not None:
            o[i] = np.copy(fix[i])
        else:
            o[i] = x[j]
            j += 1
    return o

def trans_fix_param(x, fix=fix_input, trans=logtransform_x.transform):
    o = np.zeros(len(fix))
    j = 0
    for i in range(len(o)):
        if fix[i] is not None:
            o[i] = trans(np.copy(fix[i]))
        else:
            o[i] = x[j]
            j += 1
    return o

def fit_param(x, fix=fix_input):
    o = []
    for i in range(len(fix)):
        if fix[i] is None:
            o.append(x[i])
    return np.array(o)

n_fit_param = list(fix_input.values()).count(None)

# Create a model for PINTS 
# To perform the search in a log-transformed space (the model takes log-transformed parameters).
model = nn.NNFullModelForPints(trained_nn_model, stim_nodes, stim_relative_position, stim_pos, shape,
        transform_x=None,
        transform=logtransform_y.inverse_transform)

# Boundaries of the input parameters. [p0, p1, p2, p3, p4]
# p0: basal lumen diameter, p1: infill density, p2: taper ratio, p3: cochlear width, p4: cochlear height 
lower_input = [1.98, 0, 0.55, 7.34, 3.53]
upper_input = [2.5, 100, 0.96, 12.66, 4.95]

# Update bounds. Apply log transform.
lower = logtransform_x.transform(lower_input) 
upper = logtransform_x.transform(upper_input)
lower = fit_param(lower) 
upper = fit_param(upper) 
log_prior = pints.UniformLogPrior(lower, upper) # set the prior

def merge_list(l1, l2):
    l1, l2 = list(l1), list(l2)
    out = list(l1 + list(set(l2) - set(l1)))
    out.sort()
    return out

# Go through each input in the input file
for i, input_id in enumerate(input_ids):
    print('Predicting ' + input_id + ' ...')
    saveas = 'id_' + input_id
    fd = path2data + '/' + input_id + '.txt'

    # Load EFI data and mask data from unavailable electrodes
    raw_data = method.io.load(fd)
    filtered_data = method.io.mask(raw_data, x=main_unavailable_electrodes)
    n_readout, n_stimuli = filtered_data.shape

    filter_list = merge_list(main_unavailable_electrodes,
            all_unavailable_electrodes[input_id])
    mask = lambda y: method.io.mask(y, x=filter_list)

    # Inital guess
    guessparams = [2.34, 0, 0.88, 10.53, 4.5]
    transform_guessparams = logtransform_x.transform(guessparams)
    guessparams = fit_param(guessparams)
    transform_guessparams = fit_param(transform_guessparams)

    # Summary statistics
    # Specify the part of EFIs for comparison, only 2-18.5mm along the lumen 
    index_low = stim_relative_position.index(list(filter(lambda k: k >= - 0.1, stim_relative_position))[0])
    index_up = stim_relative_position.index(list(filter(lambda k: k <=16, stim_relative_position))[-1])
  
    summarystats = nn.RootMeanSquaredError(model, raw_data, index_low=index_low, index_up=index_up,
                                           mask=mask, fix=[trans_fix_param, n_fit_param], transform=None)

    print('Summary statistics at guess input parameters: ',
            summarystats(transform_guessparams))
    for _ in range(10):
        assert(summarystats(transform_guessparams) ==\
                summarystats(transform_guessparams))

    try:
        # Load true input value if exists
        fi = path2input + '/' + input_id + '.txt'
        input_value = method.io.load_input(fi)
        transform_input_value = logtransform_x.transform(input_value)
        fit_input_value = fit_param(input_value)
        fit_transform_input_value = fit_param(transform_input_value)
        has_input = True
        print('Summary statistics at true input parameters: ',
                summarystats(fit_transform_input_value))
    
    except (OSError, OSError) as e:
        has_input = False
        
    # Perform ABC inference using PINTS
    abc = pints.ABCController(summarystats, log_prior, method=pints.ABCSMC)
    MAPE_threshold_array = np.array([1,0.5,0.2,0.15,0.1,0.09,0.08,0.07])
    
    abc.sampler().set_threshold_schedule(MAPE_threshold_array)    
    abc.set_n_target(1000) # Number of samples drawn from the final approximate posterior distribution
    abc.set_log_to_screen(True)
    samples = abc.run()

    # De-transform parameters
    samples_param = np.zeros(samples.shape)
    c_tmp = np.copy(samples)
    samples_param[:, :] = logtransform_x.inverse_transform(c_tmp[:, :])
    
    # Save the predicted parameters (de-transformed version)
    none_index = []
    for i in range(len(fix_input)):        
        if fix_input[i] is None:
            none_index.append(1)
        else: 
           none_index.append(0) 
    
    # convert infill density to resistivity (kohm.cm) 
    if none_index[1] == 1 and none_index[0] == 0: 
        p1 = samples_param[:,0]
        header = 'p1, resistivity (kohm.cm)'
    elif none_index[1] == 1 and none_index[0] == 1: 
        p1 = samples_param[:,1] 
        header = 'p0, p1, p2, p3, p4, resistivity (kohm.cm)'
        
    void_pc = 0.4792*(p1/100)+0.0008
    resist = (1/((6.74*10**(-3))*(void_pc-0.035)**1.73))/1000
        
    out = np.column_stack((samples_param,resist))
    del(c_tmp)

    np.savetxt('%s/%s-samples.csv' % (savedir, saveas), out,
               delimiter=',', fmt='%10.4f', header=header)

    # Plot
    if has_input:
        transform_x0 = fit_transform_input_value
        x0 = fit_input_value
    else:
        transform_x0 = None
        x0 = None
    
    # debug
    '''
    pints.plot.histogram([samples_param], ref_parameters=x0)
    plt.savefig('%s/%s-fig.png' % (savedir, saveas))
    plt.close('all')   
    
    pints.plot.histogram([samples], ref_parameters=transform_x0)
    plt.savefig('%s/%s-fig1-transformed.png' % (savedir, saveas))
    plt.close('all')

    if len(x0) > 1:
        pints.plot.pairwise(samples_param, kde=False, ref_parameters=x0)
        plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
        plt.close('all')
    '''

    print('Done.')
