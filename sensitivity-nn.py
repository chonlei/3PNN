#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from SALib.sample import saltelli
from SALib.analyze import sobol
import method.io
import method.transform as transform
import method.nn as nn
import method.plot as plot
from method.feature import baseline, curve_fit, PowerLawBaseline, peaks

"""
Sensitivity analysis - analysing the EFI profile sensitivity with a trained neural network.

To run this:
    1. Specify the electrodes not to be included in prediction in line 63. 
    
    2. Specify the electrode array position information in lines 65-68. By default, 
    predicton of EFI of 1J electrode array is made. (np.linspace(3, 22.5, 16) 
    should be used for predictions of slimJ electrode array).

    3. Run sensitivity-nn.py, with argument [str:nn_name].
    
    'nn_name' is the name of the trained NN model, which is the name of the txt 
    file contains the list of training file IDs fitted in fit-nn.py. Note that the 
    first argument is without '.txt'.

Output: All outputs will be saved in './out-nn/[str:nn_name]-sensitivity' folder
‘A_left_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                             indices for the coefficient A of the [i]th 
                                             stimulus spread toward the base-side
                                             of cochlea.
‘A_right_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                              indices for the coefficient A of the [i]th 
                                              stimulus spread toward the apex-side
                                              of cochlea.
‘b_left_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                             indices for the coefficient b of the [i]th 
                                             stimulus spread toward the base-side
                                             of cochlea.
‘b_right_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                              indices for the coefficient b of the [i]th 
                                              stimulus spread toward the apex-side
                                              of cochlea.
‘Ab_left_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                              indices for the coefficient product Ab of the 
                                              [i]th stimulus spread toward the base-side
                                              of cochlea.
‘Ab_right_stim_[i]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                               indices for the coefficient product Ab of the 
                                               [i]th stimulus spread toward the apex-side
                                               of cochlea.
‘EFI_mega_i[i]_j[j]_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order
                                                indices for the EFI matrix at entry [i],[j],
                                                where i is the stimulating electrode number
                                                and j is the recording electrode number. 
‘peak_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order indices for
                                  the peak of the EFI.
‘baseline_[first|second|total].csv’ : Sobol sensitivity [first|second|total]-order indices for
                                      the baseline of the EFI.
"""

try:
    loadas_pre = sys.argv[1]  # trained NN name
except IndexError:
    print('Usage: python %s [str:nn_name]' % os.path.basename(__file__))
    sys.exit()

loaddir = './out-nn/' # directory of the trained model

# Save directory 
savedir = './out-nn/%s-sensitivity/' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)
nn.tf.random.set_seed(fit_seed)


# Load electrode information
main_unavailable_electrodes = [] # the electrode number not to be included in prediction
# Positions of the electrodes in prediction - if 1J, np.linspace(2, 18.5, 16);
# if slimJ, np.linspace(3, 22.5, 16).
electrode_pos_pred = np.linspace(2, 18.5, 16)  


# Positions of the electrodes in trained model. 1J is used in this study. 
electrode_pos_train = np.linspace(2, 18.5, 16) 
stim_nodes = range(16) # Number of electrodes 
# Positions of electrodes in prediction relative to the positions of electrodes in trained model
stim_relative_position = [(electrode_pos_train[-1] - pred_i)/(electrode_pos_train[1]-electrode_pos_train[0]) for pred_i in electrode_pos_pred[::-1]] 


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


# Sensitivity analysis boundaries
lower = [1.98, 20, 0.58, 7.34, 3.53]
upper = [2.50, 100, 0.89, 12.66, 4.95]

# SALib problem setting
problem = {
    'num_vars': 5,
    'names': ['x1', 'x2', 'x3', 'x4', 'x5'],
    'bounds': np.array([lower, upper]).T
}

param_values = saltelli.sample(problem, 14000)


# Go through each input in the samples
baselines = []
peak2s = []
Als = [] # Coefficient A towards base
Ars = [] # Coefficient A towards apex
Bls = [] # Coefficient b towards base
Brs = [] # Coefficient b towards apex
ABls = [] # Coefficient Ab towards base
ABrs = [] # Coefficient Ab towards apex
ABl_means = [] # Mean coefficient Ab towards base
ABr_means = [] # Mean coefficient Ab towards apex
EFI_mega = []

powerlaw = PowerLawBaseline()

predict_stims = []
for j_stim, j_stim_pos in zip(stim_nodes, stim_relative_position):
    if (j_stim + 1) in main_unavailable_electrodes:
        continue
    predict_stims.append(j_stim + 1)

for i_param, param in enumerate(param_values):
    # Create predict output
    predict_y_means = []

    for j_stim, j_stim_pos in zip(stim_nodes, stim_relative_position): 
        # j_stim = stimulated electrode number.
        # j_stim_pos = relative pos. of the stimulated electrode.

        if (j_stim + 1) in main_unavailable_electrodes:
            continue 

        # laod input parameters and transform
        predict_x = [np.append([i, j_stim_pos], logtransform_x.transform(
                    param)) for i in electrode_pos_pred]
        predict_x = np.asarray(predict_x).reshape(len(predict_x), -1)
        # Predict transimpedance magnitude using the trained NN model
        predict_y = trained_nn_model.predict(predict_x)         
        # Inverse transform prediction and turn it into 1D array. 
        predict_y_mean = logtransform_y.inverse_transform(predict_y)
        predict_y_means.append(predict_y_mean[:, 0]) 

    predict_xs = np.asarray(predict_x)[:, 0] # position of electrodes
    predict_y_means = np.asarray(predict_y_means).T # predicted EFI profile

    # Compute QoIs:
    b = baseline(predict_y_means, method=2)  # minimum value of the whole EFI
    baselines.append(b)

    p = np.max(peaks(predict_y_means))
    peak2s.append(p)

    EFI_mega.append(predict_y_means)

    powerlaw.set_baseline(b)
    y = np.full((16, 16), np.NaN)  # pad with NaN for curve_fit
    y[:, np.array(predict_stims) - 1] = predict_y_means
    cc = curve_fit(y[:, ::-1], powerlaw, predict_xs)
    Al = []
    Ar = []
    Bl = []
    Br = []
    ABl = []
    ABr = []

    for i in predict_stims:
        j = 16 - i  # stimulus order reversed

        if cc[j][1] is not None:
            Al.append(cc[j][1][0])
            Bl.append(cc[j][1][1])
            ABl.append(cc[j][1][0]*cc[j][1][1])
        else:
            Al.append(np.NaN)
            Bl.append(np.NaN)
            ABl.append(np.NaN)

        if cc[j][0] is not None:
            Ar.append(cc[j][0][0])
            Br.append(cc[j][0][1])
            ABr.append(cc[j][0][0]*cc[j][0][1])
        else:
            Ar.append(np.NaN)
            Br.append(np.NaN)
            ABr.append(np.NaN)

    Als.append(Al)
    Ars.append(Ar)
    Bls.append(Bl)
    Brs.append(Br)
    ABls.append(ABl)
    ABrs.append(ABr)
    ABl_means.append(np.nanmean(ABl))
    ABr_means.append(np.nanmean(ABr))

    if (i_param % 10) == 0:
        print(i_param)

# To numpy array
baselines = np.array(baselines)
peak2s = np.array(peak2s)
Als = np.array(Als)
Ars = np.array(Ars)
Bls = np.array(Bls)
Brs = np.array(Brs)
ABls = np.array(ABls)
ABrs = np.array(ABrs)
ABl_means = np.array(ABl_means)
ABr_means = np.array(ABr_means)
EFI_mega = np.array(EFI_mega)


# Sensitivity analysis for baseline value
baseline_Si = sobol.analyze(problem, baselines)
total_Si, first_Si, second_Si = baseline_Si.to_df()
total_Si.to_csv('%s/baseline_total.csv' % (savedir))
first_Si.to_csv('%s/baseline_first.csv' % (savedir))
second_Si.to_csv('%s/baseline_second.csv' % (savedir))

# Sensitivity analysis for mean AB value towards base
ABl_means_Si = sobol.analyze(problem, ABl_means)
total_Si, first_Si, second_Si = ABl_means_Si.to_df()
total_Si.to_csv('%s/mean_AB_left_total.csv' % (savedir))
first_Si.to_csv('%s/mean_AB_left_first.csv' % (savedir))
second_Si.to_csv('%s/mean_AB_left_second.csv' % (savedir))

# Sensitivity analysis for mean AB value towards apex
ABr_means_Si = sobol.analyze(problem, ABr_means)
total_Si, first_Si, second_Si = ABr_means_Si.to_df()
total_Si.to_csv('%s/mean_AB_right_total.csv' % (savedir))
first_Si.to_csv('%s/mean_AB_right_first.csv' % (savedir))
second_Si.to_csv('%s/mean_AB_right_second.csv' % (savedir))

# Sensitivity analysis for peak value
peak_Si = sobol.analyze(problem, peak2s)
total_Si, first_Si, second_Si = peak_Si.to_df()
total_Si.to_csv('%s/peak_total.csv' % (savedir))
first_Si.to_csv('%s/peak_first.csv' % (savedir))
second_Si.to_csv('%s/peak_second.csv' % (savedir))


# Sensitivity analysis for EFI matrix
for i in range(np.array(EFI_mega).shape[1]):
    for j in range(np.array(EFI_mega).shape[2]):
        EFI_ij_Si = sobol.analyze(problem, EFI_mega[:,i,j])
        total_Si, first_Si, second_Si = EFI_ij_Si.to_df()
        total_Si.to_csv('%s/EFI_mega_i%s_j%s_total.csv' % (savedir, i, j))
        first_Si.to_csv('%s/EFI_mega_i%s_j%s_first.csv' % (savedir, i, j))
        second_Si.to_csv('%s/EFI_mega_i%s_j%s_second.csv' % (savedir, i, j))


# Sensitivity analysis for coefficients A, b in |z| = A|x|^{-b} + baseline
for i in range(len(stim_nodes) - len(main_unavailable_electrodes)):
    if all(np.isfinite(Als[:, i])) and all(np.isfinite(Bls[:, i])):
        Ali_Si = sobol.analyze(problem, Als[:, i])
        Bli_Si = sobol.analyze(problem, Bls[:, i])
        ABli_Si = sobol.analyze(problem, ABls[:, i])

        total_Si, first_Si, second_Si = Ali_Si.to_df()
        total_Si.to_csv('%s/A_left_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/A_left_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/A_left_stim_%s_second.csv' % (savedir, predict_stims[i]))

        total_Si, first_Si, second_Si = Bli_Si.to_df()
        total_Si.to_csv('%s/b_left_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/b_left_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/b_left_stim_%s_second.csv' % (savedir, predict_stims[i]))

        total_Si, first_Si, second_Si = ABli_Si.to_df()
        total_Si.to_csv('%s/Ab_left_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/Ab_left_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/Ab_left_stim_%s_second.csv' % (savedir, predict_stims[i]))

    if all(np.isfinite(Ars[:, i])) and all(np.isfinite(Brs[:, i])):
        Ari_Si = sobol.analyze(problem, Ars[:, i])
        Bri_Si = sobol.analyze(problem, Brs[:, i])
        ABri_Si = sobol.analyze(problem, ABrs[:, i])

        total_Si, first_Si, second_Si = Ari_Si.to_df()
        total_Si.to_csv('%s/A_right_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/A_right_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/A_right_stim_%s_second.csv' % (savedir, predict_stims[i]))

        total_Si, first_Si, second_Si = Bri_Si.to_df()
        total_Si.to_csv('%s/b_right_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/b_right_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/b_right_stim_%s_second.csv' % (savedir, predict_stims[i]))

        total_Si, first_Si, second_Si = ABri_Si.to_df()
        total_Si.to_csv('%s/Ab_right_stim_%s_total.csv' % (savedir, predict_stims[i]))
        first_Si.to_csv('%s/Ab_right_stim_%s_first.csv' % (savedir, predict_stims[i]))
        second_Si.to_csv('%s/Ab_right_stim_%s_second.csv' % (savedir, predict_stims[i]))
