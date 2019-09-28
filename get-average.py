#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import method.io
import method.feature as feature

try:
    file_ids = sys.argv[1:]
except IndexError:
    print('Usage: python %s [str:file_id_1], ...' % os.path.basename(__file__))
    sys.exit()

savedir = './averaged-features'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = file_ids[0]

fit_seed = '542811797'

# Load input parameters and features
input_parameters = []
peaks = []
baselines = []
curve_parameters = {}
for i in range(16):
    curve_parameters[i] = []
    for j in range(2):
        curve_parameters[i].append([])

for file_id in file_ids:
    f_feature = './out-features/%s-%s' % (file_id, fit_seed)
    f_input = './input/%s.txt' % file_id

    # input
    input_parameters.append(method.io.load_input(f_input))

    # features
    peaks.append(method.io.load_peaks(f_feature))
    baselines.append(method.io.load_baseline(f_feature))
    f = method.io.load_curve_parameters(f_feature)
    for i in range(16):
        for j in range(2):
            if f[i][j] is None:
                curve_parameters[i][j] = None
            else:
                curve_parameters[i][j].append(f[i][j])

# Check input are the same
if all(all(i == input_parameters[0]) for i in input_parameters):
    input_parameter = input_parameters[0]
else:
    raise ValueError('Input are the not same.')

# Compute average
mean_peaks = np.mean(peaks, axis=0)
mean_baselines = np.mean(baselines, axis=0)
mean_curve_parameters = {}
for i in range(16):
    mean_curve_parameters[i] = []
    for j in range(2):
        if curve_parameters[i][j] is None:
            mean_curve_parameters[i].append(None)
        else:
            mean = np.mean(curve_parameters[i][j], axis=0)
            mean_curve_parameters[i].append(mean)

# Output averaged features
method.io.save_input('%s/averaged-%s-input.txt' % (savedir, saveas),
        input_parameter)
method.io.save_peaks('%s/averaged-%s-%s' % (savedir, saveas, fit_seed),
        mean_peaks)
method.io.save_baseline('%s/averaged-%s-%s' % (savedir, saveas, fit_seed),
        mean_baselines)
method.io.save_curve_parameters('%s/averaged-%s-%s' \
        % (savedir, saveas, fit_seed), mean_curve_parameters, n_parameters=2)

