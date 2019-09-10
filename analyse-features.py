#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import re

import method.io
import method.plot as plot

analyse_index = 2

savedir = './fig'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Remove data
remove_data = ['175426', '173154', '162224']

# Plot if only more than n data points
min_points = 3

# Get all input files
files = glob.glob('./input/[0-9]*.txt')

# Load input parameters and features
input_parameters = {}
peaks = {}
baselines = {}
curve_parameters = {}
for f in files:
    file_id = re.findall('\.\/input\/(\w+)\.txt', f)[0]
    if file_id in remove_data:
        continue
    f_feature = './out-features/%s-542811797' % file_id

    # input
    input_parameters[file_id] = method.io.load_input(f)

    # features
    peaks[file_id] = method.io.load_peaks(f_feature)
    baselines[file_id] = method.io.load_baseline(f_feature)
    curve_parameters[file_id] = method.io.load_curve_parameters(f_feature)

splitted_k, splitted_fixp = method.io.split(input_parameters,
        index=analyse_index)

# Get those to plot and get x (input parameter)
plot_k, plot_fixp, plot_x = [], [], []
for k, p in zip(splitted_k, splitted_fixp):
    if len(k) >= min_points:
        x = []
        for key in k:
            x.append(input_parameters[key][analyse_index])
        sorted_idx = np.argsort(x)
        plot_k.append([k[i] for i in sorted_idx])
        plot_fixp.append(p)
        plot_x.append([x[i] for i in sorted_idx])


# Get y: Peaks
xylabels = ['Parameter %s' % (analyse_index + 1), r'Peaks (k$\Omega$)']
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        plot_y.append(peaks[key])
    plot_y = np.asarray(plot_y)

    if i == 0:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=None, axes=None, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=xylabels)
    else:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=fig, axes=axes, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=None)
plt.savefig('%s/sensitivity-peaks-parameter%s' % (savedir, analyse_index + 1),
        bbox_inch='tight', dpi=200)
plt.close()


# Gey y: Baseline
fig, axes = plt.subplots(1, 1, figsize=(6, 4))
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        plot_y.append(baselines[key])
    plot_y = np.asarray(plot_y)

    axes.plot(plot_x[i], plot_y, c='C%s' % i, marker='o', ls='--', label='')
axes.set_xlabel('Parameter %s' % (analyse_index + 1))
axes.set_ylabel(r'Baselines (k$\Omega$)')
plt.savefig('%s/sensitivity-baselines-parameter%s'
        % (savedir, analyse_index + 1), bbox_inch='tight', dpi=200)
plt.close()


# Get y: left_c
xylabels = ['Parameter %s' % (analyse_index + 1), r'Left c (k$\Omega$/idx)']
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        yy = []
        for ii in range(16):
            y = curve_parameters[key][ii][0]
            if y is not None:
                yy.append(y[0])
            else:
                yy.append(np.NaN)
        plot_y.append(yy)
    plot_y = np.asarray(plot_y)

    if i == 0:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=None, axes=None, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=xylabels)
    else:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=fig, axes=axes, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=None)
plt.savefig('%s/sensitivity-left_c-parameter%s' % (savedir, analyse_index + 1),
        bbox_inch='tight', dpi=200)
plt.close()


# Get y: left_d
xylabels = ['Parameter %s' % (analyse_index + 1), r'Left d']
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        yy = []
        for ii in range(16):
            y = curve_parameters[key][ii][0]
            if y is not None:
                yy.append(y[1])
            else:
                yy.append(np.NaN)
        plot_y.append(yy)
    plot_y = np.asarray(plot_y)

    if i == 0:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=None, axes=None, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=xylabels)
    else:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=fig, axes=axes, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=None)
plt.savefig('%s/sensitivity-left_d-parameter%s' % (savedir, analyse_index + 1),
        bbox_inch='tight', dpi=200)
plt.close()


# Get y: right_c
xylabels = ['Parameter %s' % (analyse_index + 1), r'Right c (k$\Omega$/idx)']
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        yy = []
        for ii in range(16):
            y = curve_parameters[key][ii][1]
            if y is not None:
                yy.append(y[0])
            else:
                yy.append(np.NaN)
        plot_y.append(yy)
    plot_y = np.asarray(plot_y)

    if i == 0:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=None, axes=None, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=xylabels)
    else:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=fig, axes=axes, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=None)
plt.savefig('%s/sensitivity-right_c-parameter%s' % (savedir, analyse_index + 1),
        bbox_inch='tight', dpi=200)
plt.close()


# Get y: right_d
xylabels = ['Parameter %s' % (analyse_index + 1), r'Right d']
for i, k in enumerate(plot_k):
    plot_y = []
    for key in k:
        yy = []
        for ii in range(16):
            y = curve_parameters[key][ii][1]
            if y is not None:
                yy.append(y[1])
            else:
                yy.append(np.NaN)
        plot_y.append(yy)
    plot_y = np.asarray(plot_y)

    if i == 0:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=None, axes=None, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=xylabels)
    else:
        fig, axes = plot.sensitivity_analyse_splitted(plot_x[i], plot_y,
            fig=fig, axes=axes, c='C%s' % i, marker='o', ls='--',
            label='', xylabels=None)
plt.savefig('%s/sensitivity-right_d-parameter%s' % (savedir, analyse_index + 1),
        bbox_inch='tight', dpi=200)
plt.close()


print('Done.')
