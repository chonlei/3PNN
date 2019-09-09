#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

import read
import feature
import plot

try:
    file_id = sys.argv[1]
except IndexError:
    print('Usage: python test.py [str:file_id]')
    sys.exit()
path2files = 'data'
filename = path2files + '/' + file_id + '.txt'

savedir = './out-features'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = file_id

# Load data
raw_data = read.load(filename)
filtered_data = read.mask(raw_data, x=[12, 16])
n_readout, n_stimuli = filtered_data.shape

# Find the peaks
peaks = feature.peaks(filtered_data)

# Find the baseline
baseline = feature.baseline(filtered_data)

# Simply plot the raw data, peaks, and baseline
fig, axes = plot.basic_plot(filtered_data, palette='Blues')
x = np.arange(n_readout) + 1
axes.plot(x, peaks, marker='s', c='C2')
axes.axhline(baseline, c='C2')
plt.savefig('%s/%s-peaks-baseline-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()

# Find the curvatures
curve_parameters = feature.curve_fit(filtered_data, feature.powerlaw)

# Plot the raw data and fitted curves in a split-plot format
fig, axes = plot.basic_plot_splitted(filtered_data, c='C0')
fig, axes = plot.fitted_curves_splitted(curve_parameters, feature.powerlaw,
        fig=fig, axes=axes, c='C2')
plt.savefig('%s/%s-curve-fit-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.show()

#TODO
read.save_peaks('%s/%s-%s' % (savedir, saveas, fit_seed), peaks)
read.save_baseline('%s/%s-%s' % (savedir, saveas, fit_seed), baseline)
read.save_curve_parameters('%s/%s-%s' % (savedir, saveas, fit_seed),
        curve_parameters)

print('----' * 20)
print(r'Peaks at each electrode (k$\Omega$): ', peaks)
print(r'Baseline value (k$\Omega$): ', baseline)
print('Fitted curve parameters:')
for i in range(16):
    print(r'Electrode %s: ' % (i + 1), curve_parameters[i])
