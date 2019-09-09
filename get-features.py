#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import method.io
import method.feature as feature
import method.plot as plot

try:
    file_id = sys.argv[1]
except IndexError:
    print('Usage: python %s [str:file_id]' % os.path.basename(__file__))
    sys.exit()
path2files = 'data'
filename = path2files + '/' + file_id + '.txt'

savedir = './out-features'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = file_id

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

broken_electrodes = []

# Load data
raw_data = method.io.load(filename)
filtered_data = method.io.mask(raw_data, x=broken_electrodes)
n_readout, n_stimuli = filtered_data.shape

# Find the peaks
peaks = feature.peaks(filtered_data)

# Find the baseline
baseline = feature.baseline(filtered_data)

# Simply plot the raw data, peaks, and baseline
fig, axes = plot.basic_plot(filtered_data, palette='Blues')
x = np.arange(n_readout) + 1
axes.plot(x, peaks, marker='s', c='C2', label='Peaks')
axes.axhline(baseline, c='C2', label='Baseline')
axes.legend()
plt.savefig('%s/%s-peaks-baseline-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()

# Find the curvatures
curve_parameters = feature.curve_fit(filtered_data, feature.powerlaw)

# Plot the raw data and fitted curves in a split-plot format
fig, axes = plot.basic_plot_splitted(filtered_data, c='C0')
fig, axes = plot.fitted_curves_splitted(curve_parameters, feature.powerlaw,
        fig=fig, axes=axes, c='C2')
plt.savefig('%s/%s-curve_fit-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()

# Output features
method.io.save_peaks('%s/%s-%s' % (savedir, saveas, fit_seed), peaks)
method.io.save_baseline('%s/%s-%s' % (savedir, saveas, fit_seed), baseline)
method.io.save_curve_parameters('%s/%s-%s' % (savedir, saveas, fit_seed),
        curve_parameters, n_parameters=2)

# To lead them...
# peaks = method.io.load_peaks('%s/%s-%s' % (savedir, saveas, fit_seed))
# baseline = method.io.load_baseline('%s/%s-%s' % (savedir, saveas, fit_seed))
# curve_parameters = method.io.load_curve_parameters('%s/%s-%s' % (savedir,
#         saveas, fit_seed))

