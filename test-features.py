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
# file_id = '115419'
path2files = 'data'

filename = path2files + '/' + file_id + '.txt'

# Load data
raw_data = read.load(filename)
filtered_data = read.mask(raw_data, x=[12, 16])
n_readout, n_stimuli = filtered_data.shape

# Simply plot the raw data
fig, axes = plot.basic_plot(filtered_data, palette='Blues')
x = np.arange(n_readout) + 1

# Find the peaks
peaks = feature.peaks(filtered_data)
axes.plot(x, peaks, marker='s', c='C2')

# Find the baseline
baseline = feature.baseline(filtered_data)
axes.axhline(baseline, c='C2')

plt.show()

# Find the curvatures
curve_parameters = feature.curve_fit(filtered_data, feature.powerlaw)
fig, axes = plot.basic_plot_splitted(filtered_data, color='C0')
fig, axes = plot.fitted_curves_splitted(curve_parameters, feature.powerlaw,
        fig=fig, axes=axes, color='C2')

plt.show()

print('----' * 20)
print(r'Peaks at each electrode (k$\Omega$): ', peaks)
print(r'Baseline value (k$\Omega$): ', baseline)
print('Fitted curve parameters:')
for i in range(16):
    print(r'Electrode %s: ' % (i + 1), curve_parameters[i])
