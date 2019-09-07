#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

import read
import feature
import plot

try:
    ID = sys.argv[1]
except IndexError:
    print('Usage: python test.py [str:file_name]')
    sys.exit()
# ID = '115419'
path2files = 'data'

filename = path2files + '/' + ID + '.txt'

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

