#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import method.model as m
import method.io
import method.plot as plot

# Create model
model = m.FirstOrderLeakyTransmissionLineNetwork(n_electrodes=16)

# Test parameters (Vanpoucke et al. 2004 smoothing constraint fits)
rt = np.array([
    73.5534,
    74.3194,
    59.4313,
    37.4357,
    18.5653,
    7.6150,
    3.9673,
    7.5092,
    18.0529,
    34.3136,
    51.3494,
    55.4153,
    36.1409,
    18.5566,
    7.5303,
    3.7847,
])
rl = np.array([
    0.1469,
    0.1418,
    0.1496,
    0.1637,
    0.1597,
    0.1453,
    0.1320,
    0.1106,
    0.1038,
    0.1093,
    0.1267,
    0.1646,
    0.2401,
    0.3435,
    0.5509,
])
p = np.append(rt, rl)

# Simulate the model
simulation = model.simulate(p)

# Filter simulation
simulation = method.io.mask(simulation)

# Load data
filename = 'data/Vanpoucke2004Fig2.txt'
raw_data = method.io.load(filename)
filtered_data = method.io.mask(raw_data)

# Plot the raw data
fig, axes = plot.basic_plot_splitted(filtered_data, c='C0', ls='')

# Plot the simulated data
fig, axes = plot.basic_plot_splitted(simulation, fig=fig, axes=axes, c='C1',
        ls='-')

plt.show()

# Also plot the parameters
fig, axes = plot.parameters(rt, rl, fig=None, axes=None, c='C1', ls='-',
        marker='o')

plt.show()

