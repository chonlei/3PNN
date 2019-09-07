#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import model as m
import read
import plot

# Create model
model = m.FirstOrderLeakyTransmissionLineNetwork(n_electrodes=16)

# Test parameters (just some random parameters to test here!)
rt = np.ones(15) * 4.
rbasel = 100.
rl = np.arange(1., 16.)[::-1] / 10.
p = np.append(np.append(rt, rbasel), rl)

# Simulate the model
simulation = model.simulate(p)

# Filter simulation
simulation = read.mask(simulation)

# Simply plot the raw data
fig, axes = plot.basic_plot(simulation, palette='Blues')

plt.show()

