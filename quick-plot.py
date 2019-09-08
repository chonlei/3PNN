import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pints

import read
import model as m
import plot

try:
    file_id = sys.argv[1]
except IndexError:
    print('Usage: python test.py [str:file_id] [int:n_repeats]')
    sys.exit()
path2files = 'data'
filename = path2files + '/' + file_id + '.txt'

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

saveas = file_id

fit_seed = 542811797

# Load data
raw_data = read.load(filename)
n_readout, n_stimuli = raw_data.shape
assert(n_readout == n_stimuli)  # assume it is a sqaure matrix for EFI
sigma_noise = 0.1  # guess in range of 0.1 kOhms

broken_electrodes = [12, 16]

# Create mask to filter data
mask = lambda y: read.mask(y, x=broken_electrodes)

# Create model
model = m.FirstOrderLeakyTransmissionLineNetwork(n_electrodes=n_readout,
        transform=None)

obtained_parameters0 = np.loadtxt('out/%s-solution-%s-1.txt' %
        (file_id, fit_seed))
obtained_parameters1 = np.loadtxt('out/%s-solution-%s-2.txt' %
        (file_id, fit_seed))
obtained_parameters2 = np.loadtxt('out/%s-solution-%s-3.txt' %
        (file_id, fit_seed))

fig, axes = plot.basic_plot_splitted(raw_data, fig=None, axes=None, c='C0',
        ls='')
sol0 = model.simulate(obtained_parameters0[:-1])
sol1 = model.simulate(obtained_parameters1[:-1])
sol2 = model.simulate(obtained_parameters2[:-1])
fig, axes = plot.basic_plot_splitted(sol0, fig=fig, axes=axes, c='C1', ls='-')
fig, axes = plot.basic_plot_splitted(sol1, fig=fig, axes=axes, c='C2', ls='--')
fig, axes = plot.basic_plot_splitted(sol2, fig=fig, axes=axes, c='C3', ls=':')
plt.subplots_adjust(hspace=0)
#plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed),
#        bbox_inches='tight')
plt.show()
plt.close()

rt0, rl0 = model.split_parameters(obtained_parameters0[:-1])
rt1, rl1 = model.split_parameters(obtained_parameters1[:-1])
rt2, rl2 = model.split_parameters(obtained_parameters2[:-1])
fig, axes = plot.parameters(rt0, rl0, fig=None, axes=None, c='C1', ls='-',
        marker='o', label='Solution 1')
fig, axes = plot.parameters(rt1, rl1, fig=fig, axes=axes, c='C2', ls='--',
        marker='s', label='Solution 2')
fig, axes = plot.parameters(rt2, rl2, fig=fig, axes=axes, c='C3', ls=':',
        marker='^', label='Solution 3')
plt.subplots_adjust(hspace=0)
plt.legend()
#plt.savefig('%s/%s-parameters-%s.png' % (savedir, saveas, fit_seed),
#        bbox_inches='tight')
plt.show()
plt.close()
