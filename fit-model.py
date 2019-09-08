#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import read
import model as m
import plot

"""
Run fitting for the cochlea EFI model.
"""

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

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

# Load data
raw_data = read.load(filename)
n_readout, n_stimuli = raw_data.shape
assert(n_readout == n_stimuli)  # assume it is a sqaure matrix for EFI
sigma_noise = 0.1  # guess in range of 0.1 kOhms

broken_electrodes = [12, 16]

# Create mask to filter data
mask = lambda y: read.mask(y, x=broken_electrodes)

# Parameter transformation (remove positive constraint)
transform_to_model_param = m.log_transform_to_model_param
transform_from_model_param = m.log_transform_from_model_param

# Create model
model = m.FirstOrderLeakyTransmissionLineNetwork(n_electrodes=n_readout,
        transform=transform_to_model_param)

# Prior parameters (initial guess)
rt = np.ones(n_readout - 1) * 10.  # transversal resistance (electrode order)
rbasel = 5.  # Basel resistance (next to the last electrode)
rl = np.ones(n_readout - 1) * 0.5  # longitudinal resistance (electrode order)

modelparams = np.append(np.append(rt, rbasel), rl)
priorparams = np.append(modelparams, sigma_noise)
transform_modelparams = transform_from_model_param(modelparams)
transform_priorparams = np.append(transform_modelparams, sigma_noise)

# Create likelihood, prior, and posterior
loglikelihood = m.GaussianLogLikelihood(model, raw_data, mask)
logmodelprior = m.UniformLogPrior(n_electrodes=n_readout,
        transform=transform_to_model_param,
        inv_transform=transform_from_model_param)
lognoiseprior = pints.UniformLogPrior([0.1 * sigma_noise], [10. * sigma_noise])
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
print('Score at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Run fits
try:
    n_repeats = int(sys.argv[2])
except IndexError:
    n_repeats = 3

params, logposteriors = [], []

for i in range(n_repeats):

    if True:
        x0 = transform_priorparams
    else:  # TODO
        # Randomly pick a starting point
        x0 = logprior.sample(n=1)[0]
    print('Starting point: ', x0)

    # Create optimiser
    print('Starting logposterior: ', logposterior(x0))
    opt = pints.OptimisationController(logposterior, x0.T, method=pints.CMAES)
    opt.set_max_iterations(None)
    opt.set_parallel(True)

    # Run optimisation
    try:
        with np.errstate(all='ignore'):
            # Tell numpy not to issue warnings
            p, s = opt.run()
            p[:-1] = transform_to_model_param(p[:-1])  # last one is sigma
            params.append(p)
            logposteriors.append(s)
            print('Found solution:' )
            for k, x in enumerate(p):
                print(pints.strfloat(x))
    except ValueError:
        import traceback
        traceback.print_exc()

# Order from best to worst
order = np.argsort(logposteriors)[::-1]  # (use [::-1] for LL)
logposteriors = np.asarray(logposteriors)[order]
params = np.asarray(params)[order]

# Show results
bestn = min(3, n_repeats)
print('Best %d logposteriors:' % bestn)
for i in xrange(bestn):
    print(logposteriors[i])
print('Mean & std of logposterior:')
print(np.mean(logposteriors))
print(np.std(logposteriors))
print('Worst logposterior:')
print(logposteriors[-1])

# Extract best 3
obtained_logposterior0 = logposteriors[0]
obtained_parameters0 = params[0]
obtained_logposterior1 = logposteriors[1]
obtained_parameters1 = params[1]
obtained_logposterior2 = logposteriors[2]
obtained_parameters2 = params[2]

# Show results
print('Found solution:' )
# Store output
with open('%s/%s-solution-%s-1.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters0):
        print(pints.strfloat(x))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:' )
# Store output
with open('%s/%s-solution-%s-2.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters1):
        print(pints.strfloat(x))
        f.write(pints.strfloat(x) + '\n')
print('Found solution:' )
# Store output
with open('%s/%s-solution-%s-3.txt' % (savedir, saveas, fit_seed), 'w') as f:
    for k, x in enumerate(obtained_parameters2):
        print(pints.strfloat(x))
        f.write(pints.strfloat(x) + '\n')

# Simple plots
fig, axes = plot.basic_plot_splitted(raw_data, fig=None, axes=None, c='C0',
        ls='')
sol0 = model.simulate(transform_from_model_param(obtained_parameters0[:-1]))
sol1 = model.simulate(transform_from_model_param(obtained_parameters1[:-1]))
sol2 = model.simulate(transform_from_model_param(obtained_parameters2[:-1]))
fig, axes = plot.basic_plot_splitted(sol0, fig=fig, axes=axes, c='C1', ls='-')
fig, axes = plot.basic_plot_splitted(sol1, fig=fig, axes=axes, c='C2', ls='--')
fig, axes = plot.basic_plot_splitted(sol2, fig=fig, axes=axes, c='C3', ls=':')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
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
plt.savefig('%s/%s-parameters-%s.png' % (savedir, saveas, fit_seed),
        bbox_inches='tight')
plt.close()

print('Done.')
