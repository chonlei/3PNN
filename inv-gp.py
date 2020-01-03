#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pints

import method.io
import method.transform as transform
import method.gp as gp
import method.plot as plot

"""
Inverse problem: given a cochlea EFI data, find the input parameters of the
trained Gaussian process that best fits the data.
"""

try:
    loadas_pre = sys.argv[1]  # trained gp name
    input_file = sys.argv[2]  # file path containing input ID to predict
except IndexError:
    print('Usage: python %s [str:gp_name]' % os.path.basename(__file__) \
            + ' [str:input_file(predict)]')
    sys.exit()
path2data = 'data'
path2input = 'input'

input_ids = []
with open(input_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            input_ids.append(l.split()[0])

loaddir = './out-gp'

savedir = './out-gp/%s-inv-predict' % loadas_pre
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)

all_broken_electrodes = method.io.load_broken_electrodes(
        'data/available-electrodes.csv')
main_broken_electrodes = [1, 12, 16]
logtransform_x = transform.NaturalLogarithmicTransform()
logtransform_y = transform.NaturalLogarithmicTransform()

main_broken_electrodes_idx = np.array(main_broken_electrodes) - 1
stim_nodes = list(set(range(16)) - set(main_broken_electrodes_idx))
stim_dict = np.loadtxt('./input/stimulation_positions.csv', skiprows=1,
        delimiter=',')
stim_positions = {}
for i, x in stim_dict:
    stim_positions[i] = x
del(stim_dict)
stim_pos = [stim_positions[i] for i in stim_nodes]
shape = (16, 16)

# Load trained gp model
trained_gp_models = {}
print('Loading trained Gaussian process models...')
for j_stim in stim_nodes:
    if (j_stim + 1) in main_broken_electrodes:
        continue  # TODO: just ignore it?
    loadas = loadas_pre + '-stim_%s' % (j_stim + 1)
    trained_gp_models[j_stim + 1] = joblib.load('%s/gpr-%s.pkl' % \
            (loaddir, loadas))

# Create PINTS model
model = gp.GPModelForPints(trained_gp_models, stim_nodes, stim_pos, shape,
        transform=None)
transformed_model = gp.GPModelForPints(trained_gp_models, stim_nodes, stim_pos,
        shape, transform=logtransform_y.inverse_transform)

def merge_list(l1, l2):
    l1, l2 = list(l1), list(l2)
    out = list(l1 + list(set(l2) - set(l1)))
    out.sort()
    return out

# Go through each input in the input file
for i, input_id in enumerate(input_ids):
    print('Predicting ' + input_id + ' ...')
    saveas = 'id_' + input_id
    fd = path2data + '/' + input_id + '.txt'

    # Load data
    raw_data = method.io.load(fd)
    filtered_data = method.io.mask(raw_data, x=main_broken_electrodes)
    n_readout, n_stimuli = filtered_data.shape

    # Create mask to filter data
    filter_list = merge_list(main_broken_electrodes,
            all_broken_electrodes[input_id])
    mask = lambda y: method.io.mask(y, x=filter_list)

    # Inital guess
    priorparams = [18000, 60, 5, 1, 1]
    transform_priorparams = logtransform_x.transform(priorparams)

    # Likelihood
    loglikelihood = gp.GaussianLogLikelihood(model, raw_data, mask=mask,
            transform=logtransform_y.transform)

    print('Log-likelihood at prior parameters: ',
            loglikelihood(transform_priorparams))
    for _ in range(10):
        assert(loglikelihood(transform_priorparams) ==\
                loglikelihood(transform_priorparams))

    try:
        # Load true input value if exists
        fi = path2input + '/' + input_id + '.txt'
        input_value = method.io.load_input(fi)
        has_input = True
        print('Log-likelihood at true input parameters: ',
                loglikelihood(logtransform_x.transform(input_value)))
    except FileNotFoundError:
        has_input = False

    # Run fits
    try:
        n_repeats = int(sys.argv[3])
    except IndexError:
        n_repeats = 3

    params, loglikelihoods = [], []

    for i in range(n_repeats):

        if True:
            x0 = transform_priorparams
        else:  # TODO
            # Randomly pick a starting point
            x0 = logprior.sample(n=1)[0]
        print('Starting point: ', x0)

        # Create optimiser
        print('Starting loglikelihood: ', loglikelihood(x0))
        opt = pints.OptimisationController(loglikelihood, x0,
                method=pints.CMAES)
        opt.set_max_iterations(None)
        opt.set_parallel(False)  # model is fast, no need to run in parallel

        # Run optimisation
        try:
            with np.errstate(all='ignore'):
                # Tell numpy not to issue warnings
                p, s = opt.run()
                p = logtransform_x.inverse_transform(p)
                params.append(p)
                loglikelihoods.append(s)
                if has_input:
                    print('Found solution:          True parameters:' )
                    for k, x in enumerate(p):
                        print(pints.strfloat(x) + '    ' + \
                                pints.strfloat(input_value[k]))
                else:
                    print('Found solution:' )
                    for k, x in enumerate(p):
                        print(pints.strfloat(x))
        except ValueError:
            import traceback
            traceback.print_exc()

    # Order from best to worst
    order = np.argsort(loglikelihoods)[::-1]  # (use [::-1] for LL)
    loglikelihoods = np.asarray(loglikelihoods)[order]
    params = np.asarray(params)[order]

    # Show results
    bestn = min(3, n_repeats)
    print('Best %d loglikelihoods:' % bestn)
    for i in range(bestn):
        print(loglikelihoods[i])
    print('Mean & std of loglikelihood:')
    print(np.mean(loglikelihoods))
    print(np.std(loglikelihoods))
    print('Worst loglikelihood:')
    print(loglikelihoods[-1])

    # Extract best 3
    obtained_loglikelihood0 = loglikelihoods[0]
    obtained_parameters0 = params[0]
    obtained_loglikelihood1 = loglikelihoods[1]
    obtained_parameters1 = params[1]
    obtained_loglikelihood2 = loglikelihoods[2]
    obtained_parameters2 = params[2]

    # Show results
    if has_input:
        print('Found solution 1:          True parameters:' )
    else:
        print('Found solution 1:' )
    filename_1 = '%s/%s-solution-%s-1.txt' % (savedir, saveas, fit_seed)
    with open(filename_1, 'w') as f:
        for k, x in enumerate(obtained_parameters0):
            if has_input:
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(input_value[k]))
            else:
                print(pints.strfloat(x))
            f.write(pints.strfloat(x) + '\n')

    if has_input:
        print('Found solution 2:          True parameters:' )
    else:
        print('Found solution 2:' )
    filename_2 = '%s/%s-solution-%s-2.txt' % (savedir, saveas, fit_seed)
    with open(filename_2, 'w') as f:
        for k, x in enumerate(obtained_parameters1):
            if has_input:
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(input_value[k]))
            else:
                print(pints.strfloat(x))
            f.write(pints.strfloat(x) + '\n')

    if has_input:
        print('Found solution 3:          True parameters:' )
    else:
        print('Found solution 3:' )
    filename_3 = '%s/%s-solution-%s-3.txt' % (savedir, saveas, fit_seed)
    with open(filename_3, 'w') as f:
        for k, x in enumerate(obtained_parameters2):
            if has_input:
                print(pints.strfloat(x) + '    ' + \
                        pints.strfloat(input_value[k]))
            else:
                print(pints.strfloat(x))
            f.write(pints.strfloat(x) + '\n')
