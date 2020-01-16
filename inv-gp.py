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

from fix_param import fix_input

"""
Inverse problem: given a cochlea EFI data, find the input parameters of the
trained Gaussian process that best fits the data.
"""

try:
    loadas_pre = sys.argv[1]  # trained gp name
    input_file = sys.argv[2]  # file path containing input ID to predict
except IndexError:
    print('Usage: python %s [str:gp_name]' % os.path.basename(__file__) \
            + ' [str:input_file(inv)]')
    sys.exit()
path2data = 'data'
path2input = 'input'

debug = '--debug' in sys.argv  # debug mode

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
stim_pos = [stim_positions[i + 1] for i in stim_nodes]
shape = (16, 16)

# Load trained gp model
trained_gp_models = {}
scaletransform_xs = {}
scaletransform_ys = {}
print('Loading trained Gaussian process models...')
for j_stim in stim_nodes:
    if (j_stim + 1) in main_broken_electrodes:
        continue  # NOTE: These are not fitted.
    loadas = loadas_pre + '-stim_%s' % (j_stim + 1)
    trained_gp_models[j_stim + 1] = joblib.load('%s/gpr-%s.pkl' % \
            (loaddir, loadas))
    scaletransform_xs[j_stim + 1] = joblib.load(
            '%s/scaletransform_x-%s.pkl' % (loaddir, loadas))
    scaletransform_ys[j_stim + 1] = joblib.load(
            '%s/scaletransform_y-%s.pkl' % (loaddir, loadas))

def combined_inverse_transform_y(y, s=stim_nodes, t1=logtransform_y,
                                 t2=scaletransform_ys):
    out = np.full(np.asarray(y).shape, np.NaN)
    for j in s:
        t_j = t2[j + 1]
        out[:, j] = t1.inverse_transform(t_j.inverse_transform(y[:, j]))
    return out

def combined_transform_y(y, s=stim_nodes, t1=logtransform_y,
                         t2=scaletransform_ys):
    out = np.full(np.asarray(y).shape, np.NaN)
    for j in s:
        t_j = t2[j + 1]
        t1_y = t1.transform(y[:, j])
        out[:, j] = t_j.transform(t1_y.reshape(-1, 1)).ravel()
    return out

scaletransform_xs_transform = {}
for j_stim in stim_nodes:
    scaletransform_xs_transform[j_stim + 1] = \
            scaletransform_xs[j_stim + 1].transform

# Handle fixed parameters
def fix_param(x, fix=fix_input):
    o = np.zeros(len(fix))
    j = 0
    for i in range(len(o)):
        if fix[i] is not None:
            o[i] = np.copy(fix[i])
        else:
            o[i] = x[j]
            j += 1
    return o

def trans_fix_param(x, fix=fix_input, trans=logtransform_x.transform):
    o = np.zeros(len(fix))
    j = 0
    for i in range(len(o)):
        if fix[i] is not None:
            o[i] = trans(np.copy(fix[i]))
        else:
            o[i] = x[j]
            j += 1
    return o

def fit_param(x, fix=fix_input):
    o = []
    for i in range(len(fix)):
        if fix[i] is None:
            o.append(x[i])
    return np.array(o)

n_fit_param = list(fix_input.values()).count(None)

# Create PINTS model
model = gp.GPModelForPints(trained_gp_models, stim_nodes, stim_pos, shape,
        transform_x=scaletransform_xs_transform, transform=None)

# Boundaries
lower_input = [17700, 20, 0, 0.8, 0.8]
upper_input = [19000, 130, 15, 1.2, 1.2]
# Update bounds
lower = logtransform_x.transform(lower_input)
upper = logtransform_x.transform(upper_input)
lower = fit_param(lower)
upper = fit_param(upper)
boundaries = pints.RectangularBoundaries(lower, upper)

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
    priorparams = fit_param(priorparams)
    transform_priorparams = fit_param(transform_priorparams)

    # Likelihood
    loglikelihood = gp.GaussianLogLikelihood(model, raw_data, mask=mask,
            fix=[trans_fix_param, n_fit_param], transform=combined_transform_y)

    print('Log-likelihood at prior parameters: ',
            loglikelihood(transform_priorparams))
    for _ in range(10):
        assert(loglikelihood(transform_priorparams) ==\
                loglikelihood(transform_priorparams))

    try:
        # Load true input value if exists
        fi = path2input + '/' + input_id + '.txt'
        input_value = method.io.load_input(fi)
        transform_input_value = logtransform_x.transform(input_value)
        fit_input_value = fit_param(input_value)
        fit_transform_input_value = fit_param(transform_input_value)
        has_input = True
        print('Log-likelihood at true input parameters: ',
                loglikelihood(fit_transform_input_value))
        if debug:
            #a = model.simulate(logtransform_x.transform(input_value))
            a = model.simulate(trans_fix_param(fit_transform_input_value))
            a = combined_inverse_transform_y(a)
            fig, axes = plt.subplots(4, 4, sharex=True, figsize=(10, 8))
            offset = 10
            import seaborn as sns
            c = sns.color_palette('Blues', n_readout + offset)
            cd = sns.color_palette('Oranges', n_readout + offset)
            for i, j_stim in enumerate(stim_nodes):
                ax = axes[j_stim // 4, j_stim % 4]
                ci = j_stim + offset
                ax.set_title('Stim %s' % (j_stim + 1))
                ax.plot(stim_pos, a[stim_nodes, j_stim], c=c[ci])
                data_i = range(1, n_readout + 1)
                # Convert to physical positions
                x = [stim_positions[i] for i in data_i]
                #ax.plot(x, filtered_data[:, j_stim], 'x', c=cd[ci])
                ax.plot(stim_pos, filtered_data[stim_nodes, j_stim], 'x',
                        c=cd[ci])

                # Re try
                gpr = trained_gp_models[j_stim + 1]
                scaletransform_x = scaletransform_xs[j_stim + 1]
                scaletransform_y = scaletransform_ys[j_stim + 1]
                predict_x = [np.append(i, logtransform_x.transform(input_value))
                        for i in np.linspace(2, 18.5, 100)]
                predict_x_scaled = scaletransform_x.transform(predict_x)
                predict_y = gpr.predict(predict_x_scaled, return_std=True)
                predict_y_mean = logtransform_y.inverse_transform(
                        scaletransform_y.inverse_transform(predict_y[0]))
                ax.plot(np.linspace(2, 18.5, 100), predict_y_mean, c='C2')
                
            fig.text(-0.4, 1.2, r'Transimpedence (k$\Omega$)', va='center',
                    ha='center', rotation=90, transform=axes[2, 0].transAxes,
                    clip_on=False)
            fig.text(1.1, -0.35, 'Distance from round window (mm)',
                    va='center', ha='center', transform=axes[-1, 1].transAxes,
                    clip_on=False)
            plt.tight_layout(rect=[0.03, 0.03, 1, 1])
            plt.savefig('inv-gp-debug')
    except FileNotFoundError:
        has_input = False

    # Run fits
    try:
        n_repeats = int(sys.argv[3])
    except IndexError:
        n_repeats = 3

    params, loglikelihoods = [], []

    for i in range(n_repeats):

        if i == 0:
            x0 = transform_priorparams
        else:
            # Randomly pick a starting point
            x0 = boundaries.sample(1)[0]
        print('Starting point: ', x0)

        # Create optimiser
        print('Starting loglikelihood: ', loglikelihood(x0))
        opt = pints.OptimisationController(loglikelihood, x0,
                boundaries=boundaries, method=pints.NelderMead)
        opt.set_max_iterations(None)
        opt.set_parallel(False)  # model is fast, no need to run in parallel

        # Run optimisation
        try:
            with np.errstate(all='ignore'):
                # Tell numpy not to issue warnings
                p, s = opt.run()
                p = trans_fix_param(p)
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

