#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    print('Usage: python %s [str:gp_name]' % os.path.basename(__file__)
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

stim_nodes = range(16)
stim_dict = np.loadtxt('./input/stimulation_positions.csv', skiprows=1,
        delimiter=',')
stim_positions = {}
for i, x in stim_dict:
    stim_positions[i] = x
del(stim_dict)

# Load trained gp model
trained_gp_models = {}
print('Loading trained Gaussian process models...')
for j_stim in stim_nodes:
    if (j_stim + 1) in main_broken_electrodes:
        continue  # TODO: just ignore it?
    loadas = loadas_pre + '-stim_%s' % (j_stim + 1)
    trained_gp_models[j_stim + 1] = joblib.load('%s/gpr-%s.pkl' % \
            (loaddir, loadas))


model = gp.GPModelForPints(trained_gp_models, stim_nodes, stim_positions,
        transform=None)
transformed_model = gp.GPModelForPints(trained_gp_models, stim_nodes,
        stim_positions, transform=logtransform_y.inverse_transform)

def merge_list(l1, l2):
    l1, l2 = list(l1), list(l2)
    return list(l1 + list(set(l2) - set(l1))).sort()

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
    loglikelihood = gp.GaussianLogLikelihood(model, raw_data, mask,
            transform=logtransform_y.transform)

    print('Score at prior parameters: ',
            logposterior(transform_priorparams))
    for _ in range(10):
        assert(logposterior(transform_priorparams) ==\
                logposterior(transform_priorparams))



