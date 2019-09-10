#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import re

import method.io
import method.plot as plot

# Get all input files
files = glob.glob('./input/[0-9]*.txt')

# Load input parameters and features
input_parameters = {}
peaks = {}
baselines = {}
curve_parameters = {}
for f in files:
    file_id = re.findall('\.\/input\/(\w+)\.txt', f)[0]
    f_feature = './out-features/%s-542811797' % file_id

    # input
    input_parameters[file_id] = method.io.load_input(f)

    # features
    peaks[file_id] = method.io.load_peaks(f_feature)
    baselines[file_id] = method.io.load_baseline(f_feature)
    curve_parameters[file_id] = method.io.load_curve_parameters(f_feature)


