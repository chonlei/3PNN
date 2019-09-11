#!/usr/bin/env python3
import sys
import os
import numpy as np
import glob
import re

import method.io

try:
    input_param = np.array([np.float(i) for i in sys.argv[1:]])
except IndexError:
    print('Usage: python %s [arr:input_param]' % os.path.basename(__file__))
    sys.exit()

l = ['p%s=%s' % (i, v) for i, v in zip(range(len(input_param)), input_param)]

# Remove data
remove_data = ['175426', '173154', '162224']

# Get all input files
files = glob.glob('./input/[0-9]*.txt')

# Load input parameters and features
input_parameters = {}
match_ids = []
for f in files:
    f_basename = os.path.basename(f)
    file_id = re.findall('(\w+)\.txt', f_basename)[0]
    if file_id in remove_data:
        continue
    f_feature = './out-features/%s-542811797' % file_id

    # input
    input_parameters[file_id] = method.io.load_input(f)

    if len(input_parameters[file_id]) != len(input_param):
        raise ValueError('Number of input parameters does not match input '
                + 'file parameters')

    if all(input_parameters[file_id] == input_param):
        match_ids.append(file_id)

# Print on console
print('Experimental ID(s) that matches [%s]:' % ', '.join(l))
if match_ids:
    for i in match_ids:
        print(i)
else:
    print('None.')

