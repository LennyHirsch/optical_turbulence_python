# calculate the spectrum, calculate the COM at each point in time
# save both as csvs

import numpy as np
import propagation_functions as prop
import csv
import os
from tqdm import tqdm
import correlation_analysis as corr_anl
import h5py
import re

L_RANGE = 10
P_RANGE = 10

SAVE_FILE_SPEC = '/Users/ultandaly/Desktop/tmp_analysis/comp_freq_single_cell/spec_test_l'
SAVE_FILE_COM = '/Users/ultandaly/Desktop/tmp_analysis/comp_freq_single_cell/com_test_l'
SAVE_FILE_COM_INT = '/Users/ultandaly/Desktop/tmp_analysis/comp_freq_single_cell/com_int_test'

def main():
    mylist = []
    # change appropriately
    filepath = '/Volumes/Newton/V4_results/20230210/comp_freq/'
    screen_width = 0.0125*20
    transmit_ap_width = 0.075

    files = os.listdir(filepath)
    files = files[0:401]

    for file in tqdm(files):
        with h5py.File(filepath + file, 'r') as f:
            keys = []
            f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
            for i in range(4):
                for key in keys[i:-1:8]:
                    data = f[key][()]
                    analysis_function(data, screen_width, key, transmit_ap_width)


def analysis_function(data, screen_width, key, transmit_ap_width):


    space_regex = re.compile(r'\[.*\]')
    space_string = space_regex.search(key)
    spc = space_string.group()

    l_val_regex = re.compile(r'l_\d\d')
    l_val_str = l_val_regex.search(key)
    l_val = int(l_val_str.group()[-2:])

    w0 = transmit_ap_width/np.sqrt(l_val + 1)

    spec = corr_anl.spec_calc(data, L_RANGE, P_RANGE, np.shape(data)[0], screen_width, w0)
    com = corr_anl.center_of_mass(data)
    com_int = corr_anl.center_of_mass_int(data)

    if not os.path.exists(SAVE_FILE_COM):
        os.makedirs(SAVE_FILE_COM)

    if not os.path.exists(SAVE_FILE_COM_INT):
        os.makedirs(SAVE_FILE_COM_INT)

    if not os.path.exists(SAVE_FILE_SPEC):
        os.makedirs(SAVE_FILE_SPEC)

    with open(SAVE_FILE_SPEC + '/' + str(l_val).zfill(2)  + '_' + spc + '.csv', 'a') as f:
        f_writer = csv.writer(f)
        f_writer.writerow(spec)
        f.close()
    
    with open(SAVE_FILE_COM  + '/' + str(l_val).zfill(2) + '_' + spc + '.csv', 'a') as f:
        f_writer = csv.writer(f)
        f_writer.writerow(com)
        f.close()
    
    with open(SAVE_FILE_COM_INT  + '/' + str(l_val).zfill(2) + '_' + spc + '.csv', 'a') as f:
        f_writer = csv.writer(f)
        f_writer.writerow(com_int)
        f.close()

if __name__ == main():
    main()
