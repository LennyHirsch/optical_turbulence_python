import sys
import os
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import numpy as np
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt

import propagation_functions as prop
import propagation_svd as svd_funcs

if __name__ == '__main__':
    #point to relevant data directory
    data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages/svd_scripts/data/20230418/lg_prop_l_5_p_5_v1'

    if not os.path.isdir(data_dir):
        print('Data directory does not exist. Exiting program...')
        quit()
    
    #relevant parameters
    #check these align with the readme file in relevant directory

    #system parameters
    delz = 1000
    inp_ap_width = 0.15
    rec_ap_width = 0.15

    #input basis parameters
    waist = 0.02

    #peforming lg propagation
    l_pos_min = 5
    p_max = 5
    mode_num = (p_max+1) * (l_pos_min*2 + 1)

    # simulation parameters
    screen_width = 0.4
    num_of_steps = 20 + 1
    wavelength = 1550e-9
    res = 512
    delz_step = delz/num_of_steps

    compact_res = int(np.ceil(res * (rec_ap_width/screen_width)))
    trans_modes_num = 66

    #load data from numpy bin files
    res_beams = np.load(data_dir + '/res_beams.npy')
    inp_beams = np.load(data_dir + '/inp_beams.npy')
    t_screens = np.load(data_dir + '/turb_screens.npy')

    #need to define the turbulent screens as PhaseScreen objects

    t_screens_obj = [prop.PhaseScreen(
        screen_width, res, 1.0 * (num_of_steps - 1)**(3/5), 1.0, 1.0
    ) for i in range(num_of_steps - 1)]

    for i, t in enumerate(t_screens):

        t_screens_obj[i].phz_lo = 0.0
        t_screens_obj[i].phz = t

    u, s, v = svd_funcs.svd_calc(res_beams, res, compact_res, screen_width)
    print(np.shape(v))
    print(f"\nFirst {trans_modes_num} singular values: \n {s[0:trans_modes_num]**2.0}")
    svd_trans_modes = svd_funcs.svd_inp_modes_calc(v, inp_beams, mode_num, res,trans_modes_num, screen_width)

    #for i, trans_mode in enumerate(svd_trans_modes):
    #    svd_trans_modes[i] = trans_mode /(np.abs)
    #svd_rec_modes = np.zeros((trans_modes_num,
    #                              res, res), dtype=np.complex64)
    
    print(f"Propagating first {trans_modes_num} SVD modes through channel...")
    svd_rec_modes = svd_funcs.channel_propagtion(
        svd_trans_modes, t_screens_obj, inp_ap_width, rec_ap_width, screen_width, delz_step, trans_modes_num, res)


    crss = np.zeros((15, 15))

    for ii in range(15):
        for j in range(15):

            crss[ii, j] = np.abs(np.trapz(
                    np.trapz(svd_rec_modes[ii] * np.conj(svd_rec_modes[j]), dx=screen_width/res), dx=screen_width/res))

    plt.imshow(crss)

    plt.title('SVD Received Modes Crosstalk')
    plt.colorbar()
    plt.show()
