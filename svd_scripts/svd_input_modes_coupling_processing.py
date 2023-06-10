# A quick script that will take the results from svd_input_modes_coupling (assuming that this is pointing to the right directory) and determine the power that is captured by the aperture, and the power that is coupled into the modal basis

import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os
import glob
import pickle

sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/')
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/')

import params
import svd_scripts.lg_propagation as lg_prop
import packages.svd_prop_funcs as svd_funcs
import packages.propagation_functions as prop

def profile_pow_calc(beam):
    pixel_size = params.screen_width/params.res

    power = np.abs(np.trapz(np.trapz(beam * beam.conj(), dx = pixel_size), dx = pixel_size))

    return power

def file_pows(hg_dirs):
    pows_lg = []
    for hg in hg_dirs:
        pows_hg = []
        beam_profiles = np.load(hg + '/svd_rec_modes.npy')
        for beam in beam_profiles:
            pows_hg.append(profile_pow_calc(beam))
        pows_lg.append(pows_hg)
            
    return pows_lg

def power_in_ap(lg_dirs):
    pows_full = []
    for l_dir in lg_dirs:
        #find all hg_dirs contained within lg_dir
        hg_dirs = sorted(glob.glob(l_dir +'/hg*'))
        pows_full.append(file_pows(hg_dirs))
    return pows_full

def file_coupled_pows(hg_dirs):

    pows_lg = []

    for hg in hg_dirs:
        beams = np.load(hg +'/svd_rec_modes.npy')
        pascals_row = int(hg[-2:])

        hg_decomp = np.asarray(svd_funcs.hg_decomp_calc(beams, pascals_row, params.waist_lst[0], params.res, params.screen_width, params.wavelength))

        pow_sing = []
        for hg_decom in hg_decomp:
            pow_sing.append(np.sum(np.abs(hg_decom * np.conj(hg_decom))))
        pows_lg.append(pow_sing)

    return pows_lg
    

def hg_coupled_power(lg_dirs):
    #hg decomposition
    pows_full = []

    for l_dir in lg_dirs:
        hg_dirs = sorted(glob.glob(l_dir +'/hg*'))
        pows_full.append(file_coupled_pows(hg_dirs))



    return pows_full

def main(data_dir, save_dir):
    # find all LG directories
    '''
    lg_dirs = sorted(glob.glob(data_dir + '/l_*[!p_00]'))

    for lg_dir in lg_dirs:
        print(lg_dir)
    #find power coupled into aperture
    ap_power = power_in_ap(lg_dirs)
    #find poewr coupled into modal basis
    coupled_power = hg_coupled_power(lg_dirs)
    #save powers
    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_ap_pow_inc_p.pkl', 'wb') as f:
        pickle.dump(ap_power, f)
    
    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_coupled_pow_inc_p.pkl', "wb") as f:
        pickle.dump(coupled_power, f)

    #perform same process for propagations with no p values
    lg_dirs = sorted(glob.glob(data_dir + '/l_*p_00'))
    for lg_dir in lg_dirs:
        print(lg_dir)
    #find power coupled into aperture
    ap_power = power_in_ap(lg_dirs)
    #find poewr coupled into modal basis
    coupled_power = hg_coupled_power(lg_dirs)

    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_ap_pow_no_p.pkl', 'wb') as f:
        pickle.dump(ap_power, f)

    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_coupled_pow_no_p.pkl', "wb") as f:
        pickle.dump(coupled_power, f)
    # calculate power in aperture
    # calculate power coupled into modal basis

    #save results in their own directory
    '''

    hg_dirs = sorted(glob.glob(data_dir + '/hg_*'))
    for hg_dir in hg_dirs:
        print(hg_dir)

    ap_power = power_in_ap(hg_dirs)
    coupled_power = hg_coupled_power(hg_dirs)

    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_hg_inps.pkl', 'wb') as f:
        pickle.dump(ap_power, f)

    with open(save_dir + f'/realisation_{str(data_dir[-3:])}_hg_inps_coupled.pkl', "wb") as f:
        pickle.dump(coupled_power, f)


if __name__ == '__main__':
    parent_dir = '/Users/ultandaly/Desktop/input_modes_coupling/'
    data_dirs = sorted(glob.glob(parent_dir + '/realisation*'))
    for data_dir in data_dirs:
        print(data_dir)
        save_dir = parent_dir + '/coupled_power/'
        main(data_dir, save_dir)