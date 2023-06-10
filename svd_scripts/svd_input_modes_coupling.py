#This script will determine the SVD modes for a range of input modes and output coupling basis numbers for later analysis

#it's occured to me that this is an inefficient method to run this program. I am propagating through a basis set every time, even if it is decomposed multiple times by different HG sets. Would be better to instead propagate all modes through, then decompose these all at once
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import os

sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/')
sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/')

import packages.propagation_functions as prop
import packages.svd_prop_funcs as svd_funcs
import svd_scripts.lg_propagation as lg_prop
import params

def lg_generation(l_max_min, p_max, waist, res, screen_width, wavelength):   
    print("\nCalculating input modes...")
    inp = prop.BeamProfile(res, screen_width, wavelength)

    inp_beams = []

    for l in tqdm(range(-l_max_min, l_max_min + 1)):
        for p in range(p_max + 1):
            inp.laguerre_gaussian_beam(l, p, waist, z=0)
            #inp.hard_ap(params.inp_ap_width)
            inp_beams.append(inp.field)

    inp_beams = np.asarray(inp_beams)
    
    return inp_beams

def hg_generation(pascals_row, waist):

    hg_modes = svd_funcs.generate_hg_modes(pascals_row, params.res, params.screen_width, params.wavelength, waist)
    print(np.shape(hg_modes))
    return hg_modes


def turbscreen_load(data_dir):

    t_screens = np.load(data_dir + 'turb_screens.npy')

    t = [[] for _ in range(len(t_screens))]

    r0 = params.r0_tot * (params.num_of_steps - 1) ** 3/5
    for i, sc in enumerate(t_screens):
        t[i] = prop.PhaseScreen(params.screen_width, params.res, r0, params.l0, params.L0)

        t[i].phz = sc
        t[i].phz_lo = 0.0

    return t

def simulation(l_max_min, p_max, pascals_row_rec, pascals_row_inp, data_dir, coll_beams, inp_basis):
    # i can't easily change the value in params, so I need to set this loop myself
    # propagate for |l| < n and p < n
    # look into SBP of different combinations
    # propagate p = 0
    # propagate different pascals of HG modes
    delz_step = params.delz/params.num_of_steps

    if inp_basis.lower() == 'lg':
        inp_beams = lg_generation(l_max_min, p_max, params.waist, params.res, params.screen_width, params.wavelength)

        mode_num = ((2 * l_max_min) + 1) * (p_max + 1)
    elif inp_basis.lower() == 'hg':

        inp_beams = hg_generation(pascals_row_inp, params.waist)
        #double check this, but it should work
        mode_num = ((pascals_row_inp) * (pascals_row_inp + 1))//2

    else:
        print(f'Error: inp_basis variable is incorrect. Current value is {inp_basis}. Please select from either "hg" or "lg"...')

    rec_orthogonality = ((pascals_row_rec) * (pascals_row_rec + 1))//2

    t_screens_obj = turbscreen_load(data_dir)

    rec_beams = lg_prop.propagate_modes(inp_beams, t_screens_obj, delz_step, coll_beams, params.wavelength)

    #perform SVD
    u, s, v, t_mat = svd_funcs.svd_calc_hg(rec_beams, params.res, pascals_row_rec, params.waist_lst[0], params.screen_width)
    # also look at decomposing using different number of HG modes

    print(f"Transmission matrix dims: {np.shape(t_mat)}")

    svd_trans_modes = svd_funcs.svd_inp_modes_calc(
        v, inp_beams, mode_num, params.res, rec_orthogonality, params.screen_width)
        
    svd_rec_modes = np.zeros((rec_orthogonality,
                                  params.res, params.res), dtype=np.complex128)

    svd_rec_modes = svd_funcs.channel_propagtion(
        svd_trans_modes, t_screens_obj, params.inp_ap_width, params.rec_ap_width, params.screen_width, delz_step, rec_orthogonality, params.res)
    
    if coll_beams == 1:
        svd_rec_modes = prop.coll_calc(
            svd_rec_modes, params.res, params.screen_width, params.wavelength, params.waist, params.delz)
    
    return svd_trans_modes, svd_rec_modes
    

def main(data_dir, save_dir):
    # i can't easily change the value in params, so I need to set this loop myself
    # propagate for |l| < n and p < n
    # look into SBP of different combinations
    # propagate p = 0
    # propagate different pascals of HG modes

    coll_beams = 1
    #propagating all lg < l_max and p < p_max
    '''
    #Code currently breaks if there are more receiving modes than input modes
    for lg_max in range(7, 8):
        for pascals_row in range(2, 8):

            num_of_rec_modes = ((pascals_row) * (pascals_row + 1))//2
            num_of_trans_modes = ((lg_max * 2) + 1) * (lg_max + 1)

            if num_of_rec_modes > num_of_trans_modes:
                continue

            full_dir = save_dir + f'/l_{str(lg_max).zfill(2)}_p_{str(lg_max).zfill(2)}/hg_row_{str(pascals_row).zfill(2)}/'

            if os.path.exists(full_dir):
                owrite = input('Error: Path already exists. Run iteration and overwrite results?     ')

                if owrite[0].lower() != 'y':
                    continue
            else:
                os.makedirs(full_dir)
            svd_trans_modes, svd_rec_modes = simulation(lg_max, lg_max, pascals_row, 0, data_dir, coll_beams, 'lg')

            np.save(full_dir + 'svd_trans_modes.npy', svd_trans_modes)
            np.save(full_dir + 'svd_rec_modes.npy', svd_rec_modes)
    
    
    #propagating for p_max = 0
    for lg_max in range(20, 23):
        for pascals_row in range(2, 8):

            num_of_rec_modes = ((pascals_row) * (pascals_row + 1))//2
            num_of_trans_modes = ((lg_max * 2) + 1)

            if num_of_rec_modes > num_of_trans_modes:
                continue

            full_dir = save_dir + f'/l_{str(lg_max).zfill(2)}_p_{str(0).zfill(2)}/hg_row_{str(pascals_row).zfill(2)}/'

            if os.path.exists(full_dir):
                owrite = input(
                    'Error: Path already exists. Run iteration and overwrite results?     ')

                if owrite[0].lower() != 'y':
                    continue
            else:
                os.makedirs(full_dir)
            svd_trans_modes, svd_rec_modes = simulation(
                lg_max, 0, pascals_row, 0, data_dir, coll_beams, 'lg')
            
            np.save(full_dir + 'svd_trans_modes.npy', svd_trans_modes)
            np.save(full_dir + 'svd_rec_modes.npy', svd_rec_modes)
    '''
            #SAVE BOTH VALUES

    for pascals_row_trans in range(3, 8):
        for pascals_row in range(2, pascals_row_trans + 1):
            num_of_rec_modes = ((pascals_row) * (pascals_row + 1))//2
            num_of_trans_modes = ((pascals_row_trans) * (pascals_row_trans + 1))//2

            if num_of_rec_modes > num_of_trans_modes:
                continue

            full_dir = save_dir + \
                f'/hg_rows_trans_{str(pascals_row_trans).zfill(2)}/hg_row_{str(pascals_row).zfill(2)}/'

            if os.path.exists(full_dir):
                owrite = input(
                    'Error: Path already exists. Run iteration and overwrite results?     ')

                if owrite[0].lower() != 'y':
                    continue
            else:
                os.makedirs(full_dir)
            svd_trans_modes, svd_rec_modes = simulation(
                1, 1, pascals_row, pascals_row_trans, data_dir, coll_beams, 'hg')

            np.save(full_dir + 'svd_trans_modes.npy', svd_trans_modes)
            np.save(full_dir + 'svd_rec_modes.npy', svd_rec_modes)
        
        
    # save svd_rec_modes


    #COMPARE
        #COUPLING EFFICIENCY
        #CAPTURED POWER
        #LOOK AT ORTHOGONALITY

if __name__ == '__main__':
    
    #point to directory containting turbulent phase screens
    data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v013/reference/'

    #point to directory that will be made
    save_dir = '/Users/ultandaly/Desktop/input_modes_coupling/realisation_013/'
    main(data_dir, save_dir)

    #point to directory containting turbulent phase screens
    data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v015/reference/'

    #point to directory that will be made
    save_dir = '/Users/ultandaly/Desktop/input_modes_coupling/realisation_015/'

    main(data_dir, save_dir)
