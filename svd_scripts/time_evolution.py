import sys
import os
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import numpy as np
from tqdm import tqdm
from datetime import date
import time
from multiprocessing import Pool

import propagation_functions as prop
import svd_prop_funcs as svd_funcs


sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts')

import lg_propagation as lg_prop
import svd_hg_decomposition as hg_decomp

#ENSURE THIS IS POINTING TO CORRECT PARAMS FILE
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data')

import params

def main(mean_v, del_v, time_step, dim_inc = 4, num_of_rels = 1, dir_num = 1):

    rel_dir = f'/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230424/time_evolution/realisation_{str(dir_num).zfill(3)}'

    os.chdir(rel_dir)

    svd_dir = 'svd_modes/'

    if not os.path.exists(svd_dir):
        os.makedirs(svd_dir)

    wst = 0.024
    #because this function is generating new turbulent screens for every time I need to ensure that I calculate the svd modes for the first realisation
    delz_step = params.delz/params.num_of_steps

    inp_beams = lg_prop.calculate_input_modes(0)
    print(np.shape(inp_beams))
    print('\nGenerating turblent screens...')

    t_screens = [prop.PhaseScreen(
        params.screen_width * dim_inc, 128 * dim_inc, params.r0_tot *
        (params.num_of_steps - 1)**(3/5), params.l0, params.L0
    ) for i in range(params.num_of_steps - 1)]
    phz_screens = []

    for t in t_screens:
        t.mvk_screen()
        t.mvk_sh_screen()
        #reformat phase screens for later saving
        phz_screens.append(t.phz + t.phz_lo)
    
    phz_screens = np.asarray(phz_screens)


    part_t_screens = [prop.PhaseScreen(
        params.screen_width, params.res, params.r0_tot *
        (params.num_of_steps - 1)**(3/5), params.l0, params.L0
    ) for i in range(params.num_of_steps - 1)]

    #determine v speeds for all screeens
    # I can just move each screen in a single dimension rather than worrying about 
    v_mags = np.random.normal(mean_v, del_v, params.num_of_steps-1)

    #rnd_vecs = np.asarray([np.random.uniform(0, 1, 2) for i in range(params.num_of_steps-1)])

    #v_vals = v_mags * rnd_vecs
    pixel_step = params.res * v_mags * time_step / params.screen_width
    
    #calculate first svd mode
    for i in range(params.num_of_steps - 1):

        part_t_screens[i].phz = phz_screens[i, 0:params.res, 0:params.res]
        part_t_screens[i].phz_lo = 0.0

    svd_trans_modes, svd_rec_modes = determine_svd_modes(inp_beams, part_t_screens, delz_step, wst, phz_screens[:, 0:params.res, 0:params.res])

    lg_prop.save_data([np.zeros(1), np.zeros(1), svd_rec_modes], svd_dir)

    for rel in range(num_of_rels):
        pixel_shift_lst = pixel_step * rel
        print(pixel_shift_lst)
        for i in range(params.num_of_steps - 1):
            pixel_shift_tot = int(np.rint(pixel_shift_lst[i]))
            part_t_screens[i].phz = phz_screens[i, pixel_shift_tot:params.res + pixel_shift_tot, pixel_shift_tot:params.res + pixel_shift_tot]
            part_t_screens[i].phz_lo = np.zeros((params.res, params.res))
    
        res_beams = lg_prop.propagate_modes(svd_trans_modes, part_t_screens, delz_step, 1)
        
        time_step_dir = f'time_step_{str(rel).zfill(3)}/'

        if not os.path.exists(time_step_dir):
            os.makedirs(time_step_dir)

        #save the resultant beams
        lg_prop.save_data([np.zeros(1), np.zeros(1), res_beams], time_step_dir)

def determine_svd_modes(inp_beams, t_screens, delz_step, wst, phz_screens):

    res_beams = lg_prop.propagate_modes(inp_beams, t_screens, delz_step, 1)


    t_screens_obj = [prop.PhaseScreen(
        params.screen_width, params.res, 1.0 * (params.num_of_steps - 1)**(3/5), 1.0, 1.0) for i in range(params.num_of_steps - 1)]

    for i, t in enumerate(t_screens):
        t_screens_obj[i].phz_lo = np.zeros((params.res, params.res))
        t_screens_obj[i].phz = phz_screens[i]

    u, s, v, t_mat = svd_funcs.svd_calc_hg(res_beams, params.res, params.pascals_row, wst, params.screen_width)

    svd_trans_modes = svd_funcs.svd_inp_modes_calc(
        v, inp_beams, params.mode_num, params.res, params.trans_modes_num, params.screen_width)

    svd_rec_modes = np.zeros((params.trans_modes_num,
                              params.res, params.res), dtype=np.complex128)

    svd_rec_modes = svd_funcs.channel_propagtion(
        svd_trans_modes, t_screens_obj, params.inp_ap_width, params.rec_ap_width, params.screen_width, delz_step, params.trans_modes_num, params.res)

    coll_mode = prop.BeamProfile(params.res, 0.4, params.wavelength)
    coll_mode.hermite_gaussian_beam(0, 0, wst, params.delz)

    for i in range(len(svd_rec_modes)):
        svd_rec_modes[i] *= np.exp(1j * np.angle(coll_mode.field))

    return svd_trans_modes, svd_rec_modes


        
if __name__ == '__main__':
    main()