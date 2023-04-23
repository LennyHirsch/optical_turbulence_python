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

#ENSURE THIS IS POINTING TO CORRECT PARAMS FILE
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data')
import params


def main(mean_v, del_v, time_step, inp_beams, dim_inc = 4, num_of_rels = 1):

    delz_step = params.delz/params.num_of_steps
    print('\nGenerating turblent screens...')

    t_screens = [prop.PhaseScreen(
        params.screen_width * dim_inc, params.res * dim_inc, params.r0_tot *
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

    v_mags = np.abs(np.random.normal(mean_v, del_v, params.num_of_steps-1))

    rnd_vecs = np.asarray([np.random.uniform(0, 1, 2) for i in range(params.num_of_steps-1)])

    v_vals = v_mags * rnd_vecs
    pixel_vals = v_vals * time_step / params.screen_width
    
    for rel in range(num_of_rels):
        for i in range(params.num_of_steps):
            part_t_screens[i] = t_screens[i, 0:params.res - 1, 0:params.res - 1]
    
    res_beams = propagate_modes(inp_beams, t_screens, delz_step)

if __name__ == '__main__':
    main()