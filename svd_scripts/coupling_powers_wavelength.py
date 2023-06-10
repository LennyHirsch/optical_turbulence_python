#THIS SCRIPT WILL LOAD ALL OF THE RECEIVED MODES AT DIFFERENT WAVELENGTHS AND DETERMINE
#-TOTAL POWER IN APERTURE
#TOTAL COUPLING STRENGTH

import numpy as np
import sys
import datetime
from tqdm import tqdm

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages/')

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/')

import propagation_functions as prop
import svd_prop_funcs as svd_funcs
import params
import svd_scripts.lg_propagation as lg_prop
import svd_scripts.svd_hg_decomposition as hg_decomp


def input_beam_calc(l_pos_min, p_max, z = 0):

    print("\nCalculating input modes...")

    inp = prop.BeamProfile(
        params.res, params.screen_width, params.wavelength)

    inp_beams = []

    for l in tqdm(range(-l_pos_min, l_pos_min + 1)):
        for p in range(p_max + 1):
            inp.laguerre_gaussian_beam(l, p, params.waist, z)
            inp_beams.append(inp.field)

    inp_beams = np.asarray(inp_beams)
    
    return inp_beams

def power_in_ap(modes, pixel_size):
    pws = []
    for md in modes:
        pws.append(np.trapz(...))
    return pws
    

def hg_basis():
    
def coupled_power():
    pass


def main(l_rng, p_max):

    collimate_beams = 1

    inp_beams = input_beam_calc(l_rng, p_max)
    #LOAD PHASE SCREENS
    t_screens = [prop.PhaseScreen(
        params.screen_width, params.res, params.r0_tot *
        (params.num_of_steps - 1)**(3/5), params.l0, params.L0
    ) for i in range(params.num_of_steps - 1)]

    phz_screens = []
    for t in t_screens:
        phz_screens.append(t.phz + t.phz_lo)
        phz_screens = np.asarray(phz_screens)

    # propagate LG modes
    res_beams = lg_prop.propagate_modes(inp_beams, t_screens, params.delz_step, collimate_beams)

    #Save resultant_beams
    lg_prop.save_data()

    #determin SVD modes of the system
    svd_trans_modes, svd_rec_modes = hg_decomp.calc_svd_modes(data_dir, params.wst)

    # save results

    # determine received power in all of the modes
    for md in svd_rec_modes:
        pw = np.trapz(np.trapz(md, dx = pixel_size), dx = pixel_size)
    
    #decompose into Hg modes
    for md in svd_rec_modes:
        olap = ...
        pass

    # determine coupling power in all modes
    for 

if __name__ == '__main__':
    main()