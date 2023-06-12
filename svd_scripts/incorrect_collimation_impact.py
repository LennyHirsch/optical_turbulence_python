#I realised that the results that I have sent to Aleks are actually slightly incorrect. The collimation was performed using the decomposition beam waist, instead of the input beam waist. This script is to check the impact of this difference, and see if it results in significantly worse results than using the correct decompostion

import numpy as np
import sys
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import svd_prop_funcs as svd_funcs
import propagation_functions as prop

def crosstalk_matrix(beams, res, screen_width, ref_wvl, pixel_size):
    for beam in beams:
        beam_pow = np.trapz(
            np.trapz(np.abs(beam * np.conj(beam)), dx=pixel_size), dx=pixel_size)
        beam /= np.sqrt(beam_pow)


    hg_decomp = svd_funcs.hg_decomp_calc(beams, 5, 0.024, res, screen_width, ref_wvl)

    for hg_decom in hg_decomp:
        hg_decom /= np.sqrt(np.sum(np.abs(hg_decom * np.conj(hg_decom)) ))

    olap = np.asarray(svd_funcs.hg_crsstalk_matrix(hg_decomp, hg_decomp))

    crss = np.abs(olap * np.conj(olap))

    return crss


def main(realisation):
    
    # parameters defined in initial readme file. Ensure that these are correct for every situation

    r0 = 0.02   
    delz = 1000
    num_of_steps = 21
    screen_width = 0.4
    inp_ap_width = 0.15
    rec_ap_width = 0.15
    res = 512
    mode_num = 15
    waist = 0.02
    ref_wvl = 1.55e-6

    pixel_size = screen_width/res

    delz_step = delz/num_of_steps

    #import initial screens
    #import svd transmission modes

    if realisation == 13:
        data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/svd_modes/ecoc/svd_alex_files/new_new_svd_modes/process_screens13/'

        screens = np.load('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v013/reference/turb_screens.npy')


    elif realisation == 15:
        data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/svd_modes/ecoc/svd_alex_files/collimated_modes/process_screens015/'

        screens = np.load('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v015/reference/turb_screens.npy')



    trans_modes = np.load( data_dir + 'trans_modes_0_024.npy')

    #convert phase screens to objects
    t_screens = []
    for i in range(len(screens)):
        t_screen = prop.PhaseScreen( screen_width, res, r0, 0.001, 100.0)
        t_screen.phz = screens[i]
        t_screen.phz_lo = 0
        t_screens.append(t_screen)
    
    print('Performing 1550nm propagation as reference...')
    #propagate and collimate 1550nm beam. decompose into HG modes
    ref_beams = svd_funcs.channel_propagtion(trans_modes, t_screens, inp_ap_width, rec_ap_width, screen_width, delz_step, mode_num, res)


    coll_beams_correct = prop.coll_calc(ref_beams, res, screen_width, ref_wvl, waist, delz)

    coll_beams_incorr = prop.coll_calc(ref_beams, res, screen_width, ref_wvl, 0.024, delz)


    coll_corr_crss = crosstalk_matrix(coll_beams_correct, res, screen_width, ref_wvl, pixel_size)

    coll_incorrect_crss = crosstalk_matrix(coll_beams_incorr, res, screen_width, ref_wvl, pixel_size)

    #plt figures
    fig, axs = plt.subplots(1, 2)
    
    im0 = axs[0].imshow(10 * np.log10(coll_corr_crss))
    axs[0].set_title('correct collimation')
    #im0 = axs[0].imshow(np.angle(coll_beams_incorr[1]))
    im1 = axs[1].imshow(10 * np.log10(coll_incorrect_crss))
    axs[1].set_title('incorrect collimation')
    #im1 = axs[1].imshow(np.angle(coll_beams_correct[1]))

    #fig.colorbar(im0, ax = axs[0])
    #fig.colorbar(im1, ax = axs[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    main(13)
    main(15)