import numpy as np
import sys
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import svd_prop_funcs as svd_funcs
import propagation_functions as prop

def calc_wavelengths(start_freq, end_freq, num):

    c = 2.99792458 * 10 ** 8
    frequency_list = np.linspace(start_freq, end_freq, num)
    wavelengths = c/(frequency_list * 10 ** 12)

    return wavelengths

def wvl_propagation(trans_modes, screens, inp_ap_width, rec_ap_width, screen_width, delz_step, mode_num, res, wvl, ref_wvl):


    k_new = 2 * np.pi / wvl
    k_ini = 2 * np.pi/ref_wvl

    screens = screens * (k_new/k_ini)

    #convert phase screens to objects
    t_screens = []

    for i in range(len(screens)):
        t_screen = prop.PhaseScreen(screen_width, res, 0.02, 0.001, 100.0)
        t_screen.phz = screens[i]
        t_screen.phz_lo = 0
        t_screens.append(t_screen)

    beams_wvl = svd_funcs.channel_propagtion(
        trans_modes, t_screens, inp_ap_width, rec_ap_width, screen_width, delz_step, mode_num, res, wvl)

    return beams_wvl

def hg_reconstruction(hg_decomp, ref_beams, pascals_row, decomp_wst, res, screen_width, mode_num):
    hg_modes = svd_funcs.generate_hg_modes(
    pascals_row, res, screen_width, 1.55e-9, decomp_wst)

    recon_beam = np.zeros((res, res), dtype = np.complex128)

    for i in range(mode_num):
        recon_beam += hg_modes[i] * hg_decomp[10][i]
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.abs(recon_beam))
    axs[1].imshow(np.abs(ref_beams[10]))
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.angle(recon_beam))
    axs[1].imshow(np.angle(ref_beams[10]))
    plt.show()
    plt.close()

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

    start_freq = 190.1
    end_freq = 197.25
    num_of_freqs = 144
    delz_step = delz/num_of_steps
    wavelength_lst = calc_wavelengths(start_freq, end_freq, num_of_freqs)
    wavelength_lst = wavelength_lst

    #import initial screens
    #import svd transmission modes

    if realisation == 13:
        data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/svd_modes/ecoc/svd_alex_files/new_new_svd_modes/process_screens13/'

        screens = np.load('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v013/reference/turb_screens.npy')

        save_dir = '/Users/ultandaly/Desktop/wavelength_invariance_test_results/realisation013/'

    elif realisation == 15:
        data_dir = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/svd_modes/ecoc/svd_alex_files/collimated_modes/process_screens015/'

        screens = np.load('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/wavelength_test/coll_fin_l_5_p_5_v015/reference/turb_screens.npy')

        save_dir = '/Users/ultandaly/Desktop/wavelength_invariance_test_results/realisation015/'

    print('\nSaving Wavelength list to file...')
    for i, wvl in enumerate(wavelength_lst):
        line = f'\nWavelength {str(i).zfill(3)}: {str(round(wvl*10**6, 5))}'
        with open(save_dir + '../wavlength_lst.txt', 'a') as f:
            f.write(line)

    print('Creating Save Directories...')
    wavelength_dirs = []
    for i in range(len(wavelength_lst)):
        wavelength_dir = save_dir + 'wavelength_' + str(i).zfill(4)
        if not os.path.exists(wavelength_dir):
            os.makedirs(wavelength_dir)
        wavelength_dirs.append(wavelength_dir)

    trans_modes = np.load( data_dir + 'trans_modes_0_024.npy')

    '''
    #DONT USE, IT LOOKS LIKE SOMETHING MIGHT HAVE GONE WRONG WITH THE INITIAL CALCULATION, AND RESULTS ARE SLIGHTLY WORSE. USED TO CONFIRM THAT THE DECOMPOSITION IS WORKING AS EXPECTED. IT IS INDEED WORKING

    #rec_beams_ini = np.load(data_dir + 'screens_0_024.npy')
    #hg_decomp = np.load(data_dir + 'hg_decomp_0_024.npy')
    #hg_decomp_1550 = hg_decomp_calc(
    #    rec_beams_ini, 5, 0.024, res, screen_width, ref_wvl, mode_num)
    #for i in range(15):
    #    print(f'\nHg Mode {i}: {hg_decomp_1550[0][i]}, {hg_decomp[0, i]}')

'''
    
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

    ref_beams = prop.coll_calc(ref_beams, res, screen_width, ref_wvl, waist, delz)

    
    hg_decomp_1550 = np.asarray(svd_funcs.hg_decomp_calc(ref_beams, 5, 0.024, res, screen_width, ref_wvl))

    for iii, hg_decom in enumerate(hg_decomp_1550):
        hg_decomp_1550[iii] /= np.sqrt(np.sum(np.abs(hg_decom * np.conj(hg_decom))))
        
    
    #hg_reconstruction(hg_decomp_1550, ref_beams, 5, 0.024, res, screen_width, mode_num)
    #wavelength_lst = [1.551e-6, 1.55e-6]
    #quit()
    # propagate all wavelengths through channel
    for ii, wvl in enumerate(wavelength_lst):
        print(f"Wavelength propagation {ii+1} of {len(wavelength_lst)}...")
        beams_wvl = wvl_propagation(trans_modes, screens, inp_ap_width, rec_ap_width, screen_width, delz_step, mode_num, res, wvl, ref_wvl)
        
        beams_wvl = prop.coll_calc(beams_wvl, res,
                          screen_width, ref_wvl, waist, delz)
        
        #save beams themselves
        np.save(wavelength_dirs[ii] + '/collimated_beam_profiles.npy' , beams_wvl)

    for i in range(len(wavelength_lst)):

        print(f"Wavelength decomposition {i + 1} of {len(wavelength_lst)}...")

        tst = np.load(wavelength_dirs[i] + '/collimated_beam_profiles.npy')

        for iii, beam in enumerate(tst):
            beam_pow = np.trapz(np.trapz(np.abs(beam * np.conj(beam)), dx = pixel_size), dx = pixel_size)
            tst[iii] /= np.sqrt(beam_pow)

        
        hg_decomp_new = svd_funcs.hg_decomp_calc(tst, 5, 0.024, res, screen_width, ref_wvl)
        #save hg decompositions
        
        #this will effectively capture the actual coupling efficiencies into each mode
        np.save(wavelength_dirs[i] + '/hg_decomposition_powers.npy', hg_decomp_new)

        for iii, hg_decom in enumerate(hg_decomp_new):
            hg_decomp_new[iii] /= np.sqrt(np.sum(np.abs(hg_decom * np.conj(hg_decom))))

        # this will capture the relative coupling poewr into each mode
        np.save(wavelength_dirs[i] + '/hg_decomposition.npy', hg_decomp_new)
        
        #total power in each column of the crosstalk matrix should equal the power in the reconstructed mode
        #CROSSTALK POWER
        olap = np.asarray(svd_funcs.hg_crsstalk_matrix(hg_decomp_1550, hg_decomp_new))

        crss = np.abs(olap * np.conj(olap))

        #beams are the rows in the plot and basis vecs are the columns
        np.save(wavelength_dirs[i] + '/crosstalk.npy', crss)

              

if __name__ == '__main__':

    main(13)
    main(15)