import sys
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random

from packages import propagation_functions as prop

#need to restructure this so that modes are provided as an argument, rather than calculated in this function

def channel_propagtion(inp_beams, t_screens, l_pos_min, p_max, waist, inp_ap_width, rec_ap_width, screen_width, delz_step, wavelength = 1550e-9):
    # need to introduce option to instead use hg modes for input modes
    # define class for generating modes
    inp = prop.BeamProfile(res, screen_width, wavelength)

    # variables used to hold results of input and output modes
    res_beam = np.zeros((2 * l_pos_min  + 1, p_max + 1, res, res)) * (0 + 0j)


    for l in tqdm(range(-l_pos_min, l_pos_min + 1)):
        for p in range(p_max + 1):

            inp.laguerre_gaussian_beam(l, p, waist)

            inp.hard_ap(inp_ap_width)
            inp.free_space_prop(delz_step/2)

            for t in t_screens:
                phz_sc = t.phz + t.phz_lo

                inp.apply_phase_screen(phz_sc)
                inp.apply_sg_ap(0.9 * screen_width, 8)
                inp.free_space_prop(delz_step)
            inp.apply_sg_ap(0.9 * screen_width, 8)

            inp.free_space_prop(delz_step/2)

            inp.hard_ap(rec_ap_width)
            res_beam[l + l_pos_min, p] = inp.field

    beams = res_beam
    return(beams)

def plot_inp_out(inp_beams, res_beams, ind_1, ind_2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        inp_plt = ax1.imshow(np.abs(inp_beams[ind_1, ind_2]))

        res_plt = ax2.imshow(np.abs(res_beams[ind_1, ind_2]))
    except IndexError:
        print('IndexError: Values for indices relating to plotting are out of bounds. Continuing without plotting.')
        return False
    
    fig.colorbar(inp_plt, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(res_plt, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

def crosstalk_plot(beams, mode_num, res):
    beams_flt = np.array(beams.reshape(mode_num, res, res))
    cross = np.zeros((mode_num, mode_num))
    for i in tqdm(range(mode_num)):
        for j in range(mode_num):
        # for this case I do not normalise my inut vectors
            olap=np.trapz(np.trapz(beams_flt[i] * np.conj(beams_flt[j])))
            cross[i, j] = np.abs(olap)


    fig, ax = plt.subplots(1)
    im = ax.imshow(cross)

    plt.colorbar(im, fraction = 0.046, pad = 0.04)
    plt.show()


def trans_matrix_calc(res_beams, res, compact_res):
    t_mat = []

    low_bound = res//2 - (compact_res//2)
    hi_bound = res//2 + (compact_res//2)

    #flatten resultant arrays
    # currently decomposing in pixel basis. This is fine as long as resolution is 512 x 512 and economy SVD is performed

    res_beam_flat = np.reshape(
        res_beams[:, :, low_bound:hi_bound, low_bound:hi_bound], (mode_num, compact_res, compact_res))
    for beam in tqdm(res_beam_flat):

        vec_beam = np.conj(np.reshape(beam, compact_res*compact_res))
        # 64-bit complex to reduce memory load
        t_mat.append(vec_beam.astype('complex64'))
    return t_mat

def svd_calc(compact_res, res_beams, res):

    # use compact_res to partition only area around the aperture
    t_mat = trans_matrix_calc(res_beams, res, compact_res)
    u, s, v = np.linalg.svd(np.asarray(t_mat).T, full_matrices=False)
    return u, s, v

def svd_inp_modes_calc(v, inp_beams, mode_num, res, trans_modes_num):

    inp_arr = np.reshape(inp_beams, (mode_num, res, res))
    svd_trans_modes = np.zeros((trans_modes_num, res, res)) * (0 + 0j)

    for k in tqdm(range(15)):
        for i, j in enumerate(inp_arr):
            #DO NOT CONJUGATE OR TRANSPOSE. NUMPY SVD RETURNS HERMITIaN V!!!!!
            svd_trans_modes[k] += j * v[k, i]

    # normalise power in all transmiddion modes
    for k in range(15):
        print(np.sqrt(np.sum(np.abs(svd_trans_modes[k])**2.0)))
        svd_trans_modes[k] = svd_trans_modes[k] / np.sqrt(np.sum(np.abs(svd_trans_modes[k])**2.0))

    return svd_trans_modes

    
if __name__ == '__main__':

    #system parameters
    delz = 1000
    inp_ap_width = 0.15
    rec_ap_width = 0.15

    #turbulence parameters
    r0_tot = 0.01
    L0 = 1e9
    l0 = 1e-9

    # simulation parameters
    screen_width = 0.4
    
    num_of_steps = 20 + 1
    wavelength = 1550e-9
    res = 512
    l_pos_min = 1
    p_max = 1
    waist = 0.015

    delz_step = delz/num_of_steps
    mode_num = (p_max+1) * (l_pos_min*2 + 1) 
    trans_modes_num = 15
    svd_wavelengths = [1550e-9]
    # i can actually calculate this based on screen_width and rec_ap_width
    compact_res = np.ceil(res * (rec_ap_width/screen_width))

    # get system fresnel number. Provides indication for expected number of free-space modes in non-turbulent system
    fres_num = prop.fresnel_calc(inp_ap_width, rec_ap_width, 1550e-9, delz)
    print(f"Fresnel Number for system: {fres_num}")

    # generate the turbulent screens
    # at the moment I am generating these screens for single propagtion. Need to adjust this so that I have random windspeeds that give me a coherence time. This will likely involve generating large screens, and propagating through in a loop

    t_screens = [prop.PhaseScreen(
    screen_width, res, r0_tot * (num_of_steps - 1)**(3/5), l0, L0) for i in range(num_of_steps - 1)]
    
    for t in t_screens:
        t.mvk_screen()
        t.mvk_sh_screen()

    # calculate input modes
    inp = prop.BeamProfile(res, screen_width, wavelength)
    inp_beams = np.zeros((2 * l_pos_min + 1, p_max + 1, res, res)) * (0 + 0j) 
    for l in tqdm(range(-l_pos_min, l_pos_min + 1)):
        for p in range(p_max + 1):
            inp.laguerre_gaussian_beam(l, p, waist)
            inp_beams[l + l_pos_min, p] = inp.field

    # perform propagtion of all modes
    res_beams = channel_propagtion(inp_beams, t_screens, l_pos_min, p_max, waist, inp_ap_width, rec_ap_width, screen_width, delz_step)

    # optional plotting of example input and output functions. Comment out to ignore

    #plot_inp_out(inp_beams, res_beams, 0, 0)

    
    # perform crosstalk matrix calculation for input. Optional. Comment out to ignore. If we get significant crosstalk then the aperture is too small for the input modes

    #crosstalk_plot(inp_beams, mode_num, res)

    # perform svd in pixel basis at receiver
    u, s, v = svd_calc(res_beams, res, compact_res)

    # plot singular values
    print(s[0:trans_modes_num])

    #fig, ax = plt.subplots()
    #ax.plot(s[0:200]/s[0])
    #plt.title('66LG Input, Pixel Output, Normalised, No Turb, 15cm - 15cm, 1500m')
    #plt.ylabel('Singular Value')
    #plt.xlabel('Modes')
    #plt.show()

    # perform propagation of new SVD modes
    svd_trans_modes = svd_inp_modes_calc(v, inp_beams, mode_num, res, trans_modes_num)

    svd_rec_modes = np.zeros((np.size(svd_wavelengths), svd_trans_modes))
    for i, wvl in enumerate(tqdm(svd_wavelengths)):
        svd_rec_modes[i] = channel_propagtion(t_screens, l_pos_min, p_max)
