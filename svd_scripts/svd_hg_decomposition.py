import sys
import os
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import numpy as np
from tqdm import tqdm
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

import propagation_functions as prop
import svd_prop_funcs as svd_funcs

#ENSURE THIS IS POINTING TO CORRECT PARAMS FILE
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data')
import params

def load_data(data_dir: str):
    """
    The load_data function loads previously saved data from the specified directory.

    Args:

    data_dir (str): the directory path where the data is saved
    Returns:

    res_beams (numpy.ndarray): an array of resolution beams
    inp_beams (numpy.ndarray): an array of input beams
    t_screens (numpy.ndarray): an array of turbulence screens"""

    res_beams = np.load(data_dir + '/res_beams.npy')
    inp_beams = np.load(data_dir + '/inp_beams.npy')
    t_screens = np.load(data_dir + '/turb_screens.npy')

    return res_beams, inp_beams, t_screens

def get_mean_coupling_strength(data_dirs: list, res: int, waist_lst: float, pascals_row: int, screen_width: float, plt_bool: int):
    """
    Calculate the mean coupling strength of a range of waists for a set of data directories.

    Args:
        data_dirs (list): A list of strings containing the paths of the directories containing the data to analyze.
        res (int): The resolution of the beams in pixels.
        waist_lst (float): The range of waist values to analyze.
        pascals_row (int): The Pascal's row number to use for the Hermite-Gaussian modes.
        screen_width (float): The width of the turbulent screens used in the propagation.
        plt_bool (int): A flag indicating whether to plot the results.

    Returns:
        np.ndarray: An array containing the mean proportional coupling strength for each calculated mode.
    """
#NEEDS AMENDMENT. WILL NOT DEAL WITH MULTIPLE WAISTS ACCURATELY AS OF YET
    num_of_loops = len(data_dirs)
    s_tot = []

    for i, data_dir in enumerate(data_dirs):
        print(f"\nLoop {i + 1} of {num_of_loops}")
        res_beams, _, _ = load_data(data_dir)
   
        print('Performing SVD calculation on range of waists ')
        svd_vals = svd_funcs.svd_hg_waists(
            res_beams, res, waist_lst, pascals_row, screen_width)
        
        for svd_val in svd_vals:
            s = svd_val[2]
            wst = svd_val[0]
            print([s, np.sum(s)])
            s_tot.append(s)

    mean_s = np.mean(s_tot, axis = 0)
    std_s = np.std(s_tot, axis = 0)
    x = np.linspace(1, 15, 15)

    if plt_bool == 1:
        plt.errorbar(x, mean_s, yerr = std_s, capsize=2.0)
        plt.title(f'Mean Coupling Strength ({len(data_dirs)} Realisations) Waist: {wst}')
        plt.ylabel('Proportional Coupling Strength')
        plt.xlabel('Calculated Mode')
        plt.show()

    print(sum(mean_s))
    return mean_s
        #for rec in sv

def save_svd_propagation(data_dir: str, wst: float, mode_plt: list = [-1]):
    """
    The function save_svd_propagation saves the SVD results, including the overlap and reconstruction matrices, in a specified directory after calculating the SVD modes, overlap, and recreation using the calc_svd_modes and generate_hg_modes functions. It also allows an optional parameter mode_plt to plot the Gaussian reconstruction for a specific mode number.

    Parameters:

    data_dir : str : path to the directory containing the data to be analyzed.
    wst : float : waist size for generating HG modes.
    mode_plt : list : optional list of mode numbers to plot the Gaussian reconstruction.
    Returns:

    ret : bool : a boolean value indicating whether the save was successful or not.
"""
    svd_trans_modes, svd_rec_modes = calc_svd_modes(data_dir, wst)
    
    hg_beams = svd_funcs.generate_hg_modes(params.pascals_row, params.res, params.screen_width, params.wavelength, wst)

    # should be own function
    print("\nCalculating overlap for recreation...")
    all_recs = []
    olap_hld = []
    for beam in tqdm(svd_rec_modes):
        rec = np.zeros((params.res, params.res), dtype=np.complex128)
        olap_tmp = []
        for md in hg_beams:
            olap = np.trapz(np.trapz(
                beam * np.conj(md), dx=params.screen_width/params.res), dx=params.screen_width/params.res)
            rec += olap * md
            olap_tmp.append(olap)
        all_recs.append(rec)
        olap_hld.append(olap_tmp)

    all_recs = np.asarray(all_recs)
    olap_hld = np.asarray(olap_hld)

    data_save_dir = os.path.abspath(os.path.join(os.path.dirname(data_dir), '..', 'process_screens/'))

    save_dir_str = data_dir[-4:]
    data_save_dir += save_dir_str

    ret = save_svd_results(data_save_dir, wst, olap_hld, svd_trans_modes, svd_rec_modes)

    #allow option plotting recreated and geneated modes
    #move this into own function that loads already generated modes
    for md in mode_plt:
        if ((md < 0) or (md > params.trans_modes_num)) :
            continue
        svd_funcs.plot_gaussian_reconstruction(all_recs, svd_rec_modes, md)

def save_svd_results(data_save_dir: str, wst: float, olap_hld: list, svd_trans_modes: list, svd_rec_modes: list):
    """
    Save the SVD results in numpy format (.npy) to a given directory.

    Args:
    - data_save_dir (str): Path to the directory where the data will be saved.
    - wst (float): Waist value used for the SVD calculation.
    - olap_hld (list): List of overlap coefficients between the reconstructed modes and the generated modes.
    - svd_trans_modes (list): List of singular value matrices representing the transmitted modes.
    - svd_rec_modes (list): List of singular value matrices representing the reconstructed modes.

    Returns:
    - int: 1 if the save is successful.

    Raises:
    - None
    """
    wst_str = str(wst).replace('.', '_')

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    #save all data
        np.save(data_save_dir + '/hg_decomp_' + str(wst_str) + '.npy', olap_hld)
        np.save(data_save_dir + '/screens_' + str(wst_str) + '.npy', svd_rec_modes)
        np.save(data_save_dir + '/trans_modes_' + str(wst_str) + '.npy', svd_trans_modes)
    else:
        ovw = input('Warning: Saving directory already exists. Risk overwriting files? (y/n)')
        if ovw == 'y':
            np.save(data_save_dir + '/hg_decomp_' + str(wst_str) + '.npy', olap_hld)
            np.save(data_save_dir + '/screens_' + str(wst_str) + '.npy', svd_rec_modes)
            np.save(data_save_dir + '/trans_modes_' + str(wst_str) + '.npy', svd_trans_modes)

    return 1

def calc_svd_modes(data_dir: str, wst: float):
    """
    Calculate SVD modes for a given data directory and waist size.

    Args:
    data_dir (str): The directory where the data is stored.
    wst (float): The waist size to use for the calculation.

    Returns:
    Tuple of two 3D numpy arrays representing the SVD transformed input modes and reconstructed modes.
    """
    if not os.path.isdir(data_dir):
        print('Data directory does not exist. Exiting program...')
        quit()
    
    print(f"\nLoading data from: {data_dir}")
    res_beams, inp_beams, t_screens = load_data(data_dir)

    delz_step = params.delz/params.num_of_steps
    t_screens_obj = [prop.PhaseScreen(
        params.screen_width, params.res, 1.0 * (params.num_of_steps - 1)**(3/5), 1.0, 1.0) for i in range(params.num_of_steps - 1)]

    for i, t in enumerate(t_screens):
        t_screens_obj[i].phz_lo = 0.0
        t_screens_obj[i].phz = t

    print(f'Performing SVD calculation using waist: {wst}')
    svd_val = svd_funcs.svd_hg_waists(
        res_beams, params.res, [wst], params.pascals_row, params.screen_width)
    
    wst = svd_val[0][0]
    u = svd_val[0][1]
    s = svd_val[0][2]
    v = svd_val[0][3]
    t_mat = svd_val[0][4]

    svd_trans_modes = svd_funcs.svd_inp_modes_calc(
        v, inp_beams, params.mode_num, params.res, params.trans_modes_num, params.screen_width)

    svd_rec_modes = np.zeros((params.trans_modes_num,
                                params.res, params.res), dtype=np.complex128)

    svd_rec_modes = svd_funcs.channel_propagtion(
        svd_trans_modes, t_screens_obj, params.inp_ap_width, params.rec_ap_width, params.screen_width, delz_step, params.trans_modes_num, params.res)
    
    coll_mode = prop.BeamProfile(params.res, params.screen_width, params.wavelength)
    coll_mode.hermite_gaussian_beam(0, 0, wst, params.delz)

    for i in range(len(svd_rec_modes)):
        svd_rec_modes[i] *= np.exp(1j * np.angle(coll_mode.field))

    return svd_trans_modes, svd_rec_modes

def wavelength_crsstlk(data_dir, wst, wvl_lst):
    pass

def main(data_dir: str):
    #refactor so that data_dir is passed and can be passed as a list
    #refactor so that it can take a range of wavelengths and range of waists
    #data_dir_start = '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages/svd_scripts/data/20230418/lg_prop_rep/l_2_p_2_v'
    #data_dirs = [data_dir_start + str(i).zfill(3) for i in range(100)]

    #get_mean_coupling_strength(data_dirs, params.res, params.waist_lst, params.pascals_row, params.screen_width, 1)

    #quit()
    #point to relevant data directory
    #data_dir = f'/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages/svd_scripts/data/20230418/lg_prop_rep/l_2_p_2_v{str(10).zfill(3)}'

    if not os.path.isdir(data_dir):
        print('Data directory does not exist. Exiting program...')
        quit()

    delz_step = params.delz/params.num_of_steps

    #load data from numpy bin files
    res_beams, inp_beams, t_screens = load_data(data_dir)

    #need to define the turbulent screens as PhaseScreen objects

    t_screens_obj = [prop.PhaseScreen(
        params.screen_width, params.res, 1.0 * (params.num_of_steps - 1)**(3/5), 1.0, 1.0) for i in range(params.num_of_steps - 1)]

    for i, t in enumerate(t_screens):

        t_screens_obj[i].phz_lo = 0.0
        t_screens_obj[i].phz = t

    print('Performing SVD calculation on range of waists ')
    svd_vals = svd_funcs.svd_hg_waists(
        res_beams, params.res, params.waist_lst, params.pascals_row, params.screen_width)

    reconstruct = []

    for i, svd_val in enumerate(svd_vals):

        wst = svd_val[0]
        u = svd_val[1]
        s = svd_val[2]
        v = svd_val[3]
        t_mat = svd_val[4]

        print([wst, s, sum(s)])

        #I have confirmed that these modes are the same as the ones I use for the wavelength invariance test
        svd_trans_modes = svd_funcs.svd_inp_modes_calc(
            v, inp_beams, params.mode_num, params.res, params.trans_modes_num, params.screen_width)
        
        svd_rec_modes = np.zeros((params.trans_modes_num,
                                  params.res, params.res), dtype=np.complex128)

        svd_rec_modes = svd_funcs.channel_propagtion(
            svd_trans_modes, t_screens_obj, params.inp_ap_width, params.rec_ap_width, params.screen_width, delz_step, params.trans_modes_num, params.res)

        #COLLIMATE THESE MODES
        #svd_rec_modes = coll_calc(svd_rec_modes, 1, 512, 0.4, 1.55e-6, 0.02, 1000)
        hg_rec = np.zeros((params.res, params.res))

        print("\nCalculating HG modes for overlap...")
        '''
        inp = prop.BeamProfile(
            params.res, params.screen_width, params.wavelength)
        hg_beams = []
        for n in tqdm(range(params.pascals_row)):
            m_max = params.pascals_row-n
            for m in range(m_max):
                inp.hermite_gaussian_beam(n, m, wst)
                #print(f'{n}, {m}')
                norm_const = np.trapz(np.trapz(
                    inp.field * np.conj(inp.field), dx=params.screen_width/params.res), dx=params.screen_width/params.res)
                tmp = inp.field/np.sqrt(norm_const)
                #print(np.trapz(np.trapz(tmp * np.conj(tmp), dx = screen_width/res), dx = screen_width/res))
                hg_beams.append(inp.field / np.sqrt(norm_const))
        '''

        # make these programatic instead of defined
        hg_beams =svd_funcs.generate_hg_modes(5, 512, 0.4, 1.55e-9, 0.024)

        print("\nCalculating overlap for recreation...")
        all_recs = []
        olap_hld = []

        #THIS LOOKS TO BE TOO MANY STEPS AT ONCE, THERE IS A DIFFERENCE BETWEEN MY WAVELENGTH INVARIANT RECREATION AND THIS. INVESTIGATE FURTHER
        svd_rec_modes  = prop.coll_calc(svd_rec_modes, params.res, params.screen_width, 1.55e-6, 0.02, 1000)
        
        svd_hg_decomp = np.asarray(svd_funcs.hg_decomp_calc(svd_rec_modes, params.pascals_row, params.waist, params.res, params.screen_width, params.wavelength))

        for iii, hg_decomp in enumerate(svd_hg_decomp):
            svd_hg_decomp[iii] /=np.sqrt(np.sum(np.abs(hg_decomp * np.conj(hg_decomp))))

        '''

        for beam in tqdm(svd_rec_modes):
            rec = np.zeros((params.res, params.res), dtype=np.complex128)
            olap_tmp = []
            for md in hg_beams:
                olap = np.trapz(np.trapz(
                    beam * np.conj(md), dx=params.screen_width/params.res), dx=params.screen_width/params.res)
            #    rec += olap * md
                olap_tmp.append(olap)
            #all_recs.append(rec)
            olap_hld.append(olap_tmp)

#ALL NEW
            
        for iii, hg_decomp in enumerate(olap_hld):
            olap_hld[iii] /= np.sqrt(np.sum(np.abs(hg_decomp * np.conj(hg_decomp))))
        '''
        for ii in range(15):
            rec = np.zeros((params.res, params.res), dtype=np.complex128)
            for jj in range(15):
                rec += hg_beams[jj] * svd_hg_decomp[ii][jj]
            all_recs.append(rec)
            
#TO HEERE
        all_recs = np.asarray(all_recs)
        #np.save('/Users/ultandaly/Desktop/tmp_svd/np_arrs/hg_decomp.npy', olap_hld)
        #np.save('/Users/ultandaly/Desktop/tmp_svd/np_arrs/screens.npy', svd_rec_modes)
        #print(np.sum(all_recs[0] * np.conj(all_recs[0])))
        svd_funcs.plot_gaussian_reconstruction(all_recs, svd_rec_modes, 0)
        #svd_funcs.plot_gaussian_reconstruction(all_recs, svd_rec_modes, 0)
        svd_funcs.plot_gaussian_reconstruction(all_recs, svd_rec_modes, 10)

        #svd_funcs.crosstalk_plot(all_recs, svd_rec_modes, trans_modes_num, 512)
        #print(np.abs(np.sum(all_recs[0] * np.conj(svd_rec_modes[0]))))

        crss = np.zeros((15, 15))

        for iii, rec in enumerate(all_recs):
            all_recs[iii] /= np.sqrt(np.trapz(np.trapz(all_recs[iii] * np.conj(all_recs[iii]))))

        for ii in range(15):
            for j in range(15):

                # /np.sqrt(np.trapz(np.trapz(all_recs[ii] * np.conj(all_recs[ii]), dx = params.screen_width/params.res), dx = params.screen_width/params.res))
                tmp = all_recs[ii]
                # /np.sqrt(np.trapz(np.trapz(svd_rec_modes[j] * np.conj(svd_rec_modes[j]), dx = params.screen_width/params.res), dx = params.screen_width/params.res))
                tmp_2 = svd_rec_modes[j]

                crss[ii, j] = np.abs(np.trapz(np.trapz(all_recs[ii] * np.conj(all_recs[j]),
                                     dx=params.screen_width/params.res), dx=params.screen_width/params.res))
        print(crss)
        plt.imshow(10*np.log10((crss)))
        plt.show()
    #crss_lst_mean = np.mean(crss_lst, axis = 0)
    #plt.imshow(crss_lst_mean)
    #plt.title('SVD Received Modes Crosstalk')
    #plt.colorbar()
    #plt.show()

        #for rec in svd_rec_modes:
        #    print(np.sqrt(np.sum(np.abs(rec)**2.0)))

        #reconstruct, olap = svd_funcs.svd_reconstruct(err_mode_num, res_beams, #svd_rec_modes[0:err_mode_num], res)

       # svd_funcs.plot_gaussian_reconstruction(reconstruct, res_beams, 0)
    # are my received SVD modes orthogonal?
        #print(np.shape(svd_rec_modes))
        #svd_funcs.crosstalk_plot(svd_rec_modes, err_mode_num, res)

if __name__ == '__main__':
    main('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps/coll_fin_l_5_p_5_v013/')
