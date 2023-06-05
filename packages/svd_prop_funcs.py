#To Do:

#Introduce time evolution to the simulation 


#Clean up code!!!
# Document the code properly!!!

import sys
sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/packages')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import random
import time
from multiprocessing import Pool

import propagation_functions as prop


#Consider using parallel propagation instead
def channel_propagtion(inp_beams: list, t_screens: list, inp_ap_width: float, rec_ap_width: float, screen_width: float, delz_step: float, mode_num: int, res: int, wavelength: float = 1550e-9) -> np.ndarray:

    """
    Propagates a set of input beams through a turbulent channel.

    Args:
    - inp_beams (np.ndarray): Input beams.
    - t_screens (List[Screen]): List of screens representing the channel turbulence.
    - inp_ap_width (float): Width of the input aperture.
    - rec_ap_width (float): Width of the output (receiving) aperture.
    - screen_width (float): Width of the screen.
    - delz_step (float): Step size for propagating the beam through the screens.
    - mode_num (int): Number of modes.
    - res (int): Resolution of the output beam.
    - wavelength (float): Wavelength of the beam.

    Returns:
    - res_beam (np.ndarray): Output beams after propagation through the channel.
    """
    
    # need to introduce option to instead use hg modes for input modes
    # define class for generating modes
    probe_beam = prop.BeamProfile(res, screen_width, wavelength)
    #probe_beam.laguerre_gaussian_beam(0, 0, 1)

    # variables used to hold results of input and output modes
    res_beam = np.zeros((mode_num,  res, res), dtype=np.complex128)

    for i, beam in enumerate(tqdm(inp_beams)):

        probe_beam.field = beam
        probe_beam.hard_ap(inp_ap_width)
        probe_beam.free_space_prop(delz_step/2)

        for t in t_screens:
            phz_sc = t.phz + t.phz_lo

            probe_beam.apply_phase_screen(phz_sc)
            probe_beam.apply_sg_ap(0.9 * screen_width, 8)
            probe_beam.free_space_prop(delz_step)
        probe_beam.apply_sg_ap(0.9 * screen_width, 8)

        probe_beam.free_space_prop(delz_step/2)

        probe_beam.hard_ap(rec_ap_width)
        #probe_beam.low_pass_filter(30)
        res_beam[i] = probe_beam.field

    return(res_beam)

# DON'T USE

def channel_propagation_pll(lst: list) -> np.ndarray:
    """
    Simulate the propagation of an optical beam through a series of phase screens and 
    calculate the resulting field at the receiver aperture.

    Args:
    - lst: A list containing the following parameters:
        - inp_beam: A 2D numpy array representing the input beam profile
        - t_screens: A list of turbulence screen objects representing the phase screens
        - inp_ap_width: The diameter of the input aperture in meters
        - rec_ap_width: The diameter of the receiver aperture in meters
        - screen_width: The size of the turbulence screens in meters
        - delz_step: The step size for propagating through the screens in meters
        - res: The number of pixels per side of the simulation grid
        - wavelength: The wavelength of the incident beam in meters

    Returns:
    - A 2D numpy array representing the field at the receiver aperture after propagating 
      through the turbulence screens.
    """

    inp_beam = lst[0]
    t_screens = lst[1]
    inp_ap_width = lst[2]
    rec_ap_width = lst[3]
    screen_width = lst[4]
    delz_step = lst[5]
    res = lst[6]
    wavelength = lst[7]


    probe_beam = prop.BeamProfile(res, screen_width, wavelength)
    #res_beam = np.zeros((res, res), dtype = np.complex128)

    probe_beam.field = inp_beam
    probe_beam.hard_ap(inp_ap_width)
    probe_beam.free_space_prop(delz_step/2)

    for t in t_screens:
        phz_sc = t.phz + t.phz_lo

        probe_beam.apply_phase_screen(phz_sc)
        probe_beam.apply_sg_ap(0.9 * screen_width, 8)
        probe_beam.free_space_prop(delz_step)
    probe_beam.apply_sg_ap(0.9 * screen_width, 8)

    probe_beam.free_space_prop(delz_step/2)

    probe_beam.hard_ap(rec_ap_width)
    #probe_beam.low_pass_filter(30)
    return(probe_beam.field)

def trans_matrix_calc(res_beams: list, res: int, compact_res: int, screen_width: float) -> list:
    """
    Calculates the overlap integral between Hermite-Gaussian modes and beam profiles

    Args:
        res_beams (numpy.ndarray): Array of beam profiles with shape (num_beams, res, res)
        res (int): Resolution of beam profiles
        pascals_row (int): The highest index of the Hermite-Gaussian mode to be calculated
        waist (float): Waist size of the Hermite-Gaussian beam
        screen_width (float): Width of the screen in meters

    Returns:
        numpy.ndarray: A matrix of overlap integrals with shape (num_beams, num_modes)
    """
    t_mat = []

    low_bound = res//2 - (compact_res//2)
    hi_bound = res//2 + (compact_res//2)
    # currently decomposing in pixel basis. This is fine as long as 
    # resolution is 512 x 512 and economy SVD is performed

    for beam in tqdm(res_beams[:, low_bound:hi_bound, low_bound:hi_bound]):

        vec_beam = np.conj(np.reshape(beam, compact_res*compact_res) * (screen_width/res))

        t_mat.append(vec_beam.astype('complex128'))
    return t_mat

#ADJUST THIS TO MAKE USE OF GENERATE HG MODES
def trans_matrix_hg_calc(res_beams: list, res: int, pascals_row: int, waist: float, screen_width: float) -> list:
    """
    The function then iterates over the list of res_beams and calculates the overlap integral between each beam and the modes in inp_beams. The result is a matrix of overlap coefficients, t_mat, where each row corresponds to a beam in res_beams and each column corresponds to a mode in inp_beams.

    Inputs:

    res_beams: A list of 2D numpy arrays representing beam profiles.
    res: The resolution of the beam profiles.
    pascals_row: The maximum value of n and m for the Hermite-Gaussian modes to be generated.
    waist: The waist size of the Hermite-Gaussian modes.
    screen_width: The width of the screen on which the beams are projected.
    
    Outputs:

    t_mat: A 2D numpy array representing the overlap coefficients between the beams in res_beams and the Hermite-Gaussian modes."""
    t_mat = []
    inp = prop.BeamProfile(res, screen_width, 1.0)

    inp_beams = []
    mode_num = int(((1/2) * pascals_row) * (pascals_row + 1))

    #calculate hg modes with different waist sizes
    inp_beams = []
    for n in range(pascals_row):
        m_max = pascals_row-n
        for m in range(m_max):
            inp.hermite_gaussian_beam(n, m, waist)
            norm_const = np.trapz(np.trapz(
                inp.field * np.conj(inp.field), dx=screen_width/res), dx=screen_width/res)
            inp_beams.append(inp.field / np.sqrt(norm_const))
            #print(f"n: {n}, m: {m}, {np.trapz(np.trapz(inp.field * np.conj(inp.field), dx = screen_width/res), dx = screen_width/res)}")

    #TO DO: THIS CONJUGATE LOOKS TO BE THE WRONG WAY AROUND, BUT IT CANCELS WITH THE FACT THAT I AM NOT TAKING THE CONJUGATE OF THE V MATRIX
    for beam in tqdm(res_beams):
        olap = []
        for md in inp_beams:
            olap.append(
                np.trapz(np.trapz(md * np.conj(beam), dx=screen_width/res), dx=screen_width/res))

        t_mat.append(olap)
    
    return t_mat

def trans_matrix_hg_calc_circular(res_beams: list, res: int, pascals_row: int, waist: float, screen_width: float) -> list:
    """
    Calculate the transfer matrix for the circular aperture case using the Hermite-Gaussian (HG) modes.

    Parameters:
    res_beams (np.ndarray): Input beam profiles.
    res (int): Resolution of the beam profiles.
    pascals_row (int): Maximum index of the HG modes.
    waist (float): Waist size of the HG modes.
    screen_width (float): Width of the screen.

    Returns:
    np.ndarray: The transfer matrix for the circular aperture case.
    """
    t_mat = []
    inp = prop.BeamProfile(res, screen_width, 1.0)

    inp_beams = []
    mode_num = int(((1/2) * pascals_row) * (pascals_row + 1))

    #calculate hg modes with different waist sizes
    inp_beams = []
    for n in range(pascals_row):
        m_max = pascals_row-n
        for m in range(m_max):
            if n >=m:
                inp.hermite_gaussian_beam(n, m, waist)
                hld = inp.field
                hld = np.real((hld)/np.sqrt(np.trapz(np.trapz(
                    hld * np.conj(hld), dx=screen_width/res), dx=screen_width/res)))

                inp.hermite_gaussian_beam(m, n, waist)
                hld_2 = inp.field
                hld_2 = np.imag((hld_2)/np.sqrt(np.trapz(np.trapz(
                    hld_2 * np.conj(hld_2), dx=screen_width/res), dx=screen_width/res)))
                field = (hld + 1j * np.imag(hld_2))
            else:
                inp.hermite_gaussian_beam(n, m, waist)
                hld = inp.field
                hld = np.real((hld)/np.sqrt(np.trapz(np.trapz(
                    hld * np.conj(hld), dx=screen_width/res), dx=screen_width/res)))

                inp.hermite_gaussian_beam(m, n, waist)
                hld_2 = inp.field
                hld_2 = np.imag((hld_2)/np.sqrt(np.trapz(np.trapz(
                    hld_2 * np.conj(hld_2), dx=screen_width/res), dx=screen_width/res)))
                field = (hld - 1j * np.imag(hld_2))
                print(np.trapz(np.trapz(
                    field * np.conj(field), dx=screen_width/res), dx=screen_width/res))

            inp_beams.append(field)
            #print(f"n: {n}, m: {m}, {np.trapz(np.trapz(inp.field * np.conj(inp.field), dx = screen_width/res), dx = screen_width/res)}")

    for beam in tqdm(res_beams):
        olap = []
        for md in inp_beams:
            olap.append(
                np.trapz(np.trapz(md * np.conj(beam), dx=screen_width/res), dx=screen_width/res))

        t_mat.append(olap)
    
    return t_mat

def svd_hg_waists(res_beams: list, res: int, waist_lst: list, pascals_row: int, screen_width: float) -> tuple:
    """
    Calculates the singular value decompositions (SVDs) for a range of waist sizes of Hermite-Gaussian beams.

    Args:
    - res_beams (np.ndarray): Array of beam profiles with dimensions (num_beams, res, res).
    - res (int): Resolution of each beam profile in pixels.
    - waist_lst (np.ndarray): Array of waist sizes to calculate SVDs for.
    - pascals_row (int): Maximum row of Pascal's triangle to use in calculating Hermite-Gaussian modes.
    - screen_width (float): Width of the screen in meters.

    Returns:
    - svd_vals (list): List of lists containing the waist size, left singular vectors (u), singular values (s), right singular vectors (v), and transmission matrix (t_mat) for each SVD calculation.
    """
    #pascals_row = 15
    #waist_lst = np.linspace(0.025, 0.035, 10)

    svd_vals = []
    for wst in waist_lst:
        u, s, v, t_mat = svd_calc_hg(res_beams, res, pascals_row, wst, screen_width)
        svd_vals.append([wst, u, s, v, t_mat])

    #svd_vals_np = np.asarray(svd_vals)
    return svd_vals

def svd_calc_hg(res_beams: list, res: int, pascals_row: int, waist: float, screen_width: float) -> tuple:
    """
    Calculates the singular value decomposition (SVD) for Hermite-Gaussian beams with given waist size.

    Args:
    - res_beams (np.ndarray): Array of beam profiles with dimensions (num_beams, res, res).
    - res (int): Resolution of each beam profile in pixels.
    - pascals_row (int): Maximum row of Pascal's triangle to use in calculating Hermite-Gaussian modes.
    - waist (float): Waist size of Hermite-Gaussian beams in meters.
    - screen_width (float): Width of the screen in meters.

    Returns:
    - u (np.ndarray): Left singular vectors of the SVD.
    - s (np.ndarray): Singular values of the SVD.
    - v (np.ndarray): Right singular vectors of the SVD.
    - t_mat (list): Transmission matrix used in the SVD calculation.
    """

    t_mat = trans_matrix_hg_calc(res_beams, res, pascals_row, waist, screen_width)
    u, s, v = np.linalg.svd(np.asarray(t_mat).T, full_matrices=False)
    return u, s, v, t_mat

def svd_calc(res_beams: list, res: int, compact_res: int, screen_width: float) -> tuple:
    """
    Calculates the singular value decomposition (SVD) for given beam profile.

    Args:
    - res_beams (np.ndarray): Array of beam profiles with dimensions (num_beams, res, res).
    - res (int): Resolution of each beam profile in pixels.
    - compact_res (int): Resolution of the area around the aperture to use in the SVD calculation.
    - screen_width (float): Width of the screen in meters.

    Returns:
    - u (np.ndarray): Left singular vectors of the SVD.
    - s (np.ndarray): Singular values of the SVD.
    - v (np.ndarray): Right singular vectors of the SVD.
"""
    # use compact_res to partition only area around the aperture
    t_mat = trans_matrix_calc(res_beams, res, compact_res, screen_width)

    u, s, v = np.linalg.svd(np.asarray(t_mat).T, full_matrices=False)
    return u, s, v

def svd_inp_modes_calc(v: np.ndarray, inp_beams: np.ndarray, mode_num: int, res: int, trans_modes_num: int, screen_width: float) -> np.ndarray:
    """
    Calculates the input modes in the transmission basis using singular value decomposition (SVD).
    
    Args:
        v: A numpy array of shape (mode_num, mode_num) containing the right singular vectors of the
           overlap matrix.
        inp_beams: A numpy array of shape (mode_num*res*res,) containing the input beams to be decomposed.
        mode_num: An integer representing the number of input modes.
        res: An integer representing the number of pixels in each dimension of the input beams.
        trans_modes_num: An integer representing the number of transmission modes.
        screen_width: A float representing the width of the screen in meters.
    
    Returns:
        A numpy array of shape (trans_modes_num, res, res) representing the input modes in the transmission basis.

        Note: Do not conjugate or transpose v as returned from np.linalg.svd
    """
    inp_arr = np.reshape(inp_beams, (mode_num, res, res))
    svd_trans_modes = np.zeros((trans_modes_num, res, res), dtype = np.complex128)

    for k in tqdm(range(trans_modes_num)):
        for i, j in enumerate(inp_arr):
            #DO NOT CONJUGATE OR TRANSPOSE. NUMPY SVD RETURNS HERMITIaN V!!!!!
            svd_trans_modes[k] += j * v[k, i]

    # normalise power in all transmission modes
    for k in range(trans_modes_num):
        norm_const = np.abs(np.trapz(np.trapz(svd_trans_modes[k] * np.conj(svd_trans_modes[k]), dx = screen_width/res), dx = screen_width/res))
        
        svd_trans_modes[k] = svd_trans_modes[k] / np.sqrt(norm_const)
    return svd_trans_modes

def svd_reconstruct(mode_num: int, meas_beams: list, svd_rec_modes:list, res: int) -> tuple:
    """
    Reconstructs the input beams from the measured beams and the SVD-processed received modes.

    Parameters:
    -----------
    mode_num: int
        Number of modes to reconstruct.
    meas_beams: List[np.ndarray]
        List of numpy arrays containing the measured beams.
    svd_rec_modes: np.ndarray
        Array of numpy arrays containing the received modes.
    res: int
        The resolution of the beam profiles.
    
    Returns:
    --------
    Tuple containing:
    reconstruct: np.ndarray
        Array of numpy arrays containing the reconstructed input modes.
    olap_res: List[List[float]]
        List of lists containing the overlap values for each reconstructed input mode with each received mode.
    """
    # I have to normalise the received modes to get accurate overlap calculations
    # This doesn't make sense to me. If the received modes are orthogonal
    # then why does it matter if the modes are orthongonal? I am calculating the
    # overlap? This should account for differences in amplitude?? It could be 
    # something to do with interference and phase causing issues as we add more
    # and more modes together. I don't understand it, but this is what Lubomir 
    # told me to do
    #     
    reconstruct = np.zeros((mode_num, res, res), dtype = np.complex128)
    olap_res =[]
    for i, meas in enumerate(tqdm(meas_beams[0:mode_num])):
        olap_beam = []
        for rec_modes in svd_rec_modes:
            norm_mode = rec_modes/(np.sqrt(np.sum(np.abs(rec_modes)**2.0)))
            olap = np.trapz(np.trapz(meas * np.conj(norm_mode)))
            reconstruct[i] += olap * rec_modes
            olap_beam.append(olap)
        olap_res.append(olap_beam)
    return reconstruct, olap_res

def generate_hg_modes(pascals_row: int, res: int ,screen_width: float, wavelength: float, wst: float) -> np.ndarray:
    """
    The `generate_hg_modes` function calculates the Hermite-Gaussian (HG) modes for overlap.

    Parameters:
    - `pascals_row`: The maximum order of HG modes to calculate (integer).
    - `res`: The number of points per dimension of the simulation grid (integer).
    - `screen_width`: The width of the screen (in meters) that the HG modes will be projected onto (float).
    - `wavelength`: The wavelength of the light used in the simulation (in meters) (float).
    - `wst`: The waist of the beam in meters (float).

    Returns:
    - `hg_beams`: A list of numpy arrays containing the HG modes (complex)."""

    print("\nCalculating HG modes for overlap...")
    inp = prop.BeamProfile(
            res, screen_width, wavelength)
    
    hg_beams = []

    #should be own function
    for n in tqdm(range(pascals_row)):
        m_max = pascals_row-n
        for m in range(m_max):
            inp.hermite_gaussian_beam(n, m, wst)
                
            norm_const = np.trapz(np.trapz(
                    inp.field * np.conj(inp.field), dx=screen_width/res), dx=screen_width/res)

            hg_beams.append(inp.field / np.sqrt(norm_const))

    return hg_beams


def hg_decomp_calc(ref_beams, pascals_row, decomp_wst, res, screen_width, wvl):
    """
    Calculates the decomposition matrix of ref_beams into SP Hg's. Consider normalisation after to get more accurate results for reconstruction

    params:
    - ref_beams: a list of beams to be decomposed
    - pascals_row: dictates the max 'row' of HGS to use for the decomposition
    - decomp_wst: the waist of the decomposition beams
    - res: the simulation resolution (int -> square screen)
    - screen_width: the simulation screen width
    - wvl: the wavelength of the beams

    returns:
    - a 2D overlap matrix between all of the input beams and the HG beams
    """

    hg_modes = generate_hg_modes(
        pascals_row, res, screen_width, wvl, decomp_wst)

    pixel_size = screen_width/res

    olap = []

    for i in range(len(ref_beams)):
        olap_hld = []
        for j in range(len(hg_modes)):
            olap_ini = np.trapz(np.trapz(
                ref_beams[i] * np.conj(hg_modes[j]), dx=pixel_size), dx=pixel_size)
            olap_hld.append(olap_ini)
        olap.append(olap_hld)

    return olap

def hg_crsstalk_matrix(basis_vecs, beams):
    """
    finds the crosstalk matrix between two vectors. The idea is that one vector will be the HG decomposition of the basis vectors, and the other will be the HG decomposition of the received beams. After all, the HG values are all that the superpixels 'should' be able to determine
    """
    olap_wvl = []
    mode_num = len(basis_vecs)

    for i in range(mode_num):
        olap_hld = []
        for j in range(mode_num):
            olap_ini = np.dot(beams[i], np.conj(basis_vecs[j]))
            olap_hld.append(olap_ini)
        olap_wvl.append(olap_hld)

    return olap_wvl

#Plotting functions
def plot_gaussian_reconstruction(reconstruct: np.ndarray, res_beams: np.ndarray, plt_mode: list):
    """
    Plots the reconstructed and measured Gaussian beam profiles for a given mode.

    Args:
        reconstruct (np.ndarray): Array of reconstructed beam profiles.
        res_beams (np.ndarray): Array of measured beam profiles.
        plt_mode (list[int]): List of modes to be plotted.

    Returns:
        None
    """

    fig_1, (ax1, ax2) = plt.subplots(1, 2)
    recon_plt_1 = ax1.imshow(np.abs(reconstruct[plt_mode]))
    res_plt_1 = ax2.imshow(np.abs(res_beams[plt_mode]))
    fig_1.colorbar(recon_plt_1, ax=ax1, fraction=0.046, pad=0.04)
    fig_1.colorbar(res_plt_1, ax=ax2, fraction=0.046, pad=0.04)

    ax1.set_title('Reconstructed')
    ax2.set_title('Measured')

    fig_1.tight_layout()

    fig_2, (ax3, ax4) = plt.subplots(1, 2)
    recon_plt_2 = ax3.imshow(np.angle(
        reconstruct[plt_mode]), cmap='hsv', interpolation='none')
    res_plt_2 = ax4.imshow(
        np.angle(res_beams[plt_mode]), cmap='hsv', interpolation='none')
    fig_2.colorbar(recon_plt_2, ax=ax3, fraction=0.046, pad=0.04)
    fig_2.colorbar(res_plt_2, ax=ax4, fraction=0.046, pad=0.04)

    ax3.set_title('Reconstructed')
    ax4.set_title('Measured')

    fig_2.tight_layout()

    plt.draw_all()
    plt.show()

#LEGACY

'''
# Plotting functions
def plot_inp_out(inp_beams: np.ndarray, res_beams: np.ndarray, ind_1: int):
    """
    Plots the input and output beams of a single mode.

    Args:
    - inp_beams (np.ndarray): A 3D array of input beams for all modes.
    - res_beams (np.ndarray): A 3D array of reconstructed beams for all modes.
    - ind_1 (int): An integer index of the mode.

    Returns:
    - bool: Returns False if an IndexError occurs, otherwise no return value.

    Raises:
    - IndexError: If the values for indices relating to plotting are out of bounds.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        inp_plt = ax1.imshow(np.abs(inp_beams[ind_1]))

        res_plt = ax2.imshow(np.abs(res_beams[ind_1]))
    except IndexError:
        print('IndexError: Values for indices relating to plotting are out of bounds. Continuing without plotting.')
        return False
    
    fig.colorbar(inp_plt, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(res_plt, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

#It was decided that this attempted to do too many steps at once. It has been replaced with smaller functions
def svd_calc_hg_decomp(res_beams: np.ndarray, res: int, pascals_row: int, waist_lst: list):
    """
    Calculate the singular value decomposition (SVD) of the measured beams and perform a Hermite-Gaussian decomposition
    with different waist sizes.

    Args:

    res_beams: 2D array containing the measured beams
    res: int value representing the number of pixels per axis
    pascals_row: int value representing the number of rows in Pascal's triangle to be considered
    waist_lst: list of float values representing the different waist sizes
    Returns:

    reconstruct: 4D array containing the reconstructed beams for each waist size
    """

    inp = prop.BeamProfile(res, screen_width, wavelength)

    inp_beams_waists = []
    mode_num = int(((1/2) * pascals_row) * (pascals_row + 1))

    #calculate hg modes with different waist sizes
    for waist in waist_lst:
        inp_beams = []
        for n in range(pascals_row):
            m_max = pascals_row-n
            for m in range(m_max):
                inp.hermite_gaussian_beam(n, m, waist)
                inp_beams.append(inp.field)

        inp_beams_waists.append(inp_beams)

    inp_beams_waists = np.asarray(inp_beams_waists)
    reconstruct = np.zeros(
        (len(waist_lst), mode_num, res, res), dtype=np.complex128)

    #attempt to reconstruct the svd_received modes using different HG waist sizes
    for i, wst in enumerate(inp_beams_waists):
        flat_wst = np.reshape(wst, (mode_num, res, res))
        for j, beam in enumerate(res_beams[0:mode_num]):
            for md in flat_wst:
                olap = np.trapz(np.trapz(beam * np.conj(md)))
                reconstruct[i, j] += olap * md

    return reconstruct

def err_det(reconstruct_modes, meas_beams, res):

    mode_num = np.shape(reconstruct_modes)
    err = []
    for i in range(mode_num):
        arr_err = reconstruct_modes[i] - meas_beams[i]

def crosstalk_plot(beams_true: np.ndarray, beams_tst: np.ndarray, mode_num: int, res: int):
    """
    Inputs:

    beams_true (numpy.ndarray): Input array of true beam profiles with shape (mode_num * res * res).
    beams_tst (numpy.ndarray): Input array of predicted beam profiles with shape (mode_num * res * res).
    mode_num (int): Number of modes being considered.
    res (int): Resolution of the beam profiles.
    
    Output:

    None.

    Description:
    This function takes in two arrays of beam profiles beams_true and beams_tst and computes the crosstalk between them. The input beam profiles are reshaped according to mode_num and res and the crosstalk is calculated as the overlap integral between each pair of true and predicted beam profiles. The function then plots the resulting crosstalk matrix using imshow and displays a colorbar for reference.
    """
    beams_true_flt = np.array(beams_true.reshape(mode_num, res, res))
    beams_tst_flt = np.array(beams_tst.reshape(mode_num, res, res))
    cross = np.zeros((mode_num, mode_num))
    for i in tqdm(range(mode_num)):
        for j in range(mode_num):
        # for this case I do not normalise my inut vectors
            olap=np.trapz(np.trapz(beams_true_flt[i] * np.conj(beams_tst_flt[j])))
            cross[i, j] = np.abs(olap)


    fig, ax = plt.subplots(1)
    im = ax.imshow(cross/cross[0, 0])

    plt.colorbar(im, fraction = 0.046, pad = 0.04)
    plt.show()

'''

if __name__ == '__main__':

    #Has been incorporated into other files, should now not be run directly
    #I shall keep the code regardless should it be required at some point
    print("Error: File should not be run dircetly. Please adjust code is you would like to run this directly.")
    quit()

    #system parameters
    delz = 500
    inp_ap_width = 0.15
    rec_ap_width = 0.15

    #turbulence parameters
    r0_tot = 0.02
    L0 = 1e9
    l0 = 1e-9

    #input basis parameters
    input_basis = 'lg'
    waist = 0.02

    if input_basis == 'lg':
        
        l_pos_min = 5
        p_max = 5
        mode_num = (p_max+1) * (l_pos_min*2 + 1)

    elif input_basis == 'hg':
        pascals_row = 15
        mode_num = int(((1/2) * pascals_row) * (pascals_row + 1))

    # simulation parameters
    screen_width = 0.4
    num_of_steps = 20 + 1
    wavelength = 1550e-9
    res = 512
    delz_step = delz/num_of_steps

    #svd_parameters
    # how many modes I want to actually get from the SVD
    trans_modes_num = 10

    #how many SVD modes I want to use to reconstruct res_beams
    err_mode_num = 10
    svd_wavelengths = [1550e-9]
    compact_res = int(np.ceil(res * (rec_ap_width/screen_width)))
 
    # get system fresnel number. Provides indication for expected number of 
    # free-space modes in non-turbulent system
    fres_num = prop.fresnel_calc(inp_ap_width, rec_ap_width, 1550e-9, delz)
    print(f"Fresnel Number for system: {fres_num}")

    # generate the turbulent screens
    # at the moment I am generating these screens for single propagtion. Need 
    # to adjust this so that I have random windspeeds that give me a coherence 
    # time. This will likely involve generating large screens, and propagating through in a loop

    print('\nGenerating turblent screens...')
    t_screens = [prop.PhaseScreen(
        screen_width, res, r0_tot * (num_of_steps - 1)**(3/5), l0, L0
        ) for i in range(num_of_steps - 1)]
    
    for t in t_screens:
        t.mvk_screen()
        t.mvk_sh_screen()

    # calculate input modes
    print("\nCalculating input modes...")
    inp = prop.BeamProfile(res, screen_width, wavelength)

    #inp_beams = np.zeros((2 * l_pos_min + 1, p_max + 1, res, res), dtype = np.complex128)
    inp_beams = []

    # code for calculating LG modes as inputs

    if input_basis == 'lg':
        plt_mode = ((p_max) * (l_pos_min + 1))
        for l in tqdm(range(-l_pos_min, l_pos_min + 1)):
            for p in range(p_max + 1):
                inp.laguerre_gaussian_beam(l, p, waist)
                inp_beams.append(inp.field)
                #inp_beams[l + l_pos_min, p] = inp.field
                
    elif input_basis == 'hg':
        plt_mode = 0
    # code for calculating hg modes as inputs
        for n in tqdm(range(pascals_row)):
            m_max = pascals_row-n
            for m in range(m_max):
                inp.hermite_gaussian_beam(n, m, waist)
                inp_beams.append(inp.field)

    #inp_beams = np.reshape(inp_beams, (mode_num, res, res))
    inp_beams = np.asarray(inp_beams)

    # perform propagtion of all modes
    print("\nPerforming propagations through channel...")
    
    #start = time.time()
    #res_beams = channel_propagtion(inp_beams, t_screens, inp_ap_width, 
    #                               rec_ap_width, screen_width, delz_step, 
    #                               mode_num, res)
    #end = time.time()
    #print(f"\nTime for sequential execution: {end-start}")

    #generate data list for parallel propagation
    data_lst =[]
    for beam in inp_beams:
        stor = [beam, t_screens, inp_ap_width, rec_ap_width, screen_width,
                delz_step, res, wavelength]
        data_lst.append(stor)

    start = time.time()
    with Pool(9) as p:
        res_beams = p.map(channel_propagation_pll, data_lst)

    end = time.time()
    print(f"\n Time for parallel execution: {end-start}")
    res_beams = np.asarray(res_beams)
    #del(data_lst)

    #print(np.allclose(np.abs(res_beams), np.abs(parallel_beams)))
    #print(np.allclose(np.angle(res_beams), np.angle(parallel_beams)))

    # optional plotting of example input and output functions. Comment out to ignore
    #plot_inp_out(inp_beams, res_beams, 0)
    
    # perform crosstalk matrix calculation for input. Optional. Comment out to ignore. If we get significant crosstalk then the aperture is too small for the input modes
    #crosstalk_plot(inp_beams, mode_num, res)

    # perform svd in pixel basis at receiver
   

    print('Performing SVD calculation on range of waists ')
    svd_vals = svd_hg_waists(res_beams, res)

    # get overlap and cross talk

    data_lst = []

    reconstruct =[]
    for i, svd_val in enumerate(svd_vals):

        wst = svd_val[0]
        u = svd_val[1]
        s = svd_val[2]
        v = svd_val[3]
        t_mat = svd_val[4]

        print([wst, s])
        
        recon = np.zeros((1, res, res), dtype = np.complex128)
        pascals_row = 15
        hg_rec = prop.BeamProfile(res, screen_width, wavelength)

    #calculate hg modes with different waist sizes
        hg_beams = []
        for n in range(pascals_row):
            m_max = pascals_row-n
            for m in range(m_max):
                hg_rec.hermite_gaussian_beam(n, m, wst)
                hg_beams.append(hg_rec.field)

            svd_trans_modes = svd_inp_modes_calc(v, inp_beams, mode_num, res,
                                            trans_modes_num)

        svd_rec_modes = np.zeros((trans_modes_num,
                              res, res), dtype=np.complex128)  
        
        
        svd_rec_modes = channel_propagtion(svd_trans_modes, t_screens, inp_ap_width, rec_ap_width, screen_width, delz_step, mode_num, res)
        print(np.shape(svd_rec_modes))

        reconstruct, olap_res = svd_reconstruct(
            err_mode_num, svd_rec_modes, hg_beams, res)
        print(olap_res)
 
        crosstalk_plot(np.asarray(reconstruct), 10, 512)
        # for n in range(pascals_row):
        #    m_max = pascals_row-n
        #    for m in range(m_max):
        #        inp.hermite_gaussian_beam(n, m, wst)
        #        recon += inp.field * v[i, ii]
        #        ii += 1

        #plot_gaussian_reconstruction(recon, res_beams, 0)
        # determine reconstruction
        #reconstruct.append(svd_reconstruct(err_mode_num, res_beams, recon, res))
        # plot gaussian reconstruction
        #save plots

    quit()



    u, s, v = svd_calc(res_beams, res, compact_res)

    # plot singular values
    print(f"\nFirst {trans_modes_num} singular values: \n {s[0:trans_modes_num]/s[0]}")

    #fig, ax = plt.subplots()
    #ax.plot(s[0:200]/s[0])
    #plt.title('66LG Input, Pixel Output, Normalised, No Turb, 15cm - 15cm, 1500m')
    #plt.ylabel('Singular Value')
    #plt.xlabel('Modes')
    #plt.show()

    # perform propagation of new SVD modes
    print("\nCalculating input SVD modes...")
    svd_trans_modes = svd_inp_modes_calc(v, inp_beams, mode_num, res, 
                                         trans_modes_num)

    svd_rec_modes = np.zeros((np.size(svd_wavelengths), trans_modes_num, 
                              res, res), dtype = np.complex128) 

    print('\nPerforming SVD modes propagation...')
    print(f"Number of wavelengths: {np.size(svd_wavelengths)}")

    for i, wvl in enumerate(svd_wavelengths):

        print(f"\nWavelength Num: {i + 1} of {np.size(svd_wavelengths)}")
        print(f"Wavelength: {wvl}")

        data_lst_svd = []
        for beam in svd_trans_modes:
            stor = [beam, t_screens, inp_ap_width, rec_ap_width, screen_width,
                delz_step, res, wvl]
            data_lst_svd.append(stor)

        with Pool(9) as p:
            svd_rec_modes[i] = p.map(channel_propagation_pll, data_lst_svd)

    #reconstructing modes using SVD basis 
    print(f"\nAttempting to reconstruct measured modes using SVD basis...")
    reconstruct = svd_reconstruct(err_mode_num, res_beams, svd_rec_modes[0], res)
    
    #print("\nDetermining optimum beam waist for reconstruction of received modes in HG basis...")
    #pascals_row = 20
    #waist_lst = np.linspace(0.005, 0.02, 10)
    #reconstruct_hg = svd_calc_hg_decomp(svd_rec_modes[0], res, pascals_row, waist_lst)

    #print(f"\nDetermining reconstruction error...")
    #err = err_det(reconstruct, res_beams, res)

    #reconstruct_hg = np.asarray(reconstruct_hg)
    #diffs = np.zeros((len(waist_lst)))

#    for i, rec_hg in enumerate(reconstruct_hg):
#        for ii in range(len(rec_hg)):
#            sq_diff = np.sum(np.abs(rec_hg[ii] - svd_rec_modes[0,ii])**2.0)
#            diffs[i] += sq_diff
#    print(diffs)
#    print(waist_lst)

    #plot gaussian propagated mode and reconstructed mode

    plot_gaussian_reconstruction(reconstruct, res_beams, 0)

    #diffs = np.zeros((20))
    #for j , beam in enumerate(res_beams):
    #    for i in range(20):
    #        sq_diff = np.sum(np.abs(((i/10)*reconstruct[j]) - beam)**2.0)
    #        diffs[i] += sq_diff
    #plt.plot(np.log(diffs))
    #plt.show()
    #print(diffs)

    #crosstalk_plot(svd_rec_modes[0], mode_num, res)
