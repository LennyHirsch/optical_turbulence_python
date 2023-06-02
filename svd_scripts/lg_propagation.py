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

def propagation_loop(delz_step: float, data_dir: str, collimate_beams: int = 0):
    """
    Function: propagation_loop(delz_step, data_dir, collimate_beams = 0)
    
    This function generates turbulent screens, calculates input modes, propagates those modes through the turbulent screens, and saves the resulting data to a specified directory.

    Parameters:

        delz_step: the step size for propagation through the turbulent screens.
        data_dir: the directory where the resulting data will be saved.
        collimate_beams: an optional parameter that determines whether or not the propagated beams will be collimated. If not specified, it defaults to 0. Current collimation assumes that the beam can be collimated with a free-space beam equivalent gaussian propagation. This should be resolved
    
    Returns:
        None
    """
    print('\nGenerating turblent screens...')

    t_screens = [prop.PhaseScreen(
        params.screen_width, params.res, params.r0_tot *
        (params.num_of_steps - 1)**(3/5), params.l0, params.L0
    ) for i in range(params.num_of_steps - 1)]

    phz_screens = []

    for t in t_screens:
        t.mvk_screen()
        t.mvk_sh_screen()
        #reformat phase screens for later saving
        phz_screens.append(t.phz + t.phz_lo)
    phz_screens = np.asarray(phz_screens)

    inp_beams = calculate_input_modes()
    res_beams = propagate_modes(inp_beams, t_screens, delz_step, collimate_beams)

    data = [phz_screens, inp_beams, res_beams]

    save_data(data, data_dir)

def calculate_input_modes(z:float = 0):
    """The function calculate_input_modes calculates the input modes of the optical system using Laguerre-Gaussian beams. The function takes a single optional argument z, which is the propagation distance from the input aperture. The function returns a NumPy array of the calculated input beams.

    Parameters:

    z (optional, default=0): The propagation distance from the input aperture.
    Returns:

    inp_beams: A NumPy array of the calculated input beams.
    Note:

    This function assumes that params is a global object containing the necessary system parameters.
    This function uses the prop.BeamProfile class to generate Laguerre-Gaussian beams."""
    # calculate input modes
    print("\nCalculating input modes...")
    inp = prop.BeamProfile(
        params.res, params.screen_width, params.wavelength)

    #inp_beams = np.zeros((2 * l_pos_min + 1, p_max + 1, res, res), dtype = np.complex128)
    inp_beams = []
    # lgs = [[0, 0],
    #        [0, 1],
    #        [0, 2],
    #        [1, 0],
    #        [-1, 0],
    #        [1, 1],
    #        [-1, 1],
    #        [2, 0],
    #        [-2, 0],
    #        [2, 1],
    #        [-2, 1],
    #        [3, 0],
    #        [-3, 0],
    #        [4, 0],
    #        [-4, 0]]

    #for lg in tqdm(lgs):
    #    l = lg[0]
    #    p = lg[1]
    #    inp.laguerre_gaussian_beam(l, p, params.waist)
    #    inp_beams.append(inp.field)

    for l in tqdm(range(-params.l_pos_min, params.l_pos_min + 1)):
        for p in range(params.p_max + 1):
            inp.laguerre_gaussian_beam(l, p, params.waist, z)
            inp_beams.append(inp.field)

    inp_beams = np.asarray(inp_beams)
    
    return inp_beams

def propagate_modes(inp_beams: np.ndarray, t_screens: np.ndarray, delz_step: float, collimate_beams: int = 0, wavelength: float = params.wavelength):
    """
    Function: propagate_modes(inp_beams, t_screens, delz_step, collimate_beams = 0, wavelength = params.wavelength)

    This function performs the propagation of input beams through a series of turbulent screens, using the svd_funcs.channel_propagation_pll() function to perform the actual propagation calculations. The resulting propagated beams are optionally collimated and returned as an array.

    Parameters:

    inp_beams: an array of input beams to be propagated.
    t_screens: an array of turbulent screens through which the input beams will be propagated.
    delz_step: the step size for propagation through the turbulent screens.
    collimate_beams: an optional parameter that determines whether or not the propagated beams will be collimated. If not specified, it defaults to 0.
    wavelength: the wavelength of the input beams. If not specified, it defaults to the wavelength parameter from params.
    
    Returns:
    
    None"""
    print(f'\nCurrent Wavelength is: {wavelength}')
    print("\nPerforming propagations through channel...")
    data_lst = []
    for beam in inp_beams:
        stor = [beam, t_screens, params.inp_ap_width, params.rec_ap_width, params.screen_width,
                delz_step, params.res, wavelength]
        data_lst.append(stor)

    start = time.time()
    with Pool(9) as p:
        res_beams = p.map(svd_funcs.channel_propagation_pll, data_lst)

    end = time.time()
    print(f"\n Time for parallel execution: {end-start}")
    res_beams = np.asarray(res_beams)

    #collimate beams
    #This method will only work under the assumption that the gaussian mode is the 30th entry of the multidimensional array
    if collimate_beams == 1:
        print('\nCollimating beams...')
       # free_space_diff = calculate_input_modes(params.delz)
        free_space_diff = prop.BeamProfile(params.res, params.screen_width, wavelength)
        free_space_diff.laguerre_gaussian_beam(0, 0, params.waist, params.delz)
        for i in range(len(res_beams)):
            res_beams[i] *= np.exp(1j * np.angle(free_space_diff.field))

    return res_beams

def save_data(data: list, data_dir: str):
    """
    Save generated data to the specified directory as numpy arrays and create a readme file with the simulation
    parameters.

    Parameters:
        data(List): List containing the generated data as numpy arrays. The list should be of the form:
            [phz_screens, inp_beams, res_beams]
        data_dir(str): Path of the directory where the data will be saved.

    Returns:
        None
    """
    f_name = ['turb_screens', 'inp_beams', 'res_beams']
    for i, f in enumerate(f_name):
        if os.path.isfile(data_dir + f + '.npy'):
            
            cont = input(
                f'\nWARNING: {data_dir + f}.npy already exists. Replace file? (y/n): ')
            
            if cont != 'y':
                continue
            
        print(f'\nWriting file: {data_dir + f}.npy')
        np.save(data_dir + f + '.npy', data[i])

    pms = [f'r0: {params.r0_tot}\n',
              f'Propagation distance: {params.delz}\n'
              f'Number of steps: {params.num_of_steps}\n'
              f'Screen width: {params.screen_width}\n',
              f'Inp Aperture Diameter: {params.inp_ap_width}\n'
              f'Receiver Aperture Diameter: {params.rec_ap_width}\n'
              f'Outer scale of turbulence: {params.L0}\n'
              f'Inner Scale of turbulence: {params.l0}\n'
              f'Waist: {params.waist}\n'
              f'Number of modes: {params.mode_num}'
              ]

    #param_write = input('Write new parameters readme? (y/n): ')
    if os.path.isfile(data_dir + 'readme.txt'):
        param_write = input('Readme already exists in location. Overwrite file? (y/n): ')
        if param_write != 'y':
            return 0
    with open(data_dir + 'readme.txt', 'w') as f:
        f.writelines(pms)

def main(num_of_loops: int, dir_name: str, collimate_beams: int):
    delz_step = params.delz/params.num_of_steps
    """This is the main function that orchestrates the simulation loop. It takes in three parameters: num_of_loops, dir_name, and collimate_beams.

    Parameters:

    num_of_loops: an integer that specifies the number of iterations to perform the simulation loop.
    dir_name: a string that specifies the directory name to save the simulation data.
    collimate_beams: an integer that specifies whether to collimate the beams after propagation (1 for True and 0 for False).
    The function starts by calculating the step size (delz_step) for the simulation loop. It then calculates the Fresnel number for the system using the input and receiver aperture diameters, wavelength, and propagation distance.

    Next, it enters the simulation loop, where it creates a new directory for each iteration of the loop to store the simulation data. If the directory already exists, it will prompt the user to continue or not.

    For each iteration, it calls the propagation_loop function, passing in the delz_step, data_dir, and collimate_beams parameters.

    The function doesn't return any values; it only prints information about the progress of the simulation loop."""
    # get system fresnel number. Provides indication for expected number of
    # free-space modes in non-turbulent system
    fres_num = prop.fresnel_calc(
        params.inp_ap_width, params.rec_ap_width, 1550e-9, params.delz)
    print(f"Fresnel Number for system: {fres_num}")
    print(f"\nFried parameter: {params.r0_tot}")

    # generate the turbulent screens
    # at the moment I am generating these screens for single propagtion. Need
    # to adjust this so that I have random windspeeds that give me a coherence
    # time. This will likely involve generating large screens, and propagating through in a loop

    for i in range(num_of_loops):

        #file saving parameters
        today = date.today()
        d1 = today.strftime("%Y%m%d")
        data_dir = f"data/{d1}" + dir_name + f"l_{params.l_pos_min}_p_{params.p_max}_v{str(i).zfill(3)}/"

        if not os.path.isdir(data_dir):
            print(f"\nCreating dir: {data_dir}")
            os.makedirs(data_dir)
        else:
            print(f"\nWARNING: Directory {data_dir} already exists. Data may be overwritten. Continuing using {data_dir} as saving directory...")

        print(f"\nLoop {i + 1} of {num_of_loops}")
        propagation_loop(delz_step, data_dir, collimate_beams)

if __name__ == '__main__':
    num_of_loops = 1
    dir_name = '/lg_prop_rep/'
    main(num_of_loops, dir_name)
