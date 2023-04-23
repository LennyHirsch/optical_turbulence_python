import numpy as np
import h5py
import propagation_functions as prop
import cfd_processing
import sys
import os
import csv
from datetime import datetime
from tqdm import tqdm

def main():
    #HANDLE DIRECTORIES AND LOAD RO AND MVALS
    #----------------------------------------------------
    r0_path = "/Users/ultandaly/Desktop/PhD_Data/python_tst_vals/r0_vals.h5"
    m_vals_path = "/Users/ultandaly/Desktop/PhD_Data/python_tst_vals/mVals.h5"

    parent_folder = "/Volumes/Newton/V4_results/"
    u_file_path = parent_folder + "python_formatting_u/"
    cfd_file_path = parent_folder + "python_formatting/"

    now = datetime.now()
    date_string = now.strftime("%Y%m%d")

    res_dir = parent_folder + date_string + '/fixed_time_ev_1'
    hf_dir = res_dir + '/hf_screens/'

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if not os.path.exists(hf_dir):
        os.makedirs(hf_dir)

    with h5py.File(r0_path, 'r') as r0_raw:
        dset_name = list(r0_raw.keys())[0]
        r0 = r0_raw[dset_name][()]

    with h5py.File(m_vals_path, 'r') as m_vals_raw:
        dset_name = list(m_vals_raw.keys())[0]
        m_vals = m_vals_raw[dset_name][()]

    seeds_file = res_dir + '/seeds.csv'
    #------------------------------------------------------

    # DECLARE ALL SIMULATION CONSTANTS
    #------------------------------------------------------

    #free_space_between_cells = [
    #    [0, 13.33333, 13.33333, 13.3333, 0],
    #    [40.0, 0.0, 0.0, 0.0, 0.0],
    #    [0.0, 0.0, 0.0, 0.0, 40.0],
    #    [0.0, 5.0, 5.0, 10.0, 20.0],
    #    [20.0, 10.0, 5.0, 5.0, 0.0]
    #]
    free_space_between_cells = [
        [60.0, 0],
        [ 0, 60.0]]

    transmitApWidth = 0.075
    lg_p = 0
    max_lg = 1

    channel_length = 45
    part_width = 20
    avg_screens = 20
    pixel_width = 0.0125
    screen_width = pixel_width * part_width
    target_res = 512
    filt_point = 106.629

    ini_channel_width = 101

    delz = pixel_width * avg_screens

    wavelength = 1550 * 10**-9.0
    l0 = 0.0002
    L0 = 0.125

    prop_start = 2
    prop_end = 41
    screens_per_cell = prop_end - prop_start + 1

    section_break = 100

    section_time_length = 500 + section_break
    number_of_sections = 1
    initial_start_time = 0
    initial_end_time = initial_start_time + section_time_length - 1 - section_break

    start_time = np.linspace(initial_start_time, initial_start_time +
                             (number_of_sections-1) * section_time_length, number_of_sections)

    end_time = np.linspace(initial_end_time, initial_end_time +
                           (number_of_sections-1) * section_time_length, number_of_sections)

    dim_mult = 2

    tot_prop_screens = screens_per_cell * number_of_sections

    m_vals_full = (np.tile(m_vals[prop_start:prop_end + 1], number_of_sections))

    hf_save_space = 25

    sg_ap_width = 0.94*screen_width
    sg_ap_power = 8

    #--------------------------------------------------------

    #SET RANDOM SEED
    #Need to pass through a different random seed to each screen generation
    #each number should pass through to a different generator in each process, and then impact the generator in spr. Be careful to ensure that different seeds are passed each time. I think the best way to save the seeds is to save them at each time step IN THEIR OWN DIRECTORY. Adding them to datasets associated with composite screens will be messy

    rng = np.random.default_rng(100)
    # save entire simulation seed. Might make life easier

    #convert r0 to single list containing only values considered in the propagation
    r0_flat = np.ndarray.flatten(r0[0:number_of_sections, prop_start:prop_end + 1])
    cn2 = [prop.r0_to_cn2(r, wavelength, delz) for r in r0_flat]

    #declare cfd class
    cfd_screens_channel = cfd_processing.cfdChannel(
        cfd_file_path, "h5", pixel_width, wavelength, part_width)

    #set random seed for each screen
    seeds = rng.integers(1, 2**16, len(cn2))
    #save these seeds in a file

    with open(seeds_file, 'w') as f:
        row_name = 'initial_seeds'
        writer = csv.writer(f)
        seeds_list = list(seeds)
        seeds_list.insert(0, row_name)
        writer.writerow(seeds_list)


    screens = [cfd_processing.cfdScreens(
        int(target_res), dim_mult, filt_point, wavelength, part_width, screen_width) for i in range(len(cn2))]

    for i, screen in enumerate(screens):
        screen.hf_filtered_screen(cn2[i], l0, L0, delz, seeds[i])

    u_files_channel = cfd_processing.cfdChannelWind(
        u_file_path, ".h5", pixel_width, part_width, ini_channel_width - 1)


    all_files_u = u_files_channel.files

    #gets the file associated with each cell
    #dim0 contains the first cell time stamp times
    #dim1 contains the second time stamp files etc.
    all_files_u = [all_files_u[int(start_time[i]):int(
        end_time[i] + 1)] for i in range(number_of_sections)]

    #organise files so that all cells in a single propagation are in the same list
    all_files_u = np.transpose(all_files_u)

    #time stamp loop
    for i, cell_list in enumerate(tqdm(all_files_u)):

        #clear cfd screens upon every loop
        cfd_screens = np.ndarray((number_of_sections, channel_length, part_width, part_width))

        #load wind vels and m vals
        # gets the timestamp associated with the files in cell_list
        filenames = [a.split("/")[-1] for a in cell_list]
        timestamp = [(filename.split("_")[-1]).split(".")[0]
                     for filename in filenames]

        vels = [u_files_channel.mean_wind_speed(t, avg_screens) for t in timestamp]

        pixel_shift = [cell_vels[prop_start:prop_end + 1] /
                       (1000 * pixel_width) for cell_vels in vels]

        pixel_shift = np.reshape(pixel_shift, (tot_prop_screens, 2))

        #get raw cfd screens for time stamp
        for ii, time in enumerate(start_time + i):
            cfd_screens_channel.get_opd_screens(avg_screens, int(time), delz)
            cfd_screens[ii] = np.asarray(cfd_screens_channel.opd_part)

        #reshape cfd screens into single dimension and drop screens outside of propstart - propend
        cfd_screens = np.reshape(
            cfd_screens[:, prop_start:prop_end + 1], (tot_prop_screens, part_width, part_width))

    # Calculate hf components for new channel
    # perform interpolation

        seeds = rng.integers(1, 2**16, len(cn2))

        # append seeds to seed csv
        with open(seeds_file, 'a') as f:
            row_name = int(timestamp[0])
            writer = csv.writer(f)
            seeds_list = list(seeds)
            seeds_list.insert(0, row_name)
            writer.writerow(seeds_list)

        for ii, screen in enumerate(screens):
            screen.time_ev(cn2[ii], m_vals_full[ii],
                           pixel_shift[ii], l0, L0, delz, seeds[ii])
            screen.interpolate_cfd(cfd_screens[ii])

        beams = [prop.BeamProfile(target_res, screen_width, wavelength)
                 for ii in range(max_lg + 1)]

        time_str = str(int(timestamp[0])).zfill(5)

    # save selected hf screens
        if np.mod(i, hf_save_space) == 0:
            pass
            hf_filename = hf_dir + '/timestamp_' + time_str + '.h5' 
            with h5py.File(hf_filename, 'w') as f:
                for ii, screen in enumerate(screens):
                    f.create_dataset('screen_' + str(ii).zfill(4), data=screen.high_freq_phz)

        # refactor so that I am not using nested ifs to figure out which phase screen to use
        for ii in range(3):
            for free_space in free_space_between_cells:
                free_space_str = str(free_space).replace(", ", "_")
                for l_val, beam in enumerate(beams):

                    w0 = transmitApWidth/(2 * np.sqrt(l_val + 1))
                    beam.laguerre_gaussian_beam(l_val, lg_p, w0)

                    for sc_num, sc in enumerate(screens):

                        if np.mod(sc_num, screens_per_cell) == 0:
                            prop_elem = int(sc_num/screens_per_cell)
                            beam.free_space_prop(free_space[prop_elem])
                            beam.apply_sg_ap(sg_ap_width, sg_ap_power)
                        beam.free_space_prop(delz)
                        if ii == 0:
                            comp_screen = sc.high_freq_phz[0:512,
                                                           0:512] + sc.low_freq_phz
                        elif ii == 1:
                            comp_screen = sc.high_freq_phz[0:512, 0:512]
                        elif ii == 2:
                            comp_screen = sc.low_freq_phz
                        else:
                            sys.exit('Ended up in impossible zone.')

                        beam.apply_phase_screen(comp_screen)

                    beam.free_space_prop(free_space[-1])
                if ii == 0:
                    phz_string = 'comp_freq'
                elif ii == 1:
                    phz_string = 'high_freq'
                elif ii == 2:
                    phz_string = 'low_freq'
                else:
                    sys.exit('Ended up in impossible zone.')            

                full_res_dir = res_dir + '/' + phz_string
                if not os.path.exists(full_res_dir):
                    os.makedirs(full_res_dir)

                file_str = full_res_dir +'/timestamp_' + time_str +'.h5'
                with h5py.File(file_str, 'a') as f:
                    for l_val, beam in enumerate(beams):
                        f.create_dataset('beam_profile_l_' + str(l_val).zfill(2) + "/" + free_space_str, data = beam.field)        

if __name__ == "__main__":
    main()