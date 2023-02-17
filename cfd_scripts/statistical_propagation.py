import numpy as np
import propagation_functions as prop
import h5py
from datetime import datetime
import os
from tqdm import tqdm

def main():
    
    r0_path = "/Users/ultandaly/Desktop/PhD_Data/python_tst_vals/r0_vals.h5"
    parent_folder = "/Volumes/Newton/V4_results/"

    now = datetime.now()
    date_string = now.strftime("%Y%m%d")

    res_dir = parent_folder + date_string

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with h5py.File(r0_path, 'r') as r0_raw:
        dset_name = list(r0_raw.keys())[0]
        r0 = r0_raw[dset_name][()]

    stat_dir = res_dir  + '/statistical_propagation_L0_0.125_two_steps'
    
    transmitApWidth = 0.075
    lg_p = 0
    max_lg = 5

    part_width = 20
    pixel_width = 0.0125
    screen_width = pixel_width * part_width
    target_res = 512
    wavelength = 1550 * 10**-9.0

    number_of_sections = 1
    prop_start = 2
    prop_end = 41

    L0 = 0.125
    l0 = 0.0002

    realisations = 400
    tot_channel_length = 80
    section_dis = tot_channel_length/number_of_sections

    sg_ap_width = 0.94*screen_width
    sg_ap_power = 8

    r0_flat = np.ndarray.flatten(r0[0:number_of_sections, prop_start:prop_end + 1])

    r0_tot = np.power(np.sum(np.power(r0_flat, -5.0/3.0)), -3.0/5.0)
    r0_per_sec = np.power(number_of_sections, 3.0/5.0) * r0_tot

    for i in tqdm(range(realisations)):

        beams = [prop.BeamProfile(target_res, screen_width, wavelength) for ii in range(max_lg + 1)]

        screens = [prop.PhaseScreen(screen_width, target_res, r0_per_sec, l0, L0) for ii in range(number_of_sections)]

        for l_val, beam in enumerate(beams):
            w0 = transmitApWidth/(2 * np.sqrt(l_val + 1))
            beam.laguerre_gaussian_beam(l_val, lg_p, w0)

            for screen in screens:

                screen.mvk_sh_screen()
                screen.mvk_screen()

                beam.free_space_prop(section_dis/2)
                beam.apply_phase_screen(screen.phz + screen.phz_lo)

                beam.free_space_prop(section_dis/2)
                beam.apply_sg_ap(sg_ap_width, sg_ap_power)
            
            full_res_dir = stat_dir + '/' + 'L0_' + str(L0)

            if not os.path.exists(full_res_dir):
                os.makedirs(full_res_dir)
            
            file_str = full_res_dir + '/realisation_' + str(i).zfill(5) + '.h5'

            with h5py.File(file_str, 'a') as f:
                f.create_dataset('beam_profile_l_' + str(l_val).zfill(2), data = beam.field)  

if __name__ == "__main__":
    main()