import os
import sys
import numpy as np
import glob

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts')

sys.path.append(
    '/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/pacakges')

import svd_hg_decomposition as svd_decomp
import lg_propagation as lg_prop
import svd_prop_funcs as svd_funcs
import propagation_functions as prop

sys.path.append('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data')
import params

if __name__ == '__main__':
    os.chdir('/Users/ultandaly/Library/CloudStorage/OneDrive-UniversityofGlasgow/Projects/python_simulation/svd_scripts/data/20230419/lg_prop_rep_many_inps')
    trgt_dirs = glob.glob('l*')

    waist_lst = [0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
    uncoll_s_lst = []
    coll_s_lst = []

    t_screens = [prop.PhaseScreen(
        params.screen_width, params.res, params.r0_tot *
        (params.num_of_steps - 1)**(3/5), params.l0, params.L0
    ) for i in range(params.num_of_steps - 1)]

    for wst in waist_lst:
        for trgt in trgt_dirs:
            t_screens_phz = np.load(trgt + '/turb_screens.npy')
            inp_beams = np.load(trgt + '/inp_beams.npy')

            for i in range(params.num_of_steps - 1):
                t_screens[i].phz = t_screens_phz
                t_screens[i].phz_lo = 0.0

            uncollimated = lg_prop.propagate_modes(inp_beams, t_screens, params.delz/   params.num_of_steps, 0)
            print('\nPropagated Beams')
#    collimated = lg_prop.propagate_modes(inp_beams, t_screens, params.delz/params.num_of_steps, 1)

            svd_vals = svd_funcs.svd_hg_waists(
            uncollimated, params.res, [wst], params.pascals_row, params.screen_width)

            uncoll_s = svd_vals[2]
            print(sum(uncoll_s))
            uncoll_s_lst.append(sum(uncoll_s))

    print(uncoll_s_lst)