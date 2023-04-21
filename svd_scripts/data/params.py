#system parameters
delz = 1000
inp_ap_width = 0.15
rec_ap_width = 0.15

#turbulence parameters
r0_tot = 0.02
L0 = 100.0
l0 = 0.001

#input basis parameters
waist = 0.02

#peforming lg propagation
l_pos_min = 5
p_max = 5
mode_num = (p_max+1) * (l_pos_min*2 + 1)

# simulation parameters
screen_width = 0.4
num_of_steps = 20 + 1
wavelength = 1550e-9
res = 512

#svd parameters
# how many modes I want to actually get from the SVD
trans_modes_num = 15

#how many SVD modes I want to use to reconstruct res_beams
err_mode_num = 15
svd_wavelengths = [1550e-9]
plt_mode = ((p_max) * (l_pos_min + 1))

#pascals_row = 5 -> first 15 hg modes
pascals_row = 5

#waist_lst = np.linspace(0.015, 0.035, 11)
waist_lst = [0.025]