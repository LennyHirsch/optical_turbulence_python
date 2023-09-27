import propagation_functions as prop
import numpy as np
import matplotlib.pyplot as plt


res = 512 #i wouldn't recommend going too much higher than this for resolution
screen_width = 0.01 # the cross sectional width and height of the simulation
number_of_particles = 5000
wvl = 6.5e-9
mode = 5
pixel_size = screen_width / res # if we let screen_width be in mm, we can calculate pixel size. From this we can calculate how large the particulates should be.
particle_size = 11e-6 / pixel_size

# generate instance of beam class
beam = prop.BeamProfile(res, screen_width, wvl)

#generate the beam specifics, there is also a hg option, but with these options it is a simple collimated gaussian beam 
beam_waist = 1.5e-3
initial_z = 0
beam.laguerre_gaussian_beam(mode, 0, beam_waist, initial_z)

#propagate the beam some distance 
dis = 0.5
beam.free_space_prop(dis)

#THIS WILL NEED TO BE CHANGED TO ABSORBERS RATHER THAN A RI SCREEN BUT THE PRINCIPLE SHOULD BE THE SAME

absorbers = prop.AbsorberScreen(screen_width, res, particle_size, number_of_particles)
absorbers.generate_absorbers()
beam.apply_absorber_screen(absorbers.grid)

#generate the phase screen - don't worry too much
l0 = 0.001
L0 = 100.0
r0 = 0.01

phz = prop.PhaseScreen(screen_width, res, r0, l0, L0)
phz.mvk_screen()
phz.mvk_sh_screen()

# # #APPLY THE PHASE SCREEN TO BEAM 
beam.apply_phase_screen(phz.phz + phz.phz_lo)
beam.free_space_prop(dis)

#THE BEAM FIELD IS ACCESSED FROM THE VARIABLE BEAM.FIELD
#I've plot the phase here for a quick example
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(np.abs(beam.field), cmap = 'plasma')
ax1.imshow(np.angle(beam.field), cmap = 'hsv')
plt.show()