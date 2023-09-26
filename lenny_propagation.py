import propagation_functions as prop
import numpy as np
import matplotlib.pyplot as plt


res = 2048 #i wouldn't recommend going too much higher than this for resolution
screen_width = 4 # the cross sectional width and height of the simulation
wvl = 0.65e-9 

# generate instance of beam class
beam = prop.BeamProfile(res, screen_width, wvl)

#generate the beam specifics, there is also a hg option, but with these options it is a simple collimated gaussian beam 
beam_waist = 0.5
initial_z = 0
beam.laguerre_gaussian_beam(2, 0, beam_waist, initial_z)

#propagate the beam some distance 
dis = 10000
beam.free_space_prop(dis)

#THIS WILL NEED TO BE CHANGED TO ABSORBERS RATHER THAN A RI SCREEN BUT THE PRINCIPLE SHOULD BE THE SAME

absorbers = prop.AbsorberScreen(screen_width, res, 11e-6, 1000)
absorbers.generate_absorbers()
grid = absorbers.grid
circles = absorbers.plot_circles(grid)
beam.apply_absorber_screen(circles)

#generate the phase screen - don't worry too much
# l0 = 0.001
# L0 = 100.0
# r0 = 0.1

# phz = prop.PhaseScreen(screen_width, res, r0, l0, L0)
# phz.mvk_screen()
# phz.mvk_sh_screen()

# # #APPLY THE PHASE SCREEN TO BEAM 
# beam.apply_phase_screen(phz.phz + phz.phz_lo)
beam.free_space_prop(dis)

#THE BEAM FIELD IS ACCESSED FROM THE VARIABLE BEAM.FIELD
#I've plot the phase here for a quick example
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(np.abs(beam.field))
ax1.imshow(np.angle(beam.field), cmap = 'hsv')
plt.show()