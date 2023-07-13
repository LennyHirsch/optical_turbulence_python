import numpy as np
from matplotlib import pyplot as plt

# CALCULATING CONSTANTS
pi = 3.14159
cell_radius = 1 # in cm
cell_area = pi * cell_radius**2 # in cm^2

def total_particles(vol_maallox, vol_water):
    pi = 3.14159
    density_Mg = 2.42 # in g/cm^3
    density_Al = 2.34 # in g/cm^3
    particle_volume = 4/3 * pi * (0.0011/2)**3 # in cm^3

    mass_Mg = density_Mg * particle_volume # in g
    mass_Al = density_Al * particle_volume # in g

    mass_per_unit_volume = 200/5 # in mg/ml, so can be trivially converted to ug/ul
    parts_per_ul_maalox = (mass_per_unit_volume / (mass_Mg*(10**6))) + (mass_per_unit_volume / (mass_Al*(10**6)))

    parts = parts_per_ul_maalox * vol_maallox
    parts_per_ml = parts / vol_water

    return(parts, parts_per_ml)

# Params:
#   spot_size: diameter of AOI in cm
#   flow_rate: flow rate in cell in ml/s
#   maalox_vol: amount of maalox added in ul
#   water_vol: total amount of water in ml
# Returns:
#   tuple: (particles passing through AOI per second, time taken for particle to pass through sphere with diameter spot_size)
def particles_per_sec(spot_size, flow_rate, maalox_vol, water_vol):
    velocity = flow_rate / pi
    volume_thru_focal_area_per_sec = pi * ((spot_size/2)**2) * velocity
    parts_per_ml = (48246 * maalox_vol) / water_vol
    parts_per_sec = parts_per_ml * volume_thru_focal_area_per_sec
    time = 0.0011 + spot_size / velocity

    return (parts_per_sec, time)

spot_diameter = 0.002
volume_maalox = 30 # in ul
volume_water = 60 # in ml
flow_rate = 2 # in ml/s
cell_velocity = flow_rate / cell_area # velocity inside cell, in cm/s
dataset_time = 1.6 # seconds

(parts_per_sec, time) = particles_per_sec(spot_diameter, flow_rate, volume_maalox, volume_water)
print(f"Particles per second: {parts_per_sec}")
print(f"Seconds per particle: {1/parts_per_sec}s")
print(f"Time to pass: {time*10**3}ms")