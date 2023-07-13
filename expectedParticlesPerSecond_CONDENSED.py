import numpy as np
from matplotlib import pyplot as plt

# CALCULATING CONSTANTS
pi = 3.14159
cell_radius = 1 # in cm
cell_area = pi * cell_radius**2 # in cm^2

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