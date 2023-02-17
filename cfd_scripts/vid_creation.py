import matplotlib.pyplot as plt
import subprocess
import numpy as np
import glob
import h5py
import os
import matplotlib as mpl

plt.rcParams['figure.figsize'] = (7, 5)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 20}

mpl.rc('font', **font)

screen_width = 0.25
delta = 1000* screen_width / 512

working_dir = "/Volumes/Newton/V4_results/20230210/"
files = glob.glob(working_dir + 'fixed_time_ev_4_secs/comp_freq/'+ '*.h5')

for i in range(250):

    with h5py.File(files[i], 'r') as f:
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(
        f[key], h5py.Dataset) else None)
        data = f[keys[1]][()]

    plt.imshow(np.abs(data[128:256 + 128, 128:256 + 128])
               ** 2.0, cmap='gist_gray',
    extent=[-127.5*delta, 127.5*delta, -127.5*delta, 127.5*delta], interpolation='none', vmin = 0, vmax = 10.94)

    if i == 0:
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label('Intensity')


        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')

    plt.savefig(working_dir + 'fixed_time_ev_4_secs/vid_figs/' +
                "/file%02d.png" % i, bbox_inches='tight')

os.chdir(working_dir + 'fixed_time_ev_4_secs/vid_figs/')
subprocess.call([
    'ffmpeg', '-framerate', '25', '-i', 'file%02d.png', '-r', '25', '-pix_fmt', 'yuv420p',
    'video_name.mp4'])

for file_name in glob.glob("*.png"):
    os.remove(file_name)