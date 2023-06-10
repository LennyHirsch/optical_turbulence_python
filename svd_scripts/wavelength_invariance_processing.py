import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import glob
from PIL import Image
import cmasher as cmr

def create_plot(crss, save_dir, i_wvl):

    #cmap = cmr.rainforest
    #cmap = plt.get_cmap('cmr.rainforest')
    #for i in range(15):
    #    crss[i, i] = np.min(crss)
    cmap = 'viridis'
    
    #transpose so that basis vectors are on the y-axis
    plt.imshow(crss.T, cmap=cmap)
    plt.colorbar()
    plt.title(f'Wavelength: {str(i_wvl).zfill(4)}')

    plt.xlabel('Input Modes')
    plt.ylabel('Basis Vectors')

    fname = f'crss_plt_{str(i_wvl).zfill(4)}.png'

    plt.savefig(save_dir + fname)
    plt.close()
    print('\nSaved Figure....')

def create_gif(path):
    all_files = sorted(glob.glob(path +'*.png'))
    ims = []
    for f in all_files:
        ims.append(Image.open(f))
    ims[0].save(path+"../out.gif", save_all=True, append_images=ims[1:], duration = 100, loop = 0)

def main(working_dir, save_dir = 'tmp_images/'):
    wvl_dirs = glob.glob(working_dir + '/wavelength*')
    print(np.shape(wvl_dirs))
    img_dir = 'crosstalk_plots_images/'
    if not os.path.exists(save_dir + img_dir):
        os.makedirs(save_dir + img_dir)
    for i, wvl_dir in enumerate(sorted(wvl_dirs)):
        print(wvl_dir)
        crss = np.load(wvl_dir + '/crosstalk.npy')
        
        #normalise columns, I have confrimed this is the columns, not rows
        #for ii in range(len(crss)):
        #    crss[:, ii] /= np.max(crss[:, ii])

        crss = 10*np.log10(crss)
        create_plot(crss, save_dir + img_dir,wvl_dir[-4:] )

    create_gif(save_dir + img_dir)
    #convert to dbs
    #plot gif of all crosstalk matrices
    #repeat for both realisations


if __name__ == '__main__':

    os.chdir('../')
    print(os.getcwd())
    save_dir = 'svd_scripts/data/20230601/wavelength_invariance/realisation015/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    working_dir = '/Users/ultandaly/Desktop/wavelength_invariance_test_results/realisation015/'
    main(working_dir, save_dir)

    save_dir = 'svd_scripts/data/20230601/wavelength_invariance/realisation013/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    working_dir = '/Users/ultandaly/Desktop/wavelength_invariance_test_results/realisation013/'
    main(working_dir, save_dir)