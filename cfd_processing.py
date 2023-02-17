#Consider the work flow, so I can know what the next steps are
# we already have mVals and r0, so this is not a concern
# - load the cfdChannel x
# - Calculate the hf components x
# - composite screens using interpolated components x
# - Perform propagation x
# - save resultant beam ~
# - load next cfd channel x
# - calculate time evolution of hf screens x
# I have all the parts. I just need to create a script that pulls it all together
# 

import glob
import numpy as np
import h5py
import propagation_functions as prop
from scipy import ndimage
import re


class cfdFiles():
    def __init__(self, filepath, extension):
        all_files = glob.glob(filepath + "/*" + extension)
        self.files = sorted(all_files)
    

class cfdChannel(cfdFiles):
    def __init__(self, filepath, extension, delta, wavelength, part_width):
        super().__init__(filepath, extension)
        self.delta = delta
        self.wavelength = wavelength
        self.part_width = part_width

        self.screen_width = part_width * delta

    def load_channel(self, filenum):
        channel_raw = h5py.File(self.files[filenum], 'r')

        dset_name = list(channel_raw.keys())[0]
        dset = channel_raw[dset_name]
        self.channel = dset

    def ri_calc(self, avg_screens):

        alpha = 7.76 * 10**-7
        pt_rel = self.channel[1]/self.channel[0]
        wvl_term = (7.52 * 10**-3) * ((self.wavelength * 10 ** 6)**-2)

        dims = pt_rel.shape
        self.new_screen_num = np.floor(dims[0]/avg_screens)

        pt_part = pt_rel[:(int(self.new_screen_num)*avg_screens)]

        # have to reshape the screens in this way to make sure that I take the correct dimension for slices
        pt_reshape = pt_part.reshape(int(
            self.new_screen_num), avg_screens, dims[-2], dims[-1])

        mean_pt = np.mean(pt_reshape, axis = 1)

        mean_ri = 1 + alpha * (1 +wvl_term) * mean_pt
        self.ri = mean_ri


    def get_opd_screens(self, avg_screens, filenum, del_z):

        self.load_channel(filenum)
        self.ri_calc(avg_screens)

        channel_width = np.shape(self.ri)[-1] - 1
        start_point = int((channel_width - self.part_width)/2)
        end_point = int(((channel_width + self.part_width)/2))
        
        self.ri = [self.ri[i] - np.mean(self.ri[i]) for i in range(int(self.new_screen_num))]

        ri_part = np.asarray(self.ri)[:, start_point:end_point, start_point:end_point]
        opd_screens = ri_part * 2 * np.pi * del_z / self.wavelength

        self.opd_part = opd_screens

    
class cfdChannelWind(cfdFiles):
    def __init__(self, filepath, extension, delta, part_width, channel_width):
        super().__init__(filepath, extension)
        self.delta = delta
        self.part_width = part_width
        self.channel_width = channel_width

        self.extension = extension

        self.screen_width = part_width * delta

    def wind_speed_partition(self, time_stamp):
        # I'm reloading the files that have already been provided
        
        pattern = re.compile(".*" + time_stamp + ".*")
        target_file = list(filter(pattern.match, self.files))
        
        if len(target_file) > 1:
            print(
                "Warning: multiple files match time stamp. Proceeding with first found file...")

        channel_raw = h5py.File(target_file[0], 'r')
        dset_name = list(channel_raw.keys())[0]
        u_vals = channel_raw[dset_name]

        start_pixel = int((self.channel_width - self.part_width) / 2.0)
        end_pixel = int(((self.channel_width + self.part_width) / 2.0))

        ux = u_vals[1, :, start_pixel:end_pixel, start_pixel:end_pixel]
        uy = u_vals[2, :, start_pixel:end_pixel, start_pixel:end_pixel]

        vels = [ux, uy]

        return(vels)

    def mean_wind_speed(self, time_stamp, avg_screens):
        vels = self.wind_speed_partition(time_stamp)

        max_screen_length = int(np.floor_divide(
            np.shape(vels)[1], avg_screens))

        res_u = np.zeros(
            (2, max_screen_length, self.part_width, self.part_width))

        ux_prop_part = [np.mean(vels[0][(ii * avg_screens): ((ii+1) * avg_screens)])
                        for ii in range(max_screen_length)]

        uy_prop_part = [np.mean(vels[1][(ii * avg_screens): ((ii+1) * avg_screens)])
                        for ii in range(max_screen_length)]


        # little bit messy, as I have to take a transpose. Not the end of the world, but I should try to refactor this to make it more readable
        res_u = np.transpose([ux_prop_part, uy_prop_part])

        return(res_u)

class cfdScreens():
    def __init__(self, target_res, dim_mult, filt_point, wavelength, part_width,screen_width):

        self.res = target_res
        self.dim_mult = dim_mult
        self.filt_point = filt_point
        self.wavelength = wavelength
        self.part_width = part_width
        self.screen_width = screen_width

        self.delta = screen_width/target_res

    def hf_filtered_screen(self, cn2Val, l0, L0, del_z, seed):

        delf = 1.0/self.screen_width

        r0 = prop.cn2_to_r0(cn2Val, self.wavelength, del_z)

        high_res = self.res * self.dim_mult
        large_screen_width = self.screen_width * self.dim_mult

        turbHigh = prop.PhaseScreen(large_screen_width, high_res, r0, l0, L0)
        psd = turbHigh.mvk()


        turbHigh.psd = self.high_pass_filt(psd, delf, high_res)

        high_freq_phz = turbHigh.mvk_screen(turbHigh.psd, seed)
        self.high_freq_phz = high_freq_phz

        # to remove this return f2 in time ev must be refactored
        return(high_freq_phz)

    def high_pass_filt(self, psd, delf, high_res):

        delk = 2.0 * np.pi * delf / self.dim_mult

        x = np.linspace(-high_res/2, high_res/2 - 1, high_res) * delk
        y = np.linspace(-high_res/2, high_res/2 - 1, high_res) * delk

        xx, yy = np.meshgrid(x, y)

        new_psd = psd * np.where(xx**2.0 + yy**2.0 <
                                 self.filt_point**2.0, 0.0, 1.0)

        #refactor hf_filtered screen to remove this return and instead change class attribute
        return(new_psd)

    def interpolate_cfd(self, phz):

        zoom_ratio = self.res/self.part_width
        upsample = ndimage.zoom(phz, zoom_ratio)
        self.low_freq_phz = upsample

    def time_ev(self, cn2Val, alpha, pixelshift, l0, L0, del_z, seed = None):

        f1 = self.high_freq_phz

        large_screen_width = self.screen_width * self.dim_mult
        delf = 1.0/large_screen_width

        high_res = self.res * self.dim_mult
    
        fx = np.linspace(-high_res/2, high_res/2 - 1, high_res)
        fy = np.linspace(-high_res/2, high_res/2 - 1, high_res)

        fxx, fyy = np.meshgrid(fx, fy)


        f2 = self.hf_filtered_screen(cn2Val, l0, L0, del_z, seed)

        f1_freq = prop.ft2(f1, self.delta)
        f2_freq = prop.ft2(f2, self.delta)
        denom = (1.0 + alpha**2.0)**(1.0/2.0)

        comb_screen = (f1_freq + alpha * f2_freq) / denom
        
        rot = (fxx * pixelshift[0] + fyy * pixelshift[1])/high_res

        comb_screen = comb_screen * np.exp(-1j * 2.0 * np.pi * rot)
        self.high_freq_phz = np.real(prop.ift2(comb_screen, delf))
        #return(self.high_freq_phz)
