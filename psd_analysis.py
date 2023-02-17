import numpy as np
from skimage.filters import window
import propagation_functions as prop

#In progress:
#This class is supposed to mirror some processing performed in Mathematica, but I ahve run into some issues

# results between mathematica and python disagree slightly due to different window definitions. Analysis should be kept self consistent within the chosen platform
class psd():
    def __init__(self, screen, screen_width, wind_type = None):
        
        self.screen = screen
        self.res = np.shape(screen)[0]
        self.screen_width = screen_width
        self.delta = self.screen_width/self.res
        self.delta_f = 1/self.screen_width
        self.wind_type = wind_type

        self.detrend_2d()
        self.window()
        self.psd_calc()

    def opd_to_ri_psd(self, wvl, del_z):
        k = 2 * np.pi/wvl
        conv_denom = 2 * np.pi * k**2.0  * del_z
        self.psd = self.psd / conv_denom

    def ri_to_opd_psd(self, wvl, del_z):
        k = 2 * np.pi/wvl
        conv_numer = 2 * np.pi * k**2.0  * del_z
        self.psd = self.psd * conv_numer

    def radial_units(self):
        self.rho = np.sqrt(self.xx**2.0 + self.yy*2.0)
        self.phi = np.arctan2(self.yy, self.xx)


    def detrend_2d(self):
        """for the time being this function doesn't doo too much"""
        self.detrend_screen = self.screen

    def units_calc(self):
        k = 2 * np.pi * self.delta_f
        x = np.linspace(-self.res/2, self.res/2 - 1, self.res) * k
        y = np.linspace(-self.res/2, self.res/2 - 1, self.res) * k

        self.xx, self.yy = np.meshgrid(x, y)

    def psd_fit_processing(self):

        self.units_calc()
        self.radial_units()

        radial_psd_prelim = list(zip(self.rho, self.psd))
        radial_psd = np.transpose(radial_psd_prelim, (0, 2, 1))
        
        flat_psd = np.reshape(radial_psd, (self.res * self.res, 2))
        unique_psd = np.unique(flat_psd)

        self.sort_psd = np.sort(unique_psd, axis = 0)


    def window(self):
        """currently only supports windows that do not need additional parameters"""    
        if self.wind_type is not None:
            self.wind = window(self.wind_type, (self.res, self.res))
            self.wind_screen = self.detrend_screen * self.wind
        else:
            self.wind_screen = self.detrend_screen
            self.wind = np.ones((self.res, self.res))

    def psd_calc(self):
        fourier_screen = prop.ft2(self.wind_screen, self.delta)
        winInt = np.sum(self.wind ** 2.0) / (self.res ** 2.0)

        psd_norm = 1/(4* (np.pi ** 2.0) * winInt * self.screen_width ** 2.0)
  
        self.psd = psd_norm * np.abs(fourier_screen) ** 2.0

