import numpy as np
from scipy.special import genlaguerre

'''if there are any issues considering changing powers to numpy.power()'''

def ft2(g, delta):
    """
    Find the DFT of a 2D array
    
    This function returns a scaled and amplitude corrected DFT of the input array using numpy.fft.fft2 as a starting point.
    
    Parameters
    --------
    g : array_like
        input array, can be complex
    delta : numeric
        scaling constant for the transformed array. This should be the pixel size, the DFT is scaled by a factor of delta**2.0

    Returns
    --------
    complex ndarray
    
    See Also
    --------
    ift2: 2D inverse DFT
    """
    G = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2.0
    return(G)

def ift2(G, delta_f):
    res = np.size(G, 1)
    g = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (res * delta_f)**2
    return(g)

def cn2_to_r0(cn2: float, wavelength: float, delz: float) -> float:
    """
    Convert a (log10) cn2 value to an r0 value
    
    Equation based on Field guide to atmospheric optics - Andrews
    All values are expected in units of meters. Can be scaled appropriately
    
    Parameters
    ----------
    cn2: numeric
        The log10 value of cn2
    
    wavelength: numeric
        The wavelength of the propagation beam
    
    delz: numeric
        The propagation distance
    
   returns
   -----------
   numeric
   """
    wvl_term = (2*np.pi / wavelength ) ** 2.0
    cn2_term = np.power(10.0, cn2)
    r0 = np.power(0.423 * wvl_term * cn2_term * delz, -3.0/5.0)

    return(r0)

def r0_to_cn2(r0: float, wavelength: float, delz: float) -> float:
    """
    Convert an r0 value to a (log10) cn2 value

    Equation is based on Field guide to atmospheric optics - Andrews
    All values are assumed to be in units of meters

    Parameters
    -----------
    r0: numeric
        The Fried parameter
    
    wavelength : numeric
        The wavelength of light propagating through the channel

    delz: numeric
        The propagation distance

    Returns
    -----------
    numeric
    """
    r0_term = np.power(r0, -5.0/3.0)
    denom = delz * 0.423 * (2.0 * np.pi / wavelength)**2.0
    cn2 = np.log10(r0_term/denom)

    return(cn2)

def rytov_var(r0: float, wavelength: float, delz: float) -> float:
    wvl_term = np.power(2.0 * np.pi / wavelength, -5.0/6.0)
    z_term = 2.9078 * np.power(delz, 5.0/6.0)
    r0_term = np.power(r0, -5.0/3.0)
    return(r0_term * z_term * wvl_term)

class BeamProfile:

    def __init__(self, res: int, screen_width: float, wavelength: float) -> np.ndarray:

        self.res = res
        self.screen_width = screen_width
        self.wavelength = wavelength

    def laguerre_gaussian_beam(self, l: int, p: int, beam_waist: float) -> np.ndarray:
        """"""

        x = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)
        y = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)

        xx, yy = np.meshgrid(x, y)

        self.beam_waist = beam_waist
        phase = np.exp(1j * l * np.arctan2(yy, xx))

        gauss_profile = np.exp(-(xx ** 2 +
                               yy ** 2) / (beam_waist ** 2))

        lg_profile = np.power(np.sqrt(xx**2.0 + yy**2.0) * np.sqrt(2) / beam_waist, np.abs(l))

        lg_poly_term = (2*(xx**2 + yy**2))/(beam_waist**2)

        laguerre_poly = genlaguerre(p, np.abs(l))(lg_poly_term)

        self.field = laguerre_poly * lg_profile * gauss_profile * phase
        #return(self.field)

    def free_space_prop(self, dis: float) -> None:
        """"""

        df = 1/self.screen_width
        delta = self.screen_width/self.res
        x = np.linspace(-self.res/2, self.res/2 - 1, self.res)
        y = np.linspace(-self.res/2, self.res/2 - 1, self.res)

        k = 2 * np.pi / self.wavelength

        fx, fy = np.meshgrid(x, y)

        fx = fx.astype('float') * df
        fy = fy.astype('float') * df

        prop_direction = np.sqrt(k**2 - (2 * np.pi * 1*fx)**2 - (2 * np.pi*1*fy)**2)

        angular_spec = np.exp(1j * dis * prop_direction)

        new_field = ift2(ft2(self.field, delta) * angular_spec, df)

        self.field = new_field

    def apply_phase_screen(self, phasescreen: np.ndarray) -> None:
        """"""
        self.field = self.field * np.exp(1j * phasescreen)

    def apply_sg_ap(self, ap_width, sg_power):

        x = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)
        y = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)

        xx, yy = np.meshgrid(x, y)
        rr_sq = xx ** 2.0 + yy ** 2.0
        ap_rad = ap_width / 2.0

        sg = np.exp(-2.0 * np.power(rr_sq/ap_rad ** 2.0, sg_power))
        self.field = self.field * sg
        
class PhaseScreen:
    def __init__(self, screen_width: float, res: int, r0 : float, l0: float, L0: float):
        self.screen_width = screen_width
        self.res = res
        self.l0 = l0
        self.L0 = L0
        self.r0 = r0

    def spr(self, n: int, seed: int =None):

        rng = np.random.default_rng(seed)
        #set optional random seed argument
        # need to move towards generators in numpy
        real_draws = rng.normal(0.0, 1.0, (n, n))
        imag_draws = rng.normal(0.0, 1.0, (n, n))
        gaussian_draw = real_draws + 1j * imag_draws
        return(gaussian_draw)

    def mvk(self):

        x = np.linspace(-self.res/2, self.res/2 - 1, self.res)
        y = np.linspace(-self.res/2, self.res/2 - 1, self.res)

        fx, fy = np.meshgrid(x/self.screen_width, y/self.screen_width)

        f_abs = np.sqrt(fx**2 + fy**2)
        r0_scale = 0.023*np.power(self.r0, -5.0/3.0)

        fm = 5.92/self.l0 / (2.0 * np. pi)
        vk_numer = np.exp(-(f_abs/fm)**2.0)

        vk_denom = np.power(f_abs**2 + (1/self.L0)**2, 11.0/6.0)

        psd = r0_scale * vk_numer/vk_denom

        psd[int(self.res/2) , int(self.res/2) ] = 0.0

        self.psd = psd
        return(psd)

    def mvk_sh_psd(self, p: int):

        delf = 1/((3.0**p) * self.screen_width)
        f_vals = np.linspace(-delf, delf, 3)
        fx, fy = np.meshgrid(f_vals, f_vals)
        f = np.sqrt(fx**2 + fy**2)

        fm = 5.92/self.l0/(2*np.pi)
        f0 = 1/self.L0
        r0_scale = 0.023*np.power(self.r0, -5.0/3.0)

        numer = np.exp(-np.power(f/fm, 2))
        denom = np.power(f**2 + f0**2, 11.0/6.0)

        psd_low = r0_scale * numer/denom


        psd_low[1, 1] = 0
        self.psd_low = psd_low
        
        return(psd_low)

    def turblowFN3(self, p: int, psd_low: np.ndarray, seed: int = None):

        sh = np.zeros((self.res, self.res))

        gaussian_draw = self.spr(3, seed)

        for i in range(3):
           for jj in range(3):

               delta = self.screen_width/self.res

               x = np.linspace(-self.res/2, self.res/2 - 1,
                               self.res) * delta
               y = np.linspace(-self.res/2, self.res/2 - 1,
                               self.res) * delta

               xx, yy = np.meshgrid(x, y)

               delf = 1/(3.0**p * self.screen_width)
               fx = np.array([-1.0, 0.0, 1.0]) * delf
               fx, fy = np.meshgrid(fx, fx)

               x_comp = fx[i, jj] * xx
               y_comp = fy[i, jj] * yy

               psd_comp = gaussian_draw[i, jj] * \
                   np.sqrt(psd_low[i, jj]) * delf

               sh = sh + psd_comp*np.exp((x_comp + y_comp) * 2 * np.pi * 1j)

        return(sh)

    def mvk_sh_screen(self):

        phz_lo = np.zeros((self.res, self.res))

        for p in range(1, 4):
            psd_sh = self.mvk_sh_psd(p)
            sh = self.turblowFN3(p, psd_sh)
            phz_lo = phz_lo + sh

        phz_lo = np.real(phz_lo) - np.mean(np.real(phz_lo))

        self.phz_lo = phz_lo
        return(phz_lo)

    def mvk_screen(self, psd: np.ndarray = 0, seed: int = None):

        delf = 1/(self.screen_width)

        if isinstance(psd, int):
            psd = self.mvk()
        
        cn = self.spr(self.res, seed) * np.sqrt(psd) * delf
        phz = np.real(ift2(cn, 1))

        self.phz = phz
        return(phz)