#Add in correction for a lens
import numpy as np
from scipy.special import genlaguerre, eval_hermite

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

def r0_gaussian_to_cn2(r0:float, wavelength: float, delz:float, w0: float) -> float:
    """
    Calculates the refractive index structure parameter Cn^2 from the atmospheric coherence length (r0),the wavelength of light, the distance of propagation (delz), and the beam waist size (w0) assuming a Gaussian beam profile.

    Parameters:
    -----------
    r0 : float
        Atmospheric coherence length (in meters).
    wavelength : float
        Wavelength of the light (in meters).
    delz : float
        Distance of propagation (in meters).
    w0 : float
        Beam waist size (in meters).

    Returns:
    --------
    cn2 : float
        Refractive index structure parameter Cn^2.

    References:
    -----------
    [1] Schmidt, "Numerical Simulation of Optical Wave Propagation", SPIE, 2010

    [2] Andrews, "Field Guide to Atmospheric Optics", SPIE, 2007    
    """
    k = 2*np.pi / wavelength
    Gamma_0 = 2 * delz /(k * w0 ** 2.0)
    Theta = 1/(1+Gamma_0 ** 2.0)
    Gamma = Gamma_0/(1 + Gamma_0**2.0)
    a = (1-np.power(Theta, 8.0/3.0))/(1 - Theta)
    rho = r0 / 2.1

    denom = np.power(0.55 * k**2.0 * delz * 
                     (a + (0.62 * np.power(Gamma, 11.0/6.0))), -3.0/5.0)
    
    cn2 = np.power(rho/denom, -5.0/3.0)
    return(cn2)


#Alek functions
def cart2pol(x, y):
    
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def symmetric_rotated_meshgrid(res_x, res_y, pixel_size, rotation_deg, polar = False):

    X = (np.arange(0, res_x) - (res_x / 2 - 0.5)) * pixel_size
    Y = (np.arange(0, res_y) - (res_y / 2 - 0.5)) * pixel_size
    
    X, Y = np.meshgrid(X, Y)
    
    radius, angle = cart2pol(X, Y)
    
    if polar:
        return radius, angle + (rotation_deg * np.pi / 180.0)
    else:
        return pol2cart(radius, angle + (rotation_deg * np.pi / 180.0))
    

def coll_calc(beams, res, screen_width, wvl, waist, delz):
    """
    Intended to perform collimation in the situation where the input beams are arrays, rather than members of the BeamProfile class. Note that collimation is performed w.r.t. the beam waist at the focal point.

    Parameters:
    - beams: a list of the beam fields that need to be collimated
    - res: the screen resolution
    - screen_width: the screen width
    - wvl: the wavelength of the propagated beams
    - waist: the beam waist of the input beams
    - delz: the total propagation distance from the beam waist
    """

    l = 0
    p = 0

    beam_class = BeamProfile(res, screen_width, wvl)
    beam_class.laguerre_gaussian_beam(l, p, waist, delz)

    new_beams = []

    for beam in beams:
        beam_class.field = beam
        beam_class.collimation(delz)
        new_beams.append(beam_class.field)

    new_beams = np.asarray(new_beams)
    return new_beams

def rytov_var(r0: float, wavelength: float, delz: float) -> float:
    wvl_term = np.power(2.0 * np.pi / wavelength, -5.0/6.0)
    z_term = 2.9078 * np.power(delz, 5.0/6.0)
    r0_term = np.power(r0, -5.0/3.0)
    return(r0_term * z_term * wvl_term)

def fresnel_calc(d1: float, d2: float, wavelength: float, delz: float) -> float:
    numerator = np.pi * d1 * d2
    denominator = 4 * wavelength * delz
    fres = (numerator/denominator) ** 2.0

    return(fres)

class BeamProfile:

    def __init__(self, res: int, screen_width: float, wavelength: float) -> np.ndarray:

        self.res = res
        self.screen_width = screen_width
        self.wavelength = wavelength
        self.field = None

    def hermite_gaussian_beam(self, n: int, m: int, beam_waist: float, z: float = 0.0) -> np.ndarray:
        #x = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)
        #y = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)

        #xx, yy = np.meshgrid(x, y)

        #Aleks has requested 45 degree rotated HG decomposition. This code is kept in temporarily to accomdate
        
        xx, yy = symmetric_rotated_meshgrid(self.res, self.res, self.screen_width/self.res, 45)

        self.beam_waist = beam_waist
        zr = (np.pi * beam_waist**2.0) /self.wavelength
        k = 2 * np.pi / self.wavelength

        if np.abs(z) >= 1e-9:
            wz = beam_waist * np.sqrt(1 + (z/zr) ** 2.0)
            rz = z * (1 + (zr/z)**2.0)
            gouyz = (n + m + 1) * np.arctan(z/zr)

            gouy_term = np.exp(1j * gouyz)
            curve_term = np.exp(-1j * k * ((xx**2.0) + (yy**2.0))/(2*rz))
        else:
            wz = beam_waist
            curve_term = 1.0
            gouy_term = 1.0

        hn = eval_hermite(n, np.sqrt(2) * xx/wz)
        hm = eval_hermite(m, np.sqrt(2) * yy/wz)
        gauss_term = np.exp(-(xx**2.0 + yy ** 2.0)/wz ** 2.0)
        a = beam_waist/wz

        self.field = a * hn * hm * gauss_term * gouy_term * curve_term

    def laguerre_gaussian_beam(self, l: int, p: int, beam_waist: float, z: float = 0.0) -> np.ndarray:
        """"""

        x = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)
        y = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)

        xx, yy = np.meshgrid(x, y)

        self.beam_waist = beam_waist
        zr = (np.pi * beam_waist**2.0) /self.wavelength
        k = 2 * np.pi / self.wavelength

        if np.abs(z) >= 1e-9:
            wz = beam_waist * np.sqrt(1 + (z/zr) ** 2.0)
            rz = z * (1 + (zr/z)**2.0)
            gouyz = (np.abs(l) + (2 * p) + 1) * np.arctan(z/zr)

            gouy_term = np.exp(1j * gouyz)
            curve_term = np.exp(-1j * k * ((xx**2.0) + (yy**2.0))/(2*rz))
        else:
            wz = beam_waist
            curve_term = 1.0
            gouy_term = 1.0
        
        phase = np.exp(-1j * l * np.arctan2(yy, xx))

        gauss_profile = np.exp(-(xx ** 2 +
                               yy ** 2) / (wz ** 2))

        lg_profile = np.power(2 * (xx**2.0 + yy**2.0) / wz**2.0, np.abs(l/2))

        lg_poly_term = (2*(xx**2 + yy**2))/(wz**2)

        laguerre_poly = genlaguerre(p, np.abs(l))(lg_poly_term)

        norm_numer = 2 * np.math.factorial(p)
        norm_denom = np.pi * np.math.factorial(p + np.abs(l))

        norm = np.sqrt(norm_numer/norm_denom)/wz

        self.field = norm * laguerre_poly * lg_profile * gauss_profile * phase * gouy_term * curve_term

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

    def sq_ap(self, ap_width):

        ap = np.zeros((self.res, self.res))

        propor_width = int((ap_width/self.screen_width) * self.res)
     
        x = np.linspace(-self.res/2, self.res/2, self.res)
        y = np.linspace(-self.res/2, self.res/2, self.res)

        xx, yy = np.meshgrid(x, y)

        ap_x = np.where(np.abs(xx.astype(int)) <= propor_width/2, 1, 0)
        ap_y = np.where(np.abs(yy.astype(int)) <= propor_width/2, 1, 0)
        ap = ap_x * ap_y

        ap_bound_x = np.where(np.abs(xx.astype(int)) == propor_width/2, True, False)
        ap_bound_y = np.where(np.abs(yy.astype(int)) == propor_width/2, True, False)

        ap_bound =  (ap_bound_x | ap_bound_y) / 2
        ap_corner = np.logical_and(ap_bound_x, ap_bound_y) / 4
        
        ap = ap*(ap - ap_bound.astype(float) - ap_corner.astype(float))

        if self.field is None:
            self.field = ap
        else:
            self.field = self.field * ap

    def hard_ap(self, ap_width: float):
        '''Applies a hard boundaried circular aperture to the beam, centered on the central pixel. 
        
        Input: 
        ap_width : numeric
            the diameter of the circular aperture in units of metres'''
        x = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)
        y = np.linspace(-self.screen_width/2, self.screen_width/2, self.res)

        xx, yy = np.meshgrid(x, y)
        rr_sq = xx ** 2.0 + yy ** 2.0
        ap_rad = ap_width / 2.0
        circ = np.where(np.sqrt(rr_sq) <= ap_rad, 1, 0)
        self.field = self.field * circ
    
    def low_pass_filter(self, ap_pixels: int):

        x = np.linspace(-self.res/2, self.res/2, self.res)
        y = np.linspace(-self.res/2, self.res/2, self.res)

        xx, yy = np.meshgrid(x, y)
        rr_sq = xx ** 2.0 + yy ** 2.0
        ap_rad = ap_pixels / 2.0
        circ = np.where(np.sqrt(rr_sq) <= ap_rad, 1, 0)

        ft = ft2(self.field, 1) * circ
        
        self.field = ift2(ft, 1/self.res)

    def collimation(self, delz_total:float):
        correction_beam = BeamProfile(self.res, self.screen_width, self.wavelength)

        correction_beam.laguerre_gaussian_beam(0, 0, self.beam_waist, delz_total)

        # I currently have to conjugate here as the beam is propagating in the wrong direction in free_space_prop
        #IF THE PROPAGATION DIRECTION IS FIXED THIS ALSO NEEDS TO BE FIXED
        self.wavefront_correction(np.conj(correction_beam.field))
        del(correction_beam)


    def wavefront_correction(self, correction_beam:list):
        self.field *= np.exp(-1j * np.angle(correction_beam))

class PhaseScreen:
    def __init__(self, screen_width: float, res: int, r0 : float, l0: float, L0: float):
        self.screen_width = screen_width
        self.res = res
        self.l0 = l0
        self.L0 = L0
        self.r0 = r0

        #allows me to define a screen with a pre-defined phase screen
        self.phz = 0.0
        self.phz_lo = 0.0

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
    
    def r0_approx(self, sample_num: int = 0, phz_screen: np.ndarray = [0], max_dims: int = 0) -> np.ndarray:
        
        if sample_num == 0:
            print("Need to supply number of sample points to take from screen")
            return None
        
        #if phz_screen.all() == 0:
        #    phz_screen = self.phz + self.phz_lo

        if max_dims == 0:
            max_dims = [[0, self.res], [0, self.res]]

        pts = np.zeros((2, 2))
        max_dims = np.asarray(max_dims)

        ret_arr = []

        for i in range(sample_num):

            # lazy way, should use list comprehension
            pts[0, 0] = np.random.randint(max_dims[0, 0], max_dims[0, 1])
            pts[0, 1] = np.random.randint(max_dims[0, 0], max_dims[0, 1])

            pts[1, 0] = np.random.randint(max_dims[1, 0], max_dims[1, 1])
            pts[1, 1] = np.random.randint(max_dims[1, 0], max_dims[1, 1])

            dis = np.sqrt((pts[1, 1] - pts[0, 1])**2.0 + 
                          (pts[1, 0] - pts[0, 0])**2.0)
            
            #convert pixels distance to meters
            dis = dis * (self.screen_width/self.res)

            phz_diff = phz_screen[int(pts[1, 0]), int(pts[1, 1])] 
            - phz_screen[int(pts[0, 0]), int(pts[1, 0])]

            ret_arr.append([dis, phz_diff])
        
        return np.asarray(ret_arr)

            
        
