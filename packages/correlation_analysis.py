import numpy as np
from propagation_functions import BeamProfile
from scipy import ndimage, stats
from scipy.linalg import pinv

def img_list_autocovariance(lis, ref_point=0, mean_image = 0):
    if isinstance(mean_image, int):
        mean_image = np.mean(lis, 0)

    corr_lis = lis - mean_image
    corr_lis = np.array(corr_lis)
    
    elems = range(len(lis))
    auto_corr = [
        stats.pearsonr(corr_lis[i].flatten(),
                        corr_lis[ref_point].flatten())[0] for i in elems]

    return(auto_corr)

def center_of_mass(img):
    com = ndimage.measurements.center_of_mass(np.abs(img))
    return(com)


def center_of_mass_int(img):
    com = ndimage.measurements.center_of_mass(np.abs(img) ** 2.0)
    return(com)

def spec_calc(input_beam: np.ndarray, l_max: int, p_max: int, res: int, screen_width:  float, w0: float)-> np.ndarray:
    
    wavelength = 660 * 10 ** -9.0
    
    test_beam = BeamProfile(res, screen_width, wavelength)

    spec = np.ndarray(((2 * l_max) + 1))
    spec[:] = 0.0

    for l in range(-l_max, l_max + 1):
        for p in range(p_max + 1):
                test_beam.laguerre_gaussian_beam(l, p, w0)
                test_beam_phz = np.exp(1j * np.angle(test_beam.field))
                olap = np.conj(input_beam) * test_beam_phz
                spec[l+l_max] += np.abs(np.sum(olap))

    return(spec)


def vortex_splitting(input_beam: np.ndarray, prop_dist: float, res: int, screen_width: float, wavelength: float, l: int, p: int, w0: float):
    
     corrected_beam = diffraction_correction(input_beam, prop_dist, res, screen_width, wavelength, l, p, w0)

def diffraction_correction(input_beam: np.ndarray, prop_dist: float, res: int, screen_width: float, wavelength: float, l: int, p: int, w0: float):
    correction_beam = BeamProfile(res, screen_width, wavelength)
    correction_beam.laguerre_gaussian_beam(l, p, w0)
    correction_beam.free_space_prop(prop_dist)

    corrected_beam = input_beam * np.exp(-1j * np.angle(correction_beam.field))
    return(corrected_beam)

def gaussian_smoothing(input_beam: np.ndarray, gaussian_width: float):
    return(ndimage.gaussian_filter(input_beam, gaussian_width))

def zernike_generation(n:int, m:int, res:int, aperture)->np.ndarray:

    m_abs = np.abs(m)
    radial_term = np.zeros((res, res))
    k_max = ((n - m_abs) // 2) + 1
    
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)

    XX, YY = np.meshgrid(x, y)
    rho = np.sqrt(XX**2 + YY**2)
    theta = np.arctan2(XX, YY)


    for k in range(k_max ):
        numer = ((-1) ** k) * (np.math.factorial(n-k))

        denom_1 = np.math.factorial(k)
        denom_2 = np.math.factorial((n + m_abs)/2 - k)
        denom_3 = np.math.factorial((n - m_abs)/2 - k)

        radial_term += (numer * rho ** (n - (2*k)))/(denom_1 * denom_2 * denom_3)

    if m < 0:
        angle_term = np.sin(m_abs * theta)
    else:
        angle_term = np.cos(m_abs * theta)

    if aperture == 1:
        rho_mask = np.where(rho > 1, 0, 1)
        radial_term *= rho_mask

    return radial_term * angle_term

def zernike_decomp(wavefront:np.ndarray, max_n:int, aperture):
    '''this is not the most efficient method, but it does work
    uses the pseudoinverse to determine the zernike components that make up a wavefront. '''
    calc_weightings = []
    z_lst = []
    res = np.shape(wavefront)[0]

    for n in range(max_n):
        for m in range(-n, n+1, 2):
            zernike = zernike_generation(n, m, res, aperture)
            z_lst.append(zernike.flatten())
            calc_weightings.append([n, m])

    z_lst = np.asarray(z_lst).T
    weightings = pinv(z_lst) @ wavefront.flatten()

    for i, weighting in enumerate(weightings):
        calc_weightings[i].append(weighting)

    return calc_weightings