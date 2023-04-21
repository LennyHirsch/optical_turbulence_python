import numpy as np
from propagation_functions import BeamProfile
from scipy import ndimage, stats

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