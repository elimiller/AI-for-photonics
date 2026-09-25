import numpy as np

def beam_rad(wl, w0, z, Msq=1.0):
    '''Calculates the beam radius of a Gaussian beam'''
    radius = w0 * np.sqrt(1 + Msq*(z * wl / (np.pi * w0**2))**2)
    return radius

def wavefront_roc(wl, w0, z, Msq=1.0):
    roc = z * (1 + Msq*(np.pi * w0**2 / (z * wl))**2)
    return roc

def rayleigh_range(wl, w0):
    rr = np.pi * w0**2 / wl
    return rr

def divergence(wl, w0):
    theta = wl/ (np.pi * w0)


def intensity_envelope(r, w):
    """
    Gaussian beam intensity envelope (radial profile).

    Parameters
    ----------
    r : array-like or float
        Radial position (m)
    w : float
        Beam waist / radius (m), defined at 1/e^2 intensity

    Returns
    -------
    I : array-like or float
        Intensity at each r (W/m^2)
    """
    env = np.exp(-2 * r**2 / w**2)
    return env

def peak_intensity(total_pwr, area):
    I = 2 * total_pwr / area
    return I


def w0_transf(wl, w0_input, f, z1):
    zR = np.pi * w0_input**2 / wl
    w0_prime = w0_input * f / np.sqrt((f - z1)**2 + (zR)**2)
    return w0_prime

def z_transf(wl, w0_input, f, z1):
    zR = np.pi * w0_input**2 / wl
    z2 = f + f**2 * (z1 - f) / ((z1 - f)**2 + (zR)**2)
    return z2