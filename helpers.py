"""Contains some helpful functions too small for a separate file."""

import numpy as np
from astropy.cosmology import Planck15 as cosmo

def nonlin(x):
    """non-linear function for Eulerian-Lagrangian transformations."""
    return -1.35*(1+x)**(-2/3) + 0.78785*(1+x)**(-0.58661) - 1.12431*(1+x)**(-1/2) + 1.68647

def RtoM( R):
    CMperMPC = 3.086e24
    Msun = 1.989e33
    critical_density = cosmo.critical_density0.value*CMperMPC**3/Msun
    return (4.0/3.0)*np.pi*(R**3)*(cosmo.Om0*critical_density)

def MtoR (M):
    CMperMPC = 3.086e24
    Msun = 1.989e33
    critical_density = cosmo.critical_density0.value*CMperMPC**3/Msun
    return (3*M/(4*np.pi*cosmo.Om0*critical_density))**(1.0/3.0)

def V_cell(M_bias):
    R = MtoR(M_bias)
    return 4*np.pi/3 * R**3
