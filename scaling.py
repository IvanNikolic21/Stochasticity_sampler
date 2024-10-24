"""Temporary file that includes all the scaling laws."""


from astropy.cosmology import Planck18 as cosmo
import numpy as np
from astropy import units as u


def sfr_ms_mh_21cmmc(z, get_stellar_mass=True):
    """
        Get scaling relations for SFR-Mh or SFR-Mstar and Mstar-Mh from 21cmmc.
        Parameters
        ----------
        z: float,
            redshift at which scaling laws are evaluated. Important for
            SFR - M_stellar or SFR - M_halo relation.
        get_stellar_mass: boolean; optional,
            if True, two scaling laws are produced: SFR - M_stellar and
            M_stellar - M_halo. Else, only SFR - M_halo relation is returned.
    """
    a_SFR = 1.0 #the same number in both cases
    a_stellar_mass = 1.5
    b_stellar_mass = np.log10(0.0076) - 5
    b_SFR = -np.log10(0.43) + np.log10(cosmo.H(z).to(u.yr**(-1)).value)

    if get_stellar_mass:
        return a_SFR, b_SFR, a_stellar_mass, b_stellar_mass
    else:
        return a_SFR, b_stellar_mass + b_SFR


def ms_mh_flattening(mh):
    """
        Get scaling relations for SHMR based on Davies+in prep.
        Parameters
        ----------
        mh: float,
            halo mass at which we're evaluating the relation.
        Returns
        ----------
        ms_mean: floats; optional,
            a and b coefficient of the relation.
    """
    f_star_mean = 0.0076 * (2.6e11 / 1e10) ** 0.5
    f_star_mean /= (mh / 2.6e11) ** -0.5 + (mh / 2.6e11) ** 0.61
    return f_star_mean * mh


def sigma_SHMR_constant():
    """
        Get scatter of the SHMR relation. Constant as per discussion.
        Returns
        ----------
        sigma: float,
            constant scatter for the shmr relation.
    """
    return 0.25 #new FirstLight update
    #return 0.6 / np.log(10)

def sigma_SFR_constant():
    """
        Get scatter of the SFR-M_stellar relation. Constant!
    """
    return 0.5 / np.log(10)


def sigma_SFR_Hassan():
    """
        Get scatter from Hassan+21.
    Returns
    -------
        sigma_SRF: float,
            sigma from Hassan+21.
    """
    return 0.3

def sigma_SFR_variable(Mstar):
    """
        Variable scatter of SFR-Mstar relation.
        It's based on FirstLight database.
    Parameters
    ----------
    Mstar: stellar mass at which the relation is taken

    Returns
    -------
    sigma: sigma of the relation
    """
    a_sig_SFR = -0.11654893
    b_sig_SFR = 1.35289501
    sigma = a_sig_SFR * np.log10(Mstar) + b_sig_SFR

    if Mstar > 10**10:
        return 0.18740570999999995
    else:
        return sigma
def Brorby_lx(Z=None):
    """
        Get scaling law for Lx - SFR relation from Brorby+16.
        Parameters
        ----------
        Z: float,
            metalictiy.
        Returns
        ----------
        a_Lx: float,
            leadin factor in the relation.
        b_Lx: float,
            intercept of the relation.
        Notes
        ----------
        metalicities are given as 12 + log(O/H).
        Solar metalicity is given as 12 + log(O/H)_sun = 8.69.
    """
    if Z:
        a_Lx = 1.03
        b_Z = -0.64
        b_Lx = b_Z * np.log10(10**(Z-12) / 10**(8.69 - 12)) + 39.46
        sigma_Lx = 0.34
        return a_Lx, b_Lx
    else:
        a_Lx = 1.00
        b_Lx = 39.85
        sigma_Lx = 0.25
        return a_Lx, b_Lx

def Lx_SFR_Davies(Z = None):
    """
        Get scaling law for Lx - SFR relation, based on Lehmer+20.
        Parameters
        ----------
        Z: float,
            metalicity.
        Returns
        ----------
    """
    Z_r = Z / 0.05
    return 10 ** 40.5 / ((Z_r) ** 0.64 + 1)

def Lx_SFR(Z = None):
    """
        Get scaling law for Lx - SFR relation, based on Davies+in prep.
        Parameters
        ----------
        Z: float,
            metalicity.
        Returns
        ----------
    """
    a_Lx = -0.106 * Z + 2.304
    b_Lx = -0.312 * Z + 41.775
    return a_Lx, b_Lx

def sigma_Lx_const():
    """
        Get scatter of the Lx_SFR relation, based on Lehmer+20.
        Returns
        ----------
        sigma: float,
            scatter of the relation.
    """
    return 0.511

def Zahid_metal(Mstell, z):
    """
        Get metalicity from Zahid+14.
        Parameters
        ----------
        Mstell: float,
            stellar mass.
        z: float,
            redshift.
        
        Notes
        ----------
        metalicities are given as 12 + log(O/H).
    """

    Z0 = 9.1
    bZ = 9.135 + 2.64 * np.log10(1+z)
    gammaz = 0.522
    M0 = 10**bZ
    Z = Z0 + np.log10(1-np.exp(-(Mstell/M0)**gammaz))
    
    #delta_b0 =  - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * np.log(10) * M0 * 0.003
    #delta_Z0 = 0.002
    #delta_gammaz = - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * 0.009
    #delta_c0 = - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * np.log(10) * M0 * np.log10(1+z) * 0.05
    
    #delta_tot = np.sqrt(delta_b0 ** 2 + delta_Z0 ** 2 + delta_gammaz ** 2 + delta_c0 ** 2)
    
    return Z#, delta_tot

def metalicity_from_FMR ( M_star, SFR):
    """
    metalicity from Curti+19
    
    -----
    
    Function takes in the stellar mass and SFR and outputs the metallicity 12+log(O/H)
    """
    Z_0 = 8.779
    gamma = 0.31
    beta = 2.1
    m_0 = 10.11
    m_1 = 0.56
    M_0 = 10**(m_0 + m_1 * np.log10(SFR))
    return Z_0 - gamma/beta * np.log10(1+(M_star/M_0)**(-beta))

def sigma_metalicity_const():
    """
    Get the scatter of the FMR. Based on Curti+19.
    Returns
    ----------
    sigma: float,
        returns the scatter of the relation.
    """
    #return 0.054
    return 0.1 #motivated by the fact that galaxies show a larger scatter at higher redshifts

def OH_to_mass_fraction(Z_OH):
    """
    Convert 12+log(O/H) metalicty to mass fraction one.
    Very important note! So far I haven't accounted for solar metallicity being
    0.02!
    """
    return 10**(Z_OH - 8.69) * 0.02

def DeltaZ_z(z):
    """
        Evolution of the normalization of FMR. Based on Curti+23.
    Parameters
    ----------
    z: redshift

    Returns
    -------
    Delta Z: offset from FMR
    """
    a_d = -0.0553952
    b_d = 0.0635493
    return a_d * z + b_d
