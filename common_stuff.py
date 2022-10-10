"""Here are some functions common for all samplers."""

from helpers import RtoM, nonlin, metalicity_from_FMR
from astropy.cosmology import Planck15 as cosmo
import hmf
import numpy as np
from hmf import integrate_hmf as ig_hmf
from scipy import integrate
from astropy import units as u

from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import scipy.integrate as intg

def hmf_integral_gtm(M, dndm, mass_density=False):
    """
    Cumulatively integrate dn/dm.
    Parameters
    ----------
    M : array_like
        Array of masses.
    dndm : array_like
        Array of dn/dm (corresponding to M)
    mass_density : bool, `False`
        Whether to calculate mass density (or number density).
    Returns
    -------
    ngtm : array_like
        Cumulative integral of dndm.
    Examples
    --------
    Using a simple power-law mass function:
    >>> import numpy as np
    >>> m = np.logspace(10,18,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    The function always integrates to m=1e18, and extrapolates with a spline
    if data not provided:
    >>> m = np.logspace(10,12,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    """
    # Eliminate NaN's
    m = M[np.logical_not(np.isnan(dndm))]
    dndm = dndm[np.logical_not(np.isnan(dndm))]
    dndlnm = m * dndm

    if len(m) < 4:
        raise NaNException(
            "There are too few real numbers in dndm: len(dndm) = %s, #NaN's = %s"
            % (len(M), len(M) - len(dndm))
        )

    # Calculate the mass function (and its integral) from the highest M up to 10**18
    if m[-1] < m[0] * 10**18 / m[3]:
        m_upper = np.arange(
            np.log(m[-1]), np.log(10**18), np.log(m[1]) - np.log(m[0])
        )
        mf_func = _spline(np.log(m), np.log(dndlnm), k=1)
        mf = mf_func(m_upper)

        if not mass_density:
            int_upper = intg.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even="first")
        else:
            int_upper = intg.simps(
                np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even="first"
            )
    else:
        int_upper = 0

    # Calculate the cumulative integral (backwards) of [m*]dndlnm
    if not mass_density:
        ngtm = np.concatenate(
            (
                intg.cumtrapz(dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1],
                np.zeros(1),
            )
        )
    else:
        ngtm = np.concatenate(
            (
                intg.cumtrapz(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[
                    ::-1
                ],
                np.zeros(1),
            )
        )

    return ngtm# + int_upper


def _sample_densities(z, N, Mmin, Mmax, dlog10m, R_bias):
    """
        sample overdensities a number of times.
    """

    l = -0.999 #left most Eulerian overdensity

    conditional_mass = RtoM(R_bias)
    
    hmfi = hmf.MassFunction(
                                z = z, 
                                Mmin = Mmin, 
                                Mmax = Mmax, 
                                dlog10m = dlog10m, 
                                conditional_mass = conditional_mass)
    
    gauss_lagr = np.zeros(shape = 100000)
    delta_lagr = np.linspace(l/hmfi.growth_factor, 15, 100000)
    Ri = R_bias * (1.0 + delta_lagr * hmfi.growth_factor)**(1/3.0)    

    delta_shift = nonlin(delta_lagr * hmfi.growth_factor)/hmfi.growth_factor
    sigma_nn = hmfi.sigma[~np.isnan(hmfi.sigma)]
    radii_nn = hmfi.radii[~np.isnan(hmfi.sigma)] / cosmo.h
    gf_inv = 1 / hmfi.growth_factor
    pref = 1/np.sqrt(2*np.pi)
    for i,d in enumerate(delta_shift):
        s = np.interp(Ri[i], radii_nn, sigma_nn) * gf_inv
        gauss_lagr[i] = pref / s * np.exp( - 0.5 * d**2 /s**2 )
    
    N_mean_cumsum = integrate.cumtrapz(gauss_lagr, delta_shift)
    cumsum = N_mean_cumsum / N_mean_cumsum[-1]
    random_numb = np.random.uniform(size = N)
    delta_list = np.zeros(shape = N)
    
    for index, random in enumerate(random_numb):
        delta_list[index] = np.interp(random, cumsum, delta_lagr[:-1])
        delta_list[index] = nonlin(delta_list[index] * hmfi.growth_factor)
    
    return delta_list

def _sample_halos(Mmin, Mmax, nbins, mx, mf, mass_coll,Vb, sample_hmf = True):
    """
        sample halo masses from the given hmf.
    """
    N_actual = np.zeros(nbins)
    m_haloes = []
    counter=0
    while np.sum(m_haloes) <= mass_coll:
        mbin = 10 ** np.linspace(Mmin, Mmax, nbins + 1)
        N_mean_list = np.zeros(nbins)
        for k in range(nbins):
            if mbin[k] < mx[-1]:
                inds = [b for b,m in enumerate(mx) if m > mbin[k] and m < mbin[k+1]]
  #              try:
                if len(inds)>=4:
                    N_cs = hmf_integral_gtm(mx[inds], mf[inds]) * Vb
                    N_mean_list[k] = N_cs[0]
                    N_cs = N_cs / N_cs[0]
                #except:
                else:
                    continue
                if sample_hmf:
                    N_actual[counter+k] = np.random.poisson(N_mean_list[k])
                else:
                    N_actual[counter+k] = round(N_mean_list[k])
            
                random_number_this_mass_bin = np.random.uniform(size = int(N_actual[counter+k]))
                for index, rn in enumerate(random_number_this_mass_bin):
                    m_haloes.append(np.interp(rn, np.flip(N_cs), np.flip(mx[inds])))
        counter+=nbins
        if nbins>1:
            nbins = int(nbins/2)
        if nbins == 1 and N_actual[-1]<1:
           # print("looks like there is no problem here, no haloes")
            break
            #print(nbins, mass_coll, np.sum(m_haloes))
        N_actual = np.concatenate((N_actual, np.zeros(nbins)))
    m_haloes = np.sort(np.array(m_haloes))
    while np.sum(m_haloes)>=mass_coll:
        m_haloes = np.delete(m_haloes, 0)
    N_this_iter = len(m_haloes)
    return N_this_iter, m_haloes

def sfr_ms_mh_21cmmc(Mh, z,get_stellar_mass = True):
    """
        Get scaling relations for SFR-Mh or SFR-Mstar and Mstar-Mh from 21cmmc.
    """
    a_SFR = 1.0 #the same number in both cases
    a_stellar_mass = 1.5
    b_stellar_mass = np.log10(0.0076) - 5
    b_SFR = -np.log10(0.43) + np.log10(cosmo.H(z).to(u.yr**(-1)).value)

    if get_stellar_mass:
        return a_SFR, b_SFR, a_stellar_mass, b_stellar_mass
    else:
        return a_SFR, b_stellar_mass + b_SFR
       
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
        return a_Lx, b_Lx, sigma_Lx
    else:
        a_Lx = 1.00
        b_Lx = 39.85
        sigma_Lx = 0.25
        return a_Lx, b_Lx, sigma_Lx

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
    M0 = 10**bz
    Z = Z0 + np.log10(1-np.exp(-(Mstell/M0)**gammaz))
    
    delta_b0 =  - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * np.log(10) * M0 * 0.003
    delta_Z0 = 0.002
    delta_gammaz = - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * 0.009
    delta_c0 = - gammaz / (np.exp(+(Mstell/M0)**gammaz)-1) * (Mstell / M0)**(gammaz-1) * np.log(10) * M0 * np.log10(1+z) * 0.05
    
    delta_tot = np.sqrt(delta_b0 ** 2 + delta_Z0 ** 2 + delta_gammaz ** 2 + delta_c0 ** 2)
    
    return Z, delta_tot
