"""Here are some functions common for all samplers."""

import warnings
from helpers import RtoM, nonlin
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
    >>> ngtm =     >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
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

    if not nbins:
        nbins = 1
    
    N_actual = np.zeros(nbins)
    m_haloes = []
    counter=0
    print(nbins, "This is nbins")
    max_iter = 10000
    for iter_num in range(max_iter):
        mbin = 10 ** np.linspace(Mmin, Mmax, nbins + 1)
        if nbins > 1:
            N_mean_list = np.zeros(nbins)
        if nbins==1:
            inds  = [b for b,m in enumerate(mx) if m > 10**Mmin and m<10**Mmax]
            
            if len(inds)>=4:
                    N_cs = hmf_integral_gtm(mx[inds], mf[inds]) * Vb
                    N_mean_list = N_cs[0]
                    N_cs = N_cs / N_cs[0]
            else:
                raise ValueError("You have 1 bin but less than 4 hmf elements inside it. Select a more detailed hmf or increase you limits!")
            
            if sample_hmf:
                N_actual[counter] = np.random.poission(N_mean_list)
  
            else:
                N_actual[counter] = round(N_mean_list)
            random_number_this_mass_bin = np.random.uniform(size = int(N_actual[counter]))
            for index, rn in enumerate(random_number_this_mass_bin):
                m_haloes.append(np.interp(rn, np.flip(N_cs), np.flip(mx[inds])))
        else:
            for k in range(nbins):
                if mbin[k] < mx[-1]:
                    inds = [b for b,m in enumerate(mx) if m > mbin[k] and m < mbin[k+1]]
                    
  #                  try:
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
        if np.sum(m_haloes) >= mass_coll:
            break
        print("This is the sum of haloes", np.sum(m_haloes), "and this is where I want to be", mass_coll)
        counter+=nbins
        if nbins>1:
            nbins = int(nbins/2)
        #if nbins == 1 and N_actual[-1]<1:
           # print("looks like there is no problem here, no haloes")
        #    break
            #print(nbins, mass_coll, np.sum(m_haloes))
        N_actual = np.concatenate((N_actual, np.zeros(nbins)))
    if iter_num == max_iter - 1:
        warnings.warn("In {} iterations couldn't find all haloes. Moving on, but beware!".format(max_iter))
    #while np.sum(m_haloes) <= mass_coll:
#
 #       mbin = 10 ** np.linspace(Mmin, Mmax, nbins + 1)
  #      N_mean_list = np.zeros(nbins)
#
 #       for k in range(nbins):
  #          if mbin[k] < mx[-1]:
   #             inds = [b for b,m in enumerate(mx) if m > mbin[k] and m < mbin[k+1]]
    #            print(inds)
  #  #            try:
      #          if len(inds)>=4:
       #             N_cs = hmf_integral_gtm(mx[inds], mf[inds]) * Vb
        #            N_mean_list[k] = N_cs[0]
         #           N_cs = N_cs / N_cs[0]
          #      #except:
           #     else:
            #        continue
             #   if sample_hmf:
              #      N_actual[counter+k] = np.random.poisson(N_mean_list[k])
               # else:
#                    N_actual[counter+k] = round(N_mean_list[k])
 #           
  #              random_number_this_mass_bin = np.random.uniform(size = int(N_actual[counter+k]))
   #             for index, rn in enumerate(random_number_this_mass_bin):
    #                m_haloes.append(np.interp(rn, np.flip(N_cs), np.flip(mx[inds])))
     #   counter+=nbins
      #  if nbins>1:
       #     nbins = int(nbins/2)
        #if nbins == 1 and N_actual[-1]<1:
         #  # print("looks like there is no problem here, no haloes")
          #  break
           # #print(nbins, mass_coll, np.sum(m_haloes))
#        N_actual = np.concatenate((N_actual, np.zeros(nbins)))
    m_haloes = np.sort(np.array(m_haloes))
    index_of_sum = 1
    for index, mass in  enumerate(m_haloes):
        if np.sum(m_haloes[index:]) <= mass_coll:
            break
    N_this_iter = len(m_haloes[index:])
    return N_this_iter, m_haloes[index:]

