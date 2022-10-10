"""Contains the sampler function"""

from lx import LxvsSFR_lin
from chmf import chmf
from helpers import RtoM, nonlin, metalicity_from_FMR
from common_stuff import _sample_densities, _sample_halos, sfr_ms_mh_21cmmc, Brorby_lx, Zahid_metal
from sfr import SFRvsMh_lin

import random
import math
import time
import numpy as np
import hmf
from hmf import integrate_hmf as ig_hmf
from scipy import integrate
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u 
from numpy.random import default_rng as rng
from multiprocessing import Pool, cpu_count, Process, Manager
import os
    
def Sampler_x(emissivities_list,
            z=10,
            delta_bias = 0.0,
            R_bias=5.0,
            log10_Mmin = 5,
            log10_Mmax = 15,
            dlog10m = 0.01,
            N_iter = 1000,
            sample_hmf = True,
            sample_densities = True, 
            sample_SFR = 0,          #0 is for sampling using data from Ceverino+18, 1 is for Nikolic+22,
                                        #2 is not sampling but using mean from Ceverino+18, 3 is for mean from Nikolic+22
            sample_Lx = 0,           #0 is for sampling using data from Lehmer+20, 1 is from Brorby+16
                                        #2 is not sampling but using mean from Lehmer+20, 3 is for mean from Brorby+16
            calculate_2pcc = False,     #whether to calculate the 2point-correlation Poission correction

            duty_cycle = True,          #whether to turn of duty cycle in the sampler.
            logger = False,          #just printing stuff
            sample_Ms = True,        #also sampling metlaicity with this
            mass_binning = False,     #bin halo mass fucntion
           ):
    
    M_turn = 5*10**8  #Park+19 parametrization
    V_bias = 4.0 / 3.0  * np.pi * R_bias ** 3    
#    print("starting now")
    
    ##if not sample densities, no need to calculate hmf every time

    if sample_Ms:   #initiate SFR data
        a_SFR, b_SFR_ini, sSFR, a_Ms, b_Ms_ini, sMs = SFRvsMh_lin(z)
    elif sample_Ms:
        a_Ms, b_Ms_ini, sMs = SFRvsMh_lin(z,sample_Ms=False)
   
    if sample_densities:
        
        delta_list = _sample_densities(z, 
                                       N_iter, 
                                       log10_Mmin, 
                                       log10_Mmax, 
                                       dlog10m, 
                                       R_bias)
        
        hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
        hmf_this.prep_for_hmf_st(log10_Mmin, log10_Mmax, dlog10m )

        delta_nonlin = np.linspace(-0.99,10)
        delta_lin_values= nonlin(delta_nonlin)
    
    else:

        if delta_bias==0.0:
            hmf_this = hmf.MassFunction(z = z, 
                                        Mmin = log10_Mmin, 
                                        Mmax = log10_Mmax, 
                                        dlog10m = dlog10m)
            mass_func = hmf_this.dndm 
            masses = hmf_this.m
            N_mean_cs = ig_hmf.hmf_integral_gtm(masses, 
                                                mass_func, 
                                                mass_density=False) * V_bias
            cumulative_mass = ig_hmf.hmf_integral_gtm(masses, 
                                                      mass_func, 
                                                      mass_density=True)

        else:
            hmf_this = chmf(z=z, delta_bias = delta_bias, R_bias = R_bias)
            hmf_this.prep_for_hmf_st(log10_Mmin, log10_Mmax, dlog10m)
            masses, mass_func = hmf_this.run_hmf_st(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            N_mean_cs = ig_hmf.hmf_integral_gtm(masses[:index_to_stop], 
                                                mass_func[:index_to_stop])

        N_mean = int((N_mean_cs)[0])
        N_cs_normalized = N_mean_cs/N_mean_cs[0]

    emissivities = np.zeros(shape=N_iter)
    halo_masses_saved = []
    for i in range(N_iter):
        if sample_densities:
        
            delta_bias = delta_list[i]
            delta_bias_before = float(delta_bias)
            delta_bias = np.interp(delta_bias, delta_lin_values, delta_nonlin)
            delta_bias /= hmf_this.dicke()

            masses, mass_func = hmf_this.run_hmf_st(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            mass_coll = hmf_this.f_coll_st(10**log10_Mmin) * V_bias * hmf_this.critical_density

            N_mean_cs = ig_hmf.hmf_integral_gtm(masses, 
                                                mass_func) * V_bias
            N_mean = int((N_mean_cs)[0])
            N_cs_norm = N_mean_cs/N_mean_cs[0]

            if mass_binning:
                N_this_iter, mhs = _sample_halos(log10_Mmin, 
                                                 log10_Mmax, 
                                                 mass_binning,
                                                 masses, 
                                                 mass_func,
                                                 mass_coll,
                                                 V_bias, 
                                                 sample_hmf)
                if i==0:
                    halo_masses_saved.append(mhs)
        N_this_iter = int(N_this_iter)
        if not mass_binning:
            N_this_iter = N_mean
            masses_of_haloes = np.zeros(shape = N_this_iter)
                            
            for ind, rn in enumerate(range(N_this_iter)):
                rand = np.random.uniform()
                mhs[ind] = np.interp(rand, np.flip(N_cs_norm), np.flip(masses))

        masses_saved = []
        if duty_cycle:
            for index, mass in enumerate(mhs):
                if np.random.binomial(1, np.exp(-M_turn/mass)):
                    masses_saved.append(mass)
                    
        Lx = np.zeros(shape = int(N_this_iter))

        if sample_Ms:
            ms_samples = np.zeros(shape=N_this_iter)
            Z_samples = np.zeros(shape=N_this_iter)
        SFR_samples_this_iter = np.zeros(shape=N_this_iter)
        
        for j,mass in enumerate(masses_saved):
            logm = np.log10(mass)            
            a_Lx, b_Lx, sigma_Lx = LxvsSFR_lin()
            if sample_Ms:
                b_SFR = b_SFR_ini
                b_Ms = b_Ms_ini
            else:
                b_Ms = b_Ms_ini
######################SFR PART BEGINS###########################################

            if sample_SFR == 0 and sample_Ms:
#                print(b_Ms, "Ms now")
                if sample_Lx==0 or sample_Lx==1:
                    b_Ms -= a_Lx * a_SFR * np.log(10) * sMs**2 / 2
                else:
                    b_Ms -= a_SFR * np.log(10) * sMs**2/2

                Ms_sample = 10**(np.random.normal((a_Ms*logm + b_Ms), sMs))
                logmstar = np.log10(Ms_sample)

                if sample_Lx==0 or sample_Lx==1:
                    b_SFR -= a_Lx *np.log(10) * sSFR**2 / 2
                else:
                    b_SFR -= np.log(10) * sSFR**2 / 2
#                print(b_Ms, "and later")
                SFR_samp = 10**(np.random.normal((a_SFR*logmstar+b_SFR), sSFR))
                
            elif sample_SFR==1 and sample_Ms:
                a_SFR, b_SFR, a_stellar_mass, b_stellar_mass = sfr_ms_mh_21cmmc(mass, z)

                if sample_Lx==0 or sample_Lx==1:
                    b_stellar_mass -= a_Lx * a_SFR * np.log(10) * sMs**2 / 2
                else:
                    b_stellar_mass -= a_SFR * np.log(10) * sMs**2/2
                
                Ms_sample = 10**(np.random.normal((a_stellar_mass*logm + b_stellar_mass), sMs))
                logmstar = np.log10(Ms_sample)

                if sample_Lx==0 or sample_Lx==1:
                    b_SFR -= a_Lx *np.log(10) * sSFR**2 / 2
                else:
                    b_SFR -= np.log(10) * sSFR**2 / 2

                SFR_samp = 10**(np.random.normal(a_SFR * logmstar + b_SFR, sSFR))
                
                
            elif sample_SFR==2 and sample_Ms:
                
                if sample_Lx==0 or sample_Lx==1:
                    b_Ms -= a_Lx * a_SFR * np.log(10) * sMs**2 / 2
                else:
                    b_Ms -= a_SFR * np.log(10) * sMs**2/2

                Ms_sample = 10**(np.random.normal(a_Ms * logm + b_Ms, sMs))
                SFR_samp = 10**(a_SFR * np.log10(Ms_sample) + b_SFR)
                
            elif sample_SFR==3 and sample_Ms:
                a_SFR, b_SFR, a_st_m, b_st_m = sfr_ms_mh_21cmmc(mass, z)

                if sample_Lx==0 or sample_Lx==1:
                    b_st_m -= a_Lx * a_SFR * np.log(10) * sMs**2 / 2
                else:
                    b_st_m -= a_SFR * np.log(10) * sMs**2/2
                
                Ms_sample = 10**(np.random.normal(a_st_m * logm + b_st_m, sMs))
                SFR_samp = 10**(a_SFR * np.log10(Ms_sample) + b_SFR)
#                print(SFR_samp, Ms_sample, a_SFR, b_SFR, a_st_m, b_st_m, logm)
            elif sample_SFR == 0 and not sample_Ms:
                if sample_Lx==0 or sample_Lx==1:
                    b_Ms -= a_Lx * np.log(10) * sMs**2 / 2
                else:
                    b_Ms -= np.log(10) * sMs**2/2

                SFR_samp = 10**(np.random.normal((a_Ms * logm + b_Ms), sMs))

            elif sample_SFR == 2 and not sample_Ms:
                SFR_samp = 10**(a_Ms * logm + b_Ms)
            
            elif sample_SFR == 1 and not sample_Ms:
                a_SFR, b_SFR = sfr_ms_mh_21cmmc(mass, z, False)

                if sample_Lx==0 or sample_Lx==1:
                    b_SFR -= a_Lx * np.log(10) * sMs**2 / 2
                else:
                    b_SFR -= np.log(10) * sMs**2/2

                SFR_samp = 10**(np.random.normal((a_SFR * logm + b_SFR), sMs))
            elif sample_SFR == 3 and not sample_Ms:
                a_SFR, b_SFR = sfr_ms_mh_21cmmc(mass, z, False)
                SFR_samp = 10**(a_SFR * logm + b_SFR)
                
###########################SFR PART DONE########################################
##########################LX PART BEGINS########################################
            if sample_Lx==0 and sample_Ms:
#                print("This is the SFR", SFR_samp, "\n")                 
                Z_mean, sigma_Z = metalicity_from_FMR(Ms_sample, SFR_samp)
               
                ### Since I need a_Lx to estimate the shift of the relation,
                ### I'll first call a_Lx with current Z_mean and the reduce
                ### that Z_mean.
                
                a_Lx_try, _, _ = LxvsSFR_lin(Z_mean)
                Z_mean -= np.log(10) * a_Lx_try * sigma_Z**2 / 2

                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))
                logsfr = np.log10(SFR_samp)

                a_Lx, b_Lx, sigma_Lx = LxvsSFR_lin(Z_sample)

                b_Lx -= np.log(10) * sigma_Lx**2 / 2   #shift to median      
 
                Lx_sample = 10**np.random.normal(a_Lx * logsfr + b_Lx,sigma_Lx)
                
            elif sample_Lx==1 and sample_Ms:
                logsfr = np.log10(SFR_samp)
                Z_mean, sigma_Z = metalicity_from_FMR(Ms_sample, SFR_samp)

                ### Since I need a_Lx to estimate the shift of the relation,
                ### I'll first call a_Lx with current Z_mean and the reduce
                ### that Z_mean.
                a_Lx_try, _, _ = Brorby_lx(Z_mean)
                Z_mean -= np.log(10) * a_Lx_try * sigma_Z**2 / 2

                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))

                a_Lx, b_Lx, sigma_Lx = Brorby_lx(Z_sample)
                
                b_Lx -= np.log(10) * sigma_Lx**2 / 2 #shift to median

                Lx_sample = 10**np.random.normal((a_Lx*logsfr+b_Lx), sigma_Lx)
        #        print("Lx value", Lx_sample, "coeffs", a_Lx, b_Lx, "and sfr", logsfr)
                
            elif sample_Lx==2 and sample_Ms:
                Z_mean, sigma_Z =metalicity_from_FMR(Ms_sample, SFR_samp)
                Z_mean -= np.log(10) * sigma_Z**2 / 2
                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))
                a_Lx, b_Lx, sigma_Lx = LxvsSFR_lin(Z_sample)
                Lx_sample = 10**(a_Lx * np.log10(SFR_samp) + b_Lx)
                
            elif sample_Lx==3 and sample_Ms:
                Z_mean, sigma_Z = metalicity_from_FMR(Ms_sample, SFR_samp)
                Z_mean -= np.log(10) * sigma_Z**2 / 2
                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))

                a_Lx, b_Lx, sigma_Lx = Brorby_lx(Z_sample)
                Lx_sample = 10**(a_Lx*logsfr+b_SFR)

                
            elif sample_Lx == 0 and not sample_Ms:
                logsfr = np.log10(SFR_samp)
             
                b_Lx -= np.log(10) * sigma_Lx**2 / 2 #shift to median

                Lx_sample = 10**np.random.normal(a_Lx * logsfr + b_Lx,sigma_Lx)
                
            elif sample_Lx == 2 and not sample_Ms:
                Lx_sample = 10**(a_Lx * np.log10(SFR_samp) + b_Lx)

            elif sample_Lx == 1 and not sample_Ms:
                a_Lx, b_Lx, sigma_Lx = Brorby_lx()

                b_Lx -= np.log(10) * sigma_Lx**2 / 2
                Lx_sample = 10**np.random.normal((a_Lx*logsfr+b_SFR), sigma_Lx)
           
            elif sample_Lx == 3 and not sample_Ms:
                a_Lx, b_Lx, sigma_Lx = Brorby_lx()
                Lx_sample = 10**(a_Lx * logsfr + b_SFR)
###########################End Lx part###################################

            Lx[j] = Lx_sample
            SFR_samples_this_iter[j] = SFR_samp
            if sample_Ms:
                ms_samples[j]=Ms_sample
                Z_samples[j]=Z_sample
            
        if logger:
            end_sampling = time.time()
        #    print("sampling took", end_sampling - end_duty)
        emissivities[i] = np.sum (Lx)
        
        if np.round(i/N_iter*100000)/1000%10 == 0:
            print("done with {} % of the data".format(math.ceil(i/N_iter*100)))
    #return emissivities, sfr_samples,Lx_for_samples, delta_bias, halo_mass_total
    print("Checking if here is the problem,", emissivities[:10])
    np.save('/home/inikolic/halo_masses' + str(z) + '.npy', np.array(halo_masses_saved))
    emissivities_list.append(emissivities)
