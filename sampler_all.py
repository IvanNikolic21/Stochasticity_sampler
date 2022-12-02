""""Contains sampler of all three things at the same time"""
import os
import math
from hmf import integrate_hmf as ig_hmf
from sfr import SFRvsMh_lin
from scaling import sfr_ms_mh_21cmmc, sigma_SHMR_constant, sigma_SFR_constant
from scaling import metalicity_from_FMR, sigma_metalicity_const, Lx_SFR
from scaling import sigma_Lx_const, ms_mh_21cmmc, Brorby_lx
from chmf import chmf
from helpers import RtoM, nonlin
from common_stuff import _sample_halos, _sample_densities
from bpass_read import bpass_loader
import numpy as np
from fesc import fesc_distr
from numpy.random import normal
import time

def Sampler_ALL(emissivities_x_list,
                emissivities_lw_list,
                emissivities_uv_list,
                z=10,
                delta_bias = 0.0,
                R_bias=5.0,
                log10_Mmin = 5,
                log10_Mmax = 15,
                dlog10m = 0.01,
                N_iter = 1000,
                sample_hmf = True,
                sample_densities = True, 
                sample_SFR = True,
                sample_emiss = True,        
                calculate_2pcc = False,     #2pcc correction to # of halos.
                duty_cycle = True,          #whether to turn of duty cycle.
                sample_Ms = True,        #also sampling metlaicity with this
                mass_binning = False,     #bin halo mass fucntion
                f_esc_option = 'binary', #f_esc distribution option
                bpass_read = None,
           ):
    
    M_turn = 5*10**8  #Park+19 parametrization
    V_bias = 4.0 / 3.0  * np.pi * R_bias ** 3

    ########################INITIALIZE SOME SCALING LAWS########################
    np.random.seed(seed = (os.getpid() * int(time.time()) % 123456789))    
 
    if sample_densities:

        delta_list = _sample_densities(z, 
                                       N_iter, 
                                       log10_Mmin, 
                                       log10_Mmax, 
                                       dlog10m, 
                                       R_bias)
        
        hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
        hmf_this.prep_for_hmf_st(log10_Mmin, log10_Mmax, dlog10m)
        hmf_this.prep_collapsed_fractions()
    
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
            hmf_this.prep_for_hmf(log10_Mmin, log10_Mmax, dlog10m)
            masses, mass_func = hmf_this.run_hmf(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            N_mean_cs = ig_hmf.hmf_integral_gtm(masses[:index_to_stop], 
                                                mass_func[:index_to_stop])

        N_mean = int((N_mean_cs)[0])
        N_cs_normalized = N_mean_cs/N_mean_cs[0]
        
    emissivities_x = np.zeros(shape = int(N_iter))
    emissivities_lw = np.zeros(shape = int(N_iter))
    emissivities_uv = np.zeros(shape = int(N_iter))

    for i in range(N_iter):
        if sample_densities:

            delta_bias = delta_list[i]
            delta_bias_before = float(delta_bias)
            delta_bias = np.interp(delta_bias, delta_lin_values, delta_nonlin)
            delta_bias /= hmf_this.dicke()

            masses = hmf_this.bins
            mass_func = hmf_this.ST_hmf(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            #######################TEMPORARY MINIMUM MASS#######################
            Mmin_temp = 7.7
           
            mass_coll = hmf_this.mass_coll_grt_ST(delta_bias, mass=Mmin_temp)
            
            N_mean_cs = ig_hmf.hmf_integral_gtm(masses, 
                                                mass_func) * V_bias
            N_mean = int((N_mean_cs)[0])
            N_cs_norm = N_mean_cs/N_mean_cs[0]

            time_is_up = time.time()
            if mass_binning:

                N_this_iter, mhs = _sample_halos(Mmin_temp, 
                                                 log10_Mmax, 
                                                 mass_binning,
                                                 masses, 
                                                 mass_func, 
                                                 mass_coll,
                                                 V_bias, 
                                                 sample_hmf)

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

        L_UV = np.zeros(shape = N_this_iter)
        L_X = np.zeros(shape = N_this_iter)
        L_LW  = np.zeros(shape = N_this_iter)
                
        for j,mass in enumerate(masses_saved):
            logm = np.log10(mass)            
                
            if sample_Ms:      
                if sample_SFR:
                    a_SFR, b_SFR, a_Ms, b_Ms = sfr_ms_mh_21cmmc(
                                                            z,
                                                            True,
                                                        )
                    sSFR = sigma_SFR_constant()
                    sMs = sigma_SHMR_constant()
 
                    if sample_emiss:
                        #find a_Lx
                        #find a_SFR
                        Ms_mean = 10 ** (a_Ms * logm + b_Ms)
                        SFR_mean = 10 ** (a_SFR * np.log10(Ms_mean) + b_SFR)  
                        Z_mean = metalicity_from_FMR(Ms_mean, SFR_mean)
                        a_Lx_mean, b_Lx_mean =  Lx_SFR(Z_mean)
                        b_Ms -= 1/2  *  np.log(10) * sMs**2
                        b_SFR -= np.log(10) * sSFR**2 / 2

                    else:
                        b_Ms -= np.log(10) * sMs**2 / 2
                        b_SFR -= np.log(10) * sSFR**2 / 2
                    Ms_sample = 10**(np.random.normal((a_Ms*logm + b_Ms), sMs))
                    logmstar = np.log10(Ms_sample)

                    SFR_samp = 10**(normal((a_SFR * logmstar + b_SFR), sSFR))
                    #print("Currently halo mass", logm, "stellar mass", logmstar, "SFR now", np.log10(SFR_samp))     
                else:
                    
                    a_SFR, b_SFR, a_Ms, b_Ms = sfr_ms_mh_21cmmc(
                                                            z,
                                                            True,
                                                        )
                    sMs = sigma_SHMR_constant()
                    b_Ms -= np.log(10) * sMs ** 2 / 2
                    Ms_sample = 10**(normal((a_Ms*logm + b_Ms), sMs))
                    logmstar = np.log10(Ms_sample)
                    
                    SFR_samp = 10**(a_SFR * logmstar + b_SFR)
            
            else:
                if sample_SFR:
                    a_SFR, b_SFR = sfr_ms_mh_21cmmc(
                                                z,
                                                get_stellar_mass = False,
                                            )
           	
                    sSFR = sigma_SFR_constant() #might not be accurate here
                    b_SFR -= np.log(10) * sSFR**2 / 2
                    SFR_samp = 10**(normal((a_SFR*logm + b_SFR), sSFR))
                else:
                    a_SFR, b_SFR = sfr_ms_mh_21cmmc(
                                                z,
                                                get_stellar_mass = False,
                                            )
                    SFR_samp = 10**(a_SFR * logm + b_SFR)
            ######################LUMINOSITIES PART#############################
            #########################X_RAYS FIRST###############################
            if sample_Ms:
                if sample_emiss :
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    sigma_Z = sigma_metalicity_const()
                    Z_mean -= np.log(10) * sigma_Z ** 2 / 2

                    Z_sample = 10 ** (normal((np.log10(Z_mean)), sigma_Z))
                    logsfr = np.log10(SFR_samp)

                    a_Lx, b_Lx = Lx_SFR(Z_sample)
                    #a_Lx, b_Lx = Brorby_lx(Z_sample)
                    sigma_Lx = sigma_Lx_const()
                    b_Lx -= np.log(10) * sigma_Lx**2 / 2   #shift to median      
 
                    Lx_sample = 10**normal(a_Lx * logsfr + b_Lx,sigma_Lx)
                    #print("Currently SFR and stellar mass", logsfr, logmstar, "Metalicity is this", Z_sample, "and finally Lx", np.log10(Lx_sample))
                else:
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    sigma_Z = sigma_metalicity_const()
                    Z_mean -= np.log(10) * sigma_Z**2 / 2

                    Z_sample = 10**(normal((np.log10(Z_mean)), sigma_Z))

                    a_Lx, b_Lx = Lx_SFR(Z_sample)
                    Lx_sample = 10**(a_Lx*logsfr+b_SFR)
            else:
                if sample_emiss:
                    a_Ms, b_Ms = ms_mh_21cmmc()
                    Ms_sample = 10**(a_Ms * logm + b_Ms)
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    
                    a_Lx, b_Lx = Lx_SFR(Z_mean)
                    sigma_Lx = sigma_Lx_const()
                    b_Lx -= np.log(10) * sigma_Lx**2 / 2 
                    Lx_sample = 10**normal(a_Lx * logsfr + b_Lx,sigma_Lx)
                
                else:
                    a_Ms, b_Ms = ms_mh_21cmmc()
                    Ms_sample = 10**(a_Ms * logm + b_Ms)
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    
                    a_Lx, b_Lx = Lx_SFR(Z_mean)
                    Lx_sample = 10**(a_Lx * logsfr + b_Lx)
                
            #######################END OF LX PART###############################
            ######################START OF UV PART##############################
            #assert a_Lx=='something stupid', "Something stupid happened"
            if not sample_Ms:
                
                Z_sample, _ = metalicity_from_FMR(Ms_sample, SFR_samp)
                F_UV = bpass_read.get_UV(Z_sample, 'UV', SFR_samp,z)
                
            else:
                F_UV = bpass_read.get_UV(Z_sample, 'UV', SFR_samp,z)

            #######################END OF UV PART###############################
            #####################START OF LW PART###############################
            if sample_Ms:
                F_LW = bpass_read.get_LW(Z_sample, 'LW', SFR_samp, z)
            
            else:
                Z_sample, _ = metalicity_from_FMR(Ms_sample, SFR_samp)
                F_LW = bpass_read.get_LW(Z_sample, 'LW', SFR_samp, z)
            
            #####################END OF LW PART#################################
            ####################FESC FOR LW FIRST###############################
            if f_esc_option == 'binary':
                f_esc = fesc_distr()
                F_LW *= f_esc

            elif f_esc_option == 'ksz_inference':
                f_esc = fesc_distr(f_esc_option,mass)
                F_LW *= f_esc
            ###################END OF F_ESC FOR LW##############################
            #######################STAR OF F_ESC FOR UV#########################

            if f_esc_option == 'binary':
                f_esc = fesc_distr()
                F_UV *= f_esc
            elif f_esc_option == 'ksz_inference':
                f_esc = fesc_distr(f_esc_option,mass)
                F_UV *= f_esc
                
                
            L_X[j] = Lx_sample
            L_UV[j] = F_UV
            L_LW[j] = F_LW
        print("Here are the luminosities", L_X, "and here's the number of them", np.shape(L_X))
        emissivities_x[i] = np.sum(L_X)
        emissivities_uv[i] = np.sum(L_UV)
        emissivities_lw[i] = np.sum(L_LW)

    emissivities_x_list.append(emissivities_x)
    emissivities_uv_list.append(emissivities_uv)
    emissivities_lw_list.append(emissivities_lw)