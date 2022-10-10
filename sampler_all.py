""""Contains sampler of all three things at the same time"""

import math
from hmf import integrate_hmf as ig_hmf
from sfr import SFRvsMh_lin
import time
from chmf import chmf
from helpers import RtoM, nonlin, metalicity_from_FMR
from common_stuff import _sample_halos, _sample_densities
from bpass_read import bpass_loader
import numpy as np
from fesc import fesc_distr

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
                sample_SFR = 0,          #0 is for sampling using data from Ceverino+18, 1 is for Tacchella+18,
                                        #2 is not sampling but using mean from Ceverino+18, 3 is for mean from Tacchella
                sample_emiss = 0,           #for x-rays only
                calculate_2pcc = False,     #whether to calculate the 2point-correlation Poission correction

                duty_cycle = True,          #whether to turn of duty cycle in the sampler.
                logger = False,          #just printing stuff
                sample_Ms = True,        #also sampling metlaicity with this
                mass_binning = False,     #bin halo mass fucntion
                f_esc_option = 'binary', #f_esc distribution option
                bpass_read = None,
           ):
    
    M_turn = 5*10**8  #Park+19 parametrization
    V_bias = 4.0 / 3.0  * np.pi * R_bias ** 3
    #bpass_read = bpass_loader()
    
    ##if not sample densities, no need to calculate hmf every time
    if (sample_SFR == 0 or sample_SFR==2) and sample_Ms:   #initiate SFR data
        a_SFR, b_SFR, sSFR, a_Ms, b_Ms, sMs = SFRvsMh_lin(z)
    elif (sample_SFR == 0 or sample_SFR==2) and not sample_Ms:
        a_Ms, b_Ms, sMs = SFRvsMh_lin(z,sample_Ms=False)
    
    time_2 = time.time()

    if sample_densities:
        
        delta_list = _sample_densities(z, 
                                       N_iter, 
                                       log10_Mmin, 
                                       log10_Mmax, 
                                       dlog10m, 
                                       R_bias)
        time_inter = time.time()
        print("Actual smapling of densities took,", time_inter-time_2)
        hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
        hmf_this.prep_for_hmf(log10_Mmin, log10_Mmax, dlog10m )
        time_inter2 = time.time()
        print("First chmf part took", time_inter2 - time_inter)

        delta_nonlin = np.linspace(-0.99,10)
        delta_lin_values= nonlin(delta_nonlin)
        time_3 = time.time()
        print("Rest of chmf took", time_3-time_inter2)
        
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

            masses, mass_func = hmf_this.run_hmf(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            mass_coll = hmf_this.mass_coll_grt()

            N_mean_cs = ig_hmf.hmf_integral_gtm(masses, 
                                                mass_func) * V_bias
            N_mean = int((N_mean_cs)[0])
            N_cs_norm = N_mean_cs/N_mean_cs[0]
            time_4 = time.time()
            print("Starting halo sampling, previous stuff took", time_4-time_3)
            if mass_binning:
                N_this_iter, mhs = _sample_halos(log10_Mmin, 
                                                 log10_Mmax, 
                                                 mass_binning,
                                                 masses, 
                                                 mass_func, 
                                                 mass_coll,
                                                 V_bias, 
                                                 sample_hmf)
        time_5 = time.time()
        print("Sampling hmf took", time_5-time_4)
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
        time_6 = time.time()
        print("Duty cycle finished", time_6-time_5)
        for j,mass in enumerate(masses_saved):
            logm = np.log10(mass)            
                      
            if sample_SFR == 0 and sample_Ms:
                Ms_sample = 10**(np.random.normal((a_Ms*logm + b_Ms), sMs))
                logmstar = np.log10(Ms_sample)
                SFR_samp = 10**(np.random.normal((a_SFR*logmstar+b_SFR), sSFR))
                
            elif sample_SFR==1:
                Ms_sample = 10**(np.random.normal((a_Ms*logm + b_Ms), sMs))
                logmstar = np.log10(Ms_sample)
                b_SFR = -15.365-1.67 * np.log10(0.15)
                a_SFR = 1.67
                SFR_samp = 10**(np.random.normal(a_SFR * logmstar + b_SFR, 0.2))
                
            elif sample_SFR==2 and sample_Ms:
                Ms_sample = 10**((a_Ms * logm + b_Ms))
                SFR_samp = 10**(a_SFR * np.log10(Ms_sample) + b_SFR)
                
            elif sample_SFR==3:
                Ms_sample = 10**((a_Ms * logm + b_Ms))
                b_SFR = -15.365-1.67*np.log(0.15)
                a_SFR = 1.67
                SFR_samp = 10**(a_SFR * np.log10(Ms_sample) + b_SFR)
                
            elif sample_SFR == 0 and not sample_Ms:
                SFR_samp = 10**(np.random.normal((a_Ms * logm + b_Ms), sMs))
            elif sample_SFR == 2 and not sample_Ms:
                SFR_samp = 10**(a_Ms * logm + b_Ms)
            time_7 = time.time()
            print("Sampling SFR/Mstell took", time_7-time_6)
            ######################LUMINOSITIES PART#############################
            #########################X_RAYS FIRST###############################
            
            if sample_Lx==0 and sample_Ms:
                print("This is the SFR", SFR_samp, "\n")                 
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

                a_Lx_try, _, _ = LxvsSFR_lin(Z_mean)
                Z_mean -= np.log(10) * a_Lx_try * sigma_Z**2 / 2

                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))

                a_Lx, b_Lx, sigma_Lx = Brorby_lx(Z_sample)
                
                b_Lx -= np.log(10) * sigma_Lx**2 / 2 #shift to median

                Lx_sample = 10**np.random.normal((a_Lx*logsfr+b_SFR), sigma_Lx)
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
                
            #######################END OF LX PART###############################
            ######################START OF UV PART##############################
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
        emissivities_x[i] = np.sum(L_X)
        emissivties_uv[i] = np.sum(L_UV)
        emissivities_lw[i] = np.sum(L_LW)
    emissivities_x_list.append(emissivities_x)
    emissivities_uv_list.append(emissivities_uv)
    emissivities_lw_list.append(emissivities_lw)