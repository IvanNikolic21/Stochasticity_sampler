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
from save import HdF5Saver

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
    time_enter_sampler = time.time()
    M_turn = 5*10**7  #Park+19 parametrization
    V_bias = 4.0 / 3.0  * np.pi * R_bias ** 3

    ########################INITIALIZE SOME SCALING LAWS########################
    np.random.seed(seed = (os.getpid() * int(time.time()) % 123456789))    

    #initialize h5 file
    container = HdF5Saver(
        z,
        os.getpid(),
        '/home/inikolic/projects/stochasticity/samples/'
    )
    container.create_file()
    container.create_redshift()

    if sample_densities:

        delta_list = _sample_densities(z, 
                                       N_iter, 
                                       log10_Mmin, 
                                       log10_Mmax, 
                                       dlog10m, 
                                       R_bias)
        #np.savetxt('/home/inikolic/projects/stochasticity/samples/density{}.txt'.format(z), np.array(delta_list)) 
        hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
        hmf_this.prep_for_hmf_st(log10_Mmin, log10_Mmax, dlog10m)
        hmf_this.prep_collapsed_fractions()
    
        delta_nonlin = np.linspace(-0.99,10)
        delta_lin_values= nonlin(delta_nonlin)
        time_finished_densities = time.time()
        print("h5 initialization, and density sampling took", time_finished_densities-time_enter_sampler)        
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
    tot_mass = np.zeros(shape=int(N_iter))
    #print("Starting the iteration", flush=True)
    for i in range(N_iter):
        start = time.time()
        if sample_densities:

            delta_bias = delta_list[i]
            delta_bias_before = float(delta_bias)
            delta_bias = np.interp(delta_bias, delta_lin_values, delta_nonlin)
            delta_bias /= hmf_this.dicke()

            delta_container = container.add_delta_group(delta_bias)

            masses = hmf_this.bins
            mass_func = hmf_this.ST_hmf(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            #######################TEMPORARY MINIMUM MASS#######################
            Mmin_temp = 7.6
           
            mass_coll = hmf_this.mass_coll_grt_ST(delta_bias, mass=Mmin_temp)
            time_finished_hmf_initialization = time.time()
            print("Time it took from density sampling to hmf initialization: ", time_finished_hmf_initialization - time_finished_densities)
            N_mean_cs = ig_hmf.hmf_integral_gtm(masses, 
                                                mass_func) * V_bias
            #np.savetxt('/home/inikolic/projects/stochasticity/samples/mass{}.txt'.format(delta_bias), np.array(mass_func))
            N_mean = int((N_mean_cs)[0])
            N_cs_norm = N_mean_cs/N_mean_cs[0]

            time_is_up = time.time()
            if mass_binning:
     #           print("starting to sample the halos for the first time", flush=True)
                #print(Mmin_temp, log10_Mmax, mass_binning)
                #print( len(masses), sep=", ")
                #print( len(mass_func), sep=", ")
                #print(mass_coll, V_bias, sample_hmf, "These are the ingredients", delta_bias, "and the previous number is delta")
                N_this_iter, mhs = _sample_halos(Mmin_temp, 
                                                 log10_Mmax, 
                                                 mass_binning,
                                                 masses[:len(mass_func)], 
                                                 mass_func, 
                                                 mass_coll,
                                                 V_bias, 
                                                 sample_hmf)
                time_for_halo_sampling = time.time()
                print("Halo sampling is a bitch, and here's why:", time_for_halo_sampling - time_finished_hmf_initialization)
                #print("These are the masses:", mhs, flush=True)
            #    np.savetxt('/home/inikolic/projects/stochasticity/samples/halos{}.txt'.format(delta_bias), np.array(mhs))
                #print("Here's one file for you to analyze", flush=True)
                if np.sum(mhs) < 0.5 *  mass_coll:
                    print("minimal temperature", Mmin_temp, "log10_Mmax", log10_Mmax,"mass_binning", mass_binning, "redshift", z)
                    print( len(masses), sep=", ")
                    print( len(mass_func), sep=", ")
                    print(mass_coll, V_bias, sample_hmf, "These are the ingredients", delta_bias, "and the previous number is delta")
                    raise ValueError("For this iteration sampling halos failed")
                #assert len(mhs) < 1, "only one mass, aborting"
        time_is_now = time.time()
        #print("Time for mass sampling", time_is_now - time_is_up)
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

        container.add_halo_masses(np.array(masses_saved))

        Mstar_samples = []
        metalicity_samples = []
        SFR_samples = []
        beta_samples = []

        tot_mass[i] = np.sum(masses_saved)
        L_UV = np.zeros(shape = N_this_iter)
        L_X = np.zeros(shape = N_this_iter)
        L_LW  = np.zeros(shape = N_this_iter)
        time_to_start_getting_quantities = time.time()
        print("Starting the for loop:", time_to_start_getting_quantities - time_for_halo_sampling)        
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
                    time_inside_loop = time.time() 
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
            time_for_stellar_mass = time.time()
            print("Getting stellar mass took:", time_for_stellar_mass - time_inside_loop, flush=True)
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
            time_to_get_X = time.time()
            print("Time it took to get X-rays", time_to_get_X - time_for_stellar_mass, flush=True)
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

            time_to_get_LW = time.time()
            print("Other emissivities took", time_to_get_LW - time_to_get_X)
            ###########GET_BETAS#######
            beta_samples.append(bpass_read.get_beta(Z_sample, SFR_samp, z))
            finally_beta = time.time()
            print("And finally beta", finally_beta - time_to_get_LW)

            Mstar_samples.append(Ms_sample)
            SFR_samples.append(SFR_samp)
            metalicity_samples.append(Z_sample)
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
        container.add_stellar_masses(np.array(Mstar_samples))
        container.add_SFR(np.array(SFR_samples))
        container.add_metal(np.array(metalicity_samples))
        container.add_beta(np.array(beta_samples))
        container.add_Lx(L_X)
        container.add_L_LW(L_LW)
        container.add_L_UV(L_UV)

        #print("Here are the luminosities", L_X, "and here's the number of them", np.shape(L_X))
        emissivities_x[i] = np.sum(L_X)
        emissivities_uv[i] = np.sum(L_UV)
        emissivities_lw[i] = np.sum(L_LW)
        end = time.time()
        print("time for one iteraton", end-start)
    #np.savetxt('/home/inikolic/projects/stochasticity/samples/tot_halo_mass{}.txt'.format(z), tot_mass)
    emissivities_x_list.append(emissivities_x)
    emissivities_uv_list.append(emissivities_uv)
    emissivities_lw_list.append(emissivities_lw)

    container.add_X(emissivities_x)
    container.add_LW(emissivities_lw)
    container.add_UV(emissivities_uv)
    container = None
