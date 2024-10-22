""""Contains sampler of all three things at the same time"""
import os
from hmf import integrate_hmf as ig_hmf
from scaling import sfr_ms_mh_21cmmc, sigma_SHMR_constant, OH_to_mass_fraction
from scaling import metalicity_from_FMR, sigma_metalicity_const, Lx_SFR
from scaling import sigma_Lx_const, ms_mh_flattening, Lx_SFR_Davies
from scaling import sigma_SFR_variable, DeltaZ_z, sigma_SFR_Hassan
from chmf import chmf
from common_stuff import _sample_halos
from common_stuff import  _get_loaded_halos
import numpy as np
from fesc import fesc_distr
from numpy.random import normal
import time
import h5py


class SamplerOutput:
    """
        Class that contains all the information from the simulation. It's
        directly passed to the saver algorithm, and it's accessed directly
        throughout the code. It's main parameter is the overdensity.
    """

    def __init__(self, delta):
        self.delta = delta


def sampler_all_func(
        emissivities_x_list,
        emissivities_lw_list,
        emissivities_uv_list,
        z=10,
        delta_bias=0.0,
        r_bias=5.0,
        log10_m_min=5,
        log10_m_max=15,
        dlog10m=0.01,
        N_iter=1000,
        sample_hmf=True,
        sample_dens=True,
        sample_SFR=True,
        sample_emiss=True,
        sample_met=True,
        duty_cycle=True,          #whether to turn of duty cycle.
        sample_Mstar=True,
        mass_binning=False,     #bin halo mass fucntion
        f_esc_option='binary', #f_esc distribution option
        bpass_read=None,
        filename=None,
        control_run=False,
        proc_number=0,
        get_previous=False,
        density_inst=None,
        hmf_this=None,
        SFH_samp=None,
        iter_num=0,
        shift_scaling=False,
        literature_run=None,
        flattening=True,
        mass_sampler='binning',
):

    M_turn = 5*10**8  #Park+19 parametrization
    V_bias = 4.0 / 3.0  * np.pi * r_bias ** 3
    #SFH_samp = SFH_sampler(z)
    ########################INITIALIZE SOME SCALING LAWS########################
    np.random.seed(seed = (os.getpid() * int(time.time()) % 123456789))
    a_SFR, b_SFR, a_Ms, b_Ms = sfr_ms_mh_21cmmc(
        z,
        True,
    )

    if literature_run == 'Hassan21' and sample_Mstar:
        raise ValueError("Hassan21 requires only SFR sampling. "
                         "Check your setup")

    # #initialize h5 file
    # container = HdF5Saver(
    #     z,
    #     main_pid,
    #     '/home/inikolic/projects/stochasticity/samples/dir_080323/full/'
    # )

    if sample_dens and not control_run and not get_previous:
        pass
    #
    #     delta_list = _sample_densities(z,
    #                                    N_iter,
    #                                    log10_Mmin,
    #                                    log10_Mmax,
    #                                    dlog10m,
    #                                    R_bias)
    #     #np.savetxt('/home/inikolic/projects/stochasticity/samples/density{}.txt'.format(z), np.array(delta_list))
    #     hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
    #     hmf_this.prep_for_hmf_st(log10_Mmin, log10_Mmax, dlog10m)

    elif not sample_dens and not control_run and get_previous=='False':

        if delta_bias==0.0:
            pass #this I haven't used for 2 years so I don't think I should be.
            # hmf_this = hmf.MassFunction(z = z,
            #                             Mmin=log10_m_min,
            #                             Mmax = log10_m_max,
            #                             dlog10m = dlog10m)
            # mass_func = hmf_this.dndm
            # masses = hmf_this.m
            # N_mean_cs = ig_hmf.hmf_integral_gtm(masses,
            #                                     mass_func,
            #                                     mass_density=False) * V_bias
            # cumulative_mass = ig_hmf.hmf_integral_gtm(masses,
            #                                           mass_func,
            #                                           mass_density=True)

        else:
            hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = r_bias)
            hmf_this.prep_for_hmf(log10_m_min, log10_m_max, dlog10m)
            masses, mass_func = hmf_this.run_hmf(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element == 0 or np.isnan(mass_func_element):
                    break
            masses = masses[:index_to_stop]

    list_of_outputs = []

    for i in range(N_iter):

        if not control_run and get_previous=='False':

            # new 06/04/23: this is a Lagrangian density at z=z going into chmf.
            if hasattr(density_inst, '__len__'):
                delta_bias = density_inst[i]
            else:
                delta_bias = density_inst
            if sample_dens:
                class_int = SamplerOutput(delta_bias)
            else:
                random_dens = np.random.random()
                class_int = SamplerOutput(random_dens)
            setattr(class_int, 'filename', filename)
            setattr(class_int, 'redshift', z)

            masses = hmf_this.bins
            mass_func = hmf_this.hmf_coll(delta_bias)

            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element == 0 or np.isnan(mass_func_element):
                    break
            masses = masses[:index_to_stop]
            mass_func = mass_func[:index_to_stop]

            m_min_temp = 7.6
            
            mass_coll = hmf_this.mass_coll_grt_st(delta_bias, mass=m_min_temp)
            if mass_binning:
                n_this_iter, mhs = _sample_halos(masses[:len(mass_func)],
                                                 mass_func,
                                                 m_min_temp,
                                                 log10_m_max,
                                                 V_bias,
                                                 mode=mass_sampler,
                                                 Poisson=sample_hmf,
                                                 nbins=2,
                                                 mass_coll=mass_coll,
                                                 )

                if np.sum(mhs) < 0.5 * mass_coll:
                    print(len(masses), sep=", ")
                    print(len(mass_func), sep=", ")
                    print(
                        np.sum(mhs),
                        mass_coll,
                        V_bias,
                        sample_hmf,
                        "These are the ingredients",
                        delta_bias,
                        "and the previous number is delta"
                    )

        elif control_run:
            fake_delta = float(str(iter_num) + str(os.getpid()))
            class_int = Sampler_Output(fake_delta)
            setattr(class_int, 'filename', filename)
            setattr(class_int, 'redshift', z)
            n_this_iter, mhs = _get_loaded_halos(
                z,
                direc='/home/inikolic/projects/stochasticity/_cache'
            )
        elif get_previous != 'False':
            with h5py.File(
                    '/home/inikolic/projects/stochasticity/_cache/Mh_'+get_previous+'.h5',
                    'r'
            ) as f_prev:
                print(
                    "Is this proc number okay?",
                    str(float(proc_number)),
                    "for z=",
                    str(z),
                    "and this iter",
                    str(float(iter_num))
                )
                delta_bias = f_prev[
                    str(z)
                ][
                    str(float(proc_number))
                ][
                    str(float(iter_num * N_iter + i))
                ].attrs['delta']
                if delta_bias == 0.0:
                    delta_bias = 9.0 + np.random.random() #random delta_bias
                class_int = SamplerOutput(delta_bias)
                setattr(class_int, 'filename', filename)
                setattr(class_int, 'redshift', z)
                mhs = np.array(
                    f_prev[
                        str(z)
                    ][
                        str(float(proc_number))
                    ][
                        str(float(iter_num * N_iter + i))
                    ]['Mh']
                )
                n_this_iter = len(mhs)
                if len(mhs) == 1 and mhs == np.zeros((1,)):
                    mhs = []
                    n_this_iter = 0
            #print("Time for mass sampling", time_is_now - time_is_up)
            n_this_iter = int(n_this_iter)
        time_2 = time.time()
        #print("Got masses now", time_2 - time_1)


        if not mass_binning:
            raise ValueError("This is not implemented anymore!")
        
        masses_saved = []
        if duty_cycle and not control_run and (
                get_previous == 'bigz2' or get_previous == 'False' or get_previous[-6:] == 'hmfnew' or get_previous[-3:] == 'ms1' or get_previous[-11:] == 'hmfnew_noPo' or get_previous[-5]=='James'
        ):
            for index, mass in enumerate(mhs):
                if np.random.binomial(1, np.exp(-M_turn/mass)):
                    masses_saved.append(mass)
            len_mass = len(masses_saved)
        elif control_run or (get_previous!='False' and get_previous != 'bigz2'):
            masses_saved = mhs # duty cycle already applied
            len_mass = len(masses_saved)

        #container.add_halo_masses(np.array(masses_saved))
        setattr(class_int, 'halo_masses', np.array(mhs))
        Mstar_samples = np.zeros(shape = len_mass)
        metalicity_samples = np.zeros(shape = len_mass)
        SFR_samples = np.zeros(shape = len_mass)
        beta_samples = np.zeros(shape=len_mass)
        SFH_samples = []
        n_ion_samples = np.zeros(shape = len_mass)

            #tot_mass[i] = np.sum(masses_saved)
        L_UV = np.zeros(shape = len_mass)
        L_X = np.zeros(shape = len_mass)
        L_LW  = np.zeros(shape = len_mass)
        L_LyC = np.zeros(shape=len_mass)
        time_to_start_getting_quantities = time.time()
        time_3  = time.time()
        #print("before sampling masses", time_3 - time_2)
        #print("Starting the for loop:", time_to_start_getting_quantities - time_for_halo_sampling)        
        for j,mass in enumerate(masses_saved):
            logm = np.log10(mass)
                
            if sample_Mstar:
                if sample_SFR:

                    sMs = sigma_SHMR_constant()
                    time_inside_loop = time.time()
                    if sample_emiss:
                        #find a_Lx
                        #find a_SFR
                        if flattening:
                            Ms_mean = ms_mh_flattening(mass)
                        else:
                            Ms_mean = 10 ** (a_Ms * logm + b_Ms)
                        SFR_mean = 10 ** (a_SFR * np.log10(Ms_mean) + b_SFR)
                        Z_mean = metalicity_from_FMR(Ms_mean, SFR_mean)
                        a_Lx_mean, b_Lx_mean =  Lx_SFR(Z_mean)
                    #b_Ms -= 1/2  *  np.log(10) * sMs**2
                    #b_SFR -= np.log(10) * sSFR**2 / 2

                #else:
                    #b_Ms -= np.log(10) * sMs**2 / 2
                    #b_SFR -= np.log(10) * sSFR**2 / 2
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sMs**2
                    else:
                        fct = 0.0
                    if flattening:
                        Ms_sample = 10**(np.random.normal(np.log10(ms_mh_flattening(mass)), sMs))
                    else:
                        Ms_sample = 10**(np.random.normal((a_Ms*logm + b_Ms - fct), sMs))
                    logmstar = np.log10(Ms_sample)
                    if literature_run=='Hassan21':
                        sSFR = sigma_SFR_Hassan()
                    else:
                        sSFR = sigma_SFR_variable(Ms_sample)
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sSFR**2
                    else:
                        fct = 0.0
                    SFR_samp = 10**(normal((a_SFR * logmstar + b_SFR-fct), sSFR))
                #print("Currently halo mass", logm, "stellar mass", logmstar, "SFR now", np.log10(SFR_samp))
                else:

                    sMs = sigma_SHMR_constant()
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sMs**2
                    else:
                        fct = 0.0
                    #b_Ms -= np.log(10) * sMs ** 2 / 2
                    if flattening:
                        Ms_sample = 10**(np.random.normal(np.log10(ms_mh_flattening(mass)), sMs))
                    else:
                        Ms_sample = 10**(normal((a_Ms*logm + b_Ms - fct), sMs))
                    logmstar = np.log10(Ms_sample)
                    
                    SFR_samp = 10**(a_SFR * logmstar + b_SFR)
            else:
                if sample_SFR:
                    #a_SFR, b_SFR = sfr_ms_mh_21cmmc(
                    #                            z,
                    #                            get_stellar_mass = False,
                    #                        )
                    if flattening:
                        Ms_sample = ms_mh_flattening(mass)
                    else:
                        Ms_sample = 10**(a_Ms * logm + b_Ms)
                    if literature_run=='Hassan21':
                        sSFR = sigma_SFR_Hassan()
                    else:
                        sSFR = sigma_SFR_variable(Ms_sample) #might not be accurate here
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sSFR**2
                    else:
                        fct = 0.0
                    logmstar = np.log10(Ms_sample)
                    #b_SFR -= np.log(10) * sSFR**2 / 2
                    SFR_samp = 10**(normal((a_SFR*logmstar + b_SFR-fct), sSFR))
                else:
                    #a_SFR, b_SFR = sfr_ms_mh_21cmmc(
                    #                            z,
                    #                            get_stellar_mass = False,
                    #                        )
                    if flattening:
                        Ms_sample = ms_mh_flattening(mass)
                    else:
                        Ms_sample = 10 ** (a_Ms * logm + b_Ms)
                    SFR_samp = 10**(a_SFR * logm + b_SFR)
            time_for_stellar_mass = time.time()
            #print("Getting stellar mass took:", time_for_stellar_mass - time_3, flush=True)
            ######################LUMINOSITIES PART#############################
            #########################X_RAYS FIRST###############################
            if sample_met:
                if sample_emiss :
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    Z_mean += DeltaZ_z(z)
                    sigma_Z = sigma_metalicity_const()
                #Z_mean -= np.log(10) * sigma_Z ** 2 / 2   #I don't think metalicity should be lognormal

                    #Z_sample = 10 ** (normal((np.log10(Z_mean)), sigma_Z))
                    Z_sample = normal(Z_mean, sigma_Z)
                    logsfr = np.log10(SFR_samp)

                    #a_Lx, b_Lx = Brorby_lx(Z_sample)
                    metal_sample = OH_to_mass_fraction(Z_sample)

                    metal_sample = metal_sample / 10 ** 0.42 / 0.02 #in units of solar

                    Lx_mean = Lx_SFR_Davies(metal_sample) * SFR_samp
                    sigma_Lx = sigma_Lx_const()
                    #b_Lx -= np.log(10) * sigma_Lx**2 / 2   #shift to median
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sigma_Lx**2
                    else:
                        fct = 0.0
                    Lx_sample = 10**normal(np.log10(Lx_mean)-fct,sigma_Lx)
                    #print("Currently SFR and stellar mass", logsfr, logmstar, "Metalicity is this", Z_sample, "and finally Lx", np.log10(Lx_sample))
                else:
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    Z_mean += DeltaZ_z(z)
                    sigma_Z = sigma_metalicity_const()
                    #Z_mean -= np.log(10) * sigma_Z**2 / 2

                    Z_sample = normal(Z_mean, sigma_Z)

                    metal_sample = OH_to_mass_fraction(Z_sample)

                    metal_sample = metal_sample / 10 ** 0.42 / 0.02  # in units of solar

                    Lx_sample = Lx_SFR_Davies(metal_sample) * SFR_samp

            else:
                if sample_emiss:
                    #a_Ms, b_Ms = ms_mh_21cmmc()
                    #Ms_sample = 10**(a_Ms * logm + b_Ms)
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    Z_mean += DeltaZ_z(z)
                    Z_sample = Z_mean

                    metal_sample = OH_to_mass_fraction(Z_sample)

                    metal_sample = metal_sample / 10 ** 0.42 / 0.02  # in units of solar

                    Lx_mean = Lx_SFR_Davies(metal_sample) * SFR_samp
                    sigma_Lx = sigma_Lx_const()
                    #b_Lx -= np.log(10) * sigma_Lx**2 / 2
                    if shift_scaling:
                        fct = np.log(10) * 0.5 * sigma_Lx**2
                    else:
                        fct = 0.0
                    Lx_sample = 10**normal(np.log10(Lx_mean)-fct,sigma_Lx)
                
                else:
                    #a_Ms, b_Ms = ms_mh_21cmmc()
                    #Ms_sample = 10**(a_Ms * logm + b_Ms)
                    logsfr = np.log10(SFR_samp)
                    Z_mean = metalicity_from_FMR(Ms_sample, SFR_samp)
                    Z_mean += DeltaZ_z(z)
                    Z_sample = Z_mean

                    metal_sample = OH_to_mass_fraction(Z_sample)

                    metal_sample = metal_sample / 10 ** 0.42 / 0.02  # in units of solar

                    Lx_sample = Lx_SFR_Davies(metal_sample) * SFR_samp

            time_to_get_X = time.time()
            #print("Time it took to get X-rays", time_to_get_X - time_for_stellar_mass, flush=True)
            #######################END OF LX PART###############################
            ######################START OF UV PART##############################
            #assert a_Lx=='something stupid', "Something stupid happened"
            if not sample_met:
                
                Z_sample = metalicity_from_FMR(Ms_sample, SFR_samp)
                Z_sample += DeltaZ_z(z)
                F_UV = bpass_read.get_UV(Z_sample, Ms_sample, SFR_samp,z, SFH_samp = SFH_samp)
           #     F_UV = SFR_samp / 1.15 * 1e28
                
            else:
                F_UV = bpass_read.get_UV(Z_sample, Ms_sample, SFR_samp,z, SFH_samp=SFH_samp)
            #    F_UV = SFR_samp / 1.15 * 1e28 
            #######################END OF UV PART###############################
            #####################START OF LW PART###############################
            if sample_met:
                F_LW = bpass_read.get_LW(Z_sample, Ms_sample, SFR_samp, z)
            
            else:
                Z_sample  = metalicity_from_FMR(Ms_sample, SFR_samp)
                Z_sample += DeltaZ_z(z)

                F_LW = bpass_read.get_LW(Z_sample, Ms_sample, SFR_samp, z)
###LyC
            if sample_met:
                F_LyC = bpass_read.get_LyC(Z_sample, Ms_sample, SFR_samp, z)

            else:
                Z_sample = metalicity_from_FMR(Ms_sample, SFR_samp)
                Z_sample += DeltaZ_z(z)
                F_LyC = bpass_read.get_LyC(Z_sample, Ms_sample, SFR_samp, z)
            
            #let's perturb emissivities as well
            if sample_emiss:
                mag_filt = 0.1
                if shift_scaling:
                    fct = np.log(10) * 0.5 * mag_filt**2
                else:
                    fct = 0.0
                F_UV = 10**normal(np.log10(F_UV)-fct, mag_filt)
                F_LW = 10**normal(np.log10(F_LW)-fct, mag_filt)
                F_LyC = 10**normal(np.log10(F_LyC)-fct, mag_filt)
                
            time_to_get_LW = time.time()
            #print("Other emissivities took", time_to_get_LW - time_to_get_X)
            ###########GET_BETAS#######
            #beta_samples[j] = bpass_read.get_beta(Z_sample, SFR_samp, Ms_sample, z)
            #finally_beta = time.time()
            #print("And finally beta", finally_beta - time_to_get_LW)


            #get number of ionizing photons (produced!)
            n_ion_samples_temp = bpass_read.get_nion(Z_sample, SFR_samp, Ms_sample, z)
            if sample_emiss:
                n_ion_samples[j] = 10**normal(np.log10(n_ion_samples_temp), mag_filt)
            else:
                n_ion_samples[j] = n_ion_samples_temp

            Mstar_samples[j] = Ms_sample
            SFR_samples[j] = SFR_samp
            metalicity_samples[j] = Z_sample
            #####################END OF LW PART#################################
            ####################FESC FOR LW FIRST###############################
            #if f_esc_option == 'binary':
            #    f_esc = fesc_distr()
            #    F_LW *= f_esc

            #elif f_esc_option == 'ksz_inference':
            #    f_esc = fesc_distr(f_esc_option,mass)
            #    F_LW *= f_esc
            ###################END OF F_ESC FOR LW##############################
            #######################STAR OF F_ESC FOR UV#########################

            if f_esc_option == 'binary':
                if sample_emiss:
                    f_esc = fesc_distr()
                    F_LyC *= f_esc
                else:
                    f_esc = fesc_distr(option='mean')
                    F_LyC *= f_esc #this is new, default option is log-normal
                    #there's gotta be a better way to do this. But we'll keep it for now
                    #F_LyC *= 0.053 #constant value
            elif f_esc_option == 'Mascia+23':
                if sample_emiss:
                    f_esc = fesc_distr(option='Mascia+23')
                    F_LyC *= f_esc
                else:
                    f_esc = fesc_distr(option='mean')
                    F_LyC *= f_esc

            elif f_esc_option == 'ksz_inference':
                f_esc, scat = fesc_distr(f_esc_option,mass)
                if sample_emiss:
                    f_esc = np.clip(10**(normal(f_esc, scat)), a_min=0.0, a_max=1.0)
                    F_LyC *= f_esc
                else:
                    F_LyC *= 10**f_esc
            SFH_samples.append(bpass_read.SFH)
            L_X[j] = Lx_sample
            L_UV[j] = F_UV
            L_LW[j] = F_LW
            L_LyC[j] = F_LyC

        #M_uv = get_Muv(L_UV, solar_mult=True)
        #UV_lf, _ = get_uvlf(M_uv, Rbias=R_bias)
        setattr(class_int, 'stellar_masses', Mstar_samples)
        setattr(class_int, 'SFR', SFR_samples)
        setattr(class_int, 'metallicity', metalicity_samples)
        #setattr(class_int, 'beta', beta_samples)
        setattr(class_int, 'nion', n_ion_samples)
        setattr(class_int, 'L_x', L_X)
        setattr(class_int, 'L_lw', L_LW)
        setattr(class_int, 'L_uv', L_UV)
        setattr(class_int, 'L_lyc', L_LyC)
        #setattr(class_int, 'uv_lf', np.array(UV_lf))
        setattr(class_int, 'proc_number', proc_number)
        setattr(class_int, 'iter_number', iter_num * N_iter + i)

        # container.add_stellar_masses(np.array(Mstar_samples))
        # container.add_SFR(np.array(SFR_samples))
        # container.add_metal(np.array(metalicity_samples))
        # container.add_beta(np.array(beta_samples))
        # container.add_nion(np.array(n_ion_samples))
        # container.add_Lx(L_X)
        # container.add_L_LW(L_LW)
        # container.add_L_UV(L_UV)
        # container.add_L_LyC(L_LyC)
        # container.add_uvlf(UV_lf)

        if len(SFH_samples)>0:
            max_len_SFH = max([len(haj) for haj in SFH_samples])
            SFH_array = np.zeros((len(SFH_samples), max_len_SFH))
            for haj in range(len(SFH_samples)):
                SFH_array[haj,:len(SFH_samples[haj])] = SFH_samples[haj]
        else:
            SFH_array = np.zeros((42))

        setattr(class_int, 'SFH', np.array(SFH_array))

        list_of_outputs.append(class_int)
        #container.add_SFH(SFH_array)

    #    emissivities_x[i] = np.sum(L_X)
    #    emissivities_uv[i] = np.sum(L_UV)
    #    emissivities_lw[i] = np.sum(L_LW)
        end = time.time()
    #return class_int
    #np.savetxt('/home/inikolic/projects/stochasticity/samples/tot_halo_mass{}.txt'.format(z), tot_mass)
 #   emissivities_x_list.append(emissivities_x)
 #   emissivities_uv_list.append(emissivities_uv)
 #   emissivities_lw_list.append(emissivities_lw)

    #container.add_X(emissivities_x)
    #container.add_LW(emissivities_lw)
    #container.add_UV(emissivities_uv)
  #  container = None

    return list_of_outputs
