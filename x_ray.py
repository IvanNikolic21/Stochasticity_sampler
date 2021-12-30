import mcint
import random
import math
import time
import numpy as np
import hmf
from hmf import integrate_hmf
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt
import json
from numpy.random import default_rng as rng

def nonlin(x):
    return -1.35*(1+x)**(-2/3) + 0.78785*(1+x)**(-0.58661) - 1.12431*(1+x)**(-1/2) + 1.68647

def Sampler(z=10,
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
            sample_Lx = 0,           #0 is for sampling using data from Lehmer+20, 1 is from Kouroumpatzakis+20
                                        #2 is not sampling but using mean from Lehmer+20, 3 is for mean from Kouroumpatzakis+20
            calculate_2pcc = False,     #whether to calculate the 2point-correlation Poission correction
            interpolating = True,       #whether to interpolate the hmf to better sample the masses,
                                        #it's not slow and is done once so True is the best option
            duty_cycle = True,          #whether to turn of duty cycle in the sampler.
            logger = False,          #just printing stuff
            sample_Ms = True,        #also sampling metlaicity with this
           ):
    ##let's intiate the halo mass function
    M_turn = 5*10**8  #Park+19 parametrization

    #sample densities for a given radius:

    ##########################################TOOOOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOOOO######################################

    ##if not sample densities, no need to calculate hmf every time

    if (sample_SFR == 0 or sample_SFR==2) and sample_Ms:   #initiate SFR data
        means_Ms_relation, sigma_Ms_relation, a_Ms_relation, b_Ms_relation, \
        means_SFR_relation, sigma_SFR_relation, a_SFR_relation, b_SFR_relation = SFRvsMh(z)
    elif (sample_SFR == 0 or sample_SFR==2) and not sample_Ms:
        means_Ms_relation, sigma_Ms_relation, a_Ms_relation, b_Ms_relation = SFRvsMh(z,sample_Ms=False)

    if sample_densities:
        hmf_this = hmf.MassFunction(z = z, Mmin = log10_Mmin, Mmax = log10_Mmax, dlog10m = dlog10m, conditional_mass = RtoM(R_bias))   #just a quick one to calculate

        #intialize samples of densities:
        gauss_lagr = []
        delta_lagrangian = np.linspace(-7,15, 100000)
        delta_other= nonlin(delta_lagrangian*hmf_this.growth_factor)/hmf_this.growth_factor
        sigma_non_nan = hmf_this.sigma[~np.isnan(hmf_this.sigma)]
        radii_non_nan = hmf_this.radii[~np.isnan(hmf_this.sigma)]
        for index, i in enumerate(delta_other):
            sigma = np.interp(R_bias*(1+delta_lagrangian[index]*hmf_this.growth_factor)**(1/3.0), radii_non_nan/cosmo.h, sigma_non_nan)\
                    /hmf_this.growth_factor
            gauss_lagr.append(1/np.sqrt(2*np.pi)/sigma* np.exp(-(i)**2/2/(sigma)**2) )
        N_mean_cumsum = integrate.cumtrapz(gauss_lagr, delta_other)
        cumsum = N_mean_cumsum / N_mean_cumsum[-1]
        random_numb = np.random.uniform(size=N_iter)
        delta_list = np.zeros(shape=N_iter)
        for index, random in enumerate(random_numb):
            delta_list[index] = np.interp(random, cumsum, delta_lagrangian[:-1])
            delta_list[index] = nonlin(delta_list[index] * hmf_this.growth_factor)
        if logger:
            print("This many delta are positive:", np.sum(delta_list>0), "and this many negative", np.sum(delta_list<0))
        hmf_this = chmf(z=z, delta_bias=delta_bias, R_bias = R_bias)
        hmf_this.prep_for_hmf(log10_Mmin = log10_Mmin, log10_Mmax = log10_Mmax, dlog10m = dlog10m )

        delta_nonlin = np.linspace(-0.99,10)
        delta_lin_values= nonlin(delta_nonlin)
        new_values = np.linspace(-10,10)

    if not sample_densities:
        if delta_bias==0.0:
            ##calculate mass_bin resolution
            hmf_this = hmf.MassFunction(z = z, Mmin = log10_Mmin, Mmax = log10_Mmax, dlog10m = dlog10m)   #at this redshift, larger mass halos are non-existent
            mass_func = hmf_this.dndm
            masses = hmf_this.m
            N_mean_cumsum = integrate_hmf.hmf_integral_gtm(masses, mass_func, mass_density=False) * 4/3 * np.pi *R_bias**3
            cumulative_mass = integrate_hmf.hmf_integral_gtm(masses, mass_func, mass_density=True)

        else:
            hmf_this = chmf(z=z, delta_bias = delta_bias, R_bias = R_bias)        #TODO fix this still, it's in my to do list.
            hmf_this.prep_for_hmf(log10_Mmin = log10_Mmin, log10_Mmax = log10_Mmax, dlog10m = dlog10m)
            masses, mass_func = hmf_this.run_hmf(delta_bias)
            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]
            N_mean_cumsum = integrate_hmf.hmf_integral_gtm(masses[:index_to_stop], mass_func[:index_to_stop])
            #cumulative_mass = integrate_hmf(masses)
        #print(N_mean_cumsum)

        N_mean = int((N_mean_cumsum)[0])
        if logger:
            print("without sampling densities I have ", N_mean, "objects")

        N_cumsum_normalized = N_mean_cumsum/N_mean_cumsum[0]
    emissivities = np.zeros(shape=N_iter)


    for i in range(N_iter):

        if sample_densities:
            delta_bias = delta_list[i]
            delta_bias_before = float(delta_bias)
            delta_bias = np.interp(delta_bias, delta_lin_values, delta_nonlin)
            delta_bias /= hmf_this.dicke()
#            hmf_this = chmf(z=z, delta_bias = delta_bias, R_bias = R_bias)        #TODO fix this still, it's in my to do list.
            masses, mass_func = hmf_this.run_hmf(delta_bias)
            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element==0:
                    break
            masses=masses[:index_to_stop]
            mass_func=mass_func[:index_to_stop]

            N_mean_cumsum = integrate_hmf.hmf_integral_gtm(masses[:index_to_stop], mass_func[:index_to_stop]) * 4/3 * np.pi *R_bias**3
            N_mean = int((N_mean_cumsum)[0])
            N_cumsum_normalized = N_mean_cumsum/N_mean_cumsum[0]

        if not sample_hmf:
            N_this_iter = N_mean
        else:
            if not calculate_2pcc:
                N_this_iter = np.random.poisson(N_mean)
            else:
                delta = delta_scc(hmf_this)
                beta = 1- np.sqrt(1+delta/N_mean)
                N_s = np.logspace(1, np.log(N_mean)+1)
                super_poisson = N_mean/np.sqrt(2*np.pi*N_s)*np.e**(N_s/2-N_mean(1-beta)-N_s*beta)\
                                *(1-beta)/N_s**(N_s/2)*(N_mean(1-beta)+N_s*beta)**(N_s-1)
                super_poisson_cumsum = integrate_hmf.hmf_integral_gtm(N_s, super_poisson, mass_density=False)
                super_poisson_cumsum = super_poisson_cumsum/super_poisson_cumsum[0]
                give_me_random = np.random.uniform(size=1)
                N_this_iter = np.interp(give_me_random, np.flip(np.array(super_poisson_cumsum)), np.flip(np.array(N_s)))

        random_numbers = np.random.uniform(size=N_this_iter)
        masses_of_haloes = np.zeros(shape = N_this_iter)
        if logger: start_interpolating =time.time()
        if interpolating:
            for index, rn in enumerate(random_numbers):
                masses_of_haloes[index] = np.interp(rn, np.flip(np.array(N_cumsum_normalized)), np.flip(np.array(masses)))
        else:
            for index, rn in enumerate(random_numbers):
                for k, cum_mass in enumerate(cumulative_mass):
                    if cum_mass>rn:
                        masses_of_haloes[index] =  masses[k]
                        break
        if logger:
            end_interpolating = time.time
            print("getting the masses took", end_interpolating - start_interpolating)
        masses_saved = []
        if duty_cycle:
            for index, mass in enumerate(masses_of_haloes):
                if np.random.binomial(1, np.exp(-M_turn/mass)):
                    masses_saved.append(mass)
        if logger:
            end_duty = time.time()
            print("implementing the duty cycle took", end_duty - end_interpolating)
        Lx = np.zeros(shape = N_this_iter)
        if logger:
            print(len(masses_saved))
        sfr_samples = np.zeros(shape=N_this_iter)
        if sample_Ms:
            ms_samples = np.zeros(shape=N_this_iter)
            Z_samples = np.zeros(shape=N_this_iter)

        for j,mass in enumerate(masses_saved):

            a_ms, b_ms, sigma_ms = give_me_sigma(means_Ms_relation, sigma_Ms_relation, a_Ms_relation, b_Ms_relation, mass) #need to fix this as it does smth weird

            if sample_SFR == 0 and sample_Ms:
                Ms_sample = 10**(np.random.normal((a_Ms_relation*np.log10(mass)+ b_Ms_relation), sigma_Ms_relation))
                a_sfr, b_sfr, sigma_sfr = give_me_sigma(means_SFR_relation, sigma_SFR_relation, a_SFR_relation, b_SFR_relation, Ms_sample)
                SFR_sample = 10**(np.random.normal((a_SFR_relation*np.log10(Ms_sample)+b_SFR_relation), sigma_SFR_relation))

            elif sample_SFR==1:
                Ms_sample = 10**(np.random.normal((a_Ms_relation*np.log10(mass)+ b_Ms_relation), sigma_Ms_relation))
                SFR_sample = 10**(np.random.normal(1.67*np.log10(Ms_sample)-15.365-1.67*np.log10(0.15), 0.2))

            elif sample_SFR==2 and sample_Ms:
                Ms_sample = 10**((a_Ms_relation*np.log10(mass)+ b_Ms_relation))
                a_sfr, b_sfr, sigma_sfr = give_me_sigma(means_SFR_relation, sigma_SFR_relation, a_SFR_relation, b_SFR_relation, Ms_sample)
                SFR_sample = 10**(a_SFR_relation*np.log10(Ms_sample)+b_SFR_relation)

            elif sample_SFR==3:
                Ms_sample = 10**((a_Ms_relation*np.log10(mass)+ b_Ms_relation))
                SFR_sample = 10**(1.67*np.log10(Ms_sample)-15.365-1.67*np.log(0.15))

            elif sample_SFR == 0 and not sample_Ms:
                SFR_sample = 10**(np.random.normal((a_Ms_relation*np.log10(mass)+b_Ms_relation), sigma_Ms_relation))
            elif sample_SFR == 2 and not sample_Ms:
                SFR_sample = 10**(a_Ms_relation*np.log10(mass)+b_Ms_relation)

            if sample_Lx==0 and sample_Ms:
                if sample_SFR==0 or sample_SFR==1:
                    pass
                else:
                    Ms_sample = 10**(a_Ms_relation*np.log10(mass)+ b_Ms_relation)

                Z_mean, sigma_Z = metalicity_from_FMR(Ms_sample, SFR_sample)
                Z_sample = 10**(np.random.normal((np.log10(Z_mean)), sigma_Z))
                logLx_over_sfr, logsigma_lx_over_sfr = give_me_logLxoverSFR(SFR_sample, Z_sample)
                Lx_sample = 10**np.random.normal(logLx_over_sfr + np.log10(SFR_sample),logsigma_lx_over_sfr)

            elif sample_Lx==1:
                Lx_sample = 10**np.random.normal((0.8*np.log10(SFR_sample)+39.40), 0.9)

            elif sample_Lx==2 and sample_Ms:
                if sample_SFR==2:
                    pass
                else:
                    Ms_sample = 10**(np.random.normal((a_Ms_relation*np.log10(mass)+ b_Ms_relation), sigma_Ms_relation))

                Z_sample, _ =metalicity_from_FMR(Ms_sample, SFR_sample)

                logLx_over_sfr, _ = give_me_logLxoverSFR(SFR_sample, Z_sample)
                Lx_sample = 10**(logLx_over_sfr + np.log10(SFR_sample))

            elif sample_Lx==3:
                Lx_sample = 10**(0.8*np.log10(SFR_sample)+39.40)

            elif sample_Lx == 0 and not sample_Ms:
                logLx_over_sfr, logsigma_lx_over_sfr = give_me_logLxoverSFR(SFR_sample)
                Lx_sample = 10**np.random.normal(logLx_over_sfr + np.log10(SFR_sample),logsigma_lx_over_sfr)

            elif sample_Lx == 2 and not sample_Ms:
                logLx_over_sfr, logsigma_lx_over_sfr = give_me_logLxoverSFR(SFR_sample)
                Lx_sample = 10**(logLx_over_sfr + np.log10(SFR_sample))

            Lx[j] = Lx_sample
            sfr_samples[j]=SFR_sample
            if sample_Ms:
                ms_samples[j]=Ms_sample
                Z_samples[j]=Z_sample

        if logger:
            end_sampling = time.time()
            print("sampling took", end_sampling - end_duty)
        emissivities[i] = np.sum (Lx)
        if np.round(i/N_iter*100000)/1000%10 == 0:
            print("done with {} % of the data".format(math.ceil(i/N_iter*100)))
    return emissivities, ms_samples, sfr_samples,Lx, logsigma_lx_over_sfr, sigma_SFR_relation#, Z_samples

class chmf:
    def z_drag_calculate(self):
        z_drag = 0.313*(self.omhh**-0.419) * (1 + 0.607*(self.omhh** 0.674));

        z_drag *= 1291 * self.omhh**0.251 / (1 + 0.659*self.omhh**0.828);
        return z_drag

    def alpha_nu_calculation(self):
        alpha_nu = (self.f_c/self.f_cb) * (2*(self.p_c+self.p_cb)+5)/(4*self.p_cb+5.0)
        alpha_nu *= 1 - 0.553*self.f_nub+0.126*(self.f_nub**3);
        alpha_nu /= 1-0.193*np.sqrt(self.f_nu)+0.169*self.f_nu;
        alpha_nu *= (1+self.y_d)**(self.p_c-self.p_cb);
        alpha_nu *= 1+ (self.p_cb-self.p_c)/2.0 * (1.0+1.0/(4.0*self.p_c+3.0)/(4.0*self.p_cb+7.0))/(1.0+self.y_d)
        return alpha_nu

    def MtoR(self,M):
        if (self.FILTER == 0): ##top hat M = (4/3) PI <rho> R^3
            return ((3*M/(4*np.pi*cosmo.Om0*self.critical_density))**(1.0/3.0))

    def RtoM(self, R):
        if (self.FILTER == 0):
            return ((4.0/3.0)*np.pi*R**3*(cosmo.Om0*self.critical_density))

    def __init__(self, z, delta_bias, R_bias):

        self.CMperMPC = 3.086e24
        self.Msun = 1.989e33
        self.TINY = 10**-30
        self.Deltac = 1.686
        self.FILTER = 0
        self.T_cmb = cosmo.Tcmb0.value
        self.theta_cmb = self.T_cmb /2.7
        self.critical_density = cosmo.critical_density0.value*self.CMperMPC*self.CMperMPC*self.CMperMPC/self.Msun
        self.z = z
        self.delta_bias = delta_bias
        self.R_bias = R_bias
        self.omhh=cosmo.Om0*(cosmo.h)**2
        self.z_equality = 25000*self.omhh*self.theta_cmb**-4 - 1.0
        self.k_equality = 0.0746*self.omhh/(self.theta_cmb**2)
        self.z_drag = self.z_drag_calculate()
        self.y_d = (1 + self.z_equality) / (1.0 + self.z_drag)
        self.f_nu = cosmo.Onu0 /cosmo.Om0
        self.f_baryon = cosmo.Ob0 / cosmo.Om0
        self.p_c = -(5 - np.sqrt(1 + 24*(1 - self.f_nu-self.f_baryon)))/4.0
        self.p_cb = -(5 - np.sqrt(1 + 24*(1 - self.f_nu)))/4.0
        self.f_c = 1 - self.f_nu - self.f_baryon
        self.f_cb = 1 - self.f_nu
        self.f_nub = self.f_nu+self.f_baryon
        self.alpha_nu = self.alpha_nu_calculation()
        self.R_drag = 31.5 * cosmo.Ob0*cosmo.h**2 * (self.theta_cmb**-4) * 1000 / (1.0 + self.z_drag)
        self.R_equality = 31.5 * cosmo.Ob0*cosmo.h**2 * (self.theta_cmb**-4) * 1000 / (1.0 + self.z_equality)
        self.sound_horizon = 2.0/3.0/self.k_equality * np.sqrt(6.0/self.R_equality) *np.log( (np.sqrt(1+self.R_drag) \
                    + np.sqrt(self.R_drag+self.R_equality)) / (1.0 + np.sqrt(self.R_equality)) )
        self.beta_c = 1.0/(1.0-0.949*self.f_nub)
        self.N_nu = (1.0)
        self.POWER_INDEX = 0.9667
        self.Radius_8 = 8.0/cosmo.h
        self.SIGMA_8 = 0.8159
        self.M_bias = self.RtoM(self.R_bias)

    def dicke(self):

        OmegaM_z=cosmo.Om(self.z)
        dick_z = 2.5*OmegaM_z / ( 1.0/70.0 + OmegaM_z*(209-OmegaM_z)/140.0 + pow(OmegaM_z, 4.0/7.0) )
        dick_0 = 2.5*cosmo.Om0 / ( 1.0/70.0 + cosmo.Om0*(209-cosmo.Om0)/140.0 + pow(cosmo.Om0, 4.0/7.0) )
        return dick_z / (dick_0 * (1.0+self.z))

    def TFmdm(self,k):
        q = k*self.theta_cmb**2/self.omhh
        gamma_eff=np.sqrt(self.alpha_nu) + (1.0-np.sqrt(self.alpha_nu))/(1.0+(0.43*k*self.sound_horizon)** 4)
        q_eff = q/gamma_eff
        TF_m= np.log(np.e+1.84*self.beta_c*np.sqrt(self.alpha_nu)*q_eff)
        TF_m /= TF_m + q_eff**2 * (14.4 + 325.0/(1.0+60.5*(q_eff**1.11)))
        q_nu = 3.92*q/np.sqrt(self.f_nu/self.N_nu)
        TF_m *= 1.0 + (1.2*(self.f_nu**0.64)*(self.N_nu**(0.3+0.6*self.f_nu))) /((q_nu**-1.6)+(q_nu**0.8))

        return TF_m

    def dsigma_dk(self, k, R):

        T = self.TFmdm(k)
        p = k**self.POWER_INDEX * T * T
        kR = k*R

        #if ( (kR) < 1.0e-4 ):
        #    w = 1.0
        #else:
        w = 3.0 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        return k*k*p*w*w

    def sigma_norm(self,):
        kstart = 1.0*(10**-99)/self.Radius_8
        kend = 350.0/self.Radius_8

        lower_limit = kstart
        upper_limit = kend

        result = integrate.quad(self.dsigma_dk, 0, np.inf, args=(self.Radius_8,), limit=1000, epsabs=10**-20)[0]
        return self.SIGMA_8/np.sqrt(result)

    def sigma_z0(self, M):

        Radius = self.MtoR(M);
        kstart = 1.0*10**(-99)/Radius;
        kend = 350.0/Radius;

        lower_limit = kstart
        upper_limit = kend

        integral = integrate.quad(self.dsigma_dk, 0, np.inf, args=(Radius,), limit=1000, epsabs=10**-20)
        result = integral[0]

        return self.sigma_norm() * np.sqrt(result)

    def dsigmasq_dm(self, k, R):

        T = self.TFmdm(k);
        p = k**self.POWER_INDEX * T * T;

        kR = k * R;
        #if ( (kR) < 1.0*10**(-4) ):
        #    w = 1.0;
        #else:
        w = 3.0 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)


        #if ( (kR) < 1.0*10**(-10) ):
        #    dwdr = 0
        #else:
        dwdr = 9*np.cos(kR)*k/kR**3 + 3*np.sin(kR)*(1 - 3/(kR*kR))/(kR*R)
        drdm = 1.0 / (4.0*np.pi *cosmo.Om0* self.critical_density * R*R);

        return k*k*p*2*w*dwdr*drdm;

    def dsigmasqdm_z0(self, M):
        Radius = self.MtoR(M)

        kstart = 1.0e-99/Radius;
        kend = 350.0/Radius;

        lower_limit = kstart
        upper_limit = kend
        result = integrate.quad (self.dsigmasq_dm, 0, np.inf, args=(Radius,), limit=1000, epsabs=10**-20)
        return self.sigma_norm() * self.sigma_norm() * result[0]



    def dnbiasdM( self, M):
        if ((self.M_bias-M) < self.TINY):
            #print("Mass of the halo bigger than the overdensity mass, not good, stopping now and returing 0")
            return(0)
        delta = self.Deltac/self.dicke() - self.delta_bias

        sig_o = self.sigma_z0(self.M_bias);
        sig_one = self.sigma_z0(M);
        sigsq = sig_one*sig_one - sig_o*sig_o;
        return -(self.critical_density*cosmo.Om0)/M /np.sqrt(2*np.pi) \
            *delta*((sig_one**2 - sig_o**2)**(-1.5))*(np.e**( -0.5*delta**2/(sig_one**2-sig_o**2))) \
            *self.dsigmasqdm_z0(M)

    def prep_for_hmf(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01):
        self.log10_Mmin = log10_Mmin
        self.log10_Mmax = log10_Mmax
        self.dlog10m = dlog10m
        self.bins = 10 ** np.arange(self.log10_Mmin, self.log10_Mmax, self.dlog10m)
        self.sigma_z0_array = np.zeros(len(self.bins))
        self.sigma_derivatives = np.zeros(len(self.bins))
        for index, mass in enumerate(self.bins):
            self.sigma_z0_array[index] = self.sigma_z0(mass)
            self.sigma_derivatives[index] = self.dsigmasqdm_z0(mass)

    def run_hmf(self, delta_bias):
        delta = self.Deltac/self.dicke() - delta_bias
        sigma_array = self.sigma_z0_array**2 - self.sigma_cell()**2
        self.hmf = np.zeros(len(self.bins))
        if delta<0:
            print("Something went wrong, cell overdensity bigger than collapse\n")
            print("this cell collapsed already, returning 0")
            return 0
        for index, mass in enumerate(self.bins):
            if mass<self.M_bias:
                self.hmf[index] = -(self.critical_density*cosmo.Om0)/mass/np.sqrt(2*np.pi) * delta * ((sigma_array[index])**(-1.5)) * \
                                (np.e**(-0.5*delta**2/(sigma_array[index]))) * self.sigma_derivatives[index]
            else:
                self.hmf[index] = 0.0
        return self.bins, self.hmf

    def dndMnormal( self, M):
        if ((self.M_bias-M) < self.TINY):
            #print("Mass of the halo bigger than the overdensity mass, not good, stopping now and returing 0")
            return(0)
        delta = self.Deltac/self.dicke()
        sig_one = self.sigma_z0(M);
        sigsq = sig_one*sig_one
        return -(self.critical_density*cosmo.Om0)/M /np.sqrt(2*np.pi) \
            *delta*((sig_one**2)**(-1.5))*(np.e**( -0.5*delta**2/(sig_one**2))) \
            *self.dsigmasqdm_z0(M)

    def sigma_cell(self):
        return self.sigma_z0(self.M_bias)

#     def run_hmf(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01 ):
#         self.log10_Mmin = log10_Mmin
#         self.log10_Mmax = log10_Mmax
#         self.dlog10m = dlog10m
#         self.bins = 10 ** np.arange(self.log10_Mmin, self.log10_Mmax, self.dlog10m)
#         self.hmf = np.zeros(len(self.bins))
#         for i, mass in enumerate(self.bins):
#             self.hmf[i] = self.dnbiasdM(mass * cosmo.h)
#         return (self.bins, self.hmf)

    def run_hmf_normal(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01 ):
        self.log10_Mmin_normal = log10_Mmin
        self.log10_Mmax_normal = log10_Mmax
        self.dlog10m_normal = dlog10m
        self.bins_normal = 10 ** np.arange(self.log10_Mmin_normal, self.log10_Mmax_normal, self.dlog10m_normal)
        self.hmf_normal = np.zeros(len(self.bins_normal))
        for i, mass in enumerate(self.bins_normal):
            self.hmf_normal[i] = self.dndMnormal(mass)
        return (self.bins_normal, self.hmf_normal)

    def cumulative_number(self):
        dndlnm = self.bins * self.hmf
        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]), np.log(10 ** 18), np.log(self.bins[1]) - np.log(self.bins[0])
            )
            mf_func = _spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)

            int_upper = integrate.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even="first")
        else:
            int_upper = 0

        # Calculate the cumulative integral (backwards) of [m*]dndlnm
        self.cumnum = np.concatenate(
            (
                integrate.cumtrapz(dndlnm[::-1], dx=np.log(self.bins[1]) - np.log(self.bins[0]))[::-1],
                np.zeros(1),
            )
        )
        self.cumnum+=int_upper
        return self.cumnum

    def cumulative_mass(self):
        dndlnm = self.bins * self.hmf

        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]), np.log(10 ** 18), np.log(self.bins[1]) - np.log(self.bins[0])
            )
            mf_func = _spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)
            int_upper = integrate.simps(
                np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even="first"
            )
        else:
            int_upper = 0

        self.cummass = np.concatenate(
            (
                integrate.cumtrapz(self.bins[::-1] * dndlnm[::-1], dx=np.log(self.bins[1]) - np.log(self.bins[0]))[
                    ::-1
                ],
                np.zeros(1),
            )
        )
        self.cummass += int_upper
        return self.cummass

def SFRvsMh(z, sample_Ms = True):
    ff=open('FirstLight_database.dat', 'r')
    database = json.load(ff)
    ff.close()
    extract_keys_base=[ 'FL633', 'FL634', 'FL635', 'FL637', 'FL638', 'FL639', 'FL640', 'FL641', 'FL642', 'FL643', 'FL644', 'FL645', 'FL646',
                       'FL647', 'FL648', 'FL649', 'FL650', 'FL651', 'FL653', 'FL655', 'FL656', 'FL657', 'FL658', 'FL659', 'FL660', 'FL661',
                       'FL662', 'FL664', 'FL665', 'FL666', 'FL667', 'FL668', 'FL669', 'FL670', 'FL671', 'FL673', 'FL675', 'FL676', 'FL678',
                       'FL679', 'FL680', 'FL681', 'FL682', 'FL683', 'FL684', 'FL685', 'FL686', 'FL687', 'FL688', 'FL689', 'FL690', 'FL691',
                       'FL692', 'FL693', 'FL694', 'FL695', 'FL696', 'FL697', 'FL698', 'FL700', 'FL701', 'FL702', 'FL703', 'FL704', 'FL705',
                       'FL706', 'FL707', 'FL708', 'FL709', 'FL710', 'FL711', 'FL712', 'FL713', 'FL714', 'FL715', 'FL716', 'FL717', 'FL718',
                       'FL719', 'FL720', 'FL721', 'FL722', 'FL723', 'FL724', 'FL725', 'FL726', 'FL727', 'FL728', 'FL729', 'FL730', 'FL731',
                       'FL732', 'FL733', 'FL734', 'FL735', 'FL736', 'FL739', 'FL740', 'FL744', 'FL745', 'FL746', 'FL747', 'FL748', 'FL750',
                       'FL752', 'FL753', 'FL754', 'FL755', 'FL756', 'FL757', 'FL758', 'FL760', 'FL761', 'FL763', 'FL767', 'FL768', 'FL769',
                       'FL770', 'FL771', 'FL772', 'FL773', 'FL774', 'FL775', 'FL777', 'FL778', 'FL779', 'FL780', 'FL781', 'FL782', 'FL783',
                       'FL784', 'FL788', 'FL789', 'FL792', 'FL793', 'FL794', 'FL796', 'FL800', 'FL801', 'FL802', 'FL803', 'FL805', 'FL806',
                       'FL808', 'FL809', 'FL810', 'FL811', 'FL814', 'FL815', 'FL816', 'FL818', 'FL819', 'FL823', 'FL824', 'FL825', 'FL826',
                       'FL827', 'FL829',         'FL652', 'FL654', 'FL699', 'FL738', 'FL741', 'FL742', 'FL749', 'FL762', 'FL764', 'FL766',
                       'FL776', 'FL785', 'FL786', 'FL787', 'FL790', 'FL795', 'FL797', 'FL798', 'FL799', 'FL804', 'FL807', 'FL812', 'FL813',
                       'FL817', 'FL820', 'FL822', 'FL828', 'FL839', 'FL850', 'FL855', 'FL862', 'FL870', 'FL874', 'FL879', 'FL882', 'FL891',
                       'FL896', 'FL898', 'FL903', 'FL904', 'FL906', 'FL915', 'FL916', 'FL917', 'FL918', 'FL919', 'FL921', 'FL922', 'FL923',
                       'FL924', 'FL925', 'FL926', 'FL928', 'FL930', 'FL932', 'FL934', 'FL935', 'FL765', 'FL835', 'FL836', 'FL841', 'FL843',
                       'FL844', 'FL845', 'FL846', 'FL847', 'FL848', 'FL849', 'FL851', 'FL856', 'FL857', 'FL858', 'FL859', 'FL860', 'FL861',
                       'FL865', 'FL866', 'FL867', 'FL868', 'FL871', 'FL877', 'FL881', 'FL884', 'FL885', 'FL887', 'FL888', 'FL893', 'FL895',
                       'FL897', 'FL899', 'FL900', 'FL901', 'FL907', 'FL908', 'FL909', 'FL911', 'FL912', 'FL837', 'FL838', 'FL840', 'FL852',
                       'FL853', 'FL863', 'FL864', 'FL869', 'FL872', 'FL873', 'FL883', 'FL890', 'FL894', 'FL902', 'FL834', 'FL854', 'FL876',
                       'FL913', 'FL927', 'FL931', 'FL933', 'FL938', 'FL939', 'FL940', 'FL942', 'FL943', 'FL946', 'FL947', 'FL914', 'FL920',
                       'FL929', 'FL936', 'FL937', 'FL941', 'FL944' ]
    extract_a='a'
    extract_run='run'
    extract_field1='Mvir'
    extract_field2='Ms'
    extract_field3='SFR'
    s='_'
    possible_a = np.array([0.059, 0.060, 0.062, 0.064, 0.067, 0.069, 0.071, 0.074, 0.077, 0.080, 0.083, 0.087, 0.090, 0.095, 0.1, 0.105, 0.111, 0.118, 0.125,
                 0.133, 0.142, 0.154, 0.160])
    possible_a_str = list(map("{:.3f}".format, possible_a))
    z_poss = 1./possible_a - 1
    z_actual = min(z_poss, key=lambda x:abs(x-z))
    #print(possible_a_str,)
    index_actual = np.argmin(abs(np.array(z_poss)-z))
    #print(possible_a_str[index_actual])
    Ms = []
    Mh = []
    SFR = []
    for j in range(len(possible_a_str)):
        extract_a_value = possible_a_str[index_actual]
        for i in range(len(extract_keys_base)):
            extract_keys_run= extract_keys_base[i] + s + extract_a_value + s + extract_run
            extract_keys_a= extract_keys_base[i] + s + extract_a_value + s + extract_a
            extract_keys_field1= extract_keys_base[i] + s + extract_a_value + s + extract_field1
            extract_keys_field2= extract_keys_base[i] + s + extract_a_value + s + extract_field2
            extract_keys_field3= extract_keys_base[i] + s + extract_a_value + s + extract_field3
            if ((extract_keys_field1) in database) and ((extract_keys_field2) in database):
                #print(database[extract_keys_a], database[extract_keys_run], "%.2e" % database[extract_keys_field1],
                #      "%.2e" % database[extract_keys_field2], database[extract_keys_field3])
                Mh.append(database[extract_keys_field1])
                Ms.append(database[extract_keys_field2])
                SFR.append(database[extract_keys_field3])
#    return Mh, Ms, SFR
    Mh_uniq=set()
    Ms_uniq=set()
    SFR_uniq=set()
    Mh_uniq_list = []
    SFR_uniq_list = []
    Ms_uniq_list = []
    for i, mh in enumerate(Mh):
        if (mh, SFR[i], Ms[i]) not in Mh_uniq:
            Mh_uniq.add((mh, SFR[i], Ms[i]))
            Mh_uniq_list.append(mh)
            SFR_uniq_list.append(SFR[i])
            Ms_uniq_list.append(Ms[i])
    zipped = zip(Mh_uniq_list,SFR_uniq_list, Ms_uniq_list)
    zipped = sorted(zipped)
    Mh_uniq_sorted, SFR_uniq_sorted, Ms_uniq_sorted = zip(*zipped)
#    print(Mh_uniq_sorted, SFR_uniq_sorted)
    if sample_Ms:
        a_ms, b_ms = np.polyfit (np.log10(Mh_uniq_sorted), np.log10(Ms_uniq_sorted), 1 )
        bins = 2
        binned_masses=np.linspace(np.log10(min(Mh_uniq_sorted)),np.log10(max(Mh_uniq_sorted)), bins)
        means_ms=np.zeros(bins-1)
        number_of_masses=np.zeros(bins-1)
        errs_ms = np.zeros(bins-1)

        for i, mass in enumerate(Mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if j<(bins-2) and (np.log10(mass)>=bining) and (np.log10(mass)<binned_masses[j+1]):
                    err_ms[j]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(Ms_uniq_sorted[i]))**2)
                    number_of_masses[j]+=1
                    means_ms[j]+=(mass)
                    break
                elif(j==bins-2):
                    errs_ms[bins-2]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(Ms_uniq_sorted[i]))**2)
                    number_of_masses[bins-2]+=1
                    means_ms[bins-2]+=(mass)
                    break
        errs_ms=np.sqrt(errs_ms/number_of_masses)
        means_ms=np.sqrt(means_ms/number_of_masses)
        #do the same for Ms-SFR

        a_sfr, b_sfr = np.polyfit (np.log10(Ms_uniq_sorted), np.log10(SFR_uniq_sorted), 1 )
        binned_stellar_masses=np.linspace(np.log10(min(Ms_uniq_sorted)),np.log10(max(Ms_uniq_sorted)), bins)
        means_sfr=np.zeros(bins-1)
        errs_sfr = np.zeros(bins-1)
        number_of_stellar_masses = np.zeros(bins-1)
        for i, stellar_mass in enumerate(Ms_uniq_sorted):
            for j, stellar_bining in enumerate(binned_stellar_masses):
                if j<(bins-2) and (np.log10(stellar_mass)>=stellar_bining) and (np.log10(stellar_mass)<binned_stellar_masses[j+1]):
                    err_sfr[j]+=(((a_sfr*np.log10(stellar_mass)+b_sfr) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_stellar_masses[j]+=1
                    means_sfr[j]+=(stellar_mass)
                    break
                elif(j==bins-2):
                    errs_sfr[bins-2]+=(((a_sfr*np.log10(stellar_mass)+b_sfr) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_stellar_masses[bins-2]+=1
                    means_sfr[bins-2]+=(stellar_mass)
                    break
        errs_sfr = np.sqrt(errs_sfr/number_of_stellar_masses)
        means_sfr = np.sqrt(means_sfr/number_of_stellar_masses)
        return (means_ms, errs_ms, a_ms,b_ms, means_sfr, errs_sfr, a_sfr, b_sfr)

    else:
        a_ms, b_ms = np.polyfit (np.log10(Mh_uniq_sorted), np.log10(SFR_uniq_sorted), 1 )
        bins = 2
        binned_masses=np.linspace(np.log10(min(Mh_uniq_sorted)),np.log10(max(Mh_uniq_sorted)), bins)
        means_ms=np.zeros(bins-1)
        number_of_masses=np.zeros(bins-1)
        errs_ms = np.zeros(bins-1)

        for i, mass in enumerate(Mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if j<(bins-2) and (np.log10(mass)>=bining) and (np.log10(mass)<binned_masses[j+1]):
                    err_ms[j]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_masses[j]+=1
                    means_ms[j]+=(mass)
                    break
                elif(j==bins-2):
                    errs_ms[bins-2]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_masses[bins-2]+=1
                    means_ms[bins-2]+=(mass)
                    break
        errs_ms=np.sqrt(errs_ms/number_of_masses)
        means_ms=np.sqrt(means_ms/number_of_masses)

        return (means_ms, errs_ms, a_ms,b_ms)

def give_me_sigma(mass,means, errs, a, b): ##binning is off and only mean is given. Returns all the parameters for the relation.
        if len(means)==1:
            return a,b, errs
        else:
            return a,b,np.interp(np.log10(mass),np.log10(means), np.sqrt(np.array(errs)))

def give_me_logLxoverSFR(SFR_data, Z_sample=None):

    if Z_sample:
        indices = [7.0,7.2,7.4,7.6,7.8,8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2]

    SFRdot01_x = [38.16, 38.16, 38.16, 38.16, 38.16, 38.15, 38.15, 38.15, 38.14, 38.14, 38.14, 38.14]
    SFRdot01_s = [0.75, 0.77, 0.77, 0.77, 0.74, 0.73, 0.72, 0.70, 0.69, 0.69, 0.68, 0.67]
    SFRdot1_x = [38.82, 38.89, 38.93, 38.89, 38.83, 38.77, 38.71, 38.67, 38.65, 38.62, 38.62, 38.60]
    SFRdot1_s = [1.53, 1.56, 1.59, 1.64, 1.65,1.58, 1.37, 1.00, 0.75, 0.64, 0.58, 0.55]
    SFR1_x = [39.92, 40.04, 40.11, 40.13, 40.09, 40.00, 39.85, 39.63, 39.37, 39.14, 39.01, 38.95]
    SFR1_s = [0.31, 0.30, 0.31, 0.33, 0.38, 0.46, 0.59, 0.67, 0.57, 0.59, 0.49, 0.38]
    SFR10_x = [39.95, 40.07, 40.14, 40.16, 40.14, 40.07, 39.96, 39.81, 39.64, 39.46, 36.28, 39.13]
    SFR10_s = [0.08, 0.08, 0.08, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19, 0.20, 0.22, 0.20]
    SFR100_x = [39.96, 40.07, 40.14, 40.17, 40.14, 40.07, 39.97, 39.83, 39.67, 39.50, 39.33, 39.19]
    SFR100_s = [0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.07]

    if Z_sample:
        SFRdot01_x_sample = np.interp(Z_sample, indices, SFRdot01_x)
        SFRdot01_s_sample = np.interp(Z_sample, indices, SFRdot01_s)
        SFRdot1_x_sample = np.interp(Z_sample, indices, SFRdot1_x)
        SFRdot1_s_sample = np.interp(Z_sample, indices, SFRdot1_s)
        SFR1_x_sample = np.interp(Z_sample, indices, SFR1_x)
        SFR1_s_sample = np.interp(Z_sample, indices, SFR1_s)
        SFR10_x_sample = np.interp(Z_sample, indices, SFR10_x)
        SFR10_s_sample = np.interp(Z_sample, indices, SFR10_s)
        SFR100_x_sample = np.interp(Z_sample, indices, SFR100_x)
        SFR100_s_sample = np.interp(Z_sample, indices, SFR100_s)

    if Z_sample is None:
        SFRdot01_meanx = np.mean(SFRdot01_x)
        SFRdot1_meanx = np.mean(SFRdot1_x)
        SFR1_meanx = np.mean(SFR1_x)
        SFR10_meanx = np.mean(SFR10_x)
        SFR100_meanx = np.mean(SFR100_x)

        SFRdot01_means = np.mean(SFRdot01_s)
        SFRdot1_means = np.mean(SFRdot1_s)
        SFR1_means = np.mean(SFR1_s)
        SFR10_means = np.mean(SFR10_s)
        SFR100_means = np.mean(SFR100_s)

        SFRdot01_tots = np.sqrt(SFRdot01_means**2+np.std(SFRdot01_x)**2)
        SFRdot1_tots = np.sqrt(SFRdot1_means**2+np.std(SFRdot1_x)**2)
        SFR1_tots = np.sqrt(SFR1_means**2+np.std(SFR1_x)**2)
        SFR10_tots = np.sqrt(SFR10_means**2+np.std(SFR10_x)**2)
        SFR100_tots = np.sqrt(SFR100_means**2+np.std(SFR100_x)**2)

    SFR = [0.01, 0.1, 1. , 10.0, 100.0]
    if Z_sample is None:
        values = [SFRdot01_meanx, SFRdot1_meanx, SFR1_meanx, SFR10_meanx, SFR100_meanx]
        values_sigma = [SFRdot01_tots, SFRdot1_tots, SFR1_tots, SFR10_tots, SFR100_tots]

    if Z_sample:
        if hasattr(SFRdot01_x_sample, '__len__'):
            values = [SFRdot01_x_sample[0], SFRdot1_x_sample[0], SFR1_x_sample[0], SFR10_x_sample[0], SFR100_x_sample[0]]
            values_sigma = [SFRdot01_s_sample[0], SFRdot1_s_sample[0], SFR1_s_sample[0], SFR10_s_sample[0], SFR100_s_sample[0]]
        else:
            values = [SFRdot01_x_sample, SFRdot1_x_sample, SFR1_x_sample, SFR10_x_sample, SFR100_x_sample]
            values_sigma = [SFRdot01_s_sample, SFRdot1_s_sample, SFR1_s_sample, SFR10_s_sample, SFR100_s_sample]

    logLxoverSFR = np.interp(SFR_data, SFR, values)
    logsigma = np.interp(SFR_data, SFR, values_sigma)

    return logLxoverSFR, logsigma

def bias_function(M, hmf_class):
    a_bias = 0.707
    b_bias = 0.5
    c_bias = 0.6
    nu = hmf_class.Deltac/hmf_class.dicke()/np.sqrt(hmf_class.sigma_z0(M))
    nu_1 = nu * np.sqrt(a_bias)
    return 1+ 1/(hmf_class.Deltac/hmf_class.dicke()) * (nu_1**2  + b_bias*nu_1**(2*(1-c_bias)) - (nu_1**(2*c_bias) / \
            np.sqrt(a_bias))/(nu_1**(2*c_bias)+b_bias*(1-c_bias)*(1-c_bias/2)))

def bias_func(M, hmf_class):  #this supposes that the hmf class was initiated
    hmf_value = hmf_class.dnbiasdM(M)
    bias_value = bias_function(M, hmf_class)
    return M*hmf_value*bias_value

def delta_scc(hmf_class):  #could add manual calculation of N_mean and sigma_cell_sq (these are already calculated in any case)
    bias_integral = integrate.quad(bias_func, 10**hmf_class.log10_Mmin, 10**hmf_class.log10_Mmax, args=(hmf_class,))[0]
    return (hmf_class.cumulative_number()[0]/critical_density)**2 * hmf_class.sigma_cell()**2 * (bias_integral)**2

def metalicity_from_FMR ( M_star, SFR):
    """
    metalicity from Curti+19

    -----

    Function takes in the stellar mass and SFR and outputs the metallicity 12+log(O/H)
    """
    sigma_met = 0.054  #value they qoute for the sigma_FMR
    Z_0 = 8.779
    gamma = 0.31
    beta = 2.1
    m_0 = 10.11
    m_1 = 0.56
    M_0 = 10**(m_0 + m_1 * np.log10(SFR))
    return (Z_0 - gamma/beta * np.log10(1+(M_star/M_0)**(-beta))), sigma_met

emissivities,_,_,_,_ = Sampler(N_iter = 5000)
np.save('emissivities_z10_full.npy', np.array(emissivities))
