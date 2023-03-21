"""Contains all necessary ingredients for manipulating bpass files"""
from scipy.optimize import curve_fit
import numpy as np
from scipy.integrate import simpson as simps
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from multiprocessing import Pool
from astropy import constants as const

from scaling import OH_to_mass_fraction
from common_stuff import get_SFH_stoch_const, get_SFH_exp, SFH_sampler

def wv_to_freq(wvs):
    """
    Converts a wavelength to frequency.
    Input
    ----------
        wvs: scalar or ndarray-like.
            wavelengths in Angstroms.
    Output:
        freq: scalar of ndarray-like.
            frequencies in Herz.
    """

    return const.c.cgs.value / (wvs * 1e-8)

def reader(name):
    """
    Function that reads and splits a string into separate strings.
    """
    return open(name).read().split('\n')

def splitter(fp):
    """Function that splits SEDs. Useful for parallelizing."""
    wv_b = int(1e5)
    return [[float(fp[i].split()[j]) for i in range(wv_b)] for j in range(1,52)]

class bpass_loader:
    """
    This Class contains all of the properties calculated using BPASS. Class
    structure is used to improve the speed.
    """
    def __init__(self, parallel = None, filename = '/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-imf135_300.a+00.'):
        """
        Input
        ----------
        parallel : boolean,
            Whether first processing is parallelized.
        filename : string,
            Which BPASS file is used.
        """
        self.metal_avail = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004, 
                                     0.006, 0.008, 0.01, 0.14, 0.02, 0.03, 
                                     0.04])
        self.metal_avail_names = ['zem5', 'zem4', 'z001', 'z002', 'z003', 
                                  'z004', 'z006', 'z008', 'z010', 'z014', 
                                  'z020', 'z030', 'z040']
        
        
        self.SEDS_raw = []
        
        names = []
        for index, metal_name in enumerate(self.metal_avail_names):
            names.append(filename + metal_name + '.dat')
        
        if parallel:
            pool = Pool(parallel)
            self.SEDS_raw = pool.map(reader, names)
        else:
            for index, name in enumerate(self.metal_avail_names):
                self.SEDS_raw.append(open(filename + name + '.dat').read().split('\n'))
            
        self.wv_b = len(self.SEDS_raw[0])-1
        self.wv = np.linspace(1,1e5+1, self.wv_b+1)
        if parallel:
            pool = Pool(parallel)
            self.SEDS = pool.map(splitter, self.SEDS_raw)
            self.SEDS = np.array(self.SEDS)
        else:
            self.SEDS = np.array([[[float(fp[i].split()[j]) for i in range(self.wv_b)] for j in range(1,52)] for fp in self.SEDS_raw])
        self.ages = 52
        self.ag = np.array([0] + [10**(6.05 + 0.1 * i) for i in range(1,52)])
        
        self.t_star = 0.36

    def get_UV(self, metal, Mstar, SFR, z, SFH_samp = None):
        """
        Function returs the specific luminosity at 1500 angstroms averaged over
        100 angstroms.
        Input
        ----------
            metal : float,
                Metallicity of the galaxy.
            Mstar : float,
                Stellar mass of the galaxy.
            SFR : float,
                Star formation rate of the galaxy.
            z : float,
                redshift of observation.
            SFH_samp : boolean,
                whether SFH is sampled or it's given by previous properties.
                So far sampling does nothing so it's all the same.
        Output
        ----------
            UV_final : float,
                UV luminosity in ergs Hz^-1
        """
        metal = OH_to_mass_fraction(metal)

        #to get solar metalicity need to take 0.42 according to Strom+18

        metal = metal / 10**0.42

        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]
        
        SEDp = self.SEDS[i-1]
        SEDn = self.SEDS[i]

        try:
            self.SFH
        except AttributeError:

            if SFH_samp is None:
                SFH_short, self.index_age =  get_SFH_exp(Mstar, SFR, z)
            else:
                SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages-1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 10**6
        
        wv_UV = self.wv[1450:1550]
        UV_p = np.zeros(self.ages-1)
        UV_n = np.zeros(self.ages-1)
        for i in range(self.ages-1):
            UV_p[i] = np.sum(np.array(SEDp[i][1449:1549]))/100 * self.SFH[i] * (self.ag[i+1]-self.ag[i]) * (1/const.c.cgs.value * 1500**2 * 1e-8)
            UV_n[i] = np.sum(np.array(SEDn[i][1449:1549]))/100 * self.SFH[i] * (self.ag[i+1]-self.ag[i]) * (1/const.c.cgs.value * 1500**2 * 1e-8)
        
        #if index_age!=(self.ages-1):
        #    missing_piecep = np.interp(self.ag[index_age], self.ag[1:], UV_p, right=0)
        #    missing_piecen = np.interp(self.ag[index_age], self.ag[1:], UV_n, right=0)

        #    UV_p_to_sum = np.append(UV_p[:ages_UV], missing_piecep)
        #    UV_n_to_sum = np.append(UV_n[:ages_UV], missing_piecen)
        
        FUV_p = np.sum(UV_p)
        FUV_n = np.sum(UV_n)
        
        UV_final = np.interp(metal, [met_prev, met_next], [FUV_p, FUV_n])

        #a_UV = (FUV_n - FUV_p) / (met_next - met_prev)
        #b_UV = (- FUV_n * met_prev + FUV_p * met_next) / (met_next - met_prev)
        
        return UV_final
    
    def get_LyC(self, metal, Mstar, SFR, z, SFH_samp = None):
        """
                Function returs the specific luminosity at 912 angstroms.
                Input
                ----------
                    metal : float,
                        Metallicity of the galaxy.
                    Mstar : float,
                        Stellar mass of the galaxy.
                    SFR : float,
                        Star formation rate of the galaxy.
                    z : float,
                        redshift of observation.
                    SFH_samp : boolean,
                        whether SFH is sampled or it's given by previous properties.
                        So far sampling does nothing so it's all the same.
                Output
                ----------
                    LyC_final : float,
                        Lyc luminosity in ergs Hz^-1
                """

        metal = OH_to_mass_fraction(metal)

        metal = metal / 10 ** 0.42 #see above

        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]

        SEDp = self.SEDS[i-1]
        SEDn = self.SEDS[i]

        try:
            self.SFH
        except AttributeError:

            if SFH_samp is None:
                SFH_short, self.index_age = get_SFH_exp(Mstar, SFR, z)
            else:
                SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 10 ** 6

        LyC_p = np.zeros(self.ages-1)
        LyC_n = np.zeros(self.ages-1)
        for i in range(self.ages-1):
            LyC_p[i] = np.array(SEDp[i][911]) * self.SFH[i] * (self.ag[i+1]-self.ag[i]) * (1/const.c.cgs.value * 911**2 * 1e-8)
            LyC_n[i] = np.array(SEDn[i][911]) * self.SFH[i] * (self.ag[i+1]-self.ag[i]) * (1/const.c.cgs.value * 911**2 * 1e-8)

        #if index_age!=(self.ages-1):
        #    missing_piecep = np.interp(self.ag[index_age], self.ag[1:], UV_p, right=0)
        #    missing_piecen = np.interp(self.ag[index_age], self.ag[1:], UV_n, right=0)

        #    UV_p_to_sum = np.append(UV_p[:ages_UV], missing_piecep)
        #    UV_n_to_sum = np.append(UV_n[:ages_UV], missing_piecen)

        FLyC_p = np.sum(LyC_p)
        FLyC_n = np.sum(LyC_n)

        LyC_final = np.interp(metal, [met_prev, met_next], [FLyC_p, FLyC_n])

        #a_UV = (FUV_n - FUV_p) / (met_next - met_prev)
        #b_UV = (- FUV_n * met_prev + FUV_p * met_next) / (met_next - met_prev)

        return LyC_final
    
    def get_LW(self, metal, Mstar, SFR, z, SFH_samp=None):
        """
                Function returs the luminsoity from 912 to 1108 angstroms
                represnting the Lyman-Werner luminosity.
                Input
                ----------
                    metal : float,
                        Metallicity of the galaxy.
                    Mstar : float,
                        Stellar mass of the galaxy.
                    SFR : float,
                        Star formation rate of the galaxy.
                    z : float,
                        redshift of observation.
                    SFH_samp : boolean,
                        whether SFH is sampled or it's given by previous properties.
                        So far sampling does nothing so it's all the same.
                Output
                ----------
                    LW_final : float,
                        LW luminosity in ergs.
                """

        metal=OH_to_mass_fraction(metal)

        metal = metal / 10 ** 0.42 #see above


        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]
        
        SEDp = self.SEDS[i-1]
        SEDn = self.SEDS[i]

        try:
            self.SFH
        except AttributeError:

            if SFH_samp is None:
                SFH_short, self.index_age = get_SFH_exp(Mstar, SFR, z)
            else:
                SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 10 ** 6

        wv_LW = self.wv[911:1107]
        LW_p = np.zeros(self.ages-1)
        LW_n = np.zeros(self.ages-1)
        for i in range(self.ages-1):
            LW_p[i] = simps(np.array(SEDp[i][911:1107]), wv_LW) * self.SFH[i] * (self.ag[i+1]- self.ag[i])
            LW_n[i] = simps(np.array(SEDn[i][911:1107]), wv_LW) * self.SFH[i] * (self.ag[i+1]- self.ag[i])

        FLW_p = np.sum(LW_p)
        FLW_n = np.sum(LW_n)

        LW_final = np.interp(metal, [met_prev, met_next], [FLW_p, FLW_n])
        
        return LW_final

    def get_nion(self, metal, Mstar, SFR, z, SFH_samp = None):
        """
                Function returs the number of ionizing photons
                Input
                ----------
                    metal : float,
                        Metallicity of the galaxy.
                    Mstar : float,
                        Stellar mass of the galaxy.
                    SFR : float,
                        Star formation rate of the galaxy.
                    z : float,
                        redshift of observation.
                Output
                ----------
                    nion_final : float,
                        Number of ionizing photons produced per second.
                """
        metal = OH_to_mass_fraction(metal)

        metal = metal / 10 ** 0.42 #see above

        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i != 0:
            met_prev = self.metal_avail[i - 1]
        met_next = self.metal_avail[i]

        SEDp = self.SEDS[i - 1]
        SEDn = self.SEDS[i]

        try:
            self.SFH
        except AttributeError:

            if SFH_samp is None:
                SFH_short, self.index_age = get_SFH_exp(Mstar, SFR, z)
            else:
                SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 10 ** 6

        wv_nion = self.wv[:912]
        nion_p = np.zeros(self.ages - 1)
        nion_n = np.zeros(self.ages - 1)
        for i in range(self.ages - 1):
            nion_p[i] = np.sum(np.array(SEDp[i][:912]) / (6.626 * 1e-27 * wv_to_freq(wv_nion))) *self.SFH[i] * (self.ag[i + 1] - self.ag[i]) * 3.826 * 1e33
            nion_n[i] = np.sum(np.array(SEDn[i][:912]) / (6.626 * 1e-27 * wv_to_freq(wv_nion))) *self.SFH[i] * (self.ag[i + 1] - self.ag[i]) * 3.826 * 1e33

        nion_p_tot = np.sum(nion_p)
        nion_n_tot = np.sum(nion_n)

        nion_final = np.interp(metal, [met_prev, met_next], [nion_p_tot, nion_n_tot])

        return nion_final

    def get_beta(self, metal, SFR, Mstar, z, SFH_samp=None):
        """
                Function returs the UV slope for the galaxy.
                Input
                ----------
                    metal : float,
                        Metallicity of the galaxy.
                    Mstar : float,
                        Stellar mass of the galaxy.
                    SFR : float,
                        Star formation rate of the galaxy.
                    z : float,
                        redshift of observation.
                Output
                ----------
                    beta_slope : float,
                        beta slope for the galay.
                """
        metal = OH_to_mass_fraction(metal)

        metal = metal / 10 ** 0.42 #see above

        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]

        SEDp = self.SEDS[i - 1]
        SEDn = self.SEDS[i]

        try:
            self.SFH
        except AttributeError:

            if SFH_samp is None:
                SFH_short, self.index_age = get_SFH_exp(Mstar, SFR, z)
            else:
                SFH_short, self.index_age = SFH_samp.get_SFH_exp(Mstar, SFR)
            self.SFH = np.zeros(self.ages - 1)
            self.SFH[:len(SFH_short)] = np.array(SFH_short)
            self.SFH /= 10 ** 6

        wv_bet = self.wv[1216:3200]        #beta is usually derived redwards of Ly-a
        bet_p = np.zeros((self.ages - 1, len(wv_bet)))
        bet_n = np.zeros((self.ages - 1, len(wv_bet)))

        for i in range(self.ages-1):
            bet_p[i] = np.array(SEDp[i][1215:3199]) * self.SFH[i] * (self.ag[i+1]- self.ag[i])
            bet_n[i] = np.array(SEDn[i][1215:3199]) * self.SFH[i] * (self.ag[i+1]- self.ag[i])
        
        bet_p_to_sum = []
        bet_n_to_sum = []
        #if ages_bet!=(self.ages):
        #    for wv_ind in range(len(wv_bet)):
        #        missing_piecep = np.interp(t_age, self.ag[1:], bet_p[:,wv_ind], right=0)
        #        missing_piecen = np.interp(t_age, self.ag[1:], bet_n[:,wv_ind], right=0)

         #       bet_p_to_sum.append(np.append(bet_p[:ages_bet, wv_ind], missing_piecep))
         #       bet_n_to_sum.append(np.append(bet_n[:ages_bet, wv_ind], missing_piecen))
        #bet_p_to_sum = np.array(bet_p_to_sum)
        #bet_n_to_sum = np.array(bet_n_to_sum)
        SED_summed_p = np.sum(bet_p, axis=0)
        SED_summed_n = np.sum(bet_n, axis=0)
        print(np.shape(SED_summed_p))
        SED_interp = np.zeros(np.shape(SED_summed_p)[0])
        for wv_ind in range(np.shape(SED_summed_p)[0]):
            SED_interp[wv_ind] = np.interp(metal, [met_prev, met_next], [SED_summed_p[wv_ind], SED_summed_n[wv_ind]])
        
        beta_slope, _ = np.polyfit(np.log10(wv_bet),np.log10(SED_interp), 1)

        return beta_slope

def loader(metal, band,SFR,z, filename = '/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-imf135_300.a+00.'):
    """
        Old function to load and manipulate with the bpass data. Not used anymore.
        Parameters
        ----------
        filename: string
            filename extension of the files
        Note: this includes the path to the files and the everything except the
              metalicity in the filename. That is it include info about the 
              popluation (sin or bin, bin taken here), imf (imf135_300 is 
              default for bpass) and fraction of mass in alpha elements relative
              to Iron (0.02 taken here, to be close to the previous versions).
    """
    metal_avail = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.003, 0.004, 0.006, 0.008,
                            0.01, 0.02, 0.03, 0.04])

    for i, met_cur in enumerate(metal_avail):
        if metal < met_cur:
            break
    
    met_prev = None
    if i!=0:
        met_prev = metal_avail[i-1]
    met_next = metal_avail[i]
 
    if met_prev is not None and met_prev==1e-5:
        met_prev_filename = 'zem5'
    elif met_prev is not None and met_prev==1e-4:
        met_prev_filename = 'zem4'
    elif met_prev is not None and met_prev<0.01:
        met_prev_filename = 'z' + str(met_prev)[2:]
    elif met_prev is not None and met_prev>=0.01:
        met_prev_filename = 'z' + str(met_prev)[2:] + '0'

    if met_next==1e-5:
        met_next_filename = 'zem5'
    if met_next==1e-4:
        met_next_filename = 'zem4'
    elif met_next is not None and met_next<0.01:                                                   
        met_next_filename = 'z' + str(met_next)[2:] 
    elif met_next is not None and met_next>=0.01:
        met_next_filename = 'z' + str(met_next)[2:] + '0'
    
    fp = open(filename + met_prev_filename + '.dat').read().split("\n")
    fn = open(filename + met_next_filename + '.dat').read().split("\n")

    wv_b = len(fp)-1
    wv = np.linspace(1,1e5, wv_b+1)
    
    SEDp = [[float(fp[i].split()[j]) for i in range(wv_b)] for j in range(1,52)]
    SEDn = [[float(fn[i].split()[j]) for i in range(wv_b)] for j in range(1,52)]
    ages = len(SEDp)
    ag = [0] + [10**(6.05+ 0.1*i) for i in range(1,52)]
    
    #need to estimate the age of galaxy 
    t_star = 0.36
    t_age = t_star * (cosmo.H(z).to(u.yr**(-1)).value)**-1
    
    for index, age in enumerate(ag):
        if age > t_age:
            ages = index - 1
            break
    
    unit_conv = 3.826 * 10**33 * 1e-7 / const.h.value * wv * 1e-10 
    mburst = SFR / 10**6
 
    if band == 'LW':
        wv_LW = wv[911:1107]

        LW_p = np.zeros(len(ag)-1)
        LW_n = np.zeros(len(ag)-1)

        for i in range(len(ag)-1):
            LW_p[i] = simps(SEDp[i][911:1107], wv_LW) * mburst * (ag[i+1]-ag[i])
            LW_n[i] = simps(SEDn[i][911:1107], wv_LW) * mburst * (ag[i+1]-ag[i])

        if ages!=len(ag):
            missing_piecep = np.interp(t_age, ag[1:], LW_p, right=0)
            missing_piecen = np.interp(t_age, ag[1:], LW_n, right=0)

            LW_p_to_sum = np.append(LW_p[:ages+1], missing_piecep)
            LW_n_to_sum = np.append(LW_n[:ages+1], missing_piecen)

        FLW_p = np.sum(LW_p_to_sum)
        FLW_n = np.sum(LW_n_to_sum)
    
        a_LW = (FLW_n - FLW_p) / (met_next - met_prev)
        b_LW = ( - FLW_n * met_prev + FLW_p * met_next) / (met_next - met_prev)
        
        return a_LW * metal + b_LW

    elif band == 'UV': 
        wv_UV = wv[1600]

        UV_p = np.zeros(len(ag)-1)
        UV_n = np.zeros(len(ag)-1)

        for i in range(len(ag)-1):                               #to get per Hz       to get per cm
            UV_p[i] = SEDp[i][1600] * mburst * (ag[i+1]-ag[i]) * const.c.cgs.value / (1e-8)
            UV_n[i] = SEDn[i][1600] * mburst * (ag[i+1]-ag[i]) * const.c.cgs.value / (1e-8)
        
        if ages!=(len(ag)):
            missing_piecep = np.interp(t_age, ag[1:], UV_p, right=0)
            missing_piecen = np.interp(t_age, ag[1:], UV_n, right=0)

            UV_p_to_sum = np.append(UV_p[:ages+1], missing_piecep)
            UV_n_to_sum = np.append(UV_n[:ages+1], missing_piecen)

        FUV_p = np.sum(UV_p_to_sum)
        FUV_n = np.sum(UV_n_to_sum)
        print(FUV_p, FUV_n)

        a_UV = (FUV_n - FUV_p) / (met_next - met_prev)
        b_UV = (FUV_p * met_next - FUV_n * met_prev) / (met_next - met_prev)
        
        return a_UV * metal + b_UV
