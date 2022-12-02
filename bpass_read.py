"""Contains all necessary ingredients for manipulating bpass files"""

import numpy as np
from scipy.integrate import simpson as simps
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from multiprocessing import Pool
from astropy import constants as const

def reader(name):
    return open(name).read().split('\n')

def splitter(fp):
    wv_b = int(1e5)
    return [[float(fp[i].split()[j]) for i in range(wv_b)] for j in range(1,52)]

class bpass_loader:
    
    def __init__(self, parallel = None, filename = '/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-imf135_300.a+00.'):
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
        else:
            self.SEDS = [[[float(fp[i].split()[j]) for i in range(self.wv_b)] for j in range(1,52)] for fp in self.SEDS_raw]
        self.ages = 52
        self.ag = [0] + [10**(6.05 + 0.1 * i) for i in range(1,52)]
        
        self.t_star = 0.36
        
    def get_UV(self, metal, band, SFR, z):
        
        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]
        
        SEDp = self.SEDS[i-1]
        SEDn = self.SEDS[i]
        
        t_age = self.t_star * (cosmo.H(z).to(u.yr**(-1)).value)**-1
        
        for index, age in enumerate(self.ag):
            if age > t_age:
                ages_UV = index -1
                break
        if index==self.ages-1:
            ages_UV=self.ages-1
        
        mburst = SFR / 10**6
        
        wv_UV = self.wv[1600]
        UV_p = np.zeros(self.ages-1)
        UV_n = np.zeros(self.ages-1)
        for i in range(self.ages-1):
            UV_p[i] = SEDp[i][1600] * mburst * (self.ag[i+1]-self.ag[i]) * const.c.cgs.value / (1e-8)
            UV_n[i] = SEDn[i][1600] * mburst * (self.ag[i+1]-self.ag[i]) * const.c.cgs.value / (1e-8)
        
        if ages_UV!=(self.ages):
            missing_piecep = np.interp(t_age, self.ag[1:], UV_p, right=0)
            missing_piecen = np.interp(t_age, self.ag[1:], UV_n, right=0)

            UV_p_to_sum = np.append(UV_p[:ages_UV], missing_piecep)
            UV_n_to_sum = np.append(UV_n[:ages_UV], missing_piecen)
        
        FUV_p = np.sum(UV_p_to_sum)
        FUV_n = np.sum(UV_n_to_sum)
        
        a_UV = (FUV_n - FUV_p) / (met_next - met_prev)
        b_UV = (- FUV_n * met_prev + FUV_p * met_next) / (met_next - met_prev)
        
        return a_UV * metal + b_UV
    
    def get_LW(self, metal, band, SFR, z):
        
        for i, met_cur in enumerate(self.metal_avail):
            if metal < met_cur:
                break
        met_prev = None
        if i!=0:
            met_prev = self.metal_avail[i-1]
        met_next = self.metal_avail[i]
        
        SEDp = self.SEDS[i-1]
        SEDn = self.SEDS[i]
        
        t_age = self.t_star * (cosmo.H(z).to(u.yr**(-1)).value)**-1
        
        for index, age in enumerate(self.ag):
            if age > t_age:
                ages_LW = index -1
                break
        if index==self.ages-1:
            ages_LW=self.ages-1
        
        mburst = SFR / 10**6
        
        wv_LW = self.wv[911:1107]
        LW_p = np.zeros(self.ages-1)
        LW_n = np.zeros(self.ages-1)
        for i in range(self.ages-1):
            LW_p[i] = simps(SEDp[i][911:1107], wv_LW) * mburst * (self.ag[i+1]- self.ag[i])
            LW_n[i] = simps(SEDn[i][911:1107], wv_LW) * mburst * (self.ag[i+1]- self.ag[i])
        
        if ages_LW!=(self.ages):
            missing_piecep = np.interp(t_age, self.ag[1:], LW_p, right=0)
            missing_piecen = np.interp(t_age, self.ag[1:], LW_n, right=0)

            LW_p_to_sum = np.append(LW_p[:ages_LW], missing_piecep)
            LW_n_to_sum = np.append(LW_n[:ages_LW], missing_piecen)
        
        FLW_p = np.sum(LW_p_to_sum)
        FLW_n = np.sum(LW_n_to_sum)
        
        a_LW = (FLW_n - FLW_p) / (met_next - met_prev)
        b_LW = (- FLW_n * met_prev + FLW_p * met_next) / (met_next - met_prev)
        
        return a_LW * metal + b_LW
        
                
def loader(metal, band,SFR,z, filename = '/home/inikolic/projects/stochasticity/stoc_sampler/BPASS/spectra-bin-imf135_300.a+00.'):
    """
        Function to load and manipulate with the bpass data
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