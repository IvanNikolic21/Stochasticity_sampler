"""Contains all the x-ray functions."""

import numpy as np

def LxvsSFR_lin(Z_sample=None, pol=1):
    """
        Returns the coefficients of the linear fit to the log(Lx/SFR) vs log(SFR) relation.
        Parameters
        ----------
        Z_sample: boolean, optional.
            Metalicity for which the relations are evaluated.
            Default is None for which the metalciities are marginalized over.
        pol: int, optinal.
            Polynomial order which is going to be fit. Default is 1.
        returns
        ----------
        a_Lx: float,
            linear coefficient of the relation.
        b_Lx: float,
            b coefficient of the relation.
        sigma_Lx: float,
            standard deviation of the relation. Given as a mean over the sigma's
    """    
    if Z_sample:
        indices = [7.0,7.2,7.4,7.6,7.8,8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2]
    
    SFRdot01_x = [38.16, 38.16, 38.16, 38.16, 38.16, 38.15, 38.15, 38.15, 38.14, 38.14, 38.14, 38.14]
    SFRdot01_s = [0.75, 0.77, 0.77, 0.77, 0.74, 0.73, 0.72, 0.70, 0.69, 0.69, 0.68, 0.67]
    SFRdot1_x = [38.82, 38.89, 38.93, 38.89, 38.83, 38.77, 38.71, 38.67, 38.65, 38.62, 38.62, 38.60]
    SFRdot1_s = [1.53, 1.56, 1.59, 1.64, 1.65,1.58, 1.37, 1.00, 0.75, 0.64, 0.58, 0.55]
    SFR1_x = [39.92, 40.04, 40.11, 40.13, 40.09, 40.00, 39.85, 39.63, 39.37, 39.14, 39.01, 38.95]
    SFR1_s = [0.31, 0.30, 0.31, 0.33, 0.38, 0.46, 0.59, 0.67, 0.57, 0.59, 0.49, 0.38]
    SFR10_x = [39.95, 40.07, 40.14, 40.16, 40.14, 40.07, 39.96, 39.81, 39.64, 39.46, 39.28, 39.13]
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

            
    #This part is going to be new
    a_Lx, b_Lx = np.polyfit(np.log10(np.array(SFR)), np.array(values) + np.log10(np.array(SFR)), pol )
    sigma_Lx = np.mean(values_sigma)
    
    return a_Lx, b_Lx, sigma_Lx
