"""Here escape fraction lies"""

import numpy as np

def fesc_distr(option = 'binary', Mh = None):
    """
        The function is used to give escape fraction distribution.
        Parameters
        ----------
        option: string,
            what distribution to use for the escaep fraction.
            Possible options are 'binary' for simple binary distribution and
            'ksz_inference' for the mean which is given by Nikolic+22.
        Mh: float; optional,
            Halo mass for this sample.
            Only for option = 'ksz_inference'.
        Returns
        ----------
    """
    
    if option == 'binary':
        return np.random.binomial(1, 0.5) #maximum variance.
    elif option == 'ksz_inference':
        f_esc10 = 0.053
        alpha_esc = 0.00
        
        #scatter estimate based on Yeh+22 (Thesan project)
        logM1 = 9.0
        logM2 = 11.0
        logs1 = 0.4
        logs2 = 0.2
        logM = np.log10(Mh)
        scat_value = np.interp(logM, [logM1, logM2], [logs1, logs2])
         
        if M is None:
            raise ValueError('Please enter Mh if using ksz_inference option')

        f_esc_mean = f_esc10 * (Mh / 10**10) ** alpha_esc
        logf_esc_mean = np.log10(f_esc_mean) - np.log(10) * scat_value**2 * 0.5
        return logf_esc_mean, scat_value

    elif option == 'binary':
        pass
    #TBD
        
