# Stochasticity_sampler
Sampler for estimating PDF of emissivities due to various stochasticities for high redshift Universe.


For chmf, a simple hmf instance can be obtained via:

hmf_this = chmf(z=5, delta_bias=0.0, R_bias = 5)                                            #delta_bias is a global delta_bias for this instance, R_bias, the region. z is the redshift
hmf_this.prep_for_hmf_st(log10_Mmin = 5, log10_Mmax = 15, dlog10m = 0.01 )                                #log10_Min minimum mass for hmf, log10_Mmax is the maximu, and                                                                                              #dlog10m is the interval length in log space
masses, mass_func = hmf_this.run_hmf_st(delta_bias = -0.6828119768123684)                                          #Note, this one will actually be used
