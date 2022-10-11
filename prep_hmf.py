from chmf import chmf

hmf_this = chmf(z=20, delta_bias = 0, R_bias = 5)
hmf_this.prep_for_hmf_st(5,20, 0.01)
hmf_this.prep_collapsed_fractions()
