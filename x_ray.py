"""
Sampler for sampling the x-ray emissivity at high redshifts.

Stochasticity comes from various sources which are explained in the code.
"""

import json

import hmf
import numpy as np
from astropy.cosmology import Planck15 as Cosmo
from hmf import integrate_hmf
from numpy.random import normal
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


def nonlin(x):
    """Non-linear transform from Eulerian to Lagrangian overdensity."""
    return (
        -1.35 * (1 + x) ** (-2 / 3)
        + 0.78785 * (1 + x) ** (-0.58661)
        - 1.12431 * (1 + x) ** (-1 / 2)
        + 1.68647
    )


def sampler(
    z=10,
    delta_bias=0.0,
    r_bias=5.0,
    log10_mass_min=5,
    log10_mass_max=15,
    dlog10m=0.01,
    n_iter=1000,
    sample_hmf=True,
    sample_overdensities=True,
    sample_sfr=0,  # 0 is for sampling using data from Ceverino+18,
    # 1 is for Tacchella+18,
    # 2 is not sampling but using mean from Ceverino+18,
    # 3 is for mean from Tacchella
    sample_lx=0,  # 0 is for sampling using data from Lehmer+20,
    # 1 is from Kouroumpatzakis+20
    # 2 is not sampling but using mean from Lehmer+20,
    # 3 is for mean from Kouroumpatzakis+20
    calculate_2pcc=False,  # if True
    # calculate 2point-correlation Poission correction
    interpolating=True,  # if True
    # interpolate hmf to better sample the mass function
    duty_cycle=True,  # if True turn off duty cycle in the sampler.
    sample_metal=True,  # if True, sample metallicity
):
    """Return an array of emissivities for each iteration of the sampler."""
    # let's intiate the halo mass function
    m_turn = 5 * 10 ** 8  # Park+19 parametrization

    # if not sample densities, no need to calculate hmf every time

    if (sample_sfr == 0 or sample_sfr == 2) and sample_metal:  # initiate SFR
        (
            means_ms_rel,
            sigma_ms_rel,
            a_ms_rel,
            b_ms_rel,
            means_sfr_rel,
            sigma_sfr_rel,
            a_sfr_rel,
            b_sfr_rel,
        ) = sfrvsmh(z)
    elif (sample_sfr == 0 or sample_sfr == 2) and not sample_metal:
        (
            means_ms_rel,
            sigma_ms_rel,
            a_ms_rel,
            b_ms_rel,
        ) = sfrvsmh(z, sample_metal=False)

    if sample_overdensities:
        hmf_this = hmf.MassFunction(
            z=z,
            Mmin=log10_mass_min,
            Mmax=log10_mass_max,
            dlog10m=dlog10m,
            conditional_mass=rtom(r_bias),
        )  # just a quick one to calculate

        # intialize samples of densities:
        gauss_lagr = []
        delta_lag = np.linspace(-7, 15, 100000)
        delta_other = nonlin(delta_lag * hmf_this.growth_f) / hmf_this.growth_f
        sigma_non_nan = hmf_this.sigma[~np.isnan(hmf_this.sigma)]
        radii_non_nan = hmf_this.radii[~np.isnan(hmf_this.sigma)]
        for index, i in enumerate(delta_other):
            mult_f = (1 + delta_lag[index] * hmf_this.growth_f) ** (1 / 3.0)
            sigma = (
                np.interp(
                    r_bias * mult_f * radii_non_nan / Cosmo.h,
                    sigma_non_nan,
                )
                / hmf_this.growth_f
            )
            exp_f = np.exp(-((i) ** 2) / 2 / (sigma) ** 2)
            gauss_lagr.append(1 / np.sqrt(2 * np.pi) / sigma * exp_f)
        n_mean_cumsum = integrate.cumtrapz(gauss_lagr, delta_other)
        cumsum = n_mean_cumsum / n_mean_cumsum[-1]
        random_numb = np.random.uniform(size=n_iter)
        delta_list = np.zeros(shape=n_iter)
        for index, random in enumerate(random_numb):
            delta_list[index] = np.interp(random, cumsum, delta_lag[:-1])
            delta_list[index] = nonlin(delta_list[index] * hmf_this.growth_f)

        hmf_this = Chmf(z=z, delta_bias=delta_bias, r_bias=r_bias)
        hmf_this.prep_for_hmf(
            log10_mass_min=log10_mass_min,
            log10_mass_max=log10_mass_max,
            dlog10m=dlog10m,
        )

        delta_nonlin = np.linspace(-0.99, 10)
        delta_lin_values = nonlin(delta_nonlin)

    if not sample_overdensities:
        if delta_bias == 0.0:
            # calculate mass_bin resolution
            hmf_this = hmf.MassFunction(
                z=z, Mmin=log10_mass_min, Mmax=log10_mass_max, dlog10m=dlog10m
            )  # at this redshift, larger mass halos are non-existent
            mass_func = hmf_this.dndm
            masses = hmf_this.m
            n_mean_cumsum = (
                integrate_hmf.hmf_integral_gtm(masses, mass_func, mass_d=False)
                * 4
                / 3
                * np.pi
                * r_bias ** 3
            )
            cumulative_mass = integrate_hmf.hmf_integral_gtm(
                masses, mass_func, mass_d=True
            )

        else:
            hmf_this = Chmf(
                z=z, delta_bias=delta_bias, r_bias=r_bias
            )  # TODO fix this still, it's in my to do list.
            hmf_this.prep_for_hmf(
                log10_mass_min=log10_mass_min,
                log10_mass_max=log10_mass_max,
                dlog10m=dlog10m,
            )
            masses, mass_func = hmf_this.run_hmf(delta_bias)
            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element == 0:
                    break
            masses = masses[:index_to_stop]
            mass_func = mass_func[:index_to_stop]
            n_mean_cumsum = (
                integrate_hmf.hmf_integral_gtm(
                    masses[:index_to_stop], mass_func[:index_to_stop]
                )
                * 4
                / 3
                * np.pi
                * r_bias ** 3
            )

        n_mean = int((n_mean_cumsum)[0])
    n_cumsum_normalized = n_mean_cumsum / n_mean_cumsum[0]
    emissivities = np.zeros(shape=n_iter)

    for i in range(n_iter):

        if sample_overdensities:
            delta_bias = delta_list[i]
            delta_bias = np.interp(delta_bias, delta_lin_values, delta_nonlin)
            delta_bias /= hmf_this.dicke()
            masses, mass_func = hmf_this.run_hmf(delta_bias)
            for index_to_stop, mass_func_element in enumerate(mass_func):
                if mass_func_element == 0:
                    break
            masses = masses[:index_to_stop]
            mass_func = mass_func[:index_to_stop]

            n_mean_cumsum = (
                integrate_hmf.hmf_integral_gtm(
                    masses[:index_to_stop], mass_func[:index_to_stop]
                )
                * 4
                / 3
                * np.pi
                * r_bias ** 3
            )
            n_mean = int((n_mean_cumsum)[0])
            n_cumsum_normalized = n_mean_cumsum / n_mean_cumsum[0]

        if not sample_hmf:
            n_this_iter = n_mean
        else:
            if not calculate_2pcc:
                n_this_iter = np.random.poisson(n_mean)
            else:
                delta = delta_scc(hmf_this)
                beta = 1 - np.sqrt(1 + delta / n_mean)
                n_s = np.logspace(1, np.log(n_mean) + 1)
                super_poisson = (
                    n_mean
                    / np.sqrt(2 * np.pi * n_s)
                    * np.e ** (n_s / 2 - n_mean(1 - beta) - n_s * beta)
                    * (1 - beta)
                    / n_s ** (n_s / 2)
                    * (n_mean(1 - beta) + n_s * beta) ** (n_s - 1)
                )
                super_p_cumsum = integrate_hmf.hmf_integral_gtm(
                    n_s, super_poisson, mass_d=False
                )
                super_p_cumsum = super_p_cumsum / super_p_cumsum[0]
                give_me_random = np.random.uniform(size=1)
                n_this_iter = np.interp(
                    give_me_random,
                    np.flip(np.array(super_p_cumsum)),
                    np.flip(np.array(n_s)),
                )

        random_numbers = np.random.uniform(size=n_this_iter)
        masses_of_haloes = np.zeros(shape=n_this_iter)

        if interpolating:
            for index, rn in enumerate(random_numbers):
                masses_of_haloes[index] = np.interp(
                    rn,
                    np.flip(np.array(n_cumsum_normalized)),
                    np.flip(np.array(masses)),
                )
        else:
            for index, rn in enumerate(random_numbers):
                for k, cum_mass in enumerate(cumulative_mass):
                    if cum_mass > rn:
                        masses_of_haloes[index] = masses[k]
                        break

        masses_saved = []
        if duty_cycle:
            for index, mass in enumerate(masses_of_haloes):
                if np.random.binomial(1, np.exp(-m_turn / mass)):
                    masses_saved.append(mass)

        lx = np.zeros(shape=n_this_iter)

        sfr_samples = np.zeros(shape=n_this_iter)
        if sample_metal:
            ms_samples = np.zeros(shape=n_this_iter)
            z_samples = np.zeros(shape=n_this_iter)

        for j, mass in enumerate(masses_saved):

            a_ms, b_ms, sigma_ms = give_me_sigma(
                means_ms_rel, sigma_ms_rel, a_ms_rel, b_ms_rel, mass
            )  # need to fix this as it does smth weird

            if sample_sfr == 0 and sample_metal:
                ms_s = 10 ** (
                    normal(
                        (a_ms_rel * np.log10(mass) + b_ms_rel),
                        sigma_ms_rel,
                    )
                )
                a_sfr, b_sfr, sigma_sfr = give_me_sigma(
                    means_sfr_rel,
                    sigma_sfr_rel,
                    a_sfr_rel,
                    b_sfr_rel,
                    ms_s,
                )
                sfr_s = 10 ** (
                    normal(
                        (a_sfr_rel * np.log10(ms_s) + b_sfr_rel),
                        sigma_sfr_rel,
                    )
                )

            elif sample_sfr == 1:
                ms_s = 10 ** (
                    normal(
                        (a_ms_rel * np.log10(mass) + b_ms_rel),
                        sigma_ms_rel,
                    )
                )
                sfr_v = 1.67 * np.log10(ms_s) - 15.365 - 1.67 * np.log10(0.15)
                sfr_s = 10 ** (normal(sfr_v, 0.2))

            elif sample_sfr == 2 and sample_metal:
                ms_s = 10 ** (a_ms_rel * np.log10(mass) + b_ms_rel)
                a_sfr, b_sfr, sigma_sfr = give_me_sigma(
                    means_sfr_rel,
                    sigma_sfr_rel,
                    a_sfr_rel,
                    b_sfr_rel,
                    ms_s,
                )
                sfr_s = 10 ** (a_sfr_rel * np.log10(ms_s) + b_sfr_rel)

            elif sample_sfr == 3:
                ms_s = 10 ** (a_ms_rel * np.log10(mass) + b_ms_rel)
                log_sfr_s = 1.67 * np.log10(ms_s)
                log_sfr_s = log_sfr_s - 15.365 - 1.67 * np.log(0.15)
                sfr_s = 10 ** log_sfr_s

            elif sample_sfr == 0 and not sample_metal:
                sfr_s = 10 ** (
                    normal(
                        (a_ms_rel * np.log10(mass) + b_ms_rel),
                        sigma_ms_rel,
                    )
                )
            elif sample_sfr == 2 and not sample_metal:
                sfr_s = 10 ** (a_ms_rel * np.log10(mass) + b_ms_rel)

            if sample_lx == 0 and sample_metal:
                if sample_sfr == 0 or sample_sfr == 1:
                    pass
                else:
                    ms_s = 10 ** (a_ms_rel * np.log10(mass) + b_ms_rel)

                z_mean, sigma_z = metalicity_from_fmr(ms_s, sfr_s)
                z_s = 10 ** (normal((np.log10(z_mean)), sigma_z))
                loglx_over_sfr, logsigma_lx_over_sfr = give_me_lx(sfr_s, z_s)
                lx_sample = 10 ** normal(
                    loglx_over_sfr + np.log10(sfr_s), logsigma_lx_over_sfr
                )

            elif sample_lx == 1:
                lx_sample = 10 ** normal((0.8 * np.log10(sfr_s) + 39.40), 0.9)

            elif sample_lx == 2 and sample_metal:
                if sample_sfr == 2:
                    pass
                else:
                    ms_s = 10 ** (
                        normal(
                            (a_ms_rel * np.log10(mass) + b_ms_rel),
                            sigma_ms_rel,
                        )
                    )

                z_s, _ = metalicity_from_fmr(ms_s, sfr_s)

                loglx_over_sfr, _ = give_me_lx(sfr_s, z_s)
                lx_sample = 10 ** (loglx_over_sfr + np.log10(sfr_s))

            elif sample_lx == 3:
                lx_sample = 10 ** (0.8 * np.log10(sfr_s) + 39.40)

            elif sample_lx == 0 and not sample_metal:
                loglx_over_sfr, logsigma_lx_over_sfr = give_me_lx(sfr_s)
                lx_sample = 10 ** normal(
                    loglx_over_sfr + np.log10(sfr_s), logsigma_lx_over_sfr
                )

            elif sample_lx == 2 and not sample_metal:
                loglx_over_sfr, logsigma_lx_over_sfr = give_me_lx(sfr_s)
                lx_sample = 10 ** (loglx_over_sfr + np.log10(sfr_s))

            lx[j] = lx_sample
            sfr_samples[j] = sfr_s
            if sample_metal:
                ms_samples[j] = ms_s
                z_samples[j] = z_s

        emissivities[i] = np.sum(lx)

    return (
        emissivities,
        ms_samples,
        sfr_samples,
        lx,
        logsigma_lx_over_sfr,
        sigma_sfr_rel,
    )  # , z_samples


def rtom(r):
    """Conversion from Radius to Mass."""
    cmpermpc3 = 3.086e24 ** 3
    msun = 1.989e33
    critical_density = Cosmo.critical_density0.value * cmpermpc3 / msun
    return (4.0 / 3.0) * np.pi * (r ** 3) * (Cosmo.Om0 * critical_density)


class Chmf:
    """Class that contains everyhing hmf related."""

    def z_drag_calculate(self):
        """Calculate z_drag."""
        z_drag = 0.313 * (self.omhh ** -0.419)
        z_drag *= 1 + 0.607 * (self.omhh ** 0.674)

        z_drag *= 1291 * self.omhh ** 0.251
        z_drag /= 1 + 0.659 * self.omhh ** 0.828
        return z_drag

    def alpha_nu_calculation(self):
        """Calculate alpha_nu."""
        alpha_nu = (
            (self.f_c / self.f_cb)
            * (2 * (self.p_c + self.p_cb) + 5)
            / (4 * self.p_cb + 5.0)
        )
        alpha_nu *= 1 - 0.553 * self.f_nub + 0.126 * (self.f_nub ** 3)
        alpha_nu /= 1 - 0.193 * np.sqrt(self.f_nu) + 0.169 * self.f_nu
        alpha_nu *= (1 + self.y_d) ** (self.p_c - self.p_cb)
        alpha_nu *= 1 + (self.p_cb - self.p_c) / 2.0 * (
            1.0 + 1.0 / (4.0 * self.p_c + 3.0) / (4.0 * self.p_cb + 7.0)
        ) / (1.0 + self.y_d)
        return alpha_nu

    def mtor(self, m):
        """Conversion from mass to radius."""
        if self.FILTER == 0:  # top hat M = (4/3) PI <rho> R^3
            mat_d = Cosmo.Om0 * self.critical_density
            return (3 * m / (4 * np.pi * mat_d)) ** (1.0 / 3.0)

    def rtom(self, r):
        """Conversion from radius to mass."""
        if self.FILTER == 0:
            mat_d = Cosmo.Om0 * self.critical_density
            return (4.0 / 3.0) * np.pi * r ** 3 * mat_d

    def __init__(self, z, delta_bias, r_bias):
        """Initialize variables for the class."""
        self.CMperMPC = 3.086e24
        self.Msun = 1.989e33
        self.TINY = 10 ** -30
        self.Deltac = 1.686
        self.FILTER = 0
        self.T_cmb = Cosmo.Tcmb0.value
        self.theta_cmb = self.T_cmb / 2.7
        self.critical_density = (
            Cosmo.critical_density0.value
            * self.CMperMPC
            * self.CMperMPC
            * self.CMperMPC
            / self.Msun
        )
        self.z = z
        self.delta_bias = delta_bias
        self.r_bias = r_bias
        self.omhh = Cosmo.Om0 * (Cosmo.h) ** 2
        self.z_equality = 25000 * self.omhh * self.theta_cmb ** -4 - 1.0
        self.k_equality = 0.0746 * self.omhh / (self.theta_cmb ** 2)
        self.z_drag = self.z_drag_calculate()
        self.y_d = (1 + self.z_equality) / (1.0 + self.z_drag)
        self.f_nu = Cosmo.Onu0 / Cosmo.Om0
        self.f_baryon = Cosmo.Ob0 / Cosmo.Om0
        self.p_c_var = 1 - self.f_nu - self.f_baryon
        self.p_c = -(5 - np.sqrt(1 + 24 * (self.p_c_var))) / 4.0
        self.p_cb = -(5 - np.sqrt(1 + 24 * (1 - self.f_nu))) / 4.0
        self.f_c = 1 - self.f_nu - self.f_baryon
        self.f_cb = 1 - self.f_nu
        self.f_nub = self.f_nu + self.f_baryon
        self.alpha_nu = self.alpha_nu_calculation()
        self.R_drag = (
            31.5
            * Cosmo.Ob0
            * Cosmo.h ** 2
            * (self.theta_cmb ** -4)
            * 1000
            / (1.0 + self.z_drag)
        )
        self.R_eq = (
            31.5
            * Cosmo.Ob0
            * Cosmo.h ** 2
            * (self.theta_cmb ** -4)
            * 1000
            / (1.0 + self.z_equality)
        )
        self.sound_horizon = (
            2.0
            / 3.0
            / self.k_equality
            * np.sqrt(6.0 / self.R_eq)
            * np.log(
                (np.sqrt(1 + self.R_drag) + np.sqrt(self.R_drag + self.R_eq))
                / (1.0 + np.sqrt(self.R_eq))
            )
        )
        self.beta_c = 1.0 / (1.0 - 0.949 * self.f_nub)
        self.N_nu = 1.0
        self.POWER_INDEX = 0.9667
        self.Radius_8 = 8.0 / Cosmo.h
        self.SIGMA_8 = 0.8159
        self.M_bias = self.rtom(self.r_bias)

    def dicke(self):
        """Dicke growth factor."""
        omegam_z = Cosmo.Om(self.z)
        dick_z = (
            2.5
            * omegam_z
            / (
                1.0 / 70.0
                + omegam_z * (209 - omegam_z) / 140.0
                + pow(omegam_z, 4.0 / 7.0)
            )
        )
        dick_0 = (
            2.5
            * Cosmo.Om0
            / (
                1.0 / 70.0
                + Cosmo.Om0 * (209 - Cosmo.Om0) / 140.0
                + pow(Cosmo.Om0, 4.0 / 7.0)
            )
        )
        return dick_z / (dick_0 * (1.0 + self.z))

    def tfmdm(self, k):
        """Calculate transfer function."""
        q = k * self.theta_cmb ** 2 / self.omhh
        gamma_eff = np.sqrt(self.alpha_nu) + (1.0 - np.sqrt(self.alpha_nu)) / (
            1.0 + (0.43 * k * self.sound_horizon) ** 4
        )
        q_eff = q / gamma_eff
        tf_m_10 = np.e + 1.84 * self.beta_c * np.sqrt(self.alpha_nu) * q_eff
        tf_m = np.log(tf_m_10)
        sec = q_eff ** 2 * (14.4 + 325.0 / (1.0 + 60.5 * (q_eff ** 1.11)))
        tf_m /= tf_m + sec
        q_nu = 3.92 * q / np.sqrt(self.f_nu / self.N_nu)
        tf_m *= 1.0 + (
            1.2 * (self.f_nu ** 0.64) * (self.N_nu ** (0.3 + 0.6 * self.f_nu))
        ) / ((q_nu ** -1.6) + (q_nu ** 0.8))

        return tf_m

    def dsigma_dk(self, k, r):
        """Calculate derivative of sigma over k."""
        t = self.tfmdm(k)
        p = k ** self.POWER_INDEX * t * t
        kr = k * r

        w = 3.0 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)
        return k * k * p * w * w

    def sigma_norm(
        self,
    ):
        """Calculate normalization factor for sigma."""
        result = integrate.quad(
            self.dsigma_dk,
            0,
            np.inf,
            args=(self.Radius_8,),
            limit=1000,
            epsabs=10 ** -20,
        )[0]
        return self.SIGMA_8 / np.sqrt(result)

    def sigma_z0(self, m):
        """Calculate sigma at z=0."""
        r = self.mtor(m)
        err = 10 ** -20
        integral = integrate.quad(
            self.dsigma_dk, 0, np.inf, args=(r,), limit=1000, epsabs=err
        )
        result = integral[0]

        return self.sigma_norm() * np.sqrt(result)

    def dsigmasq_dm(self, k, r):
        """Calculate derivative of sigma squared over mass factor."""
        t = self.tfmdm(k)
        p = k ** self.POWER_INDEX * t * t

        kr = k * r

        w = 3.0 * (np.sin(kr) / kr ** 3 - np.cos(kr) / kr ** 2)

        dwdr = 9 * np.cos(kr) * k / kr ** 3
        dwdr += 3 * np.sin(kr) * (1 - 3 / (kr * kr)) / (kr * r)
        drdm = 1.0 / (4.0 * np.pi * Cosmo.Om0 * self.critical_density * r * r)

        return k * k * p * 2 * w * dwdr * drdm

    def dsigmasqdm_z0(self, m):
        """Calculate derivative of sigma squared over mass."""
        r = self.mtor(m)
        er = 10 ** -20
        result = integrate.quad(
            self.dsigmasq_dm, 0, np.inf, args=(r,), limit=1000, epsabs=er
        )
        return self.sigma_norm() * self.sigma_norm() * result[0]

    def dnbiasdm(self, m):
        """Calculate biased hmf."""
        if (self.M_bias - m) < self.TINY:
            return 0
        delta = self.Deltac / self.dicke() - self.delta_bias

        sig_o = self.sigma_z0(self.M_bias)
        sig_one = self.sigma_z0(m)
        return (
            -(self.critical_density * Cosmo.Om0)
            / m
            / np.sqrt(2 * np.pi)
            * delta
            * ((sig_one ** 2 - sig_o ** 2) ** (-1.5))
            * (np.e ** (-0.5 * delta ** 2 / (sig_one ** 2 - sig_o ** 2)))
            * self.dsigmasqdm_z0(m)
        )

    def prep_for_hmf(self, log10_mass_min=6, log10_mass_max=15, dlog10m=0.01):
        """Do everything needed for hmf."""
        self.log10_mass_min = log10_mass_min
        self.log10_mass_max = log10_mass_max
        self.dlog10m = dlog10m
        self.bins = 10 ** np.arange(
            self.log10_mass_min, self.log10_mass_max, self.dlog10m
        )
        self.sigma_z0_array = np.zeros(len(self.bins))
        self.sigma_derivatives = np.zeros(len(self.bins))
        for index, mass in enumerate(self.bins):
            self.sigma_z0_array[index] = self.sigma_z0(mass)
            self.sigma_derivatives[index] = self.dsigmasqdm_z0(mass)

    def run_hmf(self, delta_bias):
        """Run one instance of hmf."""
        delta = self.Deltac / self.dicke() - delta_bias
        sigma_array = self.sigma_z0_array ** 2 - self.sigma_cell() ** 2
        self.hmf = np.zeros(len(self.bins))
        if delta < 0:
            return 0
        for index, mass in enumerate(self.bins):
            if mass < self.M_bias:
                self.hmf[index] = (
                    -(self.critical_density * Cosmo.Om0)
                    / mass
                    / np.sqrt(2 * np.pi)
                    * delta
                    * ((sigma_array[index]) ** (-1.5))
                    * (np.e ** (-0.5 * delta ** 2 / (sigma_array[index])))
                    * self.sigma_derivatives[index]
                )
            else:
                self.hmf[index] = 0.0
        return self.bins, self.hmf

    def dndmnormal(self, m):
        """Return normalized hmf."""
        if (self.M_bias - m) < self.TINY:
            return 0
        delta = self.Deltac / self.dicke()
        sig_one = self.sigma_z0(m)
        return (
            -(self.critical_density * Cosmo.Om0)
            / m
            / np.sqrt(2 * np.pi)
            * delta
            * ((sig_one ** 2) ** (-1.5))
            * (np.e ** (-0.5 * delta ** 2 / (sig_one ** 2)))
            * self.dsigmasqdm_z0(m)
        )

    def sigma_cell(self):
        """Sigma in a given volume."""
        return self.sigma_z0(self.M_bias)

    def run_hmf_nor(self, log10_mass_min=6, log10_mass_max=15, dlog10m=0.01):
        """Run unbiased hmf."""
        self.log10_mass_min_nor = log10_mass_min
        self.log10_mass_max_nor = log10_mass_max
        self.dlog10m_nor = dlog10m
        self.bins_normal = 10 ** np.arange(
            self.log10_mass_min_nor, self.log10_mass_max_nor, self.dlog10m_nor
        )
        self.hmf_normal = np.zeros(len(self.bins_normal))
        for i, mass in enumerate(self.bins_normal):
            self.hmf_normal[i] = self.dndmnormal(mass)
        return (self.bins_normal, self.hmf_normal)

    def cumulative_number(self):
        """Cumulative number of halos."""
        dndlnm = self.bins * self.hmf
        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]),
                np.log(10 ** 18),
                np.log(self.bins[1]) - np.log(self.bins[0]),
            )
            mf_func = Spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)

            int_upper = integrate.simps(
                np.exp(mf), dx=m_upper[2] - m_upper[1], even='first'
            )
        else:
            int_upper = 0

        # Calculate the cumulative integral (backwards) of [m*]dndlnm
        dx = np.log(self.bins[1]) - np.log(self.bins[0])
        self.cumnum = np.concatenate(
            (
                integrate.cumtrapz(dndlnm[::-1], dx=dx)[::-1],
                np.zeros(1),
            )
        )
        self.cumnum += int_upper
        return self.cumnum

    def cumulative_mass(self):
        """Cumulative mass of halos."""
        dndlnm = self.bins * self.hmf

        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]),
                np.log(10 ** 18),
                np.log(self.bins[1]) - np.log(self.bins[0]),
            )
            mf_func = Spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)
            int_upper = integrate.simps(
                np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even='first'
            )
        else:
            int_upper = 0

        self.cummass = np.concatenate(
            (
                integrate.cumtrapz(
                    self.bins[::-1] * dndlnm[::-1],
                    dx=np.log(self.bins[1]) - np.log(self.bins[0]),
                )[::-1],
                np.zeros(1),
            )
        )
        self.cummass += int_upper
        return self.cummass


def sfrvsmh(z, sample_metal=True):
    """Return various sfr, stellar mass and halo mass relations."""
    ff = open('FirstLight_database.dat')
    database = json.load(ff)
    ff.close()
    extract_keys_base = [
        'FL633',
        'FL634',
        'FL635',
        'FL637',
        'FL638',
        'FL639',
        'FL640',
        'FL641',
        'FL642',
        'FL643',
        'FL644',
        'FL645',
        'FL646',
        'FL647',
        'FL648',
        'FL649',
        'FL650',
        'FL651',
        'FL653',
        'FL655',
        'FL656',
        'FL657',
        'FL658',
        'FL659',
        'FL660',
        'FL661',
        'FL662',
        'FL664',
        'FL665',
        'FL666',
        'FL667',
        'FL668',
        'FL669',
        'FL670',
        'FL671',
        'FL673',
        'FL675',
        'FL676',
        'FL678',
        'FL679',
        'FL680',
        'FL681',
        'FL682',
        'FL683',
        'FL684',
        'FL685',
        'FL686',
        'FL687',
        'FL688',
        'FL689',
        'FL690',
        'FL691',
        'FL692',
        'FL693',
        'FL694',
        'FL695',
        'FL696',
        'FL697',
        'FL698',
        'FL700',
        'FL701',
        'FL702',
        'FL703',
        'FL704',
        'FL705',
        'FL706',
        'FL707',
        'FL708',
        'FL709',
        'FL710',
        'FL711',
        'FL712',
        'FL713',
        'FL714',
        'FL715',
        'FL716',
        'FL717',
        'FL718',
        'FL719',
        'FL720',
        'FL721',
        'FL722',
        'FL723',
        'FL724',
        'FL725',
        'FL726',
        'FL727',
        'FL728',
        'FL729',
        'FL730',
        'FL731',
        'FL732',
        'FL733',
        'FL734',
        'FL735',
        'FL736',
        'FL739',
        'FL740',
        'FL744',
        'FL745',
        'FL746',
        'FL747',
        'FL748',
        'FL750',
        'FL752',
        'FL753',
        'FL754',
        'FL755',
        'FL756',
        'FL757',
        'FL758',
        'FL760',
        'FL761',
        'FL763',
        'FL767',
        'FL768',
        'FL769',
        'FL770',
        'FL771',
        'FL772',
        'FL773',
        'FL774',
        'FL775',
        'FL777',
        'FL778',
        'FL779',
        'FL780',
        'FL781',
        'FL782',
        'FL783',
        'FL784',
        'FL788',
        'FL789',
        'FL792',
        'FL793',
        'FL794',
        'FL796',
        'FL800',
        'FL801',
        'FL802',
        'FL803',
        'FL805',
        'FL806',
        'FL808',
        'FL809',
        'FL810',
        'FL811',
        'FL814',
        'FL815',
        'FL816',
        'FL818',
        'FL819',
        'FL823',
        'FL824',
        'FL825',
        'FL826',
        'FL827',
        'FL829',
        'FL652',
        'FL654',
        'FL699',
        'FL738',
        'FL741',
        'FL742',
        'FL749',
        'FL762',
        'FL764',
        'FL766',
        'FL776',
        'FL785',
        'FL786',
        'FL787',
        'FL790',
        'FL795',
        'FL797',
        'FL798',
        'FL799',
        'FL804',
        'FL807',
        'FL812',
        'FL813',
        'FL817',
        'FL820',
        'FL822',
        'FL828',
        'FL839',
        'FL850',
        'FL855',
        'FL862',
        'FL870',
        'FL874',
        'FL879',
        'FL882',
        'FL891',
        'FL896',
        'FL898',
        'FL903',
        'FL904',
        'FL906',
        'FL915',
        'FL916',
        'FL917',
        'FL918',
        'FL919',
        'FL921',
        'FL922',
        'FL923',
        'FL924',
        'FL925',
        'FL926',
        'FL928',
        'FL930',
        'FL932',
        'FL934',
        'FL935',
        'FL765',
        'FL835',
        'FL836',
        'FL841',
        'FL843',
        'FL844',
        'FL845',
        'FL846',
        'FL847',
        'FL848',
        'FL849',
        'FL851',
        'FL856',
        'FL857',
        'FL858',
        'FL859',
        'FL860',
        'FL861',
        'FL865',
        'FL866',
        'FL867',
        'FL868',
        'FL871',
        'FL877',
        'FL881',
        'FL884',
        'FL885',
        'FL887',
        'FL888',
        'FL893',
        'FL895',
        'FL897',
        'FL899',
        'FL900',
        'FL901',
        'FL907',
        'FL908',
        'FL909',
        'FL911',
        'FL912',
        'FL837',
        'FL838',
        'FL840',
        'FL852',
        'FL853',
        'FL863',
        'FL864',
        'FL869',
        'FL872',
        'FL873',
        'FL883',
        'FL890',
        'FL894',
        'FL902',
        'FL834',
        'FL854',
        'FL876',
        'FL913',
        'FL927',
        'FL931',
        'FL933',
        'FL938',
        'FL939',
        'FL940',
        'FL942',
        'FL943',
        'FL946',
        'FL947',
        'FL914',
        'FL920',
        'FL929',
        'FL936',
        'FL937',
        'FL941',
        'FL944',
    ]
    extract_field1 = 'Mvir'
    extract_field2 = 'Ms'
    extract_field3 = 'SFR'
    s = '_'
    possible_a = np.array(
        [
            0.059,
            0.060,
            0.062,
            0.064,
            0.067,
            0.069,
            0.071,
            0.074,
            0.077,
            0.080,
            0.083,
            0.087,
            0.090,
            0.095,
            0.1,
            0.105,
            0.111,
            0.118,
            0.125,
            0.133,
            0.142,
            0.154,
            0.160,
        ]
    )
    possible_a_str = list(map('{:.3f}'.format, possible_a))
    z_poss = 1.0 / possible_a - 1
    index_actual = np.argmin(abs(np.array(z_poss) - z))
    ms = []
    mh = []
    sfr = []
    for j in range(len(possible_a_str)):
        extract_a_value = possible_a_str[index_actual]
        for i in range(len(extract_keys_base)):
            extract_keys_field1 = (
                extract_keys_base[i] + s + extract_a_value + s + extract_field1
            )
            extract_keys_field2 = (
                extract_keys_base[i] + s + extract_a_value + s + extract_field2
            )
            extract_keys_field3 = (
                extract_keys_base[i] + s + extract_a_value + s + extract_field3
            )
            if ((extract_keys_field1) in database) and (
                (extract_keys_field2) in database
            ):
                mh.append(database[extract_keys_field1])
                ms.append(database[extract_keys_field2])
                sfr.append(database[extract_keys_field3])
    mh_uniq = set()
    mh_uniq_list = []
    sfr_uniq_list = []
    ms_uniq_list = []
    for i, mh in enumerate(mh):
        if (mh, sfr[i], ms[i]) not in mh_uniq:
            mh_uniq.add((mh, sfr[i], ms[i]))
            mh_uniq_list.append(mh)
            sfr_uniq_list.append(sfr[i])
            ms_uniq_list.append(ms[i])
    zipped = zip(mh_uniq_list, sfr_uniq_list, ms_uniq_list)
    zipped = sorted(zipped)
    mh_uniq_sorted, sfr_uniq_sorted, ms_uniq_sorted = zip(*zipped)
    if sample_metal:
        log10_mh = np.log10(mh_uniq_sorted)
        log10_ms = np.log10(ms_uniq_sorted)
        a_ms, b_ms = np.polyfit(log10_mh, log10_ms, 1)
        bins = 2
        binned_masses = np.linspace(
            np.log10(min(mh_uniq_sorted)), np.log10(max(mh_uniq_sorted)), bins
        )
        means_ms = np.zeros(bins - 1)
        number_of_masses = np.zeros(bins - 1)
        err_ms = np.zeros(bins - 1)

        for i, mass in enumerate(mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if (
                    j < (bins - 2)
                    and (np.log10(mass) >= bining)
                    and (np.log10(mass) < binned_masses[j + 1])
                ):
                    line = a_ms * np.log10(mass) + b_ms
                    err_ms[j] += (line - np.log10(ms_uniq_sorted[i])) ** 2
                    number_of_masses[j] += 1
                    means_ms[j] += mass
                    break
                elif j == bins - 2:
                    line = a_ms * np.log10(mass) + b_ms
                    log_ms = np.log10(ms_uniq_sorted[i])
                    err_ms[bins - 2] += (line - log_ms) ** 2
                    number_of_masses[bins - 2] += 1
                    means_ms[bins - 2] += mass
                    break
        err_ms = np.sqrt(err_ms / number_of_masses)
        means_ms = np.sqrt(means_ms / number_of_masses)
        # do the same for Ms-SFR

        a_sfr, b_sfr = np.polyfit(
            np.log10(ms_uniq_sorted), np.log10(sfr_uniq_sorted), 1
        )
        binned_stellar_masses = np.linspace(
            np.log10(min(ms_uniq_sorted)), np.log10(max(ms_uniq_sorted)), bins
        )
        means_sfr = np.zeros(bins - 1)
        err_sfr = np.zeros(bins - 1)
        number_of_stellar_masses = np.zeros(bins - 1)
        for i, stellar_mass in enumerate(ms_uniq_sorted):
            for j, stellar_bining in enumerate(binned_stellar_masses):
                if (
                    j < (bins - 2)
                    and (np.log10(stellar_mass) >= stellar_bining)
                    and (np.log10(stellar_mass) < binned_stellar_masses[j + 1])
                ):
                    err_sfr[j] += (
                        (a_sfr * np.log10(stellar_mass) + b_sfr)
                        - np.log10(sfr_uniq_sorted[i])
                    ) ** 2
                    number_of_stellar_masses[j] += 1
                    means_sfr[j] += stellar_mass
                    break
                elif j == bins - 2:
                    err_sfr[bins - 2] += (
                        (a_sfr * np.log10(stellar_mass) + b_sfr)
                        - np.log10(sfr_uniq_sorted[i])
                    ) ** 2
                    number_of_stellar_masses[bins - 2] += 1
                    means_sfr[bins - 2] += stellar_mass
                    break
        err_sfr = np.sqrt(err_sfr / number_of_stellar_masses)
        means_sfr = np.sqrt(means_sfr / number_of_stellar_masses)
        return (means_ms, err_ms, a_ms, b_ms, means_sfr, err_sfr, a_sfr, b_sfr)

    else:
        log10_mh = np.log10(mh_uniq_sorted)
        log10_sfr = np.log10(sfr_uniq_sorted)
        a_ms, b_ms = np.polyfit(log10_mh, log10_sfr, 1)
        bins = 2
        binned_masses = np.linspace(
            np.log10(min(mh_uniq_sorted)), np.log10(max(mh_uniq_sorted)), bins
        )
        means_ms = np.zeros(bins - 1)
        number_of_masses = np.zeros(bins - 1)
        err_ms = np.zeros(bins - 1)

        for i, mass in enumerate(mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if (
                    j < (bins - 2)
                    and (np.log10(mass) >= bining)
                    and (np.log10(mass) < binned_masses[j + 1])
                ):
                    line = a_ms * np.log10(mass) + b_ms
                    err_ms[j] += (line - np.log10(sfr_uniq_sorted[i])) ** 2
                    number_of_masses[j] += 1
                    means_ms[j] += mass
                    break
                elif j == bins - 2:
                    line = a_ms * np.log10(mass) + b_ms

                    err_ms[bins - 2] += (line - log10_sfr[i]) ** 2
                    number_of_masses[bins - 2] += 1
                    means_ms[bins - 2] += mass
                    break
        err_ms = np.sqrt(err_ms / number_of_masses)
        means_ms = np.sqrt(means_ms / number_of_masses)

        return (means_ms, err_ms, a_ms, b_ms)


def give_me_sigma(mass, means, errs, a, b):
    """Return only stuff that's needed."""
    if len(means) == 1:
        return a, b, errs
    else:
        i = np.interp(np.log10(mass), np.log10(means), np.sqrt(np.array(errs)))
        return a, b, i


def give_me_lx(sfr_data, z_sample=None):
    """Return Lx over SFR relation, with optional metallicity."""
    if z_sample:
        indices = [7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2]

    sfrdot01_x = [
        38.16,
        38.16,
        38.16,
        38.16,
        38.16,
        38.15,
        38.15,
        38.15,
        38.14,
        38.14,
        38.14,
        38.14,
    ]
    sfrdot01_s = [
        0.75,
        0.77,
        0.77,
        0.77,
        0.74,
        0.73,
        0.72,
        0.70,
        0.69,
        0.69,
        0.68,
        0.67,
    ]
    sfrdot1_x = [
        38.82,
        38.89,
        38.93,
        38.89,
        38.83,
        38.77,
        38.71,
        38.67,
        38.65,
        38.62,
        38.62,
        38.60,
    ]
    sfrdot1_s = [
        1.530,
        1.560,
        1.590,
        1.640,
        1.650,
        1.580,
        1.370,
        1.000,
        0.750,
        0.640,
        0.580,
        0.550,
    ]
    sfr1_x = [
        39.92,
        40.04,
        40.11,
        40.13,
        40.09,
        40.00,
        39.85,
        39.63,
        39.37,
        39.14,
        39.01,
        38.95,
    ]
    sfr1_s = [
        0.310,
        0.300,
        0.310,
        0.330,
        0.380,
        0.460,
        0.590,
        0.670,
        0.570,
        0.590,
        0.490,
        0.380,
    ]
    sfr10_x = [
        39.95,
        40.07,
        40.14,
        40.16,
        40.14,
        40.07,
        39.96,
        39.81,
        39.64,
        39.46,
        36.28,
        39.13,
    ]
    sfr10_s = [
        0.080,
        0.080,
        0.080,
        0.080,
        0.090,
        0.110,
        0.130,
        0.160,
        0.190,
        0.200,
        0.220,
        0.200,
    ]
    sfr100_x = [
        39.96,
        40.07,
        40.14,
        40.17,
        40.14,
        40.07,
        39.97,
        39.83,
        39.67,
        39.50,
        39.33,
        39.19,
    ]
    sfr100_s = [
        0.020,
        0.020,
        0.020,
        0.030,
        0.030,
        0.030,
        0.040,
        0.050,
        0.060,
        0.070,
        0.070,
        0.070,
    ]

    if z_sample:
        sfrdot01_x_sample = np.interp(z_sample, indices, sfrdot01_x)
        sfrdot01_s_sample = np.interp(z_sample, indices, sfrdot01_s)
        sfrdot1_x_sample = np.interp(z_sample, indices, sfrdot1_x)
        sfrdot1_s_sample = np.interp(z_sample, indices, sfrdot1_s)
        sfr1_x_sample = np.interp(z_sample, indices, sfr1_x)
        sfr1_s_sample = np.interp(z_sample, indices, sfr1_s)
        sfr10_x_sample = np.interp(z_sample, indices, sfr10_x)
        sfr10_s_sample = np.interp(z_sample, indices, sfr10_s)
        sfr100_x_sample = np.interp(z_sample, indices, sfr100_x)
        sfr100_s_sample = np.interp(z_sample, indices, sfr100_s)

    if z_sample is None:
        sfrdot01_mean_x = np.mean(sfrdot01_x)
        sfrdot1_mean_x = np.mean(sfrdot1_x)
        sfr1_mean_x = np.mean(sfr1_x)
        sfr10_mean_x = np.mean(sfr10_x)
        sfr100_mean_x = np.mean(sfr100_x)

        sfrd01_mean_s = np.mean(sfrdot01_s)
        sfrd1_mean_s = np.mean(sfrdot1_s)
        sfr1_mean_s = np.mean(sfr1_s)
        sfr10_mean_s = np.mean(sfr10_s)
        sfr100_mean_s = np.mean(sfr100_s)

        sfrdot01_tot_s = np.sqrt(sfrd01_mean_s ** 2 + np.std(sfrdot01_x) ** 2)
        sfrdot1_tot_s = np.sqrt(sfrd1_mean_s ** 2 + np.std(sfrdot1_x) ** 2)
        sfr1_tot_s = np.sqrt(sfr1_mean_s ** 2 + np.std(sfr1_x) ** 2)
        sfr10_tot_s = np.sqrt(sfr10_mean_s ** 2 + np.std(sfr10_x) ** 2)
        sfr100_tot_s = np.sqrt(sfr100_mean_s ** 2 + np.std(sfr100_x) ** 2)

    sfr = [0.01, 0.1, 1.0, 10.0, 100.0]
    if z_sample is None:
        values = [
            sfrdot01_mean_x,
            sfrdot1_mean_x,
            sfr1_mean_x,
            sfr10_mean_x,
            sfr100_mean_x,
        ]
        values_sigma = [
            sfrdot01_tot_s,
            sfrdot1_tot_s,
            sfr1_tot_s,
            sfr10_tot_s,
            sfr100_tot_s,
        ]

    if z_sample:
        if hasattr(sfrdot01_x_sample, '__len__'):
            values = [
                sfrdot01_x_sample[0],
                sfrdot1_x_sample[0],
                sfr1_x_sample[0],
                sfr10_x_sample[0],
                sfr100_x_sample[0],
            ]
            values_sigma = [
                sfrdot01_s_sample[0],
                sfrdot1_s_sample[0],
                sfr1_s_sample[0],
                sfr10_s_sample[0],
                sfr100_s_sample[0],
            ]
        else:
            values = [
                sfrdot01_x_sample,
                sfrdot1_x_sample,
                sfr1_x_sample,
                sfr10_x_sample,
                sfr100_x_sample,
            ]
            values_sigma = [
                sfrdot01_s_sample,
                sfrdot1_s_sample,
                sfr1_s_sample,
                sfr10_s_sample,
                sfr100_s_sample,
            ]

    loglxoversfr = np.interp(sfr_data, sfr, values)
    logsigma = np.interp(sfr_data, sfr, values_sigma)

    return loglxoversfr, logsigma


def bias_function(m, hmf_class):
    """Return bias factor factor."""
    a_bias = 0.707
    b_bias = 0.5
    c_bias = 0.6
    nu = hmf_class.Deltac / hmf_class.dicke() / np.sqrt(hmf_class.sigma_z0(m))
    nu_1 = nu * np.sqrt(a_bias)
    return 1 + 1 / (hmf_class.Deltac / hmf_class.dicke()) * (
        nu_1 ** 2
        + b_bias * nu_1 ** (2 * (1 - c_bias))
        - (nu_1 ** (2 * c_bias) / np.sqrt(a_bias))
        / (nu_1 ** (2 * c_bias) + b_bias * (1 - c_bias) * (1 - c_bias / 2))
    )


def bias_func(m, hmf_class):  # this supposes that the hmf class was initiated
    """Return bias factor."""
    hmf_value = hmf_class.dnbiasdm(m)
    bias_value = bias_function(m, hmf_class)
    return m * hmf_value * bias_value


def delta_scc(
    hmf_class,
):
    """Return delta factor needed."""
    bias_integral = integrate.quad(
        bias_func,
        10 ** hmf_class.log10_mass_min,
        10 ** hmf_class.log10_mass_max,
        args=(hmf_class,),
    )[0]
    cmpermpc3 = 3.086e24 ** 3
    msun = 1.989e33
    critical_density = Cosmo.critical_density0.value * cmpermpc3 / msun
    return (
        (hmf_class.cumulative_number()[0] / critical_density) ** 2
        * hmf_class.sigma_cell() ** 2
        * (bias_integral) ** 2
    )


def metalicity_from_fmr(m_star, sfr):
    """
    Metalicity from Curti+19.

    -----

    Function takes in stellar mass and SFR, outputs the metallicity 12+log(O/H)
    """
    sigma_met = 0.054  # value they qoute for the sigma_FMR
    z_0 = 8.779
    gamma = 0.31
    beta = 2.1
    m_0 = 10.11
    m_1 = 0.56
    bigm_0 = 10 ** (m_0 + m_1 * np.log10(sfr))
    met = z_0 - gamma / beta * np.log10(1 + (m_star / bigm_0) ** (-beta))
    return met, sigma_met


emissivities, _, _, _, _ = sampler(n_iter=5000)
np.save('emissivities_z10_full.npy', np.array(emissivities))
