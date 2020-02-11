__author__ = 'sibirrer'

import numpy as np
import lenstronomy.GalKin.velocity_util as vel_util
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.GalKin.psf import psf_select
from lenstronomy.GalKin.aperture import aperture_select
import lenstronomy.Util.constants as const
import math


class AnalyticKinematics(object):
    """
    class to compute eqn 20 in Suyu+2010 with a Monte-Carlo from rendering from the
    light profile distribution and displacing them with a Gaussian seeing convolution

    This class assumes spherical symmetry in light and mass distribution and
        - a Hernquist light profile (parameterised by the half-light radius)
        - a power-law mass profile (parameterized by the Einstein radius and logarithmic slop)

    The analytic equations for the kinematics in this approximation are presented e.g. in Suyu et al. 2010 and
    the spectral rendering approach to compute the seeing convolved slit measurement is presented in Birrer et al. 2016.
    The stellar anisotropy is parameterised based on Osipkov 1979; Merritt 1985.

    Units
    -----
    all units are meant to be in angular arc seconds. The physical units are fold in through the angular diameter
    distances

    """
    def __init__(self, D_d, D_s, D_ds, kwargs_aperture, kwargs_psf):
        """

        :param D_d: angular diameter to the deflector [MPC]
        :param D_s: angular diameter to the source [MPC]
        :param D_ds: angular diameter from the deflector to the source [MPC]
        :param psf_type: string, point spread functino type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param fwhm: full width at half maximum seeing condition
        :param moffat_beta: float, beta parameter of Moffat profile
        """
        if D_ds <= 0 or D_s <= 0 or D_d <=0:
            raise ValueError('input angular diameter distances Dd: %s, Ds: %s, Dds: %s are not suppored for a lens model!' % (D_d, D_s, D_ds) )
        self._cosmo = Cosmo(D_d=D_d, D_s=D_s, D_ds=D_ds)
        self._psf = psf_select(**kwargs_psf)
        self.aperture = aperture_select(**kwargs_aperture)

    def vel_disp(self, gamma, theta_E, r_eff, r_ani, rendering_number=1000):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)

        :param gamma: power-law slope of the mass profile (isothermal = 2)
        :param theta_E: Einstein radius of the lens (in arcseconds)
        :param r_eff: half light radius of the Hernquist profile (or as an approximation of any other profile to be described as a Hernquist profile
        :param r_ani: anisotropy radius
        :param kwargs_aperture: keyword arguments describing the aperture of the collected spectral
        :param rendering_number: number of spectral renderings drawn from the light distribution that go through the
            slit of the observations

        :return: LOS integrated velocity dispersion in units [km/s]
        """
        sigma_s2_sum = 0
        rho0_r0_gamma = self._rho0_r0_gamma(theta_E, gamma)
        for i in range(0, rendering_number):
            sigma_s2_draw = self.vel_disp_one(gamma, rho0_r0_gamma, r_eff, r_ani)
            sigma_s2_sum += sigma_s2_draw
        sigma_s2_average = sigma_s2_sum / rendering_number
        return np.sqrt(sigma_s2_average)

    def _rho0_r0_gamma(self, theta_E, gamma):
        # equation (14) in Suyu+ 2010
        return -1 * math.gamma(gamma/2) / (np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E ** gamma / \
               self._cosmo.arcsec2phys_lens(theta_E) * self._cosmo.epsilon_crit * const.M_sun / const.Mpc ** 3

    def vel_disp_one(self, gamma, rho0_r0_gamma, r_eff, r_ani):
        """
        computes one realisation of the velocity dispersion realized in the slit

        :param gamma: power-law slope of the mass profile (isothermal = 2)
        :param rho0_r0_gamma: combination of Einstein radius and power-law slope as equation (14) in Suyu+ 2010
        :param r_eff: half light radius of the Hernquist profile (or as an approximation of any other profile to be described as a Hernquist profile
        :param r_ani: anisotropy radius
        :param kwargs_aperture: keyword arguments describing the aperture of the collected spectral
        :param FWHM: full width at half maximum of the seeing conditions, described as a Gaussian
        :return: projected velocity dispersion of a single drawn position in the potential [km/s]
        """
        a = 0.551 * r_eff
        while True:
            r = self.P_r(a)  # draw r
            R, x, y = self.R_r(r)  # draw projected R
            x_, y_ = self._psf.displace_psf(x, y)
            bool = self.aperture.aperture_select(x_, y_)
            if bool is True:
                break
        sigma_s2 = self.sigma_s2(r, R, r_ani, a, gamma, rho0_r0_gamma)
        return sigma_s2

    def P_r(self, a):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r

    def R_r(self, r):
        """
        draws a random projection from radius r in 2d and 1d
        :param r: 3d radius
        :return: R, x, y
        """
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        R = np.sqrt(x**2 + y**2)
        return R, x, y

    def sigma_s2(self, r, R, r_ani, a, gamma, rho0_r0_gamma):
        """
        projected velocity dispersion
        :param r:
        :param R:
        :param r_ani:
        :param a:
        :param gamma:
        :param phi_E:
        :return:
        """
        beta = self._beta_ani(r, r_ani)
        return (1 - beta * R**2/r**2) * self.sigma_r2(r, a, gamma, rho0_r0_gamma, r_ani)

    def sigma_r2(self, r, a, gamma, rho0_r0_gamma, r_ani):
        """
        equation (19) in Suyu+ 2010
        """
        # first term
        prefac1 = 4*np.pi * const.G * a**(-gamma) * rho0_r0_gamma / (3-gamma)
        prefac2 = r * (r + a)**3/(r**2 + r_ani**2)
        hyp1 = vel_util.hyp_2F1(a=2+gamma, b=gamma, c=3+gamma, z=1./(1+r/a))
        hyp2 = vel_util.hyp_2F1(a=3, b=gamma, c=1+gamma, z=-a/r)
        fac = r_ani**2/a**2 * hyp1 / ((2+gamma) * (r/a + 1)**(2+gamma)) + hyp2 / (gamma*(r/a)**gamma)
        return prefac1 * prefac2 * fac * (self._cosmo.arcsec2phys_lens(1.) * const.Mpc / 1000) ** 2

    def _beta_ani(self, r, r_ani):
        """
        anisotropy parameter beta
        :param r: radius
        :param r_ani: anisotropy radius
        :return: beta(r) in the OM parameterization
        """
        return r**2/(r_ani**2 + r**2)
