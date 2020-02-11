__author__ = 'sibirrer'

#this file contains a class to convert units

import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.background import Background
from lenstronomy.Cosmo.nfw_param import NFWParam


class LensCosmo(object):
    """
    class to manage the physical units and distances present in a single plane lens with fixed input cosmology
    """
    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy.cosmology instance
        """

        self.z_lens = z_lens
        self.z_source = z_source
        self.background = Background(cosmo=cosmo)
        self.nfw_param = NFWParam()

    def a_z(self, z):
        """
        convert redshift into scale factor
        :param z: redshift
        :return: scale factor
        """
        return 1. / (1. + z)

    @property
    def h(self):
        return self.background.cosmo.H(0).value / 100.

    @property
    def D_d(self):
        """

        :return: angular diameter distance to the deflector [Mpc]
        """
        return self.background.D_xy(0, self.z_lens)

    @property
    def D_s(self):
        """

        :return: angular diameter distance to the source [Mpc]
        """
        return self.background.D_xy(0, self.z_source)

    @property
    def D_ds(self):
        """

        :return: angular diameter distance from deflector to source [Mpc]
        """
        return self.background.D_xy(self.z_lens, self.z_source)

    @property
    def D_dt(self):
        """

        :return: time delay distance [Mpc]
        """
        return (1 + self.z_lens) * self.D_d * self.D_s / self.D_ds

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        :return: critical projected mass density
        """
        if not hasattr(self, '_Epsilon_Crit'):
            const_SI = const.c ** 2 / (4 * np.pi * const.G)  #c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._Epsilon_Crit = self.D_s/(self.D_d*self.D_ds) * factor #[M_sun/Mpc^2]
        return self._Epsilon_Crit

    @property
    def epsilon_crit_angle(self):
        """
        returns the critical projected mass density in units of M_sun/arcsec^2 (in physical solar mass units)
        :return: critical projected mass density
        """
        if not hasattr(self, '_Epsilon_Crit_arcsec'):
            const_SI = const.c ** 2 / (4 * np.pi * const.G)  # c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._Epsilon_Crit_arcsec = self.D_s / (self.D_d * self.D_ds) * factor * (self.D_d * const.arcsec)**2  # [M_sun/arcsec^2]
        return self._Epsilon_Crit_arcsec

    def phys2arcsec_lens(self, phys):
        """
        convert physical Mpc into arc seconds
        :param phys: physical distance [Mpc]
        :return: angular diameter [arcsec]
        """
        return phys / self.D_d/ const.arcsec

    def arcsec2phys_lens(self, arcsec):
        """
        convert angular to physical quantities for lens plane
        :param arcsec: angular size at lens plane [arcsec]
        :return: physical size at lens plane [Mpc]
        """
        return arcsec * const.arcsec * self.D_d

    def arcsec2phys_source(self, arcsec):
        """
        convert angular to physical quantities for source plane
        :param arcsec: angular size at source plane [arcsec]
        :return: physical size at source plane [Mpc]
        """
        return arcsec * const.arcsec * self.D_s

    def kappa2proj_mass(self, kappa):
        """
        convert convergence to projected mass M_sun/Mpc^2
        :param kappa: lensing convergence
        :return: projected mass [M_sun/Mpc^2]
        """
        return kappa * self.epsilon_crit

    def mass_in_theta_E(self, theta_E):
        """
        mass within Einstein radius (area * epsilon crit) [M_sun]
        :param theta_E: Einstein radius [arcsec]
        :return: mass within Einstein radius [M_sun]
        """
        mass = self.arcsec2phys_lens(theta_E) ** 2 * np.pi * self.epsilon_crit
        return mass

    def mass_in_coin(self, theta_E):
        """

        :param theta_E: Einstein radius [arcsec]
        :return: mass in coin calculated in mean density of the universe
        """
        chi_L = self.background.T_xy(0, self.z_lens)
        chi_S = self.background.T_xy(0, self.z_source)
        return 1./3 * np.pi * (chi_L * theta_E * const.arcsec) ** 2 * chi_S * self.background.rho_crit  #[M_sun/Mpc**3]

    def time_delay_units(self, fermat_pot, kappa_ext=0):
        """

        :param fermat_pot: in units of arcsec^2 (e.g. Fermat potential)
        :param kappa_ext: unit-less external shear not accounted for in the Fermat potential
        :return: time delay in days
        """
        D_dt = self.D_dt * (1. - kappa_ext) * const.Mpc  # eqn 7 in Suyu et al.
        return D_dt / const.c * fermat_pot / const.day_s * const.arcsec ** 2  # * self.arcsec2phys_lens(1.)**2

    def time_delay2fermat_pot(self, dt):
        """

        :param dt: time delay in units of days
        :return: Fermat potential in units arcsec**2 for a given cosmology
        """
        D_dt = self.D_dt * const.Mpc
        return dt * const.c * const.day_s / D_dt / const.arcsec ** 2

    def nfw_angle2physical(self, Rs_angle, alpha_Rs):
        """
        converts the angular parameters into the physical ones for an NFW profile

        :param alpha_Rs: observed bending angle at the scale radius in units of arcsec
        :param Rs: scale radius in units of arcsec
        :return: M200, r200, Rs_physical, c
        """
        Rs = Rs_angle * const.arcsec * self.D_d
        theta_scaled = alpha_Rs * self.epsilon_crit * self.D_d * const.arcsec
        rho0 = theta_scaled / (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        rho0_com = rho0 / self.h**2 * self.a_z(self.z_lens)**3
        c = self.nfw_param.c_rho0(rho0_com)
        r200 = c * Rs
        M200 = self.nfw_param.M_r200(r200 * self.h / self.a_z(self.z_lens)) / self.h
        return rho0, Rs, c, r200, M200

    def nfw_physical2angle(self, M, c):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities

        :param M: mass enclosed 200 rho_crit in units of M_sun
        :param c: NFW concentration parameter (r200/r_s)
        :return: alpha_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        rho0, Rs, r200 = self.nfwParam_physical(M, c)
        Rs_angle = Rs / self.D_d / const.arcsec  # Rs in arcsec
        alpha_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return Rs_angle,  alpha_Rs / self.epsilon_crit / self.D_d / const.arcsec

    def nfwParam_physical(self, M, c):
        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """
        r200 = self.nfw_param.r200_M(M * self.h) / self.h * self.a_z(self.z_lens)  # physical radius r200
        rho0 = self.nfw_param.rho0_c(c) * self.h**2 / self.a_z(self.z_lens)**3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def nfw_M_theta_vir(self, M):
        """
        returns virial radius in angular units of arc seconds on the sky

        :param M: physical mass in M_sun
        :return: angle (in arc seconds) of the virial radius
        """
        r200 = self.nfw_param.r200_M(M * self.h) / self.h * self.a_z(self.z_lens)  # physical radius r200
        theta_r200 = r200 / self.D_d / const.arcsec
        return theta_r200

    def sis_theta_E2sigma_v(self, theta_E):
        """
        converts the lensing Einstein radius into a physical velocity dispersion
        :param theta_E: Einstein radius (in arcsec)
        :return: velocity dispersion in units (km/s)
        """
        v_sigma_c2 = theta_E * const.arcsec / (4*np.pi) * self.D_s / self.D_ds
        return np.sqrt(v_sigma_c2)*const.c / 1000

    def sis_sigma_v2theta_E(self, v_sigma):
        """
        converts the velocity dispersion into an Einstein radius for a SIS profile
        :param v_sigma: velocity dispersion (km/s)
        :return: theta_E (arcsec)
        """
        theta_E = 4 * np.pi * (v_sigma * 1000./const.c)**2 * self.D_ds / self.D_s / const.arcsec
        return theta_E
