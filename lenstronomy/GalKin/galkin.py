from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.mass_profile import MassProfile
from lenstronomy.GalKin.aperture import aperture_select
from lenstronomy.GalKin.anisotropy import MamonLokasAnisotropy
from lenstronomy.GalKin.psf import psf_select
from lenstronomy.GalKin.cosmo import Cosmo
import lenstronomy.GalKin.velocity_util as util
import lenstronomy.Util.constants as const

import numpy as np


class Galkin(object):
    """
    Major class to compute velocity dispersion measurements given light and mass models

    The class supports any mass and light distribution (and superposition thereof) that has a 3d correspondance in their
    2d lens model distribution. For models that do not have this correspondance, you may want to apply a
    Multi-Gaussian Expansion (MGE) on their models and use the MGE to be de-projected to 3d.

    The computation follows Mamon&Lokas 2005 and performs the spectral rendering of the seeing convolved apperture with
    the method introduced by Birrer et al. 2016.

    The class supports various types of anisotropy models (see Anisotropy class) and aperture types (see Aperture class).
    Solving the Jeans Equation requires a numerical integral over the 3d light and mass profile (see Mamon&Lokas 2005).
    This class (as well as the dedicated LightModel and MassModel classes) perform those integral numerically with an
    interpolated grid.

    The seeing convolved integral over the aperture is computed by rendering spectra (light weighted LOS kinematics)
    from the light distribution.

    The cosmology assumed to compute the physical mass and distances are set via the kwargs_cosmo keyword arguments.
        D_d: Angular diameter distance to the deflector (in Mpc)
        D_s: Angular diameter distance to the source (in Mpc)
        D_ds: Angular diameter distance from the deflector to the source (in Mpc)

    The numerical options can be chosen through the kwargs_numerics keywords
        sampling_number: number of spectral rendering to compute the light weighted integrated LOS dispersion within
        the aperture. This keyword should be chosen high enough to result in converged results within the tolerance.

        interpol_grid_num: number of interpolation points in the light and mass profile (radially). This number should
        be chosen high enough to accurately describe the true light profile underneath.
        log_integration: bool, if True, performs the interpolation and numerical integration in log-scale.

        max_integrate: maximum 3d radius to where the numerical integration of the Jeans Equation solver is made.
        This value should be large enough to contain most of the light and to lead to a converged result.
        min_integrate: minimal integration value. This value should be very close to zero but some mass and light
        profiles are diverging and a numerically stabel value should be chosen.

    These numerical options should be chosen to allow for a converged result (within your tolerance) but not too
    conservative to impact too much the computational cost. Reasonable values might depend on the specific problem.

    """
    def __init__(self, mass_profile_list, light_profile_list, kwargs_aperture, kwargs_psf, anisotropy_model='isotropic',
                 kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500},
                 sampling_number=1000, interpol_grid_num=500, log_integration=False, max_integrate=10, min_integrate=0.001):
        """

        :param mass_profile_list: list of lens (mass) model profiles
        :param light_profile_list: list of light model profiles of the lensing galaxy
        :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class.
        :param kwargs_psf: keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances involved
        """
        self.massProfile = MassProfile(mass_profile_list, kwargs_cosmo, interpol_grid_num=interpol_grid_num,
                                         max_interpolate=max_integrate, min_interpolate=min_integrate)
        self.lightProfile = LightProfile(light_profile_list, interpol_grid_num=interpol_grid_num,
                                         max_interpolate=max_integrate, min_interpolate=min_integrate)
        self.aperture = aperture_select(**kwargs_aperture)
        self.anisotropy = MamonLokasAnisotropy(anisotropy_model)

        self.cosmo = Cosmo(**kwargs_cosmo)
        self._num_sampling = sampling_number
        self._interp_grid_num = interpol_grid_num
        self._log_int = log_integration
        self._max_integrate = max_integrate  # maximal integration (and interpolation) in units of arcsecs
        self._min_integrate = min_integrate  # min integration (and interpolation) in units of arcsecs
        self._psf = psf_select(**kwargs_psf)

    def vel_disp(self, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        sigma2_R_sum = 0
        for i in range(0, self._num_sampling):
            sigma2_R = self._draw_one_sigma2(kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma2_R_sum += sigma2_R
        sigma_s2_average = sigma2_R_sum / self._num_sampling
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        sigma_s2_average *= 2 * const.G  # correcting for integral prefactor
        return np.sqrt(sigma_s2_average/(const.arcsec**2 * self.cosmo.D_d**2 * const.Mpc))/1000.  # in units of km/s

    def _draw_one_sigma2(self, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in angular units for a single draw of the light distribution that
         falls in the aperture after displacing with the seeing
        """
        while True:
            R = self.lightProfile.draw_light_2d(kwargs_light, n=1)[0]  # draw r in arcsec
            x, y = util.draw_xy(R)  # draw projected R in arcsec
            x_, y_ = self._psf.displace_psf(x, y)
            bool = self.aperture.aperture_select(x_, y_)
            if bool is True:
                break
        sigma2_R = self._sigma2_R(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        return sigma2_R

    def _sigma2_R(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified projected radius

        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return:
        """
        I_R_sigma2 = self._I_R_simga2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R = self.lightProfile.light_2d(R, kwargs_light)
        return I_R_sigma2 / I_R

    def _I_R_simga2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        equation A15 in Mamon&Lokas 2005 as a logarithmic numerical integral (if option is chosen)
        modulo pre-factor 2*G

        :param R: 2d projected radius (in angular units)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integral of A15 in Mamon&Lokas 2005
        """
        R = max(R, self._min_integrate)
        if self._log_int is True:
            min_log = np.log10(R+0.001)
            max_log = np.log10(self._max_integrate)
            r_array = np.logspace(min_log, max_log, self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dlog_r * r_array
        else:
            r_array = np.linspace(R+0.001, self._max_integrate, self._interp_grid_num)
            dr = r_array[2] - r_array[1]
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dr
        IR_sigma2 = np.sum(IR_sigma2_dr) * const.arcsec * self.cosmo.D_d  # integral from angle to physical scales
        return IR_sigma2

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space) in Mamon&Lokas 2005

        :param r: 3d radius in arc seconds
        :param R: 2d projected radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return:
        """
        k_r = self.anisotropy.K(r, R, kwargs_anisotropy)
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        m_r = self.massProfile.mass_3d_interp(r, kwargs_mass)
        out = k_r * l_r * m_r / r
        return out
