__author__ = 'sibirrer'

import numpy as np
import copy
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.multi_gauss_expansion as mge
import lenstronomy.Util.constants as const


class LensProp(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc
    for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_model, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param kwargs_model: model keyword arguments
        :param cosmo: astropy.cosmology instance
        """
        self.z_d = z_lens
        self.z_s = z_source
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        self.lens_analysis = LensAnalysis(kwargs_model)
        self._lensModelExt = LensModelExtensions(self.lens_analysis.LensModel)
        self.kwargs_options = kwargs_model
        self._kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}

    def time_delays(self, kwargs_lens, kwargs_ps, kappa_ext=0):
        """
        predicts the time delays of the image positions

        :param kwargs_lens: lens model parameters
        :param kwargs_ps: point source parameters
        :param kappa_ext: external convergence (optional)
        :return: time delays at image positions for the fixed cosmology
        """
        fermat_pot = self.lens_analysis.fermat_potential(kwargs_lens, kwargs_ps)
        time_delay = self.lensCosmo.time_delay_units(fermat_pot, kappa_ext)
        return time_delay

    def velocity_dispersion(self, kwargs_lens, r_eff, R_slit, dR_slit, psf_fwhm, aniso_param=1, psf_type='GAUSSIAN',
                            moffat_beta=2.6, num_evaluate=1000):
        """
        computes the LOS velocity dispersion of the lens within a slit of size R_slit x dR_slit and seeing psf_fwhm.
        The assumptions are a Hernquist light profile and the spherical power-law lens model at the first position.

        Further information can be found in the AnalyticKinematics() class.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: deflector light parameters
        :param aniso_param: scaled r_ani with respect to the half light radius
        :param r_eff: half light radius, if not provided, will be computed from the lens light model
        :param R_slit: width of the slit
        :param dR_slit: length of the slit
        :param psf_fwhm: full width at half maximum of the seeing (Gaussian form)
        :param psf_type: string, point spread functino type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param moffat_beta: float, beta parameter of Moffat profile
        :param num_evaluate: number of spectral rendering of the light distribution that end up on the slit
        :return: velocity dispersion in units [km/s]
        """
        gamma = kwargs_lens[0]['gamma']
        theta_E = kwargs_lens[0]['theta_E']
        r_ani = aniso_param * r_eff
        analytic_kinematics = AnalyticKinematics(fwhm=psf_fwhm, moffat_beta=moffat_beta, psf_type=psf_type, **self._kwargs_cosmo)
        sigma2 = analytic_kinematics.vel_disp(gamma, theta_E, r_eff, r_ani, R_slit, dR_slit, rendering_number=num_evaluate)
        return sigma2

    def velocity_dispersion_numerical(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, psf_fwhm,
                                      aperture_type, anisotropy_model, r_eff, psf_type='GAUSSIAN', moffat_beta=2.6, kwargs_numerics={}, MGE_light=False,
                                      MGE_mass=False, lens_model_kinematics_bool=None, light_model_kinematics_bool=None,
                                      Hernquist_approx=False):
        """
        Computes the LOS velocity dispersion of the deflector galaxy with arbitrary combinations of light and mass models.
        For a detailed description, visit the description of the Galkin() class.
        Additionaly to executing the Galkin routine, it has an optional Multi-Gaussian-Expansion decomposition of lens
        and light models that do not have a three-dimensional distribution built in, such as Sersic profiles etc.

        The center of all the lens and lens light models that are part of the kinematic estimate must be centered on the
        same point.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: lens light parameters
        :param kwargs_anisotropy: anisotropy parameters (see Galkin module)
        :param kwargs_aperture: aperture parameters (see Galkin module)
        :param psf_fwhm: full width at half maximum of the seeing (Gaussian form)
        :param psf_type: string, point spread functino type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param moffat_beta: float, beta parameter of Moffat profile
        :param aperture_type: type of aperture (see Galkin module
        :param anisotropy_model: stellar anisotropy model (see Galkin module)
        :param r_eff: a rough estimate of the half light radius of the lens light in case of computing the MGE of the
         light profile
        :param kwargs_numerics: keyword arguments that contain numerical options (see Galkin module)
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore substructure, shear etc that do not describe the
            main deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore light components that do not describe the main
            deflector
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched to the half light radius of the deflector light profile to compute the kinematics
        :return: LOS velocity dispersion [km/s]
        """

        kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}

        mass_profile_list, kwargs_profile, light_profile_list, kwargs_light = self.kinematic_profiles(kwargs_lens,
                                                                                        kwargs_lens_light, r_eff=r_eff,
                                                                                        MGE_light=MGE_light, MGE_mass=MGE_mass,
                                                                                        lens_model_kinematics_bool=lens_model_kinematics_bool,
                                                                                        light_model_kinematics_bool=light_model_kinematics_bool,
                                                                                        Hernquist_approx=Hernquist_approx)
        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type,
                        anisotropy_model=anisotropy_model, fwhm=psf_fwhm, psf_type=psf_type, moffat_beta=moffat_beta,
                        kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        sigma2 = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture)
        return sigma2

    def kinematic_profiles(self, kwargs_lens, kwargs_lens_light, r_eff, MGE_light=False, MGE_mass=False,
                           lens_model_kinematics_bool=None, light_model_kinematics_bool=None, Hernquist_approx=False):
        """
        translates the lenstronomy lens and mass profiles into a (sub) set of profiles that are compatible with the GalKin module to compute the kinematics thereof.

        :param kwargs_lens: lens model parameters
        :param kwargs_lens_light: lens light parameters
        :param r_eff: a rough estimate of the half light radius of the lens light in case of computing the MGE of the
         light profile
        :param MGE_light: bool, if true performs the MGE for the light distribution
        :param MGE_mass: bool, if true performs the MGE for the mass distribution
        :param lens_model_kinematics_bool: bool list of length of the lens model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore substructure, shear etc that do not describe the
            main deflector potential
        :param light_model_kinematics_bool: bool list of length of the light model. Only takes a subset of all the models
            as part of the kinematics computation (can be used to ignore light components that do not describe the main
            deflector
        :param Hernquist_approx: bool, if True, uses a Hernquist light profile matched to the half light radius of the deflector light profile to compute the kinematics
        :return: mass_profile_list, kwargs_profile, light_profile_list, kwargs_light
        """

        mass_profile_list = []
        kwargs_profile = []
        if lens_model_kinematics_bool is None:
            lens_model_kinematics_bool = [True] * len(kwargs_lens)
        for i, lens_model in enumerate(self.kwargs_options['lens_model_list']):
            if lens_model_kinematics_bool[i] is True:
                mass_profile_list.append(lens_model)
                if lens_model in ['INTERPOL', 'INTERPOL_SCLAED']:
                    center_x, center_y = self._lensModelExt.lens_center(kwargs_lens, k=i)
                    kwargs_lens_i = copy.deepcopy(kwargs_lens[i])
                    kwargs_lens_i['grid_interp_x'] -= center_x
                    kwargs_lens_i['grid_interp_y'] -= center_y
                else:
                    kwargs_lens_i = {k: v for k, v in kwargs_lens[i].items() if not k in ['center_x', 'center_y']}
                kwargs_profile.append(kwargs_lens_i)

        if MGE_mass is True:
            lensModel = LensModel(lens_model_list=mass_profile_list)
            massModel = LensModelExtensions(lensModel)
            theta_E = massModel.effective_einstein_radius(kwargs_profile)
            r_array = np.logspace(-4, 2, 200) * theta_E
            mass_r = lensModel.kappa(r_array, np.zeros_like(r_array), kwargs_profile)
            amps, sigmas, norm = mge.mge_1d(r_array, mass_r, N=20)
            mass_profile_list = ['MULTI_GAUSSIAN_KAPPA']
            kwargs_profile = [{'amp': amps, 'sigma': sigmas}]

        light_profile_list = []
        kwargs_light = []
        if light_model_kinematics_bool is None:
            light_model_kinematics_bool = [True] * len(kwargs_lens_light)
        for i, light_model in enumerate(self.kwargs_options['lens_light_model_list']):
            if light_model_kinematics_bool[i]:
                light_profile_list.append(light_model)
                kwargs_lens_light_i = {k: v for k, v in kwargs_lens_light[i].items() if
                                       not k in ['center_x', 'center_y']}
                if 'e1' in kwargs_lens_light_i:
                    kwargs_lens_light_i['e1'] = 0
                    kwargs_lens_light_i['e2'] = 0
                kwargs_light.append(kwargs_lens_light_i)
        if Hernquist_approx is True:
            light_profile_list = ['HERNQUIST']
            kwargs_light = [{'Rs': r_eff, 'amp': 1.}]
        else:
            if MGE_light is True:
                lightModel = LightModel(light_profile_list)
                r_array = np.logspace(-3, 2, 200) * r_eff * 2
                flux_r = lightModel.surface_brightness(r_array, 0, kwargs_light)
                amps, sigmas, norm = mge.mge_1d(r_array, flux_r, N=20)
                light_profile_list = ['MULTI_GAUSSIAN']
                kwargs_light = [{'amp': amps, 'sigma': sigmas}]
        return mass_profile_list, kwargs_profile, light_profile_list, kwargs_light

    def angular_diameter_relations(self, sigma_v_model, sigma_v, kappa_ext, D_dt_model):
        """

        :return:
        """
        sigma_v2_model = sigma_v_model**2
        Ds_Dds = sigma_v**2/(1-kappa_ext)/(sigma_v2_model * self.lensCosmo.D_ds / self.lensCosmo.D_s)
        D_d = D_dt_model/(1+self.lensCosmo.z_lens)/Ds_Dds/(1-kappa_ext)
        return D_d, Ds_Dds

    def angular_distances(self, sigma_v_measured, time_delay_measured, kappa_ext, sigma_v_modeled, fermat_pot):
        """

        :param sigma_v_measured: velocity dispersion measured [km/s]
        :param time_delay_measured: time delay measured [d]
        :param kappa_ext: external convergence estimated []
        :param sigma_v_modeled: lens model velocity dispersion with default cosmology and without external convergence [km/s]
        :param fermat_pot: fermat potential of lens model, modulo MSD of kappa_ext [arcsec^2]
        :return: D_d and D_d*D_s/D_ds, units in Mpc physical
        """

        Ds_Dds = (sigma_v_measured/float(sigma_v_modeled)) ** 2 / (self.lensCosmo.D_ds / self.lensCosmo.D_s) / (1. - kappa_ext)
        DdDs_Dds = 1./(1+self.lensCosmo.z_lens)/(1. - kappa_ext) * (const.c * time_delay_measured * const.day_s)/(fermat_pot*const.arcsec**2)/const.Mpc
        return Ds_Dds, DdDs_Dds
