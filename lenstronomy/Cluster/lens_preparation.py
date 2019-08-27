import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.LensModel.numeric_lens_differentials import NumericLens
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class LensPreparation(object):
    """
    This class is used to do initialize kwargs of the lens models according to the given lens_model_list.
    (It's easy and free to choose lens_model_list, but a little tricky to generate corresponding kwargs.)
    For the situation that you know deflection maps, then you want know shear, convergency, and shift ....
    Or, you know, shear, convergency, shift, then you wanna know deflection maps.

    """
    def __init__(self,alphax,alphay,zl,zs):
       """
       :param alphax: the x deflection map corresponds to data in dataclass
       :param alphay: the y deflection map corresponds to data in dataclass
       :param zl: redshift of the lens
       :param zs:  redshift of the lens
       :param lenseq:
       """
       cosmo = LensCosmo(zl, zs)
       dlsds = cosmo.D_ds / cosmo.D_s
       self.alphax = alphax* dlsds
       self.alphay = alphay* dlsds


    def initial_kwargs_lens(self,x,y,kwargs_data,alphax_shift=0, alphay_shift=0,lens_model_list=['SHIFT','SHEAR','CONVERGENCE','FLEXIONFG']):
        """
        This function returns list type of kwargs of lens models.
        :param lens_model_list: list of strings with lens model names
        :param alphax_shift: shift the source's x position
        :param alphay_shift: shift the source's y position
        :return:  a list of kwargs of lens models corresponding to the lens models existed in lens_model_list
        """
        imageData = ImageData(**kwargs_data)
        r_cut = (np.shape(imageData.data)[0] - 1) / 2
        alphax = self.alphax[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
        alphay = self.alphay[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
        xaxes, yaxes = imageData.pixel_coordinates
        ra_center = xaxes[r_cut + 1, r_cut + 1]
        dec_center = yaxes[r_cut + 1, r_cut + 1]
        kwargs_lens_in = [{'grid_interp_x': xaxes[0], 'grid_interp_y': yaxes[:, 0], 'f_x': alphax,
                           'f_y': alphay}]

        kwargs_lens=[]
        for lens_type in lens_model_list:
            if lens_type=='INTERPOL':
                kwargs_lens.append({'grid_interp_x': xaxes[0], 'grid_interp_y': yaxes[:,0], 'f_x': alphax, 'f_y': alphay})
            elif lens_type=='SHIFT':
                alpha_x_center = alphax[r_cut + 1, r_cut + 1]
                alpha_y_center = alphay[r_cut + 1, r_cut + 1]
                kwargs_lens.append({'alpha_x': alpha_x_center - alphax_shift, 'alpha_y': alpha_y_center - alphay_shift})
            elif lens_type == 'SHEAR':
                gamma1, gamma2 = NumericLens(['INTERPOL']).gamma(util.image2array(xaxes),
                                                                 util.image2array(yaxes), kwargs=kwargs_lens_in)
                gamma_1_center, gamma_2_center = gamma1.mean(), gamma2.mean()
                kwargs_lens.append({'e1': gamma_1_center, 'e2': gamma_2_center, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'CONVERGENCE':
                kappa = NumericLens(['INTERPOL']).kappa(util.image2array(xaxes),
                                                        util.image2array(yaxes), kwargs=kwargs_lens_in)
                kappa_center = kappa.mean()
                kwargs_lens.append({'kappa_ext': kappa_center, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'FLEXION':
                g1, g2, g3, g4 = NumericLens(['INTERPOL']).Dmatrix(util.image2array(xaxes),
                                                                   util.image2array(yaxes), kwargs=kwargs_lens_in)
                g1_c, g2_c, g3_c, g4_c = g1.mean(), g2.mean(), g3.mean(), g4.mean()
                kwargs_lens.append({'g1': g1_c,'g2':g2_c,'g3':g3_c,'g4':g4_c, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'FLEXIONFG':
                g1, g2, g3, g4 = NumericLens(['INTERPOL']).Dmatrix(util.image2array(xaxes),
                                                                   util.image2array(yaxes), kwargs=kwargs_lens_in)
                g1_c, g2_c, g3_c, g4_c = g1.mean(), g2.mean(), g3.mean(), g4.mean()
                F1_c = (g1_c + g3_c) * 0.5
                F2_c = (g2_c + g4_c) * 0.5
                G1_c = (g1_c - g3_c) * 0.5 - g3_c
                G2_c = (g2_c - g4_c) * 0.5 - g4_c
                kwargs_lens.append({'F1': F1_c, 'F2': F2_c, 'G1': G1_c, 'G2': G2_c, 'ra_0': ra_center, 'dec_0': dec_center})
        magnification=NumericLens(['INTERPOL']).magnification(util.image2array(xaxes),
                                                                 util.image2array(yaxes), kwargs=kwargs_lens_in)
        magnification=np.abs(magnification.mean())
        return kwargs_lens, magnification


