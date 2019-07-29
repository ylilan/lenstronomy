import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.LensModel.numeric_lens_differentials import NumericLens
from lenstronomy.Data.imaging_data import ImageData


class Initkwargs_lens(object):
    """
    This class is used to do initialize kwargs of the lens models according to the given lens_model_list.
    (It's easy and free to choose lens_model_list, but a little tricky to generate corresponding kwargs.)
    For the situation that you know deflection maps, then you want know shear, convergency, and shift ....
    Or, you know, shear, convergency, shift, then you wanna know deflection maps.

    """
    def __init__(self,kwargs_data,alphax,alphay):
        """
        :param data_class: the data_class
        :param alphax: the x deflection map corresponds to data in dataclass
        :param alphay: the y deflection map corresponds to data in dataclass
        """
        self.ImageData = ImageData(**kwargs_data)
        self.alphax=alphax
        self.alphay=alphay
        self.xaxes, self.yaxes =self.ImageData.pixel_coordinates
        self.cutsize = (np.shape(self.ImageData.data)[0] - 1) / 2


    def initial_kwargs_lens(self,lens_model_list,alphax_shift=0, alphay_shift=0):
        """
        This function returns list type of kwargs of lens models.
        It requires: knowing the deflection maps
         with knowing deflection maps
        :param lens_model_list: list of strings with lens model names
        :param alphax_shift: shift the source's x position
        :param alphay_shift: shift the source's y position
        :return:  a list of kwargs of lens models corresponding to the lens models existed in lens_model_list
        """
        kwargs_lens=[]
        for lens_type in lens_model_list:
            if lens_type=='INTERPOL':
                kwargs_lens.append({'grid_interp_x': self.xaxes[0], 'grid_interp_y': self.yaxes[:,0], 'f_x': self.alphax, 'f_y': self.alphay})
            elif lens_type=='SHIFT':
                alpha_x_center = self.alphax[self.cutsize+1,self.cutsize+1]
                alpha_y_center = self.alphay[self.cutsize+1,self.cutsize+1]
                kwargs_lens.append({'alpha_x': alpha_x_center - alphax_shift, 'alpha_y': alpha_y_center - alphay_shift})
            elif lens_type == 'SHEAR':
                gamma_1_center,gamma_2_center= self.gamma_center()
                ra_center =self.xaxes[self.cutsize+1,self.cutsize+1]
                dec_center =self.yaxes[self.cutsize+1,self.cutsize+1]
                kwargs_lens.append({'e1': gamma_1_center, 'e2': gamma_2_center, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'CONVERGENCE':
                kappa_center = self.kappa_center()
                ra_center = self.xaxes[self.cutsize + 1,self.cutsize+1]
                dec_center = self.yaxes[self.cutsize + 1,self.cutsize+1]
                kwargs_lens.append({'kappa_ext': kappa_center, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'FLEXION':
                g1_c, g2_c, g3_c, g4_c = self.g_flexion()
                ra_center = self.xaxes[self.cutsize + 1, self.cutsize + 1]
                dec_center = self.yaxes[self.cutsize + 1, self.cutsize + 1]
                kwargs_lens.append({'g1': g1_c,'g2':g2_c,'g3':g3_c,'g4':g4_c, 'ra_0': ra_center, 'dec_0': dec_center})
            elif lens_type == 'FLEXIONFG':
                g1_c, g2_c, g3_c, g4_c = self.g_flexion()
                ra_center = self.xaxes[self.cutsize + 1, self.cutsize + 1]
                dec_center = self.yaxes[self.cutsize + 1, self.cutsize + 1]
                F1_c = (g1_c + g3_c) * 0.5
                F2_c = (g2_c + g4_c) * 0.5
                G1_c = (g1_c - g3_c) * 0.5 - g3_c
                G2_c = (g2_c - g4_c) * 0.5 - g4_c
                kwargs_lens.append({'F1': F1_c, 'F2': F2_c, 'G1': G1_c, 'G2': G2_c, 'ra_0': ra_center, 'dec_0': dec_center})
        return kwargs_lens

    def kappa_center(self):
        kwargs_lens=[{'grid_interp_x': self.xaxes[0], 'grid_interp_y': self.yaxes[:,0], 'f_x': self.alphax, 'f_y': self.alphay}]
        kappa=NumericLens(['INTERPOL']).kappa(util.image2array(self.xaxes), util.image2array(self.yaxes), kwargs=kwargs_lens)
        kappa_c = kappa.mean()

        return  kappa_c



    def gamma_center(self):
        kwargs_lens = [{'grid_interp_x': self.xaxes[0], 'grid_interp_y': self.yaxes[:, 0], 'f_x': self.alphax,'f_y': self.alphay}]
        gamma1, gamma2 = NumericLens(['INTERPOL']).gamma(util.image2array(self.xaxes), util.image2array(self.yaxes), kwargs=kwargs_lens)
        gamma1_c, gamma2_c = gamma1.mean(), gamma2.mean()

        return  gamma1_c, gamma2_c


    def g_flexion(self):
        kwargs_lens = [{'grid_interp_x': self.xaxes[0], 'grid_interp_y': self.yaxes[:, 0], 'f_x': self.alphax, 'f_y': self.alphay}]
        g1,g2,g3,g4 =  NumericLens(['INTERPOL']).Dmatrix(util.image2array(self.xaxes), util.image2array(self.yaxes), kwargs=kwargs_lens)
        g1_c,g2_c,g3_c,g4_c = g1.mean(),g2.mean(),g3.mean(),g4.mean()
        return g1_c,g2_c,g3_c,g4_c