import numpy as np
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util


class Initkwargs_data(object):
    """
    This class is used to do initialize kwargs for data
    """
    def initial_kwargs_data(self,image,x_img,y_img,deltaPix,exp_time,background_rms):
        """

        :param image: lensed image
        :param x_img: center x position (detector coordinate)
        :param y_img: center y position (detector coordinate)
        :param deltaPix: arcseconds/pixel
        :param exp_time: exposure time of the lensed image
        :param background_rms: background noise level of the lensed image
        :return:
        !Attention: (ra,dec) is (y_img,x_img) in fits file.
        """
        image_data = image
        cut_size = (np.shape(image)[0]-1)/2
        ra_at_xy_0 = (y_img - cut_size) * deltaPix #(ra,dec) is (y_img,x_img)
        dec_at_xy_0 = (x_img - cut_size) * deltaPix
        kwargs_data = {}
        kwargs_data['background_rms'] = background_rms
        kwargs_data['exposure_time'] = exp_time
        kwargs_data['ra_at_xy_0'] = ra_at_xy_0
        kwargs_data['dec_at_xy_0'] = dec_at_xy_0
        kwargs_data['transform_pix2angle'] = np.array([[1, 0], [0, 1]]) * deltaPix
        kwargs_data['image_data'] = image_data

        return  kwargs_data


    def initial_kwargs_psf(self, image_psf,deltaPix, kernel_size=None):
        if kernel_size is None:
            kernel_size=np.shape(image_psf)[0]
        image_psf_cut = kernel_util.cut_psf(image_psf, psf_size=kernel_size)
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': image_psf_cut, 'pixel_size': deltaPix}

        return kwargs_psf