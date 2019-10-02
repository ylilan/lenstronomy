__author__ = 'lilan yang'

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.io import fits

from lenstronomy.Util import kernel_util

from astropy.stats import sigma_clipped_stats
from astropy import wcs
from photutils import detect_threshold, detect_sources,deblend_sources, source_properties
from photutils.datasets import make_noise_image



class DataPreparation(object):
    """
    The class contains useful fuctions to do, e.g. cut image, calculate cutsize, make mask of images.....
    """
    def __init__(self, data, deltaPix, snr=3.0, npixels=20,exp_time=None,background_rms=None,
                 background=None, kernel = None, interaction = True):
        """

        :param hdul: fits format, image fits file.
        :param deltaPix: float, pixel size
        :param snr: float, signal-to-noise value
        :param npixels: int, number of connected pixels that can be detected as a source
        :param exp_time: float, exposure time of the fits files
        :param background_rms: float,float or array_like, the gaussian 1-sigma background noise in data.
        :param background: float or 2D array,background value of the input image.
        :param kernel: The 2D array, filter the image before thresholding.
        :param interaction:
        """
        self.hdul = fits.open(data)
        self.deltaPix = deltaPix
        self.snr = snr
        self.npixels = npixels
        if exp_time is None:
            exp_time = self.hdul[0].header['EXPTIME']
        else:
            exp_time = exp_time
        self.exp_time = exp_time
        self.background_rms = background_rms
        self.bakground = background
        self.kernel = kernel
        self.interaction= interaction
        self.image = self.hdul[0].data


    def radec2detector(self,ra,dec):
        """
        transform (ra,dec) in deg to director coordinate
        :param ra: ra in deg
        :param dec: dec in deg
        :return:
        """
        w = wcs.WCS(self.hdul[0].header)  # get world2pix information from header
        y_detector= np.int(w.wcs_world2pix([[ra, dec]], 1)[0][0])
        x_detector = (np.int(w.wcs_world2pix([[ra, dec]], 1)[0][1]))

        return x_detector, y_detector

    def cut_image(self, x, y, r_cut, image=None):
        """
         Function used to cut input image.
         :param x: x coordinate
         :param y: y coordinate
         :param r_cut: int format value, radius of cut out image
         :param image: parent image
         :return: cutted image
         """
        if image is None:
            image = self.image
        else:
            image = image
        image_cutted = image[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
        return image_cutted

    def _seg_image(self, x, y, r_cut=100):
        """
        detect and deblend sources into segmentation maps
        :param x:
        :param y:
        :param r_cut:
        :param image_name:
        :param title_name1:
        :param title_name2:
        :return:
        """
        snr=self.snr
        npixels=self.npixels
        bakground = self.bakground
        error = self.background_rms
        kernel = self.kernel
        image_cutted = self.image[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
        image_data = image_cutted
        threshold_detect_objs=detect_threshold(data=image_data, snr=snr,background=bakground,error=error)
        segments=detect_sources(image_data, threshold_detect_objs, npixels=npixels, filter_kernel=kernel)
        segments_deblend = deblend_sources(image_data, segments, npixels=npixels,nlevels=10)
        segments_deblend_info = source_properties(image_data, segments_deblend)
        nobjs = segments_deblend_info.to_table(columns=['id'])['id'].max()
        xcenter = segments_deblend_info.to_table(columns=['xcentroid'])['xcentroid'].value
        ycenter = segments_deblend_info.to_table(columns=['ycentroid'])['ycentroid'].value
        image_data_size = np.int((image_data.shape[0] + 1) / 2.)
        dist = ((xcenter - image_data_size) ** 2 + (ycenter - image_data_size) ** 2) ** 0.5
        c_index = np.where(dist == dist.min())[0][0]
        center_mask=(segments_deblend.data==c_index+1)*1 #supposed to be the data mask
        obj_masks = []
        for i in range(nobjs):
            mask = ((segments_deblend.data==i+1)*1)
            obj_masks.append(mask)
        xmin = segments_deblend_info.to_table(columns=['xmin'])['xmin'].value
        xmax = segments_deblend_info.to_table(columns=['xmax'])['xmax'].value
        ymin = segments_deblend_info.to_table(columns=['ymin'])['ymin'].value
        ymax = segments_deblend_info.to_table(columns=['ymax'])['ymax'].value
        xmin_c, xmax_c = xmin[c_index], xmax[c_index]
        ymin_c, ymax_c = ymin[c_index], ymax[c_index]
        xsize_c = xmax_c - xmin_c
        ysize_c = ymax_c - ymin_c
        if xsize_c > ysize_c:
            r_center = np.int(xsize_c)
        else:
            r_center = np.int(ysize_c)
        center_mask_info= [center_mask, r_center, xcenter, ycenter, c_index]
        return obj_masks, center_mask_info, segments_deblend


    def cutsize(self,x,y,r_cut=100):
     """

     :param x: x coordinate
     :param y: y coordinate
     :param r_cut: int format value, radius of cut out image
     :param image_name: string, name of the image
     :return: cutout size
     """
     cutsize_data = r_cut
     if self.interaction:
            m_image = self.cut_image(x, y, r_cut)
            fig_ci=plt.figure()
            plt.imshow(m_image, origin='lower',cmap="gist_heat")
            plt.title('Good framesize? ('+repr(cutsize_data*2+1)+'x'+repr(cutsize_data*2+1)+' pixels^2' + ')')
            plt.show(fig_ci)
            cutyn = raw_input('Hint: appropriate framesize? (y/n): ')
            if cutyn == 'n':
                cutsize_ = np.int(raw_input('Hint: please tell me an appropriate cutsize (framesize=2*cutsize+1)? (int format): '))
                cutsize_data = cutsize_
            elif cutyn == 'y':
                cutsize_data=cutsize_data
            else:
                raise ValueError("Please input 'y' or 'n' !")
     return cutsize_data



    def data_assemble(self, x,y, r_cut, add_mask=5):
       """
       Function to pick up the pieces of data.
       :param x: x coordinate.
       :param y: y coordinate.
       :param r_cut: radius size of the data.
       :param add_mask: number of pixels adding around picked pieces
       :return: kwargs_data
       """

       obj_masks,center_mask_info, segments_deblend_list = self._seg_image(x, y, r_cut=r_cut)
       data_masks_center, _, xcenter, ycenter, c_index = center_mask_info
       image = self.cut_image(x,y,r_cut)
       self.raw_image = image
       if self.interaction:
            self.plot_segmentation(image, segments_deblend_list, xcenter, ycenter, c_index)
            source_mask_index = input('Input segmentation index, e.g.,[0,1]. (list format)=')
            src_mask = np.zeros_like(image)
            for i in source_mask_index:
                src_mask = src_mask + obj_masks[i]
            mask = src_mask
       else:
            mask = data_masks_center
       selem = np.ones((add_mask, add_mask))
       img_mask = ndimage.binary_dilation(mask.astype(np.bool), selem)
       self.data_mask = mask
       source_mask = image * img_mask
       _, _, std = sigma_clipped_stats(image, sigma=3.0, mask=source_mask)
       tshape = image.shape
       img_bkg = make_noise_image(tshape, type='gaussian', mean=0.,
                                   stddev=std, random_state=12)
       no_source_mask = (img_mask * -1 + 1) * img_bkg
       picked_data = source_mask + no_source_mask
       self.data = picked_data
       ra_at_xy_0 = (y - r_cut) * self.deltaPix  # (ra,dec) is (y_img,x_img)
       dec_at_xy_0 = (x - r_cut) * self.deltaPix
       kwargs_data = {}
       if self.background_rms is None:
            kwargs_data['background_rms'] = std
       else:
            kwargs_data['background_rms'] = self.background_rms
       kwargs_data['exposure_time'] = self.exp_time
       kwargs_data['transform_pix2angle'] = np.array([[1, 0], [0, 1]]) * self.deltaPix
       kwargs_data['ra_at_xy_0'] = ra_at_xy_0
       kwargs_data['dec_at_xy_0'] = dec_at_xy_0
       kwargs_data['image_data'] = picked_data
       return kwargs_data


    def pick_psf(self, x, y, r_cut, pixel_size=None, kernel_size=None):
        """
        select psf
        :param x:  x coordinate.
        :param y:  y coordinate.
        :param r_cut: radius size of the psf.
        :param deltaPix: pixel size of the psf.
        :param kernel_size: kernel size of the psf.
        :return: kwargs_psf
        """
        image_psf = self.cut_image(x, y, r_cut)
        if kernel_size is None:
            kernel_size = np.shape(image_psf)[0]
        image_psf_cut = kernel_util.cut_psf(image_psf, psf_size=kernel_size)
        if pixel_size is None:
            pixel_size=self.deltaPix
        else:
            pixel_size=pixel_size
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': image_psf_cut, 'pixel_size': pixel_size}
        return kwargs_psf



    def plot_data_assemble(self,add_mask=5):
        """

        :param add_mask:
        :return:
        """

        mask = self.data_mask
        image = self.raw_image
        picked_data = self.data
        selem = np.ones((add_mask, add_mask))
        img_mask = ndimage.binary_dilation(mask.astype(np.bool), selem)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10))
        ax1.imshow(image, origin='lower', cmap="gist_heat")
        ax1.set_title('Input Image')
        ax2.imshow(img_mask+mask, origin='lower',cmap="gist_heat")
        ax2.set_title('Selected pixels')
        ax3.imshow(picked_data, origin='lower',cmap="gist_heat")
        ax3.set_title('Data')
        plt.show()
        return 0



    def plot_segmentation(self,image_data,segments_deblend,xcenter,ycenter,c_index):
        """
        show segmentation map of image_data
        :param image_data:
        :param segments_deblend:
        :param nobjs:
        :param xcenter:
        :param ycenter:
        :param c_index:
        :return:
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.imshow(image_data, origin='lower',cmap="gist_heat")
        ax1.set_title('Input Image')
        ax2.imshow(segments_deblend, origin='lower')
        for i in range(len(xcenter)):
            ax2.text(xcenter[i]*1.1, ycenter[i], 'Seg'+repr(i), color='w')
        ax2.text(image_data.shape[0]*0.5,image_data.shape[0]*0.1,'Seg '+repr(c_index)+' '+'in center',size=12,color='white')
        ax2.set_title('Segmentations (S/N >'+repr(self.snr)+')')
        plt.show()
        return 0


    def kwargs_data_psf(self,ra,dec,r_cut,x_psf,y_psf,add_mask):
        ximg_list = []
        yimg_list = []
        kwargs_data_list = []
        kwargs_psf_list = []
        for i in range(len(ra)):
            xy = self.radec2detector(ra[i], dec[i])
            cutsize = self.cutsize(xy[0], xy[1], r_cut=r_cut)
            kwargs_data = self.data_assemble(x=xy[0], y=xy[1], r_cut=cutsize,add_mask=add_mask)
            kwargs_psf = self.pick_psf(x=x_psf, y=y_psf, r_cut=cutsize)
            kwargs_psf_list.append(kwargs_psf)
            kwargs_data_list.append(kwargs_data)
            ximg_list.append(xy[0])
            yimg_list.append(xy[1])
            self.plot_data_assemble(add_mask=add_mask)
        return ximg_list,yimg_list,kwargs_data_list,kwargs_psf_list
