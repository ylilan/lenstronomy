__author__ = 'lilan yang'

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from lenstronomy.Util import kernel_util

from astropy.stats import sigma_clipped_stats
from astropy import wcs

from photutils import detect_threshold
from photutils import detect_sources,deblend_sources
from photutils import source_properties
from photutils.datasets import make_noise_image



class DataPreparation(object):
    """
    The class contains useful fuctions to do, e.g. cut image, calculate cutsize, make mask of images.....
    """
    def __init__(self, hdul,deltaPix, snr=3.0, npixels=20,exp_time=None,background_rms=None, background=None, error = None, kernel = None, interaction = True):
        self.hdul = hdul
        self.deltaPix = deltaPix
        self.snr = snr
        self.npixels = npixels
        if exp_time is None:
            exp_time = hdul[0].header['EXPTIME']
        else:
            exp_time = exp_time
        self.exp_time = exp_time
        self.background_rms = background_rms
        self.bakground = background
        self.error = error
        self.kernel = kernel
        self.interaction= interaction
        self.image = hdul[0].data


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

    def _seg_image(self, x, y, r_cut=100,image_name='segs_map.pdf',title_name1='Input Image',title_name2='Segmentation of Detected Sources'):
        snr=self.snr
        npixels=self.npixels
        bakground = self.bakground
        error = self.error
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
        masks = []
        for i in range(nobjs):
            masks.append((segments_deblend.data==i+1)*1)
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.imshow(image_data, origin='lower')
        ax1.set_title(title_name1)
        ax2.imshow(segments_deblend, origin='lower')
        for i in range(nobjs):
            ax2.text(xcenter[i]*1.1, ycenter[i], 'S'+repr(i), color='white')
        ax2.text(image_data.shape[0]*0.5,image_data.shape[0]*0.1,'S'+repr(c_index)+' '+'in center',size=15,color='white')
        ax2.set_title(title_name2)
        plt.show(fig)
        fig.savefig(image_name)
        return masks, center_mask, r_center


    def r_center(self,x,y,r_cut=100):
        """
        This fuction is used cut out the center object.
        :param image_data: 2-D array of the image that contains target object in the center.
        :param snr: float, the signal-to-noise ratio per pixel above the background for which to consider a pixel as possibly being part of a source.
        :param npixels:int,The number of connected pixels, each greater than threshold, that an object must have to be detected.
                        npixels must be a positive integer.
        :param background: float or array_like, optional
                           The background value(s) of the input data. background may either be a scalar value or a 2D image with the same shape as the input data.
                           If the input data has been background-subtracted, then set background to 0.0.
                           If None, then a scalar background value will be estimated using sigma-clipped statistics.
        :param error: The Gaussian 1-sigma standard deviation of the background noise in data.
                       error should include all sources of background error, but exclude the Poisson error of the sources.
                       If error is a 2D image, then it should represent the 1-sigma background error in each pixel of data.
                       If None, then a scalar background rms value will be estimated using sigma-clipped statistics.
        :param kernel: array-like (2D) or Kernel2D, optional
                      The 2D array of the kernel used to filter the image before thresholding.
                      Filtering the image will smooth the noise and maximize detectability of objects with a shape similar to the kernel.
        :param sigclip_sigma:float, optional
                        The number of standard deviations to use as the clipping limit when calculating the image background statistics.
        :param plt_show: plot detect objects or not.
        :param manually_option: user interaction option.
        :return: int, the cut size of the center object.
        """
        _,_,r_center= self._seg_image(x,y,r_cut=r_cut)
        cutsize_data=np.int(r_center/2.+ 10)
        if self.interaction:
            cutted_image=self.cut_image(x, y, cutsize_data)
            fig_ci=plt.figure()
            plt.imshow(cutted_image, origin='lower')
            plt.title('Good framesize? (framesize=' + repr(cutsize_data*2+1) + ')')
            plt.show(fig_ci)
            cutyn = raw_input('Hint: appropriate cutsize? (y/n): ')
            if cutyn == 'n':
                cutsize_ = np.int(input('Hint: please tell me an appropriate cutsize (framesize=2*cutsize+1)? (int fotmat): '))
                cutsize_data = cutsize_
                cutted_image_new = self.cut_image(x, y, cutsize_data)
                plt.imshow(cutted_image_new, origin='lower')
                plt.title('Cutted Data (framesize=' + repr(cutsize_data*2+1) + ')')
                plt.show()
            elif cutyn == 'y':
                cutsize_data=cutsize_data
            else:
                raise ValueError("Please input 'y' or 'n' !")
        return cutsize_data



    def pick_data(self,x,y, r_cut, add_mask=5):
       """
       Function to pick up the pieces of data.
       :param x: x coordinate.
       :param y: y coordinate.
       :param r_cut: radius size of the data.
       :param add_mask: number of pixels adding around picked pieces
       :return: kwargs_data
       """
       selem = np.ones((add_mask, add_mask))
       obj_masks, data_masks_center, _ = self._seg_image(x, y, r_cut=r_cut,image_name='resegs_map.pdf',title_name1="Cutout Image")
       image = self.cut_image(x,y,r_cut)
       if self.interaction:
            source_mask_index = input('Tell me mask index (list format)=')
            src_mask = np.zeros_like(image)
            for i in source_mask_index:
                src_mask = src_mask + obj_masks[i]
            mask = src_mask
       else:
            mask = data_masks_center
       img_mask = ndimage.binary_dilation(mask.astype(np.bool), selem)
       source_mask = image * img_mask
       _, _, std = sigma_clipped_stats(image, sigma=3.0, mask=source_mask)
       tshape = image.shape
       img_bkg = make_noise_image(tshape, type='gaussian', mean=0.,
                                   stddev=std, random_state=12)
       no_source_mask = (img_mask * -1 + 1) * img_bkg
       picked_data = source_mask + no_source_mask

       f, axes = plt.subplots(1, 2, figsize=(9, 6))
       vmax = image.max()
       vmin = image.min()
       ax1 = axes[0]
       ax1.imshow(image, origin='lower', vmin=vmin, vmax=vmax)
       ax1.set_title("Cutout Image")
       ax2 = axes[1]
       ax2.imshow(picked_data, origin='lower', vmin=vmin, vmax=vmax)
       ax2.set_title("Cleaned Image")
       plt.show()
       f.savefig('cleaned_img.pdf')
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