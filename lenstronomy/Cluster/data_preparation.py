__author__ = 'lilan yang'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

from astropy import wcs
from astropy.stats import sigma_clipped_stats

from photutils import detect_threshold,detect_sources,deblend_sources,source_properties
from photutils.datasets import make_noise_image



class DataPreparation(object):
    """
    Class used to process data in cluster.
    """

    def __init__(self, hdul):
        self.hdul = hdul
        self.image = hdul[0].data


    def test(self):

    def radec2detector(self, ra, dec):
        """
        :param ra: ra coordinate in deg
        :param dec: dec coordinate in deg
        :return: The corresponding detector corrdinate
        """
        w = wcs.WCS(self.hdul[0].header)  # get World Coordinate System (WCS) transformations information from header
        y_detector = np.int(w.wcs_world2pix([[ra, dec]], 1)[0][0])
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

    def cut_center(self,image_data, ximg, yimg, snr=2.5, npixels=10, bakground=None, error=None, kernel=None,
                             plt_show=True, manually_option=True ):
        """
         Function used to figure out the cut size of detected center object.

        :param image_data: 2-D array of the image that contains target object in the center.
        :param snr: float, the signal-to-noise ratio per pixel above the background for
                    which to consider a pixel as possibly being part of a source.
        :param npixels:int,The number of connected pixels, each greater than threshold,
                       that an object must have to be detected. npixels must be a positive integer.
        :param background: float or array_like, optional
                           The background value(s) of the input data. background may either be a scalar value or a
                           2D image with the same shape as the input data.
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
        threshold_detect_objs = detect_threshold(data=image_data, snr=snr, background=bakground, error=error)
        segments = detect_sources(image_data, threshold_detect_objs, npixels=npixels, filter_kernel=kernel)
        segments_deblend = deblend_sources(image_data, segments, npixels=npixels, nlevels=10)
        segments_deblend_info = source_properties(image_data, segments_deblend)
        nobjs = segments_deblend_info.to_table(columns=['id'])['id'].max()
        xcenter = segments_deblend_info.to_table(columns=['xcentroid'])['xcentroid'].value
        ycenter = segments_deblend_info.to_table(columns=['ycentroid'])['ycentroid'].value
        xmin = segments_deblend_info.to_table(columns=['xmin'])['xmin'].value
        xmax = segments_deblend_info.to_table(columns=['xmax'])['xmax'].value
        ymin = segments_deblend_info.to_table(columns=['ymin'])['ymin'].value
        ymax = segments_deblend_info.to_table(columns=['ymax'])['ymax'].value
        image_data_size = np.int((image_data.shape[0] + 1) / 2.)
        dist = ((xcenter - image_data_size) ** 2 + (ycenter - image_data_size) ** 2) ** 0.5
        c_index = np.where(dist == dist.min())[0][0]
        xmin_c, xmax_c = xmin[c_index], xmax[c_index]
        ymin_c, ymax_c = ymin[c_index], ymax[c_index]
        xsize_c = xmax_c - xmin_c
        ysize_c = ymax_c - ymin_c
        figc, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.imshow(image_data, origin='lower')
        ax1.set_title('Initial Data')
        ax2.imshow(segments_deblend, origin='lower')
        for i in range(nobjs):
            ax2.text(xcenter[i], ycenter[i], 'object_' + repr(i + 1), color='white')
        ax2.text(image_data.shape[0] * 0.5, image_data.shape[0] * 0.1, 'object_' + repr(c_index + 1) + 'in Center',
                 size=12, color='white')
        ax2.set_title('Segmentation Image')
        if plt_show == True:
            plt.show(figc)
        else:
            plt.close()
        if xsize_c > ysize_c:
            framesize_c = np.int(xsize_c)
        else:
            framesize_c = np.int(ysize_c)
        r_cut = np.int(framesize_c / 2. + 10)
        if manually_option:
            cutted_image = self.cut_image(ximg, yimg, r_cut)
            fig_ci = plt.figure()
            plt.imshow(cutted_image, origin='lower')
            plt.title('Good framesize? (framesize=' + repr(r_cut * 2 + 1) + ')')
            plt.show(fig_ci)
            cutyn = raw_input('Hint: appropriate cutsize? (y/n): ')
            if cutyn == 'n':
                cutsize_ = np.int(
                    input('Hint: please tell me an appropriate cutsize (framesize=2*cutsize+1)? (int fotmat): '))
                r_cut = cutsize_
                cutted_image_new = self.cut_image(ximg, yimg, r_cut)
                plt.imshow(cutted_image_new, origin='lower')
                plt.title('Cutted Data (framesize=' + repr(r_cut * 2 + 1) + ')')
                plt.show()
            elif cutyn == 'y':
                r_cut = r_cut
            else:
                raise ValueError("Please input 'y' or 'n' !")
        return r_cut

