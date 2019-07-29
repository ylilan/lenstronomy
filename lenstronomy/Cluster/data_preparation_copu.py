__author__ = 'lilan yang'




import numpy as np
from astropy import wcs

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import sigma_clipped_stats
from photutils import detect_threshold

from photutils import detect_sources,deblend_sources
from photutils import source_properties

from photutils.datasets import make_noise_image
from scipy import ndimage


def radec2detector(ra, dec, hdul):
    """
    transform (ra,dec) in deg to director coordinate
    :param ra: ra in deg
    :param dec: dec in deg
    :param hdul: fits format, fits image that contains header info
    :return:
    """
    w = wcs.WCS(hdul[0].header)  # get world2pix information from header
    y_detector = np.int(w.wcs_world2pix([[ra, dec]], 1)[0][0])
    x_detector = (np.int(w.wcs_world2pix([[ra, dec]], 1)[0][1]))
    return  x_detector,y_detector

def cut_image(x, y, r_cut, image):
    """
    Function used to cut input image.
    :param x: x coordinate
    :param y: y coordinate
    :param r_cut: int format value, radius of cut out image
    :param image: parent image
    :return: cutted image
    """
    image_cutted = image[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
    return image_cutted


def cut_center(image, x, y, snr=2.5, npixels=10, bakground=None, error=None, kernel=None,
               plt_show=True, manually_option=True,r_cut_in=100):
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
    image_cutted = image[x - r_cut_in:x + r_cut_in + 1, y - r_cut_in:y + r_cut_in + 1]
    image_data=image_cutted
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
        framesize = np.int(xsize_c)
    else:
        framesize = np.int(ysize_c)
    r_cut = np.int(framesize / 2. + 10)
    if manually_option:
        cutted_image = cut_image(x, y, r_cut, image)
        fig_ci = plt.figure()
        plt.imshow(cutted_image, origin='lower')
        plt.title('Good framesize? (framesize=' + repr(r_cut * 2 + 1) + ')')
        plt.show(fig_ci)
        cutyn = raw_input('Hint: appropriate cutsize? (y/n): ')
        if cutyn == 'n':
            cutsize_ = np.int(
                    input('Hint: please tell me an appropriate cutsize (framesize=2*cutsize+1)? (int fotmat): '))
            r_cut = cutsize_
            cutted_image_new = cut_image(x, y, r_cut, image)
            plt.imshow(cutted_image_new, origin='lower')
            plt.title('Cutted Data (framesize=' + repr(r_cut * 2 + 1) + ')')
            plt.show()
        elif cutyn == 'y':
            r_cut = r_cut
        else:
            raise ValueError("Please input 'y' or 'n', not %s",cutyn)
    return r_cut



def mask_image(image, x, y,r_cut, snr=2.5, npixels=10, background=None, error=None,kernel=None,plt_show = True):
        """
          This fuction is used to detect masks of sources in a input img_data, then retrun the masks of all objects in the image.

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
          :param plt_show: plot detect objects or not.
          :return: masks of objects,mask of the center object, masks of the brightest object.
          """
        image_cutted = image[x - r_cut:x + r_cut + 1, y - r_cut:y + r_cut + 1]
        image_data = image_cutted
        threshold_detect_objs = detect_threshold(data=image_data, snr=snr, background=background, error=error)
        segments = detect_sources(image_data, threshold_detect_objs, npixels=npixels, filter_kernel=kernel)
        segments_deblend = deblend_sources(image_data, segments, npixels=npixels,nlevels=10)
        segments_deblend_info = source_properties(image_data, segments_deblend)
        columns = ['id', 'xcentroid', 'ycentroid', 'source_sum', 'area', 'xmin', 'xmax', 'ymin', 'ymax']
        tbl = segments_deblend_info.to_table(columns=columns)
        tbl['xcentroid'].info.format = '.2f'  # optional format
        tbl['ycentroid'].info.format = '.2f'
        tbl['source_sum'].info.format = '.2f'
        nobjs = segments_deblend_info.to_table(columns=['id'])['id'].max()
        xcenter = segments_deblend_info.to_table(columns=['xcentroid'])['xcentroid'].value
        ycenter = segments_deblend_info.to_table(columns=['ycentroid'])['ycentroid'].value
        source_sum = np.array(segments_deblend_info.to_table(columns=['source_sum'])['source_sum'])
        fmax_index = np.where(source_sum == source_sum.max())[0][0]
        image_data_size=np.int((image_data.shape[0]+1)/2.)
        dist = ((xcenter - image_data_size) ** 2 + (ycenter - image_data_size) ** 2) ** 0.5
        c_index = np.where(dist == dist.min())[0][0]
        data_masks_center=(segments_deblend.data==c_index+1)*1 #supposed to be the data mask
        data_masks_max=(segments_deblend.data==fmax_index+1)*1
        obj_masks = []
        #plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6))
        ax1.imshow(image_data, origin='lower')
        ax1.set_title('Data')
        ax2.imshow(segments_deblend, origin='lower')
        for i in range(nobjs):
            ax2.text(xcenter[i], ycenter[i], 'Mask_' + repr(i), color='red')
            obj_masks.append((segments_deblend.data==i+1)*1)
        ax2.text(image_data.shape[0]*0.5,image_data.shape[0]*0.1, 'mask_'+repr(c_index)+' in center', size=12, color='white')
        ax2.set_title('Segmentation Image')
        if plt_show == True:
            plt.show(fig)
        else:
            plt.close()
        return obj_masks, data_masks_center,data_masks_max,c_index



def masked_data(self,image,add_mask=5,snr=2., npixels=20,imgname='Data_mask_default',plt_show=True, manually_option=True):
       """

       :param image:
       :param add_mask: the pixels add around the chosen mask.
       :param snr:
       :param npixels:
       :param imgname:
       :param plt_show:
       :param manually_option:
       :return:
       """
       selem = np.ones((add_mask, add_mask))
       obj_masks, data_masks_center, _, _ = self.masks_objs(image, snr=snr, npixels=npixels,plt_show=plt_show)
       if manually_option:
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
       masked_image = source_mask + no_source_mask
       self.images_plot(image,masked_image, img_bkg,imgname,plt_show=plt_show)

       return  masked_image,img_bkg,std,img_mask



def images_plot(self, image,masked_image,img_bkg,imgname,plt_show=True):
        vmax=image.max()
        vmin=image.min()
        f, axes = plt.subplots(1, 3, figsize=(25,8), sharex=False, sharey=False)
        ax = axes[0]
        im = ax.matshow(image, origin='lower',vmin=vmin,vmax=vmax)
        ax.set_title("Original lensed image")
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="4%", pad=0.01)
        plt.colorbar(im, cax=cax)

        ax = axes[1]
        im0 = ax.matshow(masked_image, origin='lower',vmin=vmin,vmax=vmax)
        ax.set_title("Uncontaminated lensed image")
        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="4%", pad=0.01)
        plt.colorbar(im0, cax=cax)


        ax = axes[2]
        im2 = ax.matshow(img_bkg, origin='lower',vmin=vmin,vmax=vmax)
        ax.set_title("Uncontaminated bkg image")
        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="4%", pad=0.01)
        plt.colorbar(im2, cax=cax)
        if plt_show == True:
            plt.show()
        else:
            plt.close()
        f.savefig(imgname+'.png')