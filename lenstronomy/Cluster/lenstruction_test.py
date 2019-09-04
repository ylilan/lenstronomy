#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:09:37 2019

@author: lilan
"""

#The following modules are assumed to have beem imported
import numpy as np
import matplotlib.pyplot as plt

#required input from user 

#data = 'hlsp_clash_hst_acs-30mas_macs0717_f555w_v1_drz.fits'
#deflection_x = 'interpolated_alphax.fits'
#deflection_y = 'interpolated_alphay.fits'
#ra=[109.381093,109.376338,109.3909811]
#dec=[37.750440,37.744602,37.76324591]
#zl=0.545
#zs=1.850
#deltaPix = 0.03
#exposure_time = None # if exposure_time is not specified, the code will read from headfile
#background_rms = None # if background_rms is not specified, the code will estimate that
#y_psf=5769
#x_psf=5396
#interaction=True
#fsize = 55


from lenstronomy.Cluster.data_preparation import DataPreparation
from lenstronomy.Cluster.source_preparation import SourcePreparation
from lenstronomy.Cluster.lens_preparation import LensPreparation
from lenstronomy.Cluster.clsr_workflow import ClsrWorkflow


#from lenstronomy.Util.analysis_util import half_light_radius
#import lenstronomy.Util.util as util

#clsr_wf = ClsrWorkflow(

# kwargs_data_joint,

# kwargs_params = lens_params, source_params

# kwargs_model,
#
# kwargs_constraints, kwargs_likelihood,)

def lenstruction_test(data,ra,dec,x_psf,y_psf,deltaPix,exposure_time,background_rms,xdeflection,ydeflection,z_lens,z_source,lenseq,
                      nmax_list,n_particles=100,n_iterations=100,sigma_scale =1,interaction=True):
    """

    :param data:
    :param ra:
    :param dec:
    :param x_psf:
    :param y_psf:
    :param deltaPix:
    :param exposure_time:
    :param background_rms:
    :param xdeflection:
    :param ydeflection:
    :param z_lens:
    :param z_source:
    :param lenseq:
    :param nmax_list:
    :param n_particles:
    :param n_iterations:
    :param sigma_scale:
    :param interaction:
    :return:
    """
    datapreparation = DataPreparation(data=data, deltaPix=deltaPix, snr=3.0, npixels=20,exp_time=exposure_time,background_rms=background_rms,interaction=interaction)
    ximg_list, yimg_list, kwargs_data_joint = datapreparation.kwargs_data_psf(ra,dec,60,x_psf,y_psf,10)

    lenspreparation=LensPreparation(alphax=xdeflection,alphay=ydeflection,zl=z_lens,zs=z_source)
    lens_params=lenspreparation.kwargs_lens_configuration(ximg_list=ximg_list, yimg_list=yimg_list, kwargs_data_list=kwargs_data_list)
    lens_model_list = lenspreparation.lens_model_list
    lens_constrain = lenspreparation.constrain()


    sourcepreparation = SourcePreparation()
    source_params = sourcepreparation.params(R_sersic_in, betax, betay, deltaPix)
    source_model_list = sourcepreparation.source_model_list
    source_constrain = sourcepreparation.constrain()

    kwargs_constraints =  lens_constrain + source_constrain

    kwargs_model = lens_model_list + source_model_list
    #    {'lens_model_list': lens_model_list, 'source_light_model_list': source_model_list,
    #                'index_lens_model_list': index_lens_model_list}
    clsrwf=ClsrWorkflow(kwargs_data_joint, kwargs_model, kwargs_constraints,lens_params,source_params)
    bic_model_list_output, chain_list_lowest, kwargs_result_lowest = clsrwf.lowest_bic(nmax_list, n_particles,n_iterations,sigma_scale,magnif)
    return 0
    #return source_results,flux_total,rh

   



 