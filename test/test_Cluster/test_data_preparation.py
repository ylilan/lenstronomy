__author__ = 'lilan yang'


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import unittest


from lenstronomy.Cluster.data_preparation import DataPreparation









class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=1, sigma_bkg=1)
            #kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {'source_light_model_list': ['GAUSSIAN']}
            lensPlot = LensModelPlot(kwargs_data, kwargs_psf={'psf_type': 'NONE'}, kwargs_numerics={},
                                     kwargs_model=kwargs_model, kwargs_lens=[],
                                     kwargs_source=[{'amp': 1, 'sigma_x': 1, 'sigma_y': 1, 'center_x': 0, 'center_y': 0}], kwargs_lens_light=[],
                                     kwargs_ps = [],
                                     arrow_size=0.02, cmap_string="gist_heat")
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax = lensPlot.source_plot(ax, numPix=10, deltaPix_source=0.1, v_min=None, v_max=None, with_caustics=False,
                                      caustic_color='yellow',
                                      fsize=15, plot_scale='bad')
            plt.close()
