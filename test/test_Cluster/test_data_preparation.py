__author__ = 'lilan yang'

import matplotlib.pyplot as plt
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Cluster.data_preparation import cut_image
import lenstronomy.Util.util as util

class TestDataPreparation(object):
    """
    Test data preparation
    """
    def setup(self):
        kwargs_data = sim_util.data_configure_simple(numPix=50, deltaPix=1, sigma_bkg=1,exposure_time=10)
        x_grid, y_grid = util.make_grid(numPix=50, deltapix=1)
        lightmodel = LightModel(['GAUSSIAN'])
        kwargs_light = [{'amp': 1, 'sigma_x': 1, 'sigma_y': 1, 'center_x': 0, 'center_y': 0}]
        self.image = util.array2image(lightmodel.surface_brightness(x=x_grid, y=y_grid, kwargs_list=kwargs_light))


    def test_cut_image(self):
        image_cutted=cut_image(x=0,y=0,r_cut=10,image=self.image)
        plt.imshow(image_cutted)
        plt.close()


