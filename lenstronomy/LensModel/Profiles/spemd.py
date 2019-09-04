__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class SPEMD(LensProfileBase):
    """
    class for smooth power law ellipse mass density profile
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.s2 = 0.00000001
        self.spp = SPP()
        self.spemd_smooth = SPEMD_SMOOTH()
        super(SPEMD, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        return self.spemd_smooth.function(x, y, theta_E, gamma, e1, e2, self.s2, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        return self.spemd_smooth.derivatives(x, y, theta_E, gamma, e1, e2, self.s2, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        return self.spemd_smooth.hessian(x, y, theta_E, gamma, e1, e2, self.s2, center_x, center_y)

    def mass_3d_lens(self, r, theta_E, gamma, e1, e2):
        """
        computes the spherical power-law mass enclosed (with SPP routiune)
        :param r:
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :return:
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)
