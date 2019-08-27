import pytest
import numpy.testing as npt
import numpy as np
import copy
from lenstronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestPositionLikelihood(object):

    def setup(self):

        # compute image positions
        lensModel = LensModel(lens_model_list=['SIE'])
        solver = LensEquationSolver(lensModel=lensModel)
        self._kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': -0.03, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = solver.image_position_from_source(sourcePos_x=0.01, sourcePos_y=-0.01, kwargs_lens=self._kwargs_lens)

        point_source_class = PointSource(point_source_type_list=['LENSED_POSITION'], lensModel=lensModel)
        self.likelihood = PositionLikelihood(point_source_class, position_uncertainty=0.005, astrometric_likelihood=True,
                 image_position_likelihood=True, ra_image_list=[x_pos], dec_image_list=[y_pos],
                 source_position_likelihood=True, check_solver=False, solver_tolerance=0.001, force_no_add_image=False,
                 restrict_image_number=False, max_num_images=None)
        self._x_pos, self._y_pos = x_pos, y_pos

    def test_image_position_likelihood(self):
        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        logL = self.likelihood.image_position_likelihood(kwargs_ps, self._kwargs_lens, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

        kwargs_ps = [{'ra_image': self._x_pos + 0.01, 'dec_image': self._y_pos}]
        logL = self.likelihood.image_position_likelihood(kwargs_ps, self._kwargs_lens, sigma=0.01)
        npt.assert_almost_equal(logL, -2, decimal=8)

    def test_astrometric_likelihood(self):
        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        kwargs_special = {'delta_x_image': [0, 0, 0, 0.], 'delta_y_image': [0, 0, 0, 0.]}
        logL = self.likelihood.astrometric_likelihood(kwargs_ps, kwargs_special, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

        kwargs_special = {'delta_x_image': [0, 0, 0, 0.01], 'delta_y_image': [0, 0, 0, 0.01]}
        logL = self.likelihood.astrometric_likelihood(kwargs_ps, kwargs_special, sigma=0.01)
        npt.assert_almost_equal(logL, -1, decimal=8)

        logL = self.likelihood.astrometric_likelihood([], kwargs_special, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

        logL = self.likelihood.astrometric_likelihood(kwargs_ps, {}, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=8)

    def test_check_additional_images(self):
        point_source_class = PointSource(point_source_type_list=['LENSED_POSITION'], additional_images_list=[True],
                                         lensModel=LensModel(lens_model_list=['SIE']))
        likelihood = PositionLikelihood(point_source_class)

        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        bool = likelihood.check_additional_images(kwargs_ps, self._kwargs_lens)
        assert bool is False

        kwargs_ps = [{'ra_image': self._x_pos[1:], 'dec_image': self._y_pos[1:]}]
        bool = likelihood.check_additional_images(kwargs_ps, self._kwargs_lens)
        assert bool is True

    def test_solver_penalty(self):
        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        logL = self.likelihood.solver_penalty(self._kwargs_lens, kwargs_ps, tolerance=0.0001, verbose=False)
        assert logL == 0

        kwargs_ps = [{'ra_image': self._x_pos + 0.01, 'dec_image': self._y_pos}]
        logL = self.likelihood.solver_penalty(self._kwargs_lens, kwargs_ps, tolerance=0.001, verbose=False)
        npt.assert_almost_equal(logL, 123518245.5117848, decimal=0)

    def test_logL(self):
        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        kwargs_special = {'delta_x_image': [0, 0, 0, 0.], 'delta_y_image': [0, 0, 0, 0.]}
        logL = self.likelihood.logL(self._kwargs_lens, kwargs_ps, kwargs_special, verbose=True)
        npt.assert_almost_equal(logL, 0 , decimal=9)

    def test_source_position_likelihood(self):
        kwargs_ps = [{'ra_image': self._x_pos, 'dec_image': self._y_pos}]
        logL = self.likelihood.source_position_likelihood(self._kwargs_lens, kwargs_ps, sigma=0.01)
        npt.assert_almost_equal(logL, 0, decimal=9)
        x_pos = copy.deepcopy(self._x_pos)
        x_pos[0] += 0.01
        kwargs_ps = [{'ra_image': x_pos, 'dec_image': self._y_pos}]
        logL = self.likelihood.source_position_likelihood(self._kwargs_lens, kwargs_ps, sigma=0.01)
        npt.assert_almost_equal(logL, -0.33011713058631054, decimal=4)


if __name__ == '__main__':
    pytest.main()
