__author__ = 'sibirrer'

"""
Tests for `prob_density` module.
"""

from lenstronomy.Util.prob_density import SkewGaussian, Approx, KDE1D
import lenstronomy.Util.prob_density as prob_density

import pytest
import numpy as np
import numpy.testing as npt
import unittest


class TestSkewGaussian(object):

    def setup(self):
        self.skewGassian = SkewGaussian()
        np.random.seed(seed=42)

    def test_pdf(self):
        x = 1
        y = self.skewGassian.pdf(x, e=0., w=1., a=0.)
        assert y == 0.24197072451914337
        x = np.array([0, 1])
        y = self.skewGassian.pdf(x, e=0., w=1., a=0.)
        assert y[0] == 0.3989422804014327
        assert y[1] == 0.24197072451914337

    def test_pdf_skew(self):
        x = 1
        y = self.skewGassian.pdf_skew(x, mu=1, sigma=1, skw=0.5)
        assert y == 0.39834240320473779

        y = self.skewGassian.pdf_skew(x, mu=1, sigma=1, skw=-0.5)
        assert y == 0.39834240320473779


class TestProbDensity(object):

    def setup(self):
        np.random.seed(seed=42)

    def gauss(self, x, simga):
        return np.exp(-(x/(simga))**2/2)

    def test_approx_cdf_1d(self):
        x_array = np.linspace(-5, 5, 500)
        sigma = 1.
        pdf_array = self.gauss(x_array, simga=sigma)
        pdf_array /= np.sum(pdf_array)

        cdf_array, cdf_func, cdf_inv_func = prob_density.approx_cdf_1d(x_array, pdf_array)
        npt.assert_almost_equal(cdf_array[-1], 1, decimal=8)
        npt.assert_almost_equal(cdf_func(0), 0.5, decimal=2)
        npt.assert_almost_equal(cdf_inv_func(0.5), 0., decimal=2)

    def test_compute_lower_upper_errors(self):
        x_array = np.linspace(-5, 5, 1000)
        sigma = 1.
        pdf_array = self.gauss(x_array, simga=sigma)
        approx = Approx(x_array, pdf_array)
        np.random.seed(42)
        sample = approx.draw(n=20000)
        mean_0, _ = prob_density.compute_lower_upper_errors(
            sample, num_sigma=0)
        mean, [[lower_sigma1, upper_sigma1]] = prob_density.compute_lower_upper_errors(
            sample, num_sigma=1)
        npt.assert_almost_equal(mean, mean_0, decimal=8)
        mean, [[lower_sigma1, upper_sigma1], [lower_sigma2, upper_sigma2]] = prob_density.compute_lower_upper_errors(sample, num_sigma=2)
        mean, [[lower_sigma1, upper_sigma1], [lower_sigma2, upper_sigma2], [lower_sigma3, upper_sigma3]] = prob_density.compute_lower_upper_errors(sample, num_sigma=3)
        npt.assert_almost_equal(mean, 0, decimal=2)
        print(lower_sigma1, lower_sigma2, lower_sigma3)
        print(upper_sigma1, upper_sigma2, upper_sigma3)
        npt.assert_almost_equal(lower_sigma1, sigma, decimal=2)
        npt.assert_almost_equal(lower_sigma2, 2*sigma, decimal=1)
        npt.assert_almost_equal(lower_sigma3, 3 * sigma, decimal=1)

        draw = approx.draw_one
        assert len(draw) == 1


class TestKDE1D(object):

    def setup(self):
        np.random.seed(seed=42)

    def gauss(self, x, mean, simga):
        return np.exp(-((x-mean)/(simga))**2/2) / np.sqrt(2*np.pi) / simga

    def test_likelihood(self):
        x_array = np.linspace(0.5, 1.5, 3000)
        sigma = .1
        mean = 1.
        pdf_array = self.gauss(x_array, mean=mean, simga=sigma)
        approx = Approx(x_array, pdf_array)
        sample = approx.draw(n=50000)
        kde = KDE1D(values=sample)

        x = -10
        likelihood = kde.likelihood(x)
        likelihood_true = self.gauss(x, mean=mean, simga=sigma)
        npt.assert_almost_equal(likelihood, likelihood_true, decimal=4)

        x = np.linspace(0.5, 1.5, 15)
        likelihood = kde.likelihood(x)
        likelihood_true = self.gauss(x, mean=mean, simga=sigma)
        #import matplotlib.pyplot as plt
        #plt.plot(x, likelihood)
        #plt.plot(x, likelihood_true)
        #plt.show()
        npt.assert_almost_equal(likelihood, likelihood_true, decimal=1)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            skewGassian = SkewGaussian()
            skewGassian.pdf_skew(x=1, mu=1, sigma=1, skw=-1)
        with self.assertRaises(ValueError):
            prob_density.compute_lower_upper_errors(sample=None, num_sigma=4)


if __name__ == '__main__':
    pytest.main()
