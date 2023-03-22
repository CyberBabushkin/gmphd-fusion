from unittest import TestCase

import numpy as np

from gmphd_fusion.data import StateVector, CovarianceMatrix
from gmphd_fusion.gm import Gaussian, GaussianMixture


class TestGaussian(TestCase):
    def setUp(self) -> None:
        Gaussian._next_label = 0
        self.m = StateVector([0, 0])
        self.c = CovarianceMatrix([[1, 0], [0, 1]])
        self.g0 = Gaussian(self.m, self.c)
        self.g10 = Gaussian(self.m, self.c, label=10)

    def test_init(self):
        self.assertTrue(self.g0.mean.shape == (2, 1))
        self.assertTrue(np.all(self.g0.mean == [[0], [0]]))
        self.assertTrue(self.g0.cov.shape == (2, 2))
        self.assertTrue(np.all(self.g0.cov == [[1, 0], [0, 1]]))
        self.assertEqual(self.g0.label, 0)
        self.assertEqual(self.g10.label, 10)

    def test_new_label(self):
        self.g0.new_label()
        self.assertEqual(self.g0.label, 1)
        self.g0.new_label()
        self.assertEqual(self.g0.label, 2)

    def test_copy(self):
        g0c = self.g0.copy(with_label=False)
        self.assertEqual(g0c.label, 1)
        g0cl = self.g0.copy(with_label=True)
        self.assertEqual(g0cl.label, 0)
        self.g0.mean[0, 0] = 5
        g0c.mean[0, 0] = 6
        g0cl.mean[0, 0] = 7

        self.assertTrue(np.all(self.g0.mean == [[5], [0]]))
        self.assertTrue(np.all(g0c.mean == [[6], [0]]))
        self.assertTrue(np.all(g0cl.mean == [[7], [0]]))


class TestGaussianMixture(TestCase):
    def setUp(self) -> None:
        self.m = StateVector([0, 0])
        self.c = CovarianceMatrix([[1, 0], [0, 1]])
        self.gaussians = list(Gaussian(self.m * i, self.c * i, label=i * 10) for i in range(1, 6))
        self.weights = list(i * 0.1 for i in range(1, 6))
        self.gm = GaussianMixture(self.gaussians, self.weights)

    def test_init(self):
        self.assertEqual(self.gm.gaussians, list(reversed(self.gaussians)))
        self.assertEqual(self.gm.weights, list(reversed(self.weights)))

    def test_add_to_mixture(self):
        g = Gaussian(self.m, self.c, label=100)
        w = 0.35
        true_g = list(reversed(self.gaussians))
        true_g = true_g[:2] + [g] + true_g[2:]
        true_w = list(reversed(self.weights))
        true_w = true_w[:2] + [w] + true_w[2:]
        self.gm.add_to_mixture(g, w)
        self.assertEqual(self.gm.gaussians, true_g)
        self.assertEqual(self.gm.weights, true_w)

    def test_remove_from_mixture(self):
        self.test_add_to_mixture()
        self.gm.remove_from_mixture(2)
        true_g = list(reversed(self.gaussians))
        true_w = list(reversed(self.weights))
        self.assertEqual(self.gm.gaussians, true_g)
        self.assertEqual(self.gm.weights, true_w)

    def test_truncate(self):
        self.test_add_to_mixture()
        self.gm.truncate(1)
        self.assertEqual(self.gm.gaussians, self.gaussians[-1:])
        self.assertEqual(self.gm.weights, self.weights[-1:])

    def test_uniquify_labels(self):
        self.gaussians[1].label = Gaussian._next_label
        self.gaussians[0].label = Gaussian._next_label + 1
        self.gm.gaussians[-1].label = self.gm.gaussians[-3].label
        self.gm.gaussians[-2].label = self.gm.gaussians[-3].label
        self.gm.uniquify_labels()
        true_g = list(reversed(self.gaussians))
        true_w = list(reversed(self.weights))
        self.assertEqual(self.gm.gaussians, true_g)
        self.assertEqual(self.gm.weights, true_w)

    def test_copy(self):
        true_g = list(reversed(self.gaussians))
        true_w = list(reversed(self.weights))
        gmc = self.gm.copy(with_labels=False)
        gmc.truncate(1)
        self.assertEqual(self.gm.gaussians, true_g)
        self.assertEqual(self.gm.weights, true_w)
        gmc = self.gm.copy(with_labels=True)
        for g in gmc.gaussians:
            g.label = 0
        gmc_labels_exp = [0 for _ in range(len(gmc.gaussians))]
        gmc_labels_true = [g.label for g in gmc.gaussians]
        gm_labels_exp = [g.label for g in true_g]
        gm_labels_true = [g.label for g in self.gm.gaussians]

        self.assertEqual(gmc_labels_exp, gmc_labels_true)
        self.assertEqual(gm_labels_exp, gm_labels_true)
        self.assertNotEqual(gmc_labels_true, gm_labels_true)
