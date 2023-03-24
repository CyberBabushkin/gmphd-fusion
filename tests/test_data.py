from unittest import TestCase

import numpy as np

from gmphd_fusion.data import Track, StateVector, StateVectors


class TestTrack(TestCase):
    def setUp(self) -> None:
        self.time = 10
        self.state_dim = 2
        self.dummy_estimate = StateVector([1, 1])
        self.track = Track(label=1)

    def test_add_estimate(self):
        self.track.add_estimate(self.dummy_estimate, self.time)
        with self.assertRaises(ValueError):
            self.track.add_estimate(self.dummy_estimate, self.time)

        self.time += 1
        self.track.add_estimate(self.dummy_estimate, self.time)

        self.time += 3
        self.track.add_estimate(self.dummy_estimate, self.time)
        self.assertTrue(
            self.track.estimates, [self.dummy_estimate, self.dummy_estimate, None, None, self.dummy_estimate]
        )

    def test_finish(self):
        self.assertEqual(self.track.is_finished(), False)
        with self.assertRaises(ValueError):
            self.track.finish(self.time)
        self.track.add_estimate(self.dummy_estimate, self.time)
        self.time += 3
        self.track.finish(self.time)
        self.assertEqual(self.track.is_finished(), True)
        self.assertEqual(self.track.estimates, [self.dummy_estimate, None, None])
        with self.assertRaises(ValueError):
            self.track.finish(self.time)


class TestStateVectors(TestCase):
    def setUp(self) -> None:
        self.nvec = 100
        self.svs = StateVectors([StateVector([i, i, i]) for i in range(self.nvec)])

    def test_init(self):
        self.assertEqual(self.svs.shape, (3, 100))

    def test_iteration(self):
        i = 0
        for vec_out in self.svs:
            vec_exp = StateVector([i, i, i])
            i += 1

            self.assertTrue(isinstance(vec_out, StateVector))
            self.assertEqual(vec_out.shape, vec_exp.shape)
            self.assertTrue(np.allclose(vec_out, vec_exp))

    def test_getitem(self):
        for i in range(self.nvec):
            vec_exp = StateVector([i, i, i])
            vec_out = self.svs[i]

            self.assertTrue(isinstance(vec_out, StateVector))
            self.assertEqual(vec_out.shape, vec_exp.shape)
            self.assertTrue(np.allclose(vec_out, vec_exp))
