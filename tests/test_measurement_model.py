from unittest import TestCase

import numpy as np

from gmphd_fusion.measurement_model import LinearCoordinateMeasurementModel


class TestLinearCoordinateMeasurementModel(TestCase):
    def setUp(self) -> None:
        self.noise = 10
        self.model24 = LinearCoordinateMeasurementModel(2, 4, self.noise)
        self.model22 = LinearCoordinateMeasurementModel(2, 2, self.noise)

    def test_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError):
            LinearCoordinateMeasurementModel(4, 2, self.noise)

    def test_measurement_matrix(self):
        self.assertTrue(np.all(np.isclose(self.model24.measurement_matrix(), [[1, 0, 0, 0], [0, 1, 0, 0]])))
        self.assertTrue(np.all(np.isclose(self.model22.measurement_matrix(), [[1, 0], [0, 1]])))

    def test_noise_matrix(self):
        expected_noise = [[self.noise**2, 0], [0, self.noise**2]]
        self.assertTrue(
            np.all(
                np.isclose(
                    self.model24.noise_matrix(),
                    expected_noise,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    self.model22.noise_matrix(),
                    expected_noise,
                )
            )
        )
