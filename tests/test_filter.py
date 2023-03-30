from unittest import TestCase

import numpy as np
from filterpy.kalman import KalmanFilter as TrueKF

from gmphd_fusion.filter import KalmanFilter as TestKF
from gmphd_fusion.measurement_model import LinearCoordinateMeasurementModel
from gmphd_fusion.motion_models import ConstantVelocityMotionModel


class TestKalmanFilter(TestCase):
    def _setup_true_filter(self):
        f = TrueKF(dim_x=self.dim_state, dim_z=self.dim_measurement)
        f.x = self.init_mean
        f.P = self.init_cov
        f.F = self.motion_model.state_transition_matrix(dt=1)
        f.Q = self.motion_model.noise_matrix(dt=1)
        f.H = self.measurement_model.measurement_matrix()
        f.R = self.measurement_model.noise_matrix()
        self.true_filter = f

    def _setup_test_filter(self):
        self.tested_filter = TestKF()
        self.tested_mean = self.init_mean
        self.tested_cov = self.init_cov

    def setUp(self) -> None:
        self.dim_state = 4
        self.dim_measurement = 2
        self.motion_noise = 0.5
        self.measurement_noise = 0.5
        self.init_mean = np.array([[1], [1], [1], [1]])
        self.init_cov = np.eye(4) * 2
        self.motion_model = ConstantVelocityMotionModel(motion_noise=self.motion_noise)
        self.measurement_model = LinearCoordinateMeasurementModel(
            self.dim_measurement, self.dim_state, self.measurement_noise
        )
        self.measurements = np.array(
            [
                self.measurement_noise * np.random.randn(2) + self.motion_noise * np.random.randn(2) + np.array([i, i])
                for i in range(2, 200)
            ]
        )
        self._setup_test_filter()
        self._setup_true_filter()

    def _step_predict(self):
        self.tested_mean, self.tested_cov = self.tested_filter.predict(
            self.tested_mean, self.tested_cov, self.motion_model
        )
        self.true_filter.predict()

    def test_predict(self):
        for _ in range(100):
            self._step_predict()
            true_mean = self.true_filter.x
            true_cov = self.true_filter.P
            self.assertTrue(np.allclose(true_mean, self.tested_mean))
            self.assertTrue(np.allclose(true_cov, self.tested_cov))

    def test_update(self):
        for m in self.measurements:
            self._step_predict()
            m = m.reshape((2, 1))

            self.tested_mean, self.tested_cov = self.tested_filter.update(
                self.tested_mean, self.tested_cov, m, self.measurement_model
            )
            self.true_filter.update(m)
            true_mean = self.true_filter.x
            true_cov = self.true_filter.P

            self.assertTrue(np.allclose(true_mean, self.tested_mean))
            self.assertTrue(np.allclose(true_cov, self.tested_cov))
