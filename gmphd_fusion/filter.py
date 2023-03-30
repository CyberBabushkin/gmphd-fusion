import abc

import numpy as np

from .data import StateVector, CovarianceMatrix
from .measurement_model import MeasurementModel, LinearCoordinateMeasurementModel
from .motion_models import MotionModel


class Filter(abc.ABC):
    @abc.abstractmethod
    def predict(
        self,
        mean: StateVector,
        cov: CovarianceMatrix,
        motion_model: MotionModel,
        *args,
        **kwargs,
    ) -> tuple[StateVector, CovarianceMatrix]:
        ...

    @abc.abstractmethod
    def predict_measurement(
        self, mean: StateVector, cov: CovarianceMatrix, measurement_model: MeasurementModel
    ) -> tuple[StateVector, CovarianceMatrix]:
        ...

    @abc.abstractmethod
    def update(
        self,
        mean: StateVector,
        cov: CovarianceMatrix,
        measurement: StateVector,
        measurement_model: MeasurementModel,
        *args,
        predicted_measurement: tuple[StateVector, CovarianceMatrix] | None = None,
        **kwargs,
    ) -> tuple[StateVector, CovarianceMatrix]:
        ...


class KalmanFilter(Filter):
    def predict(
        self,
        mean: StateVector,
        cov: CovarianceMatrix,
        motion_model: MotionModel,
        *args,
        dt: float = 1.0,
        **kwargs,
    ) -> tuple[StateVector, CovarianceMatrix]:
        mean_pred, cov_pred = motion_model(mean=mean, cov=cov, dt=dt)
        return mean_pred, cov_pred

    def predict_measurement(
        self, mean: StateVector, cov: CovarianceMatrix, measurement_model: LinearCoordinateMeasurementModel
    ) -> tuple[StateVector, CovarianceMatrix]:
        z_hat = self._z_hat(mean, measurement_model)
        _S = self._innovation_covariance(cov, measurement_model)
        return z_hat, _S

    def update(
        self,
        mean: StateVector,
        cov: CovarianceMatrix,
        measurement: StateVector,
        measurement_model: LinearCoordinateMeasurementModel,
        *args,
        predicted_measurement: tuple[StateVector, CovarianceMatrix] | None = None,
        z_hat: StateVector | None = None,
        innovation_covariance: CovarianceMatrix | None = None,
        **kwargs,
    ) -> tuple[StateVector, CovarianceMatrix]:
        """Returns posterior mean m, posterior cov P"""
        # measurement matrix
        _H = measurement_model.measurement_matrix()
        # measurement noise
        _R = measurement_model.noise_matrix()

        if predicted_measurement is None:
            # predicted measurement mean
            z_hat = _H @ mean
            # the innovation covariance S = H * P * H' + R
            _S = _H @ cov @ _H.transpose() + _R
        else:
            z_hat, _S = predicted_measurement

        # the innovation 𝜈
        nu = measurement - z_hat
        # Kalman gain W = P * H' * inv(S)
        _W = cov @ _H.transpose() @ np.linalg.inv(_S)
        # posterior state estimate
        mean_posterior = StateVector(mean + _W @ nu)
        # posterior covariance (Joseph form)
        iwh = np.eye(_W.shape[0]) - _W @ _H
        cov_posterior = iwh @ cov @ iwh.transpose() + _W @ _R @ _W.transpose()
        cov_posterior = CovarianceMatrix(cov_posterior)
        return mean_posterior, cov_posterior

    @staticmethod
    def _z_hat(mean: StateVector, measurement_model: LinearCoordinateMeasurementModel) -> StateVector:
        return StateVector(measurement_model.measurement_matrix() @ mean)

    @staticmethod
    def _innovation_covariance(
        cov: CovarianceMatrix, measurement_model: LinearCoordinateMeasurementModel
    ) -> CovarianceMatrix:
        _H = measurement_model.measurement_matrix()
        _R = measurement_model.noise_matrix()
        _S = CovarianceMatrix(_H @ cov @ _H.transpose() + _R)
        return _S
