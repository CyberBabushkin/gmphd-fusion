import abc
import numpy as np


class MeasurementModel(abc.ABC):
    def __init__(self, dim_measurement: int, dim_state: int):
        self.dim_measurement = dim_measurement
        self.dim_state = dim_state

    @abc.abstractmethod
    def measurement_matrix(self) -> np.ndarray: ...

    @abc.abstractmethod
    def noise_matrix(self) -> np.ndarray: ...


class LinearCoordinateMeasurementModel(MeasurementModel):
    """Extracts `dim_measurement` first coordinates from the state vector of length `dim_state`."""

    def __init__(self, dim_measurement: int, dim_state: int, measurement_noise: float):
        super().__init__(dim_measurement=dim_measurement, dim_state=dim_state)
        self._validate_dimensions()
        self._measurement_noise = measurement_noise

    def measurement_matrix(self) -> np.ndarray:
        return np.pad(np.eye(self.dim_measurement), [(0, 0), (0, self.dim_state - self.dim_measurement)],
                      mode='constant', constant_values=0.)

    def noise_matrix(self) -> np.ndarray:
        return self._measurement_noise ** 2 * np.eye(self.dim_measurement)

    def _validate_dimensions(self):
        if self.dim_measurement > self.dim_state:
            raise ValueError("Measurement vector dimension should be less or equal to the state vector dimension.")

    def __repr__(self):
        return f"<{self.__class__.__name__}> State dim: {self.dim_state}" \
               f", measurement dim: {self.dim_measurement}" \
               f", measurement noise: {self._measurement_noise}>"
