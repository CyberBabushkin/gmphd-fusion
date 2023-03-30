import abc

import numpy as np

from .data import StateVector, CovarianceMatrix


class MotionModel(abc.ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @abc.abstractmethod
    def __call__(
        self, mean: StateVector, cov: CovarianceMatrix, dt: float = 1.0
    ) -> tuple[StateVector, CovarianceMatrix]:
        pass

    def _validate_shapes(self, mean: StateVector, cov: CovarianceMatrix) -> None:
        expected_mean = (self.dim, 1)
        if mean.shape != expected_mean:
            raise ValueError(f"Wrong mean dimension. Got: {mean.shape}, expected: {expected_mean}")

        expected_cov = (self.dim, self.dim)
        if cov.shape != expected_cov:
            raise ValueError(f"Wrong cov dimension. Got: {cov.shape}, expected: {expected_cov}")


class LinearMotionModel(MotionModel):
    def __init__(self, dim: int, motion_noise: float):
        super().__init__(dim=dim)
        self._motion_noise = motion_noise

    @abc.abstractmethod
    def state_transition_matrix(self, dt: float) -> np.ndarray:
        ...

    @abc.abstractmethod
    def noise_matrix(self, dt: float) -> np.ndarray:
        ...

    def __call__(
        self, mean: StateVector, cov: CovarianceMatrix, dt: float = 1.0
    ) -> tuple[StateVector, CovarianceMatrix]:
        self._validate_shapes(mean, cov)
        state_transition = self.state_transition_matrix(dt)
        noise = self.noise_matrix(dt)
        mean_new = StateVector(state_transition @ mean)
        cov_new = CovarianceMatrix(state_transition @ cov @ state_transition.transpose() + noise)
        return mean_new, cov_new

    def __repr__(self):
        return f"<{self.__class__.__name__}> Motion noise = {self._motion_noise}>"


class RandomWalk(LinearMotionModel):
    """Random walk motion model.

    State vector: [x y]
    """

    def __init__(self, motion_noise: float):
        super().__init__(dim=2, motion_noise=motion_noise)

    def state_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        return np.eye(self.dim)

    def noise_matrix(self, dt: float = 1.0) -> np.ndarray:
        return self._motion_noise**2 * np.eye(self.dim)


class ConstantVelocityMotionModel(LinearMotionModel):
    """Constant Velocity (CV) motion model.

    State vector: [x y v_x v_y]
    """

    def __init__(self, motion_noise: float):
        super().__init__(dim=4, motion_noise=motion_noise)

    def state_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def noise_matrix(self, dt: float = 1.0) -> np.ndarray:
        return self._motion_noise**2 * np.array(
            [
                [dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**3 / 3, 0, dt**2 / 2],
                [dt**2 / 2, 0, dt, 0],
                [0, dt**2 / 2, 0, dt],
            ]
        )


class ConstantAccelerationMotionModel(LinearMotionModel):
    """Nearly constant velocity motion model.

    State vector: [x y dx dy ddx ddy]
    """

    def __init__(self, motion_noise: float):
        super().__init__(dim=6, motion_noise=motion_noise)

    def state_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0, dt**2 / 2, 0],
                [0, 1, 0, dt, 0, dt**2 / 2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def noise_matrix(self, dt: float = 1.0) -> np.ndarray:
        return self._motion_noise**2 * np.array(
            [
                [dt**5 / 20, 0, dt**4 / 8, 0, dt**3 / 6, 0],
                [0, dt**5 / 20, 0, dt**4 / 8, 0, dt**3 / 6],
                [dt**4 / 8, 0, dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**4 / 8, 0, dt**3 / 3, 0, dt**2 / 2],
                [dt**3 / 6, 0, dt**2 / 2, 0, dt, 0],
                [0, dt**3 / 6, 0, dt**2 / 2, 0, dt],
            ]
        )
