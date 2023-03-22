import numpy as np
from scipy.stats import multivariate_normal, poisson

from .data import StateVector, StateVectors
from .measurement_model import MeasurementModel
from .motion_models import LinearMotionModel


def generate_trajectory(
    nsamples: int, motion_model: LinearMotionModel, x0: StateVector, seed: int | None = None, dt: float = 1.0
) -> StateVectors:
    if len(x0.shape) != 2 and x0.shape[1] != 1:
        raise ValueError("x0 should be a column vector.")

    np.random.seed(seed)
    _A = motion_model.state_transition_matrix(dt=dt)
    _Q = motion_model.noise_matrix(dt=dt)
    trajectory = np.zeros((x0.shape[0], nsamples))

    x = x0
    for i in range(nsamples):
        x = _A @ x + multivariate_normal.rvs(cov=_Q).reshape((-1, 1))
        trajectory[:, i] = x.flatten()

    return StateVectors(trajectory)


def generate_measurements(
    trajectory: StateVectors,
    measurement_model: MeasurementModel,
    seed: int | None = None,
) -> StateVectors:
    np.random.seed(seed)
    _H = measurement_model.measurement_matrix()
    _R = measurement_model.noise_matrix()
    measurements = _H @ trajectory  # shape: (measurement_dim, trajectory.shape[0])
    noise = multivariate_normal.rvs(cov=_R, size=trajectory.shape[1])  # shape: (trajectory.shape[1], cov.shape[0])
    measurements = measurements + noise.T
    return StateVectors(measurements)


def generate_clutter(
    steps: int,
    poisson_intensity: float,
    uniform_min: np.ndarray,
    uniform_max: np.ndarray,
    seed: int | None = None,
) -> list[StateVectors]:
    """The number of points in each step is determined by a sample from the Poisson distribution
    with a given intensity. Each point is sampled from a uniform distribution between
    uniform_min and uniform_max. The returned value contains matrices, each of different sizes
    (point_dim, clutter_at_time_step)."""
    np.random.seed(seed)
    samples = []
    for i in range(steps):
        sample_size = poisson.rvs(poisson_intensity)
        clutter_step = np.random.default_rng().uniform(
            low=uniform_min, high=uniform_max, size=(uniform_min.shape[0], sample_size)
        )
        samples.append(StateVectors(clutter_step))
    return samples
