import numpy as np
from scipy.stats import multivariate_normal, poisson

from gmphd_fusion.data import StateVector, StateVectors, Track
from gmphd_fusion.measurement_model import MeasurementModel
from gmphd_fusion.motion_models import LinearMotionModel, ConstantVelocityMotionModel


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


def _generate_clean_track(
    x0: StateVector, n_samples: int, state_transition_matrix: np.ndarray, label: int, time0: int
) -> Track:
    track = Track(label)
    x = x0
    for i in range(n_samples):
        track.add_estimate(StateVector(x), time0 + i)
        x = state_transition_matrix @ x
    track.finish(n_samples)
    return track


def generate_tracks_paper() -> list[Track]:
    nsamples = 100
    x0_t1 = StateVector([250.0, 250.0, 2.5, -12.0])
    x0_t2 = StateVector([-250.0, -250.0, 12, -2.5])
    motion_model = ConstantVelocityMotionModel(0.0)
    track_t1 = _generate_clean_track(x0_t1, nsamples, motion_model.state_transition_matrix(1.0), -1, 1)
    track_t2 = _generate_clean_track(x0_t2, nsamples, motion_model.state_transition_matrix(1.0), -2, 1)
    return [track_t1, track_t2]


def generate_tracks_time_varying(n_tracks: int = 6) -> list[Track]:
    motion_model = ConstantVelocityMotionModel(0.0)
    tracks = []
    n_angles = int(np.ceil(n_tracks / 2))
    angle_step = np.pi / 2 / (n_angles - 1) if n_angles > 1 else 0.0
    for i in range(n_tracks):
        # tracks differ in time of birth
        start_time = i * 10
        # every track will finish after this number of samples
        n_samples = 50
        birth_position = StateVector([-1000, 750, 0, 0])
        # every object changes angle of movement
        birth_velocity = StateVector(
            [0, 0, 25 * np.cos(int(i / 2) * angle_step), -25 * np.sin(int(i / 2) * angle_step)]
        )
        # two possible birth points (left-up and right-bottom)
        init_state = (-1) ** i * (birth_position + birth_velocity)
        track = _generate_clean_track(init_state, n_samples, motion_model.state_transition_matrix(), -i, start_time)
        tracks.append(track)
    return tracks


def generate_tracks_sudden_birth() -> list[Track]:
    motion_model = ConstantVelocityMotionModel(0.0)
    tracks = generate_tracks_time_varying()
    # now we generate two tracks in positions that are not expected by the filter
    for i in range(2):
        start_time = 20 + i * 40
        n_samples = 40
        birth_position = StateVector([-1000, -1000, 0, 0])
        birth_velocity = StateVector([0, 0, 20, 20])
        init_state = (-1) ** i * (birth_position + birth_velocity)
        track = _generate_clean_track(
            init_state, n_samples, motion_model.state_transition_matrix(), -i - len(tracks), start_time
        )
        tracks.append(track)
    return tracks


def generate_measurements(
    track: Track,
    measurement_model: MeasurementModel,
    detection_prob: float = 1.0,
    seed: int | None = None,
) -> Track:
    np.random.seed(seed)
    _H = measurement_model.measurement_matrix()
    _R = measurement_model.noise_matrix()
    measurements = []
    for e in track.estimates:
        if e is None or detection_prob < np.random.random():
            measurements.append(None)
            continue
        measurement = _H @ e  # shape: (measurement_dim, 1)
        noise = multivariate_normal.rvs(cov=_R, size=1)  # shape: (_R.shape[0],)
        measurement = measurement + noise.reshape(measurement.shape)
        measurements.append(measurement)

    measured_track = Track(label=track.label, start_time=track.start_time, estimates=measurements)
    measured_track.end_time = track.end_time
    return measured_track


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
