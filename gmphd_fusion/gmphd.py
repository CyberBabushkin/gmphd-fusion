"""GM-PHD filter with labels and fusion capabilities."""
import itertools
from multiprocessing.pool import Pool

import numpy as np

from .data import Track, StateVector, StateVectors, CovarianceMatrix
from .filter import Filter
from .gm import Gaussian, GaussianMixture
from .measurement_model import MeasurementModel
from .motion_models import MotionModel
from .util import evaluate_gaussian


class GMPHD:
    def __init__(
        self,
        init_gm: GaussianMixture,
        birth_gm: GaussianMixture,
        filter: Filter,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        clutter_spatial_density: float,
        detection_prob: float,
        survival_prob: float,
        prune_threshold: float,
        merge_threshold: float,
        max_components: int,
        target_weight_threshold: float = 0.5,
        thread_pool: Pool | None = None,
    ):
        self.prior: GaussianMixture = init_gm
        self.posterior: GaussianMixture = init_gm
        self.birth_gm: GaussianMixture = birth_gm
        self.filter: Filter = filter
        self.motion_model: MotionModel = motion_model
        self.measurement_model: MeasurementModel = measurement_model
        self.clutter_spatial_density: float = clutter_spatial_density
        self.detection_prob: float = detection_prob
        self.survival_prob: float = survival_prob
        self.prune_threshold: float = prune_threshold
        self.merge_threshold: float = merge_threshold
        self.max_components: int = max_components
        self.target_weight_threshold: float = target_weight_threshold
        self.timestamp: int = 0
        self.tracks: dict[int, Track] = {}
        self.finished_tracks: dict[int, Track] = {}
        self._thread_pool = thread_pool or Pool()

    def step(self, measurements: StateVectors, timestamp: float):
        """Measurements expected shape: (measurement_dim, #measurements)"""
        if timestamp <= self.timestamp:
            raise ValueError("Time should increase.")
        dt = timestamp - self.timestamp
        self._predict(dt=dt)
        self._update(measurements)
        self.timestamp = timestamp
        self._update_tracks()

    def fuse(self, gaussian: Gaussian, weight: float) -> None:
        self.posterior.add_to_mixture(gaussian, weight)

    def estimate_states(self) -> dict[int, StateVector]:
        estimates = {}
        for g, w in self.posterior:
            if w > self.target_weight_threshold:
                estimates[g.label] = g.mean
        return estimates

    def get_tracks(self, min_length: int | None = None) -> list[Track]:
        tracks = sorted(
            list(self.tracks.values()) + list(self.finished_tracks.values()),
            key=lambda t: t.start_time,
        )
        if min_length is not None:
            tracks = [t for t in tracks if len(t) >= min_length]
        return tracks

    def finish(self) -> None:
        while self.tracks:
            l, t = self.tracks.popitem()
            t.finish(self.timestamp)
            self.finished_tracks[l] = t

    def _predict(self, dt: float = 1.0):
        birth_intensity = self._predict_birth()
        survived_intensity = self._predict_survived(dt=dt)
        self.prior = survived_intensity + birth_intensity

    def _predict_survived(self, dt: float = 1.0) -> GaussianMixture:
        mixture_predict = GaussianMixture()
        for gaussian, weight in self.posterior:
            mean_predict, cov_predict = self.filter.predict(gaussian.mean, gaussian.cov, self.motion_model, dt=dt)
            weight_predict = weight * self.survival_prob
            predict_gaussian = Gaussian(mean_predict, cov_predict, label=gaussian.label)
            mixture_predict.add_to_mixture(predict_gaussian, weight_predict)
        return mixture_predict

    def _predict_birth(self):
        return self.birth_gm.copy(with_labels=False)

    def _update(self, measurements: StateVectors):
        hypotheses = [self._create_misdetection_hypotheses()]
        measurements_hypotheses = self._create_measurements_hypotheses(measurements)
        for hypothesis_mixture in measurements_hypotheses:
            self._update_weights(hypothesis_mixture)
        hypotheses.extend(measurements_hypotheses)
        posterior_mixture = sum(hypotheses, start=GaussianMixture())
        posterior_mixture = self._prune(posterior_mixture)
        posterior_mixture = self._merge(posterior_mixture)
        posterior_mixture = self._truncate_components(posterior_mixture)
        posterior_mixture = self._process_labels(posterior_mixture)
        self.posterior = posterior_mixture
        self.prior = None

    def _create_misdetection_hypotheses(self) -> GaussianMixture:
        misdetection_mixture = GaussianMixture()
        for component, weight in self.prior:
            misdetection_weight = (1 - self.detection_prob) * weight
            misdetection_mixture.add_to_mixture(gaussian=component.copy(with_label=True), weight=misdetection_weight)
        return misdetection_mixture

    @staticmethod
    def _create_one_measurement_hypothesis(
        measurement: StateVector,
        predicted_measurements: list[tuple[StateVector, CovarianceMatrix]],
        prior: GaussianMixture,
        detection_prob: float,
        filter: Filter,
        measurement_model: MeasurementModel,
    ) -> GaussianMixture:
        measurement_mixture = GaussianMixture()
        for (component, weight), predicted_measurement in zip(prior, predicted_measurements):
            mean, cov = component.mean, component.cov
            meas_mean, meas_cov = predicted_measurement
            posterior_weight = detection_prob * weight * evaluate_gaussian(meas_mean, meas_cov, at=measurement)
            posterior_mean, posterior_cov = filter.update(
                mean,
                cov,
                measurement,
                measurement_model,
                predicted_measurement=predicted_measurement,
            )
            measurement_mixture.add_to_mixture(
                Gaussian(posterior_mean, posterior_cov, component.label),
                posterior_weight,
            )
        return measurement_mixture

    def _create_measurements_hypotheses(self, measurements: StateVectors) -> list[GaussianMixture]:
        # Measurements shape: (meas_sim, #meas)
        # eta and S for all existing components
        predicted_measurements = [
            self.filter.predict_measurement(g.mean, g.cov, self.measurement_model) for g, _ in self.prior
        ]
        arguments = zip(
            measurements,
            itertools.repeat(predicted_measurements),
            itertools.repeat(self.prior),
            itertools.repeat(self.detection_prob),
            itertools.repeat(self.filter),
            itertools.repeat(self.measurement_model),
        )
        measurement_mixtures = self._thread_pool.starmap(self._create_one_measurement_hypothesis, arguments)
        return measurement_mixtures

    def _update_weights(self, mixture: GaussianMixture) -> None:
        normalization = sum(mixture.weights) + self.clutter_spatial_density
        mixture.weights = [w / normalization for w in mixture.weights]

    def _prune(self, mixture: GaussianMixture) -> GaussianMixture:
        mixture.prune(self.prune_threshold)
        return mixture

    def _merge(self, mixture: GaussianMixture) -> GaussianMixture:
        unprocessed_indices = set(range(len(mixture)))
        merged_mixture = GaussianMixture()
        while unprocessed_indices:
            max_weight_idx = min(unprocessed_indices)  # components in the mixture are always sorted
            unprocessed_indices.remove(max_weight_idx)
            max_mean = mixture.gaussians[max_weight_idx].mean
            merged_indices = {max_weight_idx}
            for i in unprocessed_indices:
                comp_mean = mixture.gaussians[i].mean
                comp_cov = mixture.gaussians[i].cov
                dist = comp_mean - max_mean
                mahalanobis = dist.T @ np.linalg.inv(comp_cov) @ dist
                if mahalanobis.take(0) <= self.merge_threshold:
                    merged_indices.add(i)
            new_gaussian, new_weight = self._merge_gaussians(
                gaussians=[mixture.gaussians[i] for i in merged_indices],
                weights=[mixture.weights[i] for i in merged_indices],
                label=mixture.gaussians[max_weight_idx].label,
            )
            merged_mixture.add_to_mixture(new_gaussian, new_weight)
            unprocessed_indices = unprocessed_indices.difference(merged_indices)
        return merged_mixture

    @staticmethod
    def _merge_gaussians(gaussians: list[Gaussian], weights: list[float], label: int) -> tuple[Gaussian, float]:
        if len(gaussians) == 1:
            g = gaussians[0]
            w = weights[0]
            return Gaussian(mean=g.mean, cov=g.cov, label=label), w

        weights = np.array(weights).reshape((-1, 1))  # shape: (len(weights), 1)
        weight_sum = weights.sum()
        means = np.hstack([g.mean for g in gaussians]).T  # shape: (len(weights), state_dim)
        weighted_mean = StateVector((means.T @ weights) / weight_sum)  # shape: (state_dim, 1)
        covs = np.array([g.cov for g in gaussians])  # shape: (len(weights), state_dim, state_dim)
        mean_diffs = -1 * (means - weighted_mean.flatten())  # shape: (len(weights), state_dim)
        weighted_cov = CovarianceMatrix(
            np.sum(
                weights[:, :, None] * (covs + mean_diffs[:, :, np.newaxis] @ mean_diffs[:, np.newaxis, :]),
                axis=0,
            )
            / weight_sum
        )  # shape: (state_dim, state_dim)
        return Gaussian(mean=weighted_mean, cov=weighted_cov, label=label), weight_sum

    def _truncate_components(self, mixture: GaussianMixture) -> GaussianMixture:
        mixture.truncate(self.max_components)
        return mixture

    @staticmethod
    def _process_labels(mixture: GaussianMixture) -> GaussianMixture:
        mixture.uniquify_labels()
        return mixture

    def _update_tracks(self) -> None:
        states = self.estimate_states()
        self._add_track_estimates(states)
        self._finish_tracks()

    def _add_track_estimates(self, detected_states: dict[int, StateVector]) -> None:
        for label, state in detected_states.items():
            if label not in self.tracks:
                self.tracks[label] = Track(label)
            track = self.tracks[label]
            track.add_estimate(state, self.timestamp)

    def _finish_tracks(self) -> None:
        existing_labels = {g.label for g, w in self.posterior}
        track_labels = set(self.tracks.keys())
        finished_labels = track_labels.difference(existing_labels)
        for label in finished_labels:
            track = self.tracks[label]
            if not track.is_finished():
                track.finish(self.timestamp)
            del self.tracks[label]
            self.finished_tracks[label] = track
