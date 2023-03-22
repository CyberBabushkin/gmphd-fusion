# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from .data import Track
from .gm import GaussianMixture


def draw_state_vector(axis: plt.Axes, mean: np.ndarray, symbol: str, color: str) -> None:
    mean = mean.flatten()
    axis.plot(mean[0], mean[1], symbol, color=color)


def visualize_mixture(
    axis: plt.Axes,
    mixture: GaussianMixture,
    time: int,
    xlim: tuple[int, int],
    ylim: tuple[int, int],
    x_res: int = 100,
    y_res: int = 100,
) -> None:
    """Axis is expected to already have limits set up. 2D only.

    Means should be: list(mean) where mean.shape=(2,1), len(list(mean)) = #components.

    Covs should be: list(cov) where cov.shape=(2,2), len(list(cov)) = #components.

    Weights should be: list(weight) where 0<=weight<=1, len(list(weight)) = #components.

    Based on: https://stackoverflow.com/a/44947434/3870394"""
    means = [g.mean[:2] for g in mixture.gaussians]
    covs = [g.cov[:2, :2] for g in mixture.gaussians]
    weights = mixture.weights

    gaussians = [multivariate_normal(mean=m.flatten(), cov=c) for m, c in zip(means, covs)]
    x_from, x_to = xlim
    y_from, y_to = ylim
    x = np.linspace(x_from, x_to, x_res)
    y = np.linspace(y_from, y_to, y_res)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = sum([w * g.pdf(xxyy) for g, w in zip(gaussians, weights)])
    img = zz.reshape((x_res, y_res))[::-1, :]
    axis.imshow(img)
    axis.set_title(f"k = {time}")


def _rand_rgb_color() -> np.ndarray:
    return np.random.rand(
        3,
    )


def _visualize_ground_truth(
    axis: plt.Axes,
    tracks: list[Track],
) -> None:
    for t in tracks:
        vector_estimates = np.hstack(t.estimates)
        x = vector_estimates[0, :]
        y = vector_estimates[1, :]
        axis.plot(x, y, color="b", marker="|", linestyle="-", label=f"Ground truth", markersize=1, linewidth=1, alpha=0.8)


def _visualize_tracks(
    axis: plt.Axes,
    tracks: list[Track],
) -> None:
    for t in tracks:
        vector_estimates = np.hstack([est for est in t.estimates if est is not None])
        x = vector_estimates[0, :]
        y = vector_estimates[1, :]
        axis.plot(
            x,
            y,
            color=_rand_rgb_color(),
            marker="o",
            markersize=1,
            linestyle="-",
            label=f"Track {t.label}",
            alpha=0.8
        )


def _visualize_estimates(
        axis: plt.Axes,
        estimates: list[np.ndarray],
) -> None:
    if not estimates:
        return
    estimates = np.hstack(estimates)
    x = estimates[0, :]
    y = estimates[1, :]
    axis.scatter(x, y, color=_rand_rgb_color(), marker="o", label="Estimates", s=10, alpha=0.8)


def _visualize_measurements(
    axis: plt.Axes,
    measurements: np.ndarray,
) -> None:
    if not measurements.size:
        return
    x = measurements[0, :]
    y = measurements[1, :]
    axis.scatter(x, y, color="r", marker="*", label="Measurements", s=10)


def _visualize_clutter(
    axis: plt.Axes,
    clutter: np.ndarray,
) -> None:
    if not clutter.size:
        return
    x = clutter[0, :]
    y = clutter[1, :]
    axis.scatter(x, y, color="b", marker="x", s=1, label="Clutter")


def visualize_trajectories(
    axis: plt.Axes,
    ground_truth: list[Track],
    estimates: list[Track] | list[np.ndarray],
    measurements: np.ndarray,
    clutter: np.ndarray,
    time: int,
) -> None:
    """2D only.

    Clutter and measurements should be a list of matrices of size (2, n) where n is the
    number of clutter/measurements points at time step k."""

    _visualize_clutter(axis, clutter)
    _visualize_ground_truth(axis, ground_truth)
    _visualize_measurements(axis, measurements)
    if estimates and isinstance(estimates[0], np.ndarray):
        _visualize_estimates(axis, estimates)
    else:
        _visualize_tracks(axis, estimates)

    axis.set_title(f"Time = {time}")
    # axis.legend()
