from typing import Any

import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.stats import multivariate_normal

from .data import Track, StateVectors, extract_coordinate_from
from .gm import GaussianMixture

CMAPS = ("coolwarm", "turbo", "nipy_spectral", "brg")
COLORS_PER_CMAP = 64
MAX_COLORS = len(CMAPS) * COLORS_PER_CMAP


def _make_segments(x, y):
    """
    https://stackoverflow.com/a/25941474/3870394
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def _colorline(
    axis, x, y, z=None, cmap=plt.get_cmap("copper"), colors=None, norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0
):
    """
    https://stackoverflow.com/a/25941474/3870394
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)
    segments = _make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, colors=colors, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    axis.add_collection(lc)
    return lc


def _add_custom_legend_entry(
    legend_handles: list,
    legend_labels: list,
    label: str,
    entry: Any,
) -> None:
    if legend_handles is not None:
        legend_handles.append(entry)
    if legend_labels is not None:
        legend_labels.append(label)


def _rand_rgb_color() -> tuple[float, float, float, float]:
    # coolwarm, turbo, nipy_spectral, brg
    # tab10 (10pcs), Dark2(8pcs)
    cmaps = [cm.get_cmap(cmap, MAX_COLORS) for cmap in CMAPS]
    cmap = np.random.choice(cmaps)
    return cmap(np.random.random())


def _shades_gray(n: int) -> np.ndarray:
    return np.linspace(0.2, 0.6, n)


def _plot_scatter(
    axis: plt.Axes,
    vectors: list[StateVectors],
    marker: str,
    marker_size: int,
    linewidths: int,
    base_color: np.ndarray | tuple[float, float, float],
) -> None:
    """Each element of the input list is point at time k."""
    alphas = _shades_gray(len(vectors))

    colors = sum([[np.append(base_color, a)] * v.shape[1] for v, a in zip(vectors, alphas)], start=[])
    vectors = np.hstack(vectors)
    x = vectors[0, :].flatten()
    y = vectors[1, :].flatten()
    axis.scatter(x, y, color=colors, marker=marker, s=marker_size, linewidths=linewidths)


def _plot_tracks_time(
    axis: plt.Axes,
    tracks: list[Track],
    cmap_name: str = "Greys",
    nticks: int = 10,
) -> None:
    (time_min, time_max), x_all = extract_coordinate_from(tracks, 0)
    _, y_all = extract_coordinate_from(tracks, 1)
    for t, x_t, y_t in zip(tracks, x_all, y_all):
        path = mpath.Path(np.column_stack([x_t, y_t]))
        verts = path.interpolated(steps=3).vertices
        x, y = verts[:, 0], verts[:, 1]
        color_from = (t.start_time - time_min) / (time_max - time_min)
        color_to = (t.end_time - time_min) / (time_max - time_min)
        z = np.linspace(color_from, color_to, len(x))
        _colorline(axis, x, y, z, cmap=plt.get_cmap(cmap_name), linewidth=2)

    # legend
    if axis.collections:
        cb = plt.colorbar(axis.collections[-1], ax=axis)
        ticks = np.linspace(0, 1, nticks)
        tick_labels = [str(t) for t in np.linspace(time_min, time_max, nticks, dtype=int, endpoint=True)]
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels(tick_labels)
        cb.set_label("Time step")


def _plot_tracks_colors(
    axis: plt.Axes,
    tracks: list[Track],
    legend_handles: list = None,
    legend_labels: list = None,
) -> None:
    _, x_all = extract_coordinate_from(tracks, 0)
    _, y_all = extract_coordinate_from(tracks, 1)

    used_colors = set()
    for t, x_t, y_t in zip(tracks, x_all, y_all):
        color = _rand_rgb_color()
        while len(tracks) < MAX_COLORS and color in used_colors:
            color = _rand_rgb_color()
        axis.plot(x_t, y_t, color=color, marker="o", markersize=2, linestyle="--", linewidth=2, alpha=0.8)
        _add_custom_legend_entry(
            legend_handles, legend_labels, f"Track {t.label}", Line2D([0], [0], color=color, lw=1.2)
        )


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


def box_whisker_over_param(
    axis: plt.Axes,
    x_label: str,
    y_label: str,
    x_ticks: list[Any],
    measurements: list[list[float]],
) -> None:
    # https://stackoverflow.com/a/65529178
    props = {
        "boxprops": {"facecolor": "none", "edgecolor": "black"},
        "medianprops": {"color": "black"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
    }
    sns.boxplot(data=measurements, ax=axis, width=0.58, **props)
    axis.plot(np.mean(measurements, axis=1), color="red", marker="o", markersize=5)
    axis.set_xticks(list(range(len(measurements))), x_ticks)
    # axis.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


def visualize_coord_change(
    axis: plt.Axes,
    tracks: list[Track],
    coord_idx: int,
    coord_name: str,
    marker: str = None,
    legend_handles: list = None,
    legend_labels: list = None,
) -> None:
    """If marker is set, scatter will be used instead of a line plot."""
    (time_min, time_max), data = extract_coordinate_from(tracks, coord_idx)
    x = range(time_max - time_min)
    for td in data:
        if marker is None:
            axis.plot(x, td, color="black", linewidth=1.2)
        else:
            axis.scatter(x, td, facecolors="none", edgecolors="black", marker="o", s=20)

    axis.set_xticks(np.linspace(0, time_max - time_min, 11), np.linspace(time_min, time_max, 11, dtype=int))
    axis.set_xlabel("Time step")
    axis.set_ylabel(f"{coord_name} coordinate (in m)")
    axis.margins(x=0, y=0)
    _add_custom_legend_entry(legend_handles, legend_labels, "Ground truth", Line2D([0], [0], color="black", lw=1.2))


def visualize_estimated_tracks(
    axis: plt.Axes,
    tracks: list[Track],
    legend_handles: list = None,
    legend_labels: list = None,
) -> None:
    _plot_tracks_colors(axis, tracks, legend_handles=legend_handles, legend_labels=legend_labels)


def visualize_true_tracks(
    axis: plt.Axes,
    tracks: list[Track],
) -> None:
    # nipy_spectral
    _plot_tracks_time(axis, tracks, cmap_name="copper")


def visualize_measurements(
    axis: plt.Axes,
    measurements: list[Track],
    markersize: int = 15,
    linewidths: int = 1,
    legend_handles: list = None,
    legend_labels: list = None,
) -> None:
    red = (1.0, 0.0, 0.0)
    _, x_tracks_time = extract_coordinate_from(measurements, 0)
    _, y_tracks_time = extract_coordinate_from(measurements, 1)

    # the first coordinate is now the time step
    x_time_tracks = np.asarray(x_tracks_time).T
    y_time_tracks = np.asarray(y_tracks_time).T

    # create a sequence of StateVectors with all measurements for each time step
    vectors = [StateVectors(np.row_stack((x, y))) for x, y in zip(x_time_tracks, y_time_tracks)]
    _plot_scatter(axis, vectors, "*", markersize, linewidths, red)

    _add_custom_legend_entry(
        legend_handles,
        legend_labels,
        "Measurement",
        Line2D([0], [0], marker="*", color="w", markerfacecolor=np.append(red, 0.5), markersize=10),
    )


def visualize_clutter(
    axis: plt.Axes,
    clutter: list[StateVectors],
    markersize: int = 10,
    linewidths: int = 1,
    legend_handles: list = None,
    legend_labels: list = None,
) -> None:
    _plot_scatter(axis, clutter, "x", markersize, linewidths, np.zeros(3))
    _add_custom_legend_entry(
        legend_handles,
        legend_labels,
        "Clutter",
        Line2D([0], [0], marker="X", color="w", markerfacecolor=(0.0, 0.0, 0.0, 0.5), markersize=10),
    )
