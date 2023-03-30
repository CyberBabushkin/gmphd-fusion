"""Functions here may be defined in the Jupyter notebook. However, to use multiprocessing,
we have to define these in a separate module."""
import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from generator import generate_measurements, generate_clutter
from gmphd_fusion.data import Track, StateVectors
from gmphd_fusion.filter import KalmanFilter
from gmphd_fusion.gmphd import GMPHD
from gmphd_fusion.metrics import cpep_tracks, eae_targets_number
from gmphd_fusion.visualize import (
    visualize_mixture,
    visualize_clutter,
    visualize_measurements,
    visualize_true_tracks,
    visualize_estimated_tracks,
    box_whisker_over_param,
    visualize_coord_change,
)
from test_case import TestUseCase2D


def _short_hash_str(s: str) -> str:
    sha = hashlib.sha256()
    sha.update(s.encode())
    return sha.hexdigest()[:8]


def _save_close_fig(_fig: plt.Figure, _path: Path):
    _fig.savefig(str(_path), bbox_inches="tight")
    plt.close(_fig)


def save_tracks_measurements_clutter_plot(uc, clutter, measurements, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(*uc.xlim)
    ax.set_ylim(*uc.ylim)
    ax.set_facecolor("white")
    handles, labels = ax.get_legend_handles_labels()

    visualize_true_tracks(ax, uc.tracks_true)
    visualize_clutter(ax, clutter, legend_handles=handles, legend_labels=labels)
    visualize_measurements(ax, measurements, legend_handles=handles, legend_labels=labels)

    ax.legend(handles, labels, loc="upper right")
    _save_close_fig(fig, save_dir / "tracks_measurements_clutter_plot.png")


def save_tracks_estimates_plot(uc, tracks, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(*uc.xlim)
    ax.set_ylim(*uc.ylim)
    ax.set_facecolor("white")
    handles, labels = ax.get_legend_handles_labels()

    visualize_true_tracks(ax, uc.tracks_true)
    visualize_estimated_tracks(ax, tracks, legend_handles=handles, legend_labels=labels)

    ax.legend(handles, labels, loc="upper right")
    _save_close_fig(fig, save_dir / "tracks_estimates_plot.png")


def save_coordinate_change_plot(uc, tracks, coord_idx, coord_name, clutter, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, uc.samples_per_test)
    ax.set_ylim(*uc.ylim)

    # since the x coordinate is now time, we should modify clutter
    # x: time, y: selected coord
    clutter_mod = []
    for time, c_t in enumerate(clutter):
        n_t = c_t.shape[1]
        time_coord = [time] * n_t
        selected_coord = c_t[coord_idx, :].flatten()
        clutter_mod.append(StateVectors(np.row_stack((time_coord, selected_coord))))

    visualize_clutter(ax, clutter_mod)
    visualize_coord_change(ax, uc.tracks_true, coord_idx, coord_name)
    visualize_coord_change(ax, tracks, coord_idx, coord_name, marker="o", legend_handles=None, legend_labels=None)
    _save_close_fig(fig, save_dir / f"coord_{coord_name}_plot.png")


def save_posterior_plot(uc, posterior, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    x_res = y_res = 500
    ax.set_xticks(np.linspace(0, x_res, num=9), np.linspace(uc.xmin, uc.xmax, num=9))
    ax.set_yticks(np.linspace(0, y_res, num=9), np.linspace(uc.ymin, uc.ymax, num=9))

    visualize_mixture(
        axis=ax,
        mixture=posterior,
        time=uc.samples_per_test,
        xlim=uc.xlim,
        ylim=uc.ylim,
        x_res=x_res,
        y_res=y_res,
    )
    _save_close_fig(fig, save_dir / "posterior_plot.png")


def save_box_whisker_plot(param_name, labels, measurements, title, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    box_whisker_over_param(ax, param_name, labels, measurements, title)
    _save_close_fig(fig, save_dir / f"box_whisker_{_short_hash_str(title)}_plot.png")


def compute_metrics(uc: TestUseCase2D, tracks: list[Track], raw: bool = False) -> tuple[float | list[float], float]:
    state_extraction_matrix = uc.measurement_model.measurement_matrix()
    ntargets_true = []
    ntargets_est = []
    cpep_over_time = []

    for time in range(1, uc.samples_per_test + 1):
        ntt_k = [1 for t in uc.tracks_true if t.estimate_at(time) is not None]
        nte_k = [1 for t in tracks if t.estimate_at(time) is not None]
        ntargets_true.append(sum(ntt_k))
        ntargets_est.append(sum(nte_k))

        cpep_k = cpep_tracks(uc.tracks_true, tracks, time, uc.cpep_radius, state_extraction_matrix)
        cpep_over_time.append(cpep_k)

    eae = eae_targets_number(ntargets_true, ntargets_est)

    cpep_ret = np.mean(cpep_over_time) if not raw else cpep_over_time
    return cpep_ret, eae


def generate_tests(
    nsamples: int, use_cases: dict[str, TestUseCase2D], tested_params: dict[str, list[float]], base_dir: Path
) -> list[tuple[tuple, dict]]:
    param_configs = [(name, value) for name in tested_params.keys() for value in tested_params[name]]
    tests = []
    for test_name, uc in use_cases.items():
        for param_name, param_value in param_configs:
            for index in range(nsamples):
                args = (test_name, index, uc, base_dir)
                kwargs = {param_name: param_value}
                tests.append((args, kwargs))
    return tests


def run_test(test_name: str, index: int, uc: TestUseCase2D, out_dir: Path, **kwargs) -> None:
    def _pickle_obj(_path: Path, _obj):
        with _path.open("wb") as f:
            pickle.dump(_obj, f)

    if len(kwargs) != 1:
        raise ValueError("Only one additional parameter can be passed during one test.")

    param_name, param_value = kwargs.popitem()
    if param_name not in uc.__dict__:
        raise ValueError(f"Unknown param {param_name}")
    # copy is not necessary since this function will run in a different thread and objects are pickled
    setattr(uc, param_name, param_value)

    experiment_dir = out_dir / test_name / f"{param_name}={param_value:.3f}" / f"{index:03d}"
    experiment_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # disable random seed while experimenting
    # set_seed(uc.random_seed + index)
    kalman_filter = KalmanFilter()
    gmphd = GMPHD(
        init_gm=uc.init_gm,
        birth_gm=uc.birth_gm,
        filter=kalman_filter,
        motion_model=uc.motion_model,
        measurement_model=uc.measurement_model,
        clutter_spatial_density=uc.clutter_spatial_density,
        detection_prob=uc.detection_prob,
        survival_prob=uc.survival_prob,
        prune_threshold=uc.prune_threshold,
        merge_threshold=uc.merge_threshold,
        max_components=uc.max_components,
        target_weight_threshold=uc.target_weight_threshold,
    )
    measurements = [
        generate_measurements(t, uc.measurement_model, detection_prob=uc.detection_prob) for t in uc.tracks_true
    ]
    clutter = generate_clutter(uc.samples_per_test, uc.clutter_rate, uc.unif_min, uc.unif_max)

    for time in range(uc.samples_per_test):
        # fuse gaussians before acquiring measurements
        if time in uc.fuse:
            for gaussian, weight in uc.fuse[time]:
                gmphd.fuse(gaussian, weight)
        measurements_k = [t.estimate_at(time) for t in measurements]
        measurements_k = filter(lambda e: e is not None, measurements_k)
        measurements_k = list(measurements_k)
        clutter_k = clutter[time]
        meas_clut = StateVectors([*measurements_k, clutter_k])
        gmphd.step(meas_clut, timestamp=time + 1)
    gmphd.finish()
    tracks = gmphd.get_tracks(min_length=uc.track_min_len)

    # compute metrics
    m_cpep, m_eae = compute_metrics(uc, tracks, raw=True)

    # generate plots
    save_tracks_measurements_clutter_plot(uc, clutter, measurements, experiment_dir)
    save_tracks_estimates_plot(uc, tracks, experiment_dir)
    save_posterior_plot(uc, gmphd.posterior, experiment_dir)
    save_coordinate_change_plot(uc, tracks, 0, "x", clutter, experiment_dir)
    save_coordinate_change_plot(uc, tracks, 1, "y", clutter, experiment_dir)

    # save objects
    _pickle_obj(experiment_dir / "uc.pickle", uc)
    _pickle_obj(experiment_dir / "gmphd.pickle", gmphd)
    _pickle_obj(experiment_dir / "measurements.pickle", measurements)
    _pickle_obj(experiment_dir / "clutter.pickle", clutter)
    _pickle_obj(experiment_dir / "tracks.pickle", tracks)
    _pickle_obj(experiment_dir / "cpep_time.pickle", m_cpep)
    _pickle_obj(experiment_dir / "eae.pickle", m_eae)

    # create a flag to mark the experiment as finished
    (experiment_dir / "_FINISHED").touch()


def run_test_from_config(config) -> bool:
    args, kwargs = config
    try:
        run_test(*args, **kwargs)
        return True
    except:
        return False
