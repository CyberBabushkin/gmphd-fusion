import numpy as np

from .data import StateVector, StateVectors, Track


def cpep_tracks(
    tracks_true: list[Track],
    tracks_pred: list[Track],
    time: int,
    radius: float,
    state_extraction_matrix: np.ndarray | None = None,
) -> float:
    def _get_ests(_tracks):
        y = []
        for t in _tracks:
            t_est = t.estimate_at(time)
            if t_est is not None:
                y.append(t_est)
        return y

    y_true = _get_ests(tracks_true)
    y_pred = _get_ests(tracks_pred)

    if len(y_true) == 0:
        y_true = StateVectors([[]])
    if len(y_pred) == 0:
        y_pred = StateVectors([[]])

    return cpep(StateVectors(y_true), StateVectors(y_pred), radius, state_extraction_matrix)


def cpep(
    y_true: StateVectors, y_pred: StateVectors, radius: float, state_extraction_matrix: np.ndarray | None = None
) -> float:
    """Circular Position Error Probability.

    Implementation example: https://kr.mathworks.com/matlabcentral/fileexchange/46353-gaussian-mixture-cardinalized-probability-hypothesis-density-filter?focused=3818059&tab=function&s_tid=gn_loc_drop

    CEP: https://apps.dtic.mil/sti/pdfs/ADA199190.pdf"""

    def _prob(_state_true: StateVector, _states_pred: StateVectors):
        bool_vector = np.linalg.norm(_states_pred - _state_true, axis=0) > radius
        # return bool_vector.mean()
        return float(bool_vector.all())

    if y_true.shape[1] == 0 and y_pred.shape[1] == 0:
        return 0.0

    if (y_true.shape[1] != 0 and y_pred.shape[1] == 0) or (y_true.shape[1] == 0 and y_pred.shape[1] != 0):
        return 1.0

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("True and estimated shapes mismatch.")

    if state_extraction_matrix is None:
        state_extraction_matrix = np.eye(y_true.shape[0])
    states_true = StateVectors(state_extraction_matrix @ y_true)
    states_pred = StateVectors(state_extraction_matrix @ y_pred)
    cpep_metric = np.mean([_prob(s_true, states_pred) for s_true in states_true])
    return float(cpep_metric)


def eae_targets_number(num_targets_true, num_targets_pred):
    """Number of targets in time."""
    return np.mean(np.abs(np.subtract(num_targets_pred, num_targets_true)))


def gospa(y_true, y_pred, p, c, alpha):
    # TODO: maybe later, difficult to implement, needs auction algorithm
    # TODO: maybe just use stonesoup?
    raise NotImplementedError()
