import numpy as np

from .data import StateVector, StateVectors


def cpep(y_true: StateVectors, y_pred: StateVectors, radius: float, state_extraction_matrix: np.ndarray | None = None):
    """Circular Position Error Probability.

    Implementation example: https://kr.mathworks.com/matlabcentral/fileexchange/46353-gaussian-mixture-cardinalized-probability-hypothesis-density-filter?focused=3818059&tab=function&s_tid=gn_loc_drop

    CEP: https://apps.dtic.mil/sti/pdfs/ADA199190.pdf"""
    def _prob(_state_true: StateVector, _states_pred: StateVectors):
        bool_vector = np.linalg.norm(_states_pred - _state_true, axis=0) > radius
        # return bool_vector.mean()
        return float(bool_vector.all())

    if state_extraction_matrix is None:
        state_extraction_matrix = np.eye(y_true.shape[0])
    states_true = StateVectors(state_extraction_matrix @ y_true)
    states_pred = StateVectors(state_extraction_matrix @ y_pred)
    cpep_metric = np.mean([_prob(s_true, states_pred) for s_true in states_true])
    return cpep_metric


def eae_targets_number(num_targets_true, num_targets_pred):
    """Number of targets in time."""
    return np.mean(np.abs(np.subtract(num_targets_pred, num_targets_true)))


def gospa(y_true, y_pred, p, c, alpha):
    # TODO: maybe later, difficult to implement, needs auction algorithm
    # TODO: maybe just use stonesoup?
    raise NotImplementedError()
