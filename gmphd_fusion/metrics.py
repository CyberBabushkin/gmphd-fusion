import numpy as np


def cpep(y_true, y_pred, radius, state_extraction_matrix=None):
    """Circular Position Error Probability.

    The shapes are expected to be (#observations, measurement_dim).
    If state_extraction_matrix is defined, it's shape should be (state_dim, measurement_dim).

    CEP: https://apps.dtic.mil/sti/pdfs/ADA199190.pdf"""
    def _prob(_state_true, _states_pred):
        bool_vector = np.linalg.norm(_states_pred - _state_true, axis=1) > radius
        return bool_vector.mean()

    if state_extraction_matrix is None:
        state_extraction_matrix = np.eye(y_true.shape[1])
    states_true = y_true @ state_extraction_matrix.T
    states_pred = y_pred @ state_extraction_matrix.T
    cpep_metric = np.mean([_prob(s_true, states_pred) for s_true in states_true])
    return cpep_metric


def eae_targets_number(num_targets_true, num_targets_pred):
    """Number of targets in time."""
    return np.mean(np.abs(num_targets_pred - num_targets_true))


def gospa(y_true, y_pred, p, c, alpha):
    # TODO: maybe later, difficult to implement, needs auction algorithm
    # TODO: maybe just use stonesoup?
    raise NotImplementedError()
