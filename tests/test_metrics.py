from unittest import TestCase

import numpy as np

from gmphd_fusion.data import StateVector, StateVectors
from gmphd_fusion.metrics import cpep, eae_targets_number


class TestMetrics(TestCase):
    def test_cpep(self):
        def _l2(v1, v2):
            return np.sqrt(np.sum(np.power(v1 - v2, 2)))

        def _prob(_states, _state_pivot, r):
            bv = []
            for s in _states:
                bv.append(_l2(s, _state_pivot) > r)
            return sum(bv) / len(bv)

        state_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        radius = 2
        true_targets = StateVectors([StateVector([i, i, 5, 5]) for i in range(1, 6)])
        true_states = StateVectors(state_matrix @ true_targets)
        measured_targets = StateVectors([StateVector([i / 10, i * 10, 7, 7]) for i in range(1, 6)])
        measured_states = StateVectors(state_matrix @ measured_targets)

        probs = [_prob(measured_states, tt, radius) for tt in true_states]
        cpep_exp = sum(probs) / len(probs)
        cpep_true = cpep(true_targets, measured_targets, radius, state_extraction_matrix=state_matrix)
        self.assertAlmostEqual(cpep_true, cpep_exp)

    def test_eae_targets_number(self):
        true_targets = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0])
        measured_targets = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        eae_exp = 0.6
        eae_true = eae_targets_number(true_targets, measured_targets)
        self.assertEqual(eae_true, eae_exp)
