"""This file contain the definition of a Gaussian Mixture class."""
from __future__ import annotations

import bisect
from collections import defaultdict
from copy import deepcopy

from .data import StateVector, CovarianceMatrix


class Gaussian:

    # An auto-generated label, incremented
    _next_label: int = 0
    BIRTH_LABEL: int = -1

    def __init__(self, mean: StateVector, cov: CovarianceMatrix, label: int | None = None):
        """Label -1 should be used for birth components."""
        self.mean = StateVector(mean)
        self.cov = CovarianceMatrix(cov)

        if self.mean.shape[0] != self.cov.shape[0]:
            raise ValueError("Shapes of mean and cov mismatch.")

        if label is None:
            self.label = self._allocate_new_label()
        else:
            self.label = label

    def new_label(self):
        self.label = self._allocate_new_label()

    def copy(self, with_label: bool = False):
        if with_label:
            return Gaussian(mean=self.mean.copy(), cov=self.cov.copy(), label=self.label)
        else:
            return Gaussian(mean=self.mean.copy(), cov=self.cov.copy(), label=None)

    @classmethod
    def _allocate_new_label(cls):
        ret = cls._next_label
        cls._next_label += 1
        return ret

    def __lt__(self, other):
        return True


class GaussianMixture:

    def __init__(self, gaussians: list[Gaussian] = (), weights: list[float] = ()):
        """Components are always sorted by weight in the descending order."""
        self.gaussians: list[Gaussian] = list(gaussians)
        self.weights: list[float] = list(weights)
        self._sort_items()

    def add_to_mixture(self, gaussian: Gaussian, weight: float = 1.0):
        insert_position = bisect.bisect_right(self.weights, -weight, key=lambda x: -x)
        self.gaussians.insert(insert_position, gaussian)
        self.weights.insert(insert_position, weight)

    def prune(self, threshold: float) -> None:
        remove_from = bisect.bisect_right(self.weights, -threshold, key=lambda x: -x)
        self.gaussians = self.gaussians[:remove_from]
        self.weights = self.weights[:remove_from]

    def remove_from_mixture(self, index: int) -> None:
        if index >= len(self.gaussians):
            raise ValueError("Out of bounds error.")
        del self.gaussians[index]
        del self.weights[index]

    def truncate(self, max_components: int) -> None:
        self.gaussians = self.gaussians[:max_components]
        self.weights = self.weights[:max_components]

    def uniquify_labels(self) -> None:
        labels: dict[int, list[Gaussian]] = defaultdict(list)
        for g in self.gaussians:
            labels[g.label].append(g)
        for label, gs in labels.items():
            # the following will only run for lists with more than one element
            # if the length is 1, the label if already unique
            # otherwise, leave the first component (it has the highest weight) and assign new labels to others
            for g in gs[1:]:
                g.new_label()

    def __iter__(self):
        return iter(zip(self.gaussians, self.weights))

    def __add__(self, other: GaussianMixture):
        return GaussianMixture(
            gaussians=deepcopy(self.gaussians) + deepcopy(other.gaussians),
            weights=deepcopy(self.weights) + deepcopy(other.weights),
        )

    def __len__(self):
        assert len(self.gaussians) == len(self.weights)
        return len(self.gaussians)

    def copy(self, with_labels: bool = False):
        return GaussianMixture(
            gaussians=[g.copy(with_label=with_labels) for g in self.gaussians],
            weights=[w for w in self.weights],
        )

    def _sort_items(self) -> None:
        assert len(self.weights) == len(self.gaussians)
        if len(self.weights) == 0:
            return
        self.weights, self.gaussians = zip(*sorted(zip(self.weights, self.gaussians), reverse=True))
        self.weights, self.gaussians = list(self.weights), list(self.gaussians)
