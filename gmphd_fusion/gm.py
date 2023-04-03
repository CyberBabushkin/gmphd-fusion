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
        """The representation of a multivariate gaussian.

        Parameters
        ----------
        mean : StateVector
            The mean vector.
        cov : CovarianceMatrix
            The covariance matrix.
        label : int or None
            The unique label or tag for the Gaussian. If None, an auto-incremented ID
            will be assigned automatically. Label `Gaussian.BIRTH_LABEL`, or `-1`, should
            be used for birth components.
        """
        self.mean = StateVector(mean)
        self.cov = CovarianceMatrix(cov)

        if self.mean.shape[0] != self.cov.shape[0]:
            raise ValueError("Shapes of mean and cov mismatch.")

        if label is None:
            self.label = self._allocate_new_label()
        else:
            self.label = label

    def new_label(self):
        """Allocate a new label for the Gaussian."""
        self.label = self._allocate_new_label()

    def copy(self, with_label: bool = False):
        """Deep copy the Gaussian.

        Parameters
        ----------
        with_label : bool, optional
            If True, a copy will be assigned the same label. If False (default),
            the new copy will be assigned with a new unique label.

        Returns
        -------
        Gaussian
            A copy of this Gaussian.
        """
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
        """A dummy method that always returns True. Its exists because the developer
        did not have time to create a good software design and because the purpose of
        this work does not include the best software implementation."""
        return True


class GaussianMixture:
    def __init__(self, gaussians: list[Gaussian] = (), weights: list[float] = ()):
        """Represents a Gaussian mixture with Gaussians and weights.

        Components are always sorted by weight in the descending order. Weights do not sum to one.

        Parameters
        ----------
        gaussians : list of Gaussian, optional
            A list of Gaussians of this mixture. If provided, `weights` should also be passed.
        weights : list of float, optional
            A list of weights of Gaussians. If provided, `gaussians` should also be passed.
        """
        self.gaussians: list[Gaussian] = list(gaussians)
        self.weights: list[float] = list(weights)
        self._sort_items()

    def add_to_mixture(self, gaussian: Gaussian, weight: float = 1.0):
        """Add a new Gaussian to the mixture.

        Parameters
        ----------
        gaussian : Gaussian
            A gaussian to be added.
        weight : float, optional
            The weight of the gaussian. Defatuls to 1.0.
        """
        insert_position = bisect.bisect_right(self.weights, -weight, key=lambda x: -x)
        self.gaussians.insert(insert_position, gaussian)
        self.weights.insert(insert_position, weight)

    def prune(self, threshold: float) -> None:
        """Remove all Gaussians from the mixture with weights less that `threshold`.

        Parameters
        ----------
        threshold : float
            The weight threshold. All Gaussians with weights less than this number will be removed.
        """
        remove_from = bisect.bisect_right(self.weights, -threshold, key=lambda x: -x)
        self.gaussians = self.gaussians[:remove_from]
        self.weights = self.weights[:remove_from]

    def remove_from_mixture(self, index: int) -> None:
        """Removes a Gaussian and its weight at index `index` from the mixture.

        Parameters
        ----------
        index : int
            The index of a Gaussian.

        Raises
        ------
        ValueError
            If the mixture does not have a Gaussian at this index.
        """
        if index >= len(self.gaussians):
            raise ValueError("Out of bounds error.")
        del self.gaussians[index]
        del self.weights[index]

    def truncate(self, max_components: int) -> None:
        """Leave at most `max_components` components with the largest weights in the mixture.

        Parameters
        ----------
        max_components : int
            The number of Gaussians that should be left.
        """
        self.gaussians = self.gaussians[:max_components]
        self.weights = self.weights[:max_components]

    def uniquify_labels(self) -> None:
        """Force all Gaussians in the mixture to have a unique label."""
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
        """Create a copy of the mixture.

        Parameters
        ----------
        with_labels : bool
            If True, all copies of the components will be assigned the same label as their source.
            If False (default), the new copy will be assigned with a new unique label.

        Returns
        -------
        GaussianMixture
            A copied mixture.
        """
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
