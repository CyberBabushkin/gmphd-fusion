import random

import numpy as np
from scipy.stats import multivariate_normal

from .data import StateVector, CovarianceMatrix


def evaluate_gaussian(mean: StateVector, cov: CovarianceMatrix, at: StateVector) -> float:
    return multivariate_normal.pdf(at.flatten(), mean=mean.flatten(), cov=cov)


def set_seed(seed: int | None) -> None:
    np.random.seed(seed)
    random.seed(seed)
