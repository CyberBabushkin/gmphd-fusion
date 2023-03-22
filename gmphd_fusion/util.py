from scipy.stats import multivariate_normal

from .data import StateVector, CovarianceMatrix


def evaluate_gaussian(mean: StateVector, cov: CovarianceMatrix, at: StateVector) -> float:
    return multivariate_normal.pdf(at.flatten(), mean=mean.flatten(), cov=cov)
