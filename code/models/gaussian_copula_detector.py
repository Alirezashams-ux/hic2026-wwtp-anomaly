from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import norm


@dataclass
class GaussianCopulaResult:
    train_scores: np.ndarray
    test_scores: np.ndarray
    threshold_99: float


class GaussianCopulaDetector:
    def __init__(self, epsilon: float = 1e-6, regularization: float = 1e-6):
        self.epsilon = epsilon
        self.regularization = regularization
        self.sorted_train_columns: List[np.ndarray] = []
        self.covariance: np.ndarray | None = None
        self.inv_covariance: np.ndarray | None = None
        self.log_det_covariance: float | None = None
        self.dim: int | None = None

    def _fit_empirical(self, x_train: np.ndarray) -> None:
        self.sorted_train_columns = [np.sort(x_train[:, j]) for j in range(x_train.shape[1])]

    def _to_uniform(self, x: np.ndarray) -> np.ndarray:
        if not self.sorted_train_columns:
            raise RuntimeError("Detector has not been fitted.")

        u = np.zeros_like(x, dtype=float)
        for j, sorted_vals in enumerate(self.sorted_train_columns):
            ranks = np.searchsorted(sorted_vals, x[:, j], side="right")
            n = len(sorted_vals)
            u[:, j] = np.clip(ranks / (n + 1.0), self.epsilon, 1.0 - self.epsilon)
        return u

    def _to_gaussian_space(self, x: np.ndarray) -> np.ndarray:
        u = self._to_uniform(x)
        return norm.ppf(u)

    def fit(self, x_train: np.ndarray) -> None:
        self._fit_empirical(x_train)
        z_train = self._to_gaussian_space(x_train)

        cov = np.cov(z_train, rowvar=False)
        cov = cov + self.regularization * np.eye(cov.shape[0])

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite after regularization.")

        self.covariance = cov
        self.inv_covariance = np.linalg.inv(cov)
        self.log_det_covariance = float(logdet)
        self.dim = cov.shape[0]

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.inv_covariance is None or self.log_det_covariance is None or self.dim is None:
            raise RuntimeError("Detector has not been fitted.")

        z = self._to_gaussian_space(x)
        quadratic = np.einsum("bi,ij,bj->b", z, self.inv_covariance, z)
        constant = self.dim * np.log(2.0 * np.pi)
        nll = 0.5 * (quadratic + self.log_det_covariance + constant)
        return nll

    def fit_and_score(self, x_train: np.ndarray, x_test: np.ndarray) -> GaussianCopulaResult:
        self.fit(x_train)
        train_scores = self.score_samples(x_train)
        test_scores = self.score_samples(x_test)
        threshold = float(np.quantile(train_scores, 0.99))
        return GaussianCopulaResult(
            train_scores=train_scores,
            test_scores=test_scores,
            threshold_99=threshold,
        )
