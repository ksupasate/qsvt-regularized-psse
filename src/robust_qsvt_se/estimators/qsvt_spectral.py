from __future__ import annotations

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import qsvt_regularized_filter


class QSVTSpectralEstimator:
    name = "qsvt_regularized"

    def __init__(self, alpha: float) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, str | float]]:
            U, singular_values, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
            filter_values = qsvt_regularized_filter(singular_values, alpha=self.alpha)
            x_hat = Vt.T @ (filter_values * (U.T @ system.r_tilde))
            return x_hat, {
                "alpha": self.alpha,
                "simulation_scope": "classical singular-value filter simulation",
                "qsvt_target_filter": "sigma / (sigma^2 + alpha)",
            }

        return timed_solve(self.name, system, _solve)
