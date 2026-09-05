from __future__ import annotations

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import truncated_inverse_filter


class TruncatedSVDEstimator:
    name = "truncated_svd"

    def __init__(self, tau: float) -> None:
        if tau < 0.0:
            raise ValueError("tau must be nonnegative")
        self.tau = float(tau)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, float | int]]:
            U, singular_values, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
            filter_values = truncated_inverse_filter(singular_values, tau=self.tau)
            retained = int(np.count_nonzero(singular_values > self.tau))
            x_hat = Vt.T @ (filter_values * (U.T @ system.r_tilde))
            return x_hat, {"tau": self.tau, "retained_singular_directions": retained}

        return timed_solve(self.name, system, _solve)
