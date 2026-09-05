from __future__ import annotations

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import inverse_filter


class PseudoinverseEstimator:
    name = "pseudoinverse"

    def __init__(self, rcond: float = 1e-12) -> None:
        if rcond < 0.0:
            raise ValueError("rcond must be nonnegative")
        self.rcond = float(rcond)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, float]]:
            singular_values = system.singular_values()
            cutoff = self.rcond * float(np.max(singular_values)) if singular_values.size else 0.0
            filter_values = inverse_filter(singular_values, eps=cutoff)
            U, _, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
            x_hat = Vt.T @ (filter_values * (U.T @ system.r_tilde))
            return x_hat, {"rcond": self.rcond, "cutoff": cutoff}

        return timed_solve(self.name, system, _solve)
