from __future__ import annotations

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import qsvt_unregularized_inverse_filter


class QSVTUnregularizedInverseEstimator:
    name = "qsvt_unregularized_inverse"

    def __init__(self, cutoff: float = 1.0e-8) -> None:
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, bool | float | int | str]]:
            U, singular_values, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
            filter_values = qsvt_unregularized_inverse_filter(
                singular_values,
                cutoff=self.cutoff,
            )
            x_hat = Vt.T @ (filter_values * (U.T @ system.r_tilde))
            below_cutoff = int(np.count_nonzero(singular_values < self.cutoff))
            return x_hat, {
                "cutoff": self.cutoff,
                "singular_values_below_cutoff": below_cutoff,
                "max_filter_value": float(np.max(filter_values)) if filter_values.size else 0.0,
                "unstable_ablation": True,
                "simulation_scope": "classical singular-value filter simulation",
                "qsvt_target_filter": "1 / max(sigma, cutoff)",
                "caveat": "Unregularized inverse ablation; not the proposed method.",
            }

        return timed_solve(self.name, system, _solve)
