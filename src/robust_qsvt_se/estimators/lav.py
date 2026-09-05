from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import linprog

from robust_qsvt_se.estimators.base import (
    EstimatorResult,
    make_failure_result,
    make_success_result,
)
from robust_qsvt_se.measurement.linear_system import WeightedSystem


class LAVEstimator:
    name = "lav"

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        start = perf_counter()
        m_measurements = system.n_measurements
        n_states = system.n_states
        objective = np.concatenate(
            [np.zeros(n_states, dtype=np.float64), np.ones(m_measurements, dtype=np.float64)]
        )
        identity = np.eye(m_measurements, dtype=np.float64)
        constraints = np.vstack(
            [
                np.hstack([system.H_tilde, -identity]),
                np.hstack([-system.H_tilde, -identity]),
            ]
        )
        bounds_rhs = np.concatenate([system.r_tilde, -system.r_tilde])
        bounds = [(None, None)] * n_states + [(0.0, None)] * m_measurements

        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=bounds_rhs,
            bounds=bounds,
            method="highs",
        )
        diagnostics: dict[str, Any] = {
            "method": "scipy.optimize.linprog(highs)",
            "linprog_status": int(result.status),
            "linprog_message": str(result.message),
            "objective_value": float(result.fun) if result.fun is not None else None,
        }
        if not result.success:
            return make_failure_result(
                name=self.name,
                system=system,
                runtime_seconds=perf_counter() - start,
                failure_reason=str(result.message),
                extra_diagnostics=diagnostics,
            )

        x_hat = np.asarray(result.x[:n_states], dtype=np.float64)
        return make_success_result(
            name=self.name,
            system=system,
            x_hat=x_hat,
            runtime_seconds=perf_counter() - start,
            extra_diagnostics=diagnostics,
        )
