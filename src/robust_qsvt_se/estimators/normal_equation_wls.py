from __future__ import annotations

import math

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, make_failure_result, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem


class NormalEquationWLSEstimator:
    name = "normal_equation_wls"

    def __init__(self, max_gain_condition_number: float = math.inf) -> None:
        if max_gain_condition_number <= 0.0:
            raise ValueError("max_gain_condition_number must be positive")
        self.max_gain_condition_number = float(max_gain_condition_number)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, float | str]]:
            gain = system.H_tilde.T @ system.H_tilde
            right_hand_side = system.H_tilde.T @ system.r_tilde
            gain_condition = float(np.linalg.cond(gain))
            diagnostics = {
                "normal_matrix_condition_number": gain_condition,
                "H_tilde_condition_number": system.condition_number(),
                "solver": "numpy.linalg.solve on H_tilde.T @ H_tilde",
                "normal_equation_form": "G = H^T R^{-1} H = H_tilde.T @ H_tilde",
            }
            if not np.isfinite(gain_condition):
                raise np.linalg.LinAlgError("normal-equation gain matrix is singular")
            if gain_condition > self.max_gain_condition_number:
                raise np.linalg.LinAlgError(
                    "normal-equation gain matrix condition number "
                    f"{gain_condition:.3e} exceeds max_gain_condition_number="
                    f"{self.max_gain_condition_number:.3e}"
                )
            x_hat = np.linalg.solve(gain, right_hand_side)
            if not np.all(np.isfinite(x_hat)):
                raise np.linalg.LinAlgError("normal-equation solve produced nonfinite values")
            return x_hat, diagnostics

        result = timed_solve(self.name, system, _solve)
        if result.failed and "normal_matrix_condition_number" not in result.extra_diagnostics:
            gain = system.H_tilde.T @ system.H_tilde
            result.extra_diagnostics.update(
                {
                    "normal_matrix_condition_number": float(np.linalg.cond(gain)),
                    "H_tilde_condition_number": system.condition_number(),
                    "solver": "numpy.linalg.solve on H_tilde.T @ H_tilde",
                    "normal_equation_form": "G = H^T R^{-1} H = H_tilde.T @ H_tilde",
                }
            )
        return result


def make_normal_equation_failure(
    *,
    system: WeightedSystem,
    runtime_seconds: float,
    failure_reason: str,
    normal_matrix_condition_number: float,
) -> EstimatorResult:
    return make_failure_result(
        name=NormalEquationWLSEstimator.name,
        system=system,
        runtime_seconds=runtime_seconds,
        failure_reason=failure_reason,
        extra_diagnostics={
            "normal_matrix_condition_number": normal_matrix_condition_number,
            "H_tilde_condition_number": system.condition_number(),
        },
    )
