from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from robust_qsvt_se.estimators.base import (
    EstimatorResult,
    make_failure_result,
    make_success_result,
)
from robust_qsvt_se.measurement.linear_system import WeightedSystem


class HuberIRLSEstimator:
    name = "huber_irls"

    def __init__(
        self,
        *,
        delta: float,
        max_iterations: int = 50,
        tolerance: float = 1.0e-8,
    ) -> None:
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.delta = float(delta)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        start = perf_counter()
        try:
            x_hat, diagnostics, converged = self._solve_irls(system)
        except Exception as exc:
            return make_failure_result(
                name=self.name,
                system=system,
                runtime_seconds=perf_counter() - start,
                failure_reason=str(exc),
            )

        result = make_success_result(
            name=self.name,
            system=system,
            x_hat=x_hat,
            runtime_seconds=perf_counter() - start,
            extra_diagnostics=diagnostics,
        )
        result.converged = converged
        return result

    def _solve_irls(self, system: WeightedSystem) -> tuple[np.ndarray, dict[str, Any], bool]:
        x_current = _least_squares(system.H_tilde, system.r_tilde)
        weights = np.ones(system.n_measurements, dtype=np.float64)
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            residual = system.H_tilde @ x_current - system.r_tilde
            weights = _huber_weights(residual, self.delta)
            sqrt_weights = np.sqrt(weights)
            x_next = _least_squares(
                system.H_tilde * sqrt_weights[:, None],
                system.r_tilde * sqrt_weights,
            )
            step_norm = float(np.linalg.norm(x_next - x_current))
            reference_norm = 1.0 + float(np.linalg.norm(x_current))
            x_current = x_next
            iterations = iteration
            if step_norm <= self.tolerance * reference_norm:
                converged = True
                break

        residual = system.H_tilde @ x_current - system.r_tilde
        weights = _huber_weights(residual, self.delta)
        diagnostics: dict[str, Any] = {
            "delta": self.delta,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "iterations": iterations,
            "irls_converged": converged,
            "robust_objective": _huber_objective(residual, self.delta),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
        }
        return x_current, diagnostics, converged


def _least_squares(H_tilde: np.ndarray, r_tilde: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(H_tilde, r_tilde, rcond=None)
    return np.asarray(solution, dtype=np.float64)


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_residual = np.abs(residual)
    weights = np.ones_like(abs_residual, dtype=np.float64)
    mask = abs_residual > delta
    weights[mask] = delta / abs_residual[mask]
    return weights


def _huber_objective(residual: np.ndarray, delta: float) -> float:
    abs_residual = np.abs(residual)
    quadratic = abs_residual <= delta
    values = np.empty_like(abs_residual, dtype=np.float64)
    values[quadratic] = 0.5 * abs_residual[quadratic] ** 2
    values[~quadratic] = delta * (abs_residual[~quadratic] - 0.5 * delta)
    return float(np.sum(values))
