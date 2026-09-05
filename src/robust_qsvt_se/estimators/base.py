from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.measurement.linear_system import WeightedSystem

FloatArray = NDArray[np.float64]


class Estimator(Protocol):
    name: str

    def solve(self, system: WeightedSystem) -> EstimatorResult: ...


@dataclass(slots=True)
class EstimatorResult:
    name: str
    x_hat: FloatArray
    residual_norm: float
    weighted_residual: float
    weighted_residual_norm: float
    weighted_residual_quadratic: float
    rmse: float | None
    condition_number: float
    rank: int
    effective_rank: int
    singular_values: FloatArray
    converged: bool
    failed: bool
    failure_reason: str | None
    runtime_seconds: float
    extra_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x_hat": self.x_hat.tolist(),
            "residual_norm": self.residual_norm,
            "weighted_residual": self.weighted_residual,
            "weighted_residual_norm": self.weighted_residual_norm,
            "weighted_residual_quadratic": self.weighted_residual_quadratic,
            "rmse": self.rmse,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "effective_rank": self.effective_rank,
            "singular_values": self.singular_values.tolist(),
            "converged": self.converged,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "runtime_seconds": self.runtime_seconds,
            "extra_diagnostics": self.extra_diagnostics,
        }


def _common_diagnostics(system: WeightedSystem) -> tuple[FloatArray, float, int, int]:
    singular_values = system.singular_values()
    return (
        singular_values,
        system.condition_number(),
        system.rank(),
        system.effective_rank(),
    )


def make_success_result(
    *,
    name: str,
    system: WeightedSystem,
    x_hat: FloatArray,
    runtime_seconds: float,
    extra_diagnostics: dict[str, Any] | None = None,
) -> EstimatorResult:
    singular_values, condition_number, rank, effective_rank = _common_diagnostics(system)
    weighted_residual_norm = system.weighted_residual_norm(x_hat)
    return EstimatorResult(
        name=name,
        x_hat=np.asarray(x_hat, dtype=np.float64),
        residual_norm=system.residual_norm(x_hat),
        weighted_residual=weighted_residual_norm,
        weighted_residual_norm=weighted_residual_norm,
        weighted_residual_quadratic=weighted_residual_norm**2,
        rmse=system.rmse(x_hat),
        condition_number=condition_number,
        rank=rank,
        effective_rank=effective_rank,
        singular_values=singular_values,
        converged=True,
        failed=False,
        failure_reason=None,
        runtime_seconds=runtime_seconds,
        extra_diagnostics=extra_diagnostics or {},
    )


def make_failure_result(
    *,
    name: str,
    system: WeightedSystem,
    runtime_seconds: float,
    failure_reason: str,
    extra_diagnostics: dict[str, Any] | None = None,
) -> EstimatorResult:
    singular_values, condition_number, rank, effective_rank = _common_diagnostics(system)
    x_hat = np.full(system.n_states, np.nan, dtype=np.float64)
    return EstimatorResult(
        name=name,
        x_hat=x_hat,
        residual_norm=float("nan"),
        weighted_residual=float("nan"),
        weighted_residual_norm=float("nan"),
        weighted_residual_quadratic=float("nan"),
        rmse=None,
        condition_number=condition_number,
        rank=rank,
        effective_rank=effective_rank,
        singular_values=singular_values,
        converged=False,
        failed=True,
        failure_reason=failure_reason,
        runtime_seconds=runtime_seconds,
        extra_diagnostics=extra_diagnostics or {},
    )


def svd_filtered_solution(system: WeightedSystem, filter_values: FloatArray) -> FloatArray:
    U, singular_values, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
    if filter_values.shape != singular_values.shape:
        raise ValueError(
            "filter_values must match singular value shape: "
            f"{filter_values.shape} != {singular_values.shape}"
        )
    return Vt.T @ (filter_values * (U.T @ system.r_tilde))


def timed_solve(name: str, system: WeightedSystem, solve_fn: Any) -> EstimatorResult:
    start = perf_counter()
    try:
        x_hat, extra_diagnostics = solve_fn()
    except Exception as exc:
        return make_failure_result(
            name=name,
            system=system,
            runtime_seconds=perf_counter() - start,
            failure_reason=str(exc),
        )
    return make_success_result(
        name=name,
        system=system,
        x_hat=x_hat,
        runtime_seconds=perf_counter() - start,
        extra_diagnostics=extra_diagnostics,
    )
