from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb
from scipy.sparse.linalg import LinearOperator, cg

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    build_engineering_system,
    direction_metrics,
    ridge_svd_solution,
)
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.partial_observable_readout import (
    basis_probability,
    normalize_state,
    subset_probability,
)
from robust_qsvt_se.qsvt.sparse_access_oracle import (
    SparseAccessOracle,
    build_sparse_access_oracle,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

MATRIX_FREE_LIMITATION = (
    "This is a matrix-free polynomial-action proxy for the QSVT target filter, "
    "not a full dense QSVT circuit simulation."
)


@dataclass(frozen=True, slots=True)
class ExactFilterActionResult:
    update_vector: np.ndarray
    update_norm: float
    residual_norm: float
    singular_values: np.ndarray
    filter_values: np.ndarray
    runtime_seconds: float

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "update_norm": self.update_norm,
            "residual_norm": self.residual_norm,
            "sigma_max": float(self.singular_values[0]) if self.singular_values.size else 0.0,
            "sigma_min": float(self.singular_values[-1]) if self.singular_values.size else 0.0,
            "runtime_seconds": self.runtime_seconds,
        }
        if extra:
            row.update(extra)
        return row


@dataclass(frozen=True, slots=True)
class MatrixFreeQSVTActionResult:
    update_vector: np.ndarray
    normalized_update_state: np.ndarray
    update_norm: float
    residual_norm: float
    matrix_shape: tuple[int, int]
    alpha: float
    degree: int
    method: str
    matvec_calls: int
    rmatvec_calls: int
    estimated_qsvt_query_count: int
    success_probability_proxy: float
    bounded_filter_scaling_C: float
    error_vs_exact_svd: float | None
    relative_error_vs_exact_svd: float | None
    error_vs_ridge: float | None
    relative_error_vs_ridge: float | None
    runtime_seconds: float
    memory_estimate_bytes: int
    implemented_or_estimated: str
    limitation: str = MATRIX_FREE_LIMITATION

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "matrix_rows": self.matrix_shape[0],
            "matrix_cols": self.matrix_shape[1],
            "alpha": self.alpha,
            "degree": self.degree,
            "polynomial_method": self.method,
            "sparse_matvec_calls": self.matvec_calls,
            "sparse_rmatvec_calls": self.rmatvec_calls,
            "estimated_qsvt_query_count": self.estimated_qsvt_query_count,
            "residual_norm": self.residual_norm,
            "matrix_free_update_norm": self.update_norm,
            "success_probability_proxy": self.success_probability_proxy,
            "bounded_filter_scaling_C": self.bounded_filter_scaling_C,
            "error_vs_exact_svd": self.error_vs_exact_svd,
            "relative_error_vs_exact_svd": self.relative_error_vs_exact_svd,
            "error_vs_ridge": self.error_vs_ridge,
            "relative_error_vs_ridge": self.relative_error_vs_ridge,
            "runtime_seconds": self.runtime_seconds,
            "memory_estimate_bytes": self.memory_estimate_bytes,
            "implemented_or_estimated": self.implemented_or_estimated,
            "limitation": self.limitation,
        }
        if extra:
            row.update(extra)
        return row


def run_exact_svd_filter_action(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    alpha: float,
) -> ExactFilterActionResult:
    """Apply the exact Ridge/Tikhonov singular-value filter by SVD."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    matrix = np.asarray(H_tilde, dtype=np.float64)
    residual = np.asarray(r_tilde, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("H_tilde must be two-dimensional")
    if residual.shape != (matrix.shape[0],):
        raise ValueError(f"r_tilde must have shape {(matrix.shape[0],)}")
    start = time.perf_counter()
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    filter_values = ridge_filter(singular_values, alpha=float(alpha))
    update = Vt.T @ (filter_values * (U.T @ residual))
    runtime = time.perf_counter() - start
    return ExactFilterActionResult(
        update_vector=np.real(update),
        update_norm=float(np.linalg.norm(update)),
        residual_norm=float(np.linalg.norm(residual)),
        singular_values=singular_values,
        filter_values=filter_values,
        runtime_seconds=float(runtime),
    )


def run_matrix_free_polynomial_filter_action(
    oracle: SparseAccessOracle,
    r_tilde: np.ndarray,
    alpha: float,
    degree: int,
    method: str = "chebyshev",
    seed: int = 123,
) -> MatrixFreeQSVTActionResult:
    """Apply a polynomial proxy for the QSVT Ridge filter using sparse access.

    The implementation approximates ``(H^T H + alpha I)^-1`` by a Chebyshev
    polynomial and applies it to ``H^T r`` with only sparse matvec/rmatvec calls.
    """

    _ = int(seed)
    if method not in {"chebyshev", "cg_krylov"}:
        raise ValueError("method must be 'chebyshev' or 'cg_krylov'")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if degree <= 0:
        raise ValueError("degree must be positive")
    residual = np.asarray(r_tilde, dtype=np.float64)
    if residual.shape != (oracle.shape[0],):
        raise ValueError(f"r_tilde must have shape {(oracle.shape[0],)}")

    start = time.perf_counter()
    counter = _SparseCallCounter(oracle)
    beta = max(float(oracle.normalization_beta), np.finfo(float).eps)
    lambda_max = beta**2
    rhs = counter.rmatvec(residual)
    if method == "chebyshev":
        coefficients = _chebyshev_inverse_coefficients(
            alpha=float(alpha),
            lambda_max=lambda_max,
            degree=int(degree),
        )
        update = _apply_chebyshev_polynomial(
            rhs,
            coefficients=coefficients,
            lambda_max=lambda_max,
            normal_operator=counter.normal_matvec,
        )
    else:
        update = _apply_cg_krylov_polynomial(
            rhs,
            alpha=float(alpha),
            degree=int(degree),
            normal_operator=counter.normal_matvec,
        )
    runtime = time.perf_counter() - start
    exact = _exact_reference_if_available(oracle, residual, alpha=float(alpha))
    error = None
    relative_error = None
    if exact is not None:
        error = float(np.linalg.norm(update - exact.update_vector))
        relative_error = error / max(float(np.linalg.norm(exact.update_vector)), 1.0e-15)
    state, update_norm = normalize_state(update)
    residual_norm = float(np.linalg.norm(residual))
    C = _bounded_scaling_constant(beta=beta, alpha=float(alpha))
    success = 0.0
    if residual_norm > 0.0 and C > 0.0:
        success = float(min(1.0, (update_norm / (C * residual_norm)) ** 2))
    memory = int(
        oracle.data.nbytes
        + oracle.row_ptr.nbytes
        + oracle.col_ind.nbytes
        + 4 * max(oracle.shape) * np.dtype(np.float64).itemsize
    )
    return MatrixFreeQSVTActionResult(
        update_vector=np.real(update),
        normalized_update_state=state,
        update_norm=float(update_norm),
        residual_norm=residual_norm,
        matrix_shape=oracle.shape,
        alpha=float(alpha),
        degree=int(degree),
        method=method,
        matvec_calls=int(counter.matvec_calls),
        rmatvec_calls=int(counter.rmatvec_calls),
        estimated_qsvt_query_count=int(2 * int(degree) + 1),
        success_probability_proxy=success,
        bounded_filter_scaling_C=float(C),
        error_vs_exact_svd=error,
        relative_error_vs_exact_svd=relative_error,
        error_vs_ridge=error,
        relative_error_vs_ridge=relative_error,
        runtime_seconds=float(runtime),
        memory_estimate_bytes=memory,
        implemented_or_estimated="matrix_free_polynomial_action_proxy",
    )


def build_matrix_free_action_outputs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_matrix_free_action",
        "case": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "alpha": 1.0e-4,
        "degree": 51,
        "method": "cg_krylov",
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(
        {
            "case_name": resolved["case"],
            "case_source": resolved["case_source"],
            "matrix_source": resolved["matrix_source"],
            "seed": int(resolved["seed"]),
        }
    )
    oracle = build_sparse_access_oracle(system.H_tilde)
    result = run_matrix_free_polynomial_filter_action(
        oracle,
        system.r_tilde,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        method=str(resolved["method"]),
        seed=int(resolved["seed"]),
    )
    row = result.to_row(
        {
            "case": resolved["case"],
            "matrix_source": matrix_source,
            "condition_number": system.condition_number(),
        }
    )
    summary_csv = output_dir / "matrix_free_action_summary.csv"
    diagnostics_json = output_dir / "matrix_free_action_diagnostics.json"
    limitations_md = output_dir / "matrix_free_action_limitations.md"
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    write_json(
        diagnostics_json,
        {
            "summary": row,
            "oracle": oracle.to_summary_row(),
            "limitation": MATRIX_FREE_LIMITATION,
        },
    )
    limitations_md.write_text(
        _limitations_markdown({"resource_estimate_only": False}),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "matrix_free_action_summary": str(summary_csv),
            "matrix_free_action_diagnostics": str(diagnostics_json),
            "matrix_free_action_limitations": str(limitations_md),
        },
        input_config=resolved,
        claim_boundary=MATRIX_FREE_LIMITATION,
    )
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame([row]),
        "artifacts": {
            "manifest": manifest,
            "matrix_free_action_summary": summary_csv,
            "matrix_free_action_diagnostics": diagnostics_json,
            "matrix_free_action_limitations": limitations_md,
        },
    }


def run_matrix_free_ieee_experiments(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_matrix_free_ieee_experiments",
        "cases": ["ieee14", "ieee30", "ieee57"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "alphas": [1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "method": "cg_krylov",
        "seed": 123,
        "resource_estimate_only": False,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    observable_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for case in list(resolved["cases"]):
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": resolved["matrix_source"],
                "seed": int(resolved["seed"]),
            }
        )
        oracle = build_sparse_access_oracle(system.H_tilde)
        condition = system.condition_number()
        for alpha in [float(value) for value in resolved["alphas"]]:
            ridge_reference: np.ndarray | None = None
            if not bool(resolved["resource_estimate_only"]):
                ridge_reference = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=alpha)
            for degree in [int(value) for value in resolved["degrees"]]:
                if bool(resolved["resource_estimate_only"]):
                    row = _resource_only_matrix_free_row(
                        case=case,
                        matrix_source=matrix_source,
                        oracle=oracle,
                        alpha=alpha,
                        degree=degree,
                        method=str(resolved["method"]),
                        condition_number=condition,
                    )
                    summary_rows.append(row)
                    runtime_rows.append(
                        {
                            "case": case,
                            "alpha": alpha,
                            "degree": degree,
                            "runtime_seconds": 0.0,
                            "implemented_or_estimated": "resource_estimate_only",
                        }
                    )
                    continue
                result = run_matrix_free_polynomial_filter_action(
                    oracle,
                    system.r_tilde,
                    alpha=alpha,
                    degree=degree,
                    method=str(resolved["method"]),
                    seed=int(resolved["seed"]),
                )
                row = result.to_row(
                    {
                        "case": case,
                        "matrix_source": matrix_source,
                        "condition_number": condition,
                        "ridge_reference_norm": None
                        if ridge_reference is None
                        else float(np.linalg.norm(ridge_reference)),
                        "residual_norm_after_update": system.residual_norm(result.update_vector),
                    }
                )
                summary_rows.append(row)
                observable_rows.extend(
                    _observable_rows(
                        case=case,
                        alpha=alpha,
                        degree=degree,
                        result=result,
                        reference=ridge_reference,
                    )
                )
                runtime_rows.append(
                    {
                        "case": case,
                        "alpha": alpha,
                        "degree": degree,
                        "runtime_seconds": result.runtime_seconds,
                        "sparse_matvec_calls": result.matvec_calls,
                        "sparse_rmatvec_calls": result.rmatvec_calls,
                        "memory_estimate_bytes": result.memory_estimate_bytes,
                        "implemented_or_estimated": result.implemented_or_estimated,
                    }
                )
        diagnostics[case] = {
            "shape": list(oracle.shape),
            "nnz": oracle.nnz,
            "max_row_sparsity": oracle.max_row_sparsity,
            "max_col_sparsity": oracle.max_col_sparsity,
            "condition_number": condition,
            "resource_estimate_only": bool(resolved["resource_estimate_only"]),
        }
    summary_csv = output_dir / "matrix_free_ieee_summary.csv"
    observable_csv = output_dir / "matrix_free_observable_summary.csv"
    runtime_csv = output_dir / "matrix_free_runtime_summary.csv"
    limitations_md = output_dir / "matrix_free_limitations.md"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(observable_rows).to_csv(observable_csv, index=False)
    pd.DataFrame(runtime_rows).to_csv(runtime_csv, index=False)
    limitations_md.write_text(_limitations_markdown(resolved), encoding="utf-8")
    write_json(output_dir / "matrix_free_action_diagnostics.json", diagnostics)
    manifest = write_manifest(
        output_dir,
        artifacts={
            "matrix_free_ieee_summary": str(summary_csv),
            "matrix_free_observable_summary": str(observable_csv),
            "matrix_free_runtime_summary": str(runtime_csv),
            "matrix_free_limitations": str(limitations_md),
        },
        input_config=resolved,
        claim_boundary=MATRIX_FREE_LIMITATION,
    )
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame(summary_rows),
        "artifacts": {
            "manifest": manifest,
            "matrix_free_ieee_summary": summary_csv,
            "matrix_free_observable_summary": observable_csv,
            "matrix_free_runtime_summary": runtime_csv,
            "matrix_free_limitations": limitations_md,
        },
    }


class _SparseCallCounter:
    def __init__(self, oracle: SparseAccessOracle) -> None:
        self.oracle = oracle
        self.matvec_calls = 0
        self.rmatvec_calls = 0

    def matvec(self, x: np.ndarray) -> np.ndarray:
        self.matvec_calls += 1
        return self.oracle.matvec(x)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        self.rmatvec_calls += 1
        return self.oracle.rmatvec(y)

    def normal_matvec(self, x: np.ndarray) -> np.ndarray:
        return self.rmatvec(self.matvec(x))


def _chebyshev_inverse_coefficients(
    *,
    alpha: float,
    lambda_max: float,
    degree: int,
) -> np.ndarray:
    grid_size = max(512, 8 * (int(degree) + 1))
    points = np.cos(np.pi * (np.arange(grid_size) + 0.5) / grid_size)
    lambdas = 0.5 * float(lambda_max) * (points + 1.0)
    values = 1.0 / (lambdas + float(alpha))
    return np.asarray(cheb.chebfit(points, values, deg=int(degree)), dtype=np.float64)


def _apply_chebyshev_polynomial(
    vector: np.ndarray,
    *,
    coefficients: np.ndarray,
    lambda_max: float,
    normal_operator: Any,
) -> np.ndarray:
    if coefficients.size == 0:
        return np.zeros_like(vector)
    if lambda_max <= 0.0:
        return coefficients[0] * vector

    def scaled_operator(x: np.ndarray) -> np.ndarray:
        return (2.0 * normal_operator(x) / float(lambda_max)) - x

    t_previous = np.asarray(vector, dtype=np.float64)
    result = coefficients[0] * t_previous
    if coefficients.size == 1:
        return result
    t_current = scaled_operator(t_previous)
    result = result + coefficients[1] * t_current
    for index in range(2, coefficients.size):
        t_next = 2.0 * scaled_operator(t_current) - t_previous
        result = result + coefficients[index] * t_next
        t_previous, t_current = t_current, t_next
    return np.asarray(result, dtype=np.float64)


def _apply_cg_krylov_polynomial(
    vector: np.ndarray,
    *,
    alpha: float,
    degree: int,
    normal_operator: Any,
) -> np.ndarray:
    size = int(vector.size)

    def matvec(x: np.ndarray) -> np.ndarray:
        return normal_operator(x) + float(alpha) * x

    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    solution, _info = cg(
        operator,
        np.asarray(vector, dtype=np.float64),
        maxiter=int(degree),
        rtol=1.0e-10,
        atol=0.0,
    )
    return np.asarray(solution, dtype=np.float64)


def _bounded_scaling_constant(*, beta: float, alpha: float) -> float:
    gain = 1.0 / (2.0 * np.sqrt(alpha)) if np.sqrt(alpha) <= beta else beta / (beta**2 + alpha)
    return max(1.0, float(gain))


def _exact_reference_if_available(
    oracle: SparseAccessOracle,
    residual: np.ndarray,
    *,
    alpha: float,
    max_dimension: int = 400,
) -> ExactFilterActionResult | None:
    dense = oracle.to_dense_if_small(max_dimension=max_dimension)
    if dense is None:
        return None
    return run_exact_svd_filter_action(dense, residual, alpha)


def _resource_only_matrix_free_row(
    *,
    case: str,
    matrix_source: str,
    oracle: SparseAccessOracle,
    alpha: float,
    degree: int,
    method: str,
    condition_number: float,
) -> dict[str, Any]:
    return {
        "case": case,
        "matrix_source": matrix_source,
        "matrix_rows": oracle.shape[0],
        "matrix_cols": oracle.shape[1],
        "alpha": alpha,
        "degree": degree,
        "condition_number": condition_number,
        "polynomial_method": method,
        "sparse_matvec_calls": 0,
        "sparse_rmatvec_calls": 0,
        "estimated_qsvt_query_count": 2 * int(degree) + 1,
        "residual_norm": None,
        "matrix_free_update_norm": None,
        "ridge_reference_norm": None,
        "relative_error_vs_exact_svd": None,
        "relative_error_vs_ridge": None,
        "runtime_seconds": 0.0,
        "memory_estimate_bytes": int(
            oracle.data.nbytes + oracle.row_ptr.nbytes + oracle.col_ind.nbytes
        ),
        "success_probability_proxy": None,
        "implemented_or_estimated": "resource_estimate_only",
        "limitation": "resource estimate only; matrix-free action not executed",
    }


def _observable_rows(
    *,
    case: str,
    alpha: float,
    degree: int,
    result: MatrixFreeQSVTActionResult,
    reference: np.ndarray | None,
) -> list[dict[str, Any]]:
    if reference is None:
        return []
    qsvt_state = np.real(result.normalized_update_state)
    reference_state, _ = normalize_state(reference)
    reference_state = np.real(reference_state)
    rows = []
    observables = [
        ("component_0_probability", "basis_probability", [0]),
        ("component_1_probability", "basis_probability", [1] if qsvt_state.size > 1 else [0]),
        (
            "first_two_state_energy",
            "subset_probability",
            list(range(min(2, qsvt_state.size))),
        ),
    ]
    for name, kind, indices in observables:
        if kind == "basis_probability":
            q_value = basis_probability(qsvt_state, indices[0])
            r_value = basis_probability(reference_state, indices[0])
        else:
            q_value = subset_probability(qsvt_state, indices)
            r_value = subset_probability(reference_state, indices)
        rows.append(
            {
                "case": case,
                "alpha": alpha,
                "degree": degree,
                "observable_name": name,
                "observable_type": kind,
                "indices": " ".join(str(index) for index in indices),
                "matrix_free_value": q_value,
                "ridge_reference_value": r_value,
                "absolute_error": abs(q_value - r_value),
                "limitation": MATRIX_FREE_LIMITATION,
            }
        )
    metrics = direction_metrics(reference_state, qsvt_state)
    rows.append(
        {
            "case": case,
            "alpha": alpha,
            "degree": degree,
            "observable_name": "normalized_state_direction",
            "observable_type": "direction_metric",
            "indices": "all",
            "matrix_free_value": metrics["cosine_similarity"],
            "ridge_reference_value": 1.0,
            "absolute_error": abs(1.0 - metrics["cosine_similarity"]),
            "limitation": MATRIX_FREE_LIMITATION,
        }
    )
    return rows


def _limitations_markdown(config: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Matrix-Free QSVT Action Limitations",
            "",
            MATRIX_FREE_LIMITATION,
            "",
            "- The action uses sparse matvec and rmatvec calls, not a dense "
            "block-encoding unitary.",
            "- Chebyshev or Krylov degree is a polynomial-action proxy for the QSVT target filter.",
            "- Exact SVD/Ridge references are diagnostics for feasible cases.",
            "- IEEE118/IEEE300 can be run in resource-estimate-only mode.",
            f"- Resource-estimate-only mode: {bool(config['resource_estimate_only'])}",
            "",
        ]
    )
