from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.research_matrix import (
    DEFAULT_LINEARIZATION_CONFIG,
    DEFAULT_MEASUREMENT_CONFIG,
)
from robust_qsvt_se.qsvt.resources import estimate_qsvt_resources
from robust_qsvt_se.utils.seed import make_rng

DEFAULT_ALPHA_GRID = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
DEFAULT_DEGREES = [5, 11, 21, 35, 51, 71]
DEFAULT_EPSILON = 1.0e-2
RESOURCE_CAVEAT = (
    "These resource estimates support feasibility discussion only. They do not "
    "demonstrate quantum speedup or full IEEE-scale hardware execution."
)


def build_engineering_system(config: dict[str, Any] | None = None) -> tuple[WeightedSystem, str]:
    resolved = dict(config or {})
    matrix_source = str(resolved.get("matrix_source", "ieee14_ac_weighted_jacobian"))
    if matrix_source == "synthetic":
        return _synthetic_system(resolved), "synthetic_weighted_matrix"

    seed = int(resolved.get("seed", 123))
    case_name = str(resolved.get("case_name", "ieee14"))
    case_source = str(resolved.get("case_source", "pypower"))
    measurement = {**DEFAULT_MEASUREMENT_CONFIG, **dict(resolved.get("measurement", {}))}
    linearization = {
        **DEFAULT_LINEARIZATION_CONFIG,
        **dict(resolved.get("linearization", {})),
    }
    system = build_ac_weighted_system(
        case_name=case_name,
        case_source=case_source,
        linearization_config=linearization,
        measurement_config=measurement,
        rng=make_rng(seed),
        metadata={"qsvt_engineering_extension": True},
    )
    return system, f"{case_name}_{case_source}_ac_weighted_jacobian"


def ridge_svd_solution(H_tilde: np.ndarray, r_tilde: np.ndarray, *, alpha: float) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(
        np.asarray(H_tilde, dtype=np.float64),
        full_matrices=False,
    )
    return Vt.T @ (ridge_filter(singular_values, alpha=alpha) * (U.T @ r_tilde))


def singular_summary(matrix: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    condition = float("inf") if positive.size == 0 else float(positive.max() / positive.min())
    return {
        "singular_values": singular_values,
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": condition,
    }


def matrix_density(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    return float(np.count_nonzero(np.abs(values) > tolerance) / values.size)


def select_rectangular_submatrix(
    matrix: np.ndarray,
    *,
    row_count: int,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if row_count <= 0 or column_count <= 0:
        raise ValueError("row_count and column_count must be positive")
    if row_count > values.shape[0] or column_count > values.shape[1]:
        raise ValueError("requested submatrix shape exceeds source matrix")
    column_indices = _top_indices(np.linalg.norm(values, axis=0), column_count)
    row_scores = np.linalg.norm(values[:, column_indices], axis=1)
    row_indices = _top_indices(row_scores, row_count)
    return values[np.ix_(row_indices, column_indices)], row_indices, column_indices


def direction_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(ref, cand) / (ref_norm * cand_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    relative_error = (
        float(np.linalg.norm(cand - ref))
        if ref_norm == 0.0
        else float(np.linalg.norm(cand - ref) / ref_norm)
    )
    return {
        "relative_error": relative_error,
        "cosine_similarity": cosine,
        "state_fidelity": cosine**2,
    }


def bounded_scaling_C(singular_values: np.ndarray, *, alpha: float) -> float:
    gains = ridge_filter(singular_values, alpha=alpha)
    return max(1.0, float(np.max(np.abs(gains))) if gains.size else 1.0)


def estimate_degree_and_queries(
    singular_values: np.ndarray,
    *,
    alpha: float,
    epsilon: float = DEFAULT_EPSILON,
    degrees: list[int] | None = None,
    grid_size: int = 1024,
) -> dict[str, float | int | None]:
    degree_grid = list(DEFAULT_DEGREES if degrees is None else degrees)
    estimates = estimate_qsvt_resources(
        singular_values,
        alpha=float(alpha),
        degrees=degree_grid,
        grid_size=max(int(grid_size), max(degree_grid) + 2),
        target_error=float(epsilon),
    )
    recommended = next(
        (estimate.recommended_degree for estimate in estimates if estimate.recommended_degree),
        None,
    )
    selected_degree = int(recommended if recommended is not None else max(degree_grid))
    selected = next(
        (estimate for estimate in estimates if estimate.degree == selected_degree),
        estimates[-1],
    )
    return {
        "qsvt_degree_estimate": selected_degree,
        "query_count_estimate": int(2 * selected_degree + 1),
        "max_polynomial_error": float(selected.max_error),
        "target_error": float(epsilon),
        "recommended_degree": recommended,
    }


def required_case_name(system: WeightedSystem) -> str:
    return str(system.metadata.get("case_name") or "synthetic")


def _synthetic_system(config: dict[str, Any]) -> WeightedSystem:
    matrix = np.asarray(
        config.get(
            "H_tilde",
            [
                [3.0, 0.1, 0.0],
                [0.0, 1.0, 0.2],
                [0.0, 0.0, 0.02],
                [1.0, -0.1, 0.0],
                [0.0, 0.3, 0.01],
            ],
        ),
        dtype=np.float64,
    )
    x_true = np.asarray(config.get("x_true", [0.2, -0.1, 0.05]), dtype=np.float64)
    r_tilde = np.asarray(config.get("r_tilde", matrix @ x_true), dtype=np.float64)
    return WeightedSystem(
        H_tilde=matrix,
        r_tilde=r_tilde,
        x_true=x_true,
        metadata={
            "case_name": "synthetic",
            "dataset_source": "generated",
            "dataset_source_detail": "deterministic small weighted matrix",
            "mode": "synthetic_weighted_linear_system",
        },
    )


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:count])
