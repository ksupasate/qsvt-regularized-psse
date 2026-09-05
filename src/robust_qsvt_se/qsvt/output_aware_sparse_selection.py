"""Output-aware, resource-constrained sparse-support selection study.

This module implements the small-scale reproducibility campaign requested for the frozen
8x8 IEEE-14-derived workload.  It keeps local first-order sensitivity, exact Ridge loss,
and the conservative operator perturbation certificate as separate numerical objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    build_complete_wrapper_circuit,
    validate_complete_wrapper,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.tqe_revision_support_common import git_commit_hash, now_iso
from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    assign_slot_permutations,
    minimum_slot_count,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.ridge_output_certificate import (
    compute_ridge_selected_output_certificate,
    ridge_linearized_signed_error,
    ridge_selected_output_gradient,
    ridge_selected_output_via_solve,
    validate_ridge_selected_output_certificate,
)
from robust_qsvt_se.qsvt.sparse_error_precision_study import pareto_nondominated
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    _as_quantized_block,
    build_default_sparse_integrated_inputs,
    stable_array_fingerprint,
)

DEFAULT_OUTPUT_DIR = Path("outputs/output_aware_sparse_selection")
DEFAULT_CONFIG_PATH = Path("configs/output_aware_sparse_selection.json")
STUDY_ID = "output_aware_sparse_selection_v1"
BASELINE_CONFIGURATION_ID = "ieee14_sparse_quantized_8x8_d31_selected_v1"

FUNCTIONAL_IDS = (
    "coordinate_e0",
    "signed_difference_e0_minus_e1",
    "aggregate_e0_to_e3",
)
DETERMINISTIC_SELECTORS = (
    "global_magnitude",
    "balanced_magnitude",
    "leverage_weighted",
    "sensitivity_initial_mean",
    "sensitivity_initial_worst_case",
    "sensitivity_refined_mean",
    "sensitivity_refined_worst_case",
)
INITIAL_SELECTORS = DETERMINISTIC_SELECTORS[:5]
REFINED_SELECTORS = DETERMINISTIC_SELECTORS[5:]

STAGES = (
    "audit",
    "pareto-audit",
    "residuals",
    "gradients",
    "scores",
    "supports",
    "refinement",
    "certificates",
    "heldout",
    "resources",
    "qsvt",
    "finite-shot",
    "pareto",
    "verify",
)

FAILURE_CLASSES = (
    "support_budget_infeasible",
    "coverage_constraint_infeasible",
    "slot_assignment_failure",
    "optimization_failure",
    "certificate_violation",
    "common_normalization_failure",
    "polynomial_fit_failure",
    "qsvt_statevector_failure",
    "finite_shot_runtime_limit",
    "resource_compilation_limit",
    "other_verified_failure",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_study_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study configuration must contain a JSON object")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected study_id: {payload.get('study_id')}")
    training = [int(value) for value in payload["training_seed_ids"]]
    heldout = [int(value) for value in payload["held_out_seed_ids"]]
    if not training or not heldout or set(training).intersection(heldout):
        raise ValueError("training and held-out seed lists must be nonempty and disjoint")
    if list(payload.get("functionals", [])) != list(FUNCTIONAL_IDS):
        raise ValueError("the three predetermined functionals must remain frozen")
    return payload


@dataclass(slots=True)
class FrozenOutputAwareDesign:
    matrix: np.ndarray
    baseline_sparse_matrix: np.ndarray
    baseline_quantized_matrix: np.ndarray
    baseline_residual: np.ndarray
    functionals: dict[str, np.ndarray]
    alpha: float
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    source_metadata: dict[str, Any]


def build_frozen_output_aware_design(output_dir: str | Path) -> FrozenOutputAwareDesign:
    inputs = build_default_sparse_integrated_inputs(Path(output_dir))
    functionals = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in inputs.selected_functionals.items()
    }
    if tuple(functionals) != FUNCTIONAL_IDS:
        raise RuntimeError("predetermined functional ordering drifted")
    matrix = np.asarray(inputs.matrix_original, dtype=np.float64)
    if matrix.shape != (8, 8) or int(np.count_nonzero(matrix)) != 50:
        raise RuntimeError("frozen output-aware candidate matrix drifted")
    return FrozenOutputAwareDesign(
        matrix=matrix,
        baseline_sparse_matrix=np.asarray(inputs.matrix_sparsified, dtype=np.float64),
        baseline_quantized_matrix=np.asarray(inputs.matrix_quantized, dtype=np.float64),
        baseline_residual=np.asarray(inputs.residual, dtype=np.float64),
        functionals=functionals,
        alpha=float(inputs.config.alpha),
        selected_rows=tuple(int(value) for value in inputs.config.selected_rows),
        selected_columns=tuple(int(value) for value in inputs.config.selected_columns),
        source_metadata=dict(inputs.source_metadata),
    )


@dataclass(frozen=True, slots=True)
class ResidualSample:
    seed_id: int
    split: str
    residual: np.ndarray
    fingerprint: str
    status: str = "completed"
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class RidgeTask:
    task_id: str
    seed_id: int
    split: str
    residual: np.ndarray
    functional_id: str
    functional: np.ndarray


def generate_controlled_residual(
    seed_id: int,
    *,
    selected_rows: Sequence[int],
) -> np.ndarray:
    """Generate one weighted residual through the repository's controlled AC pathway."""

    system, _matrix_source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed_id),
        }
    )
    rows = np.asarray(selected_rows, dtype=np.int64)
    residual = np.asarray(system.r_tilde, dtype=np.float64)[rows]
    if residual.shape != (len(rows),) or not np.all(np.isfinite(residual)):
        raise RuntimeError("controlled residual generation returned invalid values")
    if np.linalg.norm(residual) <= 1.0e-15:
        raise RuntimeError("controlled residual generation returned a zero residual")
    return residual


def build_ridge_tasks(
    samples: Iterable[ResidualSample],
    functionals: dict[str, np.ndarray],
    *,
    required_split: str,
) -> list[RidgeTask]:
    tasks: list[RidgeTask] = []
    for sample in samples:
        if sample.status != "completed":
            continue
        if sample.split != required_split:
            raise ValueError(
                f"data leakage guard: expected {required_split}, received {sample.split}"
            )
        for functional_id in FUNCTIONAL_IDS:
            tasks.append(
                RidgeTask(
                    task_id=f"{required_split}_seed{sample.seed_id}_{functional_id}",
                    seed_id=int(sample.seed_id),
                    split=required_split,
                    residual=np.asarray(sample.residual, dtype=np.float64),
                    functional_id=functional_id,
                    functional=np.asarray(functionals[functional_id], dtype=np.float64),
                )
            )
    if not tasks:
        raise ValueError(f"no completed {required_split} residual tasks")
    return tasks


def _ridge_filter_operator(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Return the same SVD-defined Ridge operator used by ``ridge_svd_solution``."""

    values = np.asarray(matrix, dtype=np.float64)
    left, singular_values, right_t = np.linalg.svd(values, full_matrices=False)
    gains = singular_values / (singular_values**2 + float(alpha))
    return (right_t.T * gains) @ left.T


def exact_task_errors(
    matrix: np.ndarray,
    sparse_matrix: np.ndarray,
    tasks: Sequence[RidgeTask],
    *,
    alpha: float,
    y_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full outputs, sparse outputs, and stable normalized absolute errors."""

    full_operator = _ridge_filter_operator(matrix, alpha)
    sparse_operator = _ridge_filter_operator(sparse_matrix, alpha)
    full_outputs = np.asarray(
        [task.functional @ (full_operator @ task.residual) for task in tasks],
        dtype=np.float64,
    )
    sparse_outputs = np.asarray(
        [task.functional @ (sparse_operator @ task.residual) for task in tasks],
        dtype=np.float64,
    )
    normalized = np.abs(sparse_outputs - full_outputs) / np.maximum(
        np.abs(full_outputs), float(y_floor)
    )
    return full_outputs, sparse_outputs, normalized


def validate_ridge_gradient_entries(
    matrix: np.ndarray,
    tasks: Sequence[RidgeTask],
    *,
    alpha: float,
    relative_step: float,
    absolute_step_floor: float,
    entries_per_task: int,
    seed: int,
) -> pd.DataFrame:
    """Central-difference validation over deterministic small, large, and random entries."""

    values = np.asarray(matrix, dtype=np.float64)
    candidates = np.flatnonzero(values.ravel() != 0.0)
    if entries_per_task < 3 or candidates.size < entries_per_task:
        raise ValueError("entries_per_task must be at least three and fit the candidate set")
    magnitudes = np.abs(values.ravel()[candidates])
    by_small = candidates[np.argsort(magnitudes, kind="stable")]
    by_large = candidates[np.argsort(-magnitudes, kind="stable")]
    small_count = entries_per_task // 3
    large_count = entries_per_task // 3
    random_count = entries_per_task - small_count - large_count
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        rng = np.random.default_rng(int(seed) + task_index)
        preselected = list(by_small[:small_count]) + list(by_large[:large_count])
        remaining = np.asarray(
            [index for index in candidates if int(index) not in set(preselected)],
            dtype=np.int64,
        )
        random_entries = rng.choice(remaining, size=random_count, replace=False)
        selected = np.asarray([*preselected, *random_entries.tolist()], dtype=np.int64)
        analytic = ridge_selected_output_gradient(
            values, task.residual, task.functional, alpha
        ).ravel()[selected]
        finite_difference = np.empty(selected.size, dtype=np.float64)
        step_sizes = np.empty(selected.size, dtype=np.float64)
        categories = ["small"] * small_count + ["large"] * large_count + [
            "random"
        ] * random_count
        for local_index, flat_index in enumerate(selected):
            step = max(
                abs(float(values.ravel()[flat_index])) * float(relative_step),
                float(absolute_step_floor),
            )
            plus = values.copy()
            minus = values.copy()
            plus.ravel()[flat_index] += step
            minus.ravel()[flat_index] -= step
            y_plus = ridge_selected_output_via_solve(
                plus, task.residual, task.functional, alpha
            )
            y_minus = ridge_selected_output_via_solve(
                minus, task.residual, task.functional, alpha
            )
            finite_difference[local_index] = (y_plus - y_minus) / (2.0 * step)
            step_sizes[local_index] = step
        relative_frobenius = float(
            np.linalg.norm(analytic - finite_difference)
            / max(np.linalg.norm(finite_difference), np.finfo(float).eps)
        )
        for local_index, flat_index in enumerate(selected):
            row, column = np.unravel_index(int(flat_index), values.shape)
            denominator = max(abs(float(finite_difference[local_index])), 1.0e-15)
            rows.append(
                {
                    "task_id": task.task_id,
                    "seed_id": task.seed_id,
                    "split": task.split,
                    "functional_id": task.functional_id,
                    "entry_category": categories[local_index],
                    "row": int(row),
                    "column": int(column),
                    "matrix_value": float(values[row, column]),
                    "central_difference_step": float(step_sizes[local_index]),
                    "analytic_gradient": float(analytic[local_index]),
                    "finite_difference_gradient": float(finite_difference[local_index]),
                    "absolute_gradient_error": float(
                        abs(analytic[local_index] - finite_difference[local_index])
                    ),
                    "relative_entry_error": float(
                        abs(analytic[local_index] - finite_difference[local_index])
                        / denominator
                    ),
                    "epsilon_grad": relative_frobenius,
                    "gradient_semantics": "local_first_derivative_not_certificate",
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class EntryScoreBundle:
    sensitivity_mean: np.ndarray
    sensitivity_worst_case: np.ndarray
    per_functional_mean: dict[str, np.ndarray]
    per_functional_worst_case: dict[str, np.ndarray]
    normalized_task_scores: np.ndarray
    task_ids: tuple[str, ...]


def compute_output_aware_entry_scores(
    matrix: np.ndarray,
    tasks: Sequence[RidgeTask],
    *,
    alpha: float,
    epsilon: float,
) -> EntryScoreBundle:
    """Compute normalized first-order removal scores using training tasks only."""

    if not tasks or any(task.split != "training" for task in tasks):
        raise ValueError("data leakage guard: entry scores require training tasks only")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("score normalization epsilon must be positive and finite")
    values = np.asarray(matrix, dtype=np.float64)
    normalized_rows: list[np.ndarray] = []
    for task in tasks:
        gradient = ridge_selected_output_gradient(
            values, task.residual, task.functional, alpha
        )
        removal = np.abs(values * gradient)
        normalized_rows.append(removal / (float(np.sum(removal)) + float(epsilon)))
    normalized = np.stack(normalized_rows, axis=0)
    per_mean: dict[str, np.ndarray] = {}
    per_worst: dict[str, np.ndarray] = {}
    for functional_id in FUNCTIONAL_IDS:
        indices = [
            index for index, task in enumerate(tasks) if task.functional_id == functional_id
        ]
        per_mean[functional_id] = np.mean(normalized[indices], axis=0)
        per_worst[functional_id] = np.max(normalized[indices], axis=0)
    return EntryScoreBundle(
        sensitivity_mean=np.mean(normalized, axis=0),
        sensitivity_worst_case=np.max(normalized, axis=0),
        per_functional_mean=per_mean,
        per_functional_worst_case=per_worst,
        normalized_task_scores=normalized,
        task_ids=tuple(task.task_id for task in tasks),
    )


def entry_score_frame(matrix: np.ndarray, scores: EntryScoreBundle) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            per_functional = {
                functional_id: {
                    "mean": float(scores.per_functional_mean[functional_id][row, column]),
                    "worst_case": float(
                        scores.per_functional_worst_case[functional_id][row, column]
                    ),
                }
                for functional_id in FUNCTIONAL_IDS
            }
            rows.append(
                {
                    "row": row,
                    "column": column,
                    "matrix_value": float(values[row, column]),
                    "absolute_matrix_value": float(abs(values[row, column])),
                    "sensitivity_mean": float(scores.sensitivity_mean[row, column]),
                    "sensitivity_worst_case": float(
                        scores.sensitivity_worst_case[row, column]
                    ),
                    "per_functional_scores": json.dumps(
                        per_functional, sort_keys=True, separators=(",", ":")
                    ),
                    **{
                        f"{functional_id}_mean": float(
                            scores.per_functional_mean[functional_id][row, column]
                        )
                        for functional_id in FUNCTIONAL_IDS
                    },
                    **{
                        f"{functional_id}_worst_case": float(
                            scores.per_functional_worst_case[functional_id][row, column]
                        )
                        for functional_id in FUNCTIONAL_IDS
                    },
                    "candidate_status": (
                        "candidate_nonzero" if values[row, column] != 0.0 else "excluded_zero"
                    ),
                }
            )
    return pd.DataFrame(rows)


def deterministic_ridge_leverage_scores(
    matrix: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/column Ridge leverage and the documented entry score."""

    values = np.asarray(matrix, dtype=np.float64)
    system = values.T @ values + float(alpha) * np.eye(values.shape[1])
    solved_rows = np.linalg.solve(system, values.T)
    row_leverage = np.einsum("ij,ji->i", values, solved_rows)
    solved_gram = np.linalg.solve(system, values.T @ values)
    column_leverage = np.diag(solved_gram)
    row_leverage = np.maximum(row_leverage, 0.0)
    column_leverage = np.maximum(column_leverage, 0.0)
    entry = np.abs(values) * np.sqrt(
        row_leverage[:, np.newaxis] * column_leverage[np.newaxis, :]
    )
    return row_leverage, column_leverage, entry


@dataclass(frozen=True, slots=True)
class SupportConstraints:
    k_budget: int
    slot_budget: int
    coverage_enabled: bool = True


@dataclass(frozen=True, slots=True)
class SupportSelectionResult:
    support: np.ndarray | None
    objective_value: float | None
    solver_used: str
    solver_status: str
    optimality_gap: float | None
    runtime_seconds: float
    fallback_used: bool
    status: str
    failure_reason: str


def support_constraint_report(
    matrix: np.ndarray,
    support: np.ndarray,
    constraints: SupportConstraints,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    selected = np.asarray(support, dtype=bool)
    if selected.shape != values.shape:
        raise ValueError("support shape must match matrix shape")
    candidate = values != 0.0
    row_degrees = selected.sum(axis=1)
    column_degrees = selected.sum(axis=0)
    active_rows = candidate.any(axis=1)
    active_columns = candidate.any(axis=0)
    reasons: list[str] = []
    if np.any(selected & ~candidate):
        reasons.append("support_contains_non_candidate_zero")
    if int(selected.sum()) > int(constraints.k_budget):
        reasons.append("cardinality_exceeds_budget")
    if int(row_degrees.max(initial=0)) > int(constraints.slot_budget):
        reasons.append("row_degree_exceeds_slot_budget")
    if int(column_degrees.max(initial=0)) > int(constraints.slot_budget):
        reasons.append("column_degree_exceeds_slot_budget")
    if constraints.coverage_enabled:
        if np.any(row_degrees[active_rows] < 1):
            reasons.append("active_row_uncovered")
        if np.any(column_degrees[active_columns] < 1):
            reasons.append("active_column_uncovered")
    return {
        "valid": not reasons,
        "failure_reasons": reasons,
        "actual_nonzeros": int(selected.sum()),
        "row_degrees": [int(value) for value in row_degrees],
        "column_degrees": [int(value) for value in column_degrees],
        "actual_max_row_degree": int(row_degrees.max(initial=0)),
        "actual_max_column_degree": int(column_degrees.max(initial=0)),
        "active_rows_covered": bool(np.all(row_degrees[active_rows] >= 1)),
        "active_columns_covered": bool(np.all(column_degrees[active_columns] >= 1)),
    }


def _fallback_greedy_support(
    matrix: np.ndarray,
    scores: np.ndarray,
    constraints: SupportConstraints,
) -> np.ndarray:
    """Deterministic feasible fallback; intentionally not described as optimal."""

    from scipy.optimize import linear_sum_assignment

    values = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(scores, dtype=np.float64)
    candidate = values != 0.0
    active_rows = np.flatnonzero(candidate.any(axis=1))
    active_columns = np.flatnonzero(candidate.any(axis=0))
    if constraints.coverage_enabled and active_rows.size == active_columns.size:
        if constraints.k_budget < active_rows.size or constraints.slot_budget < 1:
            raise ValueError("coverage_constraint_infeasible")
        reduced = candidate[np.ix_(active_rows, active_columns)]
        scale = max(float(np.max(np.abs(weights))), 1.0)
        cost = np.where(
            reduced,
            -weights[np.ix_(active_rows, active_columns)],
            scale * 1.0e12,
        )
        row_index, column_index = linear_sum_assignment(cost)
        if np.any(~reduced[row_index, column_index]):
            raise ValueError("coverage_constraint_infeasible")
        support = np.zeros_like(candidate)
        support[active_rows[row_index], active_columns[column_index]] = True
    else:
        support = np.zeros_like(candidate)

    flat_candidates = np.flatnonzero(candidate.ravel())
    order = sorted(flat_candidates, key=lambda index: (-weights.ravel()[index], int(index)))
    for flat_index in order:
        if support.ravel()[flat_index] or int(support.sum()) >= constraints.k_budget:
            continue
        row, column = np.unravel_index(int(flat_index), values.shape)
        if support[row].sum() >= constraints.slot_budget:
            continue
        if support[:, column].sum() >= constraints.slot_budget:
            continue
        support[row, column] = True
    report = support_constraint_report(values, support, constraints)
    if not report["valid"]:
        raise ValueError("coverage_constraint_infeasible: " + ",".join(report["failure_reasons"]))
    return support


def select_resource_constrained_support(
    matrix: np.ndarray,
    scores: np.ndarray,
    constraints: SupportConstraints,
    *,
    time_limit_seconds: float = 30.0,
    relative_mip_gap: float = 0.0,
    tie_epsilon_relative: float = 1.0e-12,
    force_greedy_fallback: bool = False,
) -> SupportSelectionResult:
    """Maximize an entry score under cardinality, degree, and coverage constraints."""

    started = time.perf_counter()
    values = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or weights.shape != values.shape:
        raise ValueError("matrix and score arrays must be finite and have matching 2D shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("matrix and score arrays must be finite")
    if np.any(weights < 0.0):
        raise ValueError("support scores must be nonnegative")
    if constraints.k_budget <= 0 or constraints.slot_budget <= 0:
        raise ValueError("support and slot budgets must be positive")
    candidate_positions = np.argwhere(values != 0.0)
    candidate_count = len(candidate_positions)
    if candidate_count == 0:
        raise ValueError("matrix must contain candidate nonzeros")
    active_rows = np.flatnonzero(np.any(values != 0.0, axis=1))
    active_columns = np.flatnonzero(np.any(values != 0.0, axis=0))
    if constraints.coverage_enabled and constraints.k_budget < max(
        active_rows.size, active_columns.size
    ):
        return SupportSelectionResult(
            support=None,
            objective_value=None,
            solver_used="precheck",
            solver_status="infeasible",
            optimality_gap=None,
            runtime_seconds=time.perf_counter() - started,
            fallback_used=False,
            status="failed",
            failure_reason="coverage_constraint_infeasible: cardinality below coverage minimum",
        )

    if force_greedy_fallback:
        try:
            support = _fallback_greedy_support(values, weights, constraints)
            return SupportSelectionResult(
                support=support,
                objective_value=float(np.sum(weights[support])),
                solver_used="deterministic_constrained_greedy_fallback",
                solver_status="feasible_greedy",
                optimality_gap=None,
                runtime_seconds=time.perf_counter() - started,
                fallback_used=True,
                status="completed",
                failure_reason="",
            )
        except ValueError as exc:
            return SupportSelectionResult(
                support=None,
                objective_value=None,
                solver_used="deterministic_constrained_greedy_fallback",
                solver_status="infeasible",
                optimality_gap=None,
                runtime_seconds=time.perf_counter() - started,
                fallback_used=True,
                status="failed",
                failure_reason=str(exc),
            )

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        objective = np.asarray(
            [weights[row, column] for row, column in candidate_positions],
            dtype=np.float64,
        )
        scale = max(float(np.max(objective, initial=0.0)), 1.0)
        tie = scale * float(tie_epsilon_relative) * (
            candidate_count - np.arange(candidate_count, dtype=np.float64)
        ) / candidate_count
        c = -(objective + tie)
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        rows.append(np.ones(candidate_count, dtype=np.float64))
        lower.append(-np.inf)
        upper.append(float(constraints.k_budget))
        for row in range(values.shape[0]):
            indicator = (candidate_positions[:, 0] == row).astype(np.float64)
            rows.append(indicator)
            lower.append(1.0 if constraints.coverage_enabled and row in active_rows else -np.inf)
            upper.append(float(constraints.slot_budget))
        for column in range(values.shape[1]):
            indicator = (candidate_positions[:, 1] == column).astype(np.float64)
            rows.append(indicator)
            lower.append(
                1.0
                if constraints.coverage_enabled and column in active_columns
                else -np.inf
            )
            upper.append(float(constraints.slot_budget))
        result = milp(
            c=c,
            integrality=np.ones(candidate_count, dtype=np.int8),
            bounds=Bounds(np.zeros(candidate_count), np.ones(candidate_count)),
            constraints=LinearConstraint(
                np.vstack(rows), np.asarray(lower), np.asarray(upper)
            ),
            options={
                "time_limit": float(time_limit_seconds),
                "mip_rel_gap": float(relative_mip_gap),
                "presolve": True,
            },
        )
        status_names = {
            0: "optimal",
            1: "iteration_or_time_limit",
            2: "infeasible",
            3: "unbounded",
            4: "solver_error",
        }
        solver_status = status_names.get(int(result.status), f"status_{result.status}")
        if result.x is None or int(result.status) != 0:
            raise RuntimeError(f"MILP {solver_status}: {result.message}")
        support = np.zeros_like(values, dtype=bool)
        selected_variables = np.asarray(result.x) > 0.5
        for selected, (row, column) in zip(
            selected_variables, candidate_positions, strict=True
        ):
            if selected:
                support[row, column] = True
        report = support_constraint_report(values, support, constraints)
        if not report["valid"]:
            raise RuntimeError("MILP returned invalid support: " + ",".join(
                report["failure_reasons"]
            ))
        return SupportSelectionResult(
            support=support,
            objective_value=float(np.sum(weights[support])),
            solver_used="scipy.optimize.milp_highs",
            solver_status="optimal",
            optimality_gap=(
                float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else 0.0
            ),
            runtime_seconds=time.perf_counter() - started,
            fallback_used=False,
            status="completed",
            failure_reason="",
        )
    except (ImportError, AttributeError) as exc:
        fallback_reason = f"MILP unavailable: {type(exc).__name__}: {exc}"
    except RuntimeError as exc:
        fallback_reason = str(exc)

    try:
        support = _fallback_greedy_support(values, weights, constraints)
        return SupportSelectionResult(
            support=support,
            objective_value=float(np.sum(weights[support])),
            solver_used="deterministic_constrained_greedy_fallback",
            solver_status="feasible_greedy",
            optimality_gap=None,
            runtime_seconds=time.perf_counter() - started,
            fallback_used=True,
            status="completed",
            failure_reason=fallback_reason,
        )
    except ValueError as exc:
        return SupportSelectionResult(
            support=None,
            objective_value=None,
            solver_used="deterministic_constrained_greedy_fallback",
            solver_status="infeasible",
            optimality_gap=None,
            runtime_seconds=time.perf_counter() - started,
            fallback_used=True,
            status="failed",
            failure_reason=f"{fallback_reason}; {exc}",
        )


@dataclass(frozen=True, slots=True)
class RefinementResult:
    support: np.ndarray
    initial_objective: float
    final_objective: float
    iterations_accepted: int
    trace: tuple[dict[str, Any], ...]


def refine_support_one_swap(
    matrix: np.ndarray,
    initial_support: np.ndarray,
    tasks: Sequence[RidgeTask],
    constraints: SupportConstraints,
    *,
    alpha: float,
    y_floor: float,
    objective: str,
    max_iterations: int,
    improvement_tolerance: float,
) -> RefinementResult:
    """Deterministic exact-loss one-swap refinement using training tasks only."""

    if not tasks or any(task.split != "training" for task in tasks):
        raise ValueError("data leakage guard: refinement requires training tasks only")
    if objective not in {"mean_normalized_error", "worst_case_normalized_error"}:
        raise ValueError("unsupported refinement objective")
    if max_iterations < 0 or improvement_tolerance < 0.0:
        raise ValueError("refinement iteration and tolerance settings are invalid")
    values = np.asarray(matrix, dtype=np.float64)
    support = np.asarray(initial_support, dtype=bool).copy()
    report = support_constraint_report(values, support, constraints)
    if not report["valid"]:
        raise ValueError("initial support violates constraints")

    full_outputs = np.asarray(
        [
            task.functional
            @ ridge_svd_solution(values, task.residual, alpha=float(alpha))
            for task in tasks
        ],
        dtype=np.float64,
    )

    def loss(candidate_support: np.ndarray) -> float:
        sparse = np.where(candidate_support, values, 0.0)
        operator = _ridge_filter_operator(sparse, alpha)
        sparse_outputs = np.asarray(
            [task.functional @ (operator @ task.residual) for task in tasks],
            dtype=np.float64,
        )
        normalized = np.abs(sparse_outputs - full_outputs) / np.maximum(
            np.abs(full_outputs), float(y_floor)
        )
        if objective == "mean_normalized_error":
            return float(np.mean(normalized))
        return float(np.max(normalized))

    current = loss(support)
    initial = current
    trace: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "action": "initial",
            "removed_row": None,
            "removed_column": None,
            "added_row": None,
            "added_column": None,
            "objective_before": current,
            "objective_after": current,
            "improvement": 0.0,
            "accepted": True,
        }
    ]
    candidate_mask = values != 0.0
    accepted = 0
    for iteration in range(1, max_iterations + 1):
        selected_indices = np.flatnonzero(support.ravel())
        excluded_indices = np.flatnonzero(candidate_mask.ravel() & ~support.ravel())
        best: tuple[float, int, int, np.ndarray] | None = None
        for removed in selected_indices:
            for added in excluded_indices:
                trial = support.copy()
                trial.ravel()[removed] = False
                trial.ravel()[added] = True
                if not support_constraint_report(values, trial, constraints)["valid"]:
                    continue
                trial_loss = loss(trial)
                key = (trial_loss, int(removed), int(added))
                if best is None or key[:3] < best[:3]:
                    best = (trial_loss, int(removed), int(added), trial)
        if best is None or not (current - best[0] > float(improvement_tolerance)):
            trace.append(
                {
                    "iteration": iteration,
                    "action": "stop_no_strict_improvement",
                    "removed_row": None,
                    "removed_column": None,
                    "added_row": None,
                    "added_column": None,
                    "objective_before": current,
                    "objective_after": current if best is None else best[0],
                    "improvement": 0.0 if best is None else current - best[0],
                    "accepted": False,
                }
            )
            break
        trial_loss, removed, added, trial = best
        removed_row, removed_column = np.unravel_index(removed, values.shape)
        added_row, added_column = np.unravel_index(added, values.shape)
        trace.append(
            {
                "iteration": iteration,
                "action": "one_swap",
                "removed_row": int(removed_row),
                "removed_column": int(removed_column),
                "added_row": int(added_row),
                "added_column": int(added_column),
                "objective_before": current,
                "objective_after": trial_loss,
                "improvement": current - trial_loss,
                "accepted": True,
            }
        )
        support = trial
        current = trial_loss
        accepted += 1
    return RefinementResult(
        support=support,
        initial_objective=initial,
        final_objective=current,
        iterations_accepted=accepted,
        trace=tuple(trace),
    )


def deterministic_pareto_frontier(
    frame: pd.DataFrame,
    *,
    error_column: str,
    cost_column: str,
    tie_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a complete registry and a deterministic absolute-error frontier."""

    candidates = frame.copy()
    candidates["nondominated"] = pareto_nondominated(
        candidates, error_column=error_column, cost_column=cost_column
    )
    sort_columns = [cost_column, error_column, *tie_columns]
    candidates = candidates.sort_values(
        sort_columns, kind="stable", na_position="last"
    ).reset_index(drop=True)
    frontier = candidates[candidates["nondominated"]].reset_index(drop=True)
    return candidates, frontier


class CampaignCheckpoint:
    """Atomic stage checkpoint with explicit completed-output validation."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "checkpoint.json"

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"study_id": STUDY_ID, "stages": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"study_id": STUDY_ID, "stages": {}}
        if not isinstance(payload, dict) or payload.get("study_id") != STUDY_ID:
            return {"study_id": STUDY_ID, "stages": {}}
        payload.setdefault("stages", {})
        return payload

    def is_complete(self, stage: str) -> bool:
        record = self.read()["stages"].get(stage, {})
        if record.get("status") != "completed":
            return False
        return all((self.output_dir / relative).is_file() for relative in record.get("outputs", []))

    def clear(self, stage: str) -> None:
        payload = self.read()
        payload["stages"].pop(stage, None)
        atomic_write_json(self.path, payload)

    def mark_complete(
        self,
        stage: str,
        *,
        outputs: Sequence[str],
        result: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        payload = self.read()
        payload["stages"][stage] = {
            "status": "completed",
            "completed_at": now_iso(),
            "outputs": list(outputs),
            "result": _json_ready(result),
            "elapsed_seconds": float(elapsed_seconds),
        }
        payload["last_completed_stage"] = stage
        payload["last_update"] = now_iso()
        atomic_write_json(self.path, payload)


@dataclass(slots=True)
class OutputAwareStudyContext:
    output_dir: Path
    config_path: Path
    config: dict[str, Any]
    checkpoint: CampaignCheckpoint
    resume: bool
    force: bool
    max_workers: int
    random_seed_base: int
    _design: FrozenOutputAwareDesign | None = None

    @property
    def design(self) -> FrozenOutputAwareDesign:
        if self._design is None:
            self._design = build_frozen_output_aware_design(self.output_dir)
        return self._design


def make_context(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    resume: bool = False,
    force: bool = False,
    max_workers: int = 1,
    seed: int | None = None,
) -> OutputAwareStudyContext:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_study_configuration(config_path)
    base_seed = int(config["random_seed_base"] if seed is None else seed)
    return OutputAwareStudyContext(
        output_dir=destination,
        config_path=Path(config_path),
        config=config,
        checkpoint=CampaignCheckpoint(destination),
        resume=bool(resume),
        force=bool(force),
        max_workers=max(1, int(max_workers)),
        random_seed_base=base_seed,
    )


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _protected_file_snapshot(root: Path) -> dict[str, str]:
    protected_roots = (
        Path("manuscript"),
        Path("submission_package_tqe_final"),
        Path("outputs/sparse_integrated_chain"),
        Path("outputs/sparse_error_precision_study"),
    )
    snapshot: dict[str, str] = {}
    for relative_root in protected_roots:
        absolute = root / relative_root
        if not absolute.exists():
            continue
        for path in sorted(item for item in absolute.rglob("*") if item.is_file()):
            snapshot[str(path.relative_to(root))] = _sha256_file(path)
    return snapshot


def stage_audit(context: OutputAwareStudyContext) -> dict[str, Any]:
    audit_path = context.output_dir / "implementation_audit.md"
    if not audit_path.is_file():
        raise RuntimeError("implementation audit must exist before the campaign")
    design = context.design
    resolved = {
        **context.config,
        "root": str(Path.cwd().resolve()),
        "git_commit": git_commit_hash(),
        "physical_alpha": design.alpha,
        "matrix_shape": list(design.matrix.shape),
        "matrix_nonzeros": int(np.count_nonzero(design.matrix)),
        "matrix_fingerprint": stable_array_fingerprint(design.matrix),
        "baseline_residual_fingerprint": stable_array_fingerprint(
            design.baseline_residual
        ),
        "functionals_values": {
            key: value.tolist() for key, value in design.functionals.items()
        },
        "selected_rows": list(design.selected_rows),
        "selected_columns": list(design.selected_columns),
        "max_workers_recorded": context.max_workers,
        "random_seed_base_resolved": context.random_seed_base,
    }
    atomic_write_json(context.output_dir / "study_configuration.json", resolved)
    snapshot = _protected_file_snapshot(Path.cwd().resolve())
    atomic_write_json(
        context.output_dir / "protected_path_snapshot.json",
        {"created_at": now_iso(), "files": snapshot},
    )
    return {"matrix_fingerprint": resolved["matrix_fingerprint"], "protected_files": len(snapshot)}


def stage_pareto_audit(context: OutputAwareStudyContext) -> dict[str, Any]:
    from robust_qsvt_se.qsvt.sparse_error_precision_study import (
        build_phase_rounding_sensitivity,
        build_value_precision_pareto_tables,
    )

    audit_path = context.output_dir / "prior_pareto_audit.md"
    if not audit_path.is_file():
        raise RuntimeError("prior Pareto correction audit is missing")
    prior = Path("outputs/sparse_error_precision_study")
    grid = pd.read_csv(
        prior / "statevector_precision_grid.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    resources = pd.read_csv(
        prior / "signal_unitary_resource_registry.csv", dtype={"value_bits": str}
    )
    candidates, frontier = build_value_precision_pareto_tables(grid, resources)
    columns = [
        "frontier_kind",
        "cost_axis",
        "functional_id",
        "configuration_id",
        "value_bits",
        "phase_bits",
        "cost_value",
        "error_value",
        "error_semantics",
        "status",
        "nondominated",
    ]
    atomic_write_csv(
        context.output_dir / "prior_pareto_candidates_precision_cost_corrected.csv",
        candidates[columns],
    )
    atomic_write_csv(
        context.output_dir / "prior_pareto_frontier_precision_cost_corrected.csv",
        frontier[columns],
    )
    phase = build_phase_rounding_sensitivity(grid)
    phase_columns = [
        "curve_kind",
        "x_axis",
        "functional_id",
        "configuration_id",
        "value_bits",
        "phase_bits",
        "x_value",
        "error_value",
        "status",
        "executed_resource_frontier",
        "resource_semantics",
    ]
    atomic_write_csv(
        context.output_dir / "prior_phase_rounding_sensitivity.csv",
        phase[phase_columns],
    )
    return {
        "value_precision_candidates": len(candidates),
        "value_precision_frontier": len(frontier),
        "phase_sensitivity_rows": len(phase),
    }


def stage_residuals(context: OutputAwareStudyContext) -> dict[str, Any]:
    design = context.design
    bank_dir = context.output_dir / "residual_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, str | None] = {}
    failed = 0
    for split, seeds in (
        ("training", context.config["training_seed_ids"]),
        ("held_out", context.config["held_out_seed_ids"]),
    ):
        for raw_seed in seeds:
            seed = int(raw_seed)
            relative = Path("residual_bank") / f"{split}_seed_{seed}.npy"
            try:
                residual = generate_controlled_residual(
                    seed, selected_rows=design.selected_rows
                )
                fingerprint = stable_array_fingerprint(residual)
                _atomic_save_array(context.output_dir / relative, residual)
                status = "completed"
                failure_reason = ""
                fingerprints[str(seed)] = fingerprint
            except Exception as exc:  # retained failure, never replaced
                fingerprint = ""
                relative = Path("")
                status = "failed"
                failure_reason = f"{type(exc).__name__}: {exc}"
                fingerprints[str(seed)] = None
                failed += 1
            records.append(
                {
                    "seed_id": seed,
                    "split": split,
                    "status": status,
                    "failure_reason": failure_reason,
                    "residual_file": str(relative),
                    "residual_fingerprint": fingerprint,
                    "evaluation_matrix_fingerprint": stable_array_fingerprint(
                        design.matrix
                    ),
                    "weighting_convention": "default_diagonal_measurement_std_weighting",
                }
            )
    split_payload = {
        "declared_before_selector_evaluation": True,
        "training_seed_ids": [int(value) for value in context.config["training_seed_ids"]],
        "held_out_seed_ids": [int(value) for value in context.config["held_out_seed_ids"]],
        "residual_generation_config": context.config["residual_generation"],
        "matrix_fingerprint": stable_array_fingerprint(design.matrix),
        "residual_fingerprints": fingerprints,
        "records": records,
        "failed_residual_generations": failed,
        "failed_seed_policy": "retained_without_replacement",
    }
    atomic_write_json(context.output_dir / "residual_split.json", split_payload)
    return {"records": len(records), "failed": failed}


def load_residual_samples(
    output_dir: str | Path, *, split: str | None = None
) -> list[ResidualSample]:
    root = Path(output_dir)
    payload = json.loads((root / "residual_split.json").read_text(encoding="utf-8"))
    samples: list[ResidualSample] = []
    for record in payload["records"]:
        record_split = str(record["split"])
        if split is not None and record_split != split:
            continue
        if record["status"] == "completed":
            residual = np.load(root / record["residual_file"])
            fingerprint = stable_array_fingerprint(residual)
            if fingerprint != record["residual_fingerprint"]:
                raise RuntimeError("residual-bank fingerprint mismatch")
        else:
            residual = np.empty(0, dtype=np.float64)
            fingerprint = str(record.get("residual_fingerprint", ""))
        samples.append(
            ResidualSample(
                seed_id=int(record["seed_id"]),
                split=record_split,
                residual=residual,
                fingerprint=fingerprint,
                status=str(record["status"]),
                failure_reason=str(record.get("failure_reason", "")),
            )
        )
    return samples


def stage_gradients(context: OutputAwareStudyContext) -> dict[str, Any]:
    settings = context.config["gradient_validation"]
    samples = load_residual_samples(context.output_dir, split="training")
    completed = [sample for sample in samples if sample.status == "completed"]
    count = int(settings["training_residuals"])
    tasks = build_ridge_tasks(
        completed[:count], context.design.functionals, required_split="training"
    )
    frame = validate_ridge_gradient_entries(
        context.design.matrix,
        tasks,
        alpha=context.design.alpha,
        relative_step=float(settings["central_difference_relative_step"]),
        absolute_step_floor=float(settings["absolute_step_floor"]),
        entries_per_task=int(settings["entries_per_task"]),
        seed=context.random_seed_base,
    )
    atomic_write_csv(context.output_dir / "gradient_validation.csv", frame)
    maximum = float(frame["epsilon_grad"].max())
    median = float(frame.groupby("task_id")["epsilon_grad"].first().median())
    tolerance = float(settings["relative_frobenius_tolerance"])
    if maximum > tolerance:
        raise RuntimeError(
            f"gradient finite-difference validation failed: {maximum} > {tolerance}"
        )
    return {
        "rows": len(frame),
        "tasks": frame["task_id"].nunique(),
        "maximum_epsilon_grad": maximum,
        "median_epsilon_grad": median,
        "tolerance": tolerance,
    }


def stage_scores(context: OutputAwareStudyContext) -> dict[str, Any]:
    samples = load_residual_samples(context.output_dir, split="training")
    tasks = build_ridge_tasks(
        samples, context.design.functionals, required_split="training"
    )
    bundle = compute_output_aware_entry_scores(
        context.design.matrix,
        tasks,
        alpha=context.design.alpha,
        epsilon=float(context.config["score_normalization_epsilon"]),
    )
    frame = entry_score_frame(context.design.matrix, bundle)
    atomic_write_csv(context.output_dir / "entry_scores.csv", frame)
    metadata = {
        "task_ids": list(bundle.task_ids),
        "task_count": len(bundle.task_ids),
        "selection_data_split": "training_only",
        "score_normalization_epsilon": context.config["score_normalization_epsilon"],
        "score_formula": "abs(H_ij * gradient_ij), task-normalized before aggregation",
        "first_order_only_not_certificate": True,
    }
    atomic_write_json(context.output_dir / "entry_score_metadata.json", metadata)
    return {"entry_rows": len(frame), "training_tasks": len(tasks)}


def _load_score_arrays(output_dir: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(output_dir / "entry_scores.csv")
    arrays: dict[str, np.ndarray] = {}
    for column in ("sensitivity_mean", "sensitivity_worst_case"):
        array = np.zeros(shape, dtype=np.float64)
        for row in frame.itertuples(index=False):
            array[int(row.row), int(row.column)] = float(getattr(row, column))
        arrays[column] = array
    return arrays


def _support_id(
    selector: str,
    k_budget: int,
    slot_budget: int,
    support: np.ndarray | None,
    *,
    replicate: int | None = None,
) -> str:
    suffix = "failed" if support is None else stable_array_fingerprint(
        np.asarray(support, dtype=np.float64)
    )[:12]
    replicate_text = "" if replicate is None else f"_r{replicate:02d}"
    return f"{selector}_k{k_budget}_s{slot_budget}{replicate_text}_{suffix}"


def _write_support_file(
    output_dir: Path,
    *,
    support_id: str,
    selector: str,
    support: np.ndarray,
    matrix: np.ndarray,
    record: dict[str, Any],
) -> str:
    path = output_dir / "supports" / f"{support_id}.json"
    selected = np.argwhere(support)
    sparse_matrix = np.where(support, matrix, 0.0)
    payload = {
        "support_id": support_id,
        "selector": selector,
        "shape": list(support.shape),
        "selected_entries": [[int(row), int(column)] for row, column in selected],
        "support_fingerprint": stable_array_fingerprint(support.astype(np.float64)),
        "matrix_fingerprint": stable_array_fingerprint(sparse_matrix),
        "selection_data_split": record["selection_data_split"],
        "constraints": {
            "k_budget": int(record["k_budget"]),
            "slot_budget": int(record["slot_budget"]),
            "coverage_enabled": True,
        },
    }
    atomic_write_json(path, payload)
    return str(path.relative_to(output_dir))


def load_support(output_dir: str | Path, support_file: str) -> np.ndarray:
    root = Path(output_dir)
    payload = json.loads((root / support_file).read_text(encoding="utf-8"))
    support = np.zeros(tuple(int(value) for value in payload["shape"]), dtype=bool)
    for row, column in payload["selected_entries"]:
        support[int(row), int(column)] = True
    if stable_array_fingerprint(support.astype(np.float64)) != payload["support_fingerprint"]:
        raise RuntimeError("support fingerprint mismatch")
    return support


def _selection_record(
    *,
    matrix: np.ndarray,
    selector: str,
    constraints: SupportConstraints,
    result: SupportSelectionResult,
    output_dir: Path,
    selection_data_split: str,
    random_seed: int | None = None,
    random_replicate: int | None = None,
) -> dict[str, Any]:
    support_id = _support_id(
        selector,
        constraints.k_budget,
        constraints.slot_budget,
        result.support,
        replicate=random_replicate,
    )
    base: dict[str, Any] = {
        "support_id": support_id,
        "selector": selector,
        "k_budget": constraints.k_budget,
        "slot_budget": constraints.slot_budget,
        "selection_data_split": selection_data_split,
        "solver_used": result.solver_used,
        "solver_status": result.solver_status,
        "optimality_gap": result.optimality_gap,
        "selection_runtime": result.runtime_seconds,
        "fallback_used": result.fallback_used,
        "selection_objective_value": result.objective_value,
        "random_seed": random_seed,
        "random_replicate": random_replicate,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "stage": "supports",
        "last_completed_checkpoint": "scores",
    }
    if result.support is None:
        return {
            **base,
            "actual_nonzeros": np.nan,
            "actual_max_row_degree": np.nan,
            "actual_max_column_degree": np.nan,
            "slot_count": np.nan,
            "support_fingerprint": "",
            "matrix_fingerprint": "",
            "support_file": "",
        }
    report = support_constraint_report(matrix, result.support, constraints)
    if not report["valid"]:
        raise RuntimeError("completed selection result violates declared constraints")
    sparse_matrix = np.where(result.support, matrix, 0.0)
    slot_count = minimum_slot_count(sparse_matrix.T != 0.0)
    assignment = assign_slot_permutations(sparse_matrix.T != 0.0, slots=slot_count)
    validate_slot_assignment(sparse_matrix.T != 0.0, assignment)
    base.update(
        {
            "actual_nonzeros": report["actual_nonzeros"],
            "actual_max_row_degree": report["actual_max_row_degree"],
            "actual_max_column_degree": report["actual_max_column_degree"],
            "slot_count": int(slot_count),
            "support_fingerprint": stable_array_fingerprint(
                result.support.astype(np.float64)
            ),
            "matrix_fingerprint": stable_array_fingerprint(sparse_matrix),
            "last_completed_checkpoint": "supports",
        }
    )
    base["support_file"] = _write_support_file(
        output_dir,
        support_id=support_id,
        selector=selector,
        support=result.support,
        matrix=matrix,
        record=base,
    )
    return base


def stage_supports(context: OutputAwareStudyContext) -> dict[str, Any]:
    design = context.design
    score_arrays = _load_score_arrays(context.output_dir, design.matrix.shape)
    _row_leverage, _column_leverage, leverage_score = deterministic_ridge_leverage_scores(
        design.matrix, alpha=design.alpha
    )
    magnitude = np.abs(design.matrix)
    settings = context.config["milp"]
    records: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pair_index = 0
    for k_budget in context.config["support_budgets"]:
        for slot_budget in context.config["slot_budgets"]:
            constraints = SupportConstraints(int(k_budget), int(slot_budget), True)
            objectives = {
                "global_magnitude": magnitude,
                "balanced_magnitude": magnitude,
                "leverage_weighted": leverage_score,
                "sensitivity_initial_mean": score_arrays["sensitivity_mean"],
                "sensitivity_initial_worst_case": score_arrays[
                    "sensitivity_worst_case"
                ],
            }
            for selector, objective in objectives.items():
                result = select_resource_constrained_support(
                    design.matrix,
                    objective,
                    constraints,
                    time_limit_seconds=float(settings["time_limit_seconds"]),
                    relative_mip_gap=float(settings["relative_mip_gap"]),
                    tie_epsilon_relative=float(
                        settings["deterministic_tie_epsilon_relative"]
                    ),
                )
                split = (
                    "training_only"
                    if selector.startswith("sensitivity")
                    else "matrix_only_no_residuals"
                )
                record = _selection_record(
                    matrix=design.matrix,
                    selector=selector,
                    constraints=constraints,
                    result=result,
                    output_dir=context.output_dir,
                    selection_data_split=split,
                )
                records.append(record)
                selection_rows.append(dict(record))

            for replicate in range(int(context.config["random_supports_per_budget"])):
                random_seed = context.random_seed_base + 10_000 * pair_index + replicate
                rng = np.random.default_rng(random_seed)
                random_scores = np.zeros_like(design.matrix)
                random_scores[design.matrix != 0.0] = rng.random(
                    int(np.count_nonzero(design.matrix))
                )
                result = select_resource_constrained_support(
                    design.matrix,
                    random_scores,
                    constraints,
                    time_limit_seconds=float(settings["time_limit_seconds"]),
                    relative_mip_gap=float(settings["relative_mip_gap"]),
                    tie_epsilon_relative=float(
                        settings["deterministic_tie_epsilon_relative"]
                    ),
                )
                record = _selection_record(
                    matrix=design.matrix,
                    selector="random_feasible",
                    constraints=constraints,
                    result=result,
                    output_dir=context.output_dir,
                    selection_data_split="fixed_seed_random_no_residuals",
                    random_seed=random_seed,
                    random_replicate=replicate,
                )
                records.append(record)
                selection_rows.append(dict(record))
            pair_index += 1
    registry = pd.DataFrame(records).sort_values(
        ["k_budget", "slot_budget", "selector", "random_replicate", "support_id"],
        kind="stable",
        na_position="first",
    )
    atomic_write_csv(context.output_dir / "support_registry.csv", registry)
    atomic_write_csv(
        context.output_dir / "support_selection_results.csv",
        pd.DataFrame(selection_rows),
    )
    return {
        "supports": len(registry),
        "completed": int((registry["status"] == "completed").sum()),
        "failed": int((registry["status"] != "completed").sum()),
    }


def _training_result_rows(
    design: FrozenOutputAwareDesign,
    registry: pd.DataFrame,
    tasks: Sequence[RidgeTask],
    *,
    output_dir: Path,
    y_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in registry.itertuples(index=False):
        if record.status != "completed":
            continue
        support = load_support(output_dir, str(record.support_file))
        sparse_matrix = np.where(support, design.matrix, 0.0)
        full_outputs, sparse_outputs, normalized = exact_task_errors(
            design.matrix,
            sparse_matrix,
            tasks,
            alpha=design.alpha,
            y_floor=y_floor,
        )
        for index, task in enumerate(tasks):
            signed = float(sparse_outputs[index] - full_outputs[index])
            rows.append(
                {
                    "support_id": record.support_id,
                    "selector": record.selector,
                    "k_budget": int(record.k_budget),
                    "slot_budget": int(record.slot_budget),
                    "actual_nonzeros": int(record.actual_nonzeros),
                    "slot_count": int(record.slot_count),
                    "task_id": task.task_id,
                    "seed_id": task.seed_id,
                    "split": "training",
                    "functional_id": task.functional_id,
                    "full_ridge_output": float(full_outputs[index]),
                    "sparse_ridge_output": float(sparse_outputs[index]),
                    "signed_error": signed,
                    "absolute_error": abs(signed),
                    "normalized_error": float(normalized[index]),
                }
            )
    return pd.DataFrame(rows)


def stage_refinement(context: OutputAwareStudyContext) -> dict[str, Any]:
    design = context.design
    samples = load_residual_samples(context.output_dir, split="training")
    tasks = build_ridge_tasks(samples, design.functionals, required_split="training")
    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    selection = pd.read_csv(context.output_dir / "support_selection_results.csv")
    # Make force/retry idempotent: rebuild refined rows from their frozen initial supports.
    registry = registry[~registry["selector"].isin(REFINED_SELECTORS)].copy()
    selection = selection[~selection["selector"].isin(REFINED_SELECTORS)].copy()
    settings = context.config["refinement"]
    trace_rows: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []
    for k_budget in context.config["support_budgets"]:
        for slot_budget in context.config["slot_budgets"]:
            constraints = SupportConstraints(int(k_budget), int(slot_budget), True)
            for initial_selector, refined_selector, objective in (
                (
                    "sensitivity_initial_mean",
                    "sensitivity_refined_mean",
                    "mean_normalized_error",
                ),
                (
                    "sensitivity_initial_worst_case",
                    "sensitivity_refined_worst_case",
                    "worst_case_normalized_error",
                ),
            ):
                matched = registry[
                    (registry["selector"] == initial_selector)
                    & (registry["k_budget"] == int(k_budget))
                    & (registry["slot_budget"] == int(slot_budget))
                    & (registry["status"] == "completed")
                ]
                if matched.empty:
                    failed_result = SupportSelectionResult(
                        support=None,
                        objective_value=None,
                        solver_used="deterministic_exact_one_swap",
                        solver_status="not_run_missing_initial_support",
                        optimality_gap=None,
                        runtime_seconds=0.0,
                        fallback_used=False,
                        status="failed",
                        failure_reason="optimization_failure: initial support unavailable",
                    )
                    new_records.append(
                        _selection_record(
                            matrix=design.matrix,
                            selector=refined_selector,
                            constraints=constraints,
                            result=failed_result,
                            output_dir=context.output_dir,
                            selection_data_split="training_only",
                        )
                    )
                    continue
                initial_record = matched.sort_values("support_id", kind="stable").iloc[0]
                initial_support = load_support(
                    context.output_dir, str(initial_record["support_file"])
                )
                started = time.perf_counter()
                refined = refine_support_one_swap(
                    design.matrix,
                    initial_support,
                    tasks,
                    constraints,
                    alpha=design.alpha,
                    y_floor=float(context.config["y_floor"]),
                    objective=objective,
                    max_iterations=int(settings["max_iterations"]),
                    improvement_tolerance=float(settings["strict_improvement_tolerance"]),
                )
                elapsed = time.perf_counter() - started
                result = SupportSelectionResult(
                    support=refined.support,
                    objective_value=refined.final_objective,
                    solver_used="deterministic_exhaustive_exact_loss_one_swap",
                    solver_status="locally_refined_feasible",
                    optimality_gap=None,
                    runtime_seconds=elapsed,
                    fallback_used=False,
                    status="completed",
                    failure_reason="",
                )
                record = _selection_record(
                    matrix=design.matrix,
                    selector=refined_selector,
                    constraints=constraints,
                    result=result,
                    output_dir=context.output_dir,
                    selection_data_split="training_only",
                )
                record["refinement_initial_support_id"] = initial_record["support_id"]
                record["refinement_objective"] = objective
                record["refinement_initial_objective"] = refined.initial_objective
                record["refinement_final_objective"] = refined.final_objective
                record["refinement_iterations_accepted"] = refined.iterations_accepted
                record["stage"] = "refinement"
                record["last_completed_checkpoint"] = "refinement"
                new_records.append(record)
                for trace in refined.trace:
                    trace_rows.append(
                        {
                            "support_id": record["support_id"],
                            "selector": refined_selector,
                            "k_budget": int(k_budget),
                            "slot_budget": int(slot_budget),
                            "objective": objective,
                            "initial_support_id": initial_record["support_id"],
                            **trace,
                        }
                    )
    if new_records:
        registry = pd.concat([registry, pd.DataFrame(new_records)], ignore_index=True)
        selection = pd.concat([selection, pd.DataFrame(new_records)], ignore_index=True)
    registry = registry.sort_values(
        ["k_budget", "slot_budget", "selector", "random_replicate", "support_id"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)
    atomic_write_csv(context.output_dir / "support_registry.csv", registry)
    atomic_write_csv(context.output_dir / "support_selection_results.csv", selection)
    atomic_write_csv(
        context.output_dir / "refinement_traces.csv", pd.DataFrame(trace_rows)
    )
    training = _training_result_rows(
        design,
        registry,
        tasks,
        output_dir=context.output_dir,
        y_floor=float(context.config["y_floor"]),
    )
    atomic_write_csv(context.output_dir / "training_results.csv", training)
    traces_frame = pd.DataFrame(trace_rows)
    accepted_swaps = int(
        (
            (traces_frame.get("action", pd.Series(dtype=str)) == "one_swap")
            & traces_frame.get("accepted", pd.Series(dtype=bool)).astype(bool)
        ).sum()
    )
    return {
        "refined_supports": len(new_records),
        "accepted_swaps": accepted_swaps,
        "training_rows": len(training),
        "total_support_records": len(registry),
    }


def stage_certificates(context: OutputAwareStudyContext) -> dict[str, Any]:
    design = context.design
    samples = [
        *load_residual_samples(context.output_dir, split="training"),
        *load_residual_samples(context.output_dir, split="held_out"),
    ]
    tasks: list[RidgeTask] = []
    tasks.extend(
        build_ridge_tasks(
            [sample for sample in samples if sample.split == "training"],
            design.functionals,
            required_split="training",
        )
    )
    tasks.extend(
        build_ridge_tasks(
            [sample for sample in samples if sample.split == "held_out"],
            design.functionals,
            required_split="held_out",
        )
    )
    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    settings = context.config["certificate"]
    rows: list[dict[str, Any]] = []
    violations = 0
    for support_record in registry.itertuples(index=False):
        if support_record.status != "completed":
            continue
        support = load_support(context.output_dir, str(support_record.support_file))
        sparse_matrix = np.where(support, design.matrix, 0.0)
        full_operator = _ridge_filter_operator(design.matrix, design.alpha)
        sparse_operator = _ridge_filter_operator(sparse_matrix, design.alpha)
        for task in tasks:
            full_output = float(task.functional @ (full_operator @ task.residual))
            sparse_output = float(task.functional @ (sparse_operator @ task.residual))
            signed_error = sparse_output - full_output
            raw_certificate = compute_ridge_selected_output_certificate(
                design.matrix,
                sparse_matrix,
                task.residual,
                task.functional,
                design.alpha,
            )
            certificate = validate_ridge_selected_output_certificate(
                raw_certificate,
                abs(signed_error),
                absolute_tolerance=float(settings["absolute_validation_tolerance"]),
                relative_tolerance=float(settings["relative_validation_tolerance"]),
            )
            if not certificate.certificate_holds:
                violations += 1
            rows.append(
                {
                    "support_id": support_record.support_id,
                    "selector": support_record.selector,
                    "k_budget": int(support_record.k_budget),
                    "slot_budget": int(support_record.slot_budget),
                    "actual_nonzeros": int(support_record.actual_nonzeros),
                    "slot_count": int(support_record.slot_count),
                    "task_id": task.task_id,
                    "seed_id": task.seed_id,
                    "split": task.split,
                    "functional_id": task.functional_id,
                    "full_ridge_output": full_output,
                    "sparse_ridge_output": sparse_output,
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                    "linearized_prediction": ridge_linearized_signed_error(
                        design.matrix,
                        sparse_matrix,
                        task.residual,
                        task.functional,
                        design.alpha,
                    ),
                    "matrix_delta_spectral": certificate.matrix_delta_spectral,
                    "operator_bound_forward": certificate.operator_bound_forward,
                    "operator_bound_reverse": certificate.operator_bound_reverse,
                    "certificate_bound": certificate.selected_output_bound,
                    "certificate_holds": certificate.certificate_holds,
                    "certificate_tightness": certificate.tightness_ratio,
                    "certificate_absolute_tolerance": settings[
                        "absolute_validation_tolerance"
                    ],
                    "certificate_relative_tolerance": settings[
                        "relative_validation_tolerance"
                    ],
                    "certificate_computation_uses_actual_error": False,
                    "linearized_prediction_is_certificate": False,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_write_csv(context.output_dir / "certificate_registry.csv", frame)
    summary = (
        frame.groupby(
            ["support_id", "selector", "k_budget", "slot_budget", "split"],
            dropna=False,
            sort=True,
        )
        .agg(
            certificate_coverage=("certificate_holds", "mean"),
            median_certificate_bound=("certificate_bound", "median"),
            worst_certificate_bound=("certificate_bound", "max"),
            median_tightness=("certificate_tightness", "median"),
            worst_tightness=("certificate_tightness", "max"),
            violations=("certificate_holds", lambda values: int((~values).sum())),
            tasks=("task_id", "count"),
        )
        .reset_index()
    )
    atomic_write_csv(context.output_dir / "certificate_summary.csv", summary)
    if violations:
        raise RuntimeError(f"certificate_violation: {violations} task bounds failed")
    return {"certificate_rows": len(frame), "violations": violations}


def stage_heldout(context: OutputAwareStudyContext) -> dict[str, Any]:
    certificates = pd.read_csv(context.output_dir / "certificate_registry.csv")
    heldout = certificates[certificates["split"] == "held_out"].copy()
    heldout["normalized_error"] = heldout["absolute_error"] / np.maximum(
        heldout["full_ridge_output"].abs(), float(context.config["y_floor"])
    )
    heldout["above_declared_failure_threshold"] = heldout["normalized_error"] > float(
        context.config["normalized_error_failure_threshold"]
    )
    columns = [
        "support_id",
        "selector",
        "k_budget",
        "slot_budget",
        "actual_nonzeros",
        "slot_count",
        "task_id",
        "seed_id",
        "functional_id",
        "full_ridge_output",
        "sparse_ridge_output",
        "signed_error",
        "absolute_error",
        "normalized_error",
        "linearized_prediction",
        "certificate_bound",
        "certificate_holds",
        "certificate_tightness",
        "above_declared_failure_threshold",
    ]
    atomic_write_csv(context.output_dir / "heldout_results.csv", heldout[columns])
    per_support = (
        heldout.groupby(
            [
                "support_id",
                "selector",
                "k_budget",
                "slot_budget",
                "actual_nonzeros",
                "slot_count",
            ],
            sort=True,
        )
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            percentile_90_absolute_error=("absolute_error", lambda values: values.quantile(0.9)),
            worst_absolute_error=("absolute_error", "max"),
            mean_normalized_error=("normalized_error", "mean"),
            median_normalized_error=("normalized_error", "median"),
            percentile_90_normalized_error=(
                "normalized_error", lambda values: values.quantile(0.9)
            ),
            worst_normalized_error=("normalized_error", "max"),
            failure_fraction=("above_declared_failure_threshold", "mean"),
            certificate_coverage=("certificate_holds", "mean"),
            median_certificate_bound=("certificate_bound", "median"),
            median_certificate_tightness=("certificate_tightness", "median"),
            worst_certificate_tightness=("certificate_tightness", "max"),
            heldout_tasks=("task_id", "count"),
        )
        .reset_index()
    )
    atomic_write_csv(context.output_dir / "heldout_support_summary.csv", per_support)
    comparison = (
        per_support.groupby(["selector", "k_budget", "slot_budget"], sort=True)
        .agg(
            support_count=("support_id", "count"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            std_absolute_error_across_supports=("mean_absolute_error", "std"),
            median_absolute_error=("median_absolute_error", "mean"),
            percentile_90_absolute_error=("percentile_90_absolute_error", "mean"),
            worst_absolute_error=("worst_absolute_error", "max"),
            mean_normalized_error=("mean_normalized_error", "mean"),
            failure_fraction=("failure_fraction", "mean"),
            certificate_coverage=("certificate_coverage", "mean"),
            median_certificate_tightness=("median_certificate_tightness", "median"),
            worst_certificate_tightness=("worst_certificate_tightness", "max"),
            mean_actual_nonzeros=("actual_nonzeros", "mean"),
            mean_slot_count=("slot_count", "mean"),
        )
        .reset_index()
    )
    comparison["evaluation_split"] = "held_out_only"
    comparison["resource_matched_comparison"] = True
    atomic_write_csv(context.output_dir / "selector_comparison_summary.csv", comparison)
    return {
        "heldout_rows": len(heldout),
        "support_summaries": len(per_support),
        "certificate_coverage": float(heldout["certificate_holds"].mean()),
    }


def stage_resources(context: OutputAwareStudyContext) -> dict[str, Any]:
    from qiskit import transpile

    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    deterministic = registry[registry["selector"].isin(DETERMINISTIC_SELECTORS)].copy()
    resource_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    tolerance = float(
        context.config["resource_validation"]["wrapper_reconstruction_tolerance"]
    )
    for record in deterministic.itertuples(index=False):
        common = {
            "support_id": record.support_id,
            "selector": record.selector,
            "k_budget": int(record.k_budget),
            "slot_budget": int(record.slot_budget),
            "support_fingerprint": str(record.support_fingerprint),
        }
        if record.status != "completed":
            failure = {
                **common,
                "status": "failed",
                "failure_reason": str(record.failure_reason),
            }
            resource_rows.append(failure)
            slot_rows.append(failure)
            continue
        if record.support_fingerprint in cache:
            measured = cache[str(record.support_fingerprint)]
            resource_rows.append({**common, **measured["resource"]})
            slot_rows.append({**common, **measured["slot"]})
            continue
        try:
            support = load_support(context.output_dir, str(record.support_file))
            sparse_matrix = np.where(support, context.design.matrix, 0.0)
            pattern = sparse_matrix.T != 0.0
            minimum_slots = minimum_slot_count(pattern)
            wrapper = validate_complete_wrapper(
                _as_quantized_block(sparse_matrix, 53),
                encode_transpose=True,
                transpile_circuit=False,
            )
            slot_validation = validate_slot_assignment(pattern, wrapper.assignment)
            compiled = transpile(
                wrapper.circuit,
                basis_gates=list(context.config["resource_validation"]["basis_gates"]),
                optimization_level=int(
                    context.config["resource_validation"]["optimization_level"]
                ),
            )
            counts = {
                str(key): int(value) for key, value in compiled.count_ops().items()
            }
            if wrapper.top_left_reconstruction_error > tolerance:
                raise RuntimeError(
                    "wrapper reconstruction error "
                    f"{wrapper.top_left_reconstruction_error} exceeds {tolerance}"
                )
            resource = {
                "actual_nonzeros": int(np.count_nonzero(sparse_matrix)),
                "slot_count": int(wrapper.slots),
                "normalization_mu": float(np.max(np.abs(sparse_matrix))),
                "beta_native": float(wrapper.normalization_factor),
                "signal_unitary_gate_count": int(sum(counts.values())),
                "signal_unitary_depth": int(compiled.depth()),
                "cx_count": int(counts.get("cx", 0)),
                "controlled_value_rotations": int(wrapper.slots * sparse_matrix.shape[1]),
                "wrapper_reconstruction_error": float(
                    wrapper.top_left_reconstruction_error
                ),
                "wrapper_statevector_error": float(wrapper.statevector_max_error),
                "wrapper_unitarity_error": float(wrapper.unitarity_error),
                "resource_measurement": (
                    "executed_qiskit_transpile_u3_cx_optimization_level_1"
                ),
                "value_semantics": "exact_selected_original_values_no_quantization",
                "status": "completed",
                "failure_reason": "",
            }
            slot = {
                "actual_nonzeros": int(np.count_nonzero(sparse_matrix)),
                "minimum_slot_count": int(minimum_slots),
                "slot_count": int(wrapper.assignment.slots),
                "actual_max_row_degree_encoded": int(pattern.sum(axis=1).max()),
                "actual_max_column_degree_encoded": int(pattern.sum(axis=0).max()),
                "slot_assignment_valid": bool(slot_validation["valid"]),
                "real_edges_covered_exactly_once": bool(
                    slot_validation["real_edges_covered_exactly_once"]
                ),
                "per_slot_real_edge_counts": json.dumps(
                    slot_validation["per_slot_real_edge_counts"], separators=(",", ":")
                ),
                "augmenting_visits": int(slot_validation["augmenting_visits"]),
                "visit_budget": int(slot_validation["visit_budget"]),
                "status": "completed",
                "failure_reason": "",
            }
            cache[str(record.support_fingerprint)] = {"resource": resource, "slot": slot}
            resource_rows.append({**common, **resource})
            slot_rows.append({**common, **slot})
        except Exception as exc:  # retained deterministic resource failure
            failure_reason = f"resource_compilation_limit: {type(exc).__name__}: {exc}"
            failure = {**common, "status": "failed", "failure_reason": failure_reason}
            resource_rows.append(failure)
            slot_rows.append(failure)
    resources = pd.DataFrame(resource_rows)
    slots = pd.DataFrame(slot_rows)
    atomic_write_csv(context.output_dir / "resource_registry.csv", resources)
    atomic_write_csv(context.output_dir / "slot_validation.csv", slots)
    failures = int((resources["status"] != "completed").sum())
    if failures:
        raise RuntimeError(f"resource_compilation_limit: {failures} deterministic supports")
    return {
        "deterministic_supports": len(resources),
        "unique_wrappers_measured": len(cache),
        "failures": failures,
    }


@dataclass(frozen=True, slots=True)
class CommonPaddedWrapper:
    matrix: np.ndarray
    slots: int
    mu: float
    beta: float
    assignment: Any
    circuit: Any
    unitary: np.ndarray
    encoded_block: np.ndarray
    target_block: np.ndarray
    reconstruction_error: float
    unitarity_error: float


def build_common_padded_wrapper(
    matrix: np.ndarray,
    *,
    slots: int,
    mu: float,
) -> CommonPaddedWrapper:
    """Build the existing sparse wrapper with explicit common slot and value padding."""

    from qiskit.quantum_info import Operator

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("common padded wrapper requires a nonempty square matrix")
    if slots < minimum_slot_count(values.T != 0.0):
        raise ValueError("common slot count is below the selected support degree")
    if not np.isfinite(mu) or mu <= 0.0 or mu + 1.0e-12 < np.max(np.abs(values)):
        raise ValueError("common mu must upper-bound every selected matrix value")
    assignment = assign_slot_permutations(values.T != 0.0, slots=int(slots))
    validate_slot_assignment(values.T != 0.0, assignment)
    circuit, _diffusion = build_complete_wrapper_circuit(values.T, float(mu), assignment)
    unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    beta = float(slots) * float(mu)
    target = values.T / beta
    encoded = np.real(unitary[: values.shape[0], : values.shape[1]])
    reconstruction = float(np.max(np.abs(encoded - target)))
    unitarity = float(
        np.max(np.abs(unitary.conj().T @ unitary - np.eye(unitary.shape[0])))
    )
    return CommonPaddedWrapper(
        matrix=values.copy(),
        slots=int(slots),
        mu=float(mu),
        beta=beta,
        assignment=assignment,
        circuit=circuit,
        unitary=unitary,
        encoded_block=encoded,
        target_block=target,
        reconstruction_error=reconstruction,
        unitarity_error=unitarity,
    )


def _select_common_qsvt_subset(
    context: OutputAwareStudyContext,
) -> list[dict[str, Any]]:
    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    training = pd.read_csv(context.output_dir / "training_results.csv")
    means = (
        training.groupby("support_id", sort=True)["normalized_error"]
        .mean()
        .rename("training_mean_normalized_error")
        .reset_index()
    )
    candidates = registry.merge(means, on="support_id", how="left")
    settings = context.config["common_qsvt"]
    target_s = int(settings["validation_slot_budget"])
    subset: list[dict[str, Any]] = []

    def choose(frame: pd.DataFrame, *, role: str, k_budget: int) -> None:
        feasible = frame[
            (frame["k_budget"] == k_budget)
            & (frame["slot_budget"] == target_s)
            & (frame["status"] == "completed")
        ].copy()
        if feasible.empty:
            raise RuntimeError(
                f"qsvt_statevector_failure: no training-selected support for {role}, k={k_budget}"
            )
        selected = feasible.sort_values(
            ["training_mean_normalized_error", "support_id"], kind="stable"
        ).iloc[0]
        subset.append(
            {
                "support_id": str(selected["support_id"]),
                "selector": str(selected["selector"]),
                "role": role,
                "k_budget": int(k_budget),
                "slot_budget": target_s,
                "actual_slot_count": int(selected["slot_count"]),
                "training_mean_normalized_error": float(
                    selected["training_mean_normalized_error"]
                ),
                "support_file": str(selected["support_file"]),
            }
        )

    for raw_k in settings["validation_nonzero_budgets"]:
        k_budget = int(raw_k)
        choose(
            candidates[candidates["selector"].isin(
                ["global_magnitude", "balanced_magnitude"]
            )],
            role="best_magnitude_training_only",
            k_budget=k_budget,
        )
        choose(
            candidates[candidates["selector"].isin(
                ["sensitivity_initial_mean", "sensitivity_initial_worst_case"]
            )],
            role="best_sensitivity_training_only",
            k_budget=k_budget,
        )
        choose(
            candidates[candidates["selector"].isin(
                ["sensitivity_refined_mean", "sensitivity_refined_worst_case"]
            )],
            role="best_refined_training_only",
            k_budget=k_budget,
        )
        random = candidates[
            (candidates["selector"] == "random_feasible")
            & (candidates["random_replicate"] == 0)
        ]
        choose(random, role="random_replicate_zero_predeclared", k_budget=k_budget)
    return subset


def _common_design_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_qsvt(context: OutputAwareStudyContext) -> dict[str, Any]:
    from numpy.polynomial import Polynomial
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import (
        build_structured_qsvt_operator_circuit,
    )
    from robust_qsvt_se.qsvt.phase_synthesis import (
        synthesize_pennylane_phases_cached,
        validate_qsvt_polynomial,
    )

    settings = context.config["common_qsvt"]
    subset = _select_common_qsvt_subset(context)
    common_slots = max(int(record["actual_slot_count"]) for record in subset)
    common_mu = float(np.max(np.abs(context.design.matrix)))
    common_beta = common_slots * common_mu
    common_lambda = context.design.alpha / common_beta**2
    selected_target = None
    degree_trials: list[dict[str, Any]] = []
    for raw_degree in settings["candidate_degrees"]:
        degree = int(raw_degree)
        try:
            target = fit_codesigned_bounded_polynomial(
                beta=common_beta,
                alpha=context.design.alpha,
                domain_min=float(settings["common_domain_min"]),
                domain_max=float(settings["common_domain_max"]),
                degree=degree,
                margin=float(settings["target_margin"]),
            )
            validation = validate_qsvt_polynomial(
                np.asarray(target.coefficients),
                parity="odd",
                bound_tolerance=float(settings["bound_tolerance"]),
            )
            accepted = bool(
                target.fit_max_abs_error
                <= float(settings["uniform_approximation_tolerance"])
            )
            degree_trials.append(
                {
                    "degree": degree,
                    "fit_max_abs_error": target.fit_max_abs_error,
                    "bounded_max_abs": target.bounded_max_abs,
                    "validation": validation,
                    "accepted": accepted,
                    "failure_reason": "",
                }
            )
            if accepted:
                selected_target = target
                break
        except Exception as exc:
            degree_trials.append(
                {
                    "degree": degree,
                    "accepted": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if selected_target is None:
        raise RuntimeError("polynomial_fit_failure: no candidate degree met tolerance")
    coefficients = np.asarray(selected_target.coefficients, dtype=np.float64)
    phase_result = synthesize_pennylane_phases_cached(
        coefficients,
        angle_solver=str(settings["phase_solver"]),
        cache_dir=context.output_dir / "common_phase_cache",
        cache_metadata={
            "study_id": STUDY_ID,
            "track": "common_padded_wrapper",
            "beta": common_beta,
            "alpha": context.design.alpha,
            "degree": selected_target.degree,
        },
    )
    phases = np.asarray(phase_result.phases, dtype=np.float64)
    if phases.size != selected_target.degree + 1:
        raise RuntimeError("qsvt_statevector_failure: common phase count mismatch")
    design_payload: dict[str, Any] = {
        "study_id": STUDY_ID,
        "track": "common_padded_wrapper_primary",
        "support_subset_declared_by_training_policy": subset,
        "common_slots": common_slots,
        "common_mu": common_mu,
        "common_beta": common_beta,
        "physical_alpha": context.design.alpha,
        "common_lambda": common_lambda,
        "common_C": selected_target.bound_C,
        "degree": selected_target.degree,
        "phase_count": int(phases.size),
        "phases": phases.tolist(),
        "phase_fingerprint": stable_array_fingerprint(phases),
        "polynomial_coefficients": coefficients.tolist(),
        "polynomial_fingerprint": stable_array_fingerprint(coefficients),
        "uniform_approximation_tolerance": settings[
            "uniform_approximation_tolerance"
        ],
        "fit_max_abs_error": selected_target.fit_max_abs_error,
        "bounded_max_abs": selected_target.bounded_max_abs,
        "degree_trials": degree_trials,
        "degree_selected_before_support_specific_qsvt_outputs": True,
        "phase_refit_policy": settings["phase_refit_policy"],
        "phase_synthesis_metadata": phase_result.metadata,
        "phase_synthesis_cache_hit": phase_result.cache_hit,
        "validation_residual_seed": settings["validation_residual_seed"],
    }
    design_payload["common_design_fingerprint"] = _common_design_fingerprint(
        design_payload
    )
    # This durable declaration is written before any support-specific QSVT output.
    atomic_write_json(context.output_dir / "common_qsvt_design.json", design_payload)

    residual = np.asarray(context.design.baseline_residual, dtype=np.float64)
    residual_unit = residual / np.linalg.norm(residual)
    full_update = ridge_svd_solution(
        context.design.matrix, residual, alpha=context.design.alpha
    )
    polynomial = Polynomial(coefficients)
    rows: list[dict[str, Any]] = []
    failures = 0
    for selected in subset:
        try:
            support = load_support(context.output_dir, selected["support_file"])
            sparse_matrix = np.where(support, context.design.matrix, 0.0)
            wrapper = build_common_padded_wrapper(
                sparse_matrix, slots=common_slots, mu=common_mu
            )
            if wrapper.reconstruction_error > float(
                context.config["resource_validation"]["wrapper_reconstruction_tolerance"]
            ):
                raise RuntimeError("common padded-wrapper reconstruction failed")
            qsvt_bundle = build_structured_qsvt_operator_circuit(
                wrapper.unitary, phases, encoded_dimension=sparse_matrix.shape[1]
            )
            initial = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
            initial[: sparse_matrix.shape[0]] = residual_unit
            evolved = Statevector(initial).evolve(
                qsvt_bundle.qsvt_operator_circuit
            ).data
            encoded = np.asarray(evolved[: sparse_matrix.shape[1]], dtype=np.complex128)
            postselection = float(np.vdot(encoded, encoded).real)
            normalized_matrix = sparse_matrix.T / common_beta
            left, singular_values, right_t = np.linalg.svd(
                normalized_matrix, full_matrices=False
            )
            exact_action = (
                left @ np.diag(polynomial(singular_values)) @ right_t
            ) @ residual_unit
            action_error = float(
                np.linalg.norm(np.real(encoded) - exact_action)
                / max(np.linalg.norm(exact_action), 1.0e-30)
            )
            if action_error > 1.0e-6:
                raise RuntimeError(f"QSVT circuit/exact polynomial mismatch {action_error}")
            physical_scale = (
                selected_target.bound_C / common_beta * float(np.linalg.norm(residual))
            )
            exact_update = physical_scale * exact_action
            qsvt_update = physical_scale * np.real(encoded)
            sparse_ridge = ridge_svd_solution(
                sparse_matrix, residual, alpha=context.design.alpha
            )
            compiled_signal = transpile(
                wrapper.circuit,
                basis_gates=list(
                    context.config["resource_validation"]["basis_gates"]
                ),
                optimization_level=int(
                    context.config["resource_validation"]["optimization_level"]
                ),
            )
            signal_gate_count = int(sum(compiled_signal.count_ops().values()))
            signal_calls = int(selected_target.degree)
            phase_count = int(phases.size)
            gates_per_attempt = signal_calls * signal_gate_count + phase_count
            for functional_id in FUNCTIONAL_IDS:
                functional = context.design.functionals[functional_id]
                ridge_output = float(functional @ sparse_ridge)
                exact_output = float(functional @ exact_update)
                qsvt_output = float(functional @ qsvt_update)
                full_output = float(functional @ full_update)
                rows.append(
                    {
                        **selected,
                        "functional_id": functional_id,
                        "common_design_fingerprint": design_payload[
                            "common_design_fingerprint"
                        ],
                        "phase_fingerprint": design_payload["phase_fingerprint"],
                        "polynomial_fingerprint": design_payload[
                            "polynomial_fingerprint"
                        ],
                        "beta": common_beta,
                        "lambda": common_lambda,
                        "C": selected_target.bound_C,
                        "degree": selected_target.degree,
                        "phase_count": phase_count,
                        "common_signal_call_count": signal_calls,
                        "ridge_output_sparse_matrix": ridge_output,
                        "exact_polynomial_svt_output": exact_output,
                        "sparse_qsvt_statevector_output": qsvt_output,
                        "full_ridge_output": full_output,
                        "selected_output_qsvt_error_vs_sparse_ridge": abs(
                            qsvt_output - ridge_output
                        ),
                        "full_to_qsvt_selected_output_error": abs(
                            qsvt_output - full_output
                        ),
                        "postselection_probability": postselection,
                        "qsvt_action_error_vs_exact_polynomial": action_error,
                        "common_wrapper_reconstruction_error": (
                            wrapper.reconstruction_error
                        ),
                        "common_signal_unitary_gate_count": signal_gate_count,
                        "gates_per_attempt": gates_per_attempt,
                        "executed_cost_semantics": (
                            "measured_common_signal_unitary_times_signal_calls_plus_"
                            "projector_phase_operation_count"
                        ),
                        "per_support_phase_refit": False,
                        "ridge_and_qsvt_matrix_fingerprint": stable_array_fingerprint(
                            sparse_matrix
                        ),
                        "status": "completed",
                        "failure_reason": "",
                    }
                )
        except Exception as exc:  # retained QSVT failure
            failures += 1
            rows.append(
                {
                    **selected,
                    "status": "failed",
                    "failure_reason": f"qsvt_statevector_failure: {type(exc).__name__}: {exc}",
                    "common_design_fingerprint": design_payload[
                        "common_design_fingerprint"
                    ],
                    "per_support_phase_refit": False,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_write_csv(context.output_dir / "qsvt_validation_results.csv", frame)
    if failures:
        raise RuntimeError(f"qsvt_statevector_failure: {failures} supports failed")
    return {
        "supports": len(subset),
        "rows": len(frame),
        "degree": selected_target.degree,
        "common_beta": common_beta,
        "failures": failures,
    }


def stage_finite_shot(context: OutputAwareStudyContext) -> dict[str, Any]:
    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    completed = qsvt[qsvt["status"] == "completed"]
    unique_supports = completed.drop_duplicates("support_id")
    requested_supports = min(3, len(unique_supports))
    shots = [int(value) for value in context.config["finite_shot"]["shot_counts"]]
    seeds = int(context.config["finite_shot"]["seed_count"])
    projected_attempts = requested_supports * seeds * sum(shots)
    representative_gates = float(completed["gates_per_attempt"].median())
    projected_gates = projected_attempts * representative_gates
    reason = (
        "finite_shot_runtime_limit: optional common-wrapper campaign would require "
        f"{projected_attempts} attempted shots and approximately {projected_gates:.6e} "
        "measured-component gate applications; statevector validation completed instead"
    )
    rows: list[dict[str, Any]] = []
    for record in unique_supports.head(requested_supports).itertuples(index=False):
        for shot_count in shots:
            rows.append(
                {
                    "support_id": record.support_id,
                    "selector": record.selector,
                    "shots_attempted_requested_per_seed": shot_count,
                    "seed_count_requested": seeds,
                    "attempted_count": np.nan,
                    "postselected_count": np.nan,
                    "readout_accepted_count": np.nan,
                    "status": "skipped",
                    "failure_category": "finite_shot_runtime_limit",
                    "failure_reason": reason,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_write_csv(context.output_dir / "finite_shot_results.csv", frame)
    return {
        "status": "skipped",
        "verified_reason": reason,
        "projected_attempted_shots": projected_attempts,
        "projected_gate_applications": projected_gates,
    }


def _write_frontier_pair(
    output_dir: Path,
    stem: str,
    frame: pd.DataFrame,
    *,
    error_column: str,
    cost_column: str,
) -> tuple[int, int]:
    candidates, frontier = deterministic_pareto_frontier(
        frame,
        error_column=error_column,
        cost_column=cost_column,
        tie_columns=("support_id",),
    )
    candidates["accuracy_semantics"] = "held_out_absolute_error_no_signed_cancellation"
    frontier["accuracy_semantics"] = "held_out_absolute_error_no_signed_cancellation"
    atomic_write_csv(output_dir / f"pareto_candidates_{stem}.csv", candidates)
    atomic_write_csv(output_dir / f"pareto_frontier_{stem}.csv", frontier)
    return len(candidates), len(frontier)


def stage_pareto(context: OutputAwareStudyContext) -> dict[str, Any]:
    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    heldout = pd.read_csv(context.output_dir / "heldout_support_summary.csv")
    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    certificates = pd.read_csv(context.output_dir / "certificate_summary.csv")
    cert_heldout = certificates[certificates["split"] == "held_out"][
        [
            "support_id",
            "median_certificate_bound",
            "worst_certificate_bound",
            "certificate_coverage",
        ]
    ]
    resource_columns = resources[
        ["support_id", "signal_unitary_gate_count", "signal_unitary_depth", "cx_count"]
    ]
    candidates = registry.merge(
        heldout[
            [
                "support_id",
                "mean_absolute_error",
                "mean_normalized_error",
                "worst_absolute_error",
                "failure_fraction",
            ]
        ],
        on="support_id",
        how="left",
    ).merge(resource_columns, on="support_id", how="left")
    candidates = candidates.merge(cert_heldout, on="support_id", how="left")
    candidates["heldout_error_value"] = candidates["mean_absolute_error"]
    candidates["nonzeros_cost"] = pd.to_numeric(
        candidates["actual_nonzeros"], errors="coerce"
    )
    candidates["slots_cost"] = pd.to_numeric(candidates["slot_count"], errors="coerce")
    candidates["gates_cost"] = pd.to_numeric(
        candidates["signal_unitary_gate_count"], errors="coerce"
    )
    counts: dict[str, tuple[int, int]] = {}
    counts["error_nnz"] = _write_frontier_pair(
        context.output_dir,
        "error_nnz",
        candidates,
        error_column="heldout_error_value",
        cost_column="nonzeros_cost",
    )
    counts["error_slots"] = _write_frontier_pair(
        context.output_dir,
        "error_slots",
        candidates,
        error_column="heldout_error_value",
        cost_column="slots_cost",
    )
    counts["error_gates"] = _write_frontier_pair(
        context.output_dir,
        "error_gates",
        candidates,
        error_column="heldout_error_value",
        cost_column="gates_cost",
    )

    certificate_candidates = candidates.copy()
    certificate_candidates["certificate_error_value"] = certificate_candidates[
        "median_certificate_bound"
    ]
    counts["certificate_gates"] = _write_frontier_pair(
        context.output_dir,
        "certificate_gates",
        certificate_candidates,
        error_column="certificate_error_value",
        cost_column="gates_cost",
    )

    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    qsvt_completed = qsvt[qsvt["status"] == "completed"]
    qsvt_candidates = (
        qsvt_completed.groupby(
            ["support_id", "selector", "k_budget", "slot_budget", "role"],
            sort=True,
        )
        .agg(
            qsvt_error_value=("full_to_qsvt_selected_output_error", "mean"),
            qsvt_worst_error=("full_to_qsvt_selected_output_error", "max"),
            gates_per_attempt=("gates_per_attempt", "first"),
            postselection_probability=("postselection_probability", "first"),
            common_signal_call_count=("common_signal_call_count", "first"),
            phase_count=("phase_count", "first"),
        )
        .reset_index()
    )
    qsvt_candidates["support_id"] = qsvt_candidates["support_id"].astype(str)
    qsvt_candidates["executed_cost_value"] = qsvt_candidates["gates_per_attempt"]
    qsvt_candidates["status"] = "completed"
    qsvt_counts = _write_frontier_pair(
        context.output_dir,
        "qsvt_error_executed_cost",
        qsvt_candidates,
        error_column="qsvt_error_value",
        cost_column="executed_cost_value",
    )
    counts["qsvt_error_executed_cost"] = qsvt_counts
    return {
        name: {"candidates": value[0], "frontier": value[1]}
        for name, value in counts.items()
    }


def _compare_protected_snapshot(context: OutputAwareStudyContext) -> dict[str, Any]:
    snapshot_path = context.output_dir / "protected_path_snapshot.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    before = dict(payload["files"])
    after = _protected_file_snapshot(Path.cwd().resolve())
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    return {
        "files_before": len(before),
        "files_after": len(after),
        "changed_files": changed,
        "unchanged": not changed,
    }


def _verification_checks(context: OutputAwareStudyContext) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    split = json.loads((context.output_dir / "residual_split.json").read_text(encoding="utf-8"))
    training = set(int(value) for value in split["training_seed_ids"])
    heldout = set(int(value) for value in split["held_out_seed_ids"])
    checks["heldout_split_disjoint"] = not training.intersection(heldout)

    gradients = pd.read_csv(context.output_dir / "gradient_validation.csv")
    checks["gradient_max_epsilon"] = float(gradients["epsilon_grad"].max())
    checks["gradient_within_tolerance"] = checks["gradient_max_epsilon"] <= float(
        context.config["gradient_validation"]["relative_frobenius_tolerance"]
    )

    registry = pd.read_csv(context.output_dir / "support_registry.csv")
    constraint_failures: list[str] = []
    for record in registry[registry["status"] == "completed"].itertuples(index=False):
        support = load_support(context.output_dir, str(record.support_file))
        report = support_constraint_report(
            context.design.matrix,
            support,
            SupportConstraints(int(record.k_budget), int(record.slot_budget), True),
        )
        if not report["valid"]:
            constraint_failures.append(str(record.support_id))
    checks["support_constraint_failures"] = constraint_failures
    checks["support_constraints_hold"] = not constraint_failures
    checks["heldout_used_in_selection"] = bool(
        registry["selection_data_split"].astype(str).str.contains("held_out").any()
    )

    traces = pd.read_csv(context.output_dir / "refinement_traces.csv")
    accepted = traces[(traces["action"] == "one_swap") & traces["accepted"]]
    checks["refinement_accepted_swaps_strict"] = bool(
        (accepted["objective_before"] > accepted["objective_after"]).all()
    )
    checks["refinement_max_iterations_respected"] = bool(
        traces.groupby("support_id")["iteration"].max().max()
        <= int(context.config["refinement"]["max_iterations"])
    )

    certificates = pd.read_csv(context.output_dir / "certificate_registry.csv")
    checks["certificate_rows"] = len(certificates)
    checks["certificate_violations"] = int((~certificates["certificate_holds"]).sum())
    checks["certificate_actual_error_not_used"] = bool(
        (~certificates["certificate_computation_uses_actual_error"]).all()
    )
    checks["linear_prediction_separate"] = bool(
        (~certificates["linearized_prediction_is_certificate"]).all()
    )

    resources = pd.read_csv(context.output_dir / "resource_registry.csv")
    checks["resource_failures"] = int((resources["status"] != "completed").sum())
    checks["wrapper_reconstruction_max"] = float(
        resources["wrapper_reconstruction_error"].max()
    )

    qsvt = pd.read_csv(context.output_dir / "qsvt_validation_results.csv")
    qsvt_completed = qsvt[qsvt["status"] == "completed"]
    checks["qsvt_failures"] = int((qsvt["status"] != "completed").sum())
    checks["qsvt_common_design_count"] = int(
        qsvt_completed["common_design_fingerprint"].nunique()
    )
    checks["qsvt_per_support_refit"] = bool(qsvt_completed["per_support_phase_refit"].any())
    checks["qsvt_matrix_identity_holds"] = bool(
        qsvt_completed["ridge_and_qsvt_matrix_fingerprint"].notna().all()
    )

    protected = _compare_protected_snapshot(context)
    checks["protected_paths"] = protected
    required_boolean = (
        checks["heldout_split_disjoint"],
        checks["gradient_within_tolerance"],
        checks["support_constraints_hold"],
        not checks["heldout_used_in_selection"],
        checks["refinement_accepted_swaps_strict"],
        checks["refinement_max_iterations_respected"],
        checks["certificate_violations"] == 0,
        checks["certificate_actual_error_not_used"],
        checks["linear_prediction_separate"],
        checks["resource_failures"] == 0,
        checks["qsvt_failures"] == 0,
        checks["qsvt_common_design_count"] == 1,
        not checks["qsvt_per_support_refit"],
        checks["qsvt_matrix_identity_holds"],
        protected["unchanged"],
    )
    checks["all_internal_checks_pass"] = all(required_boolean)
    return checks


def _write_summary(context: OutputAwareStudyContext, checks: dict[str, Any]) -> None:
    design = json.loads(
        (context.output_dir / "common_qsvt_design.json").read_text(encoding="utf-8")
    )
    heldout = pd.read_csv(context.output_dir / "selector_comparison_summary.csv")
    best = heldout.sort_values(
        ["mean_normalized_error", "selector", "k_budget", "slot_budget"], kind="stable"
    ).head(12)
    certificate = pd.read_csv(context.output_dir / "certificate_summary.csv")
    cert_heldout = certificate[certificate["split"] == "held_out"]
    finite = pd.read_csv(context.output_dir / "finite_shot_results.csv")
    lines = [
        "# Output-Aware Sparse Selection Summary",
        "",
        "Controlled 8x8 IEEE-14-derived numerical and classical-simulator evidence only; ",
        "no IEEE-scale, field-data, hardware, speedup, advantage, or "
        "practical-competitiveness claim.",
        "",
        "## Frozen design",
        "",
        f"- Matrix fingerprint: `{stable_array_fingerprint(context.design.matrix)}`",
        f"- Physical alpha: `{context.design.alpha:.17g}`",
        f"- Training seeds: `{context.config['training_seed_ids']}`",
        f"- Held-out seeds: `{context.config['held_out_seed_ids']}`",
        f"- Common beta/lambda/C: `{design['common_beta']:.12g}` / "
        f"`{design['common_lambda']:.12g}` / `{design['common_C']:.12g}`",
        f"- Common degree/phases: `{design['degree']}` / `{design['phase_count']}`",
        "",
        "## Best held-out configurations (resource-matched rows remain in the registry)",
        "",
        "| selector | k | s | mean normalized | worst absolute | support count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.selector} | {int(row.k_budget)} | {int(row.slot_budget)} | "
            f"{row.mean_normalized_error:.6e} | {row.worst_absolute_error:.6e} | "
            f"{int(row.support_count)} |"
        )
    lines.extend(
        [
            "",
            "## Certificate",
            "",
            f"- Coverage: `{cert_heldout['certificate_coverage'].mean():.12g}`",
            f"- Median tightness across supports: "
            f"`{cert_heldout['median_tightness'].median():.6e}`",
            f"- Worst tightness: `{cert_heldout['worst_tightness'].max():.6e}`",
            "- The bound is mathematically valid in this campaign but is reported separately "
            "from the local first-order prediction.",
            "",
            "## Finite shot",
            "",
            f"- Status: `{finite['status'].iloc[0] if not finite.empty else 'not_recorded'}`",
            "- The optional stage records its measured-cost runtime projection and does not "
            "fabricate shot counts.",
            "",
            "## Verification",
            "",
            "- Internal campaign checks: "
            f"`{'PASS' if checks['all_internal_checks_pass'] else 'FAIL'}`",
            f"- Certificate violations: `{checks['certificate_violations']}`",
            f"- Resource failures: `{checks['resource_failures']}`",
            f"- Common-design QSVT failures: `{checks['qsvt_failures']}`",
            f"- Protected path changes during campaign: "
            f"`{len(checks['protected_paths']['changed_files'])}`",
        ]
    )
    commands = checks.get("executed_verification_commands", {})
    if commands:
        lines.extend(
            [
                f"- Ruff: `{commands['ruff']['status']}`",
                f"- Targeted tests: `{commands['targeted_tests']['passed']} passed`",
                f"- Dependent tests: `{commands['dependent_tests']['passed']} passed`",
                f"- Resumable filewise suite: "
                f"`{commands['filewise_suite']['files_passed']}/"
                f"{commands['filewise_suite']['files_passed']} files passed`",
                f"- Resume check: "
                f"`{commands['campaign_resume']['stages_resumed_from_checkpoint']} "
                "stages resumed, 0 recomputed`",
            ]
        )
    lines.append("")
    (context.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def stage_verify(context: OutputAwareStudyContext) -> dict[str, Any]:
    required = [
        "implementation_audit.md",
        "prior_pareto_audit.md",
        "study_configuration.json",
        "residual_split.json",
        "entry_scores.csv",
        "gradient_validation.csv",
        "support_registry.csv",
        "support_selection_results.csv",
        "refinement_traces.csv",
        "training_results.csv",
        "heldout_results.csv",
        "certificate_registry.csv",
        "certificate_summary.csv",
        "slot_validation.csv",
        "resource_registry.csv",
        "common_qsvt_design.json",
        "qsvt_validation_results.csv",
        "finite_shot_results.csv",
        "pareto_candidates_error_nnz.csv",
        "pareto_frontier_error_nnz.csv",
        "pareto_candidates_error_slots.csv",
        "pareto_frontier_error_slots.csv",
        "pareto_candidates_error_gates.csv",
        "pareto_frontier_error_gates.csv",
        "selector_comparison_summary.csv",
        "checkpoint.json",
    ]
    missing = [name for name in required if not (context.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"verification missing artifacts: {missing}")
    checks = _verification_checks(context)
    commands_path = context.output_dir / "verification_commands.json"
    if commands_path.is_file():
        checks["executed_verification_commands"] = json.loads(
            commands_path.read_text(encoding="utf-8")
        )
    _write_summary(context, checks)
    lines = [
        "# Output-Aware Sparse Selection Verification Report",
        "",
        f"Generated: `{now_iso()}`",
        "",
        f"Internal status: **{'PASS' if checks['all_internal_checks_pass'] else 'FAIL'}**",
        "",
        "```json",
        json.dumps(_json_ready(checks), indent=2, sort_keys=True),
        "```",
        "",
        "The analytical sensitivity is a local model, not a certificate. The separately",
        "computed operator perturbation certificate used no observed error in its bound.",
        "Existing manuscript, package, integrated-chain, and prior error-study artifacts",
        "were fingerprinted before the campaign and compared byte-for-byte here.",
        "",
    ]
    (context.output_dir / "verification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not checks["all_internal_checks_pass"]:
        raise RuntimeError("one or more internal verification checks failed")
    return checks


def refresh_manifest_and_checksums(output_dir: str | Path) -> None:
    root = Path(output_dir)
    artifacts = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"}
    )
    manifest = {
        "study_id": STUDY_ID,
        "generated_at": now_iso(),
        "git_commit": git_commit_hash(),
        "artifact_count_excluding_manifest_and_checksum_file": len(artifacts),
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifacts
        ],
        "claim_boundary": (
            "Controlled 8x8 evidence only; no IEEE-scale, hardware, field-data, speedup, "
            "advantage, or practical-competitiveness claim."
        ),
    }
    atomic_write_json(root / "manifest.json", manifest)
    checksum_targets = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_text = "\n".join(
        f"{_sha256_file(path)}  {path}" for path in checksum_targets
    ) + "\n"
    checksum_path = root / "checksums.sha256"
    temporary = checksum_path.with_name(f"{checksum_path.name}.tmp.{os.getpid()}")
    temporary.write_text(checksum_text, encoding="utf-8")
    os.replace(temporary, checksum_path)


STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "audit": (
        "implementation_audit.md",
        "study_configuration.json",
        "protected_path_snapshot.json",
    ),
    "pareto-audit": (
        "prior_pareto_audit.md",
        "prior_pareto_candidates_precision_cost_corrected.csv",
        "prior_pareto_frontier_precision_cost_corrected.csv",
        "prior_phase_rounding_sensitivity.csv",
    ),
    "residuals": ("residual_split.json",),
    "gradients": ("gradient_validation.csv",),
    "scores": ("entry_scores.csv", "entry_score_metadata.json"),
    "supports": ("support_registry.csv", "support_selection_results.csv"),
    "refinement": (
        "support_registry.csv",
        "support_selection_results.csv",
        "refinement_traces.csv",
        "training_results.csv",
    ),
    "certificates": ("certificate_registry.csv", "certificate_summary.csv"),
    "heldout": (
        "heldout_results.csv",
        "heldout_support_summary.csv",
        "selector_comparison_summary.csv",
    ),
    "resources": ("slot_validation.csv", "resource_registry.csv"),
    "qsvt": ("common_qsvt_design.json", "qsvt_validation_results.csv"),
    "finite-shot": ("finite_shot_results.csv",),
    "pareto": (
        "pareto_candidates_error_nnz.csv",
        "pareto_frontier_error_nnz.csv",
        "pareto_candidates_error_slots.csv",
        "pareto_frontier_error_slots.csv",
        "pareto_candidates_error_gates.csv",
        "pareto_frontier_error_gates.csv",
        "pareto_candidates_certificate_gates.csv",
        "pareto_frontier_certificate_gates.csv",
        "pareto_candidates_qsvt_error_executed_cost.csv",
        "pareto_frontier_qsvt_error_executed_cost.csv",
    ),
    "verify": ("summary.md", "verification_report.md"),
}

STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "pareto-audit": stage_pareto_audit,
    "residuals": stage_residuals,
    "gradients": stage_gradients,
    "scores": stage_scores,
    "supports": stage_supports,
    "refinement": stage_refinement,
    "certificates": stage_certificates,
    "heldout": stage_heldout,
    "resources": stage_resources,
    "qsvt": stage_qsvt,
    "finite-shot": stage_finite_shot,
    "pareto": stage_pareto,
    "verify": stage_verify,
}


def run_study(
    context: OutputAwareStudyContext, *, stage: str = "all"
) -> dict[str, dict[str, Any]]:
    if stage != "all" and stage not in STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage}; expected one of {('all', *STAGES)}")
    selected = list(STAGES) if stage == "all" else [stage]
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        if context.force:
            context.checkpoint.clear(name)
        if context.resume and context.checkpoint.is_complete(name):
            results[name] = {"status": "resumed_from_checkpoint"}
            continue
        started = time.perf_counter()
        result = STAGE_FUNCTIONS[name](context)
        elapsed = time.perf_counter() - started
        context.checkpoint.mark_complete(
            name,
            outputs=STAGE_OUTPUTS[name],
            result=result,
            elapsed_seconds=elapsed,
        )
        results[name] = {"status": "completed", "elapsed_seconds": elapsed, **result}
        if name == "verify":
            refresh_manifest_and_checksums(context.output_dir)
    return results
