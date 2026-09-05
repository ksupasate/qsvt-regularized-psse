"""Workstream 2 - strong exact-loss support-selection baselines.

Adds a deterministic exact-loss greedy selector, an exhaustive 4x4 oracle, and an 8x8
near-oracle beam search, and compares them against the existing magnitude / leverage /
sensitivity / refined / random selectors under the repository's normalized training loss

    E_norm^(t)(S) = |y_S^(t) - y^(t)| / max(|y^(t)|, y_floor).

Support selection uses training residuals only; held-out residuals are evaluated once,
at the end.  Infeasible budgets are retained, never dropped.  No result claims quantum
speedup, advantage, or superiority over the matched Ridge filter.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    _ridge_filter_operator,
    compute_output_aware_entry_scores,
    deterministic_ridge_leverage_scores,
    refine_support_one_swap,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    array_fingerprint,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    now_iso,
    provenance_block,
    write_manifest_and_checksums,
)

STUDY_ID = "reviewer_blocking_strong_baselines_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_reviewer_blocking_experiments/strong_baselines")
DEFAULT_CONFIG_PATH = Path("configs/tqe_reviewer_blocking/strong_baselines.json")

LEGACY_FUNCTIONALS = (
    "coordinate_e0",
    "signed_difference_e0_minus_e1",
    "aggregate_e0_to_e3",
)


# --------------------------------------------------------------- small designs


@dataclass(slots=True)
class SmallDesign:
    """A deterministic IEEE-14-derived block with a fixed Ridge alpha."""

    label: str
    dimension: int
    matrix: np.ndarray
    alpha: float
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    functionals: dict[str, np.ndarray]


def build_small_design(seed: int, dimension: int) -> SmallDesign:
    """Rebuild a ``dimension``x``dimension`` block; alpha = 4 * sigma_min_pos^2."""

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        predetermined_selected_functionals,
    )

    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    matrix, _rblock, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=dimension,
        col_count=dimension,
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    positive = singular[singular > 1.0e-10]
    if positive.size == 0:
        raise ValueError("small design has no positive singular values")
    alpha = 4.0 * float(positive.min()) ** 2
    functionals = predetermined_selected_functionals(dimension)
    return SmallDesign(
        label=f"ieee14_block_{dimension}x{dimension}_seed{seed}",
        dimension=dimension,
        matrix=np.asarray(matrix, dtype=np.float64),
        alpha=alpha,
        selected_rows=tuple(int(v) for v in rows),
        selected_columns=tuple(int(v) for v in cols),
        functionals=functionals,
    )


def frozen_8x8_design_as_small(output_dir: str | Path) -> SmallDesign:
    """Wrap the frozen campaign's 8x8 design (identical matrix + alpha) as a SmallDesign."""

    from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
        build_frozen_output_aware_design,
    )

    frozen = build_frozen_output_aware_design(output_dir)
    return SmallDesign(
        label="ieee14_sparse_frozen_8x8_seed123",
        dimension=8,
        matrix=np.asarray(frozen.matrix, dtype=np.float64),
        alpha=float(frozen.alpha),
        selected_rows=tuple(int(v) for v in frozen.selected_rows),
        selected_columns=tuple(int(v) for v in frozen.selected_columns),
        functionals={
            key: np.asarray(value, dtype=np.float64)
            for key, value in frozen.functionals.items()
        },
    )


def build_block_design(dimension: int, seed: int, output_dir: str | Path) -> SmallDesign:
    if dimension == 8:
        return frozen_8x8_design_as_small(output_dir)
    return build_small_design(seed, dimension)


def generate_block_residual(seed: int, selected_rows: tuple[int, ...]) -> np.ndarray:
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import generate_controlled_residual

    return generate_controlled_residual(seed, selected_rows=selected_rows)


def build_tasks(
    design: SmallDesign,
    seeds: list[int],
    split: str,
    functional_ids: tuple[str, ...],
) -> list[RidgeTask]:
    tasks: list[RidgeTask] = []
    for seed in seeds:
        residual = generate_block_residual(seed, design.selected_rows)
        for functional_id in functional_ids:
            tasks.append(
                RidgeTask(
                    task_id=f"{split}_seed{seed}_{functional_id}",
                    seed_id=int(seed),
                    split=split,
                    residual=np.asarray(residual, dtype=np.float64),
                    functional_id=functional_id,
                    functional=np.asarray(design.functionals[functional_id], dtype=np.float64),
                )
            )
    return tasks


# ------------------------------------------------------------- exact-loss core


@dataclass(slots=True)
class ExactLossEvaluator:
    """Normalized selected-output loss over a fixed task set; counts Ridge solves."""

    matrix: np.ndarray
    tasks: list[RidgeTask]
    alpha: float
    y_floor: float
    full_outputs: np.ndarray = field(init=False)
    solves: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.full_outputs = np.asarray(
            [
                task.functional @ ridge_svd_solution(self.matrix, task.residual, alpha=self.alpha)
                for task in self.tasks
            ],
            dtype=np.float64,
        )

    def normalized(self, support: np.ndarray) -> np.ndarray:
        sparse = np.where(support, self.matrix, 0.0)
        operator = _ridge_filter_operator(sparse, self.alpha)
        self.solves += 1
        sparse_outputs = np.asarray(
            [task.functional @ (operator @ task.residual) for task in self.tasks],
            dtype=np.float64,
        )
        return np.abs(sparse_outputs - self.full_outputs) / np.maximum(
            np.abs(self.full_outputs), self.y_floor
        )

    def loss(self, support: np.ndarray, objective: str) -> float:
        normalized = self.normalized(support)
        return float(np.mean(normalized)) if objective == "mean" else float(np.max(normalized))


def _candidate_positions(matrix: np.ndarray) -> np.ndarray:
    return np.argwhere(matrix != 0.0)


def _active_rows_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.flatnonzero(np.any(matrix != 0.0, axis=1)),
        np.flatnonzero(np.any(matrix != 0.0, axis=0)),
    )


def _feasible_additions(
    matrix: np.ndarray,
    support: np.ndarray,
    constraints: SupportConstraints,
    *,
    coverage_first: bool,
) -> list[tuple[int, int]]:
    candidate = matrix != 0.0
    row_deg = support.sum(axis=1)
    col_deg = support.sum(axis=0)
    if int(support.sum()) >= constraints.k_budget:
        return []
    active_rows, active_columns = _active_rows_columns(matrix)
    uncovered_rows = {int(r) for r in active_rows if row_deg[r] < 1}
    uncovered_columns = {int(c) for c in active_columns if col_deg[c] < 1}
    additions: list[tuple[int, int]] = []
    for row, column in np.argwhere(candidate & ~support):
        if row_deg[row] >= constraints.slot_budget or col_deg[column] >= constraints.slot_budget:
            continue
        if (
            coverage_first
            and (uncovered_rows or uncovered_columns)
            and int(row) not in uncovered_rows
            and int(column) not in uncovered_columns
        ):
            continue
        additions.append((int(row), int(column)))
    return additions


def exact_loss_greedy(
    evaluator: ExactLossEvaluator,
    constraints: SupportConstraints,
    *,
    objective: str,
) -> dict[str, Any]:
    """Deterministic coverage-first forward-addition exact-loss greedy selector."""

    started = time.perf_counter()
    matrix = evaluator.matrix
    support = np.zeros_like(matrix, dtype=bool)
    trace: list[dict[str, Any]] = []
    while int(support.sum()) < constraints.k_budget:
        additions = _feasible_additions(matrix, support, constraints, coverage_first=True)
        if not additions:
            break
        scored: list[tuple[float, int, int]] = []
        for row, column in additions:
            trial = support.copy()
            trial[row, column] = True
            scored.append((evaluator.loss(trial, objective), row, column))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        best_loss, best_row, best_column = scored[0]
        support[best_row, best_column] = True
        trace.append(
            {
                "step": len(trace) + 1,
                "added_row": best_row,
                "added_column": best_column,
                "loss_after": best_loss,
                "support_size": int(support.sum()),
            }
        )
    report = support_constraint_report(matrix, support, constraints)
    runtime = time.perf_counter() - started
    if not report["valid"]:
        return {
            "support": None,
            "status": "failed",
            "failure_reason": (
                "coverage_constraint_infeasible: " + ",".join(report["failure_reasons"])
            ),
            "final_loss": float("nan"),
            "trace": trace,
            "runtime_seconds": runtime,
            "algorithm": "coverage_first_forward_exact_loss_greedy",
        }
    return {
        "support": support,
        "status": "completed",
        "failure_reason": "",
        "final_loss": float(evaluator.loss(support, objective)),
        "trace": trace,
        "runtime_seconds": runtime,
        "algorithm": "coverage_first_forward_exact_loss_greedy",
    }


# -------------------------------------------------------------------- oracle


def _support_from_positions(
    shape: tuple[int, int], positions: tuple[tuple[int, int], ...]
) -> np.ndarray:
    support = np.zeros(shape, dtype=bool)
    for row, column in positions:
        support[row, column] = True
    return support


def enumerate_feasible_supports(matrix: np.ndarray, constraints: SupportConstraints):
    """Yield every feasible support (subset of candidate nonzeros) as a boolean array."""

    positions = [tuple(int(v) for v in pos) for pos in _candidate_positions(matrix)]
    active_rows, active_columns = _active_rows_columns(matrix)
    active_rows = set(int(r) for r in active_rows)
    active_columns = set(int(c) for c in active_columns)
    max_size = min(constraints.k_budget, len(positions))
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(positions, size):
            rows = [r for r, _ in subset]
            cols = [c for _, c in subset]
            if any(rows.count(r) > constraints.slot_budget for r in set(rows)):
                continue
            if any(cols.count(c) > constraints.slot_budget for c in set(cols)):
                continue
            if constraints.coverage_enabled and (
                not active_rows.issubset(rows) or not active_columns.issubset(cols)
            ):
                continue
            yield _support_from_positions(matrix.shape, subset)


def candidate_count_estimate(matrix: np.ndarray, constraints: SupportConstraints) -> int:
    """Upper bound on subsets of size <= k (ignoring degree/coverage pruning)."""

    from math import comb

    nnz = int(np.count_nonzero(matrix))
    return int(sum(comb(nnz, size) for size in range(1, min(constraints.k_budget, nnz) + 1)))


def exhaustive_oracle(
    evaluator: ExactLossEvaluator,
    constraints: SupportConstraints,
    *,
    objective: str,
    max_candidates: int,
) -> dict[str, Any]:
    """Exact argmin over all feasible supports when the estimate is tractable."""

    started = time.perf_counter()
    estimate = candidate_count_estimate(evaluator.matrix, constraints)
    if estimate > max_candidates:
        return {
            "support": None,
            "status": "skipped_intractable",
            "failure_reason": (
                f"candidate_count_estimate {estimate} exceeds max_candidates {max_candidates}"
            ),
            "candidate_count_estimate": estimate,
            "evaluated": 0,
            "final_loss": float("nan"),
            "runtime_seconds": time.perf_counter() - started,
            "algorithm": "exhaustive_oracle",
        }
    best: tuple[float, str, np.ndarray] | None = None
    evaluated = 0
    for support in enumerate_feasible_supports(evaluator.matrix, constraints):
        evaluated += 1
        loss = evaluator.loss(support, objective)
        fingerprint = array_fingerprint(support.astype(np.float64))
        if best is None or (loss, fingerprint) < (best[0], best[1]):
            best = (loss, fingerprint, support.copy())
    if best is None:
        return {
            "support": None,
            "status": "failed",
            "failure_reason": "no_feasible_support",
            "candidate_count_estimate": estimate,
            "evaluated": evaluated,
            "final_loss": float("nan"),
            "runtime_seconds": time.perf_counter() - started,
            "algorithm": "exhaustive_oracle",
        }
    return {
        "support": best[2],
        "status": "completed",
        "failure_reason": "",
        "candidate_count_estimate": estimate,
        "evaluated": evaluated,
        "final_loss": best[0],
        "runtime_seconds": time.perf_counter() - started,
        "algorithm": "exhaustive_oracle",
    }


def near_oracle_beam(
    evaluator: ExactLossEvaluator,
    constraints: SupportConstraints,
    *,
    objective: str,
    beam_width: int,
    max_steps: int,
) -> dict[str, Any]:
    """Deterministic coverage-first beam search; a diagnostic near-oracle."""

    started = time.perf_counter()
    matrix = evaluator.matrix
    empty = np.zeros_like(matrix, dtype=bool)
    beam: list[tuple[float, str, np.ndarray]] = [(evaluator.loss(empty, objective), "", empty)]
    best_complete: tuple[float, str, np.ndarray] | None = None
    steps = 0
    for _step in range(max_steps):
        steps += 1
        expansions: dict[str, tuple[float, np.ndarray]] = {}
        for _loss, _fp, support in beam:
            report = support_constraint_report(matrix, support, constraints)
            if report["valid"]:
                fingerprint = array_fingerprint(support.astype(np.float64))
                loss = evaluator.loss(support, objective)
                if best_complete is None or (loss, fingerprint) < (
                    best_complete[0],
                    best_complete[1],
                ):
                    best_complete = (loss, fingerprint, support.copy())
            for row, column in _feasible_additions(
                matrix, support, constraints, coverage_first=True
            ):
                child = support.copy()
                child[row, column] = True
                fingerprint = array_fingerprint(child.astype(np.float64))
                if fingerprint not in expansions:
                    expansions[fingerprint] = (evaluator.loss(child, objective), child)
        if not expansions:
            break
        ranked = sorted(
            ((loss, fp, support) for fp, (loss, support) in expansions.items()),
            key=lambda item: (item[0], item[1]),
        )
        beam = ranked[:beam_width]
    # Final sweep for completed supports left in the beam.
    for _loss, _fp, support in beam:
        report = support_constraint_report(matrix, support, constraints)
        if report["valid"]:
            fingerprint = array_fingerprint(support.astype(np.float64))
            loss = evaluator.loss(support, objective)
            if best_complete is None or (loss, fingerprint) < (best_complete[0], best_complete[1]):
                best_complete = (loss, fingerprint, support.copy())
    if best_complete is None:
        return {
            "support": None,
            "status": "failed",
            "failure_reason": "beam_found_no_feasible_support",
            "beam_width": beam_width,
            "steps": steps,
            "final_loss": float("nan"),
            "runtime_seconds": time.perf_counter() - started,
            "algorithm": "near_oracle_beam",
        }
    return {
        "support": best_complete[2],
        "status": "completed",
        "failure_reason": "",
        "beam_width": beam_width,
        "steps": steps,
        "final_loss": best_complete[0],
        "runtime_seconds": time.perf_counter() - started,
        "algorithm": "near_oracle_beam",
        "stopping_condition": (
            f"beam_width={beam_width}; deterministic top-B coverage-first expansion"
        ),
    }


def near_oracle_multistart(
    evaluator: ExactLossEvaluator,
    training_tasks: list[RidgeTask],
    constraints: SupportConstraints,
    *,
    objective: str,
    seed_supports: list[np.ndarray],
    beam_diagnostic: dict[str, Any],
    max_iterations: int = 6,
) -> dict[str, Any]:
    """Deterministic multi-start exact-loss one-swap local search.

    Polishes every feasible construction (magnitude / leverage / sensitivity / greedy /
    coverage-first beam) and keeps the training-loss minimizer.  By construction the
    result is no worse than any of its seeds, so it dominates the tested selectors and
    serves as a strong, honestly-labelled diagnostic near-oracle (not a proof of optimality).
    """

    started = time.perf_counter()
    refine_objective = (
        "mean_normalized_error" if objective == "mean" else "worst_case_normalized_error"
    )
    seen: set[str] = set()
    unique_seeds: list[np.ndarray] = []
    for support in seed_supports:
        if support is None:
            continue
        fingerprint = array_fingerprint(support.astype(np.float64))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_seeds.append(support)
    if not unique_seeds:
        return {
            "support": None,
            "status": "failed",
            "failure_reason": "no_feasible_seed_for_near_oracle",
            "final_loss": float("nan"),
            "runtime_seconds": time.perf_counter() - started,
            "algorithm": "near_oracle_multistart_local_search",
        }
    base_best = min(evaluator.loss(support, objective) for support in unique_seeds)
    best: tuple[float, str, np.ndarray] | None = None
    for support in unique_seeds:
        refined = refine_support_one_swap(
            evaluator.matrix,
            support,
            training_tasks,
            constraints,
            alpha=evaluator.alpha,
            y_floor=evaluator.y_floor,
            objective=refine_objective,
            max_iterations=max_iterations,
            improvement_tolerance=1e-12,
        )
        loss = evaluator.loss(refined.support, objective)
        fingerprint = array_fingerprint(refined.support.astype(np.float64))
        if best is None or (loss, fingerprint) < (best[0], best[1]):
            best = (loss, fingerprint, refined.support.copy())
    assert best is not None
    return {
        "support": best[2],
        "status": "completed",
        "failure_reason": "",
        "starts": len(unique_seeds),
        "base_best_construction_loss": float(base_best),
        "final_loss": float(best[0]),
        "search_gap_vs_best_construction": float(base_best - best[0]),
        "beam_final_loss": beam_diagnostic.get("final_loss"),
        "beam_width": beam_diagnostic.get("beam_width"),
        "runtime_seconds": time.perf_counter() - started,
        "algorithm": "near_oracle_multistart_local_search",
        "stopping_condition": (
            "multi-start one-swap local search over all feasible constructions + beam seed"
        ),
    }


# ---------------------------------------------------------------- selectors


def balanced_magnitude_score(matrix: np.ndarray) -> np.ndarray:
    values = np.abs(matrix)
    row_norm = np.linalg.norm(values, axis=1, keepdims=True)
    col_norm = np.linalg.norm(values, axis=0, keepdims=True)
    denom = np.sqrt(np.maximum(row_norm * col_norm, 1e-30))
    return values / denom


def _milp_support(
    matrix: np.ndarray, score: np.ndarray, constraints: SupportConstraints
) -> dict[str, Any]:
    result = select_resource_constrained_support(matrix, score, constraints)
    return {
        "support": result.support,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "solver_used": result.solver_used,
        "optimality_gap": result.optimality_gap,
        "runtime_seconds": result.runtime_seconds,
        "milp_solves": 1,
    }


def produce_supports(
    design: SmallDesign,
    training_tasks: list[RidgeTask],
    constraints: SupportConstraints,
    *,
    evaluator: ExactLossEvaluator,
    random_seed: int,
    beam_width: int,
    beam_max_steps: int,
    oracle_max_candidates: int,
    include_oracle: bool,
) -> dict[str, dict[str, Any]]:
    """Return {selector_name: support-record} for one (k, s) budget on training data."""

    matrix = design.matrix
    outcomes: dict[str, dict[str, Any]] = {}

    magnitude = np.abs(matrix)
    _row_lev, _col_lev, leverage = deterministic_ridge_leverage_scores(matrix, alpha=design.alpha)
    legacy_tasks = [task for task in training_tasks if task.functional_id in LEGACY_FUNCTIONALS]
    scores = compute_output_aware_entry_scores(
        matrix, legacy_tasks, alpha=design.alpha, epsilon=1e-15
    )
    rng = np.random.default_rng(random_seed)
    random_scores = np.zeros_like(matrix)
    random_scores[matrix != 0.0] = rng.random(int(np.count_nonzero(matrix)))

    outcomes["global_magnitude"] = _milp_support(matrix, magnitude, constraints)
    outcomes["balanced_magnitude"] = _milp_support(
        matrix, balanced_magnitude_score(matrix), constraints
    )
    outcomes["ridge_leverage"] = _milp_support(matrix, leverage, constraints)
    outcomes["random_feasible"] = _milp_support(matrix, random_scores, constraints)
    outcomes["sensitivity_initial_mean"] = _milp_support(
        matrix, scores.sensitivity_mean, constraints
    )
    outcomes["sensitivity_initial_worst_case"] = _milp_support(
        matrix, scores.sensitivity_worst_case, constraints
    )

    for initial_name, refined_name, objective in (
        ("sensitivity_initial_mean", "sensitivity_refined_mean", "mean_normalized_error"),
        (
            "sensitivity_initial_worst_case",
            "sensitivity_refined_worst_case",
            "worst_case_normalized_error",
        ),
    ):
        initial = outcomes[initial_name]
        if initial["support"] is None:
            outcomes[refined_name] = {
                "support": None,
                "status": "failed",
                "failure_reason": "initial_support_unavailable",
                "runtime_seconds": 0.0,
                "milp_solves": 0,
            }
            continue
        started = time.perf_counter()
        refined = refine_support_one_swap(
            matrix,
            initial["support"],
            legacy_tasks,
            constraints,
            alpha=design.alpha,
            y_floor=evaluator.y_floor,
            objective=objective,
            max_iterations=8,
            improvement_tolerance=1e-12,
        )
        outcomes[refined_name] = {
            "support": refined.support,
            "status": "completed",
            "failure_reason": "",
            "runtime_seconds": time.perf_counter() - started,
            "refinement_iterations_accepted": refined.iterations_accepted,
            "milp_solves": 0,
        }

    for objective, name in (
        ("mean", "exact_loss_greedy_mean"),
        ("worst", "exact_loss_greedy_worst"),
    ):
        outcomes[name] = exact_loss_greedy(evaluator, constraints, objective=objective)

    if include_oracle:
        for objective, name in (("mean", "oracle_mean"), ("worst", "oracle_worst")):
            outcomes[name] = exhaustive_oracle(
                evaluator, constraints, objective=objective, max_candidates=oracle_max_candidates
            )
    # Strong near-oracle: multi-start one-swap local search over every feasible construction
    # (plus a coverage-first beam seed). Dominates its seeds, so it is a meaningful reference.
    construction_seeds = [
        outcome["support"]
        for outcome in outcomes.values()
        if outcome.get("support") is not None and outcome.get("status") == "completed"
    ]
    for objective, name in (("mean", "near_oracle_mean"), ("worst", "near_oracle_worst")):
        beam = near_oracle_beam(
            evaluator,
            constraints,
            objective=objective,
            beam_width=beam_width,
            max_steps=beam_max_steps,
        )
        seeds = list(construction_seeds)
        if beam.get("support") is not None:
            seeds.append(beam["support"])
        outcomes[name] = near_oracle_multistart(
            evaluator,
            training_tasks,
            constraints,
            objective=objective,
            seed_supports=seeds,
            beam_diagnostic=beam,
        )
    return outcomes


# ---------------------------------------------------------------- evaluation


def _support_entry_set(support: np.ndarray | None) -> frozenset[tuple[int, int]]:
    if support is None:
        return frozenset()
    return frozenset((int(r), int(c)) for r, c in np.argwhere(support))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_support(
    design: SmallDesign,
    support: np.ndarray,
    tasks: list[RidgeTask],
) -> list[dict[str, Any]]:
    sparse = np.where(support, design.matrix, 0.0)
    full_operator = _ridge_filter_operator(design.matrix, design.alpha)
    sparse_operator = _ridge_filter_operator(sparse, design.alpha)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        full_output = float(task.functional @ (full_operator @ task.residual))
        sparse_output = float(task.functional @ (sparse_operator @ task.residual))
        signed = sparse_output - full_output
        rows.append(
            {
                "task_id": task.task_id,
                "seed_id": task.seed_id,
                "split": task.split,
                "functional_id": task.functional_id,
                "full_ridge_output": full_output,
                "sparse_ridge_output": sparse_output,
                "signed_error": signed,
                "absolute_error": abs(signed),
                "normalized_error": abs(signed) / max(abs(full_output), 1e-6),
            }
        )
    return rows


# --------------------------------------------------------------- orchestrator


def _oracle_reference_name(outcomes: dict[str, dict[str, Any]], objective: str) -> str:
    oracle_key = f"oracle_{objective}"
    if oracle_key in outcomes and outcomes[oracle_key]["support"] is not None:
        return oracle_key
    return f"near_oracle_{objective}"


def run_strong_baselines(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected study_id: {config.get('study_id')}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("matrix_seed", 123))
    y_floor = float(config.get("y_floor", 1e-6))
    functional_ids = tuple(config.get("functionals", LEGACY_FUNCTIONALS))
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    heldout_seeds = [int(s) for s in config["held_out_seed_ids"]]
    if set(training_seeds) & set(heldout_seeds):
        raise ValueError("training and held-out seeds must be disjoint")
    beam_width = int(config.get("beam_width", 8))
    beam_max_steps = int(config.get("beam_max_steps", 40))
    oracle_max_candidates = int(config.get("oracle_max_candidates", 3_000_000))
    random_seed_base = int(config.get("random_seed_base", 314159))

    support_records: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    oracle_gap_rows: list[dict[str, Any]] = []
    support_paths: dict[str, list[list[int]]] = {}

    for block in config["blocks"]:
        dimension = int(block["dimension"])
        include_oracle = bool(block.get("include_exhaustive_oracle", dimension <= 4))
        design = build_block_design(dimension, seed, destination)
        training_tasks = build_tasks(design, training_seeds, "training", functional_ids)
        heldout_tasks = build_tasks(design, heldout_seeds, "held_out", functional_ids)
        evaluator = ExactLossEvaluator(design.matrix, training_tasks, design.alpha, y_floor)
        budgets = [
            SupportConstraints(int(k), int(s), True)
            for k in block["support_budgets"]
            for s in block["slot_budgets"]
        ]
        pair_index = 0
        for constraints in budgets:
            evaluator.solves = 0
            outcomes = produce_supports(
                design,
                training_tasks,
                constraints,
                evaluator=evaluator,
                random_seed=random_seed_base + 1000 * pair_index,
                beam_width=beam_width,
                beam_max_steps=beam_max_steps,
                oracle_max_candidates=oracle_max_candidates,
                include_oracle=include_oracle,
            )
            pair_index += 1
            oracle_entry_sets: dict[str, frozenset] = {}
            for objective in ("mean", "worst"):
                ref = _oracle_reference_name(outcomes, objective)
                oracle_entry_sets[objective] = _support_entry_set(outcomes[ref]["support"])

            for selector, outcome in outcomes.items():
                support = outcome.get("support")
                feasible = support is not None and outcome.get("status") == "completed"
                objective = "worst" if "worst" in selector else "mean"
                support_id = (
                    f"{design.label}_{selector}_k{constraints.k_budget}_s{constraints.slot_budget}"
                )
                entry_set = _support_entry_set(support)
                record: dict[str, Any] = {
                    "block": design.label,
                    "dimension": dimension,
                    "selector": selector,
                    "k_budget": constraints.k_budget,
                    "slot_budget": constraints.slot_budget,
                    "support_id": support_id,
                    "status": outcome.get("status", "unknown"),
                    "feasible": bool(feasible),
                    "failure_reason": outcome.get("failure_reason", ""),
                    "algorithm": outcome.get("algorithm", "milp_or_refine"),
                    "solver_used": outcome.get("solver_used", ""),
                    "selection_runtime_seconds": float(outcome.get("runtime_seconds", 0.0)),
                    "milp_solves": int(outcome.get("milp_solves", 0)),
                    "optimality_gap": outcome.get("optimality_gap"),
                    "candidate_count_estimate": outcome.get("candidate_count_estimate"),
                    "evaluated_supports": outcome.get("evaluated"),
                    "beam_width": outcome.get("beam_width"),
                    "actual_nonzeros": int(support.sum()) if support is not None else np.nan,
                    "support_fingerprint": (
                        array_fingerprint(support.astype(np.float64)) if support is not None else ""
                    ),
                    "training_selection_only": True,
                }
                if feasible:
                    support_paths[support_id] = [[int(r), int(c)] for r, c in np.argwhere(support)]
                    training_norm = evaluator.normalized(support)
                    record["training_mean_normalized_error"] = float(np.mean(training_norm))
                    record["training_worst_normalized_error"] = float(np.max(training_norm))
                    heldout_eval = evaluate_support(design, support, heldout_tasks)
                    combined_eval = (
                        evaluate_support(design, support, training_tasks) + heldout_eval
                    )
                    task_rows.extend(
                        row
                        | {
                            "selector": selector,
                            "support_id": support_id,
                            "block": design.label,
                        }
                        for row in combined_eval
                    )
                    heldout_norm = np.asarray([r["normalized_error"] for r in heldout_eval])
                    heldout_abs = np.asarray([r["absolute_error"] for r in heldout_eval])
                    record["heldout_mean_normalized_error"] = float(np.mean(heldout_norm))
                    record["heldout_worst_normalized_error"] = float(np.max(heldout_norm))
                    record["heldout_worst_absolute_error"] = float(np.max(heldout_abs))
                    record["heldout_failure_fraction"] = float(
                        np.mean(heldout_norm > float(config.get("failure_threshold", 0.1)))
                    )
                    record["overlap_with_oracle"] = _jaccard(
                        entry_set, oracle_entry_sets[objective]
                    )
                else:
                    for column in (
                        "training_mean_normalized_error",
                        "training_worst_normalized_error",
                        "heldout_mean_normalized_error",
                        "heldout_worst_normalized_error",
                        "heldout_worst_absolute_error",
                        "heldout_failure_fraction",
                        "overlap_with_oracle",
                    ):
                        record[column] = np.nan
                support_records.append(record)

            # Oracle-gap rows: each selector vs the (near-)oracle at this budget/objective.
            for objective in ("mean", "worst"):
                ref = _oracle_reference_name(outcomes, objective)
                ref_outcome = outcomes[ref]
                if ref_outcome["support"] is None:
                    continue
                ref_loss = float(ref_outcome["final_loss"])
                for selector, outcome in outcomes.items():
                    if outcome.get("support") is None or outcome.get("status") != "completed":
                        continue
                    if ("worst" in selector) != (objective == "worst"):
                        continue
                    selector_loss = evaluator.loss(outcome["support"], objective)
                    oracle_gap_rows.append(
                        {
                            "block": design.label,
                            "dimension": dimension,
                            "objective": objective,
                            "k_budget": constraints.k_budget,
                            "slot_budget": constraints.slot_budget,
                            "reference": ref,
                            "reference_kind": (
                                "exhaustive_oracle"
                                if ref.startswith("oracle")
                                else "near_oracle_multistart_local_search"
                            ),
                            "selector": selector,
                            "selector_training_loss": selector_loss,
                            "reference_training_loss": ref_loss,
                            "optimality_gap": selector_loss - ref_loss,
                            "support_overlap_jaccard": _jaccard(
                                _support_entry_set(outcome["support"]),
                                _support_entry_set(ref_outcome["support"]),
                            ),
                            "candidate_count_estimate": ref_outcome.get("candidate_count_estimate"),
                            "beam_width": ref_outcome.get("beam_width"),
                        }
                    )

    support_frame = pd.DataFrame(support_records)
    atomic_write_csv(destination / "raw_support_records.csv", support_frame)
    atomic_write_csv(destination / "raw_task_results.csv", pd.DataFrame(task_rows))
    atomic_write_csv(destination / "oracle_gap_summary.csv", pd.DataFrame(oracle_gap_rows))
    atomic_write_json(destination / "support_paths.json", support_paths)

    selector_summary = _summarize_selectors(support_frame)
    atomic_write_csv(destination / "selector_summary.csv", selector_summary)
    runtime_summary = (
        support_frame.groupby("selector", sort=True)
        .agg(
            budgets=("support_id", "count"),
            feasible=("feasible", "sum"),
            total_selection_runtime_seconds=("selection_runtime_seconds", "sum"),
            total_milp_solves=("milp_solves", "sum"),
        )
        .reset_index()
    )
    atomic_write_csv(destination / "runtime_summary.csv", runtime_summary)

    _write_baseline_protocol(destination, config, support_frame)
    atomic_write_json(
        destination / "provenance.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID},
    )
    write_manifest_and_checksums(
        destination,
        study_id=STUDY_ID,
        extra={
            "support_records": len(support_frame),
            "task_rows": len(task_rows),
            "selectors": sorted(support_frame["selector"].unique().tolist())
            if not support_frame.empty
            else [],
        },
    )
    return {
        "support_records": len(support_frame),
        "feasible_supports": int(support_frame["feasible"].sum()) if not support_frame.empty else 0,
        "task_rows": len(task_rows),
        "oracle_gap_rows": len(oracle_gap_rows),
        "blocks": [int(block["dimension"]) for block in config["blocks"]],
    }


def _summarize_selectors(support_frame: pd.DataFrame) -> pd.DataFrame:
    if support_frame.empty:
        return support_frame
    grouped = support_frame.groupby(["block", "selector"], sort=True)
    summary = grouped.agg(
        budgets=("support_id", "count"),
        feasible=("feasible", "sum"),
        feasibility_rate=("feasible", "mean"),
        mean_training_normalized_error=("training_mean_normalized_error", "mean"),
        mean_heldout_normalized_error=("heldout_mean_normalized_error", "mean"),
        median_heldout_normalized_error=("heldout_mean_normalized_error", "median"),
        worst_heldout_absolute_error=("heldout_worst_absolute_error", "max"),
        mean_heldout_failure_fraction=("heldout_failure_fraction", "mean"),
        mean_overlap_with_oracle=("overlap_with_oracle", "mean"),
        total_runtime_seconds=("selection_runtime_seconds", "sum"),
        total_milp_solves=("milp_solves", "sum"),
    ).reset_index()
    return summary


def _write_baseline_protocol(
    destination: Path, config: dict[str, Any], support_frame: pd.DataFrame
) -> None:
    selectors = (
        sorted(support_frame["selector"].unique().tolist())
        if not support_frame.empty
        else []
    )
    lines = [
        "# Strong Exact-Loss Baseline Protocol",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Loss and data isolation",
        "",
        "All selectors are compared under the repository normalized training loss "
        "`E_norm = |y_S - y| / max(|y|, y_floor)`.  Support selection consumes training "
        "residuals only; held-out residuals are evaluated once, after selection.  Infeasible "
        "budgets are retained with their failure reason.",
        "",
        "## Selectors compared",
        "",
    ]
    lines += [f"- `{name}`" for name in selectors]
    lines += [
        "",
        "## Exact-loss algorithms (new)",
        "",
        "- **exact_loss_greedy_{mean,worst}** - deterministic coverage-first forward-addition "
        "greedy that, at each step, adds the feasible candidate minimizing the exact training "
        "loss (ties broken by flat index). Respects k, per-row/column slot budget s, and coverage.",
        "- **oracle_{mean,worst}** - exhaustive argmin over every feasible support (subsets of "
        "candidate nonzeros satisfying k, s, and coverage) for tractable blocks (4x4).",
        "- **near_oracle_{mean,worst}** - deterministic coverage-first beam search (width B) used "
        "where exhaustive enumeration is intractable (8x8); labelled near-oracle, with beam width "
        "as the stopping condition.",
        "",
        "## Reproducibility",
        "",
        f"- Blocks: `{[b['dimension'] for b in config['blocks']]}`",
        f"- Training seeds: `{config['training_seed_ids']}`",
        f"- Held-out seeds: `{config['held_out_seed_ids']}`",
        f"- Beam width: `{config.get('beam_width', 8)}`; "
        f"max steps: `{config.get('beam_max_steps', 40)}`",
        f"- Oracle max candidate estimate: `{config.get('oracle_max_candidates', 3_000_000)}`",
        f"- Generated: `{now_iso()}`",
        "",
    ]
    (destination / "baseline_protocol.md").write_text("\n".join(lines), encoding="utf-8")

