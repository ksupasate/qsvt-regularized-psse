"""Shared engine for the reviewer-blocking evidence pass.

Structure registry, per-seed physical reference, selector dispatch (reusing the frozen selector
primitives plus the two new task-aware baselines), design-time alpha regimes, and the coherent
``{y_true, y_full_Ridge, y_sparse}`` metric computation.  Physical error (vs the true update) and
support-fidelity error (vs full-block Ridge) are always kept in separate columns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from robust_qsvt_se.cross_case_validation.common import (
    build_case_design,
    build_case_full_system,
    generate_case_residual,
)
from robust_qsvt_se.cross_case_validation.selectors import (
    select_support_generalized,
)
from robust_qsvt_se.paper.tqe_revision_core import (
    select_alpha_discrepancy,
    select_alpha_gcv,
    select_alpha_heldout,
    select_alpha_l_curve,
    select_alpha_oracle_rmse,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    _ridge_filter_operator,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.qsvt.ridge_output_certificate import ridge_selected_output_gradient

# structure_id -> (case_name, dimension, uses_frozen_manuscript_alpha)
STRUCTURES: dict[str, tuple[str, int, bool]] = {
    "ieee14_8x8": ("ieee14", 8, True),
    "ieee30_8x8": ("ieee30", 8, False),
    "ieee14_16x16": ("ieee14", 16, False),
}

MATRIX_SEED = 123
PHYSICAL_FLOOR = 1e-6
SUPPORT_FLOOR = 1e-6


@dataclass(slots=True)
class StructureContext:
    structure_id: str
    case_name: str
    dimension: int
    matrix: np.ndarray  # frozen block (seed 123)
    alpha_fixed: float  # fixed-benchmark alpha
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    functionals: dict[str, np.ndarray]  # id -> unit vector (physical + legacy)
    functional_family: dict[str, str]
    physical_ids: tuple[str, ...]
    legacy_ids: tuple[str, ...]
    unavailable: list[Any]
    conditioning: dict[str, Any]
    design: Any  # CaseDesign.small (SmallDesign) for selector reuse


@lru_cache(maxsize=8)
def build_structure(structure_id: str) -> StructureContext:
    case_name, dimension, use_frozen_alpha = STRUCTURES[structure_id]
    case_design = build_case_design(case_name, MATRIX_SEED, dimension=dimension)
    small = case_design.small
    alpha_fixed = float(small.alpha)
    if use_frozen_alpha:
        from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
            build_frozen_output_aware_design,
        )

        # Match the manuscript's frozen 8x8 design alpha (a QSVT-feasible operating point),
        # not the cross-case 4*sigma_min^2 convention.
        frozen = build_frozen_output_aware_design(
            "outputs/reviewer_blocking_tqe_evidence/_frozen_cache"
        )
        if not np.allclose(frozen.matrix, small.matrix):
            raise RuntimeError("frozen matrix drift for ieee14_8x8")
        alpha_fixed = float(frozen.alpha)
        small = replace(small, alpha=alpha_fixed)
    family = {rec.functional_id: rec.family for rec in case_design.functional_records}
    return StructureContext(
        structure_id=structure_id,
        case_name=case_name,
        dimension=dimension,
        matrix=np.asarray(small.matrix, dtype=np.float64),
        alpha_fixed=alpha_fixed,
        selected_rows=small.selected_rows,
        selected_columns=small.selected_columns,
        functionals={k: np.asarray(v, dtype=np.float64) for k, v in small.functionals.items()},
        functional_family=family,
        physical_ids=case_design.physical_functional_ids,
        legacy_ids=case_design.legacy_functional_ids,
        unavailable=list(case_design.unavailable),
        conditioning=dict(case_design.conditioning),
        design=small,
    )


# ------------------------------------------------------------- per-seed reference


@lru_cache(maxsize=4096)
def _seed_reference(structure_id: str, seed: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ctx = build_structure(structure_id)
    residual = generate_case_residual(ctx.case_name, int(seed), ctx.selected_rows)
    full = build_case_full_system(ctx.case_name, int(seed))
    cols = np.asarray(ctx.selected_columns, dtype=np.int64)
    delta_true_block = np.asarray(full.x_true, dtype=np.float64)[cols]
    return tuple(float(v) for v in residual), tuple(float(v) for v in delta_true_block)


def seed_reference(structure_id: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(residual_block(seed), delta_true_block(seed))`` — per-seed physical reference."""

    residual, delta_true = _seed_reference(structure_id, int(seed))
    return np.asarray(residual, dtype=np.float64), np.asarray(delta_true, dtype=np.float64)


def build_tasks(structure_id: str, seeds: list[int], split: str) -> list[RidgeTask]:
    ctx = build_structure(structure_id)
    tasks: list[RidgeTask] = []
    for seed in seeds:
        residual, _ = seed_reference(structure_id, seed)
        for fid, vector in ctx.functionals.items():
            tasks.append(
                RidgeTask(
                    task_id=f"{split}_seed{seed}_{fid}",
                    seed_id=int(seed),
                    split=split,
                    residual=residual,
                    functional_id=fid,
                    functional=np.asarray(vector, dtype=np.float64),
                )
            )
    return tasks


# ------------------------------------------------------------- alpha regimes


def alpha_grid(
    ctx: StructureContext, points: int = 13, lo: float = 1e-3, hi: float = 1e1
) -> np.ndarray:
    singular = np.linalg.svd(ctx.matrix, compute_uv=False)
    positive = singular[singular > 1e-10]
    smin2 = float(positive.min()) ** 2
    smax2 = float(positive.max()) ** 2
    grid = np.logspace(np.log10(smin2 * lo), np.log10(smax2 * hi), points)
    return np.unique(np.concatenate([grid, [ctx.alpha_fixed]]))


def design_time_alpha_regimes(ctx: StructureContext) -> list[dict[str, Any]]:
    """Alpha regimes selected at design time (matrix seed 123) — no held-out leakage.

    Deployable methods (GCV, L-curve, discrepancy, held-out rows) and one oracle diagnostic
    (best block-update RMSE, uses ground truth) plus the fixed manuscript/benchmark alpha.
    """

    grid = alpha_grid(ctx)
    residual, delta_true = seed_reference(ctx.structure_id, MATRIX_SEED)
    regimes: list[dict[str, Any]] = [
        {
            "regime": "fixed_benchmark",
            "alpha": float(ctx.alpha_fixed),
            "deployable": True,
            "is_oracle": False,
            "note": "manuscript frozen QSVT-feasible alpha"
            if ctx.structure_id == "ieee14_8x8"
            else "4*sigma_min_pos^2 design alpha",
        },
    ]
    deployable = {
        "gcv": select_alpha_gcv,
        "l_curve": select_alpha_l_curve,
        "discrepancy": select_alpha_discrepancy,
        "heldout_rows": select_alpha_heldout,
    }
    for name, fn in deployable.items():
        try:
            alpha = float(fn(ctx.matrix, residual, grid))
            regimes.append(
                {
                    "regime": name,
                    "alpha": alpha,
                    "deployable": True,
                    "is_oracle": False,
                    "note": "design-time (matrix seed 123) block selection",
                }
            )
        except Exception as exc:  # retained as a failure regime, never silently dropped
            regimes.append(
                {
                    "regime": name,
                    "alpha": float("nan"),
                    "deployable": True,
                    "is_oracle": False,
                    "note": f"unavailable: {type(exc).__name__}: {exc}",
                }
            )
    try:
        oracle = float(select_alpha_oracle_rmse(ctx.matrix, residual, delta_true, grid))
        regimes.append(
            {
                "regime": "oracle_block_rmse",
                "alpha": oracle,
                "deployable": False,
                "is_oracle": True,
                "note": "diagnostic: minimizes block-update RMSE vs truth",
            }
        )
    except Exception as exc:
        regimes.append(
            {
                "regime": "oracle_block_rmse",
                "alpha": float("nan"),
                "deployable": False,
                "is_oracle": True,
                "note": f"unavailable: {type(exc).__name__}: {exc}",
            }
        )
    return regimes


# ------------------------------------------------------------- selectors


def adjoint_unnormalized_score(
    matrix: np.ndarray, tasks: list[RidgeTask], *, alpha: float
) -> np.ndarray:
    """Baseline A: w_ij = (1/T) sum_t |H_ij * [d y_t / d H]_ij| (no per-task normalization).

    Uses the exact same ``ridge_selected_output_gradient`` as the proposed selector; the ONLY
    difference from ``sensitivity_initial_mean`` is the omission of the per-task L1 normalization.
    """

    if not tasks or any(t.split != "training" for t in tasks):
        raise ValueError("data leakage guard: adjoint score requires training tasks only")
    values = np.asarray(matrix, dtype=np.float64)
    acc = np.zeros_like(values)
    for task in tasks:
        gradient = ridge_selected_output_gradient(values, task.residual, task.functional, alpha)
        acc += np.abs(values * gradient)
    return acc / float(len(tasks))


def exact_single_removal_score(
    matrix: np.ndarray,
    tasks: list[RidgeTask],
    *,
    alpha: float,
    y_floor: float,
) -> tuple[np.ndarray, int]:
    """Baseline B: exact training-loss increase when each single entry is removed from full support.

    For each candidate nonzero (i,j): drop only that entry, keep all others, recompute the exact
    normalized selected-output loss over the frozen training tasks, and score the entry by the
    resulting loss increase.  Higher score = more important entry (removal hurts more).  Returns
    the score matrix and the number of exact Ridge operator evaluations performed.
    """

    if not tasks or any(t.split != "training" for t in tasks):
        raise ValueError("data leakage guard: exact-removal score requires training tasks only")
    values = np.asarray(matrix, dtype=np.float64)
    full_support = values != 0.0
    full_op = _ridge_filter_operator(values, alpha)
    residuals = np.stack([t.residual for t in tasks])
    funcs = np.stack([t.functional for t in tasks])
    full_out = np.einsum("tf,fg,tg->t", funcs, full_op, residuals)
    base_loss = _mean_norm_loss(full_out, full_out, y_floor)  # 0 by construction, keeps shape

    score = np.zeros_like(values)
    evals = 0
    positions = np.argwhere(full_support)
    for i, j in positions:
        trial = full_support.copy()
        trial[i, j] = False
        op = _ridge_filter_operator(np.where(trial, values, 0.0), alpha)
        evals += 1
        out = np.einsum("tf,fg,tg->t", funcs, op, residuals)
        loss = _mean_norm_loss(out, full_out, y_floor)
        score[i, j] = float(loss - base_loss)
    return score, evals


def _mean_norm_loss(sparse_out: np.ndarray, full_out: np.ndarray, y_floor: float) -> float:
    denom = np.maximum(np.abs(full_out), y_floor)
    return float(np.mean(np.abs(sparse_out - full_out) / denom))


def select_support(
    ctx: StructureContext,
    selector: str,
    constraints: SupportConstraints,
    training_tasks: list[RidgeTask],
    score_functional_ids: tuple[str, ...],
    *,
    y_floor: float = PHYSICAL_FLOOR,
) -> dict[str, Any]:
    """Return a support record for one selector, reusing frozen primitives where possible."""

    started = time.perf_counter()
    if selector == "full_support":
        support = ctx.matrix != 0.0
        return {
            "support": support,
            "status": "completed",
            "failure_reason": "",
            "runtime_seconds": time.perf_counter() - started,
            "extra_evals": 0,
        }
    if selector in {
        "global_magnitude",
        "balanced_magnitude",
        "ridge_leverage",
        "sensitivity_initial_mean",
        "sensitivity_refined_mean",
    }:
        support = select_support_generalized(
            ctx.design, selector, constraints, training_tasks, score_functional_ids
        )
        status = "completed" if support is not None else "failed"
        reason = "" if support is not None else "milp_infeasible_or_none"
        return {
            "support": support,
            "status": status,
            "failure_reason": reason,
            "runtime_seconds": time.perf_counter() - started,
            "extra_evals": 0,
        }
    score_tasks = [t for t in training_tasks if t.functional_id in set(score_functional_ids)]
    if selector == "adjoint_unnormalized_mean":
        score = adjoint_unnormalized_score(ctx.matrix, score_tasks, alpha=ctx.alpha_fixed)
        support = select_resource_constrained_support(ctx.matrix, score, constraints).support
        return {
            "support": support,
            "status": "completed" if support is not None else "failed",
            "failure_reason": "" if support is not None else "milp_infeasible_or_none",
            "runtime_seconds": time.perf_counter() - started,
            "extra_evals": 0,
        }
    if selector == "exact_single_removal_mean":
        score, evals = exact_single_removal_score(
            ctx.matrix, score_tasks, alpha=ctx.alpha_fixed, y_floor=y_floor
        )
        support = select_resource_constrained_support(ctx.matrix, score, constraints).support
        return {
            "support": support,
            "status": "completed" if support is not None else "failed",
            "failure_reason": "" if support is not None else "milp_infeasible_or_none",
            "runtime_seconds": time.perf_counter() - started,
            "extra_evals": int(evals),
        }
    raise ValueError(f"unknown selector {selector}")


# ------------------------------------------------------------- metrics


def evaluate_triples(
    ctx: StructureContext,
    support: np.ndarray,
    alpha: float,
    seeds: list[int],
    functional_ids: list[str],
) -> list[dict[str, Any]]:
    """Coherent {y_true, y_full, y_sparse} + separated physical / support-fidelity metrics.

    All three outputs use the identical frozen block matrix and per-seed residual, so E_support
    reproduces the manuscript metric and E_physical is layered on the same outputs.
    """

    values = ctx.matrix
    full_op = _ridge_filter_operator(values, alpha)
    sparse_op = _ridge_filter_operator(np.where(support, values, 0.0), alpha)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        residual, delta_true = seed_reference(ctx.structure_id, seed)
        x_full = full_op @ residual
        x_sparse = sparse_op @ residual
        for fid in functional_ids:
            ell = ctx.functionals[fid]
            y_true = float(ell @ delta_true)
            y_full = float(ell @ x_full)
            y_sparse = float(ell @ x_sparse)
            e_full_abs = abs(y_full - y_true)
            e_sparse_abs = abs(y_sparse - y_true)
            rows.append(
                {
                    "structure_id": ctx.structure_id,
                    "case": ctx.case_name,
                    "seed": int(seed),
                    "functional_id": fid,
                    "functional_family": ctx.functional_family.get(fid, "unknown"),
                    "alpha": float(alpha),
                    "y_true": y_true,
                    "y_full_ridge": y_full,
                    "y_sparse": y_sparse,
                    "E_full_abs": e_full_abs,
                    "E_sparse_abs": e_sparse_abs,
                    "E_physical_norm": e_sparse_abs / max(abs(y_true), PHYSICAL_FLOOR),
                    "E_support_norm": abs(y_sparse - y_full) / max(abs(y_full), SUPPORT_FLOOR),
                    "B_physical_signed": y_sparse - y_true,
                    "near_zero_y_true": bool(abs(y_true) < PHYSICAL_FLOOR),
                }
            )
    return rows


def support_report(
    ctx: StructureContext, support: np.ndarray, constraints: SupportConstraints
) -> dict[str, Any]:
    report = support_constraint_report(ctx.matrix, support, constraints)
    return {
        "nnz": int(support.sum()),
        "valid": bool(report["valid"]),
        "failure_reasons": report.get("failure_reasons", []),
    }
