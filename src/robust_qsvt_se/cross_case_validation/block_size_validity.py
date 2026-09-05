"""Physical-validity versus block-size study (classical, full support only).

For each IEEE case, nested selected blocks (8x8 -> 16x16 -> 32x32 -> 64x64 where the state
dimension permits -> the full weighted system) are extracted with the frozen outcome-independent
``largest_row_col_norms`` extractor, and the full-support truth-referenced physical error
``E_physical`` is computed for the existing physical functional families under two declared
alpha policies.  No circuits and no support selector are involved: the question is whether the
block-truncation floor observed at 8x8 in the main physical audit shrinks with block size, and
at what size the selected-block surrogate becomes physically credible.

Protocol constants (held-out seeds, physical floor, near-zero handling, stage-1 aggregation)
mirror the frozen reviewer-evidence physical-accuracy protocol.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import (
    SINGULAR_TOLERANCE,
    _cached_system,
    block_conditioning,
    build_case_block_binding,
    build_case_full_system,
    generate_case_residual,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import _ridge_filter_operator
from robust_qsvt_se.reviewer_blocking.physical_functionals import (
    FunctionalRecord,
    UnavailableFunctional,
    build_area_aggregate_functionals,
    build_branch_angle_difference_functionals,
    build_coordinate_functionals,
)

STUDY_ID = "physical_validity_vs_block_size_v1"
MATRIX_SEED = 123
HELD_OUT_SEEDS = tuple(range(2000, 2020))  # frozen reviewer-evidence held-out protocol
PHYSICAL_FLOOR = 1.0e-6
SQUARE_SIZES = (8, 16, 32, 64)
CASES = ("ieee14", "ieee30", "ieee57")
ALPHA_POLICIES = ("matched_frozen_benchmark", "design_4sigma_min")
# Descriptive (not pre-registered) credibility thresholds on the stage-1 median E_physical.
CREDIBILITY_THRESHOLDS = (0.5, 0.1)


@dataclass(slots=True)
class BlockContext:
    """One extracted block with its physical-functional binding."""

    case_name: str
    size_label: str
    matrix: np.ndarray
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    functional_records: list[FunctionalRecord]
    unavailable: list[UnavailableFunctional]
    conditioning: dict[str, Any]
    design_alpha: float
    state_dimension: int
    measurement_dimension: int


def _frozen_manuscript_alpha(cache_dir: str) -> tuple[float, np.ndarray]:
    """Frozen IEEE-14 8x8 manuscript alpha (QSVT-feasible operating point) plus its matrix."""

    from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
        build_frozen_output_aware_design,
    )

    frozen = build_frozen_output_aware_design(cache_dir)
    return float(frozen.alpha), np.asarray(frozen.matrix, dtype=np.float64)


def block_size_plan(case_name: str) -> tuple[list[int], list[dict[str, Any]]]:
    """Square sizes representable for this case plus records of structurally unavailable sizes."""

    matrix, _residual, _metadata = _cached_system(case_name, MATRIX_SEED)
    rows, cols = matrix.shape
    available = [k for k in SQUARE_SIZES if k <= min(rows, cols)]
    skipped = [
        {
            "case": case_name,
            "size_label": f"{k}x{k}",
            "reason": (
                f"requested {k}x{k} block exceeds system shape {rows}x{cols}; "
                "structurally unavailable, not substituted"
            ),
        }
        for k in SQUARE_SIZES
        if k > min(rows, cols)
    ]
    return available, skipped


def build_block_context(case_name: str, size: int | str) -> BlockContext:
    """Extract one block with the frozen extractor and bind the physical functional families.

    ``size`` is a square edge length or ``"full"`` (all rows and all columns through the same
    extractor code path, which reduces to a deterministic permutation of the complete system).
    """

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block

    matrix_full, residual_full, _metadata = _cached_system(case_name, MATRIX_SEED)
    m, n = matrix_full.shape
    if size == "full":
        row_count, col_count, label = m, n, "full"
    else:
        row_count = col_count = int(size)
        label = f"{row_count}x{col_count}"
    block, _rblock, rows, cols = select_deterministic_block(
        matrix_full, residual_full, row_count=row_count, col_count=col_count
    )
    block = np.asarray(block, dtype=np.float64)
    conditioning = block_conditioning(block)
    if conditioning["min_positive_singular_value"] <= 0.0:
        raise ValueError(f"{case_name} {label}: block has no positive singular values")
    design_alpha = 4.0 * conditioning["min_positive_singular_value"] ** 2

    binding = build_case_block_binding(
        case_name, MATRIX_SEED, row_count=row_count, col_count=col_count
    )
    records: list[FunctionalRecord] = []
    records.extend(build_coordinate_functionals(binding))
    records.extend(build_branch_angle_difference_functionals(binding))
    area_records, unavailable = build_area_aggregate_functionals(binding)
    records.extend(area_records)
    records.sort(key=lambda r: (r.family, r.functional_id))
    return BlockContext(
        case_name=case_name,
        size_label=label,
        matrix=block,
        selected_rows=tuple(int(v) for v in rows),
        selected_columns=tuple(int(v) for v in cols),
        functional_records=records,
        unavailable=list(unavailable),
        conditioning=conditioning,
        design_alpha=float(design_alpha),
        state_dimension=int(n),
        measurement_dimension=int(m),
    )


def resolve_alpha(
    context: BlockContext, policy: str, frozen_alpha: float, frozen_matrix: np.ndarray
) -> float:
    """Per-block alpha under a declared policy.

    ``matched_frozen_benchmark`` reuses the reviewer-evidence engine idiom: the IEEE-14 8x8
    primary structure runs at the frozen manuscript alpha (the QSVT-feasible operating point),
    every other structure at its ``4 * sigma_min_pos^2`` design alpha.  ``design_4sigma_min``
    applies the design convention uniformly so the cross-size trend has one matched rule.
    """

    if policy == "design_4sigma_min":
        return float(context.design_alpha)
    if policy == "matched_frozen_benchmark":
        if context.case_name == "ieee14" and context.size_label == "8x8":
            if not np.allclose(frozen_matrix, context.matrix):
                raise RuntimeError("frozen ieee14 8x8 matrix drifted from the extractor block")
            return float(frozen_alpha)
        return float(context.design_alpha)
    raise ValueError(f"unknown alpha policy {policy}")


def evaluate_full_support_rows(
    context: BlockContext, alpha: float, alpha_policy: str
) -> list[dict[str, Any]]:
    """Truth-referenced full-support rows for every held-out seed and physical functional."""

    operator = _ridge_filter_operator(context.matrix, float(alpha))
    cols = np.asarray(context.selected_columns, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for seed in HELD_OUT_SEEDS:
        residual = generate_case_residual(context.case_name, int(seed), context.selected_rows)
        full = build_case_full_system(context.case_name, int(seed))
        delta_true_block = np.asarray(full.x_true, dtype=np.float64)[cols]
        x_full = operator @ residual
        for record in context.functional_records:
            ell = np.asarray(record.vector, dtype=np.float64)
            y_true = float(ell @ delta_true_block)
            y_full = float(ell @ x_full)
            e_abs = abs(y_full - y_true)
            rows.append(
                {
                    "case": context.case_name,
                    "size_label": context.size_label,
                    "block_rows": len(context.selected_rows),
                    "block_cols": len(context.selected_columns),
                    "alpha_policy": alpha_policy,
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "functional_id": record.functional_id,
                    "functional_family": record.family,
                    "y_true": y_true,
                    "y_full_ridge": y_full,
                    "E_full_abs": e_abs,
                    "E_physical_norm": e_abs / max(abs(y_true), PHYSICAL_FLOOR),
                    "near_zero_y_true": bool(abs(y_true) < PHYSICAL_FLOOR),
                }
            )
    return rows


def _stage1_median(group: pd.DataFrame) -> float:
    non_zero = group[~group["near_zero_y_true"]]
    return float(non_zero["E_physical_norm"].median()) if len(non_zero) else float("nan")


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage-1 audit aggregation (median over held-out seed-functional rows) per cell.

    Each (case, size) has exactly one extractor realization, so the structure statistic equals
    the realization median, matching the frozen audit rule "median held-out seed-functional rows
    within realization" with a single realization.
    """

    cell_rows: list[dict[str, Any]] = []
    keys = ["case", "size_label", "block_rows", "block_cols", "alpha_policy", "alpha"]
    for key, group in raw.groupby(keys, sort=True):
        non_zero = group[~group["near_zero_y_true"]]
        cell_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_rows": len(group),
                "n_near_zero_y_true": int(group["near_zero_y_true"].sum()),
                "n_functionals": int(group["functional_id"].nunique()),
                "median_E_physical_norm": _stage1_median(group),
                "mean_E_physical_norm": float(non_zero["E_physical_norm"].mean())
                if len(non_zero)
                else float("nan"),
                "median_E_full_abs": float(group["E_full_abs"].median()),
                "aggregation_rule": (
                    "median over held-out seed-functional rows (near-zero y_true excluded); "
                    "single extractor realization per case and size"
                ),
            }
        )
    family_rows: list[dict[str, Any]] = []
    fam_keys = [*keys, "functional_family"]
    for key, group in raw.groupby(fam_keys, sort=True):
        family_rows.append(
            {
                **dict(zip(fam_keys, key, strict=True)),
                "n_rows": len(group),
                "n_near_zero_y_true": int(group["near_zero_y_true"].sum()),
                "median_E_physical_norm": _stage1_median(group),
            }
        )
    return pd.DataFrame(cell_rows), pd.DataFrame(family_rows)


def _size_order(label: str) -> int:
    return 10**9 if label == "full" else int(label.split("x")[0])


def trend_assessment(summary: pd.DataFrame) -> pd.DataFrame:
    """Monotonic-or-not verdict per (case, alpha policy) with declared credibility thresholds."""

    rows: list[dict[str, Any]] = []
    for (case, policy), group in summary.groupby(["case", "alpha_policy"], sort=True):
        ordered = group.sort_values("size_label", key=lambda s: s.map(_size_order))
        labels = ordered["size_label"].tolist()
        medians = ordered["median_E_physical_norm"].to_numpy(dtype=np.float64)
        deltas = np.diff(medians)
        record: dict[str, Any] = {
            "case": case,
            "alpha_policy": policy,
            "size_sequence": ">".join(labels),
            "median_sequence": ";".join(f"{v:.6f}" for v in medians),
            "monotonic_nonincreasing": bool(np.all(deltas <= 0.0)),
            "strictly_decreasing": bool(np.all(deltas < 0.0)),
            "largest_increase": float(deltas.max()) if deltas.size else 0.0,
            "ratio_full_to_8x8": (
                float(medians[-1] / medians[0]) if medians[0] > 0 else float("nan")
            ),
        }
        for threshold in CREDIBILITY_THRESHOLDS:
            below = [lab for lab, v in zip(labels, medians, strict=True) if v <= threshold]
            record[f"first_size_median_le_{threshold}"] = below[0] if below else "never"
        rows.append(record)
    return pd.DataFrame(rows)


def nestedness(contexts: dict[tuple[str, str], BlockContext]) -> pd.DataFrame:
    """Row/column overlap between consecutive extractor blocks (factual, not asserted)."""

    rows: list[dict[str, Any]] = []
    for case in CASES:
        labels = sorted(
            (label for (c, label) in contexts if c == case), key=_size_order
        )
        for smaller, larger in itertools.pairwise(labels):
            a = contexts[(case, smaller)]
            b = contexts[(case, larger)]
            col_a, col_b = set(a.selected_columns), set(b.selected_columns)
            row_a, row_b = set(a.selected_rows), set(b.selected_rows)
            rows.append(
                {
                    "case": case,
                    "smaller": smaller,
                    "larger": larger,
                    "column_overlap_fraction": len(col_a & col_b) / len(col_a),
                    "row_overlap_fraction": len(row_a & row_b) / len(row_a),
                    "columns_nested": bool(col_a <= col_b),
                    "rows_nested": bool(row_a <= row_b),
                }
            )
    return pd.DataFrame(rows)


def block_inventory(
    contexts: dict[tuple[str, str], BlockContext], skipped: list[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (case, label), ctx in sorted(
        contexts.items(), key=lambda kv: (kv[0][0], _size_order(kv[0][1]))
    ):
        families = sorted({r.family for r in ctx.functional_records})
        singular = ctx.conditioning
        rows.append(
            {
                "case": case,
                "size_label": label,
                "block_rows": len(ctx.selected_rows),
                "block_cols": len(ctx.selected_columns),
                "state_dimension": ctx.state_dimension,
                "measurement_dimension": ctx.measurement_dimension,
                "state_coverage_fraction": len(ctx.selected_columns) / ctx.state_dimension,
                "rank": singular["rank"],
                "full_rank": singular["full_rank"],
                "min_positive_singular_value": singular["min_positive_singular_value"],
                "raw_condition_number": (
                    "inf"
                    if not np.isfinite(singular["raw_condition_number"])
                    else singular["raw_condition_number"]
                ),
                "design_alpha": ctx.design_alpha,
                "n_physical_functionals": len(ctx.functional_records),
                "available_families": ";".join(families),
                "n_unavailable_functionals": len(ctx.unavailable),
                "unavailable_reasons": " | ".join(
                    f"{u.requested_functional_id}: {u.reason_unavailable}"
                    for u in ctx.unavailable
                ),
                "status": "included",
            }
        )
    for record in skipped:
        rows.append(
            {
                "case": record["case"],
                "size_label": record["size_label"],
                "status": "structurally_unavailable",
                "unavailable_reasons": record["reason"],
            }
        )
    return pd.DataFrame(rows)


def run_study(frozen_cache_dir: str) -> dict[str, pd.DataFrame]:
    """Execute the complete study; purely classical and deterministic."""

    frozen_alpha, frozen_matrix = _frozen_manuscript_alpha(frozen_cache_dir)
    contexts: dict[tuple[str, str], BlockContext] = {}
    skipped: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for case in CASES:
        sizes, case_skipped = block_size_plan(case)
        skipped.extend(case_skipped)
        for size in [*sizes, "full"]:
            ctx = build_block_context(case, size)
            contexts[(case, ctx.size_label)] = ctx
            for policy in ALPHA_POLICIES:
                alpha = resolve_alpha(ctx, policy, frozen_alpha, frozen_matrix)
                raw_rows.extend(evaluate_full_support_rows(ctx, alpha, policy))
    raw = pd.DataFrame(raw_rows)
    summary, family_summary = summarize(raw)
    return {
        "raw": raw,
        "summary": summary,
        "family_summary": family_summary,
        "trend": trend_assessment(summary),
        "nestedness": nestedness(contexts),
        "inventory": block_inventory(contexts, skipped),
        "constants": pd.DataFrame(
            [
                {
                    "study_id": STUDY_ID,
                    "matrix_seed": MATRIX_SEED,
                    "held_out_seeds": f"{HELD_OUT_SEEDS[0]}..{HELD_OUT_SEEDS[-1]}",
                    "n_held_out_seeds": len(HELD_OUT_SEEDS),
                    "physical_floor": PHYSICAL_FLOOR,
                    "singular_tolerance": SINGULAR_TOLERANCE,
                    "frozen_ieee14_8x8_alpha": frozen_alpha,
                    "alpha_policies": ";".join(ALPHA_POLICIES),
                    "credibility_thresholds": ";".join(
                        str(t) for t in CREDIBILITY_THRESHOLDS
                    ),
                    "selector": "full_support_only",
                }
            ]
        ),
    }


__all__ = [
    "ALPHA_POLICIES",
    "CASES",
    "HELD_OUT_SEEDS",
    "MATRIX_SEED",
    "PHYSICAL_FLOOR",
    "STUDY_ID",
    "BlockContext",
    "block_size_plan",
    "build_block_context",
    "evaluate_full_support_rows",
    "resolve_alpha",
    "run_study",
    "summarize",
    "trend_assessment",
]
