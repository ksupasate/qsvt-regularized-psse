"""Shared selector-comparison orchestrator for both tracks.

Generalizes the ``run_strong_baselines`` inner loop to one deterministic ``(case, size)`` block
scored on the block's representable physical functionals.  Reuses the frozen evaluation helpers
(``evaluate_support``, ``_jaccard``, ``_support_entry_set``) and produces per-family held-out
error, support stability, exact Ridge solve counts, and retained infeasible/skipped rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import CaseDesign, build_case_tasks
from robust_qsvt_se.cross_case_validation.selectors import produce_supports_generalized
from robust_qsvt_se.qsvt.output_aware_sparse_selection import SupportConstraints
from robust_qsvt_se.reviewer_blocking.common import array_fingerprint, atomic_write_csv
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    ExactLossEvaluator,
    _jaccard,
    _support_entry_set,
    evaluate_support,
)

FAMILY_OF = {
    "coordinate": "coordinate",
    "branch_angle_difference": "branch_difference",
    "area_aggregate": "aggregate",
    "legacy_predetermined": "legacy",
}


def _functional_family_map(design: CaseDesign) -> dict[str, str]:
    return {
        rec.functional_id: FAMILY_OF.get(rec.family, rec.family)
        for rec in design.functional_records
    }


def _oracle_reference_name(outcomes: dict[str, dict[str, Any]], objective: str) -> str:
    key = f"oracle_{objective}"
    if key in outcomes and outcomes[key].get("support") is not None:
        return key
    return f"near_oracle_{objective}"


def run_selector_comparison(
    case_name: str,
    design: CaseDesign,
    *,
    output_dir: str | Path,
    support_budgets: list[int],
    slot_budgets: list[int],
    training_seeds: list[int],
    held_out_seeds: list[int],
    selectors_subset: tuple[str, ...] | None,
    beam_width: int,
    near_oracle_max_loss_evals: int,
    random_seed_base: int,
    y_floor: float,
    failure_threshold: float,
) -> dict[str, Any]:
    """Run the full/subset selector suite on one block; write raw + summary CSVs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    small = design.small
    family_map = _functional_family_map(design)
    eval_ids = design.physical_functional_ids
    score_ids = design.physical_functional_ids
    if set(training_seeds) & set(held_out_seeds):
        raise ValueError("training and held-out seeds must be disjoint")

    training_tasks = build_case_tasks(case_name, small, training_seeds, "training", eval_ids)
    heldout_tasks = build_case_tasks(case_name, small, held_out_seeds, "held_out", eval_ids)
    evaluator = ExactLossEvaluator(small.matrix, training_tasks, small.alpha, y_floor)
    beam_max_steps = max(support_budgets) + 4

    support_records: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    oracle_gap_rows: list[dict[str, Any]] = []
    support_paths: dict[str, list[list[int]]] = {}
    support_entries_by_selector: dict[str, dict[tuple[int, int], frozenset]] = {}

    budgets = [
        SupportConstraints(int(k), int(s), True)
        for k in support_budgets
        for s in slot_budgets
    ]
    for pair_index, constraints in enumerate(budgets):
        evaluator.solves = 0
        outcomes = produce_supports_generalized(
            small, training_tasks, constraints,
            evaluator=evaluator, score_functional_ids=score_ids,
            random_seed=random_seed_base + 1000 * pair_index,
            beam_width=beam_width, beam_max_steps=beam_max_steps,
            oracle_max_candidates=3_000_000, include_oracle=False,
            near_oracle_max_loss_evals=near_oracle_max_loss_evals,
            selectors=selectors_subset,
        )
        solves_this_cell = int(evaluator.solves)
        oracle_entry_sets: dict[str, frozenset] = {}
        for objective in ("mean", "worst"):
            ref = _oracle_reference_name(outcomes, objective)
            oracle_entry_sets[objective] = _support_entry_set(
                outcomes[ref]["support"] if ref in outcomes else None
            )

        for selector, outcome in outcomes.items():
            support = outcome.get("support")
            feasible = support is not None and outcome.get("status") == "completed"
            objective = "worst" if "worst" in selector else "mean"
            support_id = (
                f"{small.label}_{selector}_k{constraints.k_budget}_s{constraints.slot_budget}"
            )
            entry_set = _support_entry_set(support)
            record: dict[str, Any] = {
                "case": case_name,
                "block": small.label,
                "dimension": small.dimension,
                "selector": selector,
                "objective": objective,
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
                "exact_ridge_solves_this_cell": solves_this_cell,
                "estimated_beam_loss_evals": outcome.get("estimated_beam_loss_evals"),
                "actual_nonzeros": int(support.sum()) if support is not None else np.nan,
                "support_fingerprint": (
                    array_fingerprint(support.astype(np.float64)) if support is not None else ""
                ),
                "training_selection_only": True,
            }
            if feasible:
                support_paths[support_id] = [[int(r), int(c)] for r, c in np.argwhere(support)]
                support_entries_by_selector.setdefault(selector, {})[
                    (constraints.k_budget, constraints.slot_budget)
                ] = entry_set
                training_norm = evaluator.normalized(support)
                record["training_mean_normalized_error"] = float(np.mean(training_norm))
                record["training_worst_normalized_error"] = float(np.max(training_norm))
                heldout_eval = evaluate_support(small, support, heldout_tasks)
                combined = evaluate_support(small, support, training_tasks) + heldout_eval
                for row in combined:
                    task_rows.append(
                        row | {
                            "case": case_name, "selector": selector,
                            "support_id": support_id, "block": small.label,
                            "family": family_map.get(row["functional_id"], "unknown"),
                        }
                    )
                heldout_frame = pd.DataFrame(heldout_eval)
                heldout_frame["family"] = heldout_frame["functional_id"].map(family_map)
                heldout_norm = heldout_frame["normalized_error"].to_numpy()
                record["heldout_mean_normalized_error"] = float(np.mean(heldout_norm))
                record["heldout_median_normalized_error"] = float(np.median(heldout_norm))
                record["heldout_worst_normalized_error"] = float(np.max(heldout_norm))
                record["heldout_worst_absolute_error"] = float(
                    heldout_frame["absolute_error"].max()
                )
                record["heldout_failure_fraction"] = float(
                    np.mean(heldout_norm > failure_threshold)
                )
                for family, group in heldout_frame.groupby("family"):
                    record[f"heldout_error_family_{family}"] = float(
                        group["normalized_error"].mean()
                    )
                record["overlap_with_near_oracle"] = _jaccard(
                    entry_set, oracle_entry_sets[objective]
                )
            else:
                for column in (
                    "training_mean_normalized_error", "training_worst_normalized_error",
                    "heldout_mean_normalized_error", "heldout_median_normalized_error",
                    "heldout_worst_normalized_error", "heldout_worst_absolute_error",
                    "heldout_failure_fraction", "overlap_with_near_oracle",
                ):
                    record[column] = np.nan
            support_records.append(record)

        for objective in ("mean", "worst"):
            ref = _oracle_reference_name(outcomes, objective)
            ref_outcome = outcomes.get(ref)
            if ref_outcome is None or ref_outcome.get("support") is None:
                continue
            ref_loss = float(ref_outcome["final_loss"])
            for selector, outcome in outcomes.items():
                if outcome.get("support") is None or outcome.get("status") != "completed":
                    continue
                if ("worst" in selector) != (objective == "worst"):
                    continue
                selector_loss = evaluator.loss(outcome["support"], objective)
                oracle_gap_rows.append({
                    "case": case_name, "block": small.label, "objective": objective,
                    "k_budget": constraints.k_budget, "slot_budget": constraints.slot_budget,
                    "reference": ref, "reference_kind": "near_oracle_multistart_local_search",
                    "selector": selector, "selector_training_loss": selector_loss,
                    "reference_training_loss": ref_loss,
                    "optimality_gap": selector_loss - ref_loss,
                    "support_overlap_jaccard": _jaccard(
                        _support_entry_set(outcome["support"]),
                        _support_entry_set(ref_outcome["support"]),
                    ),
                })

    support_frame = pd.DataFrame(support_records)
    atomic_write_csv(destination / "raw_selector_results.csv", support_frame)
    atomic_write_csv(destination / "raw_task_results.csv", pd.DataFrame(task_rows))
    atomic_write_csv(destination / "oracle_gap_summary.csv", pd.DataFrame(oracle_gap_rows))

    summary = _summarize(support_frame)
    atomic_write_csv(destination / "selector_summary.csv", summary)
    stability = _support_stability(support_entries_by_selector, case_name, small.label)
    atomic_write_csv(destination / "support_stability.csv", stability)

    from robust_qsvt_se.reviewer_blocking.common import atomic_write_json

    atomic_write_json(destination / "support_paths.json", support_paths)
    return {
        "case": case_name,
        "block": small.label,
        "support_records": len(support_frame),
        "feasible_supports": int(support_frame["feasible"].sum()) if not support_frame.empty else 0,
        "selectors": (
            sorted(support_frame["selector"].unique().tolist())
            if not support_frame.empty else []
        ),
        "summary_frame": summary,
        "support_frame": support_frame,
        "stability_frame": stability,
    }


def _summarize(support_frame: pd.DataFrame) -> pd.DataFrame:
    if support_frame.empty:
        return support_frame
    grouped = support_frame.groupby(["case", "block", "selector", "objective"], sort=True)
    return grouped.agg(
        budgets=("support_id", "count"),
        feasible=("feasible", "sum"),
        feasibility_rate=("feasible", "mean"),
        mean_training_normalized_error=("training_mean_normalized_error", "mean"),
        mean_heldout_normalized_error=("heldout_mean_normalized_error", "mean"),
        median_heldout_normalized_error=("heldout_mean_normalized_error", "median"),
        worst_heldout_absolute_error=("heldout_worst_absolute_error", "max"),
        mean_heldout_failure_fraction=("heldout_failure_fraction", "mean"),
        mean_overlap_with_near_oracle=("overlap_with_near_oracle", "mean"),
        total_selection_runtime_seconds=("selection_runtime_seconds", "sum"),
        total_milp_solves=("milp_solves", "sum"),
    ).reset_index()


def _support_stability(
    entries_by_selector: dict[str, dict[tuple[int, int], frozenset]],
    case_name: str,
    block_label: str,
) -> pd.DataFrame:
    """Per-selector support stability: mean pairwise Jaccard across budget cells."""

    rows: list[dict[str, Any]] = []
    for selector, by_budget in sorted(entries_by_selector.items()):
        cells = sorted(by_budget.items())
        jaccards: list[float] = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                jaccards.append(_jaccard(cells[i][1], cells[j][1]))
        # Adjacent-k stability at fixed slot (support growth monotonicity proxy).
        rows.append({
            "case": case_name,
            "block": block_label,
            "selector": selector,
            "feasible_cells": len(cells),
            "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
            "min_pairwise_jaccard": float(np.min(jaccards)) if jaccards else np.nan,
            "max_pairwise_jaccard": float(np.max(jaccards)) if jaccards else np.nan,
        })
    return pd.DataFrame(rows)
