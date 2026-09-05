"""Phase 4 - strong task-aware support-selection baselines.

Tests whether the proposed sensitivity selector is distinguished from strong task-aware baselines
under IDENTICAL constraints, or whether any gain comes merely from using output information at all:

* Baseline A (``adjoint_unnormalized_mean``): the proposed score WITHOUT per-task normalization.
* Baseline B (``exact_single_removal_mean``): exact one-entry-removal training-loss increase.

All selectors share the same MILP, budgets, coverage, deterministic tie-break, and held-out tasks.
The near-oracle multi-start local search is included as a diagnostic reference (never deployable).
Primary metric = held-out support-fidelity loss (the manuscript's selected-output metric); physical
error at the fixed benchmark alpha is reported alongside.  Failed budgets are retained.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
)
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    provenance_block,
)
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    ExactLossEvaluator,
    near_oracle_beam,
    near_oracle_multistart,
)
from robust_qsvt_se.reviewer_evidence.engine import (
    build_structure,
    build_tasks,
    evaluate_triples,
    select_support,
    support_report,
)

STUDY_ID = "reviewer_evidence_task_aware_baselines_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")
DEFAULT_CONFIG_PATH = Path("configs/reviewer_blocking_tqe_evidence/task_aware_baselines.json")

PROPOSED = "sensitivity_refined_mean"


def _legacy_tasks(
    structure_id: str, seeds: list[int], split: str, legacy_ids: tuple[str, ...]
) -> list[RidgeTask]:
    return [
        t for t in build_tasks(structure_id, seeds, split) if t.functional_id in set(legacy_ids)
    ]


def _entry_set(support: np.ndarray | None) -> frozenset:
    if support is None:
        return frozenset()
    return frozenset((int(r), int(c)) for r, c in np.argwhere(support))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_task_aware_baselines(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    structure_ids = list(config["structures"])
    selectors = list(config["selectors"])
    budgets = [(int(k), int(s)) for k in config["support_budgets"] for s in config["slot_budgets"]]
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    held_out = [int(s) for s in config["held_out_seed_ids"]]
    y_floor = float(config.get("y_floor", 1e-6))
    beam_width = int(config.get("beam_width", 8))
    beam_max_steps = int(config.get("beam_max_steps", 40))
    include_near_oracle = bool(config.get("include_near_oracle", True))

    support_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for sid in structure_ids:
        ctx = build_structure(sid)
        legacy_ids = ctx.legacy_ids
        score_ids = (
            legacy_ids  # manuscript convention: support-fidelity on the 3 legacy functionals
        )
        train_legacy = _legacy_tasks(sid, training_seeds, "training", legacy_ids)
        held_legacy = _legacy_tasks(sid, held_out, "held_out", legacy_ids)
        held_eval = ExactLossEvaluator(ctx.matrix, held_legacy, ctx.alpha_fixed, y_floor)

        for k, s in budgets:
            cons = SupportConstraints(k, s, True)
            supports: dict[str, np.ndarray | None] = {}
            records: dict[str, dict[str, Any]] = {}
            # Standard/baseline selectors first, so the near-oracle can be seeded with all of them
            # (matching the frozen near-oracle semantics: it dominates every feasible construction).
            ordered = [s_ for s_ in selectors if s_ != "near_oracle_mean"]
            if "near_oracle_mean" in selectors:
                ordered.append("near_oracle_mean")
            for selector in ordered:
                if selector == "near_oracle_mean":
                    seed_supports = [sup for sup in supports.values() if sup is not None]
                    rec = (
                        _near_oracle(
                            ctx,
                            train_legacy,
                            cons,
                            beam_width,
                            beam_max_steps,
                            y_floor,
                            seed_supports,
                        )
                        if include_near_oracle
                        else {
                            "support": None,
                            "status": "skipped",
                            "failure_reason": "disabled",
                            "runtime_seconds": 0.0,
                            "extra_evals": 0,
                        }
                    )
                else:
                    rec = select_support(
                        ctx, selector, cons, train_legacy, score_ids, y_floor=y_floor
                    )
                supports[selector] = rec["support"]
                feasible = rec["support"] is not None and rec["status"] == "completed"
                valid = feasible and support_report(ctx, rec["support"], cons)["valid"]
                row: dict[str, Any] = {
                    "structure_id": sid,
                    "selector": selector,
                    "k_budget": k,
                    "slot_budget": s,
                    "status": rec["status"],
                    "feasible": bool(valid),
                    "failure_reason": rec.get("failure_reason", ""),
                    "selection_runtime_seconds": float(rec.get("runtime_seconds", 0.0)),
                    "exact_solves": int(rec.get("extra_evals", 0)),
                    "actual_nnz": int(rec["support"].sum())
                    if rec["support"] is not None
                    else np.nan,
                }
                if valid:
                    held_norm = held_eval.normalized(rec["support"])
                    row["heldout_support_fidelity_mean"] = float(np.mean(held_norm))
                    row["heldout_support_fidelity_median"] = float(np.median(held_norm))
                    triples = evaluate_triples(
                        ctx, rec["support"], ctx.alpha_fixed, held_out, list(ctx.physical_ids)
                    )
                    ep = np.array(
                        [t["E_physical_norm"] for t in triples if not t["near_zero_y_true"]]
                    )
                    row["heldout_physical_median"] = (
                        float(np.median(ep)) if len(ep) else float("nan")
                    )
                else:
                    row["heldout_support_fidelity_mean"] = np.nan
                    row["heldout_support_fidelity_median"] = np.nan
                    row["heldout_physical_median"] = np.nan
                records[selector] = row
                support_rows.append(row)

            # Pairwise: proposed vs each baseline (support overlap + per-task win/tie/loss).
            proposed_support = supports.get(PROPOSED)
            if proposed_support is None:
                continue
            proposed_norm = held_eval.normalized(proposed_support)
            for selector in selectors:
                if selector == PROPOSED or supports.get(selector) is None:
                    continue
                other_norm = held_eval.normalized(supports[selector])
                diff = proposed_norm - other_norm  # <0 => proposed better
                tie_tol = float(config.get("tie_tolerance", 1e-9))
                wins = int(np.sum(diff < -tie_tol))
                losses = int(np.sum(diff > tie_tol))
                ties = int(np.sum(np.abs(diff) <= tie_tol))
                pairwise_rows.append(
                    {
                        "structure_id": sid,
                        "k_budget": k,
                        "slot_budget": s,
                        "proposed": PROPOSED,
                        "baseline": selector,
                        "proposed_heldout_mean": float(np.mean(proposed_norm)),
                        "baseline_heldout_mean": float(np.mean(other_norm)),
                        "median_diff_proposed_minus_baseline": float(np.median(diff)),
                        "proposed_wins": wins,
                        "ties": ties,
                        "proposed_losses": losses,
                        "support_overlap_jaccard": _jaccard(
                            _entry_set(proposed_support), _entry_set(supports[selector])
                        ),
                        "identical_support": bool(
                            _entry_set(proposed_support) == _entry_set(supports[selector])
                        ),
                    }
                )

    support_frame = pd.DataFrame(support_rows)
    atomic_write_csv(destination / "task_aware_baseline_rows.csv", support_frame)
    pairwise_frame = pd.DataFrame(pairwise_rows)
    atomic_write_csv(destination / "task_aware_baseline_pairwise.csv", pairwise_frame)
    summary = _summarize(support_frame, pairwise_frame)
    atomic_write_csv(destination / "task_aware_baseline_summary.csv", summary)
    atomic_write_json(
        destination / "task_aware_baseline_provenance.json",
        provenance_block(config_path, config)
        | {"study_id": STUDY_ID, "claim_boundary": CLAIM_BOUNDARY},
    )
    return {
        "support_rows": len(support_frame),
        "pairwise_rows": len(pairwise_frame),
        "structures": structure_ids,
        "proposed_distinguished": _distinguished_verdict(pairwise_frame),
    }


def _near_oracle(
    ctx, train_tasks, cons, beam_width, beam_max_steps, y_floor, seed_supports=None
) -> dict[str, Any]:
    started = time.perf_counter()
    evaluator = ExactLossEvaluator(ctx.matrix, train_tasks, ctx.alpha_fixed, y_floor)
    beam = near_oracle_beam(
        evaluator, cons, objective="mean", beam_width=beam_width, max_steps=beam_max_steps
    )
    seeds = list(seed_supports) if seed_supports else []
    if beam.get("support") is not None:
        seeds.append(beam["support"])
    out = near_oracle_multistart(
        evaluator, train_tasks, cons, objective="mean", seed_supports=seeds, beam_diagnostic=beam
    )
    return {
        "support": out.get("support"),
        "status": out.get("status", "unknown"),
        "failure_reason": out.get("failure_reason", ""),
        "runtime_seconds": time.perf_counter() - started,
        "extra_evals": 0,
    }


def _summarize(support_frame: pd.DataFrame, pairwise_frame: pd.DataFrame) -> pd.DataFrame:
    if support_frame.empty:
        return support_frame
    rows = []
    for (sid, selector), g in support_frame.groupby(["structure_id", "selector"]):
        rows.append(
            {
                "structure_id": sid,
                "selector": selector,
                "budgets": len(g),
                "feasible": int(g["feasible"].sum()),
                "feasibility_fraction": float(g["feasible"].mean()),
                "mean_heldout_support_fidelity": float(
                    g["heldout_support_fidelity_mean"].mean(skipna=True)
                ),
                "mean_heldout_physical": float(g["heldout_physical_median"].mean(skipna=True)),
                "total_selection_runtime_s": float(g["selection_runtime_seconds"].sum()),
                "total_exact_solves": int(g["exact_solves"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _distinguished_verdict(pairwise_frame: pd.DataFrame) -> dict[str, Any]:
    if pairwise_frame.empty:
        return {}
    verdict = {}
    for baseline, g in pairwise_frame.groupby("baseline"):
        identical = float(g["identical_support"].mean())
        median_effect = float(g["median_diff_proposed_minus_baseline"].median())
        verdict[baseline] = {
            "fraction_identical_support": identical,
            "median_support_fidelity_diff_proposed_minus_baseline": median_effect,
            "interpretation": (
                "indistinguishable (mostly identical supports)"
                if identical >= 0.5
                else (
                    "proposed better"
                    if median_effect < -1e-6
                    else ("proposed worse" if median_effect > 1e-6 else "comparable")
                )
            ),
        }
    return verdict
