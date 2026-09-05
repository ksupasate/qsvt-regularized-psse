"""Phase 2 - physical selected-output accuracy vs the true update.

For every structure x selector x budget x alpha-regime x functional x held-out seed, compute the
coherent triple ``{y_true, y_full_Ridge, y_sparse}`` and derive physical error (vs the true update)
and support-fidelity error (vs full-block Ridge) as SEPARATE columns.  Answers whether output-aware
sparse support improves the true physical quantity, and whether support fidelity predicts physical
accuracy.  Every infeasible/failed/unavailable row is retained with a reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import build_case_design
from robust_qsvt_se.qsvt.output_aware_sparse_selection import SupportConstraints
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    atomic_write_csv,
    atomic_write_json,
    bus_set_is_connected,
    load_config,
    provenance_block,
)
from robust_qsvt_se.reviewer_evidence.engine import (
    STRUCTURES,
    build_structure,
    build_tasks,
    design_time_alpha_regimes,
    evaluate_triples,
    select_support,
    support_report,
)

STUDY_ID = "reviewer_evidence_physical_accuracy_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")
DEFAULT_CONFIG_PATH = Path("configs/reviewer_blocking_tqe_evidence/physical_accuracy.json")


def _score_ids(ctx, which: str) -> tuple[str, ...]:
    return ctx.physical_ids if which == "physical" else ctx.legacy_ids


def build_functional_mapping(structure_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in structure_ids:
        case_name, dimension, _ = STRUCTURES[sid]
        design = build_case_design(case_name, 123, dimension=dimension)
        branch_set = set(design.binding.branches)
        for rec in design.functional_records:
            physical = rec.family != "legacy_predetermined"
            branch_exists = None
            if rec.family == "branch_angle_difference" and rec.global_bus_ids:
                pair = tuple(int(b) for b in rec.global_bus_ids[:2])
                branch_exists = bool(pair in branch_set or pair[::-1] in branch_set)
            connected = None
            if rec.family == "area_aggregate" and rec.area_members:
                connected = bool(
                    bus_set_is_connected(
                        set(int(b) for b in rec.area_members), design.binding.branches
                    )
                )
            rows.append(
                {
                    "structure_id": sid,
                    "case": case_name,
                    "functional_id": rec.functional_id,
                    "family": rec.family,
                    "label": rec.label,
                    "classification": "physical" if physical else "legacy_generic_diagnostic",
                    "state_type": rec.state_type,
                    "local_block_indices": list(rec.local_block_indices),
                    "global_state_indices": list(rec.global_state_indices),
                    "bus_ids": list(rec.global_bus_ids),
                    "branch_id": rec.branch_id,
                    "branch_exists_in_network": branch_exists,
                    "area_members": list(rec.area_members),
                    "connected_area": connected,
                    "unit_norm": bool(rec.unit_norm),
                    "unit_norm_error": float(rec.unit_norm_error),
                    "unavailable_reason": "",
                }
            )
        for un in design.unavailable:
            rows.append(
                {
                    "structure_id": sid,
                    "case": case_name,
                    "functional_id": un.requested_functional_id,
                    "family": un.family,
                    "label": "UNAVAILABLE",
                    "classification": "unavailable_not_substituted",
                    "state_type": "",
                    "local_block_indices": [],
                    "global_state_indices": [],
                    "bus_ids": [],
                    "branch_id": "",
                    "branch_exists_in_network": None,
                    "area_members": [],
                    "connected_area": None,
                    "unit_norm": None,
                    "unit_norm_error": None,
                    "unavailable_reason": un.reason_unavailable,
                }
            )
    return pd.DataFrame(rows)


def run_physical_accuracy(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    structure_ids = list(config["structures"])
    selectors = list(config["selectors"])
    which_scores = str(config.get("score_functional_set", "physical"))
    budgets = [(int(k), int(s)) for k in config["support_budgets"] for s in config["slot_budgets"]]
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    held_out = [int(s) for s in config["held_out_seed_ids"]]
    if set(training_seeds) & set(held_out):
        raise ValueError("training and held-out seeds must be disjoint")

    functional_map = build_functional_mapping(structure_ids)
    atomic_write_csv(destination / "functional_mapping.csv", functional_map)

    row_records: list[dict[str, Any]] = []
    alpha_regime_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for sid in structure_ids:
        ctx = build_structure(sid)
        score_ids = _score_ids(ctx, which_scores)
        train_tasks = build_tasks(sid, training_seeds, "training")
        regimes = design_time_alpha_regimes(ctx)
        for r in regimes:
            alpha_regime_records.append({"structure_id": sid, **r})
        eval_functionals = list(ctx.functionals)

        for selector in selectors:
            budget_iter = [(None, None)] if selector == "full_support" else budgets
            for k, s in budget_iter:
                if selector == "full_support":
                    support = ctx.matrix != 0.0
                    nnz, valid, is_ref = int(support.sum()), True, True
                    sel_reason = ""
                else:
                    cons = SupportConstraints(int(k), int(s), True)
                    rec = select_support(ctx, selector, cons, train_tasks, score_ids)
                    support = rec["support"]
                    is_ref = False
                    if support is None or rec["status"] != "completed":
                        failure_records.append(
                            {
                                "structure_id": sid,
                                "selector": selector,
                                "k_budget": k,
                                "slot_budget": s,
                                "stage": "support_selection",
                                "reason": rec.get("failure_reason", "unavailable"),
                            }
                        )
                        continue
                    report = support_report(ctx, support, cons)
                    nnz, valid = report["nnz"], report["valid"]
                    sel_reason = "" if valid else ",".join(report["failure_reasons"])
                    if not valid:
                        failure_records.append(
                            {
                                "structure_id": sid,
                                "selector": selector,
                                "k_budget": k,
                                "slot_budget": s,
                                "stage": "support_constraint",
                                "reason": sel_reason,
                            }
                        )
                        continue
                for r in regimes:
                    alpha = r["alpha"]
                    if not np.isfinite(alpha):
                        failure_records.append(
                            {
                                "structure_id": sid,
                                "selector": selector,
                                "k_budget": k,
                                "slot_budget": s,
                                "stage": "alpha_regime",
                                "reason": f"{r['regime']}: {r['note']}",
                            }
                        )
                        continue
                    triples = evaluate_triples(
                        ctx, support, float(alpha), held_out, eval_functionals
                    )
                    for t in triples:
                        t.update(
                            {
                                "selector": selector,
                                "k_budget": k,
                                "slot_budget": s,
                                "alpha_regime": r["regime"],
                                "deployable_alpha": bool(r["deployable"]),
                                "oracle_alpha": bool(r["is_oracle"]),
                                "support_nnz": nnz,
                                "is_reference_full_support": is_ref,
                            }
                        )
                    row_records.extend(triples)

    rows = pd.DataFrame(row_records)
    atomic_write_csv(destination / "physical_selected_output_rows.csv", rows)
    atomic_write_csv(destination / "alpha_regime_summary.csv", pd.DataFrame(alpha_regime_records))
    atomic_write_csv(
        destination / "physical_accuracy_failures.csv",
        pd.DataFrame(failure_records)
        if failure_records
        else pd.DataFrame(columns=["structure_id", "selector", "stage", "reason"]),
    )

    summary = _summarize(rows)
    atomic_write_csv(destination / "physical_selected_output_summary.csv", summary)
    decoupling = _decoupling_analysis(rows)
    atomic_write_json(destination / "support_vs_physical_decoupling.json", decoupling)

    atomic_write_json(
        destination / "physical_accuracy_provenance.json",
        provenance_block(config_path, config)
        | {"study_id": STUDY_ID, "claim_boundary": CLAIM_BOUNDARY},
    )
    return {
        "rows": len(rows),
        "structures": structure_ids,
        "selectors": selectors,
        "failures": len(failure_records),
        "decoupling_spearman": decoupling.get("overall", {}).get("spearman_support_vs_physical"),
    }


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out: list[dict[str, Any]] = []
    keys = ["structure_id", "case", "selector", "k_budget", "slot_budget", "alpha_regime"]
    for key, group in rows.groupby(keys, dropna=False):
        non_zero = group[~group["near_zero_y_true"]]
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "alpha": float(group["alpha"].iloc[0]),
                "n_rows": len(group),
                "n_near_zero_y_true": int(group["near_zero_y_true"].sum()),
                "support_nnz": int(group["support_nnz"].iloc[0]),
                "median_E_physical_norm": float(non_zero["E_physical_norm"].median())
                if len(non_zero)
                else float("nan"),
                "mean_E_physical_norm": float(non_zero["E_physical_norm"].mean())
                if len(non_zero)
                else float("nan"),
                "median_E_support_norm": float(group["E_support_norm"].median()),
                "median_E_full_abs": float(group["E_full_abs"].median()),
                "median_E_sparse_abs": float(group["E_sparse_abs"].median()),
                "mean_B_physical_signed": float(group["B_physical_signed"].mean()),
            }
        )
    return pd.DataFrame(out)


def _decoupling_analysis(rows: pd.DataFrame) -> dict[str, Any]:
    """Does support fidelity predict physical accuracy?  Rank corr E_support vs E_physical."""

    from scipy.stats import pearsonr, spearmanr

    def corr(frame: pd.DataFrame) -> dict[str, Any]:
        valid = frame[np.isfinite(frame["E_support_norm"]) & np.isfinite(frame["E_physical_norm"])]
        valid = valid[~valid["near_zero_y_true"]]
        if len(valid) < 5:
            return {"n": len(valid), "spearman_support_vs_physical": None, "pearson": None}
        sp = spearmanr(valid["E_support_norm"], valid["E_physical_norm"])
        pe = pearsonr(valid["E_support_norm"], valid["E_physical_norm"])
        return {
            "n": len(valid),
            "spearman_support_vs_physical": float(sp.statistic),
            "spearman_pvalue": float(sp.pvalue),
            "pearson": float(pe.statistic),
            "median_E_full_abs_floor": float(valid["E_full_abs"].median()),
            "median_E_sparse_abs": float(valid["E_sparse_abs"].median()),
        }

    result: dict[str, Any] = {"overall": corr(rows)}
    for sid, group in rows.groupby("structure_id"):
        result[sid] = corr(group)
    result["interpretation"] = (
        "A weak/insignificant rank correlation means low sparse-to-full Ridge fidelity does NOT "
        "imply low error against the true physical output; physical accuracy is governed largely "
        "by block-truncation + regularization floor (E_full_abs) that support fidelity cannot see."
    )
    return result
