"""Cross-track comparison: IEEE-14 vs IEEE-30 (8x8) and 8x8 vs 16x16 (IEEE-14).

Consumes the three frozen-protocol result directories and produces paired, normalized
comparison tables plus a claim-support matrix.  Uses paired (selector, block) comparisons within
identical residual-functional tasks; residual seeds / grid cells are never treated as independent
systems.  Descriptive transfer language only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.reviewer_blocking.common import CLAIM_BOUNDARY, atomic_write_csv

STUDY_ID = "cross_case_larger_block_comparison_v1"
THRESHOLDS = (0.5, 0.6, 0.75, 1.0, 1.5)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _mean_summary(directory: Path) -> pd.DataFrame:
    summary = _read_csv(directory / "selector_summary.csv")
    if summary.empty:
        return summary
    return summary[summary["objective"] == "mean"].copy()


def _near_oracle_gap(directory: Path) -> pd.DataFrame:
    gaps = _read_csv(directory / "oracle_gap_summary.csv")
    if gaps.empty:
        return gaps
    mean_gaps = gaps[gaps["objective"] == "mean"]
    return mean_gaps.groupby("selector", as_index=False)["optimality_gap"].mean()


# ------------------------------------------------------- IEEE-14 vs IEEE-30 (8x8)


def build_ieee14_vs_new_case(
    ieee14_dir: Path, ieee30_dir: Path, destination: Path
) -> pd.DataFrame:
    a = _mean_summary(ieee14_dir).rename(columns={
        "mean_heldout_normalized_error": "ieee14_mean_heldout_error",
        "feasibility_rate": "ieee14_feasibility_rate",
        "mean_overlap_with_near_oracle": "ieee14_overlap_near_oracle",
    })
    b = _mean_summary(ieee30_dir).rename(columns={
        "mean_heldout_normalized_error": "ieee30_mean_heldout_error",
        "feasibility_rate": "ieee30_feasibility_rate",
        "mean_overlap_with_near_oracle": "ieee30_overlap_near_oracle",
    })
    cols_a = ["selector", "ieee14_mean_heldout_error", "ieee14_feasibility_rate",
              "ieee14_overlap_near_oracle"]
    cols_b = ["selector", "ieee30_mean_heldout_error", "ieee30_feasibility_rate",
              "ieee30_overlap_near_oracle"]
    merged = pd.merge(
        a[cols_a] if not a.empty else pd.DataFrame(columns=cols_a),
        b[cols_b] if not b.empty else pd.DataFrame(columns=cols_b),
        on="selector", how="outer",
    )
    # Near-oracle gap per selector on each case.
    ga = _near_oracle_gap(ieee14_dir).rename(columns={"optimality_gap": "ieee14_near_oracle_gap"})
    gb = _near_oracle_gap(ieee30_dir).rename(columns={"optimality_gap": "ieee30_near_oracle_gap"})
    if not ga.empty:
        merged = merged.merge(ga, on="selector", how="left")
    if not gb.empty:
        merged = merged.merge(gb, on="selector", how="left")
    merged = merged.sort_values("selector").reset_index(drop=True)
    atomic_write_csv(destination / "ieee14_vs_new_case.csv", merged)
    return merged


def _selector_advantage(summary: pd.DataFrame) -> dict[str, float]:
    """Best magnitude/leverage vs best sensitivity vs near-oracle held-out error."""

    if summary.empty:
        return {}
    err = summary.set_index("selector")["mean_heldout_normalized_error"].to_dict()
    agnostic = [err.get(s) for s in ("global_magnitude", "balanced_magnitude", "ridge_leverage")
                if err.get(s) is not None]
    sensitivity = [err.get(s) for s in ("sensitivity_initial_mean", "sensitivity_refined_mean")
                   if err.get(s) is not None]
    return {
        "best_agnostic": float(min(agnostic)) if agnostic else float("nan"),
        "best_sensitivity": float(min(sensitivity)) if sensitivity else float("nan"),
        "near_oracle": float(err.get("near_oracle_mean", float("nan"))),
    }


# ------------------------------------------------------- 8x8 vs 16x16 (IEEE-14)


def build_8x8_vs_16x16(
    ieee14_8x8_dir: Path, ieee14_16x16_dir: Path, destination: Path
) -> pd.DataFrame:
    block8 = _read_json(ieee14_8x8_dir / "block_inventory.json")
    block16 = _read_json(ieee14_16x16_dir / "block_inventory.json")
    nnz8 = int(block8.get("conditioning", {}).get("nonzeros", 0)) or 1
    nnz16 = int(block16.get("conditioning", {}).get("nonzeros", 0)) or 1

    raw8 = _read_csv(ieee14_8x8_dir / "raw_selector_results.csv")
    raw16 = _read_csv(ieee14_16x16_dir / "raw_selector_results.csv")
    rows: list[dict[str, Any]] = []
    for label, raw, nnz, dim in (("8x8", raw8, nnz8, 8), ("16x16", raw16, nnz16, 16)):
        if raw.empty:
            continue
        feas = raw[(raw["feasible"]) & (raw["objective"] == "mean")]
        for _, r in feas.iterrows():
            rows.append({
                "block": label, "dimension": dim, "candidate_nonzeros": nnz,
                "selector": r["selector"], "k_budget": int(r["k_budget"]),
                "slot_budget": int(r["slot_budget"]),
                "actual_nonzeros": int(r["actual_nonzeros"]),
                "relative_support_density_k_over_nnz": float(r["actual_nonzeros"]) / nnz,
                "relative_support_density_k_over_dim": float(r["actual_nonzeros"]) / dim,
                "heldout_mean_normalized_error": float(r["heldout_mean_normalized_error"]),
                "selection_runtime_seconds": float(r["selection_runtime_seconds"]),
                "exact_ridge_solves_this_cell": float(
                    r.get("exact_ridge_solves_this_cell", np.nan)
                ),
            })
    scaling = pd.DataFrame(rows)
    atomic_write_csv(destination / "8x8_vs_16x16.csv", scaling)
    return scaling


def _feasibility_by_block(directory: Path) -> dict[str, Any]:
    raw = _read_csv(directory / "raw_selector_results.csv")
    if raw.empty:
        return {}
    mean = raw[raw["objective"] == "mean"]
    return {
        "total_cells": len(mean),
        "feasible_cells": int(mean["feasible"].sum()),
        "feasibility_rate": float(mean["feasible"].mean()),
        "coverage_infeasible_cells": int((~mean["feasible"]).sum()),
    }


# ------------------------------------------------------- claim support matrix


def build_claim_support_matrix(
    ieee14_dir: Path, ieee30_dir: Path, ieee14_16x16_dir: Path, destination: Path
) -> pd.DataFrame:
    s14 = _selector_advantage(_mean_summary(ieee14_dir))
    s30 = _selector_advantage(_mean_summary(ieee30_dir))
    s16 = _selector_advantage(_mean_summary(ieee14_16x16_dir))
    q30 = _read_csv(ieee30_dir / "joint_feasibility_summary.csv")
    q14 = _read_csv(ieee14_dir / "joint_feasibility_summary.csv")
    t30 = _read_csv(ieee30_dir / "threshold_cost_summary.csv")
    t14 = _read_csv(ieee14_dir / "threshold_cost_summary.csv")

    def _q_useful_feasible(frame: pd.DataFrame) -> int:
        return int(frame["N_useful_and_feasible"].iloc[0]) if not frame.empty else -1

    def _sensitivity_beats_agnostic(adv: dict[str, float]) -> bool | None:
        if not adv or np.isnan(adv.get("best_sensitivity", np.nan)) or np.isnan(
            adv.get("best_agnostic", np.nan)
        ):
            return None
        return bool(adv["best_sensitivity"] < adv["best_agnostic"])

    def _threshold_verdict(frame: pd.DataFrame, threshold: float) -> str:
        if frame.empty:
            return "unknown"
        row = frame[frame["error_threshold"] == threshold]
        return str(row["verdict"].iloc[0]) if not row.empty else "unknown"

    rows: list[dict[str, Any]] = []

    # RQ1a: sensitivity beats magnitude/leverage on the new case.
    ref = _sensitivity_beats_agnostic(s14)
    new = _sensitivity_beats_agnostic(s30)
    rows.append({
        "claim": "sensitivity_beats_magnitude_leverage",
        "ieee14_8x8": _fmt_bool(ref), "new_case_ieee30_8x8": _fmt_bool(new),
        "ieee14_16x16": _fmt_bool(_sensitivity_beats_agnostic(s16)),
        "replicated_on_new_case": _replication(ref, new),
        "ieee14_best_agnostic": s14.get("best_agnostic"),
        "ieee14_best_sensitivity": s14.get("best_sensitivity"),
        "ieee30_best_agnostic": s30.get("best_agnostic"),
        "ieee30_best_sensitivity": s30.get("best_sensitivity"),
    })
    # RQ1b: near-oracle strictly better than deployable selectors (gap exists).
    def _near_oracle_best(adv: dict[str, float]) -> bool | None:
        if not adv or np.isnan(adv.get("near_oracle", np.nan)):
            return None
        deployable = [adv.get("best_agnostic"), adv.get("best_sensitivity")]
        deployable = [d for d in deployable if d is not None and not np.isnan(d)]
        return bool(deployable and adv["near_oracle"] <= min(deployable) + 1e-9)
    rows.append({
        "claim": "near_oracle_gap_present",
        "ieee14_8x8": _fmt_bool(_near_oracle_best(s14)),
        "new_case_ieee30_8x8": _fmt_bool(_near_oracle_best(s30)),
        "ieee14_16x16": _fmt_bool(_near_oracle_best(s16)),
        "replicated_on_new_case": _replication(_near_oracle_best(s14), _near_oracle_best(s30)),
        "ieee14_near_oracle": s14.get("near_oracle"), "ieee30_near_oracle": s30.get("near_oracle"),
    })
    # RQ1c: no jointly useful+QSVT-feasible grid cell.
    uf14, uf30 = _q_useful_feasible(q14), _q_useful_feasible(q30)
    rows.append({
        "claim": "no_jointly_useful_and_feasible_cell",
        "ieee14_8x8": _fmt_bool(uf14 == 0) if uf14 >= 0 else "unknown",
        "new_case_ieee30_8x8": _fmt_bool(uf30 == 0) if uf30 >= 0 else "unknown",
        "ieee14_16x16": "not_evaluated",
        "replicated_on_new_case": (
            _replication(uf14 == 0, uf30 == 0) if (uf14 >= 0 and uf30 >= 0) else "unknown"
        ),
        "ieee14_N_useful_feasible": uf14, "ieee30_N_useful_feasible": uf30,
    })
    # RQ1d: resource benefit at tightest threshold 0.5.
    v14_05 = _threshold_verdict(t14, 0.5)
    v30_05 = _threshold_verdict(t30, 0.5)
    rows.append({
        "claim": "output_aware_cheaper_at_threshold_0p5",
        "ieee14_8x8": v14_05, "new_case_ieee30_8x8": v30_05, "ieee14_16x16": "not_evaluated",
        "replicated_on_new_case": _replication(
            v14_05 == "output_aware_cheaper", v30_05 == "output_aware_cheaper"
        ),
    })
    # RQ1e: resource ranking reverses / disappears at loose thresholds.
    v14_15 = _threshold_verdict(t14, 1.5)
    v30_15 = _threshold_verdict(t30, 1.5)
    rows.append({
        "claim": "resource_ranking_not_output_aware_at_loose_threshold_1p5",
        "ieee14_8x8": v14_15, "new_case_ieee30_8x8": v30_15, "ieee14_16x16": "not_evaluated",
        "replicated_on_new_case": _replication(
            v14_15 != "output_aware_cheaper", v30_15 != "output_aware_cheaper"
        ),
    })
    frame = pd.DataFrame(rows)
    atomic_write_csv(destination / "claim_support_matrix.csv", frame)
    return frame


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _replication(reference: bool | None, new: bool | None) -> str:
    if reference is None or new is None:
        return "inconclusive"
    if reference and new:
        return "replicated"
    if reference and not new:
        return "did_not_replicate"
    if not reference and new:
        return "new_case_only"
    return "absent_on_both"


# ------------------------------------------------------- orchestrator


def build_all_comparisons(
    root: str | Path = Path("outputs/cross_case_larger_block_validation"),
) -> dict[str, Any]:
    root = Path(root)
    ieee14_dir = root / "ieee14_8x8_reference"
    ieee30_dir = root / "cross_case"
    ieee14_16x16_dir = root / "larger_block_16x16"
    destination = root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)

    cross = build_ieee14_vs_new_case(ieee14_dir, ieee30_dir, destination)
    scaling = build_8x8_vs_16x16(ieee14_dir, ieee14_16x16_dir, destination)
    claims = build_claim_support_matrix(ieee14_dir, ieee30_dir, ieee14_16x16_dir, destination)
    feas8 = _feasibility_by_block(ieee14_dir)
    feas16 = _feasibility_by_block(ieee14_16x16_dir)
    _write_comparison_report(
        destination, ieee14_dir, ieee30_dir, ieee14_16x16_dir,
        cross, scaling, claims, feas8, feas16,
    )
    return {
        "ieee14_vs_new_case_rows": len(cross),
        "scaling_rows": len(scaling),
        "claims": len(claims),
        "feasibility_8x8": feas8,
        "feasibility_16x16": feas16,
    }


def _write_comparison_report(
    destination, ieee14_dir, ieee30_dir, ieee14_16x16_dir,
    cross, scaling, claims, feas8, feas16,
) -> None:
    block8 = _read_json(ieee14_dir / "block_inventory.json")
    block16 = _read_json(ieee14_16x16_dir / "block_inventory.json")
    block30 = _read_json(ieee30_dir / "block_inventory.json")
    lines = [
        "# Cross-Case and Larger-Block Comparison", "", CLAIM_BOUNDARY, "",
        "## Structures compared (paired within identical residual-functional tasks)", "",
        f"- IEEE-14 8x8 (reference): rank {block8.get('conditioning', {}).get('rank')}/8, "
        f"nnz {block8.get('conditioning', {}).get('nonzeros')}",
        f"- IEEE-30 8x8 (new case): rank {block30.get('conditioning', {}).get('rank')}/8, "
        f"nnz {block30.get('conditioning', {}).get('nonzeros')}, "
        f"raw kappa {block30.get('conditioning', {}).get('raw_condition_number')}",
        f"- IEEE-14 16x16 (larger block): rank {block16.get('conditioning', {}).get('rank')}/16, "
        f"nnz {block16.get('conditioning', {}).get('nonzeros')}",
        "",
        "## Claim support matrix (transfer)", "",
        "| claim | IEEE-14 8x8 | IEEE-30 8x8 | replication |",
        "|---|---|---|---|",
    ]
    for _, row in claims.iterrows():
        lines.append(
            f"| {row['claim']} | {row['ieee14_8x8']} | {row['new_case_ieee30_8x8']} | "
            f"**{row['replicated_on_new_case']}** |"
        )
    lines += [
        "",
        "## IEEE-14 vs IEEE-30 selector held-out error (mean objective)", "",
        "| selector | IEEE-14 | IEEE-30 |",
        "|---|---:|---:|",
    ]
    if not cross.empty:
        for _, row in cross.sort_values("selector").iterrows():
            e14 = row.get("ieee14_mean_heldout_error")
            e30 = row.get("ieee30_mean_heldout_error")
            lines.append(
                f"| `{row['selector']}` | "
                f"{e14:.4g} | {e30:.4g} |" if pd.notna(e14) and pd.notna(e30)
                else f"| `{row['selector']}` | {e14} | {e30} |"
            )
    lines += [
        "",
        "## 8x8 vs 16x16 scaling (IEEE-14, normalized)", "",
        f"- feasibility: 8x8 {feas8.get('feasible_cells')}/{feas8.get('total_cells')} "
        f"(rate {feas8.get('feasibility_rate')}); 16x16 {feas16.get('feasible_cells')}/"
        f"{feas16.get('total_cells')} (rate {feas16.get('feasibility_rate')}, "
        f"{feas16.get('coverage_infeasible_cells')} coverage-infeasible cells retained)",
        "- normalized error vs relative support density (k/nnz, k/dim), runtime, and exact-Ridge "
        "solve counts per cell are in `8x8_vs_16x16.csv`. Raw k is never compared without "
        "normalizing for matrix size and nonzero count.",
        "",
        "## Statistical scope", "",
        "One new IEEE-30 structure + one 16x16 IEEE-14 block. Residual seeds, functionals, and "
        "grid cells are not independent systems. Transfer language is descriptive; no "
        "population-level cross-system generalization is claimed.",
        "",
    ]
    (destination / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
