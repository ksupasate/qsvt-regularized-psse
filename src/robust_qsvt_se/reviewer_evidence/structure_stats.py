"""Phase 5 - structure- and case-aware statistical reanalysis.

The statistical unit is the block STRUCTURE (not the residual row, functional, alpha cell, or degree
row); the IEEE case is the clustering variable; the seed realization is a repeated observation
within a structure.  Reports structure-level paired differences, case-stratified + hierarchical
bootstrap,
leave-one-case-out, per-functional-family effects, and tie-tolerance / normalization-floor
sensitivity.  A confidence interval touching zero is never called conclusive.  With only three
structures over two cases the structural evidence is inherently limited, and that limit is stated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    atomic_write_csv,
    atomic_write_json,
    load_config,
)

STUDY_ID = "reviewer_evidence_structure_stats_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")
DEFAULT_CONFIG_PATH = Path("configs/reviewer_blocking_tqe_evidence/structure_stats.json")

STRUCTURE_TO_CASE = {"ieee14_8x8": "ieee14", "ieee30_8x8": "ieee30", "ieee14_16x16": "ieee14"}


def _e_physical(e_sparse_abs: np.ndarray, y_true: np.ndarray, floor: float) -> np.ndarray:
    return e_sparse_abs / np.maximum(np.abs(y_true), floor)


def _paired_effects(
    rows: pd.DataFrame,
    proposed: str,
    baseline: str,
    alpha_regime: str,
    k_budget: int,
    floor: float,
) -> pd.DataFrame:
    """Per (structure, seed, functional): effect = E_physical(baseline) - E_physical(proposed)."""

    sub = rows[(rows.alpha_regime == alpha_regime) & (rows.k_budget == k_budget)]
    sub = sub[~sub.near_zero_y_true]
    p = sub[sub.selector == proposed].copy()
    b = sub[sub.selector == baseline].copy()
    for frame in (p, b):
        frame["E_phys"] = _e_physical(
            frame["E_sparse_abs"].to_numpy(), frame["y_true"].to_numpy(), floor
        )
    merged = p.merge(
        b[["structure_id", "seed", "functional_id", "E_phys"]],
        on=["structure_id", "seed", "functional_id"],
        suffixes=("_proposed", "_baseline"),
    )
    merged["effect"] = (
        merged["E_phys_baseline"] - merged["E_phys_proposed"]
    )  # >0 => proposed better
    merged["case"] = merged["structure_id"].map(STRUCTURE_TO_CASE)
    return merged


def _structure_level(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby("structure_id")
        .agg(
            case=("case", "first"),
            mean_effect=("effect", "mean"),
            median_effect=("effect", "median"),
            n_obs=("effect", "size"),
        )
        .reset_index()
    )


def _bootstrap_over_structures(
    structure_effects: np.ndarray, rng: np.random.Generator, n: int
) -> dict[str, Any]:
    if len(structure_effects) == 0:
        return {"n_structures": 0}
    means = np.array(
        [
            np.mean(rng.choice(structure_effects, size=len(structure_effects), replace=True))
            for _ in range(n)
        ]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_structures": len(structure_effects),
        "mean_structural_effect": float(np.mean(structure_effects)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci_touches_zero": bool(lo <= 0.0 <= hi),
    }


def _hierarchical_bootstrap(
    merged: pd.DataFrame, rng: np.random.Generator, n: int
) -> dict[str, Any]:
    """Resample cases -> structures within case -> realizations within structure."""

    cases = merged["case"].unique().tolist()
    struct_by_case = {c: merged[merged.case == c]["structure_id"].unique().tolist() for c in cases}
    means = []
    for _ in range(n):
        sampled_cases = rng.choice(cases, size=len(cases), replace=True)
        effects = []
        for c in sampled_cases:
            structs = struct_by_case[c]
            sampled_structs = rng.choice(structs, size=len(structs), replace=True)
            for sidx in sampled_structs:
                pool = merged[merged.structure_id == sidx]["effect"].to_numpy()
                if len(pool):
                    effects.append(np.mean(rng.choice(pool, size=len(pool), replace=True)))
        if effects:
            means.append(np.mean(effects))
    means = np.array(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_cases": len(cases),
        "mean_effect": float(np.mean(means)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci_touches_zero": bool(lo <= 0.0 <= hi),
    }


def _leave_one_case_out(merged: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for case in merged["case"].unique():
        kept = merged[merged.case != case]
        struct_eff = _structure_level(kept)["mean_effect"].to_numpy()
        out[f"excluding_{case}"] = {
            "structures_kept": kept["structure_id"].nunique(),
            "mean_structural_effect": float(np.mean(struct_eff))
            if len(struct_eff)
            else float("nan"),
        }
    return out


def _interpret(ci: dict[str, Any]) -> str:
    if ci.get("n_structures", 1) <= 1 and "n_structures" in ci:
        return "inconclusive (<=1 independent structure)"
    if ci.get("ci_touches_zero", True):
        mean = ci.get("mean_structural_effect", ci.get("mean_effect", 0.0))
        return "evidence_against_uniform_improvement" if abs(mean) < 1e-3 else "inconclusive"
    mean = ci.get("mean_structural_effect", ci.get("mean_effect", 0.0))
    return (
        "moderate_structural_evidence_favoring_proposed"
        if mean > 0
        else "moderate_structural_evidence_against_proposed"
    )


def run_structure_stats(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output_dir)
    rows = pd.read_csv(destination / "physical_selected_output_rows.csv")

    proposed = config["proposed_selector"]
    baseline = config["baseline_selector"]
    alpha_regime = config["alpha_regime"]
    k_budget = int(config["k_budget"])
    floor = float(config.get("normalization_floor_primary", 1e-6))
    n_boot = int(config.get("bootstrap_resamples", 10000))
    rng = np.random.default_rng(int(config.get("bootstrap_seed", 20240714)))

    merged = _paired_effects(rows, proposed, baseline, alpha_regime, k_budget, floor)
    structure_level = _structure_level(merged)
    atomic_write_csv(destination / "structure_aware_statistics.csv", structure_level)

    struct_effects = structure_level["mean_effect"].to_numpy()
    primary = _bootstrap_over_structures(struct_effects, rng, n_boot)
    hierarchical = _hierarchical_bootstrap(merged, rng, n_boot)
    loco = _leave_one_case_out(merged)

    family = (
        merged.groupby("functional_family")
        .agg(
            mean_effect=("effect", "mean"),
            median_effect=("effect", "median"),
            n=("effect", "size"),
            frac_proposed_better=("effect", lambda x: float(np.mean(x > 0))),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # Sensitivity: normalization floor.
    floor_sensitivity = {}
    for f in config.get("normalization_floor_sensitivity", [1e-4, 1e-8]):
        m = _paired_effects(rows, proposed, baseline, alpha_regime, k_budget, float(f))
        se = _structure_level(m)["mean_effect"].to_numpy()
        floor_sensitivity[str(f)] = _bootstrap_over_structures(
            se, np.random.default_rng(int(config.get("bootstrap_seed", 20240714))), n_boot
        )

    # Sensitivity: tie tolerance (win/tie/loss on the primary effect).
    tie_sensitivity = {}
    for tol in config.get("tie_tolerance_sensitivity", [0.0, 1e-6, 1e-3]):
        wins = int(np.sum(merged["effect"] > tol))
        losses = int(np.sum(merged["effect"] < -tol))
        ties = int(np.sum(np.abs(merged["effect"]) <= tol))
        tie_sensitivity[str(tol)] = {
            "proposed_wins": wins,
            "ties": ties,
            "proposed_losses": losses,
            "win_fraction": float(wins / max(len(merged), 1)),
        }

    result = {
        "study_id": STUDY_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "contrast": {
            "proposed": proposed,
            "baseline": baseline,
            "alpha_regime": alpha_regime,
            "k_budget": k_budget,
            "metric": "physical_output_error (E_physical_norm); effect = baseline-proposed",
        },
        "statistical_units": {
            "primary": "block_structure",
            "cluster": "ieee_case",
            "repeated_observation": "seed_realization",
            "n_structures": int(structure_level.shape[0]),
            "n_cases": int(merged["case"].nunique()),
        },
        "structure_level_effects": structure_level.to_dict(orient="records"),
        "primary_bootstrap_over_structures": primary,
        "hierarchical_bootstrap": hierarchical,
        "leave_one_case_out": loco,
        "per_functional_family": family,
        "normalization_floor_sensitivity": floor_sensitivity,
        "tie_tolerance_sensitivity": tie_sensitivity,
        "interpretation_primary": _interpret(primary),
        "limitation": (
            "Only 3 block structures over 2 IEEE cases are available as independent units; the "
            "structural bootstrap is therefore low-powered and its intervals are wide. The frozen "
            "per-observation result is the primary record; these structure-level intervals are a "
            "secondary robustness diagnostic and cannot, on their own, establish uniform "
            "structural improvement."
        ),
    }
    atomic_write_json(destination / "structure_aware_statistics.json", result)
    return {
        "n_structures": int(structure_level.shape[0]),
        "interpretation": result["interpretation_primary"],
        "primary_ci": [primary.get("ci95_low"), primary.get("ci95_high")],
    }
