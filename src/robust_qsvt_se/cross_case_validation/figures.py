"""Canonical figure data + rendering for the transfer study (Figures A-D).

Figure data CSVs are written before rendering.  Figures never show only favorable selectors or
budgets: every evaluated selector and threshold is included, with infeasible cells marked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.reviewer_blocking.common import atomic_write_csv

SELECTOR_ORDER = [
    "global_magnitude", "balanced_magnitude", "ridge_leverage",
    "sensitivity_initial_mean", "sensitivity_refined_mean",
    "exact_loss_greedy_mean", "near_oracle_mean",
]
FAMILY_ORDER = ["coordinate", "branch_difference", "aggregate"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ------------------------------------------------------------------- Figure A


def figure_a_cross_case_selectors(cross_case_dir: Path, figures_dir: Path, data_dir: Path) -> None:
    """Held-out error per selector, per physical functional family (IEEE-30)."""

    tasks = _read_csv(cross_case_dir / "raw_task_results.csv")
    if tasks.empty:
        return
    held = tasks[tasks["split"] == "held_out"].copy()
    # Per selector x family mean held-out normalized error, minimized over budget cells first.
    per_cell = (
        held.groupby(["selector", "family", "support_id"])["normalized_error"].mean().reset_index()
    )
    agg = per_cell.groupby(["selector", "family"])["normalized_error"].min().reset_index()
    atomic_write_csv(data_dir / "figure_a_cross_case_selector_family.csv", agg)

    selectors = [s for s in SELECTOR_ORDER if s in set(agg["selector"])]
    families = [f for f in FAMILY_ORDER if f in set(agg["family"])]
    if not selectors or not families:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / max(len(families), 1)
    x = np.arange(len(selectors))
    for i, family in enumerate(families):
        vals = [
            float(agg[(agg["selector"] == s) & (agg["family"] == family)]["normalized_error"].min())
            if not agg[(agg["selector"] == s) & (agg["family"] == family)].empty else np.nan
            for s in selectors
        ]
        ax.bar(x + i * width, vals, width, label=family)
    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(selectors, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("best held-out normalized error")
    ax.set_yscale("log")
    ax.set_title("Figure A - IEEE-30 cross-case selector comparison (per physical family)")
    ax.legend(fontsize=8, title="functional family")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_a_cross_case_selectors.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- Figure B


def figure_b_threshold_cost(cross_case_dir: Path, figures_dir: Path, data_dir: Path) -> None:
    """Output-aware vs output-agnostic min C_total at each predeclared threshold (IEEE-30)."""

    summary = _read_csv(cross_case_dir / "threshold_cost_summary.csv")
    if summary.empty:
        return
    atomic_write_csv(data_dir / "figure_b_threshold_cost.csv", summary)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    thresholds = summary["error_threshold"].tolist()
    x = np.arange(len(thresholds))
    aware = summary["output_aware_min_c_total"].to_numpy(dtype=float)
    agnostic = summary["output_agnostic_min_c_total"].to_numpy(dtype=float)
    finite_aware = np.where(np.isfinite(aware), aware, np.nan)
    finite_agnostic = np.where(np.isfinite(agnostic), agnostic, np.nan)
    ax.bar(x - 0.2, finite_aware, 0.4, label="output-aware", color="#2166ac")
    ax.bar(x + 0.2, finite_agnostic, 0.4, label="output-agnostic", color="#b2182b")
    for i, (a, g) in enumerate(zip(aware, agnostic, strict=True)):
        if not np.isfinite(a):
            ax.text(i - 0.2, ax.get_ylim()[0], "infeasible", rotation=90,
                    fontsize=7, ha="center", va="bottom")
        if not np.isfinite(g):
            ax.text(i + 0.2, ax.get_ylim()[0], "infeasible", rotation=90,
                    fontsize=7, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in thresholds])
    ax.set_xlabel("selected-output error threshold")
    ax.set_ylabel("min modeled C_total (gates / accepted sample)")
    ax.set_yscale("log")
    ax.set_title("Figure B - IEEE-30 resource cost at fixed selected-output error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_b_threshold_cost.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- Figure C


def figure_c_feasibility_map(cross_case_dir: Path, figures_dir: Path, data_dir: Path) -> None:
    """Utility x QSVT-feasibility four-quadrant map (IEEE-30), empty region made visible."""

    grid = _read_csv(cross_case_dir / "joint_feasibility_grid.csv")
    if grid.empty:
        return
    plot = grid[[
        "normalized_lambda", "rmse_ratio_to_oracle_best", "qsvt_feasible",
        "application_useful_full_state", "region",
    ]].copy()
    atomic_write_csv(data_dir / "figure_c_feasibility_map.csv", plot)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = {True: "#1a9850", False: "#d73027"}
    for feasible, group in plot.groupby("qsvt_feasible"):
        ax.scatter(
            group["normalized_lambda"], group["rmse_ratio_to_oracle_best"],
            s=26, alpha=0.7, color=colors.get(bool(feasible), "gray"),
            label=f"QSVT feasible={bool(feasible)}",
        )
    ax.axhline(1.5, color="black", ls="--", lw=1, label="useful RMSE ratio <= 1.5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("normalized lambda = alpha / beta^2")
    ax.set_ylabel("full-state RMSE ratio to oracle-best")
    n_uf = int(((plot["qsvt_feasible"]) & (plot["application_useful_full_state"])).sum())
    ax.set_title(f"Figure C - IEEE-30 utility x QSVT feasibility (jointly useful+feasible: {n_uf})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_c_feasibility_map.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- Figure D


def figure_d_scaling(comparison_dir: Path, figures_dir: Path, data_dir: Path,
                     ieee14_8x8_dir: Path, ieee14_16x16_dir: Path) -> None:
    """8x8 vs 16x16 scaling: normalized error, runtime, stability, feasibility."""

    scaling = _read_csv(comparison_dir / "8x8_vs_16x16.csv")
    if scaling.empty:
        return
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (1) normalized error vs relative support density (k/nnz)
    ax = axes[0, 0]
    for block, group in scaling.groupby("block"):
        agg = group.groupby("relative_support_density_k_over_nnz")[
            "heldout_mean_normalized_error"
        ].min().reset_index()
        ax.plot(agg["relative_support_density_k_over_nnz"], agg["heldout_mean_normalized_error"],
                marker="o", label=block)
    ax.set_xlabel("relative support density (k / candidate nnz)")
    ax.set_ylabel("best held-out normalized error")
    ax.set_yscale("log")
    ax.set_title("(a) error vs relative support density")
    ax.legend(fontsize=8)

    # (2) runtime vs candidate nnz (per selector, total)
    ax = axes[0, 1]
    rt = scaling.groupby(["block", "candidate_nonzeros", "selector"])[
        "selection_runtime_seconds"
    ].sum().reset_index()
    for selector, group in rt.groupby("selector"):
        ax.plot(group["candidate_nonzeros"], group["selection_runtime_seconds"],
                marker="s", label=selector)
    ax.set_xlabel("candidate nonzeros |Omega|")
    ax.set_ylabel("total selection runtime (s)")
    ax.set_yscale("log")
    ax.set_title("(b) runtime vs |Omega|")
    ax.legend(fontsize=6)

    # (3) support stability (mean pairwise Jaccard)
    ax = axes[1, 0]
    stab8 = _read_csv(ieee14_8x8_dir / "support_stability.csv")
    stab16 = _read_csv(ieee14_16x16_dir / "support_stability.csv")
    frames = []
    for label, s in (("8x8", stab8), ("16x16", stab16)):
        if not s.empty:
            s = s.copy()
            s["block"] = label
            frames.append(s[["block", "selector", "mean_pairwise_jaccard"]])
    if frames:
        stab = pd.concat(frames, ignore_index=True)
        atomic_write_csv(data_dir / "figure_d_support_stability.csv", stab)
        selectors = sorted(set(stab["selector"]))
        x = np.arange(len(selectors))
        for i, block in enumerate(["8x8", "16x16"]):
            vals = [
                float(stab[(stab["block"] == block)
                           & (stab["selector"] == s)]["mean_pairwise_jaccard"].mean())
                if not stab[(stab["block"] == block) & (stab["selector"] == s)].empty else np.nan
                for s in selectors
            ]
            ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, label=block)
        ax.set_xticks(x)
        ax.set_xticklabels(selectors, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("mean pairwise Jaccard")
    ax.set_title("(c) support stability")
    ax.legend(fontsize=8)

    # (4) feasibility rate
    ax = axes[1, 1]
    feas = []
    for label, d in (("8x8", ieee14_8x8_dir), ("16x16", ieee14_16x16_dir)):
        raw = _read_csv(d / "raw_selector_results.csv")
        if not raw.empty:
            mean = raw[raw["objective"] == "mean"]
            feas.append({"block": label, "feasibility_rate": float(mean["feasible"].mean())})
    if feas:
        fdf = pd.DataFrame(feas)
        atomic_write_csv(data_dir / "figure_d_feasibility.csv", fdf)
        ax.bar(fdf["block"], fdf["feasibility_rate"], color=["#2166ac", "#b2182b"])
    ax.set_ylabel("feasibility rate (mean objective)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) feasibility rate (coverage floor k=16 for 16x16)")

    atomic_write_csv(data_dir / "figure_d_scaling.csv", scaling)
    fig.suptitle("Figure D - IEEE-14 8x8 vs 16x16 scaling", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_d_scaling.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- orchestrator


def render_all_figures(
    root: str | Path = Path("outputs/cross_case_larger_block_validation"),
) -> dict[str, Any]:
    root = Path(root)
    cross_case_dir = root / "cross_case"
    ieee14_dir = root / "ieee14_8x8_reference"
    ieee14_16x16_dir = root / "larger_block_16x16"
    comparison_dir = root / "comparison"

    cross_fig = cross_case_dir / "figures"
    cross_data = cross_case_dir / "figure_data"
    comp_fig = comparison_dir / "figures"
    comp_data = comparison_dir / "figure_data"
    for directory in (cross_fig, cross_data, comp_fig, comp_data):
        directory.mkdir(parents=True, exist_ok=True)

    figure_a_cross_case_selectors(cross_case_dir, cross_fig, cross_data)
    figure_b_threshold_cost(cross_case_dir, cross_fig, cross_data)
    figure_c_feasibility_map(cross_case_dir, cross_fig, cross_data)
    figure_d_scaling(comparison_dir, comp_fig, comp_data, ieee14_dir, ieee14_16x16_dir)
    return {"figures": ["figure_a", "figure_b", "figure_c", "figure_d"]}
