"""Phase 6 - reviewer-ready tables and figures (task-owned, with machine-readable source data).

Reads the study CSV/JSON artifacts and emits LaTeX tables under reviewer_ready_tables/ and figures
(PNG + per-figure data CSV) under reviewer_ready_figures/.  Every figure ships its own machine-
readable data file.  Purely a rendering layer over the frozen study outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")

# Colorblind-safe (Okabe-Ito) selector palette.
SELECTOR_COLOR = {
    "full_support": "#000000",
    "global_magnitude": "#E69F00",
    "balanced_magnitude": "#F0E442",
    "ridge_leverage": "#009E73",
    "sensitivity_initial_mean": "#56B4E9",
    "sensitivity_refined_mean": "#0072B2",
    "adjoint_unnormalized_mean": "#D55E00",
    "exact_single_removal_mean": "#CC79A7",
    "near_oracle_mean": "#999999",
}


def _latex_table(df: pd.DataFrame, caption: str, label: str, floatfmt: str = "{:.4g}") -> str:
    cols = list(df.columns)
    header = " & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isfinite(v):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v).replace("_", r"\_").replace("%", r"\%"))
        body.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\small",
            "\\begin{tabular}{" + "l" * len(cols) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _write(
    dst: Path, name: str, df: pd.DataFrame, caption: str, label: str, floatfmt="{:.4g}"
) -> None:
    (dst / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (dst / f"{name}.tex").write_text(_latex_table(df, caption, label, floatfmt), encoding="utf-8")


def build_tables(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[str]:
    base = Path(output_dir)
    tdir = base / "reviewer_ready_tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Table 1: physical accuracy by alpha regime (primary structure, k=16).
    summ = pd.read_csv(base / "physical_selected_output_summary.csv")
    t1 = summ[
        (summ.structure_id == "ieee14_8x8")
        & ((summ.k_budget == 16) | (summ.selector == "full_support"))
    ]
    t1 = t1[
        [
            "selector",
            "alpha_regime",
            "median_E_physical_norm",
            "median_E_support_norm",
            "median_E_full_abs",
            "median_E_sparse_abs",
        ]
    ].copy()
    t1 = t1.sort_values(["alpha_regime", "selector"])
    _write(
        tdir,
        "table_physical_by_alpha_regime",
        t1,
        "Physical selected-output accuracy vs.\\ support fidelity by regularization regime "
        "(IEEE-14 $8\\times8$, $k{=}16$). Physical error is against $y_{\\mathrm{true}}="
        "\\ell^\\top\\Delta x_{\\mathrm{true}}$; support fidelity is against full-block Ridge.",
        "tab:phys_by_alpha",
    )
    written.append("table_physical_by_alpha_regime")

    # Table 2: support fidelity vs physical accuracy (decoupling).
    dec = json.loads((base / "support_vs_physical_decoupling.json").read_text())
    rows = []
    for sid in ("ieee14_8x8", "ieee30_8x8", "ieee14_16x16", "overall"):
        v = dec.get(sid, {})
        rows.append(
            {
                "structure": sid,
                "n": v.get("n"),
                "spearman_support_vs_physical": v.get("spearman_support_vs_physical"),
                "pearson": v.get("pearson"),
                "median_E_full_floor": v.get("median_E_full_abs_floor"),
                "median_E_sparse": v.get("median_E_sparse_abs"),
            }
        )
    _write(
        tdir,
        "table_support_vs_physical",
        pd.DataFrame(rows),
        "Decoupling of support fidelity and physical accuracy: rank (Spearman) and linear "
        "(Pearson) correlation of $E_{\\mathrm{support}}$ vs.\\ $E_{\\mathrm{physical}}$, with the "
        "full-support truncation floor $E_{\\mathrm{full}}$.",
        "tab:decoupling",
    )
    written.append("table_support_vs_physical")

    # Table 3: degree feasibility comparison (if available).
    hd_path = base / "high_degree_qsvt_summary.csv"
    if hd_path.exists():
        hd = pd.read_csv(hd_path)
        t3 = hd[
            [
                "structure_id",
                "selector",
                "alpha_regime",
                "degree",
                "normalized_lambda",
                "uniform_fit_error",
                "target_max_abs",
                "analytic_bounded_fit_ok"
                if "analytic_bounded_fit_ok" in hd.columns
                else "boundedness_parity_fit_ok",
                "application_useful",
                "qsvt_feasible",
            ]
        ].copy()
        t3 = t3.sort_values(["structure_id", "selector", "alpha_regime", "degree"])
        _write(
            tdir,
            "table_degree_feasibility",
            t3,
            "High-degree QSVT feasibility slice (degrees 31/63/127/255). Under the tested "
            "polynomial construction, application-useful small-$\\lambda$ alphas fail the "
            "uniform-fit criterion at every degree; analytically feasible alphas are not "
            "application-useful.",
            "tab:degree_feas",
            floatfmt="{:.3g}",
        )
        written.append("table_degree_feasibility")

    # Table 4: task-aware baseline comparison.
    bl = pd.read_csv(base / "task_aware_baseline_summary.csv")
    t4 = bl[
        [
            "structure_id",
            "selector",
            "feasibility_fraction",
            "mean_heldout_support_fidelity",
            "mean_heldout_physical",
            "total_exact_solves",
        ]
    ].copy()
    t4 = t4.sort_values(["structure_id", "mean_heldout_support_fidelity"])
    _write(
        tdir,
        "table_task_aware_baselines",
        t4,
        "Strong task-aware baseline comparison. Held-out support-fidelity loss (lower is better) "
        "and physical error under identical MILP/budget/coverage constraints. "
        "Baselines A (adjoint\\_unnormalized) and B (exact\\_single\\_removal) are new.",
        "tab:baselines",
    )
    written.append("table_task_aware_baselines")

    # Table 5: structure-aware statistics.
    ss = json.loads((base / "structure_aware_statistics.json").read_text())
    srows = [
        {
            "structure": e["structure_id"],
            "case": e["case"],
            "mean_effect": e["mean_effect"],
            "median_effect": e["median_effect"],
            "n_obs": e["n_obs"],
        }
        for e in ss["structure_level_effects"]
    ]
    pb = ss["primary_bootstrap_over_structures"]
    srows.append(
        {
            "structure": "BOOTSTRAP_CI95",
            "case": f"n={pb['n_structures']}",
            "mean_effect": pb["mean_structural_effect"],
            "median_effect": pb["ci95_low"],
            "n_obs": pb["ci95_high"],
        }
    )
    _write(
        tdir,
        "table_structure_statistics",
        pd.DataFrame(srows),
        "Structure-aware paired effect (baseline$-$proposed physical error; positive favors "
        "proposed). Last row: bootstrap-over-structures mean and 95\\% CI "
        "(median\\_effect/n\\_obs columns hold CI low/high). CI spans zero: inconclusive.",
        "tab:struct_stats",
    )
    written.append("table_structure_statistics")

    # Table 6: physical functional mapping (compact).
    fm = pd.read_csv(base / "functional_mapping.csv")
    t6 = fm[
        [
            "structure_id",
            "functional_id",
            "family",
            "classification",
            "state_type",
            "bus_ids",
            "branch_exists_in_network",
            "connected_area",
            "unit_norm",
        ]
    ].copy()
    _write(
        tdir,
        "table_functional_mapping",
        t6,
        "Physical functional mapping: every selected functional bound to network states/branches "
        "by topology and state-index metadata only.",
        "tab:func_map",
    )
    written.append("table_functional_mapping")

    # Table 7: failure and unavailable-row ledger.
    ledger_rows = []
    pf = base / "physical_accuracy_failures.csv"
    if pf.exists():
        for _, r in pd.read_csv(pf).iterrows():
            ledger_rows.append(
                {
                    "study": "physical_accuracy",
                    "structure_id": r.get("structure_id"),
                    "item": f"{r.get('selector')} k={r.get('k_budget')}",
                    "reason": r.get("reason"),
                }
            )
    for _, r in fm[fm.classification == "unavailable_not_substituted"].iterrows():
        ledger_rows.append(
            {
                "study": "functional_mapping",
                "structure_id": r["structure_id"],
                "item": r["functional_id"],
                "reason": r["unavailable_reason"],
            }
        )
    hd_fail = base / "high_degree_failures.csv"
    if hd_fail.exists():
        hf = pd.read_csv(hd_fail)
        for _, r in hf.head(20).iterrows():
            ledger_rows.append(
                {
                    "study": "high_degree",
                    "structure_id": r.get("structure_id"),
                    "item": r.get("stage"),
                    "reason": str(r.get("reason"))[:80],
                }
            )
    _write(
        tdir,
        "table_failure_ledger",
        pd.DataFrame(ledger_rows),
        "Failure and unavailable-row ledger for infeasible supports and unavailable functionals. "
        "High-degree analytic failures remain in the main degree table with numeric criteria; "
        "this ledger contains a QSVT row only when a separate execution failure is recorded.",
        "tab:failures",
    )
    written.append("table_failure_ledger")
    return written


def build_figures(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(output_dir)
    fdir = base / "reviewer_ready_figures"
    fdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
    written: list[str] = []

    # Figure A: support fidelity vs physical accuracy (per structure).
    rows = pd.read_csv(base / "physical_selected_output_rows.csv")
    figA = rows[
        (rows.alpha_regime == "fixed_benchmark")
        & (~rows.near_zero_y_true)
        & (rows.selector != "full_support")
    ].copy()
    figA_data = figA[["structure_id", "selector", "E_support_norm", "E_physical_norm"]]
    figA_data.to_csv(fdir / "figureA_support_vs_physical_data.csv", index=False)
    structures = ["ieee14_8x8", "ieee30_8x8", "ieee14_16x16"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, sid in zip(axes, structures, strict=True):
        sub = figA[figA.structure_id == sid]
        for sel, g in sub.groupby("selector"):
            ax.scatter(
                np.clip(g.E_support_norm, 0, 3),
                np.clip(g.E_physical_norm, 0, 3),
                s=6,
                alpha=0.35,
                color=SELECTOR_COLOR.get(sel, "#333333"),
                label=sel,
                edgecolors="none",
            )
        ax.set_title(sid)
        ax.set_xlabel(r"$E_{\rm support}$ (sparse vs full Ridge)")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
    axes[0].set_ylabel(r"$E_{\rm physical}$ (sparse vs $y_{\rm true}$)")
    axes[-1].legend(fontsize=5, markerscale=2, ncol=1, loc="upper right")
    fig.suptitle("Support and physical errors are not equivalent (fixed benchmark $\\alpha$)")
    fig.tight_layout()
    fig.savefig(fdir / "figureA_support_vs_physical.png", bbox_inches="tight")
    plt.close(fig)
    written.append("figureA_support_vs_physical")

    # Figure B: utility-feasibility boundary by degree (if available).
    hd_path = base / "high_degree_qsvt_rows.csv"
    if hd_path.exists():
        hd = pd.read_csv(hd_path)
        fit_col = (
            "analytic_bounded_fit_ok"
            if "analytic_bounded_fit_ok" in hd.columns
            else "boundedness_parity_fit_ok"
        )
        agg = (
            hd.groupby(["degree"])
            .agg(
                any_useful_and_bounded_fit=(
                    "application_useful_full_state",
                    lambda s: bool((s.values & hd.loc[s.index, fit_col].values).any()),
                ),
                min_uniform_fit=("uniform_fit_error", "min"),
                n_useful=("application_useful_full_state", "sum"),
                n_bounded_fit=(fit_col, "sum"),
            )
            .reset_index()
        )
        agg.to_csv(fdir / "figureB_degree_boundary_data.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        useful = hd[hd.application_useful_full_state]
        overreg = hd[~hd.application_useful_full_state]
        for label, grp, color, mk in [
            ("application-useful $\\alpha$", useful, "#0072B2", "o"),
            ("over-regularized $\\alpha$", overreg, "#D55E00", "s"),
        ]:
            m = grp.groupby("degree")["uniform_fit_error"].min()
            ax.plot(m.index, np.clip(m.values, 1e-4, 1e7), marker=mk, color=color, label=label)
        ax.axhline(0.002, ls="--", color="gray", lw=1, label="uniform-fit tolerance")
        ax.set_yscale("log")
        ax.set_xlabel("QSVT polynomial degree")
        ax.set_ylabel("min uniform fit error")
        ax.set_xticks([31, 63, 127, 255])
        ax.legend(fontsize=7)
        ax.set_title("No overlap under the tested construction through degree 255")
        fig.tight_layout()
        fig.savefig(fdir / "figureB_degree_boundary.png", bbox_inches="tight")
        plt.close(fig)
        written.append("figureB_degree_boundary")

    # Figure C: structure-level selector effects (paired, by case).
    ss = json.loads((base / "structure_aware_statistics.json").read_text())
    eff = pd.DataFrame(ss["structure_level_effects"])
    eff.to_csv(fdir / "figureC_structure_effects_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = {"ieee14": "#0072B2", "ieee30": "#D55E00"}
    x = np.arange(len(eff))
    ax.bar(x, eff["mean_effect"], color=[colors.get(c, "#333") for c in eff["case"]], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(eff["structure_id"], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("mean physical effect\n(baseline $-$ proposed)")
    ax.set_title("Structure-level mean physical effects have mixed signs")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=c, label=k) for k, c in colors.items()], fontsize=7, title="IEEE case"
    )
    fig.tight_layout()
    fig.savefig(fdir / "figureC_structure_effects.png", bbox_inches="tight")
    plt.close(fig)
    written.append("figureC_structure_effects")
    return written
