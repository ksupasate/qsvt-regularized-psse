"""Manuscript-ready figures and LaTeX tables for the TQE reviewer-blocking extensions.

Every artifact is generated from the machine-readable ``raw_*`` / summary CSVs of the three
workstreams (never from hand-typed numbers) and matches the repository house style (IEEE column
width, hidden top/right spines, booktabs tables with a ``% Source:`` provenance header). Figures use
the perceptually-uniform, colour-blind-safe ``viridis`` family for sequential fields and the paper's
existing case palette for categorical series.

Assets are written under the study output directory by default; copying them into ``manuscript/`` is
a separate integration step performed only after the experiments pass claim-support review.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CASE_COLORS = {"ieee14": "#0072B2", "ieee30": "#D55E00", "ieee57": "#009E73"}
CASE_LABELS = {"ieee14": "IEEE-14", "ieee30": "IEEE-30", "ieee57": "IEEE-57"}
EPS_COLORS = {1e-2: "#0072B2", 1e-3: "#009E73", 1e-4: "#E69F00", 1e-5: "#D55E00"}
EPS_MARKERS = {1e-2: "o", 1e-3: "s", 1e-4: "^", 1e-5: "D"}
EPS_LINESTYLES = {1e-2: "-", 1e-3: "--", 1e-4: "-.", 1e-5: ":"}
GRID_GREY = "#D8D8D8"


def _despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _scientific_tick_label(value: float) -> str:
    """Format a positive grid value as a consistent math-text power of ten."""

    exponent = int(np.floor(np.log10(float(value))))
    coefficient = float(value) / (10.0**exponent)
    if np.isclose(coefficient, 1.0, rtol=0.0, atol=1e-10):
        return rf"$10^{{{exponent}}}$"
    return rf"${coefficient:g}\!\times\!10^{{{exponent}}}$"


# ============================================================ Workstream A


def build_degree_lambda_error_map(
    grid_csv: str | Path,
    dmin_csv: str | Path,
    out_path: str | Path,
    *,
    scalar_scope: str = "scalar_validation",
) -> Path:
    """Figure: (A) fit-error heatmap over (degree, lambda) with boundedness/synthesis overlays;
    (B) minimum feasible degree d_min(lambda) per target tolerance."""

    grid = pd.read_csv(grid_csv)
    dmin = pd.read_csv(dmin_csv)
    scal = grid[grid["scope_id"] == scalar_scope].copy()
    lambdas = np.sort(scal["normalized_lambda"].unique())
    degrees = np.sort(scal["degree"].unique())

    fit = np.full((len(degrees), len(lambdas)), np.nan)
    bounded = np.zeros_like(fit, dtype=bool)
    synth = np.zeros_like(fit, dtype=bool)
    for i, d in enumerate(degrees):
        for j, lam in enumerate(lambdas):
            row = scal[(scal["degree"] == d) & (scal["normalized_lambda"] == lam)]
            if not row.empty:
                fit[i, j] = float(row["uniform_fit_error"].iloc[0])
                bounded[i, j] = bool(row["boundedness_ok"].iloc[0])
                synth[i, j] = str(row["phase_synthesis_status"].iloc[0]) == "synthesized"

    z = np.log10(np.clip(fit, 1e-15, 1e3))
    journal_style = {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "hatch.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(journal_style):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.29, top=0.87, wspace=0.39)

        ax = axes[0]
        im = ax.pcolormesh(
            np.arange(len(lambdas) + 1, dtype=float) - 0.5,
            np.arange(len(degrees) + 1, dtype=float) - 0.5,
            z,
            cmap="viridis_r",
            vmin=-15,
            vmax=3,
            shading="flat",
            antialiased=False,
            rasterized=False,
        )
        ax.set_aspect("auto")
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels(
            [_scientific_tick_label(v) for v in lambdas],
            rotation=90,
            ha="center",
        )
        ax.set_yticks(range(len(degrees)))
        ax.set_yticklabels([str(int(d)) for d in degrees])
        ax.set_xlabel(r"Normalized regularization $\lambda$")
        ax.set_ylabel(r"Odd degree $d$")
        ax.set_title("(a) Uniform polynomial-approximation error", pad=7)
        # Preserve every cell while adding redundant, print-safe state encodings.
        for i in range(len(degrees)):
            for j in range(len(lambdas)):
                if not bounded[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="//",
                            edgecolor=(0.05, 0.05, 0.05, 0.32),
                            linewidth=0.15,
                        )
                    )
                if synth[i, j]:
                    ax.plot(
                        j,
                        i,
                        marker="o",
                        markersize=4.2,
                        markerfacecolor="white",
                        markeredgecolor="#202020",
                        markeredgewidth=0.7,
                        linestyle="none",
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.025)
        if cbar.solids is not None:
            cbar.solids.set_rasterized(False)
        cbar.set_label(r"$\log_{10}\varepsilon_{\mathrm{poly}}$")
        cbar.ax.tick_params(labelsize=8.0)

        ax2 = axes[1]
        tolerance_order = (1e-2, 1e-3, 1e-4, 1e-5)
        tolerance_marker_sizes = {1e-2: 5.0, 1e-3: 6.0, 1e-4: 7.0, 1e-5: 8.0}
        for layer, eps in enumerate(tolerance_order, start=2):
            sub = dmin[
                (dmin["scope_id"] == scalar_scope)
                & np.isclose(dmin["epsilon_target"], eps, rtol=0.0, atol=1e-15)
            ].sort_values("normalized_lambda")
            x = sub["normalized_lambda"].to_numpy(dtype=float)
            y = sub["d_min_fit"].to_numpy(dtype=float)
            ax2.plot(
                x,
                y,
                marker=EPS_MARKERS[eps],
                markersize=tolerance_marker_sizes[eps],
                markerfacecolor="none",
                markeredgecolor=EPS_COLORS[eps],
                markeredgewidth=0.95,
                linewidth=1.75,
                linestyle=EPS_LINESTYLES[eps],
                color=EPS_COLORS[eps],
                label=rf"$10^{{{int(np.log10(eps))}}}$",
                drawstyle="steps-post",
                zorder=layer,
            )
        ax2.set_xscale("log")
        ax2.set_xlim(float(lambdas.min()) / 1.25, float(lambdas.max()) * 1.25)
        panel_b_ticks = lambdas[::2]
        ax2.set_xticks(panel_b_ticks)
        ax2.set_xticklabels([_scientific_tick_label(v) for v in panel_b_ticks])
        ax2.set_yscale("log", base=2)
        feasible_degrees = np.sort(
            pd.to_numeric(
                dmin.loc[
                    (dmin["scope_id"] == scalar_scope) & dmin["d_min_fit"].notna(),
                    "d_min_fit",
                ]
            ).unique()
        )
        ax2.set_yticks(feasible_degrees)
        ax2.set_yticklabels([str(int(d)) for d in feasible_degrees])
        ax2.set_xlabel(r"Normalized regularization $\lambda$")
        ax2.set_ylabel(r"Minimum tested feasible degree $d_{\min}$")
        ax2.set_title("(b) Minimum feasible tested degree", pad=7)
        ax2.grid(axis="y", which="major", color=GRID_GREY, linewidth=0.55)
        ax2.set_axisbelow(True)
        tolerance_legend = ax2.legend(
            frameon=False,
            loc="upper left",
            title=r"Tolerance $\varepsilon_{\mathrm{poly}}$" "\n(no marker: infeasible)",
            handlelength=2.6,
            labelspacing=0.4,
        )
        tolerance_legend.get_title().set_fontsize(7.8)
        _despine(ax2)

        status_handles = [
            Patch(
                facecolor="white",
                edgecolor="#707070",
                hatch="///",
                linewidth=0.5,
                label="Unbounded candidate",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor="#202020",
                markeredgewidth=0.75,
                label=r"Phases synthesized (attempted only for $d\leq31$)",
            ),
        ]
        fig.legend(
            handles=status_handles,
            loc="lower left",
            bbox_to_anchor=(0.065, 0.015),
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.6,
            handletextpad=0.65,
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"Creator": __file__},
        )
        plt.close(fig)
    return out_path


def build_postselection_tradeoff(
    grid_csv: str | Path,
    out_path: str | Path,
    *,
    degree: int = 31,
) -> Path:
    """Figure: (A) modeled/executed postselection vs lambda per matrix scope at a fixed degree;
    (B) full-state Ridge RMSE ratio vs lambda (the QSVT-feasible band over-regularizes)."""

    grid = pd.read_csv(grid_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    ax = axes[0]
    matrix = grid[(grid["scope_kind"] != "scalar") & (grid["degree"] == degree)].copy()
    for scope_id, sub in matrix.groupby("scope_id"):
        sub = sub.sort_values("normalized_lambda")
        case = str(sub["case"].iloc[0])
        color = CASE_COLORS.get(case, "#555555")
        ls = "-" if "full_jacobian" in scope_id else "--"
        ax.plot(
            sub["normalized_lambda"],
            sub["postselection_probability_modeled"],
            linestyle=ls,
            linewidth=1.6,
            color=color,
            label=scope_id.replace("_", " "),
        )
        ex = sub[sub["postselection_probability_executed"].notna()]
        if not ex.empty:
            ax.scatter(
                ex["normalized_lambda"],
                ex["postselection_probability_executed"],
                s=40,
                marker="*",
                color=color,
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"normalized regularization $\lambda$", fontsize=8)
    ax.set_ylabel(f"modeled postselection ($d={degree}$)", fontsize=8)
    ax.set_title("postselection vs regularization", fontsize=9)
    ax.grid(axis="y", which="both", color=GRID_GREY, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    _despine(ax)

    ax2 = axes[1]
    lambdas = np.sort(grid["normalized_lambda"].unique())
    feasible_band = _scalar_feasible_lambda_band(grid)
    for case in ("ieee14", "ieee30", "ieee57"):
        ratios = _full_state_rmse_ratio(case, lambdas)
        ax2.plot(lambdas, ratios, linewidth=1.6, color=CASE_COLORS[case], label=CASE_LABELS[case])
    if feasible_band is not None:
        ax2.axvspan(feasible_band[0], feasible_band[1], color="#276FBF", alpha=0.08)
    ax2.axhline(1.5, color="#888888", linewidth=0.8, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"normalized regularization $\lambda$", fontsize=8)
    ax2.set_ylabel("full-state RMSE ratio to oracle-best", fontsize=8)
    ax2.set_title("utility (shaded = QSVT-feasible band)", fontsize=9)
    ax2.grid(axis="y", which="both", color=GRID_GREY, linewidth=0.5)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    _despine(ax2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _permanent_divergence_onset(scope_rows: pd.DataFrame) -> int | None:
    """Lowest degree beyond which boundedness is irrecoverably lost (all higher degrees "
    "unbounded)."""

    ordered = scope_rows.sort_values("degree")
    degrees = ordered["degree"].to_numpy()
    bounded = ordered["boundedness_ok"].astype(bool).to_numpy()
    onset = None
    for i in range(len(degrees)):
        if not bounded[i] and not bounded[i:].any():
            onset = int(degrees[i])
            break
    return onset


def _scalar_feasible_lambda_band(grid: pd.DataFrame) -> tuple[float, float] | None:
    scal = grid[grid["scope_id"] == "scalar_validation"]
    feasible = scal[scal["boundedness_ok"] & (scal["uniform_fit_error"] <= 1e-2)]
    if feasible.empty:
        return None
    return float(feasible["normalized_lambda"].min()), float(grid["normalized_lambda"].max())


def _full_state_rmse_ratio(case: str, lambdas: np.ndarray) -> np.ndarray:
    """Ridge full-state RMSE ratio to the oracle-best over the alpha grid (utility diagnostic)."""

    from robust_qsvt_se.cross_case_validation.common import build_case_full_system
    from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution

    system = build_case_full_system(case, 123)
    beta = float(np.linalg.svd(system.matrix, compute_uv=False).max())
    ref_grid = np.logspace(-3, 7, 60)
    rmses = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        (
                            ridge_svd_solution(system.matrix, system.residual, alpha=float(a))
                            - system.x_true
                        )
                        ** 2
                    )
                )
            )
            for a in ref_grid
        ]
    )
    best = float(rmses.min())
    out = []
    for lam in lambdas:
        alpha = float(lam) * beta**2
        x = ridge_svd_solution(system.matrix, system.residual, alpha=alpha)
        rmse = float(np.sqrt(np.mean((x - system.x_true) ** 2)))
        out.append(rmse / max(best, 1e-30))
    return np.asarray(out)


def build_degree_lambda_summary_table(
    dmin_csv: str | Path,
    grid_csv: str | Path,
    out_path: str | Path,
    *,
    scalar_scope: str = "scalar_validation",
    study_id: str = "tqe_degree_lambda_error_scaling_v1",
) -> Path:
    """LaTeX table: per-lambda minimum feasible scalar degree at each tolerance + "
    "fit/boundedness."""

    dmin = pd.read_csv(dmin_csv)
    grid = pd.read_csv(grid_csv)
    scal = grid[grid["scope_id"] == scalar_scope]
    lambdas = np.sort(scal["normalized_lambda"].unique())
    tolerances = sorted(dmin["epsilon_target"].unique())

    def cell(v) -> str:
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(v))

    body = []
    for lam in lambdas:
        row = [f"{lam:g}"]
        for eps in tolerances:
            rec = dmin[
                (dmin["scope_id"] == scalar_scope)
                & (dmin["epsilon_target"] == eps)
                & (dmin["normalized_lambda"] == lam)
            ]
            row.append(cell(rec["d_min_fit"].iloc[0]) if not rec.empty else "--")
        sub = scal[scal["normalized_lambda"] == lam].sort_values("degree")
        bounded = sub[sub["boundedness_ok"]]
        best_fit = float(bounded["uniform_fit_error"].min()) if not bounded.empty else float("nan")
        onset = _permanent_divergence_onset(sub)
        row.append("--" if np.isnan(best_fit) else f"{best_fit:.1e}")
        row.append("--" if onset is None else str(onset))
        body.append(" & ".join(row) + r" \\")

    eps_headers = " & ".join(rf"$\epsilon{{=}}10^{{{int(np.log10(e))}}}$" for e in tolerances)
    caption = (
        r"Workstream~A: minimum feasible odd QSVT degree $d_{\min}$ for the bounded spectral "
        r"filter "
        r"$f_{\lambda,C}(s)=\tfrac{1}{C}\,\tfrac{s}{s^2+\lambda}$ on the controlled scalar "
        r"validation "
        r"grid, as a function of normalized regularization $\lambda$ and target uniform-fit "
        r"tolerance "
        r"$\epsilon$. ``--'' marks $(\lambda,\epsilon)$ cells for which no tested odd degree "
        r"$d\in\{7,\dots,511\}$ is simultaneously bounded ($|p|\le1$) and accurate to $\epsilon$. "
        r"Best fit is the smallest uniform error over bounded degrees. The divergence onset is the "
        r"first tested degree after which no larger tested degree regained boundedness. Within "
        r"this finite degree set and scalar grid, smaller $\lambda$ behaves as a "
        r"construction-specific approximation barrier rather than merely a degree-budget "
        r"limitation. This is not a "
        r"universal QSVT lower bound; another polynomial basis or synthesis method may move the "
        r"boundary. No claim of QSVT numerical superiority over matched Ridge is made."
    )
    lines = [
        f"% Auto-generated by robust_qsvt_se.tqe_extensions.reporting ({study_id}).",
        "% Source: outputs/tqe_degree_lambda_error_scaling/minimum_feasible_degree.csv, "
        "raw_grid.csv",
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:tqe_degree_lambda_summary}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "r" * len(tolerances) + r"rr@{}}",
        r"\toprule",
        rf"$\lambda$ & {eps_headers} & Best fit & Div.\ onset \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ============================================================ Workstream C

ROW_COLOR = "#276FBF"
ENTRY_COLOR = "#C14953"
FULL_COLOR = "#333333"


def build_entry_vs_row_physical_error(
    statistical_csv: str | Path,
    out_path: str | Path,
) -> Path:
    """Plot paired row-minus-entry benchmark-reference effects and win/tie/loss counts."""

    stats = pd.read_csv(statistical_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    ax = axes[0]
    contrast_colors = {
        "best_row_vs_best_entry": "#276FBF",
        "row_sensitivity_mean_vs_entry_sensitivity_mean": "#4C956C",
        "row_magnitude_vs_entry_magnitude": "#E09F3E",
    }
    for contrast, sub in stats.groupby("contrast"):
        sub = sub.sort_values("budget_nnz")
        color = contrast_colors.get(contrast, "#555555")
        ax.plot(
            sub["budget_nnz"],
            sub["median_effect"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=contrast.replace("_", " ")[:34],
        )
        sig = sub[sub["wilcoxon_p"] <= 0.05]
        ax.scatter(sig["budget_nnz"], sig["median_effect"], s=70, marker="*", color=color, zorder=6)
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_xlabel("nonzero budget (matched)", fontsize=8)
    ax.set_ylabel(r"median effect: row $-$ entry $E_{\rm benchmark}$", fontsize=8)
    ax.set_title(r"row vs entry (negative = row better; $\star$ Wilcoxon $p\leq0.05$)", fontsize=8)
    ax.grid(axis="y", which="both", color=GRID_GREY, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    _despine(ax)

    ax2 = axes[1]
    best = stats[stats["contrast"] == "best_row_vs_best_entry"].sort_values("budget_nnz")
    x = np.arange(len(best))
    ax2.bar(x, best["wins_row_better"], color=ROW_COLOR, label="row better")
    ax2.bar(x, best["ties"], bottom=best["wins_row_better"], color="#BBBBBB", label="tie")
    ax2.bar(
        x,
        best["losses_row_worse"],
        bottom=best["wins_row_better"] + best["ties"],
        color=ENTRY_COLOR,
        label="entry better",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(b)) for b in best["budget_nnz"]], fontsize=7)
    ax2.set_xlabel("nonzero budget (matched)", fontsize=8)
    ax2.set_ylabel("structural groups (of 12)", fontsize=8)
    ax2.set_title("best-row vs best-entry outcomes", fontsize=8)
    ax2.legend(frameon=False, fontsize=6, loc="upper right", ncol=1)
    _despine(ax2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_accuracy_resource_frontier(
    resource_csv: str | Path,
    out_path: str | Path,
) -> Path:
    """Figure: accuracy-resource frontier - median held-out (A) physical and (B) support error vs
    retained nonzeros, distinguishing entry-level, row-level, and full-support families."""

    res = pd.read_csv(resource_csv)
    entry_selectors = {"entry_global_magnitude", "entry_sensitivity_mean"}
    res = res[res["selector"] != "random_row"].copy()  # random shown separately would add noise
    res["family"] = np.where(
        res["selector"] == "full_support",
        "full",
        np.where(res["selector"].isin(entry_selectors), "entry", "row"),
    )
    # Bin retained nonzeros so the family median is taken over comparable resource levels.
    res["nnz_bin"] = res["budget_nnz"].round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    def _plot(ax, value_col, ylabel):
        for family, color in (("row", ROW_COLOR), ("entry", ENTRY_COLOR)):
            fam = res[res["family"] == family]
            agg = fam.groupby("nnz_bin")[value_col].median().reset_index().sort_values("nnz_bin")
            lo = fam.groupby("nnz_bin")[value_col].min().reindex(agg["nnz_bin"]).to_numpy()
            hi = fam.groupby("nnz_bin")[value_col].max().reindex(agg["nnz_bin"]).to_numpy()
            ax.fill_between(agg["nnz_bin"], lo, hi, color=color, alpha=0.12)
            ax.plot(
                agg["nnz_bin"], agg[value_col], marker="o", markersize=3, linewidth=1.6, color=color
            )
        full = res[res["family"] == "full"]
        if not full.empty:
            fagg = full.groupby("nnz_bin")[value_col].median().reset_index()
            ax.scatter(
                fagg["nnz_bin"], fagg[value_col], s=42, marker="D", color=FULL_COLOR, zorder=6
            )
        ax.set_xlabel("median retained nonzeros", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="both", which="both", color=GRID_GREY, linewidth=0.4)
        _despine(ax)

    _plot(axes[0], "median_E_physical", r"median $E_{\rm benchmark}$")
    _plot(axes[1], "median_E_support", r"median $E_{\rm support}$")
    axes[0].set_title("physical accuracy vs resource", fontsize=9)
    axes[1].set_title("support fidelity vs resource", fontsize=9)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=ROW_COLOR, marker="o", markersize=4, label="row-level"),
        Line2D([0], [0], color=ENTRY_COLOR, marker="o", markersize=4, label="entry-level"),
        Line2D(
            [0],
            [0],
            color=FULL_COLOR,
            marker="D",
            linestyle="none",
            markersize=5,
            label="full support",
        ),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_row_sparsification_summary_table(
    statistical_csv: str | Path,
    out_path: str | Path,
    *,
    study_id: str = "tqe_measurement_row_sparsification_v1",
) -> Path:
    """Build the best-row versus best-entry benchmark-reference table."""

    stats = pd.read_csv(statistical_csv)
    best = stats[stats["contrast"] == "best_row_vs_best_entry"].sort_values("budget_nnz")
    body = []
    for _, r in best.iterrows():
        wtl = f"{int(r['wins_row_better'])}/{int(r['ties'])}/{int(r['losses_row_worse'])}"
        body.append(
            f"{int(r['budget_nnz'])} & {r['median_effect']:+.3f} & {r['mean_effect']:+.3f} & "
            f"{wtl} & {r['wilcoxon_p']:.3f} & {r['sign_test_p']:.3f} \\\\"
        )
    caption = (
        r"Workstream~C: paired held-out benchmark-reference-error effect of whole-measurement-row selection "
        r"versus arbitrary weighted-Jacobian entry selection at matched nonzero budgets, over 12 "
        r"distinct benchmark-derived structural groups (two realizations averaged per group; the "
        r"group is the statistical unit). Effect $=$ best-row $-$ best-entry median $E_{\rm "
        r"benchmark}$ "
        r"(negative favours whole rows); W/T/L counts row-better/tie/row-worse groups; $p$ are "
        r"two-sided Wilcoxon signed-rank and sign tests. Whole-row selection is significantly more "
        r"benchmark-reference accurate at the tightest budgets and becomes inconclusive as the budget "
        r"grows. No physically optimal support or QSVT-over-Ridge superiority is claimed."
    )
    lines = [
        f"% Auto-generated by robust_qsvt_se.tqe_extensions.reporting ({study_id}).",
        "% Source: outputs/tqe_measurement_row_sparsification/statistical_summary.csv",
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:tqe_row_sparsification_summary}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}rrrrrr@{}}",
        r"\toprule",
        r"nnz budget & Median eff. & Mean eff. & W/T/L & Wilcoxon $p$ & Sign $p$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ============================================================ Workstream B

SCENARIO_COLORS = {
    "gaussian_noise_baseline": "#276FBF",
    "random_missing_measurement_stress": "#E09F3E",
    "sparse_signed_bad_data_stress": "#C14953",
}
SCENARIO_SHORT = {
    "gaussian_noise_baseline": "Gaussian",
    "random_missing_measurement_stress": "missing",
    "sparse_signed_bad_data_stress": "bad data",
}


def build_nonlinear_iteration_error(
    per_iteration_csv: str | Path,
    out_path: str | Path,
) -> Path:
    """Figure: (A) per-iteration error decomposition (rational->polynomial->statevector tiers);
    (B) postselection and cumulative state RMSE by nonlinear iteration."""

    pi = pd.read_csv(per_iteration_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    ax = axes[0]
    for col, color, label in (
        ("qsvt_matrix_action_error", "#276FBF", "poly vs rational (matrix action)"),
        ("circuit_vs_ridge_error", "#C14953", "circuit vs Ridge"),
        ("circuit_vs_polynomial_error", "#4C956C", "circuit vs polynomial"),
    ):
        agg = pi.groupby("iteration")[col].median()
        ax.plot(
            agg.index,
            np.clip(agg.to_numpy(), 1e-12, None),
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("nonlinear iteration", fontsize=8)
    ax.set_ylabel("median selected-output error", fontsize=8)
    ax.set_title("error decomposition by iteration", fontsize=9)
    ax.grid(axis="y", which="both", color=GRID_GREY, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    _despine(ax)

    ax2 = axes[1]
    post = pi.groupby("iteration")["postselection_probability_executed"].median()
    ax2.plot(
        post.index,
        post.to_numpy(),
        marker="s",
        markersize=3.5,
        linewidth=1.5,
        color="#276FBF",
        label="postselection",
    )
    ax2.set_xlabel("nonlinear iteration", fontsize=8)
    ax2.set_ylabel("median postselection", fontsize=8, color="#276FBF")
    ax2.tick_params(axis="y", labelcolor="#276FBF")
    ax2.set_ylim(0, 1)
    ax3 = ax2.twinx()
    rmse = pi.groupby("iteration")["state_rmse"].median()
    ax3.plot(
        rmse.index,
        rmse.to_numpy(),
        marker="^",
        markersize=3.5,
        linewidth=1.5,
        color="#C14953",
        label="state RMSE",
    )
    ax3.set_yscale("log")
    ax3.set_ylabel("median state RMSE", fontsize=8, color="#C14953")
    ax3.tick_params(axis="y", labelcolor="#C14953")
    ax2.set_title("postselection and convergence", fontsize=9)
    ax2.spines[["top"]].set_visible(False)
    ax3.spines[["top"]].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_nonlinear_convergence_paths(
    per_iteration_csv: str | Path,
    out_path: str | Path,
) -> Path:
    """Figure: state RMSE (A) and weighted residual (B) per nonlinear iteration for all runs."""

    pi = pd.read_csv(per_iteration_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)
    seen = set()
    for (scenario, _seed), block in pi.groupby(["scenario", "seed"]):
        block = block.sort_values("iteration")
        color = SCENARIO_COLORS.get(scenario, "#555555")
        label = SCENARIO_SHORT.get(scenario, scenario) if scenario not in seen else None
        seen.add(scenario)
        axes[0].plot(
            block["iteration"],
            block["state_rmse"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            alpha=0.85,
            color=color,
            label=label,
        )
        axes[1].plot(
            block["iteration"],
            block["weighted_residual_norm"],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            alpha=0.85,
            color=color,
        )
    for ax, ylabel, title in (
        (axes[0], "state RMSE (rel. to truth)", "state convergence"),
        (axes[1], "weighted residual norm", "residual convergence"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("nonlinear iteration", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", which="both", color=GRID_GREY, linewidth=0.5)
        _despine(ax)
    axes[0].legend(
        frameon=False, fontsize=6.5, loc="upper right", title="scenario", title_fontsize=6.5
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_nonlinear_circuit_summary_table(
    per_iteration_csv: str | Path,
    convergence_csv: str | Path,
    finite_shot_csv: str | Path,
    out_path: str | Path,
    *,
    study_id: str = "tqe_nonlinear_qsvt_circuit_loop_v1",
) -> Path:
    """LaTeX table: per-scenario circuit-in-the-loop summary (statevector fidelity + "
    "convergence)."""

    pi = pd.read_csv(per_iteration_csv)
    conv = pd.read_csv(convergence_csv)
    sv = pi[pi["evidence_tier"] == "explicit_statevector_circuit_execution"]
    body = []
    for scenario in (
        "gaussian_noise_baseline",
        "random_missing_measurement_stress",
        "sparse_signed_bad_data_stress",
    ):
        cs = conv[conv["scenario"] == scenario]
        ss = sv[sv["scenario"] == scenario]
        if cs.empty:
            continue
        n_conv = int(cs["converged"].sum())
        body.append(
            f"{SCENARIO_SHORT.get(scenario, scenario)} & {len(cs)} & {n_conv}/{len(cs)} & "
            f"{int(ss['iteration'].count())} & "
            f"{ss['circuit_vs_ridge_error'].median():.1e} & "
            f"{ss['circuit_vs_polynomial_error'].median():.1e} & "
            f"{ss['postselection_probability_executed'].median():.3f} & "
            f"{cs['final_state_rmse'].median():.1e} \\\\"
        )
    caption = (
        r"Workstream~B: nonlinear AC QSVT circuit-in-the-loop summary per scenario over three "
        r"seeds "
        r"(IEEE-14, $4\times4$ block, degree 31, $\lambda=0.1$). Every retained iteration rebuilds "
        r"the "
        r"residual, weighted Jacobian, block, bounded target, and explicit circuit; the QSVT "
        r"statevector circuit reproduces the matched block Ridge selected output up to the "
        r"polynomial "
        r"approximation error and the exact polynomial action to $\sim\!10^{-8}$. The strongest "
        r"supported claim is a completed classical statevector circuit-in-the-loop path; no "
        r"quantum "
        r"hardware execution, full-state quantum recovery, or competitiveness is claimed."
    )
    lines = [
        f"% Auto-generated by robust_qsvt_se.tqe_extensions.reporting ({study_id}).",
        "% Source: "
        "outputs/tqe_nonlinear_qsvt_circuit_loop/{per_iteration_results,convergence_summary}.csv",
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        r"\label{tab:tqe_nonlinear_circuit_summary}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Scenario & Runs & Conv. & SV its & Circ/Ridge & Circ/poly & $p_{\rm post}$ & "
        r"Final RMSE \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
