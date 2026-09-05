#!/usr/bin/env python3
"""Reconcile and plot the final 12-group manuscript structural contrast.

This manuscript-facing audit reads the frozen structure-level summaries and
does not modify any scientific evidence.  It reconstructs the declared
``sensitivity_refined_mean`` versus ``global_magnitude`` ``E_support``
contrast without importing the campaign statistics helper, independently
recomputes its statistics, and fails if the expected internally consistent
result is not reproduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "outputs" / "final_repository_backed_manuscript_completion"
SOURCE = (
    ROOT
    / "outputs"
    / "tqe_physical_alignment_and_generalization"
    / "physical_audit"
    / "structure_level_summary.csv"
)
FROZEN_SUMMARY = (
    ROOT
    / "outputs"
    / "tqe_physical_alignment_and_generalization"
    / "physical_audit"
    / "support_fidelity_pairwise_summary.csv"
)
COMPANION = (
    ROOT
    / "outputs"
    / "robust_companion_statistics"
    / "robust_companion_statistics.csv"
)
ALTERNATIVE = (
    ROOT
    / "outputs"
    / "final_contribution_evidence"
    / "figure_data"
    / "structural_group_paired_errors.csv"
)
ALTERNATIVE_CONFIG = ROOT / "configs" / "output_aware_structural_generalization.json"
CAMPAIGN_CONFIG = ROOT / "configs" / "tqe_physical_alignment" / "campaign.json"
CSV_OUTPUT = AUDIT_DIR / "structural_contrast_reconciliation.csv"
MD_OUTPUT = AUDIT_DIR / "structural_contrast_reconciliation.md"
FIGURE_OUTPUT = ROOT / "manuscript" / "figures" / "fig_paired_structural_effects.pdf"

CANDIDATE = "sensitivity_refined_mean"
BASELINE = "global_magnitude"
METRIC = "E_support"
EXPECTED_MEAN = 0.04459764166450786
EXPECTED_CI = np.array([0.0151083006101379, 0.0759614727023087])
EXPECTED_COUNTS = (8, 2, 2)
EXPECTED_SIGN_P = 0.109375
EXPECTED_WILCOXON_P = 0.009765625
EXPECTED_HOLM_SIGN_P = 0.21875
EXPECTED_HOLM_WILCOXON_P = 0.0234375
HOLM_FAMILY_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def reconcile() -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    source = pd.read_csv(SOURCE)
    selected = source.loc[
        source["selector"].isin([CANDIDATE, BASELINE])
        & source["metric"].eq(METRIC)
        & source["functional_classification"].eq("physical")
        & source["summary_scope"].eq("all_families")
    ].copy()
    if len(selected) != 24:
        raise RuntimeError(f"expected 24 selector rows, found {len(selected)}")
    if selected.duplicated(["structural_group_id", "ieee_case", "selector"]).any():
        raise RuntimeError("duplicate structure/case/selector keys in authoritative source")

    pivot = selected.pivot(
        index=["structural_group_id", "ieee_case"],
        columns="selector",
        values="structure_statistic",
    ).reset_index()
    if len(pivot) != 12 or pivot[[CANDIDATE, BASELINE]].isna().any().any():
        raise RuntimeError("authoritative source does not yield 12 complete paired rows")

    paired = pivot.rename(
        columns={
            "ieee_case": "case",
            "structural_group_id": "structural_group",
            BASELINE: "baseline_E_supp",
            CANDIDATE: "candidate_E_supp",
        }
    )
    paired.insert(2, "baseline_selector", BASELINE)
    paired.insert(3, "candidate_selector", CANDIDATE)
    paired["effect"] = paired["baseline_E_supp"] - paired["candidate_E_supp"]
    paired["outcome"] = np.select(
        [paired["effect"].gt(0.0), paired["effect"].lt(0.0)],
        ["win", "loss"],
        default="tie",
    )
    paired["source_file"] = str(SOURCE.relative_to(ROOT))
    paired["source_row_key"] = paired.apply(
        lambda row: (
            f"structural_group_id={row['structural_group']};"
            f"ieee_case={row['case']};selectors={BASELINE}|{CANDIDATE};"
            "functional_classification=physical;summary_scope=all_families;"
            f"metric={METRIC}"
        ),
        axis=1,
    )
    paired = paired.sort_values(["case", "structural_group"]).reset_index(drop=True)

    effects = paired["effect"].to_numpy(dtype=np.float64)
    wins = int(np.sum(effects > 0.0))
    losses = int(np.sum(effects < 0.0))
    ties = int(np.sum(effects == 0.0))
    mean_effect = float(effects.mean())

    config = json.loads(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    settings = config["statistics"]
    replicates = int(settings["bootstrap_replicates"])
    seed = int(settings["case_stratified_bootstrap_seed"]) + 50_000
    confidence = float(settings["confidence_level"])
    rng = np.random.default_rng(seed)
    sampled_parts = []
    for case in sorted(paired["case"].unique()):
        case_effects = paired.loc[paired["case"].eq(case), "effect"].to_numpy(dtype=np.float64)
        indices = rng.integers(
            0, len(case_effects), size=(replicates, len(case_effects))
        )
        sampled_parts.append(case_effects[indices])
    sampled_means = np.concatenate(sampled_parts, axis=1).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(sampled_means, [tail, 1.0 - tail])

    sign_p = float(
        binomtest(wins, wins + losses, p=0.5, alternative="two-sided").pvalue
    )
    wilcoxon_result = wilcoxon(effects, alternative="two-sided", method="exact")
    wilcoxon_p = float(wilcoxon_result.pvalue)

    companion = pd.read_csv(COMPANION)
    family = companion[
        ["sign_test_p_exact", "wilcoxon_p_exact"]
    ].astype(float)
    if len(family) != HOLM_FAMILY_SIZE:
        raise RuntimeError(
            f"expected Holm family of {HOLM_FAMILY_SIZE}, found {len(family)}"
        )
    holm_sign = holm_adjust(family["sign_test_p_exact"].tolist())
    holm_wilcoxon = holm_adjust(family["wilcoxon_p_exact"].tolist())
    headline_index = int(
        companion.index[
            companion["candidate_selector"].eq(CANDIDATE)
            & companion["baseline_selector"].eq(BASELINE)
            & companion["metric"].eq(METRIC)
        ][0]
    )
    holm_sign_p = float(holm_sign[headline_index])
    holm_wilcoxon_p = float(holm_wilcoxon[headline_index])

    assertions = [
        (np.isclose(mean_effect, EXPECTED_MEAN, rtol=0.0, atol=1e-15), "mean effect"),
        (np.allclose(interval, EXPECTED_CI, rtol=0.0, atol=1e-15), "bootstrap interval"),
        ((wins, losses, ties) == EXPECTED_COUNTS, "win/loss/tie counts"),
        (sign_p == EXPECTED_SIGN_P, "sign-test p-value"),
        (wilcoxon_p == EXPECTED_WILCOXON_P, "Wilcoxon p-value"),
        (holm_sign_p == EXPECTED_HOLM_SIGN_P, "Holm sign-test p-value"),
        (
            holm_wilcoxon_p == EXPECTED_HOLM_WILCOXON_P,
            "Holm Wilcoxon p-value",
        ),
    ]
    failures = [label for passed, label in assertions if not passed]
    if failures:
        raise RuntimeError("headline contrast failed reconciliation: " + ", ".join(failures))

    stats: dict[str, float | int | str] = {
        "mean_effect": mean_effect,
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_p": sign_p,
        "wilcoxon_statistic": float(wilcoxon_result.statistic),
        "wilcoxon_p": wilcoxon_p,
        "holm_sign_p": holm_sign_p,
        "holm_wilcoxon_p": holm_wilcoxon_p,
        "source_sha256": sha256(SOURCE),
        "frozen_summary_sha256": sha256(FROZEN_SUMMARY),
        "companion_sha256": sha256(COMPANION),
    }
    return paired, stats


def alternative_summary() -> dict[str, float | int | str]:
    alternative = pd.read_csv(ALTERNATIVE)
    effects = (
        alternative["baseline_group_normalized_error"]
        - alternative["candidate_group_normalized_error"]
    ).to_numpy(dtype=np.float64)
    config = json.loads(ALTERNATIVE_CONFIG.read_text(encoding="utf-8"))
    outcomes = alternative["outcome"]
    return {
        "candidate": str(alternative["candidate_selector"].iloc[0]),
        "baseline": str(alternative["baseline_selector"].iloc[0]),
        "mean_effect": float(effects.mean()),
        "wins": int(outcomes.eq("win").sum()),
        "losses": int(outcomes.eq("loss").sum()),
        "ties": int(outcomes.eq("tie").sum()),
        "tie_relative_tolerance": float(
            config["primary_comparison"]["tie_relative_tolerance"]
        ),
        "tie_epsilon": float(config["primary_comparison"]["tie_epsilon"]),
        "source_sha256": sha256(ALTERNATIVE),
    }


def write_reconciliation(
    paired: pd.DataFrame,
    stats: dict[str, float | int | str],
    alternative: dict[str, float | int | str],
) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    paired.to_csv(CSV_OUTPUT, index=False, float_format="%.17g")

    table_lines = [
        "| Case | Structural group | Baseline $E_{supp}$ | Candidate $E_{supp}$ | Effect | Outcome |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in paired.itertuples(index=False):
        table_lines.append(
            f"| {row.case} | `{row.structural_group}` | "
            f"{row.baseline_E_supp:.17g} | {row.candidate_E_supp:.17g} | "
            f"{row.effect:.17g} | {row.outcome} |"
        )
    markdown = f"""# Structural Contrast Reconciliation

## Authoritative source and row keys

- Source: `{SOURCE.relative_to(ROOT)}` (SHA-256 `{stats['source_sha256']}`).
- Frozen consistency summary: `{FROZEN_SUMMARY.relative_to(ROOT)}` (SHA-256 `{stats['frozen_summary_sha256']}`).
- Companion-test family: `{COMPANION.relative_to(ROOT)}` (SHA-256 `{stats['companion_sha256']}`).
- Row filter/key: `functional_classification=physical`, `summary_scope=all_families`, `metric=E_support`, selectors `{BASELINE}` and `{CANDIDATE}`, paired by `structural_group_id` and `ieee_case`.
- Candidate selector: `{CANDIDATE}` (refined sensitivity-guided support).
- Baseline selector: `{BASELINE}` (global magnitude-based support).
- Effect: baseline $E_{{supp}}$ minus candidate $E_{{supp}}$; exact positive/negative/zero classification, with no numerical tie tolerance.

## Reconciled 12 rows

{chr(10).join(table_lines)}

## Independently recomputed statistics

- Arithmetic mean effect: `{stats['mean_effect']:.17g}`.
- Case-stratified 95% percentile interval: `[{stats['ci_low']:.17g}, {stats['ci_high']:.17g}]`.
- Bootstrap protocol: `{stats['bootstrap_replicates']}` replicates, seed `{stats['bootstrap_seed']}`, resampling the four structural groups independently within each of IEEE-14, IEEE-30, and IEEE-57 while preserving case composition.
- Wins/losses/ties: `{stats['wins']}/{stats['losses']}/{stats['ties']}`.
- Exact two-sided sign-test p-value: `{stats['sign_p']:.17g}` (zeros omitted).
- Exact two-sided Wilcoxon signed-rank statistic and p-value: `{stats['wilcoxon_statistic']:.17g}`, `{stats['wilcoxon_p']:.17g}`.
- Holm-adjusted sign-test p-value over the four declared companion contrasts: `{stats['holm_sign_p']:.17g}`.
- Holm-adjusted Wilcoxon p-value over the same family: `{stats['holm_wilcoxon_p']:.17g}`.
- Acceptance result: **PASS**. One internally consistent 12-row source reproduces the declared mean, interval, and 8/2/2 counts.

## Why the alternative source gives 6/1/5

`{ALTERNATIVE.relative_to(ROOT)}` (SHA-256 `{alternative['source_sha256']}`) is a different, older structural-generalization contrast. It compares candidate `{alternative['candidate']}` with baseline `{alternative['baseline']}`, rather than refined sensitivity with global magnitude. Its values are the older group-normalized errors and its declared outcome rule uses relative tie tolerance `{alternative['tie_relative_tolerance']}` plus epsilon `{alternative['tie_epsilon']}`. It therefore represents a different candidate refinement stage, a different magnitude baseline, a different normalized structural summary, and a different tie rule. Its independently read mean baseline-minus-candidate effect is `{alternative['mean_effect']:.17g}`, with recorded wins/losses/ties `{alternative['wins']}/{alternative['losses']}/{alternative['ties']}`. It is not combined with, and is not used for, the headline contrast or paired figure.
"""
    MD_OUTPUT.write_text(markdown, encoding="utf-8")


def build_figure(paired: pd.DataFrame, stats: dict[str, float | int | str]) -> None:
    colors = {
        "baseline": "#1F4E79",
        "candidate": "#E69F00",
        "win": "#0072B2",
        "loss": "#D55E00",
        "tie": "#4D4D4D",
    }
    outcome_styles = {
        "win": {"linestyle": "-", "symbol": "+"},
        "loss": {"linestyle": (0, (3.0, 1.5)), "symbol": r"$-$"},
        "tie": {"linestyle": (0, (1.0, 1.5)), "symbol": r"$=$"},
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 3.15),
        sharey=True,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.70, wspace=0.18)
    for axis, case in zip(axes, ("ieee14", "ieee30", "ieee57"), strict=True):
        frame = paired.loc[paired["case"].eq(case)].reset_index(drop=True)
        x = np.arange(1, len(frame) + 1)
        for position, row in zip(x, frame.itertuples(index=False), strict=True):
            style = outcome_styles[row.outcome]
            axis.plot(
                [position, position],
                [row.baseline_E_supp, row.candidate_E_supp],
                color=colors[row.outcome],
                linestyle=style["linestyle"],
                linewidth=1.7,
                alpha=0.95,
                zorder=1,
            )
        axis.scatter(
            x,
            frame["baseline_E_supp"],
            s=43,
            marker="o",
            facecolor="white",
            edgecolor=colors["baseline"],
            linewidth=1.35,
            zorder=4,
        )
        axis.scatter(
            x,
            frame["candidate_E_supp"],
            s=28,
            marker="D",
            facecolor=colors["candidate"],
            edgecolor="#303030",
            linewidth=0.45,
            zorder=3,
        )
        for position, row in zip(x, frame.itertuples(index=False), strict=True):
            if row.baseline_E_supp == 0.0 and row.candidate_E_supp == 0.0:
                inward = 11 if position <= 2 else -11
                axis.annotate(
                    "= tie:\nsaturation",
                    (position, 0.0),
                    xytext=(inward, 22),
                    textcoords="offset points",
                    ha="left" if inward > 0 else "right",
                    va="bottom",
                    fontsize=8.0,
                    color=colors["tie"],
                    linespacing=0.95,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": colors["tie"],
                        "linewidth": 0.55,
                        "shrinkA": 1.0,
                        "shrinkB": 3.0,
                    },
                )
            elif max(abs(row.baseline_E_supp), abs(row.candidate_E_supp)) < 1e-12:
                inward = 9 if position <= 2 else -9
                axis.annotate(
                    "$-$ near\nzero",
                    (position, 0.0),
                    xytext=(inward, 18),
                    textcoords="offset points",
                    ha="left" if inward > 0 else "right",
                    va="bottom",
                    fontsize=8.0,
                    color=colors["loss"],
                    linespacing=0.95,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": colors["loss"],
                        "linewidth": 0.55,
                        "shrinkA": 1.0,
                        "shrinkB": 3.0,
                    },
                )
        case_label = case.replace("ieee", "IEEE-")
        case_counts = frame["outcome"].value_counts()
        axis.set_title(
            f"{case_label} "
            f"(W/L/T = {case_counts.get('win', 0)}/{case_counts.get('loss', 0)}/"
            f"{case_counts.get('tie', 0)})",
            pad=6,
        )
        axis.set_xticks(x, [f"G{index}" for index in x])
        axis.set_xlabel("Structural group")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_ylim(-0.008, 0.325)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "Complete-matrix Ridge\npreservation error " r"$E_{\mathrm{supp}}$"
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=colors["baseline"],
            markeredgewidth=1.3,
            linestyle="none",
            label="Global magnitude",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markersize=4.6,
            markerfacecolor=colors["candidate"],
            markeredgecolor="#303030",
            markeredgewidth=0.45,
            linestyle="none",
            label="Refined sensitivity",
        ),
        *[
            Line2D(
                [],
                [],
                color=colors[outcome],
                linestyle=outcome_styles[outcome]["linestyle"],
                linewidth=1.7,
                label=f"{outcome_styles[outcome]['symbol']} {outcome.capitalize()}",
            )
            for outcome in ("win", "loss", "tie")
        ],
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    figure.suptitle(
        (
            f"Overall W/L/T = {stats['wins']}/{stats['losses']}/{stats['ties']}; "
            "W/L/T denotes wins/losses/ties for refined support; lower is better"
        ),
        fontsize=8.0,
        y=0.84,
    )
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_OUTPUT,
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Creator": __file__},
    )
    plt.close(figure)


def main() -> None:
    paired, stats = reconcile()
    alternative = alternative_summary()
    write_reconciliation(paired, stats, alternative)
    build_figure(paired, stats)
    print(f"wrote {CSV_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {MD_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {FIGURE_OUTPUT.relative_to(ROOT)}")
    print(
        "reconciled mean/CI/W-L-T:",
        stats["mean_effect"],
        (stats["ci_low"], stats["ci_high"]),
        (stats["wins"], stats["losses"], stats["ties"]),
    )


if __name__ == "__main__":
    main()
