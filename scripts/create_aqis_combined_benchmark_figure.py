from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "outputs" / "aqis_results_summary"
FIGURE_DIR = ROOT / "figures"

ALPHA = 1.0e-4


@dataclass(frozen=True)
class NonlinearEndpoint:
    case: str
    scenario: str
    label: str
    output_dir: Path
    sweep_name: str
    sweep_value: float


AC_LINEARIZED_ENDPOINTS = [
    {
        "case": "ieee300",
        "scenario": "missing_20pct",
        "label": "IEEE300\nmiss. 20%",
        "n_seeds": 10,
        "pinv_rmse_mean": 6.325480812534229,
        "ridge_qsvt_rmse_mean": 0.6936884974599471,
        "condition_number_median": 1.369050890634372e19,
        "alpha": ALPHA,
        "source": "prompt/paper-ready AC-linearized endpoint",
    },
    {
        "case": "ieee300",
        "scenario": "bad_data_10pct",
        "label": "IEEE300\nbad 10%",
        "n_seeds": 10,
        "pinv_rmse_mean": 10.183686526511908,
        "ridge_qsvt_rmse_mean": 1.1196343621070541,
        "condition_number_median": 4.705539654593432e18,
        "alpha": ALPHA,
        "source": "prompt/paper-ready AC-linearized endpoint",
    },
    {
        "case": "ieee57",
        "scenario": "missing_20pct_control",
        "label": "IEEE57\ncontrol",
        "n_seeds": 10,
        "pinv_rmse_mean": 0.1511034382514792,
        "ridge_qsvt_rmse_mean": 0.1510017311047745,
        "condition_number_median": 17301.3151047738,
        "alpha": ALPHA,
        "source": "outputs/paper_ready_results/tables/table2_ac_linearized_high_stress.csv",
    },
]

NONLINEAR_ENDPOINTS = [
    NonlinearEndpoint(
        case="ieee300",
        scenario="missing_20pct",
        label="IEEE300\nmiss. 20%",
        output_dir=ROOT / "outputs" / "nonlinear_ac_ieee300_seed10",
        sweep_name="nonlinear_missing_sweep",
        sweep_value=0.2,
    ),
    NonlinearEndpoint(
        case="ieee300",
        scenario="bad_data_10pct",
        label="IEEE300\nbad 10%",
        output_dir=ROOT / "outputs" / "nonlinear_ac_ieee300_seed10",
        sweep_name="nonlinear_bad_data_ratio_sweep",
        sweep_value=0.1,
    ),
    NonlinearEndpoint(
        case="ieee57",
        scenario="missing_20pct_control",
        label="IEEE57\ncontrol",
        output_dir=ROOT / "outputs" / "nonlinear_ac_ieee57_seed10",
        sweep_name="nonlinear_missing_sweep",
        sweep_value=0.2,
    ),
]


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ac_df = pd.DataFrame(AC_LINEARIZED_ENDPOINTS)
    nonlinear_df = _build_nonlinear_summary()

    ac_summary_path = SUMMARY_DIR / "ac_linearized_summary.csv"
    nonlinear_summary_path = SUMMARY_DIR / "nonlinear_ac_summary.csv"
    ac_df.drop(columns=["label"]).to_csv(ac_summary_path, index=False)
    nonlinear_df.drop(columns=["label"]).to_csv(nonlinear_summary_path, index=False)

    _write_summary_md(ac_df, nonlinear_df)
    _plot_combined_figure(ac_df, nonlinear_df)


def _build_nonlinear_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for endpoint in NONLINEAR_ENDPOINTS:
        summary_path = endpoint.output_dir / "summary_metrics.csv"
        aggregate_path = endpoint.output_dir / "aggregate_metrics.csv"
        if not summary_path.exists() or not aggregate_path.exists():
            raise FileNotFoundError(
                f"Missing nonlinear output CSVs for {endpoint.case} {endpoint.scenario}"
            )

        summary = pd.read_csv(summary_path)
        aggregate = pd.read_csv(aggregate_path)
        subset = summary[
            (summary["sweep_name"] == endpoint.sweep_name)
            & np.isclose(summary["sweep_value"].astype(float), endpoint.sweep_value)
        ]

        pinv = _single_estimator_row(subset, "pseudoinverse", summary_path)
        ridge = _single_estimator_row(subset, "ridge", summary_path)
        qsvt = _single_estimator_row(subset, "qsvt_regularized", summary_path)
        if not np.isclose(
            float(ridge["rmse_mean"]),
            float(qsvt["rmse_mean"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Ridge and qsvt_regularized RMSE differ in {summary_path} "
                f"for {endpoint.sweep_name}={endpoint.sweep_value}"
            )

        agg_subset = aggregate[
            (aggregate["sweep_name"] == endpoint.sweep_name)
            & np.isclose(aggregate["sweep_value"].astype(float), endpoint.sweep_value)
        ]
        rows.append(
            {
                "case": endpoint.case,
                "scenario": endpoint.scenario,
                "n_seeds": int(pinv["n_trials"]),
                "pinv_rmse_mean": float(pinv["rmse_mean"]),
                "pinv_rmse_std": float(pinv["rmse_std"]),
                "ridge_qsvt_rmse_mean": float(ridge["rmse_mean"]),
                "ridge_qsvt_rmse_std": float(ridge["rmse_std"]),
                "pinv_converged": _converged_count(agg_subset, "pseudoinverse"),
                "ridge_qsvt_converged": _converged_count(agg_subset, "ridge"),
                "pinv_iterations_mean": float(pinv["iterations_mean"]),
                "ridge_qsvt_iterations_mean": float(ridge["iterations_mean"]),
                "alpha": ALPHA,
                "label": endpoint.label,
                "source_output": str(endpoint.output_dir.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def _single_estimator_row(df: pd.DataFrame, estimator: str, path: Path) -> pd.Series:
    rows = df[df["estimator"] == estimator]
    if len(rows) != 1:
        raise ValueError(f"Expected one {estimator} row in {path}, found {len(rows)}")
    return rows.iloc[0]


def _converged_count(df: pd.DataFrame, estimator: str) -> int:
    rows = df[df["estimator"] == estimator]
    if rows.empty:
        raise ValueError(f"Missing aggregate rows for estimator {estimator}")
    return int(rows["converged"].sum())


def _plot_combined_figure(ac_df: pd.DataFrame, nonlinear_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    colors = ["#4C566A", "#2E7D7D"]

    _plot_panel(
        axes[0],
        ac_df,
        title="(a) AC-linearized",
        colors=colors,
        annotate_conditions=True,
    )
    _plot_panel(
        axes[1],
        nonlinear_df,
        title="(b) Nonlinear AC",
        colors=colors,
        annotate_conditions=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    pdf_path = FIGURE_DIR / "figure1_benchmark_rmse_combined.pdf"
    png_path = FIGURE_DIR / "figure1_benchmark_rmse_combined.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    colors: list[str],
    annotate_conditions: bool,
) -> None:
    x = np.arange(len(df))
    width = 0.34
    pinv_values = df["pinv_rmse_mean"].to_numpy(dtype=float)
    ridge_values = df["ridge_qsvt_rmse_mean"].to_numpy(dtype=float)

    ax.bar(
        x - width / 2,
        pinv_values,
        width,
        label="Pseudoinverse",
        color=colors[0],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        ridge_values,
        width,
        label="Ridge / QSVT-target",
        color=colors[1],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Mean RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist())
    ax.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.6, alpha=0.8)
    ax.grid(axis="y", which="minor", color="#E6E6E6", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    min_value = min(float(pinv_values.min()), float(ridge_values.min()))
    max_value = max(float(pinv_values.max()), float(ridge_values.max()))
    ax.set_ylim(min_value / 2.5, max_value * (4.5 if annotate_conditions else 2.5))

    if annotate_conditions:
        for index, row in df.iterrows():
            if row["case"] == "ieee300":
                y = max(float(row["pinv_rmse_mean"]), float(row["ridge_qsvt_rmse_mean"])) * 1.45
                ax.text(
                    index,
                    y,
                    f"$\\kappa\\approx{float(row['condition_number_median']):.2e}$",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )


def _write_summary_md(ac_df: pd.DataFrame, nonlinear_df: pd.DataFrame) -> None:
    lines = [
        "# AQIS Results Summary",
        "",
        "Generated by `scripts/create_aqis_combined_benchmark_figure.py`.",
        "",
        "## Nonlinear AC Source Outputs",
        "",
        "- `outputs/nonlinear_ac_ieee300_seed10/summary_metrics.csv`",
        "- `outputs/nonlinear_ac_ieee300_seed10/aggregate_metrics.csv`",
        "- `outputs/nonlinear_ac_ieee57_seed10/summary_metrics.csv`",
        "- `outputs/nonlinear_ac_ieee57_seed10/aggregate_metrics.csv`",
        "",
        "The nonlinear AC outputs include Huber IRLS rows, but the AQIS combined figure and",
        "summary CSV exclude Huber and use only pseudoinverse, ridge, and the matching",
        "`qsvt_regularized` rows. The ridge and QSVT-target rows were checked to have",
        "identical mean RMSE at the selected endpoints.",
        "",
        "## AC-Linearized Endpoints",
        "",
        "```csv",
        ac_df.drop(columns=["label"]).to_csv(index=False).strip(),
        "```",
        "",
        "## Nonlinear AC Endpoints",
        "",
        "```csv",
        nonlinear_df.drop(columns=["label"]).to_csv(index=False).strip(),
        "```",
        "",
        "The selected nonlinear IEEE300 endpoints are real consistency-check runs over 10",
        "seeds. They do not show the large pseudoinverse instability reduction seen in the",
        "AC-linearized endpoints; ridge/QSVT-target is essentially tied with pseudoinverse",
        "and slightly higher in the two IEEE300 nonlinear rows. The IEEE57 nonlinear row is",
        "a lower-condition control with identical RMSE to numerical precision.",
    ]
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
