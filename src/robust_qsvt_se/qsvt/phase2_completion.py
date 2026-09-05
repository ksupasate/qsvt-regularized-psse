from __future__ import annotations

# ruff: noqa: E501,I001

import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import (
    CLAIM_BOUNDARY,
    current_command,
    git_commit,
    utc_timestamp,
)
from robust_qsvt_se.qsvt.phase2_preconditioned_alpha import (
    build_phase2_alpha_selection_report,
    run_phase2_preconditioned_alpha_sweeps,
    select_alpha_diagnostics,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2_VARIANT_ORDER = [
    "original_ridge",
    "coordinate_preconditioned_ridge",
    "transformed_penalty_preconditioned_ridge",
    "original_qsvt_diagnostic",
    "preconditioned_qsvt_diagnostic",
]
PHASE2_REQUIRED_CASES = ["ieee118", "ieee300"]
PHASE2_PHASE_ALPHA = 1.0e-2
PHASE2_PHASE_FULL_DOMAIN_ERROR = 4.668e-4
PHASE2_PHASE_ACTUAL_SV_ERROR = 8.673e-5
PHASE2_PHASE_DEGREE = 201
PHASE2_PHASE_COUNT = 202
PHASE2_APPROX_QUERY_COUNT = 403
DIAGNOSTIC_ALPHA_CAVEAT = (
    "Alpha selection is diagnostic and controlled-benchmark-specific. It is not a "
    "field-calibrated operational rule."
)


def build_phase2_complete_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_complete_summary_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sweep_results_path = Path(resolved["sweep_results_csv"])
    alpha_summary_path = Path(resolved["alpha_selection_summary_csv"])
    alpha_trace_path = Path(resolved["alpha_selection_trace_csv"])
    _ensure_phase2_inputs(sweep_results_path, alpha_summary_path, alpha_trace_path)

    results = pd.read_csv(sweep_results_path)
    alpha_summary = pd.read_csv(alpha_summary_path)
    alpha_trace = pd.read_csv(alpha_trace_path)
    case_names = PHASE2_REQUIRED_CASES.copy()
    optional_ieee57 = Path(resolved["optional_ieee57_results_csv"])
    if optional_ieee57.is_file():
        ieee57_results = pd.read_csv(optional_ieee57)
        if not ieee57_results.empty:
            ieee57_summary, ieee57_trace = select_alpha_diagnostics(ieee57_results)
            results = pd.concat([results, ieee57_results], ignore_index=True)
            alpha_summary = pd.concat([alpha_summary, ieee57_summary], ignore_index=True)
            alpha_trace = pd.concat([alpha_trace, ieee57_trace], ignore_index=True)
            case_names.append("ieee57")

    complete = _complete_summary_frame(results, alpha_summary, alpha_trace, case_names)
    best_alpha = _best_alpha_by_metric(alpha_summary)
    variant_comparison = _variant_comparison_frame(complete)
    case_comparison = _case_comparison_frame(complete)

    paths = {
        "phase2_complete_summary_csv": output_dir / "phase2_complete_summary.csv",
        "phase2_complete_summary_json": output_dir / "phase2_complete_summary.json",
        "phase2_best_alpha_by_metric_csv": output_dir / "phase2_best_alpha_by_metric.csv",
        "phase2_variant_comparison_csv": output_dir / "phase2_variant_comparison.csv",
        "phase2_case_comparison_csv": output_dir / "phase2_case_comparison.csv",
        "phase2_key_findings_md": output_dir / "phase2_key_findings.md",
        "phase2_manifest_json": output_dir / "phase2_manifest.json",
    }
    complete.to_csv(paths["phase2_complete_summary_csv"], index=False)
    write_json(
        paths["phase2_complete_summary_json"],
        {"rows": complete.to_dict(orient="records")},
    )
    best_alpha.to_csv(paths["phase2_best_alpha_by_metric_csv"], index=False)
    variant_comparison.to_csv(paths["phase2_variant_comparison_csv"], index=False)
    case_comparison.to_csv(paths["phase2_case_comparison_csv"], index=False)
    paths["phase2_key_findings_md"].write_text(
        _key_findings_markdown(complete, best_alpha),
        encoding="utf-8",
    )
    write_json(
        paths["phase2_manifest_json"],
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {key: str(path) for key, path in paths.items()},
            "selection_rule_for_complete_summary": "joint_score_alpha per case and variant",
            "metric_note": "Rows aggregate the selected alpha over controlled scenarios.",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "summary": complete,
        "best_alpha": best_alpha,
        "artifacts": paths,
    }


def build_phase2_figures(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_figures_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sweep_summary_path = Path(resolved["sweep_summary_csv"])
    alpha_trace_path = Path(resolved["alpha_selection_trace_csv"])
    if not sweep_summary_path.is_file():
        run_phase2_preconditioned_alpha_sweeps({"output_dir": str(sweep_summary_path.parent)})
    if not alpha_trace_path.is_file():
        build_phase2_alpha_selection_report({"output_dir": str(alpha_trace_path.parent)})

    sweep = pd.read_csv(sweep_summary_path)
    trace = pd.read_csv(alpha_trace_path)
    artifacts: dict[str, str] = {}

    figure_specs = [
        (
            "fig_phase2_ieee300_residual_vs_alpha",
            lambda: _plot_metric_vs_alpha(
                sweep,
                case_name="ieee300",
                metric="mean_residual_norm",
                ylabel="Mean residual norm",
                title="IEEE300 Phase 2 residual norm versus alpha",
            ),
        ),
        (
            "fig_phase2_ieee300_rmse_vs_alpha",
            lambda: _plot_metric_vs_alpha(
                sweep,
                case_name="ieee300",
                metric="mean_rmse_if_available",
                ylabel="Mean RMSE",
                title="IEEE300 Phase 2 RMSE versus alpha",
            ),
        ),
        (
            "fig_phase2_ieee300_qsvt_error_vs_alpha",
            lambda: _plot_metric_vs_alpha(
                sweep,
                case_name="ieee300",
                metric="mean_qsvt_full_interval_approx_error",
                ylabel="Mean full-interval QSVT approximation error",
                title="IEEE300 Phase 2 QSVT diagnostic error versus alpha",
                yscale="log",
            ),
        ),
        (
            "fig_phase2_ieee300_residual_rmse_qsvt_tradeoff",
            lambda: _plot_ieee300_tradeoff(sweep),
        ),
        (
            "fig_phase2_ieee118_qsvt_error_vs_alpha",
            lambda: _plot_metric_vs_alpha(
                sweep,
                case_name="ieee118",
                metric="mean_qsvt_full_interval_approx_error",
                ylabel="Mean full-interval QSVT approximation error",
                title="IEEE118 Phase 2 QSVT diagnostic error versus alpha",
                yscale="log",
            ),
        ),
        (
            "fig_phase2_ieee118_residual_vs_alpha",
            lambda: _plot_metric_vs_alpha(
                sweep,
                case_name="ieee118",
                metric="mean_residual_norm",
                ylabel="Mean residual norm",
                title="IEEE118 Phase 2 residual norm versus alpha",
            ),
        ),
        (
            "fig_phase2_variant_comparison_ieee300",
            lambda: _plot_ieee300_variant_comparison(sweep),
        ),
        (
            "fig_phase2_original_vs_preconditioned_kappa",
            lambda: _plot_kappa_comparison(sweep),
        ),
        (
            "fig_phase2_alpha_selection_score",
            lambda: _plot_alpha_selection_score(trace),
        ),
    ]
    for stem, factory in figure_specs:
        fig = factory()
        artifacts.update(_save_figure(fig, output_dir / stem))
        plt.close(fig)

    captions_path = output_dir / "phase2_figure_captions.md"
    captions_path.write_text(_figure_captions_markdown(), encoding="utf-8")
    manifest_path = output_dir / "phase2_figures_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {**artifacts, "phase2_figure_captions_md": str(captions_path)},
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    artifacts["phase2_figure_captions_md"] = str(captions_path)
    artifacts["phase2_figures_manifest_json"] = str(manifest_path)
    return {"output_dir": output_dir, "artifacts": artifacts}


def run_phase2_optional_ieee57(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_ieee57_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    status_json = output_dir / "ieee57_phase2_status.json"
    status_md = output_dir / "ieee57_phase2_status.md"
    results_csv = output_dir / "ieee57_phase2_results.csv"
    manifest_path = output_dir / "ieee57_phase2_manifest.json"

    if bool(resolved["reuse_existing"]) and results_csv.is_file():
        results = pd.read_csv(results_csv)
        status = _ieee57_status(
            "completed_from_existing_results", "Existing IEEE57 results reused."
        )
        status.update(_ieee57_result_summary(results))
    else:
        started = time.perf_counter()
        try:
            run = run_phase2_preconditioned_alpha_sweeps(
                {
                    "output_dir": str(output_dir),
                    "cases": ["ieee57"],
                    "case_source": resolved["case_source"],
                    "base_seed": resolved["base_seed"],
                    "fallback_to_synthetic": False,
                    "seeds": resolved["seeds"],
                    "alphas": resolved["alphas"],
                    "noise_stds": resolved["noise_stds"],
                    "missing_ratios": resolved["missing_ratios"],
                    "bad_data_ratios": resolved["bad_data_ratios"],
                }
            )
            results = run["results"]
            if results.empty:
                status = _ieee57_status(
                    "skipped",
                    "IEEE57 builder returned no successful Phase 2 rows.",
                )
            else:
                results.to_csv(results_csv, index=False)
                status = _ieee57_status("completed", "IEEE57 Phase 2 sweep completed.")
                status.update(_ieee57_result_summary(results))
        except Exception as exc:  # pragma: no cover - exercised through skip record behavior
            status = _ieee57_status(
                "skipped",
                f"IEEE57 Phase 2 extension skipped after runtime-safe attempt failed: {exc}",
            )
        status["runtime_seconds"] = time.perf_counter() - started

    write_json(status_json, status)
    status_md.write_text(_ieee57_status_markdown(status), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "status": status,
            "artifacts": {
                "ieee57_phase2_status_json": str(status_json),
                "ieee57_phase2_status_md": str(status_md),
                "ieee57_phase2_results_csv": str(results_csv) if results_csv.is_file() else "",
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "status": status,
        "artifacts": {
            "status_json": status_json,
            "status_md": status_md,
            "manifest": manifest_path,
        },
    }


def build_phase2_manuscript_text(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_manuscript_text_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    complete_path = Path(resolved["complete_summary_csv"])
    summary = pd.read_csv(complete_path) if complete_path.is_file() else pd.DataFrame()
    paths = {
        "transformed_penalty_explanation_md": output_dir / "transformed_penalty_explanation.md",
        "phase2_results_paragraph_md": output_dir / "phase2_results_paragraph.md",
        "phase2_limitations_paragraph_md": output_dir / "phase2_limitations_paragraph.md",
        "phase2_claim_safe_wording_md": output_dir / "phase2_claim_safe_wording.md",
        "phase2_methods_equations_md": output_dir / "phase2_methods_equations.md",
    }
    paths["transformed_penalty_explanation_md"].write_text(
        _transformed_penalty_explanation(),
        encoding="utf-8",
    )
    paths["phase2_results_paragraph_md"].write_text(
        _phase2_results_paragraph(summary),
        encoding="utf-8",
    )
    paths["phase2_limitations_paragraph_md"].write_text(
        _phase2_limitations_paragraph(),
        encoding="utf-8",
    )
    paths["phase2_claim_safe_wording_md"].write_text(
        _phase2_claim_safe_wording(),
        encoding="utf-8",
    )
    paths["phase2_methods_equations_md"].write_text(
        _phase2_methods_equations(),
        encoding="utf-8",
    )
    manifest_path = output_dir / "phase2_manuscript_text_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {key: str(path) for key, path in paths.items()},
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    paths["phase2_manuscript_text_manifest_json"] = manifest_path
    return {"output_dir": output_dir, "artifacts": paths}


def _ensure_phase2_inputs(
    sweep_results_path: Path,
    alpha_summary_path: Path,
    alpha_trace_path: Path,
) -> None:
    if not sweep_results_path.is_file():
        run_phase2_preconditioned_alpha_sweeps({"output_dir": str(sweep_results_path.parent)})
    if not alpha_summary_path.is_file() or not alpha_trace_path.is_file():
        build_phase2_alpha_selection_report({"output_dir": str(alpha_summary_path.parent)})


def _complete_summary_frame(
    results: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    alpha_trace: pd.DataFrame,
    case_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_name in case_names:
        for variant_name in PHASE2_VARIANT_ORDER:
            selected = _selected_alpha_row(alpha_summary, case_name, variant_name)
            if selected.empty:
                continue
            alpha = float(selected.iloc[0]["selected_alpha"])
            group = results[
                (results["case_name"].astype(str).str.lower() == case_name)
                & (results["variant_name"].astype(str) == variant_name)
                & np.isclose(results["alpha"].astype(float), alpha)
            ].copy()
            if group.empty:
                continue
            score = _alpha_trace_score(alpha_trace, case_name, variant_name, alpha)
            rows.append(_complete_summary_row(case_name, variant_name, alpha, group, score))
    return pd.DataFrame(rows, columns=_complete_summary_columns())


def _complete_summary_row(
    case_name: str,
    variant_name: str,
    alpha: float,
    group: pd.DataFrame,
    alpha_selection_score: float,
) -> dict[str, Any]:
    status_counts = group["status"].astype(str).value_counts().to_dict()
    qsvt_error = _mean_numeric(group, "qsvt_full_interval_approx_error")
    original_error = _reference_qsvt_error(group, case_name, alpha, original=True)
    preconditioned_error = _reference_qsvt_error(group, case_name, alpha, original=False)
    return {
        "case_name": case_name,
        "variant_name": variant_name,
        "alpha": alpha,
        "m": round(_median_numeric(group, "m")),
        "n": round(_median_numeric(group, "n")),
        "rank": round(_median_numeric(group, "rank")),
        "condition_number_original": _median_numeric(group, "condition_number_original"),
        "condition_number_preconditioned_if_applicable": _median_numeric(
            group,
            "condition_number_preconditioned_if_applicable",
        ),
        "residual_norm": _mean_numeric(group, "residual_norm"),
        "weighted_residual_norm": _mean_numeric(group, "weighted_residual_norm"),
        "rmse_if_available": _mean_numeric(group, "rmse_if_available"),
        "solution_norm": _mean_numeric(group, "solution_norm"),
        "relative_solution_error_vs_original_ridge": _mean_numeric(
            group,
            "relative_solution_error_vs_original_ridge",
        ),
        "relative_solution_error_vs_transformed_penalty": _mean_numeric(
            group,
            "relative_solution_error_vs_transformed_penalty",
        ),
        "qsvt_full_interval_error": qsvt_error,
        "qsvt_actual_singular_value_error": _mean_numeric(
            group,
            "qsvt_actual_singular_value_error",
        ),
        "qsvt_degree": round(_mean_numeric(group, "qsvt_degree")),
        "qsvt_query_count": round(_mean_numeric(group, "qsvt_query_count")),
        "phase_validation_status": _phase_status_for_alpha(alpha),
        "alpha_selection_score": alpha_selection_score,
        "status": _aggregate_status(status_counts),
        "interpretation": _complete_interpretation(
            variant_name,
            qsvt_error=qsvt_error,
            original_error=original_error,
            preconditioned_error=preconditioned_error,
        ),
        "caveat": _complete_caveat(variant_name),
    }


def _selected_alpha_row(
    alpha_summary: pd.DataFrame,
    case_name: str,
    variant_name: str,
) -> pd.DataFrame:
    rows = alpha_summary[
        (alpha_summary["case_name"].astype(str).str.lower() == case_name)
        & (alpha_summary["variant_name"].astype(str) == variant_name)
        & (alpha_summary["selection_criterion"].astype(str) == "joint_score_alpha")
    ]
    if not rows.empty:
        return rows
    return alpha_summary[
        (alpha_summary["case_name"].astype(str).str.lower() == case_name)
        & (alpha_summary["variant_name"].astype(str) == variant_name)
    ].head(1)


def _best_alpha_by_metric(alpha_summary: pd.DataFrame) -> pd.DataFrame:
    frame = alpha_summary.copy()
    rename_map = {
        "qsvt_resource_friendly_alpha": "legacy_qsvt_resource_friendly_alpha",
    }
    frame["selection_criterion"] = frame["selection_criterion"].replace(rename_map)
    return frame.sort_values(
        ["case_name", "variant_name", "selection_criterion"],
        kind="mergesort",
    )


def _variant_comparison_frame(complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_name, group in complete.groupby("case_name", sort=False):
        original = _variant_row(group, "original_ridge")
        transformed = _variant_row(group, "transformed_penalty_preconditioned_ridge")
        coordinate = _variant_row(group, "coordinate_preconditioned_ridge")
        pre_qsvt = _variant_row(group, "preconditioned_qsvt_diagnostic")
        if original is None:
            continue
        rows.append(
            {
                "case_name": case_name,
                "coordinate_residual_ratio_vs_original": _ratio(
                    coordinate,
                    original,
                    "residual_norm",
                ),
                "coordinate_rmse_ratio_vs_original": _ratio(
                    coordinate,
                    original,
                    "rmse_if_available",
                ),
                "transformed_residual_ratio_vs_original": _ratio(
                    transformed,
                    original,
                    "residual_norm",
                ),
                "transformed_rmse_ratio_vs_original": _ratio(
                    transformed,
                    original,
                    "rmse_if_available",
                ),
                "preconditioned_qsvt_error_ratio_vs_original": _ratio(
                    pre_qsvt,
                    original,
                    "qsvt_full_interval_error",
                ),
                "interpretation": _variant_comparison_interpretation(
                    coordinate,
                    transformed,
                    original,
                    pre_qsvt,
                ),
            }
        )
    return pd.DataFrame(rows)


def _case_comparison_frame(complete: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_name, group in complete.groupby("case_name", sort=False):
        original = _variant_row(group, "original_ridge")
        pre = _variant_row(group, "preconditioned_qsvt_diagnostic")
        if original is None or pre is None:
            continue
        rows.append(
            {
                "case_name": case_name,
                "original_condition_number": original["condition_number_original"],
                "preconditioned_condition_number": pre[
                    "condition_number_preconditioned_if_applicable"
                ],
                "condition_number_ratio": _safe_divide(
                    pre["condition_number_preconditioned_if_applicable"],
                    original["condition_number_original"],
                ),
                "original_qsvt_error": original["qsvt_full_interval_error"],
                "preconditioned_qsvt_error": pre["qsvt_full_interval_error"],
                "qsvt_error_ratio": _safe_divide(
                    pre["qsvt_full_interval_error"],
                    original["qsvt_full_interval_error"],
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_vs_alpha(
    sweep: pd.DataFrame,
    *,
    case_name: str,
    metric: str,
    ylabel: str,
    title: str,
    yscale: str = "linear",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    subset = sweep[sweep["case_name"].astype(str).str.lower() == case_name].copy()
    for variant in PHASE2_VARIANT_ORDER:
        group = subset[subset["variant_name"].astype(str) == variant].sort_values("alpha")
        if group.empty:
            continue
        ax.plot(
            group["alpha"].astype(float),
            pd.to_numeric(group[metric], errors="coerce"),
            marker=_marker_for_variant(variant),
            linewidth=1.8,
            label=_label_for_variant(variant),
        )
    _finish_alpha_axis(ax, ylabel=ylabel, title=title, yscale=yscale)
    fig.tight_layout()
    return fig


def _plot_ieee300_tradeoff(sweep: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharex=True)
    metrics = [
        ("mean_residual_norm", "Mean residual norm", "linear"),
        ("mean_rmse_if_available", "Mean RMSE", "linear"),
        ("mean_qsvt_full_interval_approx_error", "Mean QSVT error", "log"),
    ]
    subset = sweep[sweep["case_name"].astype(str).str.lower() == "ieee300"].copy()
    variants = [
        "original_ridge",
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
        "preconditioned_qsvt_diagnostic",
    ]
    for ax, (metric, ylabel, yscale) in zip(axes, metrics, strict=True):
        for variant in variants:
            group = subset[subset["variant_name"].astype(str) == variant].sort_values("alpha")
            ax.plot(
                group["alpha"].astype(float),
                pd.to_numeric(group[metric], errors="coerce"),
                marker=_marker_for_variant(variant),
                linewidth=1.7,
                label=_label_for_variant(variant),
            )
        _finish_alpha_axis(
            ax,
            ylabel=ylabel,
            title=ylabel,
            yscale=yscale,
            legend=False,
        )
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("IEEE300 Phase 2 residual, RMSE, and QSVT diagnostic tradeoff", y=1.03)
    fig.tight_layout()
    return fig


def _plot_ieee300_variant_comparison(sweep: pd.DataFrame) -> plt.Figure:
    subset = sweep[
        (sweep["case_name"].astype(str).str.lower() == "ieee300")
        & np.isclose(sweep["alpha"].astype(float), PHASE2_PHASE_ALPHA)
    ].copy()
    variants = [
        "original_ridge",
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
        "preconditioned_qsvt_diagnostic",
    ]
    metrics = [
        ("mean_residual_norm", "Residual"),
        ("mean_rmse_if_available", "RMSE"),
        ("mean_qsvt_full_interval_approx_error", "QSVT error"),
    ]
    values = []
    for variant in variants:
        row = subset[subset["variant_name"].astype(str) == variant].head(1)
        values.append(
            [
                float(pd.to_numeric(row[column], errors="coerce").iloc[0])
                if not row.empty
                else np.nan
                for column, _ in metrics
            ]
        )
    array = np.asarray(values, dtype=float)
    baseline = np.where(np.isfinite(array[0]) & (array[0] != 0), array[0], 1.0)
    normalized = array / baseline

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    x = np.arange(len(variants))
    width = 0.23
    for index, (_, label) in enumerate(metrics):
        ax.bar(x + (index - 1) * width, normalized[:, index], width=width, label=label)
    ax.axhline(1.0, color="0.25", linewidth=0.9, linestyle="--")
    ax.set_title("IEEE300 Phase 2 variant comparison at alpha = 1e-2")
    ax.set_ylabel("Ratio versus original Ridge at alpha = 1e-2")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(variant) for variant in variants], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    fig.tight_layout()
    return fig


def _plot_kappa_comparison(sweep: pd.DataFrame) -> plt.Figure:
    rows = []
    for case_name in PHASE2_REQUIRED_CASES:
        subset = sweep[
            (sweep["case_name"].astype(str).str.lower() == case_name)
            & (sweep["variant_name"].astype(str) == "preconditioned_qsvt_diagnostic")
            & np.isclose(sweep["alpha"].astype(float), PHASE2_PHASE_ALPHA)
        ].head(1)
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "case_name": case_name.upper(),
                "Original": row["median_condition_number_original"],
                "Column-equilibrated": row["median_condition_number_preconditioned_if_applicable"],
            }
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(frame))
    width = 0.34
    ax.bar(x - width / 2, frame["Original"], width=width, label="Original")
    ax.bar(
        x + width / 2,
        frame["Column-equilibrated"],
        width=width,
        label="Column-equilibrated",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["case_name"])
    ax.set_ylabel("Median condition number (log scale)")
    ax.set_title("Original versus column-equilibrated condition number")
    ax.legend()
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    fig.tight_layout()
    return fig


def _plot_alpha_selection_score(trace: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    for ax, case_name in zip(axes, PHASE2_REQUIRED_CASES, strict=True):
        subset = trace[trace["case_name"].astype(str).str.lower() == case_name].copy()
        for variant in PHASE2_VARIANT_ORDER:
            group = subset[subset["variant_name"].astype(str) == variant].sort_values("alpha")
            if group.empty:
                continue
            ax.plot(
                group["alpha"].astype(float),
                pd.to_numeric(group["joint_score"], errors="coerce"),
                marker=_marker_for_variant(variant),
                linewidth=1.6,
                label=_label_for_variant(variant),
            )
        _finish_alpha_axis(
            ax,
            ylabel="Diagnostic joint score",
            title=f"{case_name.upper()} alpha-selection score",
            legend=False,
        )
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    paths = {}
    for suffix in [".png", ".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        paths[path.name] = str(path)
    return paths


def _finish_alpha_axis(
    ax: plt.Axes,
    *,
    ylabel: str,
    title: str,
    yscale: str = "linear",
    legend: bool = True,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlabel("Ridge/Tikhonov alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", color="0.88", linewidth=0.8)
    if legend:
        ax.legend(loc="best", fontsize=8)


def _figure_captions_markdown() -> str:
    return """# Phase 2 Figure Captions

## fig_phase2_ieee300_residual_vs_alpha

Shows the IEEE300 mean residual norm across alpha for original Ridge, coordinate-preconditioned Ridge, transformed-penalty Ridge, and QSVT diagnostic rows. It supports the observation that coordinate preconditioning is a separate estimator whose residual can increase. The limitation is that these are controlled IEEE/PYPOWER generated measurement rows, not field-calibrated measurements. The claim-safe interpretation is alpha-dependent estimator behavior, not QSVT superiority.

## fig_phase2_ieee300_rmse_vs_alpha

Shows IEEE300 mean RMSE across alpha for Phase 2 variants. It supports separating coordinate-preconditioned behavior from transformed-penalty consistency. The limitation is that RMSE is available only because the benchmark has synthetic/reference states. The claim-safe interpretation is controlled-benchmark diagnostic comparison.

## fig_phase2_ieee300_qsvt_error_vs_alpha

Shows IEEE300 full-interval QSVT approximation diagnostic error across alpha. It supports that preconditioned matrices can reduce approximation difficulty at selected alpha values. The limitation is that this is a scalar/polynomial approximation diagnostic, not matrix-level hardware execution. The claim-safe interpretation is QSVT-compatible approximation evidence.

## fig_phase2_ieee300_residual_rmse_qsvt_tradeoff

Shows IEEE300 residual, RMSE, and QSVT approximation diagnostic error in one tradeoff view. It supports the key Phase 2 result: coordinate-preconditioned Ridge improves the QSVT approximation diagnostic for selected alpha values but may degrade residual/RMSE, while transformed-penalty preconditioning preserves original Ridge metrics and improves the approximation diagnostic. The limitation is that the diagnostic is alpha-dependent and controlled-benchmark-specific. The claim-safe interpretation is transformed-penalty consistency, not performance superiority over Ridge/Tikhonov.

## fig_phase2_ieee118_qsvt_error_vs_alpha

Shows IEEE118 QSVT approximation diagnostic error across alpha. It supports that the preconditioned diagnostic can be substantially smaller than the original diagnostic for selected alpha values. The limitation is that this does not establish an operational alpha rule. The claim-safe interpretation is resource-aware approximation diagnostics.

## fig_phase2_ieee118_residual_vs_alpha

Shows IEEE118 residual norm across alpha. It supports the same separation between estimator quality and approximation difficulty seen in IEEE300. The limitation is that the generated measurement scenarios are controlled. The claim-safe interpretation is variant-specific behavior under controlled perturbations.

## fig_phase2_variant_comparison_ieee300

Shows normalized IEEE300 residual, RMSE, and QSVT diagnostic error at alpha = 1e-2. It supports separate reporting of original Ridge, coordinate-preconditioned Ridge, transformed-penalty Ridge, and preconditioned QSVT diagnostic rows. The limitation is that alpha = 1e-2 is the phase-validated target, not a field-calibrated rule. The claim-safe interpretation is a manuscript table companion.

## fig_phase2_original_vs_preconditioned_kappa

Shows median original and column-equilibrated condition numbers for IEEE118 and IEEE300. It supports the explanation that column equilibration changes approximation difficulty. The limitation is that condition-number improvement alone is not an estimator-quality guarantee. The claim-safe interpretation is preconditioner diagnostics only.

## fig_phase2_alpha_selection_score

Shows the diagnostic joint score over alpha by case and variant. It supports reproducible alpha-selection traceability. The limitation is that the score is controlled-benchmark-specific. The claim-safe interpretation is diagnostic alpha selection, not an operational rule.
"""


def _key_findings_markdown(complete: pd.DataFrame, best_alpha: pd.DataFrame) -> str:
    ieee300 = complete[complete["case_name"].astype(str).str.lower() == "ieee300"]
    coord = _variant_row(ieee300, "coordinate_preconditioned_ridge")
    trans = _variant_row(ieee300, "transformed_penalty_preconditioned_ridge")
    original = _variant_row(ieee300, "original_ridge")
    pre = _variant_row(ieee300, "preconditioned_qsvt_diagnostic")
    coord_residual_ratio = _ratio(coord, original, "residual_norm")
    coord_rmse_ratio = _ratio(coord, original, "rmse_if_available")
    trans_residual_ratio = _ratio(trans, original, "residual_norm")
    pre_error_ratio = _ratio(pre, original, "qsvt_full_interval_error")
    table = _markdown_table(
        complete,
        [
            "case_name",
            "variant_name",
            "alpha",
            "residual_norm",
            "rmse_if_available",
            "qsvt_full_interval_error",
            "status",
        ],
    )
    return f"""# QSVT Phase 2 Complete Key Findings

Phase 2 consolidates IEEE118 and IEEE300 alpha/preconditioned sweeps into a manuscript-ready evidence package.

## Selected-Alpha Summary

{table}

## IEEE300 Interpretation

- Coordinate-preconditioned Ridge is a separate estimator. At the joint-score selected alpha, its residual ratio versus original Ridge is {_format_float(coord_residual_ratio)} and its RMSE ratio is {_format_float(coord_rmse_ratio)}.
- Transformed-penalty preconditioning preserves the original x-space Ridge penalty. At the joint-score selected alpha, its residual ratio versus original Ridge is {_format_float(trans_residual_ratio)}.
- The preconditioned QSVT diagnostic error ratio versus original Ridge is {_format_float(pre_error_ratio)} at the selected alpha, so the approximation diagnostic improves for the preconditioned matrix without turning the coordinate-preconditioned estimator into the original Ridge estimator.
- Alpha gains are alpha-dependent and controlled-benchmark-specific.

## Alpha Selection Rows

Rows in best-alpha table: {len(best_alpha)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _phase2_results_paragraph(summary: pd.DataFrame) -> str:
    ieee300 = summary[summary["case_name"].astype(str).str.lower() == "ieee300"]
    original = _variant_row(ieee300, "original_ridge")
    coord = _variant_row(ieee300, "coordinate_preconditioned_ridge")
    trans = _variant_row(ieee300, "transformed_penalty_preconditioned_ridge")
    pre = _variant_row(ieee300, "preconditioned_qsvt_diagnostic")
    return (
        "In the Phase 2 controlled IEEE/PYPOWER benchmark, the original and "
        "preconditioned variants are reported separately. For IEEE300, the "
        f"coordinate-preconditioned Ridge row has residual {_format_row_value(coord, 'residual_norm')} "
        f"and RMSE {_format_row_value(coord, 'rmse_if_available')}, compared with original "
        f"Ridge residual {_format_row_value(original, 'residual_norm')} and RMSE "
        f"{_format_row_value(original, 'rmse_if_available')}. The transformed-penalty "
        f"row reports residual {_format_row_value(trans, 'residual_norm')} and RMSE "
        f"{_format_row_value(trans, 'rmse_if_available')}, matching the original "
        "Ridge objective within numerical precision while using the preconditioned "
        "matrix for the QSVT approximation diagnostic. The preconditioned diagnostic "
        f"QSVT error is {_format_row_value(pre, 'qsvt_full_interval_error')}, compared "
        f"with {_format_row_value(original, 'qsvt_full_interval_error')} for the original "
        "matrix at the selected alpha. These results support an alpha-dependent, "
        "resource-aware approximation diagnostic and transformed-penalty consistency."
    )


def _phase2_limitations_paragraph() -> str:
    return (
        "Phase 2 remains a controlled benchmark over generated measurement rows. "
        "The preconditioned rows do not imply that the original IEEE300 matrix has "
        "passed the same approximation diagnostic, and coordinate-preconditioned "
        "Ridge is a distinct estimator because the penalty is applied in the "
        "transformed coordinates. The alpha-selection score is diagnostic and "
        "controlled-benchmark-specific. The phase result is scalar full-domain "
        "phase-response validation of the bounded Ridge/Tikhonov target, not "
        "block-encoded matrix execution."
    )


def _phase2_claim_safe_wording() -> str:
    return """# Phase 2 Claim-Safe Wording

## Supported Wording

```text
controlled IEEE/PYPOWER benchmark
generated measurement rows
regularized spectral filtering
QSVT-compatible implementation pathway
scalar phase-response validation
preconditioned estimator variant
transformed-penalty consistency
diagnostic alpha selection
resource-aware approximation diagnostics
claim-support traceability
```

## Avoid Wording

```text
Avoid wording: quantum speedup
Avoid wording: quantum advantage
Avoid wording: QSVT outperforms Ridge/Tikhonov
Avoid wording: hardware execution
Avoid wording: hardware validation
Avoid wording: full block-encoded IEEE-scale QSVT execution
Avoid wording: real PMU/SCADA field-data validation
Avoid wording: field-calibrated operational rule
Avoid wording: preconditioned IEEE300 means original IEEE300 passed
```
"""


def _transformed_penalty_explanation() -> str:
    return r"""# Transformed-Penalty Consistency Explanation

Coordinate preconditioning changes the geometry of the regularization term because the penalty is applied to the transformed variable. We therefore also evaluate a transformed-penalty formulation that preserves the original \(x\)-space Ridge penalty while allowing the QSVT approximation diagnostic to be applied to the preconditioned matrix. This distinction is important because the coordinate-preconditioned estimator may reduce QSVT approximation difficulty while degrading residual or RMSE, whereas the transformed-penalty formulation provides a consistency-preserving diagnostic relative to the original Ridge objective.

Let the original weighted linearized state-estimation system be

\[
A=\tilde H,\qquad b=\tilde r.
\]

Column equilibration uses a nonsingular diagonal scaling matrix \(M\) and the preconditioned matrix

\[
A_p = A M^{-1}.
\]

The original Ridge/Tikhonov estimator solves

\[
\min_x \|Ax-b\|_2^2+\alpha\|x\|_2^2.
\]

The coordinate-preconditioned Ridge estimator solves

\[
\min_y \|A_p y-b\|_2^2+\alpha\|y\|_2^2,
\qquad x=M^{-1}y.
\]

This is a separate estimator because the penalty is applied to \(y\), not to \(x\). The transformed-penalty estimator instead solves

\[
\min_y \|A_p y-b\|_2^2+\alpha\|M^{-1}y\|_2^2,
\qquad x=M^{-1}y.
\]

This transformed-penalty form preserves the original \(x\)-space Ridge penalty while exposing the preconditioned matrix in the data-fit term. Therefore coordinate-preconditioned Ridge and transformed-penalty Ridge must be evaluated separately.
"""


def _phase2_methods_equations() -> str:
    return r"""# Phase 2 Methods Equations

Original weighted system:

\[
A=\tilde H,\qquad b=\tilde r.
\]

Column-equilibrated system:

\[
A_p = A M^{-1}.
\]

Original Ridge:

\[
\min_x \|Ax-b\|_2^2+\alpha\|x\|_2^2.
\]

Coordinate-preconditioned Ridge:

\[
\min_y \|A_p y-b\|_2^2+\alpha\|y\|_2^2,
\qquad x=M^{-1}y.
\]

Transformed-penalty Ridge:

\[
\min_y \|A_p y-b\|_2^2+\alpha\|M^{-1}y\|_2^2,
\qquad x=M^{-1}y.
\]

QSVT-compatible Ridge/Tikhonov target:

\[
P_\alpha(\sigma)=\frac{\sigma}{\sigma^2+\alpha}.
\]

The QSVT-target equivalence to Ridge/Tikhonov is used as a correctness check for the same spectral filter, not as a superiority claim.
"""


def _ieee57_status(status: str, reason: str) -> dict[str, Any]:
    return {
        "case_name": "ieee57",
        "status": status,
        "reason": reason,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _ieee57_result_summary(results: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": len(results),
        "variants": sorted(results["variant_name"].astype(str).unique().tolist()),
        "alphas": sorted(float(value) for value in results["alpha"].unique().tolist()),
        "mean_residual_norm": float(
            pd.to_numeric(results["residual_norm"], errors="coerce").mean()
        ),
        "mean_qsvt_full_interval_error": float(
            pd.to_numeric(results["qsvt_full_interval_approx_error"], errors="coerce").mean()
        ),
    }


def _ieee57_status_markdown(status: dict[str, Any]) -> str:
    return f"""# Optional IEEE57 Phase 2 Extension

Status: {status["status"]}

Reason: {status["reason"]}

Rows: {status.get("row_count", "not_applicable")}

Variants: {", ".join(status.get("variants", []))}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(no rows)"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(subset.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [header, separator]
    for row in subset.itertuples(index=False):
        rows.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(rows)


def _format_cell(value: Any) -> str:
    if isinstance(value, float | np.floating):
        return _format_float(float(value))
    return str(value)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    return f"{value:.6g}"


def _format_row_value(row: pd.Series | None, column: str) -> str:
    if row is None:
        return "not_available"
    return _format_float(_safe_float(row.get(column)))


def _alpha_trace_score(
    alpha_trace: pd.DataFrame,
    case_name: str,
    variant_name: str,
    alpha: float,
) -> float:
    rows = alpha_trace[
        (alpha_trace["case_name"].astype(str).str.lower() == case_name)
        & (alpha_trace["variant_name"].astype(str) == variant_name)
        & np.isclose(alpha_trace["alpha"].astype(float), alpha)
    ]
    if rows.empty:
        return np.nan
    return _safe_float(rows.iloc[0]["joint_score"])


def _phase_status_for_alpha(alpha: float) -> str:
    if np.isclose(float(alpha), PHASE2_PHASE_ALPHA):
        return "passed_scalar_full_domain"
    return "not_validated_for_this_alpha"


def _aggregate_status(status_counts: dict[str, int]) -> str:
    if any("failed" in status for status in status_counts):
        return "failed"
    if "residual_degraded" in status_counts and len(status_counts) > 1:
        return "mixed_ok_residual_degraded"
    if "residual_degraded" in status_counts:
        return "residual_degraded"
    if "consistency_check" in status_counts:
        return "consistency_check"
    if "diagnostic_only" in status_counts:
        return "diagnostic_only"
    return "ok"


def _complete_interpretation(
    variant_name: str,
    *,
    qsvt_error: float,
    original_error: float,
    preconditioned_error: float,
) -> str:
    if variant_name == "coordinate_preconditioned_ridge":
        return (
            "Separate coordinate-penalty estimator; improves the preconditioned "
            "approximation diagnostic only if residual/RMSE remain acceptable."
        )
    if variant_name == "transformed_penalty_preconditioned_ridge":
        return (
            "Preserves the original x-space Ridge penalty while using the "
            f"preconditioned diagnostic error {_format_float(qsvt_error)}."
        )
    if variant_name == "preconditioned_qsvt_diagnostic":
        return (
            "Preconditioned QSVT approximation diagnostic; compare only with the "
            f"original diagnostic error {_format_float(original_error)}."
        )
    if variant_name == "original_qsvt_diagnostic":
        return (
            "Original-matrix QSVT approximation diagnostic; compare with "
            f"preconditioned diagnostic error {_format_float(preconditioned_error)}."
        )
    return "Original Ridge/Tikhonov reference under the same alpha grid."


def _complete_caveat(variant_name: str) -> str:
    if variant_name == "coordinate_preconditioned_ridge":
        return (
            "Coordinate-preconditioned Ridge changes the regularization geometry; "
            "do not use it as a replacement for original Ridge without metric checks."
        )
    if variant_name == "transformed_penalty_preconditioned_ridge":
        return (
            "Transformed-penalty preconditioning preserves the original x-space "
            "penalty; it is evaluated separately from coordinate preconditioning."
        )
    if "qsvt" in variant_name:
        return (
            "QSVT rows are scalar/polynomial approximation diagnostics and resource "
            "proxies, not block-encoded hardware execution."
        )
    return "Original Ridge/Tikhonov baseline; QSVT-target equivalence is a correctness check."


def _reference_qsvt_error(
    group: pd.DataFrame,
    case_name: str,
    alpha: float,
    *,
    original: bool,
) -> float:
    all_results_path = Path(
        "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv"
    )
    variant = "original_qsvt_diagnostic" if original else "preconditioned_qsvt_diagnostic"
    if all_results_path.is_file():
        all_results = pd.read_csv(all_results_path)
        rows = all_results[
            (all_results["case_name"].astype(str).str.lower() == case_name)
            & (all_results["variant_name"].astype(str) == variant)
            & np.isclose(all_results["alpha"].astype(float), alpha)
        ]
        if not rows.empty:
            return _mean_numeric(rows, "qsvt_full_interval_approx_error")
    if group["variant_name"].astype(str).eq(variant).any():
        return _mean_numeric(group, "qsvt_full_interval_approx_error")
    return np.nan


def _variant_comparison_interpretation(
    coordinate: pd.Series | None,
    transformed: pd.Series | None,
    original: pd.Series | None,
    pre_qsvt: pd.Series | None,
) -> str:
    residual_ratio = _ratio(coordinate, original, "residual_norm")
    transformed_ratio = _ratio(transformed, original, "residual_norm")
    error_ratio = _ratio(pre_qsvt, original, "qsvt_full_interval_error")
    return (
        f"coordinate residual ratio={_format_float(residual_ratio)}; "
        f"transformed residual ratio={_format_float(transformed_ratio)}; "
        f"preconditioned diagnostic error ratio={_format_float(error_ratio)}."
    )


def _variant_row(frame: pd.DataFrame, variant_name: str) -> pd.Series | None:
    if frame.empty or "variant_name" not in frame.columns:
        return None
    rows = frame[frame["variant_name"].astype(str) == variant_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _ratio(row: pd.Series | None, baseline: pd.Series | None, column: str) -> float:
    if row is None or baseline is None:
        return np.nan
    return _safe_divide(row.get(column), baseline.get(column))


def _safe_divide(numerator: Any, denominator: Any) -> float:
    top = _safe_float(numerator)
    bottom = _safe_float(denominator)
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0.0:
        return np.nan
    return float(top / bottom)


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _median_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else np.nan


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _marker_for_variant(variant: str) -> str:
    return {
        "original_ridge": "o",
        "coordinate_preconditioned_ridge": "s",
        "transformed_penalty_preconditioned_ridge": "^",
        "original_qsvt_diagnostic": "D",
        "preconditioned_qsvt_diagnostic": "v",
    }.get(variant, "o")


def _label_for_variant(variant: str) -> str:
    return {
        "original_ridge": "Original Ridge",
        "coordinate_preconditioned_ridge": "Coordinate-preconditioned Ridge",
        "transformed_penalty_preconditioned_ridge": "Transformed-penalty Ridge",
        "original_qsvt_diagnostic": "Original QSVT diagnostic",
        "preconditioned_qsvt_diagnostic": "Preconditioned QSVT diagnostic",
    }.get(variant, variant)


def _short_label(variant: str) -> str:
    return {
        "original_ridge": "Original",
        "coordinate_preconditioned_ridge": "Coordinate",
        "transformed_penalty_preconditioned_ridge": "Transformed penalty",
        "preconditioned_qsvt_diagnostic": "Preconditioned diagnostic",
    }.get(variant, variant)


def _complete_summary_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "alpha",
        "m",
        "n",
        "rank",
        "condition_number_original",
        "condition_number_preconditioned_if_applicable",
        "residual_norm",
        "weighted_residual_norm",
        "rmse_if_available",
        "solution_norm",
        "relative_solution_error_vs_original_ridge",
        "relative_solution_error_vs_transformed_penalty",
        "qsvt_full_interval_error",
        "qsvt_actual_singular_value_error",
        "qsvt_degree",
        "qsvt_query_count",
        "phase_validation_status",
        "alpha_selection_score",
        "status",
        "interpretation",
        "caveat",
    ]


def _resolve_complete_summary_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_complete_summary",
        "sweep_results_csv": (
            "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv"
        ),
        "alpha_selection_summary_csv": (
            "outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.csv"
        ),
        "alpha_selection_trace_csv": "outputs/qsvt_phase2_alpha_selection/alpha_selection_trace.csv",
        "optional_ieee57_results_csv": (
            "outputs/qsvt_phase2_optional_ieee57/ieee57_phase2_results.csv"
        ),
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_figures_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_figures",
        "sweep_summary_csv": (
            "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv"
        ),
        "alpha_selection_trace_csv": "outputs/qsvt_phase2_alpha_selection/alpha_selection_trace.csv",
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_ieee57_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_optional_ieee57",
        "case_source": "pypower",
        "base_seed": 123,
        "reuse_existing": True,
        "seeds": [10, 20, 30],
        "alphas": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
        "noise_stds": [0.0, 0.01, 0.03],
        "missing_ratios": [0.0, 0.05, 0.10],
        "bad_data_ratios": [0.0, 0.02],
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_manuscript_text_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_manuscript_text",
        "complete_summary_csv": (
            "outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv"
        ),
    }
    if config:
        resolved.update(config)
    return resolved


def copy_ieee57_results_for_complete_summary(source_dir: Path, destination_dir: Path) -> None:
    source = source_dir / "ieee57_phase2_results.csv"
    if source.is_file():
        ensure_directory(destination_dir)
        shutil.copy2(source, destination_dir / "ieee57_phase2_results.csv")
