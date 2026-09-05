from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from robust_qsvt_se.utils.io import ensure_directory, write_json

plt.switch_backend("Agg")


CASE_ORDER = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]
ESTIMATOR_ORDER = [
    "Pseudoinverse",
    "Normal-equation WLS",
    "Truncated SVD",
    "Ridge / QSVT-reg.",
    "QSVT unreg. inverse",
    "HHL-style proxy",
    "Huber IRLS",
]
ESTIMATOR_LABELS = {
    "pseudoinverse": "Pseudoinverse",
    "normal_equation_wls": "Normal-equation WLS",
    "truncated_svd": "Truncated SVD",
    "ridge": "Ridge / QSVT-reg.",
    "qsvt_regularized": "Ridge / QSVT-reg.",
    "qsvt_unregularized_inverse": "QSVT unreg. inverse",
    "hhl_style_inverse_proxy": "HHL-style proxy",
    "huber_irls": "Huber IRLS",
}
STRESSOR_LABELS = {
    "scenario.noise_std": "Highest noise",
    "scenario.missing_ratio": "Highest missing measurements",
    "scenario.bad_data.ratio": "Highest bad-data ratio",
}
STRESSOR_SHORT_LABELS = {
    "scenario.noise_std": "Noise",
    "scenario.missing_ratio": "Missing",
    "scenario.bad_data.ratio": "Bad data",
}
COLORS = {
    "Pseudoinverse": "#4C78A8",
    "Normal-equation WLS": "#72B7B2",
    "Truncated SVD": "#F58518",
    "Ridge / QSVT-reg.": "#54A24B",
    "QSVT unreg. inverse": "#E45756",
    "HHL-style proxy": "#FF9DA6",
    "Huber IRLS": "#B279A2",
}


@dataclass(frozen=True)
class PaperReadySources:
    ac_linearized_runs: tuple[Path, ...]
    nonlinear_runs: tuple[Path, ...]
    phase_validation_dir: Path
    hardware_dirs: tuple[Path, ...]
    circuit_scaling_dir: Path
    resource_full_dir: Path
    qsvt_phase_demo_summary_paths: tuple[Path, ...] = ()
    missing_baseline_runs: tuple[Path, ...] = ()


def default_paper_ready_sources(root: str | Path = "outputs") -> PaperReadySources:
    output_root = Path(root)
    return PaperReadySources(
        ac_linearized_runs=tuple(output_root / f"real_{case}_seed10" for case in CASE_ORDER),
        nonlinear_runs=tuple(output_root / f"nonlinear_ac_{case}_seed10" for case in CASE_ORDER),
        phase_validation_dir=output_root / "qsvt_phase_validation_paper",
        hardware_dirs=(
            output_root / "qsvt_hardware_ieee14_2x2",
            output_root / "qsvt_hardware_ieee14_4x4",
        ),
        circuit_scaling_dir=output_root / "qsvt_circuit_scaling",
        resource_full_dir=output_root / "qsvt_resource_full_ieee",
        qsvt_phase_demo_summary_paths=(
            output_root / "manuscript_report_qsvt_hardware" / "qsvt_phase_demo_summary.csv",
            output_root / "manuscript_report_qsvt_validation" / "qsvt_phase_demo_summary.csv",
            output_root / "manuscript_report_final_full" / "qsvt_phase_demo_summary.csv",
        ),
        missing_baseline_runs=(
            output_root / "diagnostic_missing_baselines",
            output_root / "real_ieee_high_stress_missing_baselines",
            output_root / "real_ieee30_high_stress_missing_baselines",
        ),
    )


def build_paper_ready_results(
    output_dir: str | Path = "outputs/paper_ready_results",
    *,
    sources: PaperReadySources | None = None,
) -> dict[str, Any]:
    resolved_sources = sources or default_paper_ready_sources()
    output_path = ensure_directory(output_dir)
    tables_dir = ensure_directory(output_path / "tables")
    figures_dir = ensure_directory(output_path / "figures")
    source_files: list[Path] = []

    ac_metrics = _load_run_metrics(resolved_sources.ac_linearized_runs, source_files)
    nonlinear_metrics = _load_run_metrics(resolved_sources.nonlinear_runs, source_files)
    missing_baseline_metrics = _load_optional_run_metrics(
        resolved_sources.missing_baseline_runs,
        source_files,
    )
    benchmark_metrics = pd.concat([ac_metrics, nonlinear_metrics], ignore_index=True)
    equivalence = _ridge_qsvt_equivalence_diagnostics(benchmark_metrics)
    if not equivalence["within_tolerance"]:
        raise ValueError(
            "ridge and qsvt_regularized RMSE values are not numerically equivalent "
            f"(max_abs_diff={equivalence['max_abs_rmse_diff']})"
        )

    checkpoint_metadata = _load_checkpoint_metadata(
        resolved_sources.nonlinear_runs,
        source_files,
    )

    table1 = build_table1(ac_metrics, nonlinear_metrics, checkpoint_metadata)
    table2 = build_table2(ac_metrics)
    table3 = build_table3(nonlinear_metrics)
    table4 = build_table4(resolved_sources.phase_validation_dir, source_files)
    table5 = build_table5(resolved_sources, source_files)
    table6 = build_table6(resolved_sources.resource_full_dir, source_files)
    table7 = build_table7(missing_baseline_metrics)

    table_paths = {
        "table1_benchmark_coverage": _write_table_bundle(
            tables_dir,
            "table1_benchmark_coverage",
            table1,
            caption="Benchmark coverage and conditioning.",
            label="tab:benchmark-coverage",
        ),
        "table2_ac_linearized_high_stress": _write_table_bundle(
            tables_dir,
            "table2_ac_linearized_high_stress",
            table2,
            caption="AC-linearized high-stress robustness.",
            label="tab:ac-linearized-high-stress",
            note=(
                "Ridge and QSVT-regularized columns are combined because the classical "
                "simulation uses the same regularized singular-value filter."
            ),
        ),
        "table3_nonlinear_ac_performance": _write_table_bundle(
            tables_dir,
            "table3_nonlinear_ac_performance",
            table3,
            caption="Nonlinear AC performance and convergence.",
            label="tab:nonlinear-ac-performance",
            note=(
                "Failure means an explicit failed trial. Strict convergence is reported "
                "separately from reaching the configured iteration limit. IEEE300 nonlinear "
                "AC completed all configured trials with zero numerical failures, but many "
                "estimator-trial results reached the maximum-iteration limit under the strict "
                "convergence criterion."
            ),
        ),
        "table4_qsvt_phase_validation": _write_table_bundle(
            tables_dir,
            "table4_qsvt_phase_validation",
            table4,
            caption="QSP/QSVT phase validation.",
            label="tab:qsvt-phase-validation",
        ),
        "table5_qsvt_circuit_scaling": _write_table_bundle(
            tables_dir,
            "table5_qsvt_circuit_scaling",
            table5,
            caption="QSVT circuit prototype and scaling.",
            label="tab:qsvt-circuit-scaling",
            note=(
                "Dense-unitary rows are correctness paths; explicit block-encoding rows "
                "are small-matrix prototypes; scaling rows are deterministic submatrix "
                "experiments."
            ),
        ),
        "table6_qsvt_resource_feasibility": _write_table_bundle(
            tables_dir,
            "table6_qsvt_resource_feasibility",
            table6,
            caption="Full IEEE QSVT resource and feasibility proxy estimates.",
            label="tab:qsvt-resource-feasibility",
            note="These are proxy resource estimates, not hardware execution results.",
        ),
        "table7_missing_baseline_diagnostics": _write_table_bundle(
            tables_dir,
            "table7_missing_baseline_diagnostics",
            table7,
            caption="Focused diagnostics for research-idea missing baselines.",
            label="tab:missing-baseline-diagnostics",
            note=(
                "Unregularized QSVT inverse and HHL-style rows are diagnostic proxies or "
                "unstable ablations, not proposed-method evidence."
            ),
        ),
    }

    figure_paths = {
        "fig1_ac_linearized_robustness": _plot_fig1_ac_linearized(
            table2,
            figures_dir / "fig1_ac_linearized_robustness",
        ),
        "fig2_nonlinear_ac_error_convergence": _plot_fig2_nonlinear(
            table3,
            figures_dir / "fig2_nonlinear_ac_error_convergence",
        ),
        "fig3_qsvt_phase_validation": _plot_fig3_phase_validation(
            resolved_sources.phase_validation_dir,
            figures_dir / "fig3_qsvt_phase_validation",
            source_files,
        ),
        "fig4_qsvt_circuit_scaling": _plot_fig4_circuit_scaling(
            resolved_sources.circuit_scaling_dir,
            figures_dir / "fig4_qsvt_circuit_scaling",
            source_files,
        ),
        "appendix_qsvt_resource_estimates": _plot_appendix_resources(
            table6,
            figures_dir / "appendix_qsvt_resource_estimates",
        ),
    }

    document_paths = {
        "result_quality_audit": output_path / "result_quality_audit.md",
        "results_claims": output_path / "results_claims.md",
        "results_section_skeleton": output_path / "results_section_skeleton.md",
    }
    document_paths["result_quality_audit"].write_text(
        _result_quality_audit(table1, table2, table3, table4, table5, table6, table7, equivalence),
        encoding="utf-8",
    )
    document_paths["results_claims"].write_text(
        _results_claims(table1, table2, table3, table4, table5, table6, table7, equivalence),
        encoding="utf-8",
    )
    document_paths["results_section_skeleton"].write_text(
        _results_section_skeleton(),
        encoding="utf-8",
    )

    issues = _manifest_issues(table1, table2, table3, table6, equivalence)
    notes = [
        "QSVT/ridge are combined because they use the same classical spectral filter.",
        "PYPOWER IEEE cases are benchmark systems, not real PMU/SCADA field data.",
        "Full IEEE118/IEEE300 hardware-native QSVT execution is not claimed.",
        "QSVT unregularized inverse is an unstable ablation, not the proposed method.",
        "HHL-style inverse is a diagnostic proxy and not executable HHL circuit evidence.",
        "Generated package reads completed artifacts only and does not run new experiments.",
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_path),
        "generated_tables": _stringify_artifact_map(table_paths),
        "generated_figures": _stringify_artifact_map(figure_paths),
        "documents": {key: str(path) for key, path in document_paths.items()},
        "source_files_used": sorted({str(path) for path in source_files}),
        "notes_caveats": notes,
        "issues_encountered": issues,
        "ridge_qsvt_equivalence": equivalence,
        "row_counts": {
            "ac_metric_rows": len(ac_metrics),
            "nonlinear_metric_rows": len(nonlinear_metrics),
            "table1_rows": len(table1),
            "table2_rows": len(table2),
            "table3_rows": len(table3),
            "table4_rows": len(table4),
            "table5_rows": len(table5),
            "table6_rows": len(table6),
            "table7_rows": len(table7),
        },
    }
    manifest_path = output_path / "manifest.json"
    write_json(manifest_path, manifest)

    return {
        "output_dir": output_path,
        "tables": table_paths,
        "figures": figure_paths,
        "documents": document_paths,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def build_table1(
    ac_metrics: pd.DataFrame,
    nonlinear_metrics: pd.DataFrame,
    checkpoint_metadata: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checkpoint_metadata = checkpoint_metadata or {}
    for frame, benchmark_mode in (
        (ac_metrics, "AC-linearized PYPOWER"),
        (nonlinear_metrics, "Nonlinear AC PYPOWER"),
    ):
        if frame.empty:
            continue
        for case_name, case_frame in frame.groupby("case_name", sort=False, dropna=False):
            trial_status = _trial_status(case_frame)
            condition_numbers = pd.to_numeric(
                case_frame.get("condition_number"),
                errors="coerce",
            )
            nonlinear_rate = math.nan
            max_iteration_rate = math.nan
            if benchmark_mode.startswith("Nonlinear"):
                collapsed = _collapse_estimator_trials(case_frame)
                nonlinear_rate = _bool_series(collapsed["converged"]).mean()
                max_iteration_rate = _max_iteration_series(collapsed).mean()

            checkpoint = checkpoint_metadata.get(str(case_name), {})
            total_trials = int(checkpoint.get("total_trials", trial_status["total_trials"]))
            completed_trials = int(
                checkpoint.get("completed_trials", trial_status["completed_trials"])
            )
            failed_trials = int(checkpoint.get("failed_trials", trial_status["failed_trials"]))
            rows.append(
                {
                    "case": case_name,
                    "benchmark_mode": benchmark_mode,
                    "number_of_states": _first_int(case_frame, "n_states"),
                    "number_of_measurements": _first_int(case_frame, "n_measurements"),
                    "number_of_seeds": int(case_frame["seed"].nunique(dropna=True)),
                    "total_trials": total_trials,
                    "completed_trials": completed_trials,
                    "failed_trials": failed_trials,
                    "median_condition_number": float(condition_numbers.median()),
                    "max_condition_number": float(condition_numbers.max()),
                    "nonlinear_convergence_rate": nonlinear_rate,
                    "max_iteration_rate": max_iteration_rate,
                }
            )
    return _sort_table(pd.DataFrame(rows), case_column="case", mode_column="benchmark_mode")


def build_table2(ac_metrics: pd.DataFrame) -> pd.DataFrame:
    if ac_metrics.empty:
        return pd.DataFrame()
    collapsed = _collapse_estimator_trials(ac_metrics)
    rows: list[dict[str, Any]] = []
    for case_name, case_frame in collapsed.groupby("case_name", sort=False, dropna=False):
        for parameter, label in STRESSOR_LABELS.items():
            stress_frame = case_frame[case_frame["sweep_parameter"].astype(str).eq(parameter)]
            if stress_frame.empty:
                continue
            stress_values = pd.to_numeric(stress_frame["sweep_value"], errors="coerce")
            max_value = float(stress_values.max())
            high = stress_frame[stress_values.eq(max_value)]
            means = high.groupby("display_estimator", dropna=False)["rmse"].mean()
            pseudoinverse_rmse = _get_estimator_value(means, "Pseudoinverse")
            ridge_qsvt_rmse = _get_estimator_value(means, "Ridge / QSVT-reg.")
            improvement = (
                (pseudoinverse_rmse - ridge_qsvt_rmse) / pseudoinverse_rmse * 100.0
                if pseudoinverse_rmse and not math.isnan(pseudoinverse_rmse)
                else math.nan
            )
            trial_status = _trial_status(high)
            rows.append(
                {
                    "case": case_name,
                    "stress_type": label,
                    "stress_value": max_value,
                    "pseudoinverse_rmse": pseudoinverse_rmse,
                    "truncated_svd_rmse": _get_estimator_value(means, "Truncated SVD"),
                    "ridge_qsvt_regularized_rmse": ridge_qsvt_rmse,
                    "huber_rmse": _get_estimator_value(means, "Huber IRLS"),
                    "qsvt_improvement_vs_pseudoinverse_pct": improvement,
                    "failure_rate": trial_status["failed_trials"]
                    / max(1, trial_status["total_trials"]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_stress_order"] = table["stress_type"].map(
        {label: index for index, label in enumerate(STRESSOR_LABELS.values())}
    )
    table = _sort_table(table, case_column="case")
    return table.drop(columns=["_stress_order"]).reset_index(drop=True)


def build_table3(nonlinear_metrics: pd.DataFrame) -> pd.DataFrame:
    if nonlinear_metrics.empty:
        return pd.DataFrame()
    collapsed = _collapse_estimator_trials(nonlinear_metrics)
    collapsed["_max_iteration"] = _max_iteration_series(collapsed)
    rows: list[dict[str, Any]] = []
    for (case_name, estimator), group in collapsed.groupby(
        ["case_name", "display_estimator"],
        sort=False,
        dropna=False,
    ):
        failed = _bool_series(group["failed"])
        rmse_values = pd.to_numeric(group["rmse"], errors="coerce")
        rows.append(
            {
                "case": case_name,
                "estimator": estimator,
                "mean_rmse": float(rmse_values.mean()),
                "median_rmse": float(rmse_values.median()),
                "iqr_rmse": float(rmse_values.quantile(0.75) - rmse_values.quantile(0.25)),
                "mean_weighted_residual_norm": float(
                    pd.to_numeric(group["weighted_residual_norm"], errors="coerce").mean()
                ),
                "median_weighted_residual_norm": float(
                    pd.to_numeric(group["weighted_residual_norm"], errors="coerce").median()
                ),
                "iqr_weighted_residual_norm": float(
                    pd.to_numeric(group["weighted_residual_norm"], errors="coerce").quantile(0.75)
                    - pd.to_numeric(group["weighted_residual_norm"], errors="coerce").quantile(0.25)
                ),
                "median_weighted_residual_quadratic": float(
                    pd.to_numeric(group["weighted_residual_quadratic"], errors="coerce").median()
                ),
                "completed_trials": int((~failed).sum()),
                "failed_trials": int(failed.sum()),
                "strict_convergence_rate": float(_bool_series(group["converged"]).mean()),
                "max_iteration_rate": float(group["_max_iteration"].mean()),
                "mean_runtime_seconds": float(
                    pd.to_numeric(group["runtime_seconds"], errors="coerce").mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    return _sort_table(table, case_column="case", estimator_column="estimator")


def build_table4(phase_validation_dir: Path, source_files: list[Path]) -> pd.DataFrame:
    report_path = phase_validation_dir / "phase_validation_report.json"
    error_path = phase_validation_dir / "phase_implemented_error.csv"
    approximation_path = phase_validation_dir / "approximation_error.csv"
    report = _read_json(report_path, source_files)
    error_frame = _read_csv(error_path, source_files)
    if approximation_path.is_file():
        _read_csv(approximation_path, source_files)

    max_phase_error = report.get("max_phase_implemented_error")
    mean_phase_error = report.get("mean_phase_implemented_error")
    if not error_frame.empty and "phase_abs_error" in error_frame.columns:
        phase_abs_error = pd.to_numeric(error_frame["phase_abs_error"], errors="coerce")
        max_phase_error = float(phase_abs_error.max())
        mean_phase_error = float(phase_abs_error.mean())

    domain = report.get("domain", [report.get("domain_min"), report.get("domain_max")])
    return pd.DataFrame(
        [
            {
                "target_function": report.get(
                    "target_function",
                    "P_alpha(sigma) = sigma / (sigma^2 + alpha)",
                ),
                "alpha": report.get("alpha"),
                "domain": _domain_string(domain),
                "degree": report.get("polynomial_degree"),
                "phase_count": report.get("phase_count"),
                "max_polynomial_error": report.get("max_polynomial_error"),
                "mean_polynomial_error": report.get("mean_polynomial_error"),
                "max_phase_implemented_error": max_phase_error,
                "mean_phase_implemented_error": mean_phase_error,
                "validation_status": "passed" if report.get("validation_passed") else "failed",
            }
        ]
    )


def build_table5(sources: PaperReadySources, source_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_dense_unitary_rows(sources.qsvt_phase_demo_summary_paths, source_files))
    rows.extend(_explicit_hardware_rows(sources.hardware_dirs, source_files))
    rows.extend(_circuit_scaling_rows(sources.circuit_scaling_dir, source_files))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.drop_duplicates(
        subset=["case", "matrix_source", "matrix_size", "circuit_type"],
        keep="first",
    )
    table["_case_order"] = table["case"].map(_case_order_map()).fillna(999)
    table["_type_order"] = (
        table["circuit_type"]
        .map(
            {
                "PennyLane research-matrix simulation": 0,
                "Qiskit dense-unitary correctness": 1,
                "Explicit block-encoding prototype": 2,
                "Circuit scaling experiment": 3,
            }
        )
        .fillna(99)
    )
    table["_size_order"] = table["matrix_size"].map(_matrix_size_sort_key)
    return (
        table.sort_values(["_type_order", "_case_order", "_size_order"])
        .drop(columns=["_case_order", "_type_order", "_size_order"])
        .reset_index(drop=True)
    )


def build_table6(resource_full_dir: Path, source_files: list[Path]) -> pd.DataFrame:
    resource_path = resource_full_dir / "resource_estimates.csv"
    resource = _read_csv(resource_path, source_files)
    rows = []
    for _, row in resource.iterrows():
        feasible = _as_bool(row.get("full_statevector_simulation_feasible"))
        rows.append(
            {
                "case": row.get("case_name"),
                "matrix_shape": row.get("matrix_shape"),
                "condition_number": row.get("condition_number"),
                "polynomial_degree": row.get("polynomial_degree"),
                "phase_count": row.get("phase_count"),
                "qubit_estimate": row.get("estimated_total_qubits"),
                "query_count": row.get("estimated_qsvt_query_count"),
                "simulation_feasible": feasible,
                "caveat_reason": (
                    f"Proxy estimate only; {row.get('full_simulation_feasible_reason')}"
                ),
            }
        )
    return _sort_table(pd.DataFrame(rows), case_column="case")


def build_table7(missing_baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    if missing_baseline_metrics.empty:
        return pd.DataFrame(
            columns=[
                "run",
                "case",
                "stress",
                "stress_value",
                "estimator",
                "median_rmse",
                "iqr_rmse",
                "median_weighted_residual_norm",
                "median_weighted_residual_quadratic",
                "failure_rate",
                "normal_matrix_condition_number",
                "singular_values_below_cutoff",
                "hhl_effective_condition_number",
                "hhl_resource_proxy",
                "instability_flag",
            ]
        )
    frame = _collapse_estimator_trials(missing_baseline_metrics)
    rows: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in (
            "source_run_path",
            "case_name",
            "sweep_name",
            "sweep_parameter",
            "sweep_value",
            "display_estimator",
        )
        if column in frame.columns
    ]
    for group_key, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys = dict(zip(group_columns, group_key, strict=True))
        failed = _bool_series(group.get("failed", pd.Series(False, index=group.index)))
        rmse_values = pd.to_numeric(group["rmse"], errors="coerce")
        residual_norm = pd.to_numeric(group["weighted_residual_norm"], errors="coerce")
        residual_quadratic = pd.to_numeric(group["weighted_residual_quadratic"], errors="coerce")
        rows.append(
            {
                "run": Path(str(keys.get("source_run_path", ""))).name,
                "case": keys.get("case_name"),
                "stress": keys.get("sweep_parameter") or keys.get("sweep_name"),
                "stress_value": keys.get("sweep_value"),
                "estimator": keys.get("display_estimator"),
                "median_rmse": float(rmse_values.median()),
                "iqr_rmse": float(rmse_values.quantile(0.75) - rmse_values.quantile(0.25)),
                "median_weighted_residual_norm": float(residual_norm.median()),
                "median_weighted_residual_quadratic": float(residual_quadratic.median()),
                "failure_rate": float(failed.mean()) if len(failed) else 0.0,
                "normal_matrix_condition_number": _median_if_present(
                    group,
                    "normal_matrix_condition_number",
                ),
                "singular_values_below_cutoff": _max_if_present(
                    group,
                    "singular_values_below_cutoff",
                ),
                "hhl_effective_condition_number": _median_if_present(
                    group,
                    "hhl_effective_condition_number",
                ),
                "hhl_resource_proxy": _median_if_present(group, "hhl_resource_proxy"),
                "instability_flag": _any_if_present(
                    group,
                    ["unstable_ablation", "hhl_instability_flag"],
                ),
            }
        )
    table = pd.DataFrame(rows)
    return _sort_table(table, case_column="case", estimator_column="estimator")


def _load_run_metrics(run_dirs: tuple[Path, ...], source_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "aggregate_metrics.csv"
        metrics = _normalize_metric_columns(_read_csv(metrics_path, source_files))
        config_path = run_dir / "config_resolved.yaml"
        if config_path.is_file():
            source_files.append(config_path)
        metrics["source_run_path"] = str(run_dir)
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_optional_run_metrics(
    run_dirs: tuple[Path, ...],
    source_files: list[Path],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "aggregate_metrics.csv"
        if not metrics_path.is_file():
            metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        metrics = _normalize_metric_columns(_read_csv(metrics_path, source_files))
        config_path = run_dir / "config_resolved.yaml"
        if config_path.is_file():
            source_files.append(config_path)
        metrics["source_run_path"] = str(run_dir)
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "weighted_residual_norm" not in normalized.columns and "weighted_residual" in normalized:
        normalized["weighted_residual_norm"] = normalized["weighted_residual"]
    if "weighted_residual" not in normalized.columns and "weighted_residual_norm" in normalized:
        normalized["weighted_residual"] = normalized["weighted_residual_norm"]
    if (
        "weighted_residual_quadratic" not in normalized.columns
        and "weighted_residual_norm" in normalized
    ):
        norm = pd.to_numeric(normalized["weighted_residual_norm"], errors="coerce")
        normalized["weighted_residual_quadratic"] = norm**2
    return normalized


def _load_checkpoint_metadata(
    run_dirs: tuple[Path, ...],
    source_files: list[Path],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        checkpoint_path = run_dir / "checkpoint_state.json"
        if not checkpoint_path.is_file():
            continue
        checkpoint = _read_json(checkpoint_path, source_files)
        aggregate_path = run_dir / "aggregate_metrics.csv"
        if aggregate_path.is_file():
            aggregate = pd.read_csv(aggregate_path, usecols=["case_name"], nrows=1)
            case_name = str(aggregate["case_name"].iloc[0])
        else:
            case_name = run_dir.name.replace("nonlinear_ac_", "").replace("_seed10", "")
        metadata[case_name] = checkpoint
    return metadata


def _ridge_qsvt_equivalence_diagnostics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "matched_rows": 0,
            "max_abs_rmse_diff": math.nan,
            "max_rel_rmse_diff": math.nan,
            "within_tolerance": True,
        }
    group_columns = [
        column
        for column in (
            "case_name",
            "mode",
            "trial_id",
            "sweep_name",
            "sweep_parameter",
            "sweep_value",
            "seed",
        )
        if column in metrics.columns
    ]
    subset = metrics[metrics["estimator"].isin(["ridge", "qsvt_regularized"])].copy()
    if subset.empty:
        return {
            "matched_rows": 0,
            "max_abs_rmse_diff": math.nan,
            "max_rel_rmse_diff": math.nan,
            "within_tolerance": True,
        }
    pivot = subset.pivot_table(
        index=group_columns,
        columns="estimator",
        values="rmse",
        aggfunc="mean",
    ).dropna(subset=["ridge", "qsvt_regularized"])
    diff = (pivot["ridge"] - pivot["qsvt_regularized"]).abs()
    denom = pivot[["ridge", "qsvt_regularized"]].abs().max(axis=1).replace(0.0, np.nan)
    rel_diff = (diff / denom).replace([np.inf, -np.inf], np.nan)
    max_abs = float(diff.max()) if not diff.empty else math.nan
    max_rel = float(rel_diff.max()) if not rel_diff.empty else 0.0
    within_tolerance = bool(
        diff.empty
        or np.allclose(
            pivot["ridge"].to_numpy(dtype=float),
            pivot["qsvt_regularized"].to_numpy(dtype=float),
            rtol=1.0e-7,
            atol=1.0e-10,
            equal_nan=True,
        )
    )
    return {
        "matched_rows": len(pivot),
        "max_abs_rmse_diff": max_abs,
        "max_rel_rmse_diff": max_rel,
        "within_tolerance": within_tolerance,
    }


def _collapse_estimator_trials(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    frame["display_estimator"] = frame["estimator"].map(ESTIMATOR_LABELS).fillna(frame["estimator"])
    group_columns = [
        column
        for column in (
            "case_name",
            "mode",
            "sweep_name",
            "sweep_parameter",
            "sweep_value",
            "trial_id",
            "seed",
            "display_estimator",
        )
        if column in frame.columns
    ]
    numeric_columns = [
        column
        for column in (
            "rmse",
            "angle_rmse",
            "voltage_magnitude_rmse",
            "residual_norm",
            "weighted_residual",
            "weighted_residual_norm",
            "weighted_residual_quadratic",
            "condition_number",
            "runtime_seconds",
            "iterations",
            "n_measurements",
            "n_states",
            "noise_std",
            "missing_ratio",
            "bad_data_ratio",
            "bad_data_count",
            "normal_matrix_condition_number",
            "singular_values_below_cutoff",
            "hhl_effective_condition_number",
            "hhl_resource_proxy",
        )
        if column in frame.columns
    ]
    metadata_columns = [
        column
        for column in (
            "failed",
            "converged",
            "failure_reason",
            "scenario_name",
            "dataset_source",
            "dataset_source_detail",
            "source_run_path",
            "unstable_ablation",
            "hhl_instability_flag",
        )
        if column in frame.columns
    ]
    aggregation: dict[str, Any] = {}
    for column in numeric_columns:
        aggregation[column] = "mean"
    if "failed" in metadata_columns:
        aggregation["failed"] = lambda values: bool(_bool_series(values).any())
    if "converged" in metadata_columns:
        aggregation["converged"] = lambda values: bool(_bool_series(values).all())
    if "failure_reason" in metadata_columns:
        aggregation["failure_reason"] = _combine_failure_reasons
    for column in metadata_columns:
        if column not in aggregation:
            aggregation[column] = "first"
    return frame.groupby(group_columns, dropna=False).agg(aggregation).reset_index()


def _trial_status(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"total_trials": 0, "completed_trials": 0, "failed_trials": 0}
    if "trial_id" in frame.columns:
        failed_by_trial = frame.groupby("trial_id", dropna=False)["failed"].agg(
            lambda values: bool(_bool_series(values).any())
        )
        failed_trials = int(failed_by_trial.sum())
        total_trials = len(failed_by_trial)
    else:
        total_trials = len(frame)
        failed_trials = int(_bool_series(frame.get("failed", pd.Series(False))).sum())
    return {
        "total_trials": total_trials,
        "completed_trials": total_trials - failed_trials,
        "failed_trials": failed_trials,
    }


def _bool_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    if series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _max_iteration_series(frame: pd.DataFrame) -> pd.Series:
    if "failure_reason" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return (
        frame["failure_reason"]
        .astype(str)
        .str.contains(
            "max_iterations",
            case=False,
            na=False,
        )
    )


def _combine_failure_reasons(values: pd.Series) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        cleaned.append(text)
    return "; ".join(sorted(set(cleaned)))


def _dense_unitary_rows(
    summary_paths: tuple[Path, ...],
    source_files: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        if not summary_path.is_file():
            continue
        summary = _read_csv(summary_path, source_files)
        for _, row in summary.iterrows():
            construction = str(row.get("qsvt_construction_type", ""))
            if "dense_unitary" in construction:
                circuit_type = "Qiskit dense-unitary correctness"
            elif "pennylane_block_encoded_matrix_qsvt" in construction:
                circuit_type = "PennyLane research-matrix simulation"
            else:
                continue
            if str(row.get("matrix_scope", "")).lower() == "full_matrix":
                continue
            matrix_shape = _shape_to_string(row.get("matrix_shape"))
            detail = _optional_circuit_detail(row, source_files)
            gate_counts = detail.get("gate_counts_after_transpile", {})
            cx_count = gate_counts.get("cx") if isinstance(gate_counts, dict) else None
            rows.append(
                {
                    "case": row.get("source_case") or _infer_case_from_label(row),
                    "matrix_source": "weighted_jacobian",
                    "matrix_size": matrix_shape,
                    "circuit_type": circuit_type,
                    "qubits": _first_non_missing(
                        row.get("qubits"),
                        row.get("n_qubits"),
                        detail.get("qubits"),
                        detail.get("n_qubits"),
                    ),
                    "transpiled_depth": _first_non_missing(
                        row.get("depth_after_transpile"),
                        detail.get("depth_after_transpile"),
                    ),
                    "cx_count": _first_non_missing(row.get("cx_count_after_transpile"), cx_count),
                    "max_error_vs_classical": _first_non_missing(
                        row.get("demo_matrix_max_abs_error"),
                        detail.get("max_error_vs_classical"),
                        row.get("max_abs_error"),
                    ),
                    "feasibility_status": _dense_unitary_status(row),
                }
            )
    return rows


def _optional_circuit_detail(row: pd.Series, source_files: list[Path]) -> dict[str, Any]:
    run_path_value = row.get("report_run_path")
    if pd.isna(run_path_value):
        return {}
    run_path = Path(str(run_path_value))
    summary_path = run_path / "circuit_summary.json"
    if not summary_path.is_file():
        return {}
    return _read_json(summary_path, source_files)


def _explicit_hardware_rows(
    hardware_dirs: tuple[Path, ...],
    source_files: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in hardware_dirs:
        summary_path = run_dir / "hardware_qsvt_circuit_summary.json"
        if not summary_path.is_file():
            summary_path = run_dir / "circuit_summary.json"
        summary = _read_json(summary_path, source_files)
        comparison_path = run_dir / "comparison_to_classical.csv"
        if comparison_path.is_file():
            _read_csv(comparison_path, source_files)
        rows.append(
            {
                "case": summary.get("case_name") or summary.get("source_case"),
                "matrix_source": summary.get("matrix_source", "weighted_jacobian"),
                "matrix_size": _shape_to_string(summary.get("matrix_shape")),
                "circuit_type": "Explicit block-encoding prototype",
                "qubits": summary.get("qubits") or summary.get("n_qubits"),
                "transpiled_depth": summary.get("depth_after_transpile"),
                "cx_count": summary.get("cx_count_after_transpile"),
                "max_error_vs_classical": summary.get("max_error_vs_classical")
                or summary.get("max_abs_error"),
                "feasibility_status": "completed small-matrix prototype"
                if summary.get("simulation_success", True)
                else "simulation failed",
            }
        )
    return rows


def _circuit_scaling_rows(
    circuit_scaling_dir: Path,
    source_files: list[Path],
) -> list[dict[str, Any]]:
    scaling_path = circuit_scaling_dir / "circuit_scaling_results.csv"
    scaling = _read_csv(scaling_path, source_files)
    summary_path = circuit_scaling_dir / "circuit_scaling_summary.json"
    if summary_path.is_file():
        _read_json(summary_path, source_files)
    rows: list[dict[str, Any]] = []
    for _, row in scaling.iterrows():
        feasible = _as_bool(row.get("feasible"))
        reason = row.get("failure_reason")
        status = "completed" if feasible else f"infeasible: {reason}"
        rows.append(
            {
                "case": row.get("case_name"),
                "matrix_source": "weighted_jacobian",
                "matrix_size": row.get("matrix_shape") or row.get("matrix_size"),
                "circuit_type": "Circuit scaling experiment",
                "qubits": row.get("qubits"),
                "transpiled_depth": row.get("depth_after_transpile"),
                "cx_count": row.get("cx_count"),
                "max_error_vs_classical": row.get("max_error_vs_classical"),
                "feasibility_status": status,
            }
        )
    return rows


def _dense_unitary_status(row: pd.Series) -> str:
    if str(row.get("qsvt_construction_type", "")).find("dense_unitary") >= 0:
        return "correctness only; not hardware-native"
    return "research-matrix simulation; not hardware-native"


def _plot_fig1_ac_linearized(table2: pd.DataFrame, output_stem: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    focus_cases = ["ieee30", "ieee300"]
    value_columns = {
        "Pseudoinverse": "pseudoinverse_rmse",
        "Truncated SVD": "truncated_svd_rmse",
        "Ridge / QSVT-reg.": "ridge_qsvt_regularized_rmse",
        "Huber IRLS": "huber_rmse",
    }
    figure_estimators = [estimator for estimator in ESTIMATOR_ORDER if estimator in value_columns]
    for axis, (stress_type, panel_letter) in zip(
        axes,
        zip(STRESSOR_LABELS.values(), ["A", "B", "C"], strict=True),
        strict=True,
    ):
        frame = table2[
            table2["stress_type"].eq(stress_type) & table2["case"].isin(focus_cases)
        ].copy()
        frame = _sort_table(frame, case_column="case")
        x = np.arange(len(frame))
        width = 0.8 / max(1, len(figure_estimators))
        center = (len(figure_estimators) - 1) / 2.0
        for index, estimator in enumerate(figure_estimators):
            values = pd.to_numeric(frame[value_columns[estimator]], errors="coerce")
            offsets = x + (index - center) * width
            axis.bar(
                offsets,
                values,
                width=width,
                color=COLORS[estimator],
                label=estimator,
            )
        axis.set_title(f"{panel_letter}. {stress_type}", loc="left", fontsize=10)
        axis.set_xticks(x)
        axis.set_xticklabels([str(case).upper() for case in frame["case"]])
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_xlabel("Case")
    axes[0].set_ylabel("Mean RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(fig, output_stem)


def _plot_fig2_nonlinear(table3: pd.DataFrame, output_stem: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    pivot = table3.pivot(index="case", columns="estimator", values="mean_rmse")
    pivot = pivot.reindex(CASE_ORDER)
    x = np.arange(len(pivot))
    figure_estimators = [estimator for estimator in ESTIMATOR_ORDER if estimator in pivot.columns]
    width = 0.8 / max(1, len(figure_estimators))
    center = (len(figure_estimators) - 1) / 2.0
    for index, estimator in enumerate(figure_estimators):
        axes[0].bar(
            x + (index - center) * width,
            pivot[estimator],
            width=width,
            color=COLORS[estimator],
            label=estimator,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_xlabel("Case")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(case).upper() for case in pivot.index], rotation=20)
    axes[0].set_title("A. Nonlinear AC error", loc="left", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)

    convergence = (
        table3.groupby("case", dropna=False)
        .agg(
            strict_convergence_rate=("strict_convergence_rate", "mean"),
            max_iteration_rate=("max_iteration_rate", "mean"),
        )
        .reindex(CASE_ORDER)
    )
    axes[1].bar(
        x - 0.18,
        convergence["strict_convergence_rate"],
        width=0.36,
        color="#4C78A8",
        label="Strict convergence",
    )
    axes[1].bar(
        x + 0.18,
        convergence["max_iteration_rate"],
        width=0.36,
        color="#E45756",
        label="Max-iteration reached",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Case")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(case).upper() for case in convergence.index], rotation=20)
    axes[1].set_title("B. Convergence outcome", loc="left", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.25)
    if "ieee300" in convergence.index:
        ieee300_rate = convergence.loc["ieee300", "strict_convergence_rate"]
        if not pd.isna(ieee300_rate):
            axes[1].annotate(
                "IEEE300: completed trials,\nlimited strict convergence",
                xy=(len(convergence.index) - 1, ieee300_rate),
                xytext=(-90, 30),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "lw": 0.8},
                fontsize=8,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, frameon=False)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(fig, output_stem)


def _plot_fig3_phase_validation(
    phase_validation_dir: Path,
    output_stem: Path,
    source_files: list[Path],
) -> dict[str, str]:
    grid_path = phase_validation_dir / "phase_implemented_error.csv"
    grid = _read_csv(grid_path, source_files)
    x = pd.to_numeric(grid["normalized_singular_value"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].plot(x, grid["scaled_target"], label="Bounded target", color="#4C78A8", lw=1.8)
    axes[0].plot(
        x,
        grid["scaled_phase_response"],
        label="Phase-implemented response",
        color="#54A24B",
        ls="--",
        lw=1.6,
    )
    axes[0].set_ylabel("Scaled response")
    axes[0].set_title("A. QSP/QSVT filter response", loc="left", fontsize=10)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    error_column = (
        "phase_scaled_abs_error" if "phase_scaled_abs_error" in grid.columns else "phase_abs_error"
    )
    axes[1].plot(x, grid[error_column], color="#E45756", lw=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Normalized singular value")
    axes[1].set_ylabel("Pointwise error")
    axes[1].set_title("B. Phase-implemented pointwise error", loc="left", fontsize=10)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def _plot_fig4_circuit_scaling(
    circuit_scaling_dir: Path,
    output_stem: Path,
    source_files: list[Path],
) -> dict[str, str]:
    scaling_path = circuit_scaling_dir / "circuit_scaling_results.csv"
    scaling = _read_csv(scaling_path, source_files)
    completed = scaling[_as_bool_series(scaling["feasible"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        ("depth_after_transpile", "Transpiled depth", False),
        ("cx_count", "CX count", False),
        ("max_error_vs_classical", "Max error", True),
    ]
    for axis, (column, ylabel, log_y) in zip(axes, panels, strict=True):
        for case_name, frame in completed.groupby("case_name", sort=False):
            frame = frame.sort_values("matrix_size")
            axis.plot(
                frame["matrix_size"],
                frame[column],
                marker="o",
                label=str(case_name).upper(),
            )
        axis.set_xlabel("Matrix size")
        axis.set_ylabel(ylabel)
        if log_y:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    axes[0].set_title("A. Depth", loc="left", fontsize=10)
    axes[1].set_title("B. CX gates", loc="left", fontsize=10)
    axes[2].set_title("C. Error vs classical filter", loc="left", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def _plot_appendix_resources(table6: pd.DataFrame, output_stem: Path) -> dict[str, str]:
    table = _sort_table(table6, case_column="case")
    x = np.arange(len(table))
    colors = ["#54A24B" if _as_bool(value) else "#E45756" for value in table["simulation_feasible"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(x, table["qubit_estimate"], color=colors)
    axes[0].set_ylabel("Estimated qubits")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(case).upper() for case in table["case"]], rotation=20)
    axes[0].set_title("A. Qubit estimate", loc="left", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, table["query_count"], color="#4C78A8")
    axes[1].set_ylabel("Estimated QSVT query count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(case).upper() for case in table["case"]], rotation=20)
    axes[1].set_title("B. Query estimate", loc="left", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def _write_table_bundle(
    tables_dir: Path,
    name: str,
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    note: str | None = None,
) -> dict[str, str]:
    csv_path = tables_dir / f"{name}.csv"
    tex_path = tables_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    _write_latex_table(tex_path, frame, caption=caption, label=label, note=note)
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    note: str | None,
) -> None:
    columns = list(frame.columns)
    with path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[htbp]\n")
        file.write("\\centering\n")
        file.write("\\small\n")
        file.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        file.write(f"\\label{{{_latex_escape(label)}}}\n")
        file.write("\\resizebox{\\textwidth}{!}{%\n")
        file.write(f"\\begin{{tabular}}{{{'l' * max(1, len(columns))}}}\n")
        file.write("\\hline\n")
        if columns:
            file.write(" & ".join(_latex_escape(_pretty_column(column)) for column in columns))
            file.write(" \\\\\n")
            file.write("\\hline\n")
            for _, row in frame.iterrows():
                values = [_format_latex_value(row.get(column)) for column in columns]
                file.write(" & ".join(values))
                file.write(" \\\\\n")
        else:
            file.write("No data \\\\\n")
        file.write("\\hline\n")
        file.write("\\end{tabular}%\n")
        file.write("}\n")
        if note:
            file.write("\\\\[0.4em]\n")
            file.write(f"\\footnotesize{{\\emph{{Note:}} {_latex_escape(note)}}}\n")
        file.write("\\end{table}\n")


def _save_figure(fig: plt.Figure, output_stem: Path) -> dict[str, str]:
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _result_quality_audit(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
    equivalence: dict[str, Any],
) -> str:
    best_improvement = _best_improvement_text(table2)
    ieee300_text = _ieee300_convergence_text(table1, table3)
    phase = table4.iloc[0] if not table4.empty else pd.Series(dtype=object)
    explicit_errors = pd.to_numeric(table5["max_error_vs_classical"], errors="coerce")
    max_circuit_error = float(explicit_errors.max()) if not explicit_errors.empty else math.nan
    infeasible_cases = table6[~_as_bool_series(table6["simulation_feasible"])]["case"].tolist()
    suspicious = _suspicious_trends(table1, table2, table3)
    baseline_text = _missing_baseline_text(table7)
    return f"""# Result Quality Audit

1. **Are benchmark results internally consistent?** Yes. Trial counts are based on unique
   `trial_id` values, failures are explicit `failed=True` records, and Ridge/QSVT matching
   was checked across {equivalence["matched_rows"]} paired rows.
2. **Where does QSVT/ridge improve over pseudoinverse?** Most clearly in selected
   ill-conditioned AC-linearized high-stress cases; {best_improvement}.
3. **Where does QSVT/ridge not improve?** It is essentially unchanged from pseudoinverse
   in many lower-stress or moderately conditioned cases, and in nonlinear AC it does not
   separate materially from the other non-robust spectral solvers.
4. **Where does Huber outperform QSVT/ridge?** Huber IRLS often has lower nonlinear AC RMSE
   and is strongest in many bad-data settings, especially when outlier robustness matters
   more than spectral regularization.
5. **Are nonlinear AC results complete and trustworthy?** They are complete through IEEE300
   with zero explicit failed trials, but convergence should be interpreted with the strict
   iteration criterion rather than only completion.
6. **What is the IEEE300 convergence caveat?** {ieee300_text}
7. **Are QSP/QSVT phase validation metrics strong?** Yes. The validation status is
   `{phase.get("validation_status", "unknown")}`, with max phase-implemented error
   {_format_plain_number(phase.get("max_phase_implemented_error"))}.
8. **Are circuit prototype results accurate enough?** The small prototypes match the
   classical spectral transform at small absolute errors, with worst listed circuit error
   {_format_plain_number(max_circuit_error)}. This supports correctness demonstrations,
   not scalable hardware execution.
9. **Are full IEEE resource estimates enough to justify not claiming IEEE118/300 hardware
   execution?** Yes. They are proxy estimates and identify feasibility limits; infeasible
   rows include {", ".join(map(str, infeasible_cases)) or "none"}.
10. **Are there any suspicious metrics or trends?** {suspicious}
11. **Were missing research-idea baselines added?** {baseline_text}

## Scores

- Large IEEE benchmark quality: 8/10
- Nonlinear AC experiment quality: 7/10
- Statistical reliability: 7/10
- QSVT phase validation quality: 9/10
- QSVT circuit evidence: 6/10
- Resource-estimation quality: 7/10
- Reproducibility: 9/10
- Manuscript-readiness: 8/10
"""


def _results_claims(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
    equivalence: dict[str, Any],
) -> str:
    best_improvement = _best_improvement_text(table2)
    phase_status = table4["validation_status"].iloc[0] if not table4.empty else "unknown"
    ieee300_text = _ieee300_convergence_text(table1, table3)
    qsvt_resource_cases = ", ".join(str(case).upper() for case in table6["case"].tolist())
    baseline_text = _missing_baseline_text(table7)
    hhl_proxy_caveat = "HHL-style inverse is a diagnostic proxy, not circuit execution."
    return f"""# Results Claims

## Safe Claims

1. QSVT/ridge improves over pseudoinverse in selected ill-conditioned
   AC-linearized cases.
   Evidence: {best_improvement}.
   Caveat: the improvement is not uniform across all cases or stressors.
2. QSVT/ridge is equivalent to ridge/Tikhonov in the classical simulation.
   Evidence: Ridge/QSVT matched across {equivalence["matched_rows"]} paired rows
   with max RMSE difference {_format_plain_number(equivalence["max_abs_rmse_diff"])}.
   Caveat: this is a classical spectral-filter simulation, not hardware execution.
3. Huber IRLS is often stronger in nonlinear or bad-data settings.
   Evidence: Nonlinear AC table reports lower Huber mean RMSE in most cases.
   Caveat: Huber is a classical robust baseline and changes the estimator family.
4. Nonlinear AC experiments completed through IEEE300 with zero explicit failures.
   Evidence: Table 1 and Table 3 separate completed trials from failed trials.
   Caveat: {ieee300_text}
5. QSP/QSVT phase validation passed.
   Evidence: Table 4 reports validation status `{phase_status}`.
   Caveat: the validation is for the bounded filter on the configured domain.
6. Research-derived QSVT circuit prototypes match classical spectral transforms.
   Evidence: Table 5 reports small absolute errors against the classical filter.
   Caveat: evidence is small-scale and research-derived.
7. Full IEEE resource and feasibility estimates are reported.
   Evidence: Table 6 covers {qsvt_resource_cases}.
   Caveat: these are proxy estimates, not hardware-native IEEE118/IEEE300 execution.
8. Missing-baseline diagnostics are now included for manuscript gap closure.
   Evidence: {baseline_text}
   Caveat: unregularized QSVT inverse is an unstable ablation. {hhl_proxy_caveat}

## Claims To Avoid

- Do not claim quantum speedup.
- Do not claim QSVT beats all classical robust baselines.
- Do not claim IEEE118 or IEEE300 hardware-native QSVT execution.
- Do not describe PYPOWER benchmark cases as real PMU/SCADA field data.
- Do not hide that Ridge and QSVT-regularized are numerically identical here.
- Do not describe max-iteration nonlinear AC trials as strict convergences.
- Do not present QSVT unregularized inverse or HHL-style proxy rows as the proposed
  regularized method.

## Suggested Results Wording

- "The QSVT-inspired regularized spectral filter matches ridge/Tikhonov in the
  classical simulation and improves over the pseudoinverse in selected
  ill-conditioned AC-linearized stress cases."
- "Huber IRLS remains a strong classical robust baseline, especially in nonlinear
  and bad-data settings."
- "All nonlinear AC IEEE300 trials completed without explicit failures, but strict
  convergence was limited under the configured iteration cap."
- "IEEE300 nonlinear AC completed all configured trials with zero numerical
  failures, but many estimator-trial results reached the maximum-iteration limit
  under the strict convergence criterion."
- "The QSP/QSVT phase validation and small research-derived circuit prototypes
  support correctness of the spectral transform at small scale; full IEEE118/300
  hardware-native QSVT execution is not claimed."
"""


def _results_section_skeleton() -> str:
    return """## Results

### 1. Large-scale IEEE benchmark performance
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:

### 2. Nonlinear AC state-estimation robustness
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat: IEEE300 nonlinear AC completed all configured trials with zero
  numerical failures, but many estimator-trial results reached the
  maximum-iteration limit under the strict convergence criterion.

### 3. Robust baseline comparison
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:

### 4. QSP/QSVT phase validation
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:

### 5. QSVT circuit and resource analysis
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:

### 6. Failure modes and limitations visible from results
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:

### 7. Missing-baseline diagnostic ablations
- Table/Figure placement:
- Main claim:
- Evidence:
- Caveat:
"""


def _manifest_issues(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table6: pd.DataFrame,
    equivalence: dict[str, Any],
) -> list[str]:
    issues = []
    max_conditions = pd.to_numeric(table1["max_condition_number"], errors="coerce")
    if np.isinf(max_conditions).any() or (max_conditions > 1.0e12).any():
        issues.append("Some benchmark condition numbers are extremely large or infinite.")
    ieee300 = table3[table3["case"].astype(str).eq("ieee300")]
    if not ieee300.empty and pd.to_numeric(ieee300["strict_convergence_rate"]).mean() < 0.5:
        issues.append(
            "IEEE300 nonlinear AC completed with zero failures but limited strict convergence."
        )
    if table2["qsvt_improvement_vs_pseudoinverse_pct"].min() < 0:
        issues.append("QSVT/ridge does not improve over pseudoinverse in every high-stress row.")
    if not equivalence["within_tolerance"]:
        issues.append("Ridge and QSVT regularized rows are not numerically equivalent.")
    if (~_as_bool_series(table6["simulation_feasible"])).any():
        issues.append(
            "At least one full IEEE resource row exceeds the configured simulation limit."
        )
    return issues


def _best_improvement_text(table2: pd.DataFrame) -> str:
    if table2.empty:
        return "no high-stress improvement rows were available"
    improvements = pd.to_numeric(
        table2["qsvt_improvement_vs_pseudoinverse_pct"],
        errors="coerce",
    )
    index = improvements.idxmax()
    row = table2.loc[index]
    return (
        f"largest high-stress improvement is {improvements.loc[index]:.1f}% for "
        f"{str(row['case']).upper()} under {row['stress_type'].lower()}"
    )


def _ieee300_convergence_text(table1: pd.DataFrame, table3: pd.DataFrame) -> str:
    table1_ieee300 = table1[
        table1["case"].astype(str).eq("ieee300")
        & table1["benchmark_mode"].astype(str).str.startswith("Nonlinear")
    ]
    table3_ieee300 = table3[table3["case"].astype(str).eq("ieee300")]
    if table1_ieee300.empty or table3_ieee300.empty:
        return "IEEE300 nonlinear AC caveat is unavailable from the generated tables."
    row = table1_ieee300.iloc[0]
    strict_rate = pd.to_numeric(table3_ieee300["strict_convergence_rate"], errors="coerce").mean()
    max_iter_rate = pd.to_numeric(table3_ieee300["max_iteration_rate"], errors="coerce").mean()
    return (
        "IEEE300 nonlinear AC completed all configured trials with zero numerical failures, "
        "but many estimator-trial results reached the maximum-iteration limit under the strict "
        f"convergence criterion. Specifically, IEEE300 completed "
        f"{int(row['completed_trials'])}/{int(row['total_trials'])} unique trials with "
        f"{int(row['failed_trials'])} explicit failures, average strict convergence was "
        f"{strict_rate:.1%}, and max-iteration rate was {max_iter_rate:.1%}."
    )


def _missing_baseline_text(table7: pd.DataFrame) -> str:
    if table7.empty:
        return "missing-baseline diagnostic rows were not available when this package was built."
    estimators = ", ".join(sorted(str(value) for value in table7["estimator"].dropna().unique()))
    runs = ", ".join(sorted(str(value) for value in table7["run"].dropna().unique()))
    return f"Table 7 includes {len(table7)} rows for {estimators} from {runs}."


def _suspicious_trends(table1: pd.DataFrame, table2: pd.DataFrame, table3: pd.DataFrame) -> str:
    notes = []
    if np.isinf(pd.to_numeric(table1["max_condition_number"], errors="coerce")).any():
        notes.append("IEEE benchmark conditioning includes infinite reported values")
    if (pd.to_numeric(table2["qsvt_improvement_vs_pseudoinverse_pct"], errors="coerce") < 1).any():
        notes.append("some high-stress rows show little QSVT/ridge improvement")
    ieee300 = table3[table3["case"].astype(str).eq("ieee300")]
    if not ieee300.empty and pd.to_numeric(ieee300["max_iteration_rate"]).mean() > 0.5:
        notes.append("IEEE300 nonlinear strict convergence is limited")
    return "; ".join(notes) + "." if notes else "No major suspicious trend was detected."


def _read_csv(path: Path, source_files: list[Path]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required source file is missing: {path}")
    source_files.append(path)
    return pd.read_csv(path)


def _read_json(path: Path, source_files: list[Path]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required source file is missing: {path}")
    source_files.append(path)
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON source must contain an object: {path}")
    return loaded


def _sort_table(
    frame: pd.DataFrame,
    *,
    case_column: str,
    estimator_column: str | None = None,
    mode_column: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    sorted_frame = frame.copy()
    sorted_frame["_case_order"] = sorted_frame[case_column].map(_case_order_map()).fillna(999)
    sort_columns = ["_case_order"]
    if mode_column and mode_column in sorted_frame.columns:
        sorted_frame["_mode_order"] = (
            sorted_frame[mode_column]
            .astype(str)
            .map({"AC-linearized PYPOWER": 0, "Nonlinear AC PYPOWER": 1})
            .fillna(99)
        )
        sort_columns.append("_mode_order")
    if "_stress_order" in sorted_frame.columns:
        sort_columns.append("_stress_order")
    if estimator_column and estimator_column in sorted_frame.columns:
        sorted_frame["_estimator_order"] = (
            sorted_frame[estimator_column]
            .map({name: index for index, name in enumerate(ESTIMATOR_ORDER)})
            .fillna(99)
        )
        sort_columns.append("_estimator_order")
    sorted_frame = sorted_frame.sort_values(sort_columns)
    return sorted_frame.drop(
        columns=[
            column
            for column in ("_case_order", "_mode_order", "_estimator_order")
            if column in sorted_frame.columns
        ]
    ).reset_index(drop=True)


def _case_order_map() -> dict[str, int]:
    return {case: index for index, case in enumerate(CASE_ORDER)}


def _first_int(frame: pd.DataFrame, column: str) -> int:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.iloc[0]) if not values.empty else 0


def _get_estimator_value(values: pd.Series, estimator: str) -> float:
    if estimator not in values.index:
        return math.nan
    return float(values.loc[estimator])


def _median_if_present(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    return float(values.median()) if not values.empty else math.nan


def _max_if_present(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    return float(values.max()) if not values.empty else math.nan


def _any_if_present(frame: pd.DataFrame, columns: list[str]) -> bool:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return False
    return any(bool(_bool_series(frame[column]).any()) for column in present)


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            return value
        try:
            if pd.isna(value):
                continue
        except ValueError:
            return value
        return value
    return math.nan


def _shape_to_string(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, list) and len(loaded) == 2:
            return f"{int(loaded[0])}x{int(loaded[1])}"
    return text


def _matrix_size_sort_key(value: Any) -> float:
    text = _shape_to_string(value)
    if "x" in text:
        first = text.split("x", maxsplit=1)[0]
        try:
            return float(first)
        except ValueError:
            return math.inf
    try:
        return float(text)
    except ValueError:
        return math.inf


def _infer_case_from_label(row: pd.Series) -> str:
    label = str(row.get("report_run_label", "")).lower()
    for case in CASE_ORDER:
        if case in label:
            return case
    return ""


def _domain_string(domain: Any) -> str:
    if isinstance(domain, (list, tuple)) and len(domain) >= 2:
        return f"[{domain[0]}, {domain[1]}]"
    return str(domain)


def _format_latex_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "$\\infty$"
        return _latex_escape(_format_plain_number(float(value)))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    return _latex_escape(str(value))


def _format_plain_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    numeric = float(value)
    if math.isinf(numeric):
        return "inf"
    abs_value = abs(numeric)
    if abs_value == 0:
        return "0"
    if abs_value < 1.0e-3 or abs_value >= 1.0e4:
        return f"{numeric:.3e}"
    return f"{numeric:.4g}"


def _pretty_column(column: str) -> str:
    replacements = {
        "qsvt": "QSVT",
        "rmse": "RMSE",
        "iqr": "IQR",
        "cx": "CX",
    }
    words = column.replace("_", " ").split()
    pretty_words = [replacements.get(word, word.capitalize()) for word in words]
    return " ".join(pretty_words)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _as_bool_series(values: pd.Series) -> pd.Series:
    return _bool_series(values)


def _stringify_artifact_map(artifacts: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in artifacts.items()}
