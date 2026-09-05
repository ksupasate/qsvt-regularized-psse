from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.logging import configure_run_logger

DEFAULT_ESTIMATOR_ORDER = [
    "pseudoinverse",
    "normal_equation_wls",
    "ridge",
    "truncated_svd",
    "qsvt_regularized",
    "qsvt_unregularized_inverse",
    "hhl_style_inverse_proxy",
    "huber_irls",
    "lav",
]


@dataclass(frozen=True)
class ReportInput:
    path: Path
    label: str


@dataclass(frozen=True)
class ReportData:
    combined_metrics: pd.DataFrame
    combined_summary_metrics: pd.DataFrame
    qsvt_resources: pd.DataFrame
    iteration_trace: pd.DataFrame
    qsvt_phase_demo: pd.DataFrame


def load_report_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("report config file must contain a mapping")
    validate_report_config(raw)
    return raw


def validate_report_config(config: dict[str, Any]) -> None:
    report = _report_block(config)
    report_id = report.get("report_id")
    if not isinstance(report_id, str) or not report_id:
        raise ValueError("report.report_id must be a non-empty string")
    output_dir = report.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("report.output_dir must be a non-empty string")

    input_runs = report.get("input_runs")
    if not isinstance(input_runs, list) or not input_runs:
        raise ValueError("report.input_runs must be a non-empty list")
    for index, item in enumerate(input_runs):
        if isinstance(item, str):
            if not item:
                raise ValueError(f"report.input_runs[{index}] must not be empty")
            continue
        if not isinstance(item, dict):
            raise ValueError(f"report.input_runs[{index}] must be a path string or mapping")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"report.input_runs[{index}].path must be a non-empty string")
        label = item.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError(f"report.input_runs[{index}].label must be a non-empty string")

    phase_demo_inputs = report.get("phase_demo_inputs", [])
    if phase_demo_inputs is None:
        phase_demo_inputs = []
    if not isinstance(phase_demo_inputs, list):
        raise ValueError("report.phase_demo_inputs must be a list when provided")
    for index, item in enumerate(phase_demo_inputs):
        if isinstance(item, str):
            if not item:
                raise ValueError(f"report.phase_demo_inputs[{index}] must not be empty")
            continue
        if not isinstance(item, dict):
            raise ValueError(f"report.phase_demo_inputs[{index}] must be a path string or mapping")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"report.phase_demo_inputs[{index}].path must be a non-empty string")
        label = item.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError(f"report.phase_demo_inputs[{index}].label must be a non-empty string")

    resource_inputs = report.get("resource_inputs", [])
    if resource_inputs is None:
        resource_inputs = []
    if not isinstance(resource_inputs, list):
        raise ValueError("report.resource_inputs must be a list when provided")
    for index, item in enumerate(resource_inputs):
        if isinstance(item, str):
            if not item:
                raise ValueError(f"report.resource_inputs[{index}] must not be empty")
            continue
        if not isinstance(item, dict):
            raise ValueError(f"report.resource_inputs[{index}] must be a path string or mapping")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"report.resource_inputs[{index}].path must be a non-empty string")
        label = item.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError(f"report.resource_inputs[{index}].label must be a non-empty string")

    estimator_order = report.get("estimator_order", DEFAULT_ESTIMATOR_ORDER)
    if not isinstance(estimator_order, list) or not all(
        isinstance(item, str) and item for item in estimator_order
    ):
        raise ValueError("report.estimator_order must be a list of non-empty strings")

    compile_pdf = report.get("compile_pdf", "auto")
    if compile_pdf not in {"auto", True, False}:
        raise ValueError("report.compile_pdf must be one of: auto, true, false")


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    validate_report_config(config)
    report = _report_block(config)
    output_dir = ensure_directory(report["output_dir"])
    tables_dir = ensure_directory(output_dir / "tables")
    figures_dir = ensure_directory(output_dir / "figures")
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting manuscript report %s", report["report_id"])

    estimator_order = list(report.get("estimator_order", DEFAULT_ESTIMATOR_ORDER))
    inputs = _parse_report_inputs(report["input_runs"])
    phase_demo_inputs = _parse_report_inputs(report.get("phase_demo_inputs", []))
    resource_inputs = _parse_report_inputs(report.get("resource_inputs", []))
    data = load_report_data(
        inputs,
        phase_demo_inputs=phase_demo_inputs,
        resource_inputs=resource_inputs,
    )

    artifacts: dict[str, str] = {}
    combined_metrics_path = output_dir / "combined_metrics.csv"
    combined_summary_path = output_dir / "combined_summary_metrics.csv"
    ranking_path = output_dir / "estimator_ranking.csv"
    robust_path = output_dir / "robust_bad_data_comparison.csv"
    resource_summary_path = output_dir / "qsvt_resource_summary.csv"
    phase_demo_summary_path = output_dir / "qsvt_phase_demo_summary.csv"

    data.combined_metrics.to_csv(combined_metrics_path, index=False)
    data.combined_summary_metrics.to_csv(combined_summary_path, index=False)

    ranking = build_estimator_ranking(data.combined_summary_metrics, estimator_order)
    robust_comparison = build_robust_bad_data_comparison(data.combined_summary_metrics)
    resource_summary = build_qsvt_resource_summary(data.qsvt_resources)
    ranking.to_csv(ranking_path, index=False)
    robust_comparison.to_csv(robust_path, index=False)
    resource_summary.to_csv(resource_summary_path, index=False)
    data.qsvt_phase_demo.to_csv(phase_demo_summary_path, index=False)

    artifacts.update(
        {
            "combined_metrics": str(combined_metrics_path),
            "combined_summary_metrics": str(combined_summary_path),
            "estimator_ranking": str(ranking_path),
            "robust_bad_data_comparison": str(robust_path),
            "qsvt_resource_summary": str(resource_summary_path),
            "qsvt_phase_demo_summary": str(phase_demo_summary_path),
            "run_log": str(output_dir / "run.log"),
        }
    )

    table_paths = _write_latex_tables(
        tables_dir=tables_dir,
        summary=data.combined_summary_metrics,
        ranking=ranking,
        robust_comparison=robust_comparison,
        resource_summary=resource_summary,
        phase_demo_summary=data.qsvt_phase_demo,
        metrics=data.combined_metrics,
    )
    figure_paths = _write_report_figures(
        figures_dir=figures_dir,
        metrics=data.combined_metrics,
        summary=data.combined_summary_metrics,
        resources=data.qsvt_resources,
        phase_demo_inputs=phase_demo_inputs,
        estimator_order=estimator_order,
    )
    artifacts.update({name: str(path) for name, path in table_paths.items()})
    artifacts.update({name: str(path) for name, path in figure_paths.items()})

    report_tex = output_dir / "report.tex"
    _write_report_tex(
        path=report_tex,
        report_id=str(report["report_id"]),
        table_paths=table_paths,
        figure_paths=figure_paths,
    )
    artifacts["report_tex"] = str(report_tex)

    pdf_result = _compile_pdf(report_tex, report.get("compile_pdf", "auto"))
    if pdf_result["pdf_path"] is not None:
        artifacts["report_pdf"] = str(pdf_result["pdf_path"])

    manifest = {
        "report_id": report["report_id"],
        "input_runs": [{"path": str(item.path), "label": item.label} for item in inputs],
        "phase_demo_inputs": [
            {"path": str(item.path), "label": item.label} for item in phase_demo_inputs
        ],
        "resource_inputs": [
            {"path": str(item.path), "label": item.label} for item in resource_inputs
        ],
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "pdf_compiled": pdf_result["compiled"],
        "pdf_compile_reason": pdf_result["reason"],
        "n_metric_rows": len(data.combined_metrics),
        "n_summary_rows": len(data.combined_summary_metrics),
        "n_resource_rows": len(data.qsvt_resources),
        "n_phase_demo_rows": len(data.qsvt_phase_demo),
    }
    manifest_path = output_dir / "report_manifest.json"
    artifacts["report_manifest"] = str(manifest_path)
    manifest["artifacts"] = artifacts
    write_json(manifest_path, manifest)
    logger.info("Completed manuscript report %s", report["report_id"])
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "manifest": manifest,
        "data": data,
    }


def load_report_data(
    inputs: list[ReportInput],
    *,
    phase_demo_inputs: list[ReportInput] | None = None,
    resource_inputs: list[ReportInput] | None = None,
) -> ReportData:
    metrics_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    resource_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []

    for input_run in inputs:
        run_path = input_run.path
        if not run_path.is_dir():
            raise FileNotFoundError(f"input run directory does not exist: {run_path}")
        run_config = _read_run_config(run_path)
        metadata = _run_metadata(run_path, input_run.label, run_config)

        raw_metrics, source_type = _read_required_metrics(run_path)
        raw_metrics = _normalize_metric_columns(_annotate_frame(raw_metrics, metadata, source_type))
        metrics_frames.append(raw_metrics)

        summary_path = run_path / "summary_metrics.csv"
        if summary_path.is_file():
            summary = _normalize_metric_columns(pd.read_csv(summary_path))
        else:
            summary = _derive_summary_from_metrics(raw_metrics)
        summary_frames.append(
            _normalize_metric_columns(_annotate_frame(summary, metadata, source_type))
        )

        resource_path = run_path / "qsvt_resource_estimates.csv"
        if resource_path.is_file():
            resource_frames.append(
                _annotate_frame(pd.read_csv(resource_path), metadata, source_type)
            )

        trace_path = run_path / "iteration_trace.csv"
        if trace_path.is_file():
            trace_frames.append(_annotate_frame(pd.read_csv(trace_path), metadata, source_type))

    for resource_input in resource_inputs or []:
        resource_path = _first_existing(
            resource_input.path,
            ["resource_estimates.csv", "qsvt_resource_estimates.csv"],
        )
        if resource_path is None:
            raise FileNotFoundError(
                f"resource input is missing resource_estimates.csv: {resource_input.path}"
            )
        resource = pd.read_csv(resource_path)
        resource["report_run_label"] = resource_input.label
        resource["report_run_path"] = str(resource_input.path)
        resource["report_source_type"] = "resource_estimate"
        resource_frames.append(resource)

    return ReportData(
        combined_metrics=_concat_or_empty(metrics_frames),
        combined_summary_metrics=_concat_or_empty(summary_frames),
        qsvt_resources=_concat_or_empty(resource_frames),
        iteration_trace=_concat_or_empty(trace_frames),
        qsvt_phase_demo=load_phase_demo_data(phase_demo_inputs or []),
    )


def build_estimator_ranking(summary: pd.DataFrame, estimator_order: list[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    ranking = summary.copy()
    if "rmse_mean" not in ranking.columns:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("report_run_label", "sweep_name", "sweep_parameter", "sweep_value")
        if column in ranking.columns
    ]
    ranking["_rmse_for_rank"] = pd.to_numeric(ranking["rmse_mean"], errors="coerce")
    ranking["_rmse_for_rank"] = ranking["_rmse_for_rank"].fillna(math.inf)
    if "failed_count" in ranking.columns:
        failed = pd.to_numeric(ranking["failed_count"], errors="coerce").fillna(0.0) > 0.0
        ranking.loc[failed, "_rmse_for_rank"] = math.inf
    ranking["rmse_rank"] = ranking.groupby(group_columns, dropna=False)["_rmse_for_rank"].rank(
        method="dense",
        ascending=True,
    )
    ranking = _sort_by_estimator_order(ranking, estimator_order)
    return ranking.drop(columns=["_rmse_for_rank"]).reset_index(drop=True)


def build_robust_bad_data_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    label_text = (
        summary.get("report_run_label", pd.Series("", index=summary.index)).astype(str).str.lower()
    )
    parameter_text = summary.get("sweep_parameter", pd.Series("", index=summary.index)).astype(str)
    run_id_text = (
        summary.get("report_run_id", pd.Series("", index=summary.index)).astype(str).str.lower()
    )
    mask = (
        label_text.str.contains("bad_data|bad-data|robust", regex=True)
        | run_id_text.str.contains("bad_data|bad-data|robust", regex=True)
        | parameter_text.str.contains("scenario.bad_data", regex=False)
    )
    if "bad_data_count" in summary.columns:
        counts = pd.to_numeric(summary["bad_data_count"], errors="coerce").fillna(0.0)
        mask = mask | (counts > 0.0)
    return summary.loc[mask].reset_index(drop=True)


def build_qsvt_resource_summary(resources: pd.DataFrame) -> pd.DataFrame:
    if resources.empty:
        return pd.DataFrame(
            columns=[
                "report_run_label",
                "estimator",
                "resource_estimation_scope",
                "degree",
                "n_rows",
                "max_error_mean",
                "max_error_max",
                "target_error",
                "recommended_degree_min",
                "block_encoding_normalization_mean",
                "effective_condition_number_mean",
                "proxy_query_count_mean",
            ]
        )
    frame = resources.copy()
    for column in ("degree", "max_error", "recommended_degree"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    group_columns = [
        column
        for column in (
            "report_run_label",
            "case_name",
            "matrix_shape",
            "estimator",
            "resource_estimation_scope",
            "degree",
        )
        if column in frame.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for output, source, function in (
        ("n_rows", "max_error", "size"),
        ("max_error_mean", "max_error", "mean"),
        ("max_error_max", "max_error", "max"),
        ("target_error", "target_error", "first"),
        ("recommended_degree_min", "recommended_degree", "min"),
        ("block_encoding_normalization_mean", "block_encoding_normalization", "mean"),
        ("effective_condition_number_mean", "effective_condition_number", "mean"),
        ("proxy_query_count_mean", "proxy_query_count", "mean"),
        ("estimated_total_qubits", "estimated_total_qubits", "first"),
        ("estimated_qsvt_query_count", "estimated_qsvt_query_count", "first"),
        (
            "full_statevector_simulation_feasible",
            "full_statevector_simulation_feasible",
            "first",
        ),
    ):
        if source in frame.columns:
            aggregations[output] = (source, function)
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def load_phase_demo_data(inputs: list[ReportInput]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_run in inputs:
        run_path = input_run.path
        if not run_path.is_dir():
            raise FileNotFoundError(f"QSVT phase demo directory does not exist: {run_path}")
        summary_path = _first_existing(
            run_path,
            [
                "circuit_summary.json",
                "hardware_qsvt_circuit_summary.json",
                "circuit_scaling_summary.json",
                "qsvt_pennylane_summary.json",
                "qsvt_qiskit_summary.json",
            ],
        )
        error_path = run_path / "approximation_error.csv"
        results_path = _first_existing(
            run_path,
            [
                "qsvt_demo_results.csv",
                "comparison_to_classical.csv",
                "simulation_results.csv",
                "circuit_scaling_results.csv",
                "qsvt_pennylane_results.csv",
                "qiskit_simulation_results.csv",
            ],
        )
        if summary_path is None:
            raise FileNotFoundError(f"QSVT demo is missing a summary JSON file: {run_path}")
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        error_frame = pd.read_csv(error_path) if error_path.is_file() else pd.DataFrame()
        results_frame = (
            pd.read_csv(results_path)
            if results_path is not None and results_path.is_file()
            else pd.DataFrame()
        )
        row: dict[str, Any] = {
            "report_run_label": input_run.label,
            "report_run_path": str(run_path),
            "degree": summary.get("degree"),
            "polynomial_degree": summary.get("polynomial_degree"),
            "n_phase_angles": summary.get("n_phase_angles"),
            "phase_synthesis_method": summary.get("phase_synthesis_method"),
            "qsvt_method": summary.get("qsvt_method"),
            "qsvt_construction_type": summary.get("qsvt_construction_type"),
            "implementation_scope": summary.get("implementation_scope"),
            "block_encoding_type": summary.get("block_encoding_type"),
            "uses_dense_block_encoding_gate": summary.get("uses_dense_block_encoding_gate"),
            "is_dense_unitary_only": summary.get("is_dense_unitary_only"),
            "matrix_scope": summary.get("matrix_scope"),
            "full_or_submatrix": summary.get("full_or_submatrix"),
            "pennylane_available": summary.get("pennylane_available"),
            "qiskit_available": summary.get("qiskit_available"),
            "circuit_depth": summary.get("circuit_depth"),
            "depth_after_transpile": summary.get("depth_after_transpile"),
            "gate_count_total": summary.get("gate_count_total"),
            "gate_count_total_after_transpile": summary.get("gate_count_total_after_transpile"),
            "cx_count_after_transpile": summary.get("cx_count_after_transpile"),
            "max_cx_count": summary.get("max_cx_count"),
            "transpile_success": summary.get("transpile_success"),
            "validation_passed": summary.get("validation_passed"),
            "dummy_phase_check_passed": summary.get("dummy_phase_check_passed"),
            "boundedness_check_passed": summary.get("boundedness_check_passed"),
            "parity_check_passed": summary.get("parity_check_passed"),
            "target_scale": summary.get("target_scale"),
            "scale_factor": summary.get("scale_factor"),
            "is_full_matrix_qsvt": summary.get("is_full_matrix_qsvt"),
            "matrix_shape": summary.get("matrix_shape"),
            "source_case": summary.get("source_case"),
            "normalization_factor": summary.get("normalization_factor"),
            "phase_implemented_max_abs_error": summary.get("phase_implemented_max_abs_error"),
            "phase_implemented_mean_abs_error": summary.get("phase_implemented_mean_abs_error"),
            "scope_note": summary.get("scope_note"),
        }
        if not error_frame.empty:
            abs_error = _numeric_column(error_frame, "abs_error")
            row["max_abs_error"] = abs_error.max()
            row["mean_abs_error"] = abs_error.mean()
        if not results_frame.empty:
            result_error = _first_numeric_column(
                results_frame,
                [
                    "qsp_abs_error",
                    "polynomial_abs_error",
                    "abs_error_to_classical_filter",
                ],
            )
            row["demo_matrix_max_abs_error"] = result_error.max()
            row["demo_matrix_mean_abs_error"] = result_error.mean()
        for key in (
            "max_abs_error",
            "mean_abs_error",
            "scaled_max_abs_error",
            "scaled_mean_abs_error",
        ):
            if key in summary and key not in row:
                row[key] = summary[key]
        rows.append(row)
    return pd.DataFrame(rows)


def _first_existing(run_path: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_path / name
        if path.is_file():
            return path
    return None


def _first_numeric_column(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        values = _numeric_column(frame, column)
        if not values.empty:
            return values
    return pd.Series(dtype=float)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _report_block(config: dict[str, Any]) -> dict[str, Any]:
    report = config.get("report")
    if not isinstance(report, dict):
        raise ValueError("report config must contain a report mapping")
    return report


def _parse_report_inputs(items: list[Any]) -> list[ReportInput]:
    inputs = []
    for item in items:
        if isinstance(item, str):
            path = Path(item)
            label = path.name
        else:
            path = Path(item["path"])
            label = str(item.get("label") or path.name)
        inputs.append(ReportInput(path=path, label=label))
    return inputs


def _read_run_config(run_path: Path) -> dict[str, Any]:
    config_path = run_path / "config_resolved.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _run_metadata(run_path: Path, label: str, run_config: dict[str, Any]) -> dict[str, Any]:
    output = run_config.get("output", {}) if isinstance(run_config.get("output"), dict) else {}
    system = run_config.get("system", {}) if isinstance(run_config.get("system"), dict) else {}
    scenario = (
        run_config.get("scenario", {}) if isinstance(run_config.get("scenario"), dict) else {}
    )
    return {
        "report_run_id": output.get("run_id", run_path.name),
        "report_run_label": label,
        "report_run_path": str(run_path),
        "report_run_name": run_config.get("run_name", output.get("run_id", run_path.name)),
        "report_system_mode": system.get("mode"),
        "report_scenario_name": scenario.get("name"),
    }


def _read_required_metrics(run_path: Path) -> tuple[pd.DataFrame, str]:
    metrics_path = run_path / "metrics.csv"
    aggregate_path = run_path / "aggregate_metrics.csv"
    if metrics_path.is_file():
        source_type = (
            "iterative_single" if (run_path / "iteration_trace.csv").is_file() else "single"
        )
        return pd.read_csv(metrics_path), source_type
    if aggregate_path.is_file():
        source_type = "iterative_sweep" if (run_path / "iteration_trace.csv").is_file() else "sweep"
        return pd.read_csv(aggregate_path), source_type
    raise FileNotFoundError(
        f"input run is missing metrics.csv or aggregate_metrics.csv: {run_path}"
    )


def _annotate_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    source_type: str,
) -> pd.DataFrame:
    annotated = frame.copy()
    for key, value in metadata.items():
        annotated[key] = value
    annotated["report_source_type"] = source_type
    if "mode" not in annotated.columns and metadata.get("report_system_mode") is not None:
        annotated["mode"] = metadata["report_system_mode"]
    if (
        "scenario_name" not in annotated.columns
        and metadata.get("report_scenario_name") is not None
    ):
        annotated["scenario_name"] = metadata["report_scenario_name"]
    if "sweep_name" not in annotated.columns:
        annotated["sweep_name"] = "single_run"
    if "sweep_parameter" not in annotated.columns:
        annotated["sweep_parameter"] = ""
    if "sweep_value" not in annotated.columns:
        annotated["sweep_value"] = np.nan
    return annotated


def _derive_summary_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_pairs = {
        "rmse": "rmse_mean",
        "angle_rmse": "angle_rmse_mean",
        "voltage_magnitude_rmse": "voltage_magnitude_rmse_mean",
        "residual_norm": "residual_norm_mean",
        "weighted_residual": "weighted_residual_mean",
        "weighted_residual_norm": "weighted_residual_norm_mean",
        "weighted_residual_quadratic": "weighted_residual_quadratic_mean",
        "condition_number": "condition_number_mean",
        "runtime_seconds": "runtime_seconds_mean",
        "iterations": "iterations_mean",
    }
    for _, row in metrics.iterrows():
        failed = bool(row.get("failed", False))
        summary_row: dict[str, Any] = {
            "sweep_name": row.get("sweep_name", "single_run"),
            "sweep_parameter": row.get("sweep_parameter", ""),
            "sweep_value": row.get("sweep_value", np.nan),
            "estimator": row.get("estimator"),
            "n_trials": 1,
            "n_successful_trials": int(not failed),
            "n_failed_trials": int(failed),
            "failed_count": int(failed),
            "failure_rate": float(failed),
        }
        for source, target in metric_pairs.items():
            if source in row:
                summary_row[target] = row[source]
                for suffix, value in (
                    ("_std", 0.0),
                    ("_median", row[source]),
                    ("_q1", row[source]),
                    ("_q3", row[source]),
                    ("_iqr", 0.0),
                    ("_ci95_low", row[source]),
                    ("_ci95_high", row[source]),
                ):
                    summary_row[target.replace("_mean", suffix)] = value
        for metadata_column in (
            "case_name",
            "mode",
            "scenario_name",
            "noise_std",
            "missing_ratio",
            "bad_data_ratio",
            "bad_data_count",
            "bad_data_magnitude",
            "bad_data_target",
        ):
            if metadata_column in row:
                summary_row[metadata_column] = row[metadata_column]
        rows.append(summary_row)
    return pd.DataFrame(rows)


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

    summary_suffixes = (
        "_mean",
        "_std",
        "_median",
        "_q1",
        "_q3",
        "_iqr",
        "_ci95_low",
        "_ci95_high",
    )
    for suffix in summary_suffixes:
        legacy = f"weighted_residual{suffix}"
        norm_name = f"weighted_residual_norm{suffix}"
        quadratic_name = f"weighted_residual_quadratic{suffix}"
        if norm_name not in normalized.columns and legacy in normalized.columns:
            normalized[norm_name] = normalized[legacy]
        if quadratic_name not in normalized.columns and norm_name in normalized.columns:
            values = pd.to_numeric(normalized[norm_name], errors="coerce")
            normalized[quadratic_name] = values**2

    return _clip_nonnegative_ci_lows(normalized)


def _clip_nonnegative_ci_lows(frame: pd.DataFrame) -> pd.DataFrame:
    clipped = frame.copy()
    prefixes = {
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
    }
    for prefix in prefixes:
        column = f"{prefix}_ci95_low"
        if column in clipped.columns:
            clipped[column] = pd.to_numeric(clipped[column], errors="coerce").clip(lower=0.0)
    return clipped


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_latex_tables(
    *,
    tables_dir: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    robust_comparison: pd.DataFrame,
    resource_summary: pd.DataFrame,
    phase_demo_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, Path]:
    table_specs = {
        "estimator_performance": (
            _select_columns(
                ranking,
                [
                    "report_run_label",
                    "sweep_name",
                    "sweep_value",
                    "estimator",
                    "rmse_mean",
                    "weighted_residual_norm_mean",
                    "weighted_residual_quadratic_mean",
                    "failure_rate",
                    "rmse_rank",
                ],
            ),
            "Estimator performance summary.",
            "tab:estimator-performance",
        ),
        "robust_bad_data_comparison": (
            _select_columns(
                robust_comparison,
                [
                    "report_run_label",
                    "sweep_name",
                    "sweep_value",
                    "estimator",
                    "rmse_mean",
                    "weighted_residual_norm_mean",
                    "weighted_residual_quadratic_mean",
                    "failure_rate",
                ],
            ),
            "Bad-data and robust-baseline comparison.",
            "tab:robust-bad-data",
        ),
        "qsvt_resource_summary": (
            _select_columns(
                resource_summary,
                [
                    "report_run_label",
                    "estimator",
                    "resource_estimation_scope",
                    "degree",
                    "case_name",
                    "matrix_shape",
                    "max_error_mean",
                    "target_error",
                    "recommended_degree_min",
                    "estimated_total_qubits",
                    "estimated_qsvt_query_count",
                    "proxy_query_count_mean",
                ],
            ),
            "QSVT polynomial/resource proxy summary.",
            "tab:qsvt-resource",
        ),
        "iterative_convergence": (
            _select_columns(
                _iterative_summary(metrics, summary),
                [
                    "report_run_label",
                    "sweep_name",
                    "sweep_value",
                    "estimator",
                    "rmse_mean",
                    "weighted_residual_norm_mean",
                    "iterations_mean",
                    "failure_rate",
                ],
            ),
            "Iterative AC convergence summary.",
            "tab:iterative-convergence",
        ),
        "qsvt_phase_demo_summary": (
            _select_columns(
                phase_demo_summary,
                [
                    "report_run_label",
                    "degree",
                    "polynomial_degree",
                    "n_phase_angles",
                    "phase_synthesis_method",
                    "qsvt_method",
                    "qsvt_construction_type",
                    "implementation_scope",
                    "block_encoding_type",
                    "matrix_scope",
                    "full_or_submatrix",
                    "max_abs_error",
                    "phase_implemented_max_abs_error",
                    "demo_matrix_max_abs_error",
                    "source_case",
                    "matrix_shape",
                    "pennylane_available",
                    "qiskit_available",
                    "circuit_depth",
                    "depth_after_transpile",
                    "gate_count_total",
                    "gate_count_total_after_transpile",
                    "cx_count_after_transpile",
                    "max_cx_count",
                    "transpile_success",
                    "validation_passed",
                ],
            ),
            "Small-scale QSP/QSVT phase and circuit proof-of-concept summary.",
            "tab:qsvt-phase-demo",
        ),
    }
    paths: dict[str, Path] = {}
    for name, (frame, caption, label) in table_specs.items():
        path = tables_dir / f"{name}.tex"
        _write_latex_table(path, frame, caption=caption, label=label)
        paths[f"table_{name}"] = path
    return paths


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    selected = frame[[column for column in columns if column in frame.columns]].copy()
    return selected.head(80)


def _iterative_summary(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    iterative_modes = {"ac_iterative_state_estimation", "nonlinear_ac_state_estimation"}
    if "mode" in summary.columns:
        iterative = summary[summary["mode"].astype(str).isin(iterative_modes)]
        if not iterative.empty:
            return iterative
    if "mode" in metrics.columns:
        raw_iterative = metrics[metrics["mode"].astype(str).isin(iterative_modes)]
        if not raw_iterative.empty:
            return _derive_summary_from_metrics(raw_iterative)
    return pd.DataFrame()


def _write_latex_table(path: Path, frame: pd.DataFrame, *, caption: str, label: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[htbp]\n")
        file.write("\\centering\n")
        file.write("\\small\n")
        file.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        file.write(f"\\label{{{_latex_escape(label)}}}\n")
        file.write("\\resizebox{\\textwidth}{!}{%\n")
        columns = list(frame.columns)
        alignment = "l" * max(1, len(columns))
        file.write(f"\\begin{{tabular}}{{{alignment}}}\n")
        file.write("\\hline\n")
        if columns:
            file.write(" & ".join(_latex_escape(_pretty_column(column)) for column in columns))
            file.write(" \\\\\n")
            file.write("\\hline\n")
            for _, row in frame.iterrows():
                file.write(" & ".join(_format_latex_value(row.get(column)) for column in columns))
                file.write(" \\\\\n")
        else:
            file.write("No data \\\\\n")
        file.write("\\hline\n")
        file.write("\\end{tabular}%\n")
        file.write("}\n")
        file.write("\\end{table}\n")


def _write_report_figures(
    *,
    figures_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    resources: pd.DataFrame,
    phase_demo_inputs: list[ReportInput],
    estimator_order: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    rmse_path = figures_dir / "rmse_by_estimator_and_benchmark.png"
    _plot_rmse_by_estimator(summary, rmse_path, estimator_order)
    paths["figure_rmse_by_estimator_and_benchmark"] = rmse_path

    bad_data_path = figures_dir / "bad_data_rmse_sweep.png"
    if _plot_bad_data_sweep(summary, bad_data_path, estimator_order):
        paths["figure_bad_data_rmse_sweep"] = bad_data_path

    condition_path = figures_dir / "condition_vs_rmse.png"
    _plot_condition_vs_rmse(metrics, condition_path, estimator_order)
    paths["figure_condition_vs_rmse"] = condition_path

    qsvt_path = figures_dir / "qsvt_resource_degree_error.png"
    if _plot_qsvt_degree_error(resources, qsvt_path):
        paths["figure_qsvt_resource_degree_error"] = qsvt_path
    paths.update(_copy_phase_demo_figures(phase_demo_inputs, figures_dir))
    return paths


def _copy_phase_demo_figures(
    phase_demo_inputs: list[ReportInput],
    figures_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    figure_names = [
        "phase_validation_plot.png",
        "qsvt_phase_synthesis_plot.png",
        "qsvt_demo_plot.png",
        "circuit_scaling_plot_depth.png",
        "circuit_scaling_plot_cx.png",
        "circuit_scaling_plot_error.png",
    ]
    for index, input_run in enumerate(phase_demo_inputs):
        safe_label = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in input_run.label
        )
        copied = 0
        for figure_name in figure_names:
            source = input_run.path / figure_name
            if not source.is_file():
                continue
            stem = Path(figure_name).stem
            target = figures_dir / f"qsvt_phase_demo_{index}_{safe_label}_{stem}.png"
            shutil.copyfile(source, target)
            paths[f"figure_qsvt_phase_demo_{index}_{copied}"] = target
            copied += 1
    return paths


def _plot_rmse_by_estimator(
    summary: pd.DataFrame,
    path: Path,
    estimator_order: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty or "rmse_mean" not in summary.columns:
        _write_empty_plot(ax, "No RMSE data available")
    else:
        frame = (
            summary.groupby(["report_run_label", "estimator"], dropna=False)["rmse_mean"]
            .mean()
            .reset_index()
        )
        frame = _sort_by_estimator_order(frame, estimator_order)
        pivot = frame.pivot(index="report_run_label", columns="estimator", values="rmse_mean")
        pivot = pivot[[column for column in estimator_order if column in pivot.columns]]
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean state RMSE")
        ax.set_xlabel("Benchmark")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_bad_data_sweep(
    summary: pd.DataFrame,
    path: Path,
    estimator_order: list[str],
) -> bool:
    if summary.empty or "sweep_parameter" not in summary.columns:
        return False
    mask = summary["sweep_parameter"].astype(str).str.contains("scenario.bad_data", regex=False)
    frame = summary.loc[mask].copy()
    if frame.empty or "rmse_mean" not in frame.columns:
        return False
    frame = _sort_by_estimator_order(frame, estimator_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (run_label, estimator), estimator_frame in frame.groupby(
        ["report_run_label", "estimator"],
        sort=False,
        dropna=False,
    ):
        sorted_frame = estimator_frame.sort_values("sweep_value")
        ax.plot(
            sorted_frame["sweep_value"],
            sorted_frame["rmse_mean"],
            marker="o",
            label=f"{run_label}: {estimator}",
        )
    ax.set_xlabel("Bad-data sweep value")
    ax.set_ylabel("Mean state RMSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _plot_condition_vs_rmse(
    metrics: pd.DataFrame,
    path: Path,
    estimator_order: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if metrics.empty or not {"condition_number", "rmse", "estimator"}.issubset(metrics.columns):
        _write_empty_plot(ax, "No condition-number data available")
    else:
        frame = _sort_by_estimator_order(metrics.copy(), estimator_order)
        for estimator, estimator_frame in frame.groupby("estimator", sort=False):
            ax.scatter(
                estimator_frame["condition_number"],
                estimator_frame["rmse"],
                s=22,
                alpha=0.7,
                label=estimator,
            )
        ax.set_xlabel("Condition number")
        ax.set_ylabel("State RMSE")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_qsvt_degree_error(resources: pd.DataFrame, path: Path) -> bool:
    if resources.empty or not {"degree", "max_error"}.issubset(resources.columns):
        return False
    frame = (
        resources.assign(
            degree=pd.to_numeric(resources["degree"], errors="coerce"),
            max_error=pd.to_numeric(resources["max_error"], errors="coerce"),
        )
        .dropna(subset=["degree", "max_error"])
        .groupby("degree", dropna=False)
        .agg(max_error_mean=("max_error", "mean"), target_error=("target_error", "first"))
        .reset_index()
        .sort_values("degree")
    )
    if frame.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["degree"], frame["max_error_mean"], marker="o", label="Mean max error")
    if "target_error" in frame.columns:
        target_error = pd.to_numeric(frame["target_error"], errors="coerce").dropna()
        if not target_error.empty:
            ax.axhline(
                float(target_error.iloc[0]),
                linestyle="--",
                color="black",
                label="Target error",
            )
    ax.set_xlabel("Chebyshev degree")
    ax.set_ylabel("Approximation max error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _write_empty_plot(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def _write_report_tex(
    *,
    path: Path,
    report_id: str,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("\\documentclass[11pt]{article}\n")
        file.write("\\usepackage[margin=1in]{geometry}\n")
        file.write("\\usepackage{graphicx}\n")
        file.write("\\usepackage{booktabs}\n")
        file.write("\\usepackage{float}\n")
        file.write("\\title{Robust QSVT State-Estimation Benchmark Report}\n")
        file.write("\\author{Generated research-code artifact}\n")
        file.write("\\date{\\today}\n")
        file.write("\\begin{document}\n")
        file.write("\\maketitle\n")
        file.write(
            "This report summarizes completed synthetic, DC, AC-linearized, "
            "nonlinear AC, bad-data, robust-baseline, QSVT resource-proxy, "
            "and small QSP/QSVT proof-of-concept outputs. Large IEEE results "
            "are classical spectral simulations on standard benchmark cases, "
            "not PMU/SCADA field-data validation or a quantum speedup claim.\n\n"
        )
        file.write(f"\\section*{{Report ID: {_latex_escape(report_id)}}}\n")
        file.write("\\section*{Tables}\n")
        for table_path in table_paths.values():
            file.write(f"\\input{{{_latex_path(table_path, path.parent)}}}\n")
        file.write("\\clearpage\n")
        file.write("\\section*{Figures}\n")
        for figure_path in figure_paths.values():
            file.write("\\begin{figure}[H]\n")
            file.write("\\centering\n")
            figure_rel_path = _latex_path(figure_path, path.parent)
            file.write(f"\\includegraphics[width=0.95\\textwidth]{{{figure_rel_path}}}\n")
            caption = _latex_escape(figure_path.stem.replace("_", " ").title())
            file.write(f"\\caption{{{caption}}}\n")
            file.write("\\end{figure}\n")
        file.write("\\end{document}\n")


def _compile_pdf(report_tex: Path, compile_pdf: bool | str) -> dict[str, Any]:
    pdflatex = shutil.which("pdflatex")
    if compile_pdf is False:
        return {"compiled": False, "reason": "disabled", "pdf_path": None}
    if pdflatex is None:
        if compile_pdf is True:
            raise RuntimeError("compile_pdf=true but pdflatex is not available")
        return {"compiled": False, "reason": "pdflatex not available", "pdf_path": None}

    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", report_tex.name]
    completed = subprocess.run(
        command,
        cwd=report_tex.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        if compile_pdf is True:
            raise RuntimeError(f"pdflatex failed: {completed.stderr or completed.stdout}")
        return {"compiled": False, "reason": "pdflatex failed", "pdf_path": None}
    return {
        "compiled": True,
        "reason": "compiled with pdflatex",
        "pdf_path": report_tex.with_suffix(".pdf"),
    }


def _sort_by_estimator_order(frame: pd.DataFrame, estimator_order: list[str]) -> pd.DataFrame:
    if "estimator" not in frame.columns:
        return frame
    order = {name: index for index, name in enumerate(estimator_order)}
    sorted_frame = frame.copy()
    sorted_frame["_estimator_order"] = sorted_frame["estimator"].map(order).fillna(len(order))
    sort_columns = [
        column for column in ("report_run_label", "sweep_name", "sweep_value") if column in frame
    ]
    sort_columns.append("_estimator_order")
    sorted_frame = sorted_frame.sort_values(sort_columns, kind="mergesort")
    return sorted_frame.drop(columns=["_estimator_order"])


def _format_latex_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, list | tuple | dict):
        return _latex_escape(str(value))
    if isinstance(value, float) and math.isnan(value):
        return "--"
    if pd.isna(value):
        return "--"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4g}"
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return _latex_escape(str(value))


def _pretty_column(column: str) -> str:
    replacements = {
        "report_run_label": "benchmark",
        "sweep_name": "sweep",
        "sweep_value": "value",
        "rmse_mean": "rmse mean",
        "weighted_residual_norm_mean": "weighted residual norm mean",
        "weighted_residual_quadratic_mean": "weighted residual quadratic mean",
        "failure_rate": "failure rate",
        "rmse_rank": "rmse rank",
        "resource_estimation_scope": "resource scope",
        "recommended_degree_min": "recommended degree",
        "proxy_query_count_mean": "proxy queries mean",
    }
    return replacements.get(column, column.replace("_", " "))


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


def _latex_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()
