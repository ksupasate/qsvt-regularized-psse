from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robust_qsvt_se.estimators.base import EstimatorResult
from robust_qsvt_se.experiments.metrics import metrics_row
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.resources import estimate_qsvt_resources
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml


def write_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    system: WeightedSystem,
    results: list[EstimatorResult],
    save_plots: bool,
) -> dict[str, Path]:
    output_path = ensure_directory(output_dir)
    metrics = pd.DataFrame([metrics_row(result, system) for result in results])
    singular_values = system.singular_values()
    singular_values_df = pd.DataFrame(
        {"index": list(range(len(singular_values))), "singular_value": singular_values}
    )

    config_path = output_path / "config_resolved.yaml"
    metrics_path = output_path / "metrics.csv"
    results_path = output_path / "estimator_results.json"
    singular_values_path = output_path / "singular_values.csv"

    write_yaml(config_path, config)
    metrics.to_csv(metrics_path, index=False)
    resource_rows = _single_run_resource_rows(config, singular_values)
    resource_artifact = _write_resource_estimates(output_path, resource_rows)
    _attach_resource_diagnostics_to_estimator_results(results, resource_rows, "single_run")
    write_json(
        results_path,
        {
            "system_metadata": system.metadata,
            "results": [result.to_dict() for result in results],
        },
    )
    singular_values_df.to_csv(singular_values_path, index=False)

    artifacts = {
        "config": config_path,
        "metrics": metrics_path,
        "estimator_results": results_path,
        "singular_values": singular_values_path,
    }
    if resource_artifact is not None:
        artifacts["qsvt_resource_estimates"] = resource_artifact
    if save_plots:
        artifacts.update(_write_plots(output_path, metrics, singular_values_df))
    return artifacts


def write_sweep_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    aggregate_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    singular_values: pd.DataFrame,
    trial_results: list[dict[str, Any]],
    save_plots: bool,
) -> dict[str, Path]:
    output_path = ensure_directory(output_dir)
    config_path = output_path / "config_resolved.yaml"
    aggregate_path = output_path / "aggregate_metrics.csv"
    summary_path = output_path / "summary_metrics.csv"
    singular_values_path = output_path / "singular_values.csv"
    trial_results_path = output_path / "trial_results.json"
    resource_rows = (
        _iterative_resource_rows(config, singular_values)
        if "estimator" in singular_values.columns
        else _sweep_resource_rows(config, singular_values)
    )
    resource_artifact = _write_resource_estimates(output_path, resource_rows)
    _attach_resource_diagnostics_to_trial_results(trial_results, resource_rows, "sweep_trial")

    write_yaml(config_path, config)
    aggregate_metrics.to_csv(aggregate_path, index=False)
    summary_metrics.to_csv(summary_path, index=False)
    singular_values.to_csv(singular_values_path, index=False)
    write_json(trial_results_path, {"trials": trial_results})

    artifacts = {
        "config": config_path,
        "aggregate_metrics": aggregate_path,
        "summary_metrics": summary_path,
        "singular_values": singular_values_path,
        "trial_results": trial_results_path,
    }
    if resource_artifact is not None:
        artifacts["qsvt_resource_estimates"] = resource_artifact
    if save_plots:
        artifacts.update(_write_sweep_plots(output_path, summary_metrics))
    return artifacts


def write_iterative_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    problem_metadata: dict[str, Any],
    metrics: pd.DataFrame,
    iteration_trace: pd.DataFrame,
    singular_values: pd.DataFrame,
    results: list[dict[str, Any]],
    save_plots: bool,
) -> dict[str, Path]:
    output_path = ensure_directory(output_dir)
    config_path = output_path / "config_resolved.yaml"
    metrics_path = output_path / "metrics.csv"
    results_path = output_path / "estimator_results.json"
    singular_values_path = output_path / "singular_values.csv"
    trace_path = output_path / "iteration_trace.csv"
    resource_rows = _iterative_resource_rows(config, singular_values)
    resource_artifact = _write_resource_estimates(output_path, resource_rows)
    _attach_resource_diagnostics_to_result_dicts(results, resource_rows, "iterative_final_spectrum")

    write_yaml(config_path, config)
    metrics.to_csv(metrics_path, index=False)
    singular_values.to_csv(singular_values_path, index=False)
    iteration_trace.to_csv(trace_path, index=False)
    write_json(results_path, {"problem_metadata": problem_metadata, "results": results})

    artifacts = {
        "config": config_path,
        "metrics": metrics_path,
        "estimator_results": results_path,
        "singular_values": singular_values_path,
        "iteration_trace": trace_path,
    }
    if resource_artifact is not None:
        artifacts["qsvt_resource_estimates"] = resource_artifact
    if save_plots:
        artifacts.update(_write_iterative_plots(output_path, metrics, iteration_trace))
    return artifacts


def write_iterative_sweep_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    aggregate_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    iteration_trace: pd.DataFrame,
    singular_values: pd.DataFrame,
    trial_results: list[dict[str, Any]],
    save_plots: bool,
) -> dict[str, Path]:
    artifacts = write_sweep_artifacts(
        output_dir=output_dir,
        config=config,
        aggregate_metrics=aggregate_metrics,
        summary_metrics=summary_metrics,
        singular_values=singular_values,
        trial_results=trial_results,
        save_plots=save_plots,
    )
    trace_path = Path(output_dir) / "iteration_trace.csv"
    iteration_trace.to_csv(trace_path, index=False)
    artifacts["iteration_trace"] = trace_path
    if save_plots:
        artifacts.update(_write_iterative_trace_plot(Path(output_dir), iteration_trace))
    return artifacts


def _single_run_resource_rows(
    config: dict[str, Any],
    singular_values: np.ndarray,
) -> list[dict[str, Any]]:
    return _resource_rows_for_spectrum(
        config,
        singular_values,
        extra={"resource_estimation_scope": "single_run", "estimator": "qsvt_regularized"},
    )


def _sweep_resource_rows(
    config: dict[str, Any],
    singular_values: pd.DataFrame,
) -> list[dict[str, Any]]:
    if singular_values.empty:
        return []
    required_columns = {"trial_id", "sweep_name", "parameter", "value", "seed", "singular_value"}
    if not required_columns.issubset(singular_values.columns):
        return []

    rows: list[dict[str, Any]] = []
    group_columns = ["trial_id", "sweep_name", "parameter", "value", "seed"]
    for group_key, frame in singular_values.groupby(group_columns, sort=False, dropna=False):
        trial_id, sweep_name, parameter, sweep_value, seed = group_key
        rows.extend(
            _resource_rows_for_spectrum(
                config,
                frame["singular_value"].to_numpy(dtype=np.float64),
                extra={
                    "resource_estimation_scope": "sweep_trial",
                    "trial_id": trial_id,
                    "sweep_name": sweep_name,
                    "sweep_parameter": parameter,
                    "sweep_value": sweep_value,
                    "seed": seed,
                    "estimator": "qsvt_regularized",
                },
            )
        )
    return rows


def _iterative_resource_rows(
    config: dict[str, Any],
    singular_values: pd.DataFrame,
) -> list[dict[str, Any]]:
    if singular_values.empty or "estimator" not in singular_values.columns:
        return []

    rows: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in ("trial_id", "sweep_name", "parameter", "value", "seed", "estimator")
        if column in singular_values.columns
    ]
    for group_key, frame in singular_values.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        extra = {
            "resource_estimation_scope": (
                "iterative_sweep_final_spectrum"
                if "trial_id" in group_columns
                else "iterative_final_spectrum"
            )
        }
        extra.update(dict(zip(group_columns, group_key, strict=True)))
        if "value" in extra:
            extra["sweep_value"] = extra.pop("value")
        if "parameter" in extra:
            extra["sweep_parameter"] = extra.pop("parameter")
        rows.extend(
            _resource_rows_for_spectrum(
                config,
                frame["singular_value"].to_numpy(dtype=np.float64),
                extra=extra,
            )
        )
    return rows


def _resource_rows_for_spectrum(
    config: dict[str, Any],
    singular_values: np.ndarray,
    *,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    resource_config = _resource_config(config)
    if resource_config is None:
        return []
    qsvt_alpha = _qsvt_alpha(config)
    if qsvt_alpha is None:
        return []

    estimates = estimate_qsvt_resources(
        singular_values,
        alpha=qsvt_alpha,
        degrees=list(resource_config["degrees"]),
        grid_size=int(resource_config["grid_size"]),
        target_error=float(resource_config["target_error"]),
    )
    return [estimate.to_row(extra=extra) for estimate in estimates]


def _resource_config(config: dict[str, Any]) -> dict[str, Any] | None:
    resource_config = config.get("qsvt_resource")
    if not isinstance(resource_config, dict) or not bool(resource_config.get("enabled", False)):
        return None
    return resource_config


def _qsvt_alpha(config: dict[str, Any]) -> float | None:
    for estimator in config.get("estimators", []):
        if estimator.get("name") == "qsvt_regularized":
            return float(estimator["alpha"])
    return None


def _write_resource_estimates(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> Path | None:
    if not rows:
        return None
    path = output_path / "qsvt_resource_estimates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _attach_resource_diagnostics_to_estimator_results(
    results: list[EstimatorResult],
    resource_rows: list[dict[str, Any]],
    scope: str,
) -> None:
    diagnostics = _resource_diagnostics_from_rows(resource_rows, scope)
    if diagnostics is None:
        return
    for result in results:
        if result.name == "qsvt_regularized":
            result.extra_diagnostics.update(diagnostics)


def _attach_resource_diagnostics_to_result_dicts(
    results: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    scope: str,
) -> None:
    diagnostics_by_estimator = _resource_diagnostics_by_estimator(resource_rows, scope)
    if not diagnostics_by_estimator:
        return
    for result in results:
        if result.get("name") == "qsvt_regularized":
            diagnostics = diagnostics_by_estimator.get(
                "qsvt_regularized",
                next(iter(diagnostics_by_estimator.values())),
            )
            result.setdefault("extra_diagnostics", {}).update(diagnostics)


def _attach_resource_diagnostics_to_trial_results(
    trial_results: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    scope: str,
) -> None:
    rows_by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in resource_rows:
        trial_id = str(row.get("trial_id", ""))
        rows_by_trial.setdefault(trial_id, []).append(row)
    for trial in trial_results:
        trial_id = str(trial.get("trial_id", ""))
        _attach_resource_diagnostics_to_result_dicts(
            trial.get("results", []),
            rows_by_trial.get(trial_id, []),
            scope,
        )


def _resource_diagnostics_by_estimator(
    resource_rows: list[dict[str, Any]],
    scope: str,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for estimator, rows in _rows_grouped_by_estimator(resource_rows).items():
        row_diagnostics = _resource_diagnostics_from_rows(rows, scope)
        if row_diagnostics is not None:
            diagnostics[estimator] = row_diagnostics
    return diagnostics


def _rows_grouped_by_estimator(
    resource_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in resource_rows:
        estimator = str(row.get("estimator", "qsvt_regularized"))
        grouped.setdefault(estimator, []).append(row)
    return grouped


def _resource_diagnostics_from_rows(
    resource_rows: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any] | None:
    if not resource_rows:
        return None
    first = resource_rows[0]
    return {
        "resource_estimation_scope": first.get("resource_estimation_scope", scope),
        "recommended_polynomial_degree": first["recommended_degree"],
        "target_error": first["target_error"],
        "block_encoding_normalization": first["block_encoding_normalization"],
    }


def _write_plots(
    output_path: Path,
    metrics: pd.DataFrame,
    singular_values: pd.DataFrame,
) -> dict[str, Path]:
    error_plot = output_path / "error_vs_estimator.png"
    spectrum_plot = output_path / "singular_spectrum.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics["estimator"], metrics["rmse"])
    ax.set_ylabel("State RMSE")
    ax.set_xlabel("Estimator")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(error_plot, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(singular_values["index"], singular_values["singular_value"], marker="o")
    ax.set_ylabel("Singular value")
    ax.set_xlabel("Index")
    ax.set_title("Weighted system singular spectrum")
    fig.tight_layout()
    fig.savefig(spectrum_plot, dpi=150)
    plt.close(fig)

    return {"error_plot": error_plot, "spectrum_plot": spectrum_plot}


def _write_sweep_plots(output_path: Path, summary_metrics: pd.DataFrame) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    plot_specs = {
        "rmse_sweep.png": ("rmse_mean", "State RMSE"),
        "residual_sweep.png": ("residual_norm_mean", "Residual norm"),
        "condition_sweep.png": ("condition_number_mean", "Condition number"),
    }
    for filename, (metric_column, ylabel) in plot_specs.items():
        plot_path = output_path / filename
        fig, axes = _make_sweep_axes(summary_metrics)
        for ax, (sweep_name, sweep_frame) in zip(
            axes,
            summary_metrics.groupby("sweep_name", sort=False),
            strict=False,
        ):
            for estimator, estimator_frame in sweep_frame.groupby("estimator", sort=False):
                sorted_frame = estimator_frame.sort_values("sweep_value")
                ax.plot(
                    sorted_frame["sweep_value"],
                    sorted_frame[metric_column],
                    marker="o",
                    label=estimator,
                )
            ax.set_title(str(sweep_name))
            ax.set_xlabel("Sweep value")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        artifacts[filename.removesuffix(".png")] = plot_path
    return artifacts


def _make_sweep_axes(summary_metrics: pd.DataFrame) -> tuple[plt.Figure, list[plt.Axes]]:
    n_sweeps = max(1, int(summary_metrics["sweep_name"].nunique()))
    fig, axes = plt.subplots(n_sweeps, 1, figsize=(8, 3.5 * n_sweeps), squeeze=False)
    return fig, list(axes[:, 0])


def _write_iterative_plots(
    output_path: Path,
    metrics: pd.DataFrame,
    iteration_trace: pd.DataFrame,
) -> dict[str, Path]:
    artifacts = _write_iterative_trace_plot(output_path, iteration_trace)
    convergence_plot = output_path / "convergence_summary.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    residual_column = (
        "weighted_residual_norm" if "weighted_residual_norm" in metrics else "weighted_residual"
    )
    ax.bar(metrics["estimator"], metrics[residual_column])
    ax.set_ylabel("Final weighted residual norm")
    ax.set_xlabel("Estimator")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(convergence_plot, dpi=150)
    plt.close(fig)
    artifacts["convergence_summary"] = convergence_plot
    return artifacts


def _write_iterative_trace_plot(
    output_path: Path,
    iteration_trace: pd.DataFrame,
) -> dict[str, Path]:
    trace_plot = output_path / "iteration_residuals.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "trial_id" in iteration_trace.columns:
        plot_frame = (
            iteration_trace.groupby(["estimator", "iteration"], dropna=False)[
                "weighted_residual_after"
            ]
            .mean()
            .reset_index()
        )
    else:
        plot_frame = iteration_trace
    for estimator, estimator_frame in plot_frame.groupby("estimator", sort=False):
        sorted_frame = estimator_frame.sort_values("iteration")
        ax.semilogy(
            sorted_frame["iteration"],
            sorted_frame["weighted_residual_after"],
            marker="o",
            label=estimator,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted residual after update")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(trace_plot, dpi=150)
    plt.close(fig)
    return {"iteration_residuals": trace_plot}
