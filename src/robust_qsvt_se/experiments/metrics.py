from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult
from robust_qsvt_se.measurement.linear_system import WeightedSystem


def rmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    if x_hat.shape != x_true.shape:
        raise ValueError(f"x_hat shape {x_hat.shape} does not match x_true shape {x_true.shape}")
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)))


def residual_norm(system: WeightedSystem, x_hat: np.ndarray) -> float:
    return system.residual_norm(x_hat)


def weighted_residual(system: WeightedSystem, x_hat: np.ndarray) -> float:
    return system.weighted_residual(x_hat)


def weighted_residual_norm(system: WeightedSystem, x_hat: np.ndarray) -> float:
    return system.weighted_residual_norm(x_hat)


def weighted_residual_quadratic(system: WeightedSystem, x_hat: np.ndarray) -> float:
    return system.weighted_residual_quadratic(x_hat)


def condition_number(system: WeightedSystem) -> float:
    return system.condition_number()


def metrics_row(result: EstimatorResult, system: WeightedSystem) -> dict[str, Any]:
    component_metrics = _component_rmse(result, system)
    row = {
        "estimator": result.name,
        "rmse": result.rmse,
        "angle_rmse": component_metrics.get("angle_rmse"),
        "voltage_magnitude_rmse": component_metrics.get("voltage_magnitude_rmse"),
        "residual_norm": result.residual_norm,
        "weighted_residual": result.weighted_residual,
        "weighted_residual_norm": result.weighted_residual_norm,
        "weighted_residual_quadratic": result.weighted_residual_quadratic,
        "condition_number": result.condition_number,
        "rank": result.rank,
        "effective_rank": result.effective_rank,
        "runtime_seconds": result.runtime_seconds,
        "converged": result.converged,
        "failed": result.failed,
        "failure_reason": result.failure_reason,
        "case_name": system.metadata.get("case_name"),
        "dataset_source": system.metadata.get("dataset_source"),
        "dataset_source_detail": system.metadata.get("dataset_source_detail"),
        "external_case": system.metadata.get("external_case"),
        "mode": system.metadata.get("mode"),
        "n_measurements": system.n_measurements,
        "n_states": system.n_states,
        "scenario_name": system.metadata.get("scenario_name"),
        "noise_std": system.metadata.get("noise_std", 0.0),
        "missing_ratio": system.metadata.get("missing_ratio", 0.0),
        "bad_data_ratio": system.metadata.get("bad_data_ratio", 0.0),
        "bad_data_count": system.metadata.get("bad_data_count", 0),
        "bad_data_magnitude": system.metadata.get("bad_data_magnitude"),
        "bad_data_target": system.metadata.get("bad_data_target"),
        "target_condition_number": system.metadata.get("target_condition_number"),
        "achieved_condition_number": system.metadata.get("achieved_condition_number"),
    }
    row.update(_selected_extra_diagnostics(result))
    return row


def _selected_extra_diagnostics(result: EstimatorResult) -> dict[str, Any]:
    diagnostics = result.extra_diagnostics
    return {
        key: diagnostics.get(key)
        for key in (
            "normal_matrix_condition_number",
            "singular_values_below_cutoff",
            "unstable_ablation",
            "hhl_effective_condition_number",
            "hhl_resource_proxy",
            "hhl_instability_flag",
        )
        if key in diagnostics
    }


def _component_rmse(result: EstimatorResult, system: WeightedSystem) -> dict[str, float | None]:
    if system.x_true is None:
        return {"angle_rmse": None, "voltage_magnitude_rmse": None}
    metrics: dict[str, float | None] = {}
    for output_name, metadata_key in (
        ("angle_rmse", "angle_state_indices"),
        ("voltage_magnitude_rmse", "voltage_magnitude_state_indices"),
    ):
        indices = system.metadata.get(metadata_key)
        if not indices:
            metrics[output_name] = None
            continue
        index_array = np.asarray(indices, dtype=int)
        metrics[output_name] = rmse(result.x_hat[index_array], system.x_true[index_array])
    return metrics
