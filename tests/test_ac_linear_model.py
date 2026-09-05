from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.data.cases import ieee14_ac_case
from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.measurement.ac_linear import (
    ac_measurements_and_jacobian,
    build_ieee14_ac_weighted_system,
    default_ac_state_vector,
)
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config

AC_MEASUREMENT_CONFIG = {
    "include_voltage_magnitudes": True,
    "include_p_injections": True,
    "include_q_injections": True,
    "include_p_branch_flows": True,
    "include_q_branch_flows": True,
    "voltage_std": 0.01,
    "injection_p_std": 0.03,
    "injection_q_std": 0.03,
    "flow_p_std": 0.02,
    "flow_q_std": 0.02,
    "weak_area_buses": [12, 13, 14],
    "weak_area_std_multiplier": 10.0,
}


def _ac_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_ac"
    config["seed"] = 421
    config["system"]["mode"] = "ac_power_flow_linearized"
    config["system"]["measurement"] = deepcopy(AC_MEASUREMENT_CONFIG)
    config["system"]["linearization"] = {
        "angle_perturbation_std": 0.01,
        "voltage_perturbation_std": 0.01,
        "min_voltage_magnitude": 0.5,
    }
    config["scenario"]["name"] = "test_ac"
    config["scenario"]["noise_std"] = 0.002
    config["scenario"]["missing_ratio"] = 0.1
    config["estimators"] = [
        {"name": "pseudoinverse", "rcond": 1.0e-12},
        {"name": "ridge", "alpha": 1.0e-4},
        {"name": "truncated_svd", "tau": 1.0e-5},
        {"name": "qsvt_regularized", "alpha": 1.0e-4},
    ]
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_ac"
    config["output"]["save_plots"] = False
    return config


def test_ieee14_ac_case_fixture_shape() -> None:
    case = ieee14_ac_case()

    assert len(case.buses) == 14
    assert len(case.branches) == 20
    assert case.slack_bus == 1
    assert len(case.angle_state_buses) == 13
    assert len(case.voltage_state_buses) == 14


def test_ac_measurement_vector_and_jacobian_shapes() -> None:
    case = ieee14_ac_case()
    state = default_ac_state_vector(case)

    values, jacobian, rows = ac_measurements_and_jacobian(case, state, AC_MEASUREMENT_CONFIG)

    assert values.shape == (82,)
    assert jacobian.shape == (82, 27)
    assert len(rows) == 82


def test_ac_jacobian_matches_finite_differences_on_selected_rows() -> None:
    case = ieee14_ac_case()
    state = default_ac_state_vector(case)
    values, jacobian, _ = ac_measurements_and_jacobian(case, state, AC_MEASUREMENT_CONFIG)
    selected_rows = [0, 14, 28, 42, 62]
    selected_columns = [0, 5, 13, 20, 26]
    epsilon = 1.0e-6

    for row_index in selected_rows:
        for column_index in selected_columns:
            plus = state.copy()
            minus = state.copy()
            plus[column_index] += epsilon
            minus[column_index] -= epsilon
            plus_values, _, _ = ac_measurements_and_jacobian(case, plus, AC_MEASUREMENT_CONFIG)
            minus_values, _, _ = ac_measurements_and_jacobian(case, minus, AC_MEASUREMENT_CONFIG)
            finite_difference = (plus_values[row_index] - minus_values[row_index]) / (2 * epsilon)
            assert jacobian[row_index, column_index] == pytest.approx(
                finite_difference,
                abs=1.0e-4,
                rel=1.0e-4,
            )

    assert np.all(np.isfinite(values))


def test_ac_weighted_system_metadata_and_component_indices() -> None:
    rng = np.random.default_rng(421)
    system = build_ieee14_ac_weighted_system(
        linearization_config={
            "angle_perturbation_std": 0.01,
            "voltage_perturbation_std": 0.01,
            "min_voltage_magnitude": 0.5,
        },
        measurement_config=AC_MEASUREMENT_CONFIG,
        rng=rng,
    )

    assert system.metadata["mode"] == "ac_power_flow_linearized"
    assert system.n_states == 27
    assert system.n_measurements == 82
    assert len(system.metadata["angle_state_indices"]) == 13
    assert len(system.metadata["voltage_magnitude_state_indices"]) == 14
    assert np.all(np.isfinite(system.H_tilde))


def test_ac_smoke_run_writes_artifacts_and_component_metrics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")

    assert (output_dir / "config_resolved.yaml").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "estimator_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert {"angle_rmse", "voltage_magnitude_rmse"}.issubset(metrics.columns)
    assert not metrics["failed"].any()


def test_ac_sweep_run_writes_aggregate_outputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_config(tmp_path)
    config["output"]["run_id"] = "test_ac_sweep"
    config["sweeps"] = [
        {
            "name": "linearization_sweep",
            "parameter": "system.linearization.angle_perturbation_std",
            "values": [0.0, 0.01],
            "seeds": [31, 32],
        }
    ]

    run = run_experiment(config)
    output_dir = run["output_dir"]
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    summary = pd.read_csv(output_dir / "summary_metrics.csv")

    assert len(aggregate) == 2 * 2 * len(config["estimators"])
    assert {"angle_rmse_mean", "voltage_magnitude_rmse_mean"}.issubset(summary.columns)
    assert (output_dir / "trial_results.json").is_file()


def test_invalid_ac_measurement_std_fails_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_config(tmp_path)
    config["system"]["measurement"]["flow_q_std"] = 0.0

    with pytest.raises(ValueError, match="flow_q_std"):
        validate_config(config)


def test_invalid_ac_measurement_selection_fails_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_config(tmp_path)
    for key in (
        "include_voltage_magnitudes",
        "include_p_injections",
        "include_q_injections",
        "include_p_branch_flows",
        "include_q_branch_flows",
    ):
        config["system"]["measurement"][key] = False

    with pytest.raises(ValueError, match="at least one measurement group"):
        validate_config(config)
