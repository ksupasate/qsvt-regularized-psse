from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.measurement.perturbations import (
    add_bad_data_outliers,
    add_bad_data_to_measurements,
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


def _toy_system() -> WeightedSystem:
    return WeightedSystem(
        H_tilde=np.eye(8, 3),
        r_tilde=np.zeros(8),
        x_true=np.zeros(3),
        metadata={
            "measurement_buses": [[1], [2], [12], [3, 4], [13], [5], [14], [6]],
            "weak_area_buses": [12, 13, 14],
        },
    )


def _ac_bad_data_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_bad_data_ac"
    config["seed"] = 801
    config["system"]["mode"] = "ac_power_flow_linearized"
    config["system"]["measurement"] = deepcopy(AC_MEASUREMENT_CONFIG)
    config["system"]["linearization"] = {
        "angle_perturbation_std": 0.01,
        "voltage_perturbation_std": 0.01,
        "min_voltage_magnitude": 0.5,
    }
    config["scenario"]["name"] = "test_bad_data_ac"
    config["scenario"]["noise_std"] = 0.001
    config["scenario"]["missing_ratio"] = 0.1
    config["scenario"]["bad_data"] = {
        "enabled": True,
        "ratio": 0.1,
        "magnitude": 5.0,
        "target": "weak_area",
    }
    config["estimators"] = [
        {"name": "pseudoinverse", "rcond": 1.0e-12},
        {"name": "ridge", "alpha": 1.0e-4},
        {"name": "truncated_svd", "tau": 1.0e-5},
        {"name": "qsvt_regularized", "alpha": 1.0e-4},
    ]
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [2, 4],
        "grid_size": 64,
        "target_error": 1.0e6,
    }
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_bad_data_ac"
    config["output"]["save_plots"] = False
    return config


def _iterative_bad_data_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = _ac_bad_data_config(tmp_path)
    config["run_name"] = "test_iterative_bad_data"
    config["seed"] = 820
    config["system"]["mode"] = "ac_iterative_state_estimation"
    config["system"]["iteration"] = {
        "max_iterations": 8,
        "update_tolerance": 1.0e-8,
        "residual_tolerance": 1.0e-8,
        "damping": 1.0,
    }
    config["scenario"]["missing_ratio"] = 0.0
    config["scenario"]["bad_data"]["ratio"] = 0.05
    config["output"]["run_id"] = "test_iterative_bad_data"
    return config


def test_bad_data_injection_is_deterministic_for_fixed_seed() -> None:
    config = {"enabled": True, "ratio": 0.25, "magnitude": 4.0, "target": "random"}

    first = add_bad_data_outliers(
        _toy_system(),
        bad_data_config=config,
        rng=np.random.default_rng(123),
    )
    second = add_bad_data_outliers(
        _toy_system(),
        bad_data_config=config,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(first.r_tilde, second.r_tilde)
    assert first.metadata["bad_data_rows"] == second.metadata["bad_data_rows"]
    assert first.metadata["bad_data_signs"] == second.metadata["bad_data_signs"]


def test_bad_data_ratio_controls_corrupted_row_count() -> None:
    system = add_bad_data_outliers(
        _toy_system(),
        bad_data_config={"enabled": True, "ratio": 0.25, "magnitude": 4.0, "target": "random"},
        rng=np.random.default_rng(123),
    )

    assert system.metadata["bad_data_count"] == 2
    assert np.count_nonzero(system.r_tilde) == 2


def test_bad_data_weak_area_target_selects_only_weak_rows() -> None:
    system = add_bad_data_outliers(
        _toy_system(),
        bad_data_config={
            "enabled": True,
            "ratio": 0.5,
            "magnitude": 4.0,
            "target": "weak_area",
        },
        rng=np.random.default_rng(123),
    )

    weak_area = set(system.metadata["weak_area_buses"])
    measurement_buses = system.metadata["measurement_buses"]
    bad_rows = system.metadata["bad_data_rows"]
    weak_rows = [
        index
        for index in range(len(measurement_buses))
        if weak_area.intersection(measurement_buses[index])
    ]
    # Only weak-area rows are corrupted, and ratio=0.5 corrupts round(0.5 * weak rows) of them.
    assert set(bad_rows).issubset(set(weak_rows))
    assert len(bad_rows) == round(0.5 * len(weak_rows))


def test_bad_data_measurement_injection_uses_weighted_magnitude_units() -> None:
    metadata = {
        "measurement_buses": [[1], [12], [2], [13]],
        "weak_area_buses": [12, 13],
    }

    z, out_metadata = add_bad_data_to_measurements(
        np.zeros(4),
        measurement_stds=np.array([0.1, 0.2, 0.3, 0.4]),
        metadata=metadata,
        bad_data_config={"enabled": True, "ratio": 0.99, "magnitude": 5.0, "target": "weak_area"},
        rng=np.random.default_rng(123),
    )

    weighted = z / np.array([0.1, 0.2, 0.3, 0.4])
    assert out_metadata["bad_data_count"] == 2
    np.testing.assert_allclose(np.abs(weighted[out_metadata["bad_data_rows"]]), 5.0)


@pytest.mark.parametrize(
    ("bad_data", "match"),
    [
        ({"enabled": True, "ratio": 1.0, "magnitude": 1.0, "target": "random"}, "ratio"),
        ({"enabled": True, "ratio": 0.1, "magnitude": 0.0, "target": "random"}, "magnitude"),
        ({"enabled": True, "ratio": 0.1, "magnitude": 1.0, "target": "not_real"}, "target"),
    ],
)
def test_invalid_bad_data_config_fails_validation(
    bad_data: dict[str, object],
    match: str,
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["scenario"]["bad_data"] = bad_data

    with pytest.raises(ValueError, match=match):
        validate_config(config)


def test_weak_area_target_without_candidates_fails_clearly() -> None:
    system = WeightedSystem(
        H_tilde=np.eye(3),
        r_tilde=np.zeros(3),
        metadata={"measurement_buses": [[1], [2], [3]], "weak_area_buses": [12]},
    )

    with pytest.raises(ValueError, match="no eligible"):
        add_bad_data_outliers(
            system,
            bad_data_config={
                "enabled": True,
                "ratio": 0.5,
                "magnitude": 4.0,
                "target": "weak_area",
            },
            rng=np.random.default_rng(123),
        )


def test_single_step_bad_data_smoke_writes_metrics_and_resources(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_bad_data_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")

    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert {"bad_data_ratio", "bad_data_count", "bad_data_magnitude", "bad_data_target"}.issubset(
        metrics.columns
    )
    assert set(metrics["bad_data_target"]) == {"weak_area"}
    assert metrics["bad_data_count"].min() > 0


def test_bad_data_sweep_writes_aggregate_outputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _ac_bad_data_config(tmp_path)
    config["output"]["run_id"] = "test_bad_data_sweep"
    config["sweeps"] = [
        {
            "name": "bad_data_ratio_sweep",
            "parameter": "scenario.bad_data.ratio",
            "values": [0.0, 0.1],
            "seeds": [31, 32],
        }
    ]

    run = run_experiment(config)
    output_dir = run["output_dir"]
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")

    assert (output_dir / "summary_metrics.csv").is_file()
    assert (output_dir / "trial_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert {"bad_data_ratio", "bad_data_count", "bad_data_magnitude", "bad_data_target"}.issubset(
        aggregate.columns
    )
    assert len(aggregate) == 2 * 2 * len(config["estimators"])


def test_iterative_ac_bad_data_run_records_structured_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_bad_data_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")
    trace = pd.read_csv(output_dir / "iteration_trace.csv")

    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert {"iterations", "converged", "failed", "bad_data_count"}.issubset(metrics.columns)
    assert set(metrics["bad_data_target"]) == {"weak_area"}
    assert metrics["bad_data_count"].min() > 0
    assert not trace.empty
