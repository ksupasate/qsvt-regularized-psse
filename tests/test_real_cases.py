from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.data.cases import ac_measurement_count, ac_state_count, load_ac_case
from robust_qsvt_se.data.real_cases import SUPPORTED_REAL_CASES, load_power_case
from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, load_config, validate_config

REAL_MEASUREMENT_CONFIG = {
    "include_voltage_magnitudes": True,
    "include_p_injections": True,
    "include_q_injections": False,
    "include_p_branch_flows": True,
    "include_q_branch_flows": False,
    "voltage_std": 0.01,
    "injection_p_std": 0.03,
    "injection_q_std": 0.03,
    "flow_p_std": 0.02,
    "flow_q_std": 0.02,
    "weak_area_buses": [12, 13, 14],
    "weak_area_std_multiplier": 5.0,
}


def test_pypower_real_case_loader_supports_required_cases() -> None:
    expected_bus_counts = {
        "ieee14": 14,
        "ieee30": 30,
        "ieee57": 57,
        "ieee118": 118,
        "ieee300": 300,
    }

    assert set(SUPPORTED_REAL_CASES) == set(expected_bus_counts)
    for case_name, n_buses in expected_bus_counts.items():
        power_case = load_power_case(case_name)
        assert power_case.source == "pypower"
        assert power_case.bus.shape[0] == n_buses
        assert power_case.branch.shape[0] > 0
        assert power_case.gen.shape[0] > 0
        assert power_case.metadata["dataset_source"] == "pypower"


def test_real_ac_case_conversion_and_measurement_dimensions() -> None:
    case = load_ac_case("ieee30", case_source="pypower")

    assert case.source == "pypower"
    assert len(case.buses) == 30
    assert len(case.branches) == 41
    assert ac_state_count("ieee30", case_source="pypower") == 59
    assert (
        ac_measurement_count(
            case_name="ieee30",
            case_source="pypower",
            measurement_config=REAL_MEASUREMENT_CONFIG,
        )
        == 101
    )


def test_real_ac_weighted_system_records_dataset_metadata() -> None:
    system = build_ac_weighted_system(
        case_name="ieee14",
        case_source="pypower",
        linearization_config={
            "angle_perturbation_std": 0.005,
            "voltage_perturbation_std": 0.005,
            "min_voltage_magnitude": 0.5,
        },
        measurement_config=REAL_MEASUREMENT_CONFIG,
        rng=np.random.default_rng(1401),
    )

    assert system.metadata["dataset_source"] == "pypower"
    assert system.metadata["external_case"] is True
    assert system.metadata["case_name"] == "ieee14"
    assert system.n_states == 27
    assert system.n_measurements == 48
    assert np.all(np.isfinite(system.H_tilde))


def test_real_case_config_validation_files() -> None:
    for config_path in sorted(Path("configs").glob("real_ieee*.yaml")):
        if "missing_baselines" in config_path.name:
            continue
        config = load_config(config_path)
        validate_config(config)
        assert config["system"]["case_source"] == "pypower"
        assert config["output"]["run_id"].startswith("real_ieee")
        assert config["output"]["run_id"].endswith("_seed10")
        assert all(len(sweep["seeds"]) >= 10 for sweep in config["sweeps"])


def test_real_case_small_benchmark_writes_outputs(tmp_path: Path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_real_ieee14"
    config["seed"] = 1401
    config["system"]["case_name"] = "ieee14"
    config["system"]["case_source"] = "pypower"
    config["system"]["mode"] = "ac_power_flow_linearized"
    config["system"]["measurement"] = deepcopy(REAL_MEASUREMENT_CONFIG)
    config["system"]["linearization"] = {
        "angle_perturbation_std": 0.005,
        "voltage_perturbation_std": 0.005,
        "min_voltage_magnitude": 0.5,
    }
    config["scenario"] = {
        "name": "test_real_ieee14",
        "noise_std": 0.002,
        "missing_ratio": 0.1,
        "bad_data": {
            "enabled": True,
            "ratio": 0.05,
            "magnitude": 5.0,
            "target": "weak_area",
        },
    }
    config["estimators"] = [
        {"name": "pseudoinverse", "rcond": 1.0e-10},
        {"name": "ridge", "alpha": 1.0e-4},
        {"name": "truncated_svd", "tau": 1.0e-5},
        {"name": "qsvt_regularized", "alpha": 1.0e-4},
        {"name": "huber_irls", "delta": 1.5, "max_iterations": 5, "tolerance": 1.0e-7},
    ]
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [4, 8],
        "grid_size": 128,
        "target_error": 1.0e-3,
    }
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_real_ieee14"
    config["output"]["save_plots"] = False

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")

    assert (output_dir / "config_resolved.yaml").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "estimator_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert set(metrics["dataset_source"]) == {"pypower"}
    assert set(metrics["external_case"]) == {True}
    assert set(metrics["estimator"]) == {
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_regularized",
        "huber_irls",
    }
