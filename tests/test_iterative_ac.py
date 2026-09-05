from __future__ import annotations

from copy import deepcopy
from logging import getLogger

import pandas as pd
import pytest

from robust_qsvt_se.experiments.iterative_ac import (
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
    run_iterative_estimators,
)
from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config


def _iterative_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_iterative_ac"
    config["seed"] = 601
    config["system"]["mode"] = "ac_iterative_state_estimation"
    config["system"]["measurement"] = {
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
    config["system"]["linearization"] = {
        "angle_perturbation_std": 0.01,
        "voltage_perturbation_std": 0.01,
        "min_voltage_magnitude": 0.5,
    }
    config["system"]["iteration"] = {
        "max_iterations": 10,
        "update_tolerance": 1.0e-8,
        "residual_tolerance": 1.0e-8,
        "damping": 1.0,
    }
    config["scenario"]["name"] = "test_iterative_ac"
    config["scenario"]["noise_std"] = 0.0
    config["scenario"]["missing_ratio"] = 0.0
    config["estimators"] = [
        {"name": "pseudoinverse", "rcond": 1.0e-12},
        {"name": "ridge", "alpha": 1.0e-6},
        {"name": "truncated_svd", "tau": 1.0e-8},
        {"name": "qsvt_regularized", "alpha": 1.0e-6},
    ]
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_iterative_ac"
    config["output"]["save_plots"] = False
    return config


def test_iterative_ac_reduces_weighted_residual(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["estimators"] = [{"name": "pseudoinverse", "rcond": 1.0e-12}]
    problem = build_ac_nonlinear_problem(config)

    initial_residual = _weighted_residual_norm(problem, problem.initial_state)
    results, trace = run_iterative_estimators(
        config=config,
        problem=problem,
        logger=getLogger("test_iterative_ac"),
    )

    assert results[0].weighted_residual < initial_residual
    assert results[0].converged
    assert not results[0].failed
    assert not trace.empty


def test_all_iterative_ac_estimators_return_structured_results(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    problem = build_ac_nonlinear_problem(config)

    results, trace = run_iterative_estimators(
        config=config,
        problem=problem,
        logger=getLogger("test_iterative_ac"),
    )

    assert {result.name for result in results} == {
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_regularized",
    }
    assert all(result.iterations >= 1 for result in results)
    assert {"weighted_residual_before", "weighted_residual_after", "update_norm"}.issubset(
        trace.columns
    )


def test_iterative_config_validation_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["system"]["iteration"]["damping"] = 0.0

    with pytest.raises(ValueError, match="damping"):
        validate_config(config)


def test_nonlinear_ac_alias_dispatch_writes_mode_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["system"]["mode"] = "nonlinear_ac_state_estimation"
    config["estimators"] = [{"name": "pseudoinverse", "rcond": 1.0e-12}]
    config["output"]["run_id"] = "test_nonlinear_ac_alias"

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")

    assert set(metrics["mode"]) == {"nonlinear_ac_state_estimation"}
    assert (output_dir / "iteration_trace.csv").is_file()


def test_iterative_ac_detects_excessive_update_norm(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["system"]["iteration"]["max_update_norm"] = 1.0e-12
    config["estimators"] = [{"name": "pseudoinverse", "rcond": 1.0e-12}]
    problem = build_ac_nonlinear_problem(config)

    results, trace = run_iterative_estimators(
        config=config,
        problem=problem,
        logger=getLogger("test_iterative_ac"),
    )

    assert results[0].failed
    assert "max_update_norm" in str(results[0].failure_reason)
    assert bool(trace.iloc[0]["failed"])


def test_iterative_ac_smoke_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [2, 4],
        "grid_size": 64,
        "target_error": 10.0,
    }

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")
    trace = pd.read_csv(output_dir / "iteration_trace.csv")

    assert (output_dir / "config_resolved.yaml").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "estimator_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "iteration_trace.csv").is_file()
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert {"iterations", "converged", "angle_rmse", "voltage_magnitude_rmse"}.issubset(
        metrics.columns
    )
    assert not trace.empty


def test_iterative_ac_sweep_writes_aggregate_outputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_config(tmp_path)
    config["output"]["run_id"] = "test_iterative_ac_sweep"
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [2, 4],
        "grid_size": 64,
        "target_error": 10.0,
    }
    config["sweeps"] = [
        {
            "name": "initial_angle_sweep",
            "parameter": "system.linearization.angle_perturbation_std",
            "values": [0.002, 0.01],
            "seeds": [41, 42],
        }
    ]

    run = run_experiment(config)
    output_dir = run["output_dir"]
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    trace = pd.read_csv(output_dir / "iteration_trace.csv")
    resource_frame = pd.read_csv(output_dir / "qsvt_resource_estimates.csv")

    assert len(aggregate) == 2 * 2 * len(config["estimators"])
    assert (output_dir / "summary_metrics.csv").is_file()
    assert (output_dir / "trial_results.json").is_file()
    assert resource_frame["trial_id"].nunique() == 4
    assert set(resource_frame["resource_estimation_scope"]) == {"iterative_sweep_final_spectrum"}
    assert not trace.empty
