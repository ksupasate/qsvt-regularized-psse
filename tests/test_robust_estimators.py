from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.lav import LAVEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.measurement.linear_system import WeightedSystem
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

ROBUST_ESTIMATORS = [
    {"name": "pseudoinverse", "rcond": 1.0e-12},
    {"name": "ridge", "alpha": 1.0e-4},
    {"name": "truncated_svd", "tau": 1.0e-5},
    {"name": "qsvt_regularized", "alpha": 1.0e-4},
    {"name": "huber_irls", "delta": 1.5, "max_iterations": 50, "tolerance": 1.0e-8},
    {"name": "lav"},
]


def _clean_system() -> WeightedSystem:
    H_tilde = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [2.0, 1.0],
            [-1.0, 2.0],
        ]
    )
    x_true = np.array([1.0, -2.0])
    return WeightedSystem(H_tilde=H_tilde, r_tilde=H_tilde @ x_true, x_true=x_true)


def _outlier_system() -> WeightedSystem:
    system = _clean_system()
    r_tilde = system.r_tilde.copy()
    r_tilde[2] += 20.0
    return WeightedSystem(H_tilde=system.H_tilde, r_tilde=r_tilde, x_true=system.x_true)


def _robust_bad_data_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_robust_bad_data"
    config["seed"] = 901
    config["system"]["mode"] = "ac_power_flow_linearized"
    config["system"]["measurement"] = deepcopy(AC_MEASUREMENT_CONFIG)
    config["system"]["linearization"] = {
        "angle_perturbation_std": 0.01,
        "voltage_perturbation_std": 0.01,
        "min_voltage_magnitude": 0.5,
    }
    config["scenario"]["name"] = "test_robust_bad_data"
    config["scenario"]["noise_std"] = 0.001
    config["scenario"]["missing_ratio"] = 0.1
    config["scenario"]["bad_data"] = {
        "enabled": True,
        "ratio": 0.1,
        "magnitude": 5.0,
        "target": "weak_area",
    }
    config["estimators"] = deepcopy(ROBUST_ESTIMATORS)
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [2, 4],
        "grid_size": 64,
        "target_error": 1.0e6,
    }
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_robust_bad_data"
    config["output"]["save_plots"] = False
    return config


def _iterative_robust_bad_data_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = _robust_bad_data_config(tmp_path)
    config["run_name"] = "test_iterative_robust_bad_data"
    config["seed"] = 910
    config["system"]["mode"] = "ac_iterative_state_estimation"
    config["system"]["iteration"] = {
        "max_iterations": 8,
        "update_tolerance": 1.0e-8,
        "residual_tolerance": 1.0e-8,
        "damping": 1.0,
    }
    config["scenario"]["missing_ratio"] = 0.0
    config["scenario"]["bad_data"]["ratio"] = 0.05
    config["output"]["run_id"] = "test_iterative_robust_bad_data"
    return config


def test_huber_irls_recovers_clean_overdetermined_system() -> None:
    result = HuberIRLSEstimator(delta=1.5, max_iterations=50, tolerance=1.0e-10).solve(
        _clean_system()
    )

    assert not result.failed
    assert result.converged
    np.testing.assert_allclose(result.x_hat, _clean_system().x_true, atol=1.0e-9)
    assert result.extra_diagnostics["irls_converged"]


def test_huber_irls_is_less_sensitive_than_pseudoinverse_to_gross_outlier() -> None:
    system = _outlier_system()

    pinv = PseudoinverseEstimator(rcond=0.0).solve(system)
    huber = HuberIRLSEstimator(delta=1.0, max_iterations=50, tolerance=1.0e-10).solve(system)

    assert huber.rmse is not None
    assert pinv.rmse is not None
    assert huber.rmse < pinv.rmse
    assert huber.extra_diagnostics["min_weight"] < 1.0


def test_lav_solves_clean_system_and_resists_gross_outlier() -> None:
    clean = LAVEstimator().solve(_clean_system())
    outlier = LAVEstimator().solve(_outlier_system())

    assert not clean.failed
    assert not outlier.failed
    np.testing.assert_allclose(clean.x_hat, _clean_system().x_true, atol=1.0e-9)
    np.testing.assert_allclose(outlier.x_hat, _outlier_system().x_true, atol=1.0e-9)
    assert outlier.extra_diagnostics["objective_value"] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "estimator_config",
    [
        {"name": "huber_irls", "delta": 0.0, "max_iterations": 50, "tolerance": 1.0e-8},
        {"name": "huber_irls", "delta": 1.0, "max_iterations": 0, "tolerance": 1.0e-8},
        {"name": "huber_irls", "delta": 1.0, "max_iterations": 50, "tolerance": 0.0},
    ],
)
def test_invalid_huber_config_fails_validation(estimator_config: dict[str, object]) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["estimators"] = [estimator_config]

    with pytest.raises(ValueError, match="huber_irls"):
        validate_config(config)


def test_lav_failure_path_returns_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from robust_qsvt_se.estimators import lav

    monkeypatch.setattr(
        lav,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=2,
            message="synthetic LP failure",
            fun=None,
            x=None,
        ),
    )

    result = LAVEstimator().solve(_clean_system())

    assert result.failed
    assert result.failure_reason == "synthetic LP failure"
    assert result.extra_diagnostics["linprog_status"] == 2


def test_robust_bad_data_smoke_writes_all_estimator_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _robust_bad_data_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")

    assert (output_dir / "estimator_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "run.log").is_file()
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert set(metrics["estimator"]) == {item["name"] for item in ROBUST_ESTIMATORS}
    assert {"huber_irls", "lav"}.issubset(set(metrics["estimator"]))


def test_robust_bad_data_sweep_writes_robust_estimator_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _robust_bad_data_config(tmp_path)
    config["output"]["run_id"] = "test_robust_bad_data_sweep"
    config["sweeps"] = [
        {
            "name": "bad_data_ratio_sweep",
            "parameter": "scenario.bad_data.ratio",
            "values": [0.0, 0.1],
            "seeds": [41, 42],
        }
    ]

    run = run_experiment(config)
    output_dir = run["output_dir"]
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")

    assert (output_dir / "summary_metrics.csv").is_file()
    assert (output_dir / "trial_results.json").is_file()
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
    assert {"huber_irls", "lav"}.issubset(set(aggregate["estimator"]))
    assert len(aggregate) == 2 * 2 * len(config["estimators"])


def test_iterative_ac_robust_bad_data_smoke_runs_all_update_solvers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _iterative_robust_bad_data_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    metrics = pd.read_csv(output_dir / "metrics.csv")
    trace = pd.read_csv(output_dir / "iteration_trace.csv")

    assert set(metrics["estimator"]) == {item["name"] for item in ROBUST_ESTIMATORS}
    assert {"huber_irls", "lav"}.issubset(set(trace["estimator"]))
    assert (output_dir / "qsvt_resource_estimates.csv").is_file()
