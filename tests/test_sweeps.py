from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.experiments.sweeps import (
    generate_sweep_trials,
    get_by_dot_path,
    set_by_dot_path,
    summarize_sweep_metrics,
)
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config


def _small_sweep_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_sweeps"
    config["system"]["n_states"] = 6
    config["system"]["n_measurements"] = 18
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_sweeps"
    config["output"]["save_plots"] = False
    config["sweeps"] = [
        {
            "name": "noise_sweep",
            "parameter": "scenario.noise_std",
            "values": [0.0, 0.01],
            "seeds": [11, 12],
        },
        {
            "name": "missing_sweep",
            "parameter": "scenario.missing_ratio",
            "values": [0.0, 0.1],
            "seeds": [11],
        },
    ]
    return config


def test_dot_path_getter_and_setter() -> None:
    config = deepcopy(DEFAULT_CONFIG)

    assert get_by_dot_path(config, "scenario.noise_std") == 0.01
    set_by_dot_path(config, "scenario.noise_std", 0.05)

    assert config["scenario"]["noise_std"] == 0.05


def test_generate_sweep_trials_count(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_sweep_config(tmp_path)

    trials = generate_sweep_trials(config)

    assert len(trials) == 6
    assert {trial.sweep_name for trial in trials} == {"noise_sweep", "missing_sweep"}


def test_invalid_sweep_parameter_path_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_sweep_config(tmp_path)
    config["sweeps"][0]["parameter"] = "scenario.not_real"

    with pytest.raises(ValueError, match="path does not exist"):
        validate_config(config)


def test_invalid_generated_trial_config_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_sweep_config(tmp_path)
    config["sweeps"] = [
        {
            "name": "bad_missing_sweep",
            "parameter": "scenario.missing_ratio",
            "values": [0.95],
            "seeds": [11],
        }
    ]

    with pytest.raises(ValueError, match=r"removes too many rows|removes all"):
        generate_sweep_trials(config)


def test_sweep_run_writes_expected_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_sweep_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]

    assert (output_dir / "config_resolved.yaml").is_file()
    assert (output_dir / "aggregate_metrics.csv").is_file()
    assert (output_dir / "summary_metrics.csv").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "trial_results.json").is_file()
    assert (output_dir / "run.log").is_file()

    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    summary = pd.read_csv(output_dir / "summary_metrics.csv")
    assert len(aggregate) == 6 * len(config["estimators"])
    assert {
        "failure_rate",
        "rmse_mean",
        "rmse_std",
        "rmse_median",
        "rmse_q1",
        "rmse_q3",
        "rmse_iqr",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "weighted_residual_norm_mean",
        "weighted_residual_quadratic_mean",
        "n_trials",
        "n_successful_trials",
        "n_failed_trials",
    }.issubset(summary.columns)
    assert set(aggregate["estimator"]) == {
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_regularized",
    }


def test_sweep_summary_clips_nonnegative_ci_lower_bounds() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "trial_id": "trial_1",
                "sweep_name": "condition",
                "sweep_parameter": "system.condition_number",
                "sweep_value": 1.0e8,
                "estimator": "pseudoinverse",
                "rmse": 0.01,
                "weighted_residual_norm": 0.01,
                "weighted_residual_quadratic": 0.0001,
                "condition_number": 1.0e8,
                "runtime_seconds": 0.01,
                "failed": False,
            },
            {
                "trial_id": "trial_2",
                "sweep_name": "condition",
                "sweep_parameter": "system.condition_number",
                "sweep_value": 1.0e8,
                "estimator": "pseudoinverse",
                "rmse": 100.0,
                "weighted_residual_norm": 100.0,
                "weighted_residual_quadratic": 10000.0,
                "condition_number": 1.0e8,
                "runtime_seconds": 0.02,
                "failed": False,
            },
        ]
    )

    summary = summarize_sweep_metrics(aggregate)

    row = summary.iloc[0]
    assert row["rmse_ci95_low"] == 0.0
    assert row["weighted_residual_norm_ci95_low"] == 0.0
    assert row["weighted_residual_quadratic_ci95_low"] == 0.0
