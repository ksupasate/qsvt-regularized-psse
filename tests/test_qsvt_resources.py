from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.qsvt.polynomial import chebyshev_max_error
from robust_qsvt_se.qsvt.resources import estimate_qsvt_resources
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config


def _small_resource_config(tmp_path) -> dict:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_qsvt_resources"
    config["system"]["n_states"] = 6
    config["system"]["n_measurements"] = 18
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_qsvt_resources"
    config["output"]["save_plots"] = False
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [2, 4],
        "grid_size": 64,
        "target_error": 1.0e12,
    }
    return config


def test_chebyshev_approximation_error_does_not_materially_worsen() -> None:
    low_degree = chebyshev_max_error(
        alpha=0.1,
        block_encoding_normalization=1.0,
        degree=4,
        domain_min=0.05,
        domain_max=1.0,
        grid_size=256,
    )
    high_degree = chebyshev_max_error(
        alpha=0.1,
        block_encoding_normalization=1.0,
        degree=16,
        domain_min=0.05,
        domain_max=1.0,
        grid_size=256,
    )

    assert high_degree.max_error <= low_degree.max_error * 1.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": -1},
        {"grid_size": 2},
        {"alpha": 0.0},
        {"block_encoding_normalization": 0.0},
        {"domain_min": 1.0},
    ],
)
def test_chebyshev_invalid_parameters_raise(kwargs: dict[str, float | int]) -> None:
    base = {
        "alpha": 0.1,
        "block_encoding_normalization": 1.0,
        "degree": 4,
        "domain_min": 0.05,
        "domain_max": 1.0,
        "grid_size": 64,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        chebyshev_max_error(**base)


def test_resource_estimates_are_deterministic_and_select_recommended_degree() -> None:
    singular_values = np.array([3.0, 1.0, 0.2])

    first = estimate_qsvt_resources(
        singular_values,
        alpha=0.1,
        degrees=[4, 2, 4],
        grid_size=64,
        target_error=10.0,
    )
    second = estimate_qsvt_resources(
        singular_values,
        alpha=0.1,
        degrees=[4, 2, 4],
        grid_size=64,
        target_error=10.0,
    )

    assert [row.to_row() for row in first] == [row.to_row() for row in second]
    assert [row.degree for row in first] == [2, 4]
    assert all(np.isfinite(row.max_error) for row in first)
    assert all(row.recommended_degree == 2 for row in first)
    assert all(row.proxy_query_count == 2 * row.degree + 1 for row in first)


def test_resource_estimates_reject_invalid_spectra() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_qsvt_resources(
            np.zeros(3),
            alpha=0.1,
            degrees=[2],
            grid_size=32,
            target_error=0.1,
        )


def test_qsvt_resource_config_validation_errors() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["qsvt_resource"] = {
        "enabled": True,
        "degrees": [4],
        "grid_size": 4,
        "target_error": 1.0e-3,
    }

    with pytest.raises(ValueError, match="grid_size"):
        validate_config(config)


def test_single_run_writes_qsvt_resource_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_resource_config(tmp_path)

    run = run_experiment(config)
    output_dir = run["output_dir"]
    resource_frame = pd.read_csv(output_dir / "qsvt_resource_estimates.csv")
    payload = json.loads((output_dir / "estimator_results.json").read_text())
    qsvt_result = next(
        result for result in payload["results"] if result["name"] == "qsvt_regularized"
    )

    assert len(resource_frame) == len(config["qsvt_resource"]["degrees"])
    assert {"degree", "max_error", "proxy_query_count", "notes"}.issubset(resource_frame.columns)
    assert qsvt_result["extra_diagnostics"]["resource_estimation_scope"] == "single_run"
    assert qsvt_result["extra_diagnostics"]["recommended_polynomial_degree"] == 2


def test_sweep_run_writes_trial_level_qsvt_resource_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _small_resource_config(tmp_path)
    config["output"]["run_id"] = "test_qsvt_resource_sweep"
    config["sweeps"] = [
        {
            "name": "noise_sweep",
            "parameter": "scenario.noise_std",
            "values": [0.0, 0.01],
            "seeds": [11],
        }
    ]

    run = run_experiment(config)
    output_dir = run["output_dir"]
    resource_frame = pd.read_csv(output_dir / "qsvt_resource_estimates.csv")
    payload = json.loads((output_dir / "trial_results.json").read_text())

    assert resource_frame["trial_id"].nunique() == 2
    assert len(resource_frame) == 2 * len(config["qsvt_resource"]["degrees"])
    assert set(resource_frame["resource_estimation_scope"]) == {"sweep_trial"}
    for trial in payload["trials"]:
        qsvt_result = next(
            result for result in trial["results"] if result["name"] == "qsvt_regularized"
        )
        assert qsvt_result["extra_diagnostics"]["resource_estimation_scope"] == "sweep_trial"


def test_run_without_qsvt_resource_block_does_not_write_resource_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config.pop("qsvt_resource", None)
    config["system"]["n_states"] = 6
    config["system"]["n_measurements"] = 18
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_without_qsvt_resource"
    config["output"]["save_plots"] = False

    run = run_experiment(config)

    assert not (run["output_dir"] / "qsvt_resource_estimates.csv").exists()
