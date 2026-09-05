from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config


def test_smoke_experiment_writes_expected_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_smoke"
    config["output"]["save_plots"] = False
    config["system"]["n_states"] = 6
    config["system"]["n_measurements"] = 18

    run = run_experiment(config)
    output_dir = run["output_dir"]

    assert (output_dir / "config_resolved.yaml").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "estimator_results.json").is_file()
    assert (output_dir / "singular_values.csv").is_file()
    assert (output_dir / "run.log").is_file()

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert set(metrics["estimator"]) == {
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_regularized",
    }
    assert not metrics["failed"].any()


def test_config_validation_rejects_invalid_estimator() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["estimators"] = [{"name": "not_real"}]

    with pytest.raises(ValueError, match="unknown estimator"):
        validate_config(config)
