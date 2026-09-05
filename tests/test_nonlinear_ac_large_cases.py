from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.utils.config import load_config, validate_config


def test_nonlinear_large_case_configs_validate() -> None:
    for path in (
        Path("configs/nonlinear_ac_ieee118_seed10.yaml"),
        Path("configs/nonlinear_ac_ieee300_seed10.yaml"),
    ):
        config = load_config(path)
        validate_config(config)
        assert config["system"]["case_source"] == "pypower"
        assert config["system"]["mode"] == "nonlinear_ac_state_estimation"
        assert all(len(sweep["seeds"]) == 10 for sweep in config["sweeps"])


def test_reduced_nonlinear_large_case_run_records_status(tmp_path: Path) -> None:
    config = load_config("configs/nonlinear_ac_ieee118_seed10.yaml")
    config = deepcopy(config)
    config["run_name"] = "test_nonlinear_ieee118_reduced"
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_nonlinear_ieee118_reduced"
    config["output"]["save_plots"] = False
    config["sweeps"] = [
        {
            "name": "test_noise_sweep",
            "parameter": "scenario.noise_std",
            "values": [0.0],
            "seeds": [101],
        }
    ]
    config["estimators"] = [{"name": "ridge", "alpha": 1.0e-4}]
    config["system"]["iteration"]["max_iterations"] = 2
    config["qsvt_resource"]["enabled"] = False

    run = run_experiment(config)
    output_dir = run["output_dir"]
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    trace = pd.read_csv(output_dir / "iteration_trace.csv")

    assert (output_dir / "summary_metrics.csv").is_file()
    assert (output_dir / "trial_results.json").is_file()
    assert {"converged", "failed", "failure_reason", "iterations"}.issubset(aggregate.columns)
    assert not trace.empty
