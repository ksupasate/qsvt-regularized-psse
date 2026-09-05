from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.utils.config import DEFAULT_CONFIG


def _tiny_iterative_sweep_config(tmp_path):  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_resume_iterative"
    config["seed"] = 123
    config["system"]["mode"] = "nonlinear_ac_state_estimation"
    config["system"]["iteration"] = {
        "max_iterations": 1,
        "update_tolerance": 1.0e-8,
        "residual_tolerance": 1.0e-8,
        "damping": 1.0,
        "max_update_norm": 1000.0,
        "residual_growth_limit": 10000.0,
    }
    config["scenario"]["noise_std"] = 0.0
    config["scenario"]["missing_ratio"] = 0.0
    config["scenario"]["bad_data"] = {
        "enabled": False,
        "ratio": 0.0,
        "magnitude": 10.0,
        "target": "random",
    }
    config["estimators"] = [{"name": "ridge", "alpha": 1.0e-6}]
    config["sweeps"] = [
        {
            "name": "resume_noise_sweep",
            "parameter": "scenario.noise_std",
            "values": [0.0],
            "seeds": [11, 12],
        }
    ]
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_resume_iterative"
    config["output"]["save_plots"] = False
    config["output"]["overwrite"] = True
    return config


def test_iterative_sweep_writes_checkpoint_and_resume_skips_completed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _tiny_iterative_sweep_config(tmp_path)

    first = run_experiment(config)
    output_dir = first["output_dir"]
    assert (output_dir / "trial_results.jsonl").is_file()
    assert (output_dir / "checkpoint_state.json").is_file()
    assert (output_dir / "progress.log").is_file()

    with (output_dir / "trial_results.jsonl").open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    assert len(records) == 2
    assert {record["status"] for record in records} == {"completed"}

    resumed = run_experiment(config, resume=True)
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    with (output_dir / "checkpoint_state.json").open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    assert resumed["summary_metrics"].shape[0] == 1
    assert len(aggregate) == 2
    assert checkpoint["status"] == "complete"
    assert checkpoint["skipped_trials"] == 2
