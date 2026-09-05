from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from robust_qsvt_se.data.cases import ieee14_dc_case
from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.measurement.dc_linear import (
    build_dc_measurement_matrix,
    build_ieee14_dc_weighted_system,
)
from robust_qsvt_se.utils.config import DEFAULT_CONFIG, validate_config


def test_ieee14_dc_branch_flow_row_uses_slack_reference() -> None:
    case = ieee14_dc_case()
    H, rows = build_dc_measurement_matrix(
        case=case,
        measurement_config={
            "include_branch_flows": True,
            "include_bus_injections": False,
            "angle_buses": [],
        },
    )
    state_index = {bus: index for index, bus in enumerate(case.state_buses)}
    first_branch = case.branches[0]

    assert rows[0].label == "P_1_2"
    assert H[0, state_index[2]] == -first_branch.susceptance
    assert np.count_nonzero(H[0]) == 1


def test_ieee14_dc_weighted_system_is_finite_and_structured() -> None:
    rng = np.random.default_rng(123)
    system = build_ieee14_dc_weighted_system(
        angle_scale=0.05,
        measurement_config={
            "include_branch_flows": True,
            "include_bus_injections": True,
            "angle_buses": [2, 6, 9, 14],
            "flow_std": 0.02,
            "injection_std": 0.03,
            "angle_std": 0.005,
            "weak_area_buses": [12, 13, 14],
            "weak_area_std_multiplier": 30.0,
        },
        rng=rng,
    )

    assert system.metadata["mode"] == "dc_power_flow_linearized"
    assert system.n_states == 13
    assert system.n_measurements == 37
    assert np.isfinite(system.condition_number())
    assert system.rank() == 13
    assert "theta_14" in system.metadata["measurement_labels"]


def test_dc_config_validation_and_smoke_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = deepcopy(DEFAULT_CONFIG)
    config["run_name"] = "test_dc"
    config["seed"] = 321
    config["system"]["mode"] = "dc_power_flow_linearized"
    config["scenario"]["name"] = "dc_weak_observability"
    config["scenario"]["missing_ratio"] = 0.1
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_id"] = "test_dc"
    config["output"]["save_plots"] = False

    validate_config(config)
    run = run_experiment(config)
    metrics = pd.read_csv(run["output_dir"] / "metrics.csv")

    assert run["system"].metadata["mode"] == "dc_power_flow_linearized"
    assert set(metrics["mode"]) == {"dc_power_flow_linearized"}
    assert set(metrics["estimator"]) == {
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_regularized",
    }
    assert not metrics["failed"].any()
