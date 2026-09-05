from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.measurement.dc_linear import build_dc_weighted_system
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.measurement.perturbations import (
    add_bad_data_outliers,
    add_gaussian_noise,
    remove_random_rows,
)


def build_synthetic_weighted_system(
    *,
    n_measurements: int,
    n_states: int,
    condition_number: float,
    truth_scale: float,
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> WeightedSystem:
    if n_measurements < n_states:
        raise ValueError("n_measurements must be at least n_states")
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least 1")

    left_random = rng.normal(size=(n_measurements, n_states))
    right_random = rng.normal(size=(n_states, n_states))
    U, _ = np.linalg.qr(left_random, mode="reduced")
    V, _ = np.linalg.qr(right_random)
    singular_values = np.geomspace(1.0, 1.0 / condition_number, num=n_states)
    H_tilde = U @ np.diag(singular_values) @ V.T
    x_true = rng.normal(loc=0.0, scale=truth_scale, size=n_states)
    r_tilde = H_tilde @ x_true

    system_metadata = {
        "case_name": "ieee14",
        "dataset_source": "synthetic_generated",
        "dataset_source_detail": "controlled random SVD construction",
        "external_case": False,
        "mode": "synthetic_linearized",
        "note": "Milestone 1 uses a synthetic weighted linearized system, not AC IEEE14.",
        "target_condition_number": float(condition_number),
        "achieved_condition_number": float(np.linalg.cond(H_tilde)),
        "n_measurements_original": int(n_measurements),
        "n_states": int(n_states),
    }
    if metadata:
        system_metadata.update(metadata)
    return WeightedSystem(H_tilde=H_tilde, r_tilde=r_tilde, x_true=x_true, metadata=system_metadata)


def build_system_from_config(config: dict[str, Any], rng: np.random.Generator) -> WeightedSystem:
    system_config = config["system"]
    scenario_config = config["scenario"]
    metadata = {
        "case_name": system_config.get("case_name", "ieee14"),
        "case_source": system_config.get("case_source", "synthetic_generated"),
        "scenario_name": scenario_config.get("name", "controlled_ill_conditioning"),
        "seed": int(config["seed"]),
    }
    mode = system_config.get("mode", "synthetic_linearized")
    if mode == "synthetic_linearized":
        system = build_synthetic_weighted_system(
            n_measurements=int(system_config["n_measurements"]),
            n_states=int(system_config["n_states"]),
            condition_number=float(system_config["condition_number"]),
            truth_scale=float(system_config.get("truth_scale", 1.0)),
            rng=rng,
            metadata=metadata,
        )
    elif mode == "dc_power_flow_linearized":
        system = build_dc_weighted_system(
            case_name=str(system_config.get("case_name", "ieee14")),
            case_source=str(system_config.get("case_source", "builtin")),
            angle_scale=float(system_config.get("angle_scale", 0.05)),
            measurement_config=dict(system_config.get("measurement", {})),
            rng=rng,
            metadata=metadata,
        )
    elif mode == "ac_power_flow_linearized":
        system = build_ac_weighted_system(
            case_name=str(system_config.get("case_name", "ieee14")),
            case_source=str(system_config.get("case_source", "builtin")),
            linearization_config=dict(system_config.get("linearization", {})),
            measurement_config=dict(system_config.get("measurement", {})),
            rng=rng,
            metadata=metadata,
        )
    else:
        raise ValueError(f"unsupported system mode: {mode}")

    system = add_gaussian_noise(
        system,
        noise_std=float(scenario_config.get("noise_std", 0.0)),
        rng=rng,
    )
    system = remove_random_rows(
        system,
        missing_ratio=float(scenario_config.get("missing_ratio", 0.0)),
        rng=rng,
    )
    system = add_bad_data_outliers(
        system,
        bad_data_config=scenario_config.get("bad_data", {}),
        rng=rng,
    )
    return system
