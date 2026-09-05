from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robust_qsvt_se.data.cases import (
    ac_measurement_count,
    ac_state_count,
    dc_measurement_count,
    supported_case_names,
)

VALID_ESTIMATORS = {
    "pseudoinverse",
    "normal_equation_wls",
    "ridge",
    "truncated_svd",
    "qsvt_regularized",
    "qsvt_unregularized_inverse",
    "hhl_style_inverse_proxy",
    "huber_irls",
    "lav",
}
VALID_SYSTEM_MODES = {
    "synthetic_linearized",
    "dc_power_flow_linearized",
    "ac_power_flow_linearized",
    "ac_iterative_state_estimation",
    "nonlinear_ac_state_estimation",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "ieee14_spectral_smoke",
    "seed": 123,
    "system": {
        "case_name": "ieee14",
        "mode": "synthetic_linearized",
        "n_states": 20,
        "n_measurements": 60,
        "condition_number": 1.0e6,
        "truth_scale": 1.0,
        "angle_scale": 0.05,
        "measurement": {
            "include_branch_flows": True,
            "include_bus_injections": True,
            "angle_buses": [2, 6, 9, 14],
            "flow_std": 0.02,
            "injection_std": 0.03,
            "angle_std": 0.005,
            "weak_area_buses": [12, 13, 14],
            "weak_area_std_multiplier": 30.0,
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
        },
        "linearization": {
            "angle_perturbation_std": 0.01,
            "voltage_perturbation_std": 0.01,
            "min_voltage_magnitude": 0.5,
        },
        "iteration": {
            "max_iterations": 10,
            "update_tolerance": 1.0e-8,
            "residual_tolerance": 1.0e-8,
            "damping": 1.0,
        },
    },
    "scenario": {
        "name": "controlled_ill_conditioning",
        "noise_std": 0.01,
        "missing_ratio": 0.1,
        "bad_data": {
            "enabled": False,
            "ratio": 0.0,
            "magnitude": 10.0,
            "target": "random",
        },
    },
    "estimators": [
        {"name": "pseudoinverse", "rcond": 1.0e-12},
        {"name": "ridge", "alpha": 1.0e-3},
        {"name": "truncated_svd", "tau": 1.0e-4},
        {"name": "qsvt_regularized", "alpha": 1.0e-3},
    ],
    "qsvt_resource": {
        "enabled": False,
        "degrees": [4, 8, 16, 32],
        "grid_size": 512,
        "target_error": 1.0e-3,
    },
    "output": {
        "root": "outputs",
        "run_id": "ieee14_spectral_smoke_seed123",
        "save_plots": True,
        "overwrite": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping")
    config = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("run_name"), str) or not config["run_name"]:
        raise ValueError("run_name must be a non-empty string")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")

    system = _require_mapping(config, "system")
    mode = system.get("mode")
    if mode not in VALID_SYSTEM_MODES:
        raise ValueError(f"system.mode must be one of {sorted(VALID_SYSTEM_MODES)}")
    n_states, n_measurements = _validate_system_config(system)

    scenario = _require_mapping(config, "scenario")
    noise_std = float(scenario.get("noise_std", 0.0))
    if noise_std < 0.0:
        raise ValueError("scenario.noise_std must be nonnegative")
    missing_ratio = float(scenario.get("missing_ratio", 0.0))
    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError("scenario.missing_ratio must satisfy 0 <= missing_ratio < 1")
    remaining_rows = n_measurements - round(n_measurements * missing_ratio)
    if remaining_rows <= 0:
        raise ValueError("missing_ratio removes all measurement rows")
    row_limited_modes = {
        "synthetic_linearized",
        "ac_power_flow_linearized",
        "ac_iterative_state_estimation",
        "nonlinear_ac_state_estimation",
    }
    if mode in row_limited_modes and remaining_rows < n_states:
        raise ValueError("missing_ratio removes too many rows for this benchmark")

    estimators = config.get("estimators")
    if not isinstance(estimators, list) or not estimators:
        raise ValueError("estimators must be a non-empty list")
    for estimator in estimators:
        if not isinstance(estimator, dict):
            raise ValueError("each estimator config must be a mapping")
        name = estimator.get("name")
        if name not in VALID_ESTIMATORS:
            raise ValueError(f"unknown estimator name: {name}")
        if name in {"ridge", "qsvt_regularized"} and float(estimator.get("alpha", 0.0)) <= 0.0:
            raise ValueError(f"{name}.alpha must be positive")
        if name == "truncated_svd" and float(estimator.get("tau", -1.0)) < 0.0:
            raise ValueError("truncated_svd.tau must be nonnegative")
        if name == "pseudoinverse" and float(estimator.get("rcond", 0.0)) < 0.0:
            raise ValueError("pseudoinverse.rcond must be nonnegative")
        if (
            name == "normal_equation_wls"
            and float(estimator.get("max_gain_condition_number", float("inf"))) <= 0.0
        ):
            raise ValueError("normal_equation_wls.max_gain_condition_number must be positive")
        if name == "qsvt_unregularized_inverse" and float(estimator.get("cutoff", 1.0e-8)) <= 0.0:
            raise ValueError("qsvt_unregularized_inverse.cutoff must be positive")
        if name == "hhl_style_inverse_proxy":
            if float(estimator.get("cutoff", 1.0e-8)) <= 0.0:
                raise ValueError("hhl_style_inverse_proxy.cutoff must be positive")
            if float(estimator.get("precision", 1.0e-3)) <= 0.0:
                raise ValueError("hhl_style_inverse_proxy.precision must be positive")
            if float(estimator.get("instability_condition_threshold", 1.0e8)) <= 0.0:
                raise ValueError(
                    "hhl_style_inverse_proxy.instability_condition_threshold must be positive"
                )
        if name == "huber_irls":
            if float(estimator.get("delta", 0.0)) <= 0.0:
                raise ValueError("huber_irls.delta must be positive")
            max_iterations = estimator.get("max_iterations", 50)
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise ValueError("huber_irls.max_iterations must be a positive integer")
            if float(estimator.get("tolerance", 0.0)) <= 0.0:
                raise ValueError("huber_irls.tolerance must be positive")

    output = _require_mapping(config, "output")
    if not isinstance(output.get("root"), str) or not output["root"]:
        raise ValueError("output.root must be a non-empty string")
    if "run_id" in output and (not isinstance(output["run_id"], str) or not output["run_id"]):
        raise ValueError("output.run_id must be a non-empty string when provided")

    _validate_sweeps(config)
    _validate_qsvt_resource(config)
    _validate_bad_data(config)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _validate_system_config(system: dict[str, Any]) -> tuple[int, int]:
    mode = system["mode"]
    if mode == "synthetic_linearized":
        n_states = _require_positive_int(system, "n_states")
        n_measurements = _require_positive_int(system, "n_measurements")
        if n_measurements < n_states:
            raise ValueError("n_measurements must be at least n_states for the synthetic benchmark")
        condition_number = float(system.get("condition_number", 1.0))
        if condition_number < 1.0:
            raise ValueError("system.condition_number must be at least 1")
        truth_scale = float(system.get("truth_scale", 1.0))
        if truth_scale <= 0.0:
            raise ValueError("system.truth_scale must be positive")
        return n_states, n_measurements

    if mode == "dc_power_flow_linearized":
        case_name = str(system.get("case_name", "ieee14"))
        case_source = str(system.get("case_source", "builtin"))
        if case_name.lower() not in supported_case_names(case_source):
            raise ValueError(
                f"DC mode with case_source={case_source!r} supports only "
                f"{sorted(supported_case_names(case_source))}"
            )
        angle_scale = float(system.get("angle_scale", 0.05))
        if angle_scale <= 0.0:
            raise ValueError("system.angle_scale must be positive")
        measurement = _require_mapping(system, "measurement")
        flow_std = float(measurement.get("flow_std", 0.02))
        injection_std = float(measurement.get("injection_std", 0.03))
        angle_std = float(measurement.get("angle_std", 0.005))
        if flow_std <= 0.0 or injection_std <= 0.0 or angle_std <= 0.0:
            raise ValueError("DC measurement standard deviations must be positive")
        weak_multiplier = float(measurement.get("weak_area_std_multiplier", 1.0))
        if weak_multiplier <= 0.0:
            raise ValueError("weak_area_std_multiplier must be positive")
        n_measurements = dc_measurement_count(
            case_name=case_name,
            case_source=case_source,
            include_branch_flows=bool(measurement.get("include_branch_flows", True)),
            include_bus_injections=bool(measurement.get("include_bus_injections", True)),
            angle_measurement_buses=measurement.get("angle_buses", ()),
        )
        if n_measurements <= 0:
            raise ValueError("DC measurement config must include at least one row")
        return 13, n_measurements

    if mode in {
        "ac_power_flow_linearized",
        "ac_iterative_state_estimation",
        "nonlinear_ac_state_estimation",
    }:
        case_name = str(system.get("case_name", "ieee14"))
        case_source = str(system.get("case_source", "builtin"))
        if case_name.lower() not in supported_case_names(case_source):
            raise ValueError(
                f"AC mode with case_source={case_source!r} supports only "
                f"{sorted(supported_case_names(case_source))}"
            )
        measurement = _require_mapping(system, "measurement")
        if not any(
            bool(measurement.get(key, True))
            for key in (
                "include_voltage_magnitudes",
                "include_p_injections",
                "include_q_injections",
                "include_p_branch_flows",
                "include_q_branch_flows",
            )
        ):
            raise ValueError("AC measurement config must include at least one measurement group")
        for key in (
            "voltage_std",
            "injection_p_std",
            "injection_q_std",
            "flow_p_std",
            "flow_q_std",
        ):
            if float(measurement.get(key, 0.0)) <= 0.0:
                raise ValueError(f"system.measurement.{key} must be positive")
        weak_multiplier = float(measurement.get("weak_area_std_multiplier", 1.0))
        if weak_multiplier <= 0.0:
            raise ValueError("weak_area_std_multiplier must be positive")
        linearization = _require_mapping(system, "linearization")
        for key in ("angle_perturbation_std", "voltage_perturbation_std"):
            if float(linearization.get(key, -1.0)) < 0.0:
                raise ValueError(f"system.linearization.{key} must be nonnegative")
        if float(linearization.get("min_voltage_magnitude", 0.0)) <= 0.0:
            raise ValueError("system.linearization.min_voltage_magnitude must be positive")
        if mode in {"ac_iterative_state_estimation", "nonlinear_ac_state_estimation"}:
            iteration = _require_mapping(system, "iteration")
            max_iterations = iteration.get("max_iterations")
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise ValueError("system.iteration.max_iterations must be a positive integer")
            for key in ("update_tolerance", "residual_tolerance"):
                if float(iteration.get(key, 0.0)) <= 0.0:
                    raise ValueError(f"system.iteration.{key} must be positive")
            damping = float(iteration.get("damping", 0.0))
            if damping <= 0.0:
                raise ValueError("system.iteration.damping must be positive")
            if float(iteration.get("max_update_norm", 1.0e6)) <= 0.0:
                raise ValueError("system.iteration.max_update_norm must be positive")
            if float(iteration.get("residual_growth_limit", 1.0e6)) <= 1.0:
                raise ValueError("system.iteration.residual_growth_limit must be greater than 1")
        n_states = ac_state_count(case_name, case_source=case_source)
        n_measurements = ac_measurement_count(
            case_name=case_name,
            case_source=case_source,
            measurement_config=measurement,
        )
        if n_measurements <= 0:
            raise ValueError("AC measurement config must include at least one row")
        return n_states, n_measurements

    raise ValueError(f"unsupported system mode: {mode}")


def _validate_sweeps(config: dict[str, Any]) -> None:
    sweeps = config.get("sweeps", [])
    if sweeps in (None, []):
        return
    if not isinstance(sweeps, list):
        raise ValueError("sweeps must be a list when provided")
    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError("each sweep must be a mapping")
        name = sweep.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("each sweep requires a non-empty name")
        parameter = sweep.get("parameter")
        if not isinstance(parameter, str) or not parameter:
            raise ValueError(f"sweep {name} requires a non-empty parameter")
        _get_dot_path(config, parameter)
        linked_parameters = sweep.get("linked_parameters", [])
        if linked_parameters in (None, []):
            linked_parameters = []
        if not isinstance(linked_parameters, list) or not all(
            isinstance(path, str) and path for path in linked_parameters
        ):
            raise ValueError(f"sweep {name} linked_parameters must be a list of paths")
        for linked_parameter in linked_parameters:
            _get_dot_path(config, linked_parameter)
        values = sweep.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep {name} requires a non-empty values list")
        if not all(isinstance(value, int | float) for value in values):
            raise ValueError(f"sweep {name} values must be numeric")
        seeds = sweep.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"sweep {name} requires a non-empty seeds list")
        if not all(isinstance(seed, int) for seed in seeds):
            raise ValueError(f"sweep {name} seeds must be integers")


def _validate_qsvt_resource(config: dict[str, Any]) -> None:
    resource_config = config.get("qsvt_resource")
    if resource_config in (None, {}):
        return
    if not isinstance(resource_config, dict):
        raise ValueError("qsvt_resource must be a mapping when provided")
    enabled = resource_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("qsvt_resource.enabled must be a boolean")
    if not enabled:
        return

    degrees = resource_config.get("degrees")
    if not isinstance(degrees, list) or not degrees:
        raise ValueError("qsvt_resource.degrees must be a non-empty list")
    if not all(isinstance(degree, int) and degree >= 0 for degree in degrees):
        raise ValueError("qsvt_resource.degrees must contain nonnegative integers")
    grid_size = resource_config.get("grid_size")
    if not isinstance(grid_size, int) or grid_size <= max(degrees) + 1:
        raise ValueError("qsvt_resource.grid_size must be greater than max(degrees) + 1")
    if float(resource_config.get("target_error", 0.0)) <= 0.0:
        raise ValueError("qsvt_resource.target_error must be positive")
    if not any(estimator.get("name") == "qsvt_regularized" for estimator in config["estimators"]):
        raise ValueError("qsvt_resource.enabled requires a qsvt_regularized estimator")


def _validate_bad_data(config: dict[str, Any]) -> None:
    scenario = _require_mapping(config, "scenario")
    bad_data = scenario.get("bad_data", {})
    if bad_data in (None, {}):
        return
    if not isinstance(bad_data, dict):
        raise ValueError("scenario.bad_data must be a mapping when provided")
    enabled = bad_data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("scenario.bad_data.enabled must be a boolean")
    ratio = float(bad_data.get("ratio", 0.0))
    if not 0.0 <= ratio < 1.0:
        raise ValueError("scenario.bad_data.ratio must satisfy 0 <= ratio < 1")
    magnitude = float(bad_data.get("magnitude", 10.0))
    if magnitude <= 0.0:
        raise ValueError("scenario.bad_data.magnitude must be positive")
    target = str(bad_data.get("target", "random"))
    if target not in {"random", "weak_area"}:
        raise ValueError("scenario.bad_data.target must be one of ['random', 'weak_area']")
    if enabled and target == "weak_area":
        measurement = _require_mapping(_require_mapping(config, "system"), "measurement")
        weak_area_buses = measurement.get("weak_area_buses", ())
        if not isinstance(weak_area_buses, list | tuple) or not weak_area_buses:
            raise ValueError(
                "scenario.bad_data.target='weak_area' requires system.measurement.weak_area_buses"
            )


def _get_dot_path(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not part:
            raise ValueError(f"sweep parameter path does not exist: {path}")
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        raise ValueError(f"sweep parameter path does not exist: {path}")
    return current
