"""Shared real-computation primitives for the gap-closing phases.

These helpers build controlled IEEE-derived AC-linearized weighted systems, apply
stress, and run the classical estimators, returning real values only. Nothing here
fabricates results: a failed or rank-deficient configuration is reported as such.

The paper-facing estimator names (e.g. ``ridge_tikhonov``) map onto the internal
estimator classes (e.g. ``RidgeEstimator``). ``qsvt_target_classical`` is the QSVT
target filter, which is numerically identical to Ridge/Tikhonov for matched alpha.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from robust_qsvt_se.data.cases import load_ac_case
from robust_qsvt_se.estimators.base import EstimatorResult
from robust_qsvt_se.estimators.hhl_style_inverse_proxy import HHLStyleInverseProxyEstimator
from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.lav import LAVEstimator
from robust_qsvt_se.estimators.normal_equation_wls import NormalEquationWLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.measurement.perturbations import (
    add_bad_data_outliers,
    add_gaussian_noise,
    remove_random_rows,
)

DEFAULT_CASE_SOURCE = "pypower"

DEFAULT_LINEARIZATION: dict[str, float] = {
    "angle_perturbation_std": 0.005,
    "voltage_perturbation_std": 0.005,
    "min_voltage_magnitude": 0.5,
}

# Canonical full AC measurement set and per-type standard deviations (matches the
# nonlinear_ac_* benchmark configs so conditioning is comparable across the package).
DEFAULT_MEASUREMENT: dict[str, Any] = {
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
}

# Fixed (non-alpha) estimator hyperparameters reused across phases.
RCOND = 1.0e-10
TSVD_TAU = 1.0e-5
HUBER_DELTA = 1.5
HUBER_MAX_ITER = 10
HUBER_TOL = 1.0e-7

PAPER_TO_INTERNAL = {
    "pseudoinverse": "pseudoinverse",
    "ridge_tikhonov": "ridge",
    "truncated_svd": "truncated_svd",
    "huber_irls": "huber_irls",
    "qsvt_target_classical": "qsvt_regularized",
    "normal_equation_wls": "normal_equation_wls",
    "lav": "lav",
    "hhl_style_proxy": "hhl_style_inverse_proxy",
}
ALPHA_ESTIMATORS = frozenset({"ridge_tikhonov", "qsvt_target_classical"})
INTERNAL_TO_PAPER = {internal: paper for paper, internal in PAPER_TO_INTERNAL.items()}

_INCLUDE_KEYS = (
    "include_voltage_magnitudes",
    "include_p_injections",
    "include_q_injections",
    "include_p_branch_flows",
    "include_q_branch_flows",
)
_KEY_TO_TYPE = {
    "include_voltage_magnitudes": "voltage_magnitude",
    "include_p_injections": "p_injection",
    "include_q_injections": "q_injection",
    "include_p_branch_flows": "p_branch_flow",
    "include_q_branch_flows": "q_branch_flow",
}

# Measurement-subset -> set of include flags that stay True (the rest are dropped).
# Subsets that filter by topology (weak-area / contiguous) use the full set plus a
# post-build row mask and are handled separately.
SUBSET_INCLUDE_FLAGS: dict[str, set[str]] = {
    "full_ac_measurement_set": set(_INCLUDE_KEYS),
    "voltage_plus_injection_plus_branch_flow": set(_INCLUDE_KEYS),
    "voltage_only": {"include_voltage_magnitudes"},
    "voltage_plus_injection": {
        "include_voltage_magnitudes",
        "include_p_injections",
        "include_q_injections",
    },
    "injection_only": {
        "include_p_injections",
        "include_q_injections",
    },
    "branch_flow_only": {
        "include_p_branch_flows",
        "include_q_branch_flows",
    },
    # Exact P/Q measurement build-up (paper setups S1-S3); S0=voltage_only, S4=full set.
    "s1_voltage_plus_p_injection": {
        "include_voltage_magnitudes",
        "include_p_injections",
    },
    "s2_voltage_plus_p_injection_plus_p_branch_flow": {
        "include_voltage_magnitudes",
        "include_p_injections",
        "include_p_branch_flows",
    },
    "s3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection": {
        "include_voltage_magnitudes",
        "include_p_injections",
        "include_p_branch_flows",
        "include_q_injections",
    },
    "drop_voltage_rows": set(_INCLUDE_KEYS) - {"include_voltage_magnitudes"},
    "drop_injection_rows": set(_INCLUDE_KEYS) - {"include_p_injections", "include_q_injections"},
    "drop_branch_flow_rows": set(_INCLUDE_KEYS)
    - {"include_p_branch_flows", "include_q_branch_flows"},
    "drop_p_injection_rows": set(_INCLUDE_KEYS) - {"include_p_injections"},
    "drop_q_injection_rows": set(_INCLUDE_KEYS) - {"include_q_injections"},
    "drop_p_branch_flow_rows": set(_INCLUDE_KEYS) - {"include_p_branch_flows"},
    "drop_q_branch_flow_rows": set(_INCLUDE_KEYS) - {"include_q_branch_flows"},
}
TOPOLOGY_SUBSETS = frozenset({"drop_weak_area_rows", "contiguous_area_drop"})


@dataclass(frozen=True, slots=True)
class SubsetSpec:
    measurement_config: dict[str, Any]
    included_types: tuple[str, ...]
    dropped_types: tuple[str, ...]


def build_estimator(paper_name: str, alpha: float) -> Any:
    """Construct an estimator from its paper-facing name and the active alpha."""

    internal = PAPER_TO_INTERNAL.get(paper_name)
    if internal is None:
        raise ValueError(f"unknown estimator: {paper_name}")
    if internal == "pseudoinverse":
        return PseudoinverseEstimator(rcond=RCOND)
    if internal == "ridge":
        return RidgeEstimator(alpha=alpha)
    if internal == "truncated_svd":
        return TruncatedSVDEstimator(tau=TSVD_TAU)
    if internal == "qsvt_regularized":
        return QSVTSpectralEstimator(alpha=alpha)
    if internal == "huber_irls":
        return HuberIRLSEstimator(
            delta=HUBER_DELTA, max_iterations=HUBER_MAX_ITER, tolerance=HUBER_TOL
        )
    if internal == "normal_equation_wls":
        return NormalEquationWLSEstimator()
    if internal == "lav":
        return LAVEstimator()
    if internal == "hhl_style_inverse_proxy":
        return HHLStyleInverseProxyEstimator()
    raise ValueError(f"unknown estimator: {paper_name}")


def subset_spec(subset: str) -> SubsetSpec | None:
    """Return the measurement config for a type-based subset, or ``None`` if unknown.

    Topology subsets (weak-area / contiguous) return the full-set config; the caller
    applies the row mask after the system is built.
    """

    if subset in TOPOLOGY_SUBSETS:
        flags = set(_INCLUDE_KEYS)
    else:
        flags = SUBSET_INCLUDE_FLAGS.get(subset)
        if flags is None:
            return None
    measurement = dict(DEFAULT_MEASUREMENT)
    included: list[str] = []
    dropped: list[str] = []
    for key in _INCLUDE_KEYS:
        on = key in flags
        measurement[key] = on
        (included if on else dropped).append(_KEY_TO_TYPE[key])
    return SubsetSpec(measurement, tuple(included), tuple(dropped))


def build_system(
    *,
    case: str,
    measurement_config: dict[str, Any],
    seed: int,
    case_source: str = DEFAULT_CASE_SOURCE,
    linearization: dict[str, Any] | None = None,
) -> WeightedSystem:
    rng = np.random.default_rng(int(seed))
    return build_ac_weighted_system(
        case_name=case,
        case_source=case_source,
        linearization_config=linearization or dict(DEFAULT_LINEARIZATION),
        measurement_config=measurement_config,
        rng=rng,
    )


def contiguous_area_buses(case: str, *, case_source: str, size: int) -> list[int]:
    """Deterministic contiguous topological area grown by BFS from the highest bus id.

    This is a controlled benchmark assumption (no field topology metadata is used);
    it returns a connected set of buses for spatial/weak-area stress.
    """

    ac_case = load_ac_case(case, case_source=case_source)
    adjacency: dict[int, set[int]] = {bus: set() for bus in ac_case.buses}
    for branch in ac_case.branches:
        adjacency.setdefault(branch.from_bus, set()).add(branch.to_bus)
        adjacency.setdefault(branch.to_bus, set()).add(branch.from_bus)
    start = max(ac_case.buses)
    visited: list[int] = []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue and len(visited) < size:
        bus = queue.popleft()
        visited.append(bus)
        for neighbour in sorted(adjacency.get(bus, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return sorted(visited)


def filter_rows_by_buses(
    system: WeightedSystem, area_buses: set[int]
) -> tuple[WeightedSystem, int]:
    """Drop measurement rows that touch ``area_buses``; returns (system, n_dropped)."""

    buses_per_row = system.metadata.get("measurement_buses")
    if not isinstance(buses_per_row, list) or len(buses_per_row) != system.n_measurements:
        return system, 0
    keep = [
        index
        for index, buses in enumerate(buses_per_row)
        if not area_buses.intersection(int(bus) for bus in buses)
    ]
    n_dropped = system.n_measurements - len(keep)
    if n_dropped == 0:
        return system, 0
    metadata = dict(system.metadata)
    for key in ("measurement_types", "measurement_labels", "measurement_stds", "measurement_buses"):
        value = metadata.get(key)
        if isinstance(value, list) and len(value) == system.n_measurements:
            metadata[key] = [value[index] for index in keep]
    filtered = WeightedSystem(
        H_tilde=np.asarray(system.H_tilde)[keep, :],
        r_tilde=np.asarray(system.r_tilde)[keep],
        x_true=system.x_true,
        metadata=metadata,
    )
    return filtered, n_dropped


def apply_noise(system: WeightedSystem, *, noise_std: float, seed: int) -> WeightedSystem:
    return add_gaussian_noise(system, noise_std=noise_std, rng=np.random.default_rng(seed))


def apply_missing(system: WeightedSystem, *, missing_ratio: float, seed: int) -> WeightedSystem:
    return remove_random_rows(system, missing_ratio=missing_ratio, rng=np.random.default_rng(seed))


def apply_bad_data(
    system: WeightedSystem,
    *,
    ratio: float,
    magnitude: float,
    target: str,
    seed: int,
) -> WeightedSystem:
    return add_bad_data_outliers(
        system,
        bad_data_config={"enabled": True, "ratio": ratio, "magnitude": magnitude, "target": target},
        rng=np.random.default_rng(seed),
    )


def conditioning(system: WeightedSystem) -> dict[str, float]:
    singular_values = system.singular_values()
    sigma_min = float(np.min(singular_values)) if singular_values.size else float("nan")
    sigma_max = float(np.max(singular_values)) if singular_values.size else float("nan")
    state_dimension = system.n_states
    return {
        "n_rows": system.n_measurements,
        "state_dimension": state_dimension,
        "redundancy_ratio": (
            system.n_measurements / state_dimension if state_dimension else float("nan")
        ),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": system.condition_number(),
        "numerical_rank": system.rank(),
        "effective_rank": system.effective_rank(),
    }


def rank_status(system: WeightedSystem) -> str:
    """Classify whether a system is solvable, underdetermined, or rank-deficient."""

    if system.n_measurements < system.n_states:
        return "rank_deficient"
    if system.rank() < system.n_states:
        return "rank_deficient"
    return "computed"


def solve_metrics(estimator: Any, system: WeightedSystem) -> dict[str, Any]:
    result: EstimatorResult = estimator.solve(system)
    rmse = result.rmse
    return {
        "rmse": float(rmse) if rmse is not None and np.isfinite(rmse) else float("nan"),
        "weighted_residual_norm": (
            float(result.weighted_residual_norm)
            if np.isfinite(result.weighted_residual_norm)
            else float("nan")
        ),
        "converged": bool(result.converged),
        "failed": bool(result.failed),
        "failure_reason": result.failure_reason or "",
    }


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def state_split_rmse(x_hat: np.ndarray, system: WeightedSystem) -> tuple[float, float, float]:
    """Split the state RMSE into angle and voltage-magnitude components."""

    if system.x_true is None or not np.all(np.isfinite(x_hat)):
        return float("nan"), float("nan"), float("nan")
    error = np.asarray(x_hat, dtype=np.float64) - np.asarray(system.x_true, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(error**2)))
    angle_idx = system.metadata.get("angle_state_indices")
    voltage_idx = system.metadata.get("voltage_magnitude_state_indices")
    angle_rmse = (
        float(np.sqrt(np.mean(error[angle_idx] ** 2)))
        if isinstance(angle_idx, list) and angle_idx
        else float("nan")
    )
    voltage_rmse = (
        float(np.sqrt(np.mean(error[voltage_idx] ** 2)))
        if isinstance(voltage_idx, list) and voltage_idx
        else float("nan")
    )
    return rmse, angle_rmse, voltage_rmse


def solve_detailed(estimator: Any, system: WeightedSystem) -> dict[str, Any]:
    """Run an estimator and return RMSE (split), residual, conditioning, convergence."""

    result: EstimatorResult = estimator.solve(system)
    rmse, angle_rmse, voltage_rmse = state_split_rmse(result.x_hat, system)
    return {
        "rmse": rmse,
        "angle_rmse": angle_rmse,
        "voltage_rmse": voltage_rmse,
        "residual_norm": _safe_float(result.residual_norm),
        "weighted_residual_norm": _safe_float(result.weighted_residual_norm),
        "condition_number": _safe_float(result.condition_number),
        "sigma_min": _safe_float(float(np.min(result.singular_values)))
        if result.singular_values.size
        else float("nan"),
        "sigma_max": _safe_float(float(np.max(result.singular_values)))
        if result.singular_values.size
        else float("nan"),
        "converged": bool(result.converged),
        "failed": bool(result.failed),
        "failure_reason": result.failure_reason or "",
        "runtime_seconds": float(result.runtime_seconds),
    }


def high_condition_measurement_config(
    case: str, *, case_source: str, multiplier: float = 30.0
) -> dict[str, Any]:
    """Full measurement config with a deterministic weak area whose row stds are inflated.

    Inflating the weak-area standard deviations shrinks the corresponding weighted rows
    and deterministically worsens the condition number — a controlled high-condition lever.
    """

    measurement = dict(DEFAULT_MEASUREMENT)
    try:
        from robust_qsvt_se.data.cases import load_ac_case

        n_buses = len(load_ac_case(case, case_source=case_source).buses)
        size = max(2, round(n_buses * 0.18))
        measurement["weak_area_buses"] = contiguous_area_buses(
            case, case_source=case_source, size=size
        )
        measurement["weak_area_std_multiplier"] = float(multiplier)
    except Exception:
        measurement["weak_area_buses"] = []
        measurement["weak_area_std_multiplier"] = 1.0
    return measurement
