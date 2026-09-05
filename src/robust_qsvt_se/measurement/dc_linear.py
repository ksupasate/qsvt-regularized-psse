from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.data.cases import DCBranch, DCCase, load_dc_case
from robust_qsvt_se.measurement.linear_system import WeightedSystem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MeasurementRow:
    measurement_type: str
    label: str
    buses: tuple[int, ...]
    std: float


def build_ieee14_dc_weighted_system(
    *,
    angle_scale: float,
    measurement_config: dict[str, Any],
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> WeightedSystem:
    return build_dc_weighted_system(
        case_name="ieee14",
        case_source="builtin",
        angle_scale=angle_scale,
        measurement_config=measurement_config,
        rng=rng,
        metadata=metadata,
    )


def build_dc_weighted_system(
    *,
    case_name: str,
    case_source: str,
    angle_scale: float,
    measurement_config: dict[str, Any],
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> WeightedSystem:
    case = load_dc_case(case_name, case_source=case_source)
    H, rows = build_dc_measurement_matrix(case=case, measurement_config=measurement_config)
    theta_true = rng.normal(loc=0.0, scale=angle_scale, size=len(case.state_buses))
    stds = np.array([row.std for row in rows], dtype=np.float64)
    z = H @ theta_true
    H_tilde = H / stds[:, None]
    r_tilde = z / stds

    system_metadata = {
        "case_name": case.name,
        "dataset_source": case.source,
        "dataset_source_detail": case.source_detail,
        "external_case": case.source == "pypower",
        "mode": "dc_power_flow_linearized",
        "note": "DC-linearized benchmark; not a full AC estimator.",
        "slack_bus": case.slack_bus,
        "state_buses": list(case.state_buses),
        "n_branches": len(case.branches),
        "n_buses": len(case.buses),
        "n_measurements_original": int(H.shape[0]),
        "n_states": int(H.shape[1]),
        "measurement_types": [row.measurement_type for row in rows],
        "measurement_labels": [row.label for row in rows],
        "measurement_buses": [list(row.buses) for row in rows],
        "measurement_stds": stds.tolist(),
        "weak_area_buses": [int(bus) for bus in measurement_config.get("weak_area_buses", ())],
        "achieved_condition_number": float(np.linalg.cond(H_tilde)),
    }
    if metadata:
        system_metadata.update(metadata)
    return WeightedSystem(
        H_tilde=H_tilde,
        r_tilde=r_tilde,
        x_true=theta_true,
        metadata=system_metadata,
    )


def build_dc_measurement_matrix(
    *,
    case: DCCase,
    measurement_config: dict[str, Any],
) -> tuple[FloatArray, list[MeasurementRow]]:
    state_index = {bus: index for index, bus in enumerate(case.state_buses)}
    rows: list[FloatArray] = []
    row_metadata: list[MeasurementRow] = []

    include_branch_flows = bool(measurement_config.get("include_branch_flows", True))
    include_bus_injections = bool(measurement_config.get("include_bus_injections", True))
    angle_measurement_buses = tuple(int(bus) for bus in measurement_config.get("angle_buses", ()))

    if include_branch_flows:
        for branch in case.branches:
            row = np.zeros(len(case.state_buses), dtype=np.float64)
            _add_angle_difference(row, state_index, branch, case.slack_bus)
            rows.append(row)
            row_metadata.append(
                _row_metadata(
                    measurement_type="branch_flow",
                    label=f"P_{branch.from_bus}_{branch.to_bus}",
                    buses=(branch.from_bus, branch.to_bus),
                    base_std=float(measurement_config.get("flow_std", 0.02)),
                    measurement_config=measurement_config,
                )
            )

    if include_bus_injections:
        for bus in case.state_buses:
            row = np.zeros(len(case.state_buses), dtype=np.float64)
            for branch in _incident_branches(case, bus):
                neighbor = branch.to_bus if branch.from_bus == bus else branch.from_bus
                susceptance = branch.susceptance
                _add_bus_coefficient(row, state_index, bus, susceptance)
                _add_bus_coefficient(row, state_index, neighbor, -susceptance)
            rows.append(row)
            row_metadata.append(
                _row_metadata(
                    measurement_type="bus_injection",
                    label=f"P_inj_{bus}",
                    buses=(bus,),
                    base_std=float(measurement_config.get("injection_std", 0.03)),
                    measurement_config=measurement_config,
                )
            )

    for bus in angle_measurement_buses:
        if bus == case.slack_bus:
            continue
        if bus not in state_index:
            raise ValueError(f"angle measurement bus {bus} is not in case {case.name}")
        row = np.zeros(len(case.state_buses), dtype=np.float64)
        row[state_index[bus]] = 1.0
        rows.append(row)
        row_metadata.append(
            _row_metadata(
                measurement_type="angle",
                label=f"theta_{bus}",
                buses=(bus,),
                base_std=float(measurement_config.get("angle_std", 0.005)),
                measurement_config=measurement_config,
            )
        )

    if not rows:
        raise ValueError("DC measurement config must include at least one measurement row")
    H = np.vstack(rows)
    if np.linalg.matrix_rank(H) == 0:
        raise ValueError("DC measurement matrix has zero rank")
    return H, row_metadata


def _add_angle_difference(
    row: FloatArray,
    state_index: dict[int, int],
    branch: DCBranch,
    slack_bus: int,
) -> None:
    susceptance = branch.susceptance
    if branch.from_bus != slack_bus:
        row[state_index[branch.from_bus]] += susceptance
    if branch.to_bus != slack_bus:
        row[state_index[branch.to_bus]] -= susceptance


def _add_bus_coefficient(
    row: FloatArray,
    state_index: dict[int, int],
    bus: int,
    coefficient: float,
) -> None:
    if bus in state_index:
        row[state_index[bus]] += coefficient


def _incident_branches(case: DCCase, bus: int) -> list[DCBranch]:
    return [branch for branch in case.branches if bus in {branch.from_bus, branch.to_bus}]


def _row_metadata(
    *,
    measurement_type: str,
    label: str,
    buses: tuple[int, ...],
    base_std: float,
    measurement_config: dict[str, Any],
) -> MeasurementRow:
    if base_std <= 0.0:
        raise ValueError(f"{measurement_type} measurement std must be positive")
    weak_area_buses = {int(bus) for bus in measurement_config.get("weak_area_buses", ())}
    weak_multiplier = float(measurement_config.get("weak_area_std_multiplier", 1.0))
    if weak_multiplier <= 0.0:
        raise ValueError("weak_area_std_multiplier must be positive")
    std = base_std * weak_multiplier if weak_area_buses.intersection(buses) else base_std
    return MeasurementRow(
        measurement_type=measurement_type,
        label=label,
        buses=buses,
        std=float(std),
    )
