from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.data.cases import ACBranch, ACCase, load_ac_case
from robust_qsvt_se.measurement.linear_system import WeightedSystem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ACMeasurementRow:
    measurement_type: str
    label: str
    buses: tuple[int, ...]
    std: float


def build_ieee14_ac_weighted_system(
    *,
    linearization_config: dict[str, Any],
    measurement_config: dict[str, Any],
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> WeightedSystem:
    return build_ac_weighted_system(
        case_name="ieee14",
        case_source="builtin",
        linearization_config=linearization_config,
        measurement_config=measurement_config,
        rng=rng,
        metadata=metadata,
    )


def build_ac_weighted_system(
    *,
    case_name: str,
    case_source: str,
    linearization_config: dict[str, Any],
    measurement_config: dict[str, Any],
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> WeightedSystem:
    """Build a weighted AC-linearized IEEE14 benchmark.

    This constructs one Gauss-Newton-style linearized update problem around a
    perturbed state `x0`; it does not iterate to convergence.
    """

    case = load_ac_case(case_name, case_source=case_source)
    x_true = default_ac_state_vector(case)
    angle_perturbation_std = float(linearization_config.get("angle_perturbation_std", 0.01))
    voltage_perturbation_std = float(linearization_config.get("voltage_perturbation_std", 0.01))
    if angle_perturbation_std < 0.0 or voltage_perturbation_std < 0.0:
        raise ValueError("AC linearization perturbation stds must be nonnegative")

    angle_count = len(case.angle_state_buses)
    perturbation = np.concatenate(
        [
            rng.normal(0.0, angle_perturbation_std, size=angle_count),
            rng.normal(0.0, voltage_perturbation_std, size=len(case.voltage_state_buses)),
        ]
    )
    x0 = x_true + perturbation
    min_voltage = float(linearization_config.get("min_voltage_magnitude", 0.5))
    x0[angle_count:] = np.maximum(x0[angle_count:], min_voltage)

    z_true, _, true_rows = ac_measurements_and_jacobian(case, x_true, measurement_config)
    z0, H, rows = ac_measurements_and_jacobian(case, x0, measurement_config)
    if [row.label for row in rows] != [row.label for row in true_rows]:
        raise ValueError("AC measurement row layout changed between true and linearization states")

    residual = z_true - z0
    stds = np.array([row.std for row in rows], dtype=np.float64)
    H_tilde = H / stds[:, None]
    r_tilde = residual / stds
    delta_true = x_true - x0

    angle_indices = list(range(angle_count))
    voltage_indices = list(range(angle_count, angle_count + len(case.voltage_state_buses)))
    system_metadata = {
        "case_name": case.name,
        "dataset_source": case.source,
        "dataset_source_detail": case.source_detail,
        "external_case": case.source == "pypower",
        "mode": "ac_power_flow_linearized",
        "note": "AC benchmark is one weighted linearized update, not full iterative AC SE.",
        "slack_bus": case.slack_bus,
        "angle_state_buses": list(case.angle_state_buses),
        "voltage_state_buses": list(case.voltage_state_buses),
        "angle_state_indices": angle_indices,
        "voltage_magnitude_state_indices": voltage_indices,
        "state_kind": "delta_x",
        "linearization_state": x0.tolist(),
        "true_state": x_true.tolist(),
        "linearization_angle_perturbation_std": angle_perturbation_std,
        "linearization_voltage_perturbation_std": voltage_perturbation_std,
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
        x_true=delta_true,
        metadata=system_metadata,
    )


def default_ac_state_vector(case: ACCase) -> FloatArray:
    angles = np.deg2rad(np.array(case.voltage_angles_degrees, dtype=np.float64))
    voltages = np.array(case.voltage_magnitudes, dtype=np.float64)
    angle_values = np.array(
        [angles[_bus_index(case, bus)] for bus in case.angle_state_buses],
        dtype=np.float64,
    )
    return np.concatenate([angle_values, voltages])


def ac_measurements_and_jacobian(
    case: ACCase,
    state: FloatArray,
    measurement_config: dict[str, Any],
) -> tuple[FloatArray, FloatArray, list[ACMeasurementRow]]:
    theta, voltage = unpack_ac_state(case, state)
    ybus = build_ybus(case)
    rows: list[FloatArray] = []
    values: list[float] = []
    row_metadata: list[ACMeasurementRow] = []

    if bool(measurement_config.get("include_voltage_magnitudes", True)):
        for bus in case.buses:
            row = np.zeros(len(state), dtype=np.float64)
            row[_voltage_column(case, bus)] = 1.0
            rows.append(row)
            values.append(float(voltage[_bus_index(case, bus)]))
            row_metadata.append(
                _row_metadata(
                    measurement_type="voltage_magnitude",
                    label=f"V_{bus}",
                    buses=(bus,),
                    base_std=float(measurement_config.get("voltage_std", 0.01)),
                    measurement_config=measurement_config,
                )
            )

    for quantity in ("p", "q"):
        include_key = f"include_{quantity}_injections"
        if not bool(measurement_config.get(include_key, True)):
            continue
        for bus in case.buses:
            value, row = _bus_injection_value_and_jacobian(
                case=case,
                ybus=ybus,
                theta=theta,
                voltage=voltage,
                bus=bus,
                quantity=quantity,
            )
            rows.append(row)
            values.append(value)
            row_metadata.append(
                _row_metadata(
                    measurement_type=f"{quantity}_injection",
                    label=f"{quantity.upper()}_inj_{bus}",
                    buses=(bus,),
                    base_std=float(measurement_config.get(f"injection_{quantity}_std", 0.03)),
                    measurement_config=measurement_config,
                )
            )

    for quantity in ("p", "q"):
        include_key = f"include_{quantity}_branch_flows"
        if not bool(measurement_config.get(include_key, True)):
            continue
        for branch in case.branches:
            value, row = _branch_flow_value_and_jacobian(
                case=case,
                branch=branch,
                theta=theta,
                voltage=voltage,
                quantity=quantity,
            )
            rows.append(row)
            values.append(value)
            row_metadata.append(
                _row_metadata(
                    measurement_type=f"{quantity}_branch_flow",
                    label=f"{quantity.upper()}_{branch.from_bus}_{branch.to_bus}",
                    buses=(branch.from_bus, branch.to_bus),
                    base_std=float(measurement_config.get(f"flow_{quantity}_std", 0.02)),
                    measurement_config=measurement_config,
                )
            )

    if not rows:
        raise ValueError("AC measurement config must include at least one measurement row")
    return np.array(values, dtype=np.float64), np.vstack(rows), row_metadata


def ac_measurement_vector(
    case: ACCase,
    state: FloatArray,
    measurement_config: dict[str, Any],
) -> tuple[FloatArray, list[ACMeasurementRow]]:
    values, _, rows = ac_measurements_and_jacobian(case, state, measurement_config)
    return values, rows


def unpack_ac_state(case: ACCase, state: FloatArray) -> tuple[FloatArray, FloatArray]:
    state = np.asarray(state, dtype=np.float64)
    expected = len(case.angle_state_buses) + len(case.voltage_state_buses)
    if state.shape != (expected,):
        raise ValueError(f"AC state must have shape {(expected,)}, got {state.shape}")
    theta = np.zeros(len(case.buses), dtype=np.float64)
    for index, bus in enumerate(case.angle_state_buses):
        theta[_bus_index(case, bus)] = state[index]
    voltage_start = len(case.angle_state_buses)
    voltage = state[voltage_start:].copy()
    if np.any(voltage <= 0.0):
        raise ValueError("AC voltage magnitudes must be positive")
    return theta, voltage


def build_ybus(case: ACCase) -> NDArray[np.complex128]:
    ybus = np.zeros((len(case.buses), len(case.buses)), dtype=np.complex128)
    for branch in case.branches:
        from_index = _bus_index(case, branch.from_bus)
        to_index = _bus_index(case, branch.to_bus)
        y = branch.admittance
        tap = branch.complex_tap
        y_shunt = 1j * branch.line_charging / 2.0
        ybus[from_index, from_index] += (y + y_shunt) / (tap * np.conj(tap))
        ybus[to_index, to_index] += y + y_shunt
        ybus[from_index, to_index] -= y / np.conj(tap)
        ybus[to_index, from_index] -= y / tap
    return ybus


def _bus_injection_value_and_jacobian(
    *,
    case: ACCase,
    ybus: NDArray[np.complex128],
    theta: FloatArray,
    voltage: FloatArray,
    bus: int,
    quantity: str,
) -> tuple[float, FloatArray]:
    i = _bus_index(case, bus)
    g = ybus.real
    b = ybus.imag
    row = np.zeros(len(case.angle_state_buses) + len(case.voltage_state_buses), dtype=np.float64)
    p_value = 0.0
    q_value = 0.0
    for j in range(len(case.buses)):
        angle = theta[i] - theta[j]
        p_value += voltage[i] * voltage[j] * (g[i, j] * np.cos(angle) + b[i, j] * np.sin(angle))
        q_value += voltage[i] * voltage[j] * (g[i, j] * np.sin(angle) - b[i, j] * np.cos(angle))

    for bus_k in case.angle_state_buses:
        k = _bus_index(case, bus_k)
        if k == i:
            d_p = 0.0
            d_q = 0.0
            for j in range(len(case.buses)):
                if j == i:
                    continue
                angle = theta[i] - theta[j]
                d_p += (
                    voltage[i] * voltage[j] * (-g[i, j] * np.sin(angle) + b[i, j] * np.cos(angle))
                )
                d_q += voltage[i] * voltage[j] * (g[i, j] * np.cos(angle) + b[i, j] * np.sin(angle))
        else:
            angle = theta[i] - theta[k]
            d_p = voltage[i] * voltage[k] * (g[i, k] * np.sin(angle) - b[i, k] * np.cos(angle))
            d_q = -voltage[i] * voltage[k] * (g[i, k] * np.cos(angle) + b[i, k] * np.sin(angle))
        row[_angle_column(case, bus_k)] = d_p if quantity == "p" else d_q

    for bus_k in case.voltage_state_buses:
        k = _bus_index(case, bus_k)
        angle = theta[i] - theta[k]
        if k == i:
            d_p_v = 2.0 * voltage[i] * g[i, i]
            d_q_v = -2.0 * voltage[i] * b[i, i]
            for j in range(len(case.buses)):
                if j == i:
                    continue
                angle_ij = theta[i] - theta[j]
                d_p_v += voltage[j] * (g[i, j] * np.cos(angle_ij) + b[i, j] * np.sin(angle_ij))
                d_q_v += voltage[j] * (g[i, j] * np.sin(angle_ij) - b[i, j] * np.cos(angle_ij))
        else:
            d_p_v = voltage[i] * (g[i, k] * np.cos(angle) + b[i, k] * np.sin(angle))
            d_q_v = voltage[i] * (g[i, k] * np.sin(angle) - b[i, k] * np.cos(angle))
        row[_voltage_column(case, bus_k)] = d_p_v if quantity == "p" else d_q_v

    return (float(p_value), row) if quantity == "p" else (float(q_value), row)


def _branch_flow_value_and_jacobian(
    *,
    case: ACCase,
    branch: ACBranch,
    theta: FloatArray,
    voltage: FloatArray,
    quantity: str,
) -> tuple[float, FloatArray]:
    from_index = _bus_index(case, branch.from_bus)
    to_index = _bus_index(case, branch.to_bus)
    y = branch.admittance
    tap = branch.complex_tap
    y_shunt = 1j * branch.line_charging / 2.0
    yff = (y + y_shunt) / (tap * np.conj(tap))
    yft = -y / np.conj(tap)
    vi = voltage[from_index]
    vj = voltage[to_index]
    angle = theta[from_index] - theta[to_index]
    exp_delta = np.exp(1j * angle)
    s_flow = np.conj(yff) * vi**2 + np.conj(yft) * vi * vj * exp_delta

    row = np.zeros(len(case.angle_state_buses) + len(case.voltage_state_buses), dtype=np.float64)
    d_s_angle = 1j * np.conj(yft) * vi * vj * exp_delta
    d_p_angle = float(d_s_angle.real)
    d_q_angle = float(d_s_angle.imag)
    _set_angle_derivative(row, case, branch.from_bus, d_p_angle if quantity == "p" else d_q_angle)
    _set_angle_derivative(row, case, branch.to_bus, -d_p_angle if quantity == "p" else -d_q_angle)

    d_s_dvi = 2.0 * np.conj(yff) * vi + np.conj(yft) * vj * exp_delta
    d_s_dvj = np.conj(yft) * vi * exp_delta
    d_p_dvi = float(d_s_dvi.real)
    d_p_dvj = float(d_s_dvj.real)
    d_q_dvi = float(d_s_dvi.imag)
    d_q_dvj = float(d_s_dvj.imag)
    row[_voltage_column(case, branch.from_bus)] = d_p_dvi if quantity == "p" else d_q_dvi
    row[_voltage_column(case, branch.to_bus)] = d_p_dvj if quantity == "p" else d_q_dvj
    return (float(s_flow.real), row) if quantity == "p" else (float(s_flow.imag), row)


def _set_angle_derivative(row: FloatArray, case: ACCase, bus: int, value: float) -> None:
    if bus != case.slack_bus:
        row[_angle_column(case, bus)] = value


def _row_metadata(
    *,
    measurement_type: str,
    label: str,
    buses: tuple[int, ...],
    base_std: float,
    measurement_config: dict[str, Any],
) -> ACMeasurementRow:
    if base_std <= 0.0:
        raise ValueError(f"{measurement_type} measurement std must be positive")
    weak_area_buses = {int(bus) for bus in measurement_config.get("weak_area_buses", ())}
    weak_multiplier = float(measurement_config.get("weak_area_std_multiplier", 1.0))
    if weak_multiplier <= 0.0:
        raise ValueError("weak_area_std_multiplier must be positive")
    std = base_std * weak_multiplier if weak_area_buses.intersection(buses) else base_std
    return ACMeasurementRow(
        measurement_type=measurement_type,
        label=label,
        buses=buses,
        std=float(std),
    )


def _angle_column(case: ACCase, bus: int) -> int:
    if bus == case.slack_bus:
        raise ValueError("slack bus angle is not part of the AC state")
    return case.angle_state_buses.index(bus)


def _voltage_column(case: ACCase, bus: int) -> int:
    return len(case.angle_state_buses) + case.voltage_state_buses.index(bus)


def _bus_index(case: ACCase, bus: int) -> int:
    return case.buses.index(bus)
