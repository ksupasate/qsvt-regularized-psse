from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DCBranch:
    """Branch data needed for a DC linearized power-flow model."""

    from_bus: int
    to_bus: int
    reactance: float

    @property
    def susceptance(self) -> float:
        return 1.0 / self.reactance


@dataclass(frozen=True, slots=True)
class DCCase:
    """Minimal DC case fixture with a fixed slack bus."""

    name: str
    buses: tuple[int, ...]
    slack_bus: int
    branches: tuple[DCBranch, ...]
    source: str = "builtin_fixture"
    source_detail: str = "src/robust_qsvt_se/data/cases.py"
    metadata: dict[str, Any] | None = None

    @property
    def state_buses(self) -> tuple[int, ...]:
        return tuple(bus for bus in self.buses if bus != self.slack_bus)


@dataclass(frozen=True, slots=True)
class ACBranch:
    """Branch data for a dependency-free AC-linearized benchmark."""

    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    line_charging: float
    tap_ratio: float = 1.0
    phase_shift_degrees: float = 0.0

    @property
    def admittance(self) -> complex:
        return 1.0 / complex(self.resistance, self.reactance)

    @property
    def complex_tap(self) -> complex:
        ratio = self.tap_ratio if self.tap_ratio != 0.0 else 1.0
        return ratio * complex_from_angle_degrees(self.phase_shift_degrees)


@dataclass(frozen=True, slots=True)
class ACCase:
    """Small built-in AC case fixture with a fixed slack angle."""

    name: str
    buses: tuple[int, ...]
    slack_bus: int
    branches: tuple[ACBranch, ...]
    voltage_magnitudes: tuple[float, ...]
    voltage_angles_degrees: tuple[float, ...]
    source: str = "builtin_fixture"
    source_detail: str = "src/robust_qsvt_se/data/cases.py"
    metadata: dict[str, Any] | None = None

    @property
    def angle_state_buses(self) -> tuple[int, ...]:
        return tuple(bus for bus in self.buses if bus != self.slack_bus)

    @property
    def voltage_state_buses(self) -> tuple[int, ...]:
        return self.buses


def complex_from_angle_degrees(angle_degrees: float) -> complex:
    import math

    angle = math.radians(angle_degrees)
    return complex(math.cos(angle), math.sin(angle))


def load_dc_case(case_name: str, *, case_source: str = "builtin") -> DCCase:
    normalized_source = _normalize_case_source(case_source)
    if normalized_source == "pypower":
        from robust_qsvt_se.data.real_cases import load_dc_case_from_pypower

        return load_dc_case_from_pypower(case_name)

    normalized = case_name.lower()
    if normalized != "ieee14":
        raise ValueError(f"unsupported DC case: {case_name}")
    return ieee14_dc_case()


def load_ac_case(case_name: str, *, case_source: str = "builtin") -> ACCase:
    normalized_source = _normalize_case_source(case_source)
    if normalized_source == "pypower":
        from robust_qsvt_se.data.real_cases import load_ac_case_from_pypower

        return load_ac_case_from_pypower(case_name)

    normalized = case_name.lower()
    if normalized != "ieee14":
        raise ValueError(f"unsupported AC case: {case_name}")
    return ieee14_ac_case()


def supported_case_names(case_source: str = "builtin") -> set[str]:
    normalized_source = _normalize_case_source(case_source)
    if normalized_source == "pypower":
        return {"ieee14", "ieee30", "ieee57", "ieee118", "ieee300"}
    return {"ieee14"}


def _normalize_case_source(case_source: str) -> str:
    normalized = case_source.lower().replace("_", "-")
    if normalized in {"builtin", "builtin-fixture", "fixture"}:
        return "builtin"
    if normalized in {"pypower", "matpower", "external-pypower"}:
        return "pypower"
    raise ValueError("case_source must be one of: builtin, pypower")


def ieee14_dc_case() -> DCCase:
    """Return a small IEEE-14-style DC fixture.

    The topology and reactances are the standard IEEE 14-bus branch data used
    for deterministic research-code tests. Transformer taps and resistance are
    intentionally ignored in this milestone because the model is DC-linearized.
    """

    return DCCase(
        name="ieee14",
        buses=tuple(range(1, 15)),
        slack_bus=1,
        branches=(
            DCBranch(1, 2, 0.05917),
            DCBranch(1, 5, 0.22304),
            DCBranch(2, 3, 0.19797),
            DCBranch(2, 4, 0.17632),
            DCBranch(2, 5, 0.17388),
            DCBranch(3, 4, 0.17103),
            DCBranch(4, 5, 0.04211),
            DCBranch(4, 7, 0.20912),
            DCBranch(4, 9, 0.55618),
            DCBranch(5, 6, 0.25202),
            DCBranch(6, 11, 0.19890),
            DCBranch(6, 12, 0.25581),
            DCBranch(6, 13, 0.13027),
            DCBranch(7, 8, 0.17615),
            DCBranch(7, 9, 0.11001),
            DCBranch(9, 10, 0.08450),
            DCBranch(9, 14, 0.27038),
            DCBranch(10, 11, 0.19207),
            DCBranch(12, 13, 0.19988),
            DCBranch(13, 14, 0.34802),
        ),
    )


def ieee14_ac_case() -> ACCase:
    """Return a compact IEEE-14-style AC fixture.

    The branch resistance/reactance/charging and voltage profile are the
    standard IEEE 14-bus values commonly used in MATPOWER-style examples. This
    project uses them as built-in research fixtures and does not load external
    datasets.
    """

    return ACCase(
        name="ieee14",
        buses=tuple(range(1, 15)),
        slack_bus=1,
        branches=(
            ACBranch(1, 2, 0.01938, 0.05917, 0.0528),
            ACBranch(1, 5, 0.05403, 0.22304, 0.0492),
            ACBranch(2, 3, 0.04699, 0.19797, 0.0438),
            ACBranch(2, 4, 0.05811, 0.17632, 0.0340),
            ACBranch(2, 5, 0.05695, 0.17388, 0.0346),
            ACBranch(3, 4, 0.06701, 0.17103, 0.0128),
            ACBranch(4, 5, 0.01335, 0.04211, 0.0),
            ACBranch(4, 7, 0.0, 0.20912, 0.0),
            ACBranch(4, 9, 0.0, 0.55618, 0.0),
            ACBranch(5, 6, 0.0, 0.25202, 0.0),
            ACBranch(6, 11, 0.09498, 0.19890, 0.0),
            ACBranch(6, 12, 0.12291, 0.25581, 0.0),
            ACBranch(6, 13, 0.06615, 0.13027, 0.0),
            ACBranch(7, 8, 0.0, 0.17615, 0.0),
            ACBranch(7, 9, 0.0, 0.11001, 0.0),
            ACBranch(9, 10, 0.03181, 0.08450, 0.0),
            ACBranch(9, 14, 0.12711, 0.27038, 0.0),
            ACBranch(10, 11, 0.08205, 0.19207, 0.0),
            ACBranch(12, 13, 0.22092, 0.19988, 0.0),
            ACBranch(13, 14, 0.17093, 0.34802, 0.0),
        ),
        voltage_magnitudes=(
            1.060,
            1.045,
            1.010,
            1.018,
            1.020,
            1.070,
            1.062,
            1.090,
            1.056,
            1.051,
            1.057,
            1.055,
            1.050,
            1.036,
        ),
        voltage_angles_degrees=(
            0.0,
            -4.98,
            -12.72,
            -10.33,
            -8.78,
            -14.22,
            -13.37,
            -13.36,
            -14.94,
            -15.10,
            -14.79,
            -15.07,
            -15.16,
            -16.04,
        ),
    )


def dc_measurement_count(
    *,
    case_name: str,
    case_source: str = "builtin",
    include_branch_flows: bool,
    include_bus_injections: bool,
    angle_measurement_buses: list[int] | tuple[int, ...],
) -> int:
    case = load_dc_case(case_name, case_source=case_source)
    count = 0
    if include_branch_flows:
        count += len(case.branches)
    if include_bus_injections:
        count += len(case.state_buses)
    count += len(tuple(bus for bus in angle_measurement_buses if bus != case.slack_bus))
    return count


def ac_state_count(case_name: str, *, case_source: str = "builtin") -> int:
    case = load_ac_case(case_name, case_source=case_source)
    return len(case.angle_state_buses) + len(case.voltage_state_buses)


def ac_measurement_count(
    *,
    case_name: str,
    case_source: str = "builtin",
    measurement_config: dict,
) -> int:
    case = load_ac_case(case_name, case_source=case_source)
    count = 0
    if bool(measurement_config.get("include_voltage_magnitudes", True)):
        count += len(case.buses)
    if bool(measurement_config.get("include_p_injections", True)):
        count += len(case.buses)
    if bool(measurement_config.get("include_q_injections", True)):
        count += len(case.buses)
    if bool(measurement_config.get("include_p_branch_flows", True)):
        count += len(case.branches)
    if bool(measurement_config.get("include_q_branch_flows", True)):
        count += len(case.branches)
    return count
