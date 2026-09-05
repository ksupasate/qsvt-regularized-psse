from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.data.cases import ACBranch, ACCase, DCBranch, DCCase

SUPPORTED_REAL_CASES = {
    "ieee14": ("pypower.case14", "case14"),
    "ieee30": ("pypower.case30", "case30"),
    "ieee57": ("pypower.case57", "case57"),
    "ieee118": ("pypower.case118", "case118"),
    "ieee300": ("pypower.case300", "case300"),
}


@dataclass(frozen=True, slots=True)
class PowerCase:
    """External MATPOWER-compatible case loaded from PYPOWER package fixtures."""

    name: str
    source: str
    source_detail: str
    base_mva: float
    bus: NDArray[np.float64]
    branch: NDArray[np.float64]
    gen: NDArray[np.float64]
    metadata: dict[str, Any]


def load_power_case(case_name: str) -> PowerCase:
    """Load a supported external IEEE case from PYPOWER case fixtures.

    The implementation imports individual `pypower.caseXX` modules directly
    instead of `pypower.api`, because the latter imports compatibility helpers
    that are not needed for static case loading.
    """

    normalized = _normalize_real_case_name(case_name)
    module_name, function_name = SUPPORTED_REAL_CASES[normalized]
    module = import_module(module_name)
    case_function = getattr(module, function_name)
    if not isinstance(case_function, Callable):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    raw_case = case_function()
    bus = np.asarray(raw_case["bus"], dtype=np.float64)
    branch = np.asarray(raw_case["branch"], dtype=np.float64)
    gen = np.asarray(raw_case["gen"], dtype=np.float64)
    return PowerCase(
        name=normalized,
        source="pypower",
        source_detail=f"{module_name}.{function_name}",
        base_mva=float(raw_case.get("baseMVA", 100.0)),
        bus=bus,
        branch=branch,
        gen=gen,
        metadata={
            "dataset_source": "pypower",
            "dataset_source_detail": f"{module_name}.{function_name}",
            "case_name": normalized,
            "n_buses": int(bus.shape[0]),
            "n_branches": int(branch.shape[0]),
            "n_generators": int(gen.shape[0]),
            "base_mva": float(raw_case.get("baseMVA", 100.0)),
        },
    )


def load_dc_case_from_pypower(case_name: str) -> DCCase:
    power_case = load_power_case(case_name)
    buses = tuple(int(bus) for bus in power_case.bus[:, 0])
    slack_bus = _slack_bus(power_case.bus)
    branches = tuple(
        DCBranch(
            from_bus=int(row[0]),
            to_bus=int(row[1]),
            reactance=float(row[3]),
        )
        for row in _active_branches(power_case.branch)
    )
    return DCCase(
        name=power_case.name,
        buses=buses,
        slack_bus=slack_bus,
        branches=branches,
        source="pypower",
        source_detail=power_case.source_detail,
        metadata=power_case.metadata,
    )


def load_ac_case_from_pypower(case_name: str) -> ACCase:
    power_case = load_power_case(case_name)
    buses = tuple(int(bus) for bus in power_case.bus[:, 0])
    slack_bus = _slack_bus(power_case.bus)
    active_branches = _active_branches(power_case.branch)
    branches = tuple(
        ACBranch(
            from_bus=int(row[0]),
            to_bus=int(row[1]),
            resistance=float(row[2]),
            reactance=float(row[3]),
            line_charging=float(row[4]),
            tap_ratio=_tap_ratio(row),
            phase_shift_degrees=float(row[9]) if row.shape[0] > 9 else 0.0,
        )
        for row in active_branches
    )
    return ACCase(
        name=power_case.name,
        buses=buses,
        slack_bus=slack_bus,
        branches=branches,
        voltage_magnitudes=tuple(float(value) for value in power_case.bus[:, 7]),
        voltage_angles_degrees=tuple(float(value) for value in power_case.bus[:, 8]),
        source="pypower",
        source_detail=power_case.source_detail,
        metadata=power_case.metadata,
    )


def _normalize_real_case_name(case_name: str) -> str:
    normalized = case_name.lower().replace("-", "").replace("_", "")
    if normalized.startswith("case"):
        normalized = f"ieee{normalized.removeprefix('case')}"
    canonical = normalized if normalized.startswith("ieee") else f"ieee{normalized}"
    if canonical not in SUPPORTED_REAL_CASES:
        raise ValueError(
            f"unsupported real IEEE case: {case_name}; "
            f"supported cases are {sorted(SUPPORTED_REAL_CASES)}"
        )
    return canonical


def _active_branches(branch: NDArray[np.float64]) -> NDArray[np.float64]:
    if branch.shape[1] <= 10:
        return branch
    return branch[branch[:, 10] > 0.0]


def _slack_bus(bus: NDArray[np.float64]) -> int:
    slack_rows = bus[bus[:, 1] == 3.0]
    if slack_rows.size == 0:
        raise ValueError("PYPOWER case has no reference bus")
    return int(slack_rows[0, 0])


def _tap_ratio(branch_row: NDArray[np.float64]) -> float:
    if branch_row.shape[0] <= 8 or float(branch_row[8]) == 0.0:
        return 1.0
    return float(branch_row[8])
