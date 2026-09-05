from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.state_metadata import StateMetadata


def component_observable(index: int, name: str | None = None) -> dict[str, Any]:
    """Define a selected state-component probability observable."""

    selected = int(index)
    if selected < 0:
        raise ValueError("index must be nonnegative")
    return {
        "observable_name": name or f"component_{selected}_probability",
        "observable_type": "basis_probability",
        "indices": [selected],
        "notes": "computational-basis probability of one normalized update component",
    }


def subset_energy_observable(indices: Sequence[int], name: str) -> dict[str, Any]:
    """Define a normalized subset-energy observable."""

    selected = _validate_nonnegative_indices(indices)
    return {
        "observable_name": str(name),
        "observable_type": "subset_probability",
        "indices": selected,
        "notes": "sum of computational-basis probabilities over selected update indices",
    }


def difference_observable(i: int, j: int, dimension: int, name: str) -> dict[str, Any]:
    """Define a normalized linear functional ``(e_i - e_j)^T psi / sqrt(2)``."""

    first = int(i)
    second = int(j)
    dim = int(dimension)
    if dim <= 0:
        raise ValueError("dimension must be positive")
    if first == second:
        raise ValueError("difference observable requires two distinct indices")
    if first < 0 or second < 0 or first >= dim or second >= dim:
        raise IndexError("difference indices must be within dimension")
    coeffs = np.zeros(dim, dtype=np.float64)
    coeffs[first] = 1.0
    coeffs[second] = -1.0
    return {
        "observable_name": str(name),
        "observable_type": "linear_overlap_real",
        "indices": [first, second],
        "dimension": dim,
        "coefficients": coeffs.tolist(),
        "component": "real",
        "notes": "real part of normalized overlap with a signed difference vector",
    }


def branch_angle_difference_observable(
    from_index: int,
    to_index: int,
    dimension: int,
    name: str,
) -> dict[str, Any]:
    """Define a small branch-angle-difference overlap observable."""

    observable = difference_observable(from_index, to_index, dimension, name)
    observable["power_system_quantity"] = "branch_angle_difference_update_proxy"
    observable["notes"] = (
        "real overlap with a signed angle-difference vector; this is an index-level "
        "proxy unless the selected indices are confirmed angle-state indices"
    )
    return observable


def bus_angle_component(bus_id: int, metadata: StateMetadata) -> dict[str, Any]:
    """Define a selected bus-angle update probability observable."""

    index = metadata.index_for_angle_bus(int(bus_id))
    observable = component_observable(index, name=f"bus_{int(bus_id)}_angle_probability")
    observable.update(
        {
            "power_system_quantity": "selected_bus_angle_update_probability_proxy",
            "bus_id": int(bus_id),
            "state_index": int(index),
            "notes": (
                "computational-basis probability of the normalized bus-angle update component"
            ),
        }
    )
    return observable


def bus_voltage_component(bus_id: int, metadata: StateMetadata) -> dict[str, Any]:
    """Define a selected bus-voltage-magnitude update probability observable."""

    index = metadata.index_for_voltage_bus(int(bus_id))
    observable = component_observable(index, name=f"bus_{int(bus_id)}_voltage_probability")
    observable.update(
        {
            "power_system_quantity": "selected_bus_voltage_update_probability_proxy",
            "bus_id": int(bus_id),
            "state_index": int(index),
            "notes": (
                "computational-basis probability of the normalized voltage-magnitude "
                "update component"
            ),
        }
    )
    return observable


def branch_angle_difference(
    from_bus: int,
    to_bus: int,
    metadata: StateMetadata,
) -> dict[str, Any]:
    """Define a branch angle-difference overlap using bus-index metadata."""

    from_index = metadata.index_for_angle_bus(int(from_bus))
    to_index = metadata.index_for_angle_bus(int(to_bus))
    observable = difference_observable(
        from_index,
        to_index,
        metadata.dimension,
        name=f"branch_{int(from_bus)}_{int(to_bus)}_angle_difference_overlap",
    )
    observable.update(
        {
            "power_system_quantity": "branch_angle_difference_update_proxy",
            "from_bus": int(from_bus),
            "to_bus": int(to_bus),
            "notes": (
                "real overlap with a normalized signed vector for the selected "
                "branch angle-difference update"
            ),
        }
    )
    return observable


def area_update_energy(bus_ids: Sequence[int], metadata: StateMetadata) -> dict[str, Any]:
    """Define a selected-area normalized update-energy observable."""

    buses = [int(bus_id) for bus_id in bus_ids]
    indices = metadata.indices_for_buses(buses)
    if not indices:
        raise ValueError("area_update_energy requires at least one mapped state index")
    observable = subset_energy_observable(
        indices,
        name="area_" + "_".join(str(bus_id) for bus_id in buses) + "_update_energy",
    )
    observable.update(
        {
            "power_system_quantity": "selected_area_update_energy_proxy",
            "bus_ids": buses,
            "notes": (
                "sum of normalized update-state probabilities over mapped angle and "
                "voltage components for the selected buses"
            ),
        }
    )
    return observable


def jacobian_row_observable(
    row_index: int,
    H_tilde: np.ndarray,
    row_metadata: Any = None,
) -> dict[str, Any]:
    """Define a normalized linearized measurement-correction overlap observable."""

    matrix = np.asarray(H_tilde, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("H_tilde must be a matrix")
    selected = int(row_index)
    if selected < 0 or selected >= matrix.shape[0]:
        raise IndexError("row_index is out of range for H_tilde")
    coeffs = matrix[selected].astype(np.float64)
    if float(np.linalg.norm(coeffs)) <= 1.0e-15:
        raise ValueError("Jacobian row observable requires a nonzero row")
    metadata_dict = _row_metadata_dict(row_metadata, selected)
    observable = {
        "observable_name": metadata_dict.get(
            "label",
            f"jacobian_row_{selected}_linear_correction_overlap",
        ),
        "observable_type": "linear_overlap_real",
        "indices": np.flatnonzero(np.abs(coeffs) > 1.0e-12).astype(int).tolist(),
        "dimension": int(matrix.shape[1]),
        "coefficients": coeffs.tolist(),
        "component": "real",
        "row_index": selected,
        "power_system_quantity": "linearized_measurement_correction_proxy",
        "measurement_type": metadata_dict.get("measurement_type", ""),
        "measurement_buses": metadata_dict.get("buses", []),
        "notes": (
            "real overlap with the selected weighted-Jacobian row; this is a "
            "linearized measurement-correction proxy"
        ),
    }
    return observable


def _validate_nonnegative_indices(indices: Sequence[int]) -> list[int]:
    selected = [int(index) for index in indices]
    if any(index < 0 for index in selected):
        raise ValueError("indices must be nonnegative")
    return sorted(set(selected))


def _row_metadata_dict(row_metadata: Any, row_index: int) -> dict[str, Any]:
    if row_metadata is None:
        return {}
    if isinstance(row_metadata, dict):
        return dict(row_metadata)
    if isinstance(row_metadata, Sequence) and not isinstance(row_metadata, str | bytes):
        if row_index >= len(row_metadata):
            return {}
        return _row_metadata_dict(row_metadata[row_index], row_index)
    result: dict[str, Any] = {}
    for name in ("label", "measurement_type", "buses"):
        if hasattr(row_metadata, name):
            value = getattr(row_metadata, name)
            result[name] = list(value) if name == "buses" else value
    return result
