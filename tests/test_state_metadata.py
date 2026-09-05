from __future__ import annotations

import pytest

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.power_observables import (
    area_update_energy,
    branch_angle_difference,
    bus_angle_component,
    bus_voltage_component,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata


def test_state_metadata_maps_ac_angle_and_voltage_indices() -> None:
    metadata = build_state_metadata_from_system_metadata(
        {
            "angle_state_buses": [2, 3],
            "voltage_state_buses": [1, 2, 3],
            "angle_state_indices": [0, 1],
            "voltage_magnitude_state_indices": [2, 3, 4],
        }
    )

    assert metadata.dimension == 5
    assert metadata.index_for_angle_bus(2) == 0
    assert metadata.index_for_voltage_bus(3) == 4
    assert metadata.record_for_index(2).description == "Delta voltage magnitude update for bus 1"


def test_state_metadata_fallback_is_index_level_only() -> None:
    metadata = build_state_metadata_from_system_metadata({"n_states": 3})

    assert metadata.source == "generic_dimension_fallback"
    assert metadata.records[0].state_type == "unknown"
    assert metadata.assumptions


def test_power_observables_use_state_metadata() -> None:
    metadata = build_state_metadata_from_system_metadata(
        {
            "angle_state_buses": [2, 3],
            "voltage_state_buses": [1, 2, 3],
            "angle_state_indices": [0, 1],
            "voltage_magnitude_state_indices": [2, 3, 4],
        }
    )

    assert bus_angle_component(2, metadata)["indices"] == [0]
    assert bus_voltage_component(2, metadata)["indices"] == [3]
    assert branch_angle_difference(2, 3, metadata)["indices"] == [0, 1]
    assert area_update_energy([2], metadata)["indices"] == [0, 3]


def test_missing_bus_raises_clear_error() -> None:
    metadata = build_state_metadata_from_system_metadata(
        {
            "angle_state_buses": [2],
            "voltage_state_buses": [1],
        }
    )

    with pytest.raises(KeyError):
        metadata.index_for_angle_bus(9)


def test_state_metadata_maps_builtin_ieee14_ordering() -> None:
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "builtin",
            "matrix_source": "ieee14_ac_weighted_jacobian",
            "seed": 123,
        }
    )

    metadata = build_state_metadata_from_system_metadata(system.metadata)

    assert metadata.ordering_note == "Delta x = [Delta theta, Delta V]"
    assert metadata.index_for_angle_bus(2) == 0
    assert metadata.index_for_voltage_bus(1) == len(system.metadata["angle_state_buses"])
