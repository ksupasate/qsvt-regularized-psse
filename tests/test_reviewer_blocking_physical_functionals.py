"""Workstream 1 tests: physical functional determinism, mapping, and availability."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.reviewer_blocking.common import build_physical_block_binding
from robust_qsvt_se.reviewer_blocking.physical_functionals import (
    build_all_physical_functionals,
    build_area_aggregate_functionals,
    build_branch_angle_difference_functionals,
    build_coordinate_functionals,
)


def _binding():
    return build_physical_block_binding(123)


def test_block_binding_matches_frozen_state_mapping():
    binding = _binding()
    assert binding.selected_rows == (15, 17, 18, 29, 31, 32, 48, 68)
    assert binding.selected_columns == (0, 2, 3, 7, 13, 14, 16, 17)
    mapping = {(rec.state_type, rec.bus_id): rec.local_index for rec in binding.columns}
    assert mapping[("angle", 2)] == 0
    assert mapping[("voltage", 1)] == 4
    assert {rec.state_type for rec in binding.columns} == {"angle", "voltage"}


def test_all_functionals_are_unit_norm_and_length_matches_dimension():
    binding = _binding()
    records, _ = build_all_physical_functionals(binding)
    assert records, "expected at least one physical functional"
    for record in records:
        vector = np.asarray(record.vector, dtype=np.float64)
        assert vector.shape == (binding.col_count,)
        assert record.unit_norm
        assert abs(np.linalg.norm(vector) - 1.0) <= 1e-12


def test_coordinate_functionals_map_global_to_local_and_state_type():
    binding = _binding()
    records = build_coordinate_functionals(binding)
    # One coordinate per in-block state; angle vs voltage typed correctly.
    assert len(records) == len(binding.columns)
    by_id = {record.functional_id: record for record in records}
    angle2 = by_id["coordinate_angle_bus2"]
    assert angle2.state_type == "angle"
    assert angle2.local_block_indices == (0,)
    assert angle2.global_state_indices == (0,)
    assert np.argmax(np.abs(angle2.vector)) == 0
    volt1 = by_id["coordinate_voltage_bus1"]
    assert volt1.state_type == "voltage"
    assert volt1.local_block_indices == (4,)
    assert volt1.global_state_indices == (13,)


def test_branch_functionals_reference_only_real_branches():
    binding = _binding()
    records = build_branch_angle_difference_functionals(binding)
    branch_set = set(binding.branches)
    assert records
    for record in records:
        from_bus, to_bus = (int(x) for x in record.branch_id.split("-"))
        assert (from_bus, to_bus) in branch_set  # branch existence
        assert record.state_type == "angle"
        # signed-difference structure: two nonzeros, opposite sign, unit norm.
        vector = np.asarray(record.vector)
        nonzero = np.flatnonzero(vector)
        assert nonzero.size == 2
        assert np.isclose(vector[nonzero[0]], -vector[nonzero[1]])


def test_exactly_one_primary_per_coordinate_state_type_deterministically():
    binding = _binding()
    records = build_coordinate_functionals(binding)
    for state_type in ("angle", "voltage"):
        primaries = [
            r for r in records if r.state_type == state_type and r.is_primary_for_policy
        ]
        assert len(primaries) == 1
        # Primary is the smallest global state index of that type (outcome-independent).
        typed = [r for r in records if r.state_type == state_type]
        expected = min(r.global_state_indices[0] for r in typed)
        assert primaries[0].global_state_indices[0] == expected


def test_selection_uses_no_output_values():
    binding = _binding()
    records, _ = build_all_physical_functionals(binding)
    for record in records:
        assert record.output_values_available_during_selection is False
        assert record.selection_inputs == "topology_and_state_index_metadata_only"


def test_unavailable_functional_is_recorded_not_substituted():
    binding = _binding()
    _records, unavailable = build_area_aggregate_functionals(binding)
    ids = {item.requested_functional_id for item in unavailable}
    # IEEE-14 has empty weak_area_buses, so the weak-area aggregate is unavailable.
    assert "weak_area_voltage_aggregate" in ids
    item = next(
        i for i in unavailable if i.requested_functional_id == "weak_area_voltage_aggregate"
    )
    assert "weak_area_buses" in item.reason_unavailable


def test_functional_construction_is_deterministic():
    first, first_unavailable = build_all_physical_functionals(_binding())
    second, second_unavailable = build_all_physical_functionals(_binding())
    assert [r.functional_id for r in first] == [r.functional_id for r in second]
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(np.asarray(a.vector), np.asarray(b.vector))
    assert [u.requested_functional_id for u in first_unavailable] == [
        u.requested_functional_id for u in second_unavailable
    ]


def test_legacy_functionals_are_unchanged():
    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        predetermined_selected_functionals,
    )

    binding = _binding()
    records, _ = build_all_physical_functionals(binding)
    legacy = {
        r.functional_id: np.asarray(r.vector)
        for r in records
        if r.family == "legacy_predetermined"
    }
    reference = predetermined_selected_functionals(binding.col_count)
    assert set(legacy) == set(reference)
    for name, vector in reference.items():
        assert np.allclose(legacy[name], np.asarray(vector, dtype=np.float64))
