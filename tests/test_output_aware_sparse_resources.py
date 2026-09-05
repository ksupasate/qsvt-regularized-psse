"""Sparse matching, padded-wrapper, and measured-resource tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qiskit")

from robust_qsvt_se.qsvt.bipartite_slot_assignment import validate_slot_assignment
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    build_common_padded_wrapper,
    select_resource_constrained_support,
    support_constraint_report,
)


def _sparse_matrix() -> np.ndarray:
    return np.array(
        [
            [4.0, 0.0, 0.7, 0.0],
            [0.0, 3.0, 0.0, -0.4],
            [0.2, 0.0, 2.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ]
    )


def test_common_padded_wrapper_reconstructs_exact_matrix_with_zero_slots():
    matrix = _sparse_matrix()
    mu = float(np.max(np.abs(matrix)))
    wrapper = build_common_padded_wrapper(matrix, slots=3, mu=mu)
    np.testing.assert_allclose(wrapper.encoded_block, matrix.T / (3.0 * mu), atol=1.0e-12)
    assert wrapper.reconstruction_error < 1.0e-12
    assert wrapper.unitarity_error < 1.0e-10
    validation = validate_slot_assignment(matrix.T != 0.0, wrapper.assignment)
    assert validation["valid"]
    assert wrapper.assignment.slots == 3
    assert sum(validation["per_slot_real_edge_counts"]) == np.count_nonzero(matrix)


def test_common_mu_and_slot_count_are_explicitly_guarded():
    matrix = _sparse_matrix()
    with pytest.raises(ValueError, match="mu"):
        build_common_padded_wrapper(matrix, slots=3, mu=1.0)
    dense = np.ones((4, 4))
    with pytest.raises(ValueError, match="slot count"):
        build_common_padded_wrapper(dense, slots=3, mu=1.0)


def test_selected_support_values_are_exact_and_constraints_match_wrapper():
    matrix = np.array(
        [
            [8.0, 0.3, 0.2, 0.1],
            [0.4, 7.0, 0.5, 0.2],
            [0.2, 0.6, 6.0, 0.7],
            [0.5, 0.1, 0.8, 5.0],
        ]
    )
    constraints = SupportConstraints(8, 2, True)
    result = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    report = support_constraint_report(matrix, result.support, constraints)
    assert report["valid"]
    sparse = np.where(result.support, matrix, 0.0)
    assert np.array_equal(sparse[result.support], matrix[result.support])
    wrapper = build_common_padded_wrapper(
        sparse,
        slots=constraints.slot_budget,
        mu=float(np.max(np.abs(matrix))),
    )
    np.testing.assert_allclose(
        wrapper.encoded_block,
        sparse.T / (constraints.slot_budget * np.max(np.abs(matrix))),
        atol=1.0e-12,
    )


def test_campaign_resource_registry_contains_actual_measurements_if_present():
    path = Path("outputs/output_aware_sparse_selection/resource_registry.csv")
    if not path.is_file():
        pytest.skip("campaign resources not generated yet")
    frame = pd.read_csv(path)
    completed = frame[frame["status"] == "completed"]
    assert not completed.empty
    assert (completed["signal_unitary_gate_count"] > 0).all()
    assert (completed["signal_unitary_depth"] > 0).all()
    assert (completed["controlled_value_rotations"] == 8 * completed["slot_count"]).all()
    assert (completed["wrapper_reconstruction_error"] < 1.0e-9).all()
    assert set(completed["value_semantics"]) == {
        "exact_selected_original_values_no_quantization"
    }
    assert set(completed["resource_measurement"]) == {
        "executed_qiskit_transpile_u3_cx_optimization_level_1"
    }
