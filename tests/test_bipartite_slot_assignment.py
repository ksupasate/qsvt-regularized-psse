"""Tests for the deterministic bipartite slot assignment (Phase 10 WP A fix)."""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    SlotAssignment,
    assign_slot_permutations,
    minimum_slot_count,
    validate_slot_assignment,
)

# The Phase 9 8x8 sparsified nonzero pattern (row degrees all 2, max column
# degree 3) on which the old Konig augmenting-path routine did not terminate.
PHASE9_ROWS = [15, 17, 18, 29, 31, 32, 48, 68]


def _phase9_pattern() -> np.ndarray:
    pytest.importorskip("pypower")
    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
    from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import sparsify_block

    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H_block, _, rows, _ = select_deterministic_block(
        np.asarray(system.H_tilde),
        np.asarray(system.r_tilde),
        row_count=8,
        col_count=8,
        policy="largest_row_col_norms",
    )
    assert [int(v) for v in rows] == PHASE9_ROWS
    return np.abs(sparsify_block(H_block, keep_per_row=2)) > 0.0


def test_phase9_pattern_terminates_and_is_valid_at_konig_minimum():
    pattern = _phase9_pattern()
    assert minimum_slot_count(pattern) == 3
    for orientation in (pattern, pattern.T):
        assignment = assign_slot_permutations(orientation)
        assert assignment.slots == 3
        report = validate_slot_assignment(orientation, assignment)
        assert report["valid"] is True
        assert report["real_edges_covered_exactly_once"] is True
        assert assignment.augmenting_visits <= assignment.visit_budget


def test_phase9_pattern_rejects_infeasible_two_slots_instead_of_looping():
    pattern = _phase9_pattern()
    with pytest.raises(ValueError, match="maximum row/column degree 3"):
        assign_slot_permutations(pattern, slots=2)


def test_known_small_graphs():
    # Full K_{3,3} (padded to 4x4 with an isolated vertex) needs 3 slots.
    k33 = np.zeros((4, 4), dtype=bool)
    k33[:3, :3] = True
    assert minimum_slot_count(k33) == 3
    assignment = assign_slot_permutations(k33)
    validate_slot_assignment(k33, assignment)

    # A permutation pattern is a single slot.
    perm = np.eye(4, dtype=bool)[[2, 0, 3, 1]]
    assignment = assign_slot_permutations(perm)
    assert assignment.slots == 1
    validate_slot_assignment(perm, assignment)

    # A path-like degree-2 pattern colors with exactly 2 slots.
    path = np.array(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
        dtype=bool,
    )
    assignment = assign_slot_permutations(path)
    assert assignment.slots == 2
    validate_slot_assignment(path, assignment)


def test_supports_row_degree_two_with_extra_slots():
    pattern = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    assignment = assign_slot_permutations(pattern, slots=3)
    assert assignment.slots == 3
    validate_slot_assignment(pattern, assignment)


def test_deterministic_across_repeat_calls():
    pattern = _phase9_pattern()
    first = assign_slot_permutations(pattern)
    second = assign_slot_permutations(pattern)
    assert first.permutations == second.permutations
    assert first.real_edge_mask == second.real_edge_mask
    assert first.augmenting_visits == second.augmenting_visits


def test_visit_budget_guard_raises_instead_of_hanging():
    k33 = np.zeros((4, 4), dtype=bool)
    k33[:3, :3] = True
    with pytest.raises(RuntimeError, match="visit budget"):
        assign_slot_permutations(k33, max_augment_visits=1)


def test_random_patterns_decompose_at_max_degree():
    rng = np.random.default_rng(20260706)
    for _ in range(40):
        n = int(rng.integers(2, 13))
        pattern = np.zeros((n, n), dtype=bool)
        for i in range(n):
            k = int(rng.integers(0, min(n, 4) + 1))
            if k:
                pattern[i, rng.choice(n, size=k, replace=False)] = True
        if not pattern.any():
            continue
        assignment = assign_slot_permutations(pattern)
        assert assignment.slots == minimum_slot_count(pattern)
        validate_slot_assignment(pattern, assignment)


def test_validator_rejects_double_covered_real_edge():
    pattern = np.array([[1, 0], [0, 1]], dtype=bool)
    bad = SlotAssignment(
        n=2,
        slots=2,
        max_degree=1,
        permutations=((0, 1), (0, 1)),
        real_edge_mask=((True, True), (True, False)),
        augmenting_visits=0,
        visit_budget=10,
    )
    with pytest.raises(ValueError, match="more than one slot"):
        validate_slot_assignment(pattern, bad)


def test_validator_rejects_uncovered_real_edge():
    pattern = np.array([[1, 1], [1, 1]], dtype=bool)
    bad = SlotAssignment(
        n=2,
        slots=2,
        max_degree=2,
        permutations=((0, 1), (1, 0)),
        real_edge_mask=((True, True), (True, False)),
        augmenting_visits=0,
        visit_budget=10,
    )
    with pytest.raises(ValueError, match="coverage does not match"):
        validate_slot_assignment(pattern, bad)
