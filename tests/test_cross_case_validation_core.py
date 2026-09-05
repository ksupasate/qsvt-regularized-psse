"""Cross-case validation - deterministic, leakage, semantic, and arithmetic tests (fast)."""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.cross_case_validation.common import (
    build_case_block_binding,
    build_case_design,
    build_case_tasks,
    generate_case_residual,
)
from robust_qsvt_se.cross_case_validation.selectors import (
    output_aware_sensitivity_scores,
    produce_supports_generalized,
)
from robust_qsvt_se.reviewer_blocking.common import bus_set_is_connected

# ----------------------------------------------------- determinism / mappings


def test_ieee30_block_is_deterministic_and_outcome_independent():
    a = build_case_block_binding("ieee30", 123, row_count=8, col_count=8)
    b = build_case_block_binding("ieee30", 123, row_count=8, col_count=8)
    assert a.selected_rows == b.selected_rows == (35, 50, 51, 65, 80, 81, 118, 159)
    assert a.selected_columns == b.selected_columns == (2, 4, 19, 20, 32, 34, 49, 50)


def test_column_local_to_global_mapping_is_consistent():
    binding = build_case_block_binding("ieee30", 123, row_count=8, col_count=8)
    for record in binding.columns:
        assert binding.selected_columns[record.local_index] == record.global_state_index
        assert record.state_type in {"angle", "voltage"}
        assert record.bus_id >= 1


def test_representable_branches_are_real_network_branches():
    binding = build_case_block_binding("ieee30", 123, row_count=8, col_count=8)
    branch_set = set(binding.branches)
    angle_buses = {r.bus_id for r in binding.columns if r.state_type == "angle"}
    for from_bus, to_bus in binding.representable_angle_branches:
        assert (from_bus, to_bus) in branch_set
        assert from_bus in angle_buses and to_bus in angle_buses


def test_connected_area_verification_matches_topology():
    binding = build_case_block_binding("ieee30", 123, row_count=8, col_count=8)
    volt_buses = set(binding.bus_set("voltage"))
    # IEEE-30 8x8 in-block voltage buses are known to be disconnected (two components).
    connected = bus_set_is_connected(volt_buses, binding.branches)
    design = build_case_design("ieee30", 123, dimension=8)
    area_available = any(
        r.family == "area_aggregate" for r in design.functional_records
    )
    assert area_available == connected


# ----------------------------------------------------- regression anchors


def test_ieee14_8x8_reproduces_frozen_small_design():
    from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import build_small_design

    gen = build_case_design("ieee14", 123, dimension=8)
    frozen = build_small_design(123, 8)
    assert np.array_equal(gen.small.matrix, frozen.matrix)
    assert gen.small.alpha == frozen.alpha
    assert gen.small.selected_rows == frozen.selected_rows
    assert gen.small.selected_columns == frozen.selected_columns


def test_ieee14_residual_matches_frozen_controlled_residual():
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import generate_controlled_residual

    design = build_case_design("ieee14", 123, dimension=8)
    r_gen = generate_case_residual("ieee14", 1000, design.small.selected_rows)
    r_frozen = generate_controlled_residual(1000, selected_rows=design.small.selected_rows)
    assert np.array_equal(r_gen, r_frozen)


def test_sensitivity_scores_match_core_aggregate_on_legacy_tasks():
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
        compute_output_aware_entry_scores,
    )

    design = build_case_design("ieee14", 123, dimension=8)
    legacy_ids = design.legacy_functional_ids
    tasks = build_case_tasks("ieee14", design.small, [1000, 1001], "training", legacy_ids)
    core = compute_output_aware_entry_scores(
        design.small.matrix, tasks, alpha=design.small.alpha, epsilon=1e-15
    )
    mine = output_aware_sensitivity_scores(design.small.matrix, tasks, alpha=design.small.alpha)
    assert np.allclose(mine.sensitivity_mean, core.sensitivity_mean, atol=0, rtol=0)
    assert np.allclose(mine.sensitivity_worst_case, core.sensitivity_worst_case, atol=0, rtol=0)


# ----------------------------------------------------- leakage prevention


def test_score_construction_rejects_heldout_tasks():
    design = build_case_design("ieee30", 123, dimension=8)
    held = build_case_tasks(
        "ieee30", design.small, [2000], "held_out", design.physical_functional_ids
    )
    with pytest.raises(ValueError, match="leakage"):
        output_aware_sensitivity_scores(design.small.matrix, held, alpha=design.small.alpha)


# ----------------------------------------------------- four-quadrant classify


def test_four_quadrant_classification_is_correct():
    from robust_qsvt_se.reviewer_blocking.joint_feasibility import classify

    assert classify(True, True) == "application_useful_qsvt_feasible"
    assert classify(True, False) == "application_useful_qsvt_infeasible"
    assert classify(False, True) == "application_not_useful_qsvt_feasible"
    assert classify(False, False) == "neither_useful_nor_qsvt_feasible"


# ----------------------------------------------------- resource arithmetic


def test_cost_model_matches_declared_formula():
    from robust_qsvt_se.reviewer_blocking.resource_pareto import cost_model

    executed = {"signal_unitary_gate_count": 1000, "postselection_probability": 0.5}
    dimension, degree, shots = 8, 31, 100000
    result = cost_model(executed, dimension=dimension, degree=degree, shots=shots)
    c_load = 2 * dimension - 2
    c_readout = dimension
    expected = (c_load + degree * 1000 + c_readout) / 0.5
    assert result["c_total_gates"] == pytest.approx(expected)
    assert result["modeled_c_load_gates"] == c_load
    assert result["modeled_c_readout_gates"] == c_readout
    assert result["executed_c_signal_gates"] == 1000


def test_cost_model_infinite_when_postselection_zero():
    from robust_qsvt_se.reviewer_blocking.resource_pareto import cost_model

    executed = {"signal_unitary_gate_count": 100, "postselection_probability": 0.0}
    result = cost_model(executed, dimension=8, degree=31, shots=1000)
    assert result["c_total_gates"] == float("inf")


# ----------------------------------------------------- retained infeasible rows


def test_infeasible_and_skipped_rows_are_retained():
    design = build_case_design("ieee30", 123, dimension=8)
    tasks = build_case_tasks(
        "ieee30", design.small, [1000, 1001], "training", design.physical_functional_ids
    )
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import SupportConstraints
    from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import ExactLossEvaluator

    constraints = SupportConstraints(12, 3, True)
    evaluator = ExactLossEvaluator(design.small.matrix, tasks, design.small.alpha, 1e-6)
    outcomes = produce_supports_generalized(
        design.small, tasks, constraints, evaluator=evaluator,
        score_functional_ids=design.physical_functional_ids,
        random_seed=314159, beam_width=6, beam_max_steps=16,
        oracle_max_candidates=3_000_000, include_oracle=False,
        near_oracle_max_loss_evals=1,  # force near-oracle to be skipped by the ceiling
    )
    assert outcomes["near_oracle_mean"]["status"] == "skipped_compute_ceiling"
    assert outcomes["near_oracle_mean"]["support"] is None
    assert "estimated_beam_loss_evals" in outcomes["near_oracle_mean"]
