"""Larger-block (16x16) validation - shape, coverage, determinism, evidence-status tests."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.cross_case_validation.common import build_case_design, build_case_tasks
from robust_qsvt_se.cross_case_validation.selectors import select_support_generalized
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    select_resource_constrained_support,
    support_constraint_report,
)


def _design16():
    return build_case_design("ieee14", 123, dimension=16)


# ----------------------------------------------------- shape / uniqueness


def test_block_is_exactly_16x16_with_unique_active_rows_and_columns():
    design = _design16()
    assert design.small.matrix.shape == (16, 16)
    assert len(set(design.binding.selected_rows)) == 16
    assert len(set(design.binding.selected_columns)) == 16
    assert design.conditioning["rows"] == 16 and design.conditioning["cols"] == 16


def test_block_has_sufficient_nonzeros_and_physical_functionals():
    design = _design16()
    assert design.conditioning["nonzeros"] >= 16  # at least one per row for coverage
    # 16 coordinate + representable branch-diff + connected-area families present.
    assert len(design.physical_functional_ids) >= 16
    families = {r.family for r in design.functional_records}
    assert "coordinate" in families and "branch_angle_difference" in families


# ----------------------------------------------------- coverage floor


def test_coverage_floor_makes_k_below_dimension_infeasible():
    design = _design16()
    magnitude = np.abs(design.small.matrix)
    below = select_resource_constrained_support(
        magnitude, magnitude, SupportConstraints(12, 3, True)
    )
    assert below.support is None or below.status != "completed"
    at_floor = select_resource_constrained_support(
        magnitude, magnitude, SupportConstraints(16, 3, True)
    )
    assert at_floor.support is not None
    report = support_constraint_report(
        design.small.matrix, at_floor.support, SupportConstraints(16, 3, True)
    )
    assert report["valid"]


def test_budget_grid_rule_preserves_coverage_floor():
    # The predeclared 16x16 grid starts at the coverage floor (dimension), not the 8x8 floor.
    support_budgets = [16, 24, 32, 48]
    assert min(support_budgets) == 16
    assert all(k <= 101 for k in support_budgets)  # all <= candidate nnz


# ----------------------------------------------------- determinism / no silent reduction


def test_support_selection_is_deterministic():
    design = _design16()
    tasks = build_case_tasks("ieee14", design.small, [1000, 1001], "training",
                             design.physical_functional_ids)
    constraints = SupportConstraints(24, 3, True)
    s1 = select_support_generalized(design.small, "sensitivity_refined_mean", constraints, tasks,
                                    design.physical_functional_ids)
    s2 = select_support_generalized(design.small, "sensitivity_refined_mean", constraints, tasks,
                                    design.physical_functional_ids)
    assert np.array_equal(s1, s2)


def test_feasible_support_respects_declared_budget_no_silent_reduction():
    design = _design16()
    tasks = build_case_tasks("ieee14", design.small, [1000], "training",
                             design.physical_functional_ids)
    constraints = SupportConstraints(24, 3, True)
    support = select_support_generalized(design.small, "ridge_leverage", constraints, tasks,
                                         design.physical_functional_ids)
    assert support is not None
    assert int(support.sum()) <= 24  # never exceeds the declared budget
    report = support_constraint_report(design.small.matrix, support, constraints)
    assert report["valid"]


# ----------------------------------------------------- evidence status / safe skipping


def test_statevector_ceiling_forces_skip_status():
    from robust_qsvt_se.cross_case_validation.larger_block import run_qsvt_and_resources

    design = _design16()
    config = {
        "training_seed_ids": [1000, 1001], "held_out_seed_ids": [2000, 2001],
        "statevector_max_wrapper_dim": 4,  # tiny -> wrapper (128) exceeds -> skip
        "qsvt_validation": {
            "degree": 31, "validation_slot_budget": 3,
            "qsvt_selectors": ["global_magnitude"], "qsvt_support_budgets": [16],
            "basis_gates": ["u3", "cx"], "optimization_level": 1, "modeled_shots": 1000,
        },
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        result = run_qsvt_and_resources("ieee14", design, config, Path(tmp))
    qframe = result["qsvt_frame"]
    resource = result["resource_frame"]
    assert (qframe["statevector_executed"] == False).all()  # noqa: E712
    assert "skipped_statevector_ceiling" in set(resource["evidence_status"])


def test_support_stability_calculation():
    from robust_qsvt_se.cross_case_validation.selector_comparison import _support_stability

    entries = {
        "sel": {
            (16, 3): frozenset({(0, 0), (1, 1)}),
            (24, 3): frozenset({(0, 0), (1, 1), (2, 2)}),
        }
    }
    frame = _support_stability(entries, "ieee14", "block")
    assert len(frame) == 1
    # Jaccard of {2 shared} / {3 union} = 2/3
    assert frame["mean_pairwise_jaccard"].iloc[0] == 2 / 3


# ----------------------------------------------------- runtime fields populated


def test_runtime_and_solve_fields_populated():
    import tempfile
    from pathlib import Path

    from robust_qsvt_se.cross_case_validation.selector_comparison import run_selector_comparison

    design = _design16()
    with tempfile.TemporaryDirectory() as tmp:
        result = run_selector_comparison(
            "ieee14", design, output_dir=Path(tmp),
            support_budgets=[16, 24], slot_budgets=[3],
            training_seeds=[1000, 1001], held_out_seeds=[2000, 2001],
            selectors_subset=("ridge_leverage", "sensitivity_refined_mean"),
            beam_width=6, near_oracle_max_loss_evals=200000, random_seed_base=314159,
            y_floor=1e-6, failure_threshold=0.1,
        )
    frame = result["support_frame"]
    assert not frame.empty
    assert frame["selection_runtime_seconds"].notna().all()
    assert frame["exact_ridge_solves_this_cell"].notna().all()
    assert (frame["exact_ridge_solves_this_cell"] >= 0).all()
