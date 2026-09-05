"""Workstream 3 tests: oracle separation, QSVT boundedness, and evidence status."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import frozen_8x8_design_as_small
from robust_qsvt_se.reviewer_blocking.joint_feasibility import (
    build_full_system,
    classify,
    evaluate_qsvt_feasibility,
    regularization_selection,
)


def test_classify_covers_four_quadrants():
    assert classify(True, True) == "application_useful_qsvt_feasible"
    assert classify(True, False) == "application_useful_qsvt_infeasible"
    assert classify(False, True) == "application_not_useful_qsvt_feasible"
    assert classify(False, False) == "neither_useful_nor_qsvt_feasible"


def test_regularization_selection_separates_oracle_from_deployable():
    system = build_full_system(123)
    grid = np.logspace(-2, 8, 21)
    records = {r["selection_method"]: r for r in regularization_selection(system, grid)}
    assert records["oracle_rmse"]["is_oracle_diagnostic"] is True
    assert records["oracle_rmse"]["evidence_status"] == "oracle_simulation_diagnostic"
    for method in ("gcv", "l_curve", "discrepancy", "heldout_rows"):
        assert records[method]["is_oracle_diagnostic"] is False
        assert records[method]["evidence_status"] == "deployable_non_oracle"
    # Oracle achieves the best (or tied-best) true-state RMSE by construction.
    oracle_rmse = records["oracle_rmse"]["full_state_rmse"]
    for method in ("gcv", "l_curve", "discrepancy", "heldout_rows"):
        assert records[method]["full_state_rmse"] >= oracle_rmse - 1e-12


def test_nonoracle_selectors_do_not_require_truth():
    from robust_qsvt_se.paper.tqe_revision_core import (
        select_alpha_gcv,
        select_alpha_heldout,
        select_alpha_l_curve,
    )

    system = build_full_system(123)
    grid = np.logspace(-2, 8, 21)
    # These signatures take (H, r, alphas) only - never x_true.
    for selector in (select_alpha_gcv, select_alpha_l_curve, select_alpha_heldout):
        alpha = selector(system.matrix, system.residual, grid)
        assert grid.min() <= alpha <= grid.max()


def _frozen_block(tmp_path):
    design = frozen_8x8_design_as_small(tmp_path)
    return design.matrix


def test_qsvt_feasibility_unbounded_tiny_lambda_is_retained(tmp_path):
    block = _frozen_block(tmp_path)
    slots = int(minimum_slot_count(block.T != 0.0))
    beta = slots * float(np.max(np.abs(block)))
    alpha = 1e-8 * beta**2  # lambda ~ 1e-8 -> filter far too sharp to bound
    feas = evaluate_qsvt_feasibility(
        block,
        alpha,
        31,
        margin=1.05,
        bound_tolerance=0.002,
        uniform_tolerance=0.002,
        execute_statevector=False,
        residual_unit=None,
        cache_dir=tmp_path / "cache",
        attempt_phase_synthesis=False,
    )
    assert feas.boundedness_ok is False
    assert feas.phase_synthesis_status.startswith("skipped")
    assert feas.evidence_status != "executed_statevector"  # retained, not conflated


def test_qsvt_feasibility_bounded_region_without_phase_synthesis(tmp_path):
    block = _frozen_block(tmp_path)
    slots = int(minimum_slot_count(block.T != 0.0))
    beta = slots * float(np.max(np.abs(block)))
    alpha = 0.07 * beta**2  # lambda ~ 0.07, the frozen-design QSVT-feasible band
    feas = evaluate_qsvt_feasibility(
        block,
        alpha,
        31,
        margin=1.05,
        bound_tolerance=0.002,
        uniform_tolerance=0.05,
        execute_statevector=False,
        residual_unit=None,
        cache_dir=tmp_path / "cache",
        attempt_phase_synthesis=False,
    )
    assert feas.boundedness_ok is True
    assert feas.phase_synthesis_status == "not_attempted_by_request"
    assert feas.evidence_status != "executed_statevector"
    assert 0.0 < feas.normalized_lambda < 1.0


def test_evidence_status_modeled_when_statevector_disabled(tmp_path):
    block = _frozen_block(tmp_path)
    slots = int(minimum_slot_count(block.T != 0.0))
    beta = slots * float(np.max(np.abs(block)))
    feas = evaluate_qsvt_feasibility(
        block,
        0.07 * beta**2,
        31,
        margin=1.05,
        bound_tolerance=0.002,
        uniform_tolerance=0.05,
        execute_statevector=False,
        residual_unit=None,
        cache_dir=tmp_path / "cache",
        attempt_phase_synthesis=False,
    )
    assert feas.evidence_status == "modeled_analytic_fit"
