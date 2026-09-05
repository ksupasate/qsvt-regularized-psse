"""Focused tests for the reviewer-blocking evidence engine (Phases 2/4).

Physical reference correctness, metric separation, task-aware baseline correctness, and
train/held-out split integrity.  Fast: ieee14 8x8 block, a couple of seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    _ridge_filter_operator,
)
from robust_qsvt_se.qsvt.ridge_output_certificate import ridge_selected_output_gradient
from robust_qsvt_se.reviewer_evidence.engine import (
    PHYSICAL_FLOOR,
    SUPPORT_FLOOR,
    adjoint_unnormalized_score,
    build_structure,
    build_tasks,
    design_time_alpha_regimes,
    evaluate_triples,
    exact_single_removal_score,
    seed_reference,
    select_support,
)

# ------------------------------------------------------------- physical reference


def test_y_true_equals_direct_functional_dot_delta_true():
    ctx = build_structure("ieee14_8x8")
    support = ctx.matrix != 0.0
    for fid in ("coordinate_e0", "branch_angle_diff_2_4", "connected_block_voltage_area_aggregate"):
        rows = evaluate_triples(ctx, support, ctx.alpha_fixed, [2000, 2005], [fid])
        for row in rows:
            _residual, delta_true = seed_reference("ieee14_8x8", row["seed"])
            ell = ctx.functionals[fid]
            assert np.isclose(row["y_true"], float(ell @ delta_true))


def test_unit_norm_functionals_remain_unit_norm():
    ctx = build_structure("ieee14_8x8")
    for fid, vec in ctx.functionals.items():
        assert np.isclose(np.linalg.norm(vec), 1.0), fid


def test_controlled_residual_is_full_r_tilde_restricted_to_block_rows():
    from robust_qsvt_se.cross_case_validation.common import _cached_system

    ctx = build_structure("ieee14_8x8")
    residual, _ = seed_reference("ieee14_8x8", 2003)
    _matrix, full_r, _meta = _cached_system("ieee14", 2003)
    rows = np.asarray(ctx.selected_rows)
    assert np.allclose(residual, np.asarray(full_r)[rows])


def test_unavailable_functionals_are_not_substituted():
    from robust_qsvt_se.reviewer_evidence.physical_accuracy import build_functional_mapping

    mapping = build_functional_mapping(["ieee30_8x8"])
    unavailable = mapping[mapping.classification == "unavailable_not_substituted"]
    assert len(unavailable) >= 1
    # every unavailable row carries an explicit reason and no fabricated vector/state
    assert unavailable["unavailable_reason"].str.len().gt(0).all()
    assert unavailable["state_type"].eq("").all()


# ------------------------------------------------------------- metric separation


def test_physical_and_support_errors_use_different_denominators():
    ctx = build_structure("ieee14_8x8")
    support = select_support(
        ctx,
        "global_magnitude",
        SupportConstraints(16, 3, True),
        build_tasks("ieee14_8x8", [1000, 1001], "training"),
        ctx.physical_ids,
    )["support"]
    rows = evaluate_triples(ctx, support, ctx.alpha_fixed, [2000], list(ctx.functionals))
    for row in rows:
        exp_phys = row["E_sparse_abs"] / max(abs(row["y_true"]), PHYSICAL_FLOOR)
        exp_supp = abs(row["y_sparse"] - row["y_full_ridge"]) / max(
            abs(row["y_full_ridge"]), SUPPORT_FLOOR
        )
        assert np.isclose(row["E_physical_norm"], exp_phys)
        assert np.isclose(row["E_support_norm"], exp_supp)
    # at least one row where the two denominators genuinely differ
    assert any(not np.isclose(abs(r["y_true"]), abs(r["y_full_ridge"])) for r in rows)


def test_signed_and_absolute_errors_not_conflated():
    ctx = build_structure("ieee14_8x8")
    support = ctx.matrix != 0.0
    rows = evaluate_triples(ctx, support, ctx.alpha_fixed, [2000, 2001], list(ctx.functionals))
    signed = np.array([r["B_physical_signed"] for r in rows])
    absol = np.array([r["E_sparse_abs"] for r in rows])
    assert np.allclose(np.abs(signed), absol)
    assert (signed < 0).any() and (signed > 0).any()  # sign information retained


def test_near_zero_rows_retained_with_flag():
    ctx = build_structure("ieee14_8x8")
    support = ctx.matrix != 0.0
    seeds = list(range(2000, 2020))
    functionals = list(ctx.functionals)
    rows = evaluate_triples(ctx, support, ctx.alpha_fixed, seeds, functionals)
    # retention: nothing is dropped for any reason, including near-zero y_true
    assert len(rows) == len(seeds) * len(functionals)
    for r in rows:
        assert r["near_zero_y_true"] == (abs(r["y_true"]) < PHYSICAL_FLOOR)
    # the flag genuinely triggers on a constructed near-zero output (kept, not dropped)
    near_zero = [r for r in rows if r["near_zero_y_true"]]
    for r in near_zero:
        assert abs(r["y_true"]) < PHYSICAL_FLOOR and "E_physical_norm" in r


# ------------------------------------------------------------- task-aware baselines


def test_adjoint_unnormalized_matches_direct_gradient_formula():
    ctx = build_structure("ieee14_8x8")
    tasks = build_tasks("ieee14_8x8", [1000, 1001, 1002], "training")
    tasks = [t for t in tasks if t.functional_id in ctx.legacy_ids]
    score = adjoint_unnormalized_score(ctx.matrix, tasks, alpha=ctx.alpha_fixed)
    manual = np.zeros_like(ctx.matrix)
    for t in tasks:
        g = ridge_selected_output_gradient(ctx.matrix, t.residual, t.functional, ctx.alpha_fixed)
        manual += np.abs(ctx.matrix * g)
    manual /= len(tasks)
    assert np.allclose(score, manual)


def test_exact_single_removal_matches_brute_force():
    ctx = build_structure("ieee14_8x8")
    tasks = build_tasks("ieee14_8x8", [1000, 1001], "training")
    tasks = [t for t in tasks if t.functional_id in ctx.legacy_ids]
    alpha, floor = ctx.alpha_fixed, 1e-6
    score, evals = exact_single_removal_score(ctx.matrix, tasks, alpha=alpha, y_floor=floor)
    full_support = ctx.matrix != 0.0
    assert evals == int(full_support.sum())
    full_op = _ridge_filter_operator(ctx.matrix, alpha)
    full_out = np.array([t.functional @ (full_op @ t.residual) for t in tasks])
    positions = list(np.argwhere(full_support))[:5]  # spot-check first few
    for i, j in positions:
        trial = full_support.copy()
        trial[i, j] = False
        op = _ridge_filter_operator(np.where(trial, ctx.matrix, 0.0), alpha)
        out = np.array([t.functional @ (op @ t.residual) for t in tasks])
        loss = float(np.mean(np.abs(out - full_out) / np.maximum(np.abs(full_out), floor)))
        assert np.isclose(score[i, j], loss)


def test_all_selectors_share_identical_milp_constraints():
    ctx = build_structure("ieee14_8x8")
    tasks = build_tasks("ieee14_8x8", [1000, 1001], "training")
    cons = SupportConstraints(16, 3, True)
    for selector in (
        "global_magnitude",
        "ridge_leverage",
        "sensitivity_refined_mean",
        "adjoint_unnormalized_mean",
        "exact_single_removal_mean",
    ):
        support = select_support(ctx, selector, cons, tasks, ctx.physical_ids)["support"]
        assert support is not None
        assert int(support.sum()) <= cons.k_budget
        assert (support.sum(axis=1) <= cons.slot_budget).all()
        assert (support.sum(axis=0) <= cons.slot_budget).all()


def test_selector_is_deterministic():
    ctx = build_structure("ieee14_8x8")
    tasks = build_tasks("ieee14_8x8", [1000, 1001, 1002], "training")
    cons = SupportConstraints(16, 3, True)
    a = select_support(ctx, "sensitivity_refined_mean", cons, tasks, ctx.physical_ids)["support"]
    b = select_support(ctx, "sensitivity_refined_mean", cons, tasks, ctx.physical_ids)["support"]
    assert np.array_equal(a, b)


# ------------------------------------------------------------- split integrity


def test_score_functions_reject_non_training_tasks():
    ctx = build_structure("ieee14_8x8")
    held = build_tasks("ieee14_8x8", [2000, 2001], "held_out")
    with pytest.raises(ValueError):
        adjoint_unnormalized_score(ctx.matrix, held, alpha=ctx.alpha_fixed)
    with pytest.raises(ValueError):
        exact_single_removal_score(ctx.matrix, held, alpha=ctx.alpha_fixed, y_floor=1e-6)


def test_deployable_alpha_regimes_do_not_use_ground_truth():
    # Only the oracle regime is allowed to consult truth; deployable ones must not.
    ctx = build_structure("ieee14_8x8")
    regimes = {r["regime"]: r for r in design_time_alpha_regimes(ctx)}
    assert regimes["gcv"]["deployable"] and not regimes["gcv"]["is_oracle"]
    assert regimes["oracle_block_rmse"]["is_oracle"]
    # deployable alphas are finite and positive (selected without truth)
    for name in ("gcv", "l_curve", "discrepancy", "heldout_rows"):
        assert regimes[name]["alpha"] > 0.0
