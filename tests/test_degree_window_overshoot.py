from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.degree_window_overshoot import (
    DEGREE_WINDOW_CLASSES,
    SUMMARY_COLUMNS,
    detect_overshoot,
    evaluate_degree_window,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(7)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    singular_values = np.array([5.0, 4.9, 1.3, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.4, -0.3, 0.2, 0.6])
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata={})


def test_detect_overshoot_catches_bound_break_and_direction_blowup() -> None:
    # max|p| over the bound triggers overshoot even with an aligned direction.
    assert (
        detect_overshoot(
            max_abs_polynomial_on_grid=1.5, direction_error_vs_ridge=0.001, overshoot_margin=1.0
        )
        is True
    )
    # Direction-error blowup triggers overshoot even with a bounded polynomial.
    assert (
        detect_overshoot(
            max_abs_polynomial_on_grid=1.0, direction_error_vs_ridge=2.0, overshoot_margin=1.0
        )
        is True
    )
    # Off-support peak dominating the on-support response triggers overshoot.
    assert (
        detect_overshoot(
            max_abs_polynomial_on_grid=1.0, direction_error_vs_ridge=0.001, overshoot_margin=3.0
        )
        is True
    )
    # A clean, bounded, aligned, on-support target is not flagged.
    assert (
        detect_overshoot(
            max_abs_polynomial_on_grid=1.0, direction_error_vs_ridge=0.001, overshoot_margin=1.05
        )
        is False
    )


def test_degree_window_classification_is_deterministic() -> None:
    subproblem = _subproblem()
    kwargs = dict(
        subproblem=subproblem,
        alphas=[1.0e-4, 1.0e-2],
        degrees=[15, 25, 35],
        target_families=["weighted_support_ls", "residual_aware"],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
    )
    first = evaluate_degree_window(**kwargs)
    second = evaluate_degree_window(**kwargs)

    assert [row["degree_window_class"] for row in first] == [
        row["degree_window_class"] for row in second
    ]
    for row in first:
        assert row["degree_window_class"] in DEGREE_WINDOW_CLASSES
        for column in SUMMARY_COLUMNS:
            assert column in row


def test_feasible_rows_are_marked_for_gate_validation() -> None:
    rows = evaluate_degree_window(
        subproblem=_subproblem(),
        alphas=[1.0e-4],
        degrees=[15, 25, 35],
        target_families=["weighted_support_ls", "residual_aware"],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
    )
    for row in rows:
        if row["degree_window_class"] == "residual_feasible":
            assert bool(row["gate_validation_recommended"]) is True
        else:
            assert bool(row["gate_validation_recommended"]) is False
