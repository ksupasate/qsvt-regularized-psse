from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.codesigned_bounded_targets import (
    SUMMARY_COLUMNS,
    build_codesigned_solution,
    evaluate_codesigned_target_grid,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(1)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    # A near-degenerate leading pair stresses the polynomial fit (Runge overshoot).
    singular_values = np.array([5.0, 4.95, 1.2, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.4, -0.3, 0.2, 0.6])
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata={})


def test_qsvt_polynomial_validation_detects_overshoot() -> None:
    subproblem = _subproblem()
    solution = build_codesigned_solution(
        subproblem, alpha=1.0e-3, degree=25, target_family="weighted_support_ls"
    )
    assert solution is not None
    # The bounded coefficients are QSVT-safe and pass validation.
    validate_qsvt_polynomial(
        solution.bounded_coefficients, parity="odd", grid_size=8193, bound_tolerance=1.0e-4
    )
    assert solution.qsvt_safe is True

    # Doubling the bounded coefficients breaks the boundedness bound: validation must raise.
    with pytest.raises(ValueError):
        validate_qsvt_polynomial(
            2.0 * solution.bounded_coefficients,
            parity="odd",
            grid_size=8193,
            bound_tolerance=1.0e-4,
        )


def test_weighted_support_fitting_reduces_weighted_error() -> None:
    subproblem = _subproblem()
    rows, _ = evaluate_codesigned_target_grid(
        subproblem=subproblem,
        alphas=[1.0e-3],
        degrees=[35],
        target_families=["weighted_support_ls", "monotone_clipped"],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
    )
    by_family = {row["target_family"]: row for row in rows}
    weighted = float(by_family["weighted_support_ls"]["weighted_filter_error"])
    clipped = float(by_family["monotone_clipped"]["weighted_filter_error"])

    # The singular-support-weighted fit tracks the true regularized target more closely
    # on the high-weight directions than the shape-changing clipped target.
    assert weighted <= clipped + 1.0e-12


def test_codesigned_summary_rows_have_required_columns() -> None:
    subproblem = _subproblem()
    rows, _ = evaluate_codesigned_target_grid(
        subproblem=subproblem,
        alphas=[1.0e-3],
        degrees=[15],
        target_families=["weighted_support_ls", "residual_aware", "direction_preserving"],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
    )
    for row in rows:
        for column in SUMMARY_COLUMNS:
            assert column in row
        assert bool(row["qsvt_safe"]) is True
