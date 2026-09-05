"""Implementation verification: finite-difference Jacobian validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.jacobian_validation import (
    AC_DETAIL_COLUMNS,
    AC_SUMMARY_COLUMNS,
    WEIGHTED_COLUMNS,
    build_jacobian_validation,
    compare_jacobian_block,
    finite_difference_jacobian,
    has_nonpositive_variance,
    weighted_jacobian,
)


def test_finite_difference_matches_known_analytic_toy_function() -> None:
    # h(x) = [x0^2, x0*x1, sin(x2)] -> analytic Jacobian is known in closed form.
    def func(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2, x[0] * x[1], np.sin(x[2])])

    x = np.array([1.3, -0.7, 0.4])
    analytic = np.array(
        [
            [2 * x[0], 0.0, 0.0],
            [x[1], x[0], 0.0],
            [0.0, 0.0, np.cos(x[2])],
        ]
    )
    fd = finite_difference_jacobian(func, x, epsilon=1e-6)
    assert np.allclose(fd, analytic, atol=1e-6)
    metrics = compare_jacobian_block(analytic, fd)
    assert metrics["status"] == "pass"
    assert metrics["frobenius_relative_error"] < 1e-4


def test_compare_block_flags_wrong_analytic_jacobian_as_fail() -> None:
    fd = np.array([[1.0, 2.0], [3.0, 4.0]])
    wrong = fd.copy()
    wrong[0, 0] = 5.0  # large, non-near-zero error
    metrics = compare_jacobian_block(wrong, fd)
    assert metrics["status"] == "fail"


def test_near_zero_rows_do_not_cause_false_failure() -> None:
    fd = np.array([[2.0, 0.0], [0.0, 0.0]])
    analytic = np.array([[2.0, 1e-12], [1e-13, 0.0]])  # tiny near-zero deviations only
    metrics = compare_jacobian_block(analytic, fd)
    assert metrics["status"] == "pass"


def test_weighted_jacobian_equals_r_inv_sqrt_h() -> None:
    h = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    stds = np.array([0.5, 2.0, 0.1])
    expected = np.diag(1.0 / stds) @ h  # R^{-1/2} H with R = diag(std^2)
    assert np.allclose(weighted_jacobian(h, stds), expected)


def test_invalid_variance_is_caught() -> None:
    assert has_nonpositive_variance(np.array([1.0, 0.0, 2.0])) is True
    assert has_nonpositive_variance(np.array([1.0, -1.0])) is True
    assert has_nonpositive_variance(np.array([1.0, 2.0])) is False
    try:
        weighted_jacobian(np.eye(2), np.array([1.0, 0.0]))
    except ValueError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("weighted_jacobian must reject non-positive variance")


def test_ac_validation_outputs_required_columns(tmp_path: Path) -> None:
    run = build_jacobian_validation(
        {"output_dir": str(tmp_path / "jac"), "cases": ["ieee14"], "case_source": "pypower"}
    )
    detail = pd.read_csv(run["artifacts"]["ac_jacobian_finite_difference_validation"])
    summary = pd.read_csv(run["artifacts"]["ac_jacobian_validation_summary"])
    assert list(detail.columns) == AC_DETAIL_COLUMNS
    assert list(summary.columns) == AC_SUMMARY_COLUMNS
    assert run["ac_status"] == "pass"
    assert run["ac_max_relative_error"] < 1e-4


def test_ac_angle_row_recorded_not_fabricated(tmp_path: Path) -> None:
    run = build_jacobian_validation(
        {"output_dir": str(tmp_path / "jac"), "cases": ["ieee14"], "case_source": "pypower"}
    )
    summary = pd.read_csv(run["artifacts"]["ac_jacobian_validation_summary"])
    angle = summary[summary["measurement_type"] == "angle_row"]
    assert not angle.empty
    assert (angle["status"] == "not_implemented").all()
    assert (angle["num_rows_checked"] == 0).all()


def test_dc_validation_matches_finite_differences(tmp_path: Path) -> None:
    run = build_jacobian_validation(
        {"output_dir": str(tmp_path / "jac"), "cases": ["ieee14"], "case_source": "pypower"}
    )
    summary = pd.read_csv(run["artifacts"]["dc_jacobian_validation_summary"])
    assert (summary["workflow"] == "dc").all()
    computed = summary[summary["status"] != "not_implemented"]
    assert (computed["status"] == "pass").all()
    # DC angle rows are validated when angle_buses is supplied.
    assert "dc_angle_row" in set(summary["measurement_type"])
    assert run["dc_status"] == "pass"


def test_weighted_audit_records_convention_and_separates_conditioning(tmp_path: Path) -> None:
    run = build_jacobian_validation(
        {"output_dir": str(tmp_path / "jac"), "cases": ["ieee14"], "case_source": "pypower"}
    )
    weighted = pd.read_csv(run["artifacts"]["weighted_jacobian_consistency_audit"])
    assert list(weighted.columns) == WEIGHTED_COLUMNS
    consistent = weighted[weighted["status"] == "consistent"]
    assert not consistent.empty
    assert (consistent["diagonal_covariance_assumed"]).all()
    # Weighted and unweighted condition numbers are recorded separately (not conflated).
    row = consistent.iloc[0]
    assert row["condition_weighted"] != row["condition_unweighted"]
    assert "R^{-1/2}" in str(row["weighting_source"])
