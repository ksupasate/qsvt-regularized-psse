"""Finite-difference and validation tests for the exact Ridge output derivative."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from robust_qsvt_se.qsvt.ridge_output_certificate import (
    ridge_selected_output_gradient,
    ridge_selected_output_via_solve,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import predetermined_selected_functionals


def _central_difference(
    matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
    step: float = 1.0e-6,
) -> np.ndarray:
    finite = np.empty_like(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plus = matrix.copy()
            minus = matrix.copy()
            plus[row, column] += step
            minus[row, column] -= step
            finite[row, column] = (
                ridge_selected_output_via_solve(plus, residual, functional, alpha)
                - ridge_selected_output_via_solve(minus, residual, functional, alpha)
            ) / (2.0 * step)
    return finite


@pytest.mark.parametrize("seed", [3, 17, 41])
def test_analytic_gradient_matches_central_differences_for_multiple_residuals(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(8, 8))
    residual = rng.normal(size=8)
    alpha = 0.7
    for functional in predetermined_selected_functionals(8).values():
        analytic = ridge_selected_output_gradient(matrix, residual, functional, alpha)
        finite = _central_difference(matrix, residual, functional, alpha)
        relative = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
        assert relative < 2.0e-8


def test_gradient_formula_is_stable_for_small_and_large_entries():
    matrix = np.diag([1.0e-4, 2.0e-2, 1.0, 30.0])
    matrix += np.array(
        [
            [0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.1, 0.0, 0.0, 0.0],
        ]
    )
    residual = np.array([0.2, -0.7, 1.1, 0.4])
    functional = np.array([0.5, 0.5, -0.5, 0.5])
    analytic = ridge_selected_output_gradient(matrix, residual, functional, 0.25)
    finite = _central_difference(matrix, residual, functional, 0.25, step=2.0e-7)
    assert np.linalg.norm(analytic - finite) / np.linalg.norm(finite) < 5.0e-8


@pytest.mark.parametrize(
    ("matrix", "residual", "functional", "alpha", "message"),
    [
        (np.ones(4), np.ones(2), np.ones(2), 1.0, "two-dimensional"),
        (np.eye(3), np.ones(2), np.ones(3), 1.0, "residual"),
        (np.eye(3), np.ones(3), np.ones(2), 1.0, "functional"),
        (np.eye(3), np.ones(3), np.ones(3), 0.0, "alpha"),
        (np.eye(3), np.array([1.0, np.nan, 2.0]), np.ones(3), 1.0, "finite"),
    ],
)
def test_invalid_gradient_inputs_are_rejected(
    matrix, residual, functional, alpha, message
):
    with pytest.raises(ValueError, match=message):
        ridge_selected_output_gradient(matrix, residual, functional, alpha)


def test_complex_inputs_are_explicitly_rejected():
    matrix = np.eye(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="real arrays only"):
        ridge_selected_output_gradient(matrix, np.ones(3), np.ones(3), 1.0)


def test_gradient_implementation_does_not_form_an_explicit_inverse():
    source = inspect.getsource(ridge_selected_output_gradient)
    assert "np.linalg.inv" not in source
    assert "np.linalg.solve" in source
