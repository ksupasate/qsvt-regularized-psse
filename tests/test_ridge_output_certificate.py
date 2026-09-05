"""Mathematical and separation tests for the conservative Ridge certificate."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robust_qsvt_se.qsvt.ridge_output_certificate import (
    RidgeSelectedOutputCertificate,
    compute_ridge_selected_output_certificate,
    ridge_linearized_signed_error,
    ridge_selected_output_via_solve,
    validate_ridge_selected_output_certificate,
)


def _operator(matrix: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(
        matrix.T @ matrix + alpha * np.eye(matrix.shape[1]), matrix.T
    )


@pytest.fixture
def problem():
    rng = np.random.default_rng(81)
    matrix = rng.normal(size=(7, 5))
    sparse = matrix.copy()
    sparse[np.abs(sparse) < 0.7] = 0.0
    residual = rng.normal(size=7)
    functional = rng.normal(size=5)
    return matrix, sparse, residual, functional, 0.9


def test_certificate_bound_is_computed_without_actual_error(problem):
    matrix, sparse, residual, functional, alpha = problem
    certificate = compute_ridge_selected_output_certificate(
        matrix, sparse, residual, functional, alpha
    )
    assert isinstance(certificate, RidgeSelectedOutputCertificate)
    assert certificate.actual_selected_output_error is None
    assert certificate.certificate_holds is None
    assert certificate.tightness_ratio is None
    assert certificate.selected_output_bound > 0.0
    assert dataclasses.is_dataclass(certificate)


def test_forward_and_reverse_operator_forms_are_both_valid(problem):
    matrix, sparse, residual, functional, alpha = problem
    certificate = compute_ridge_selected_output_certificate(
        matrix, sparse, residual, functional, alpha
    )
    actual_operator_difference = np.linalg.norm(
        _operator(matrix, alpha) - _operator(sparse, alpha), ord=2
    )
    assert actual_operator_difference <= certificate.operator_bound_forward + 1.0e-14
    assert actual_operator_difference <= certificate.operator_bound_reverse + 1.0e-14


def test_selected_output_scaling_and_posthoc_validation(problem):
    matrix, sparse, residual, functional, alpha = problem
    raw = compute_ridge_selected_output_certificate(
        matrix, sparse, residual, functional, alpha
    )
    expected = np.linalg.norm(functional) * np.linalg.norm(residual) * min(
        raw.operator_bound_forward, raw.operator_bound_reverse
    )
    assert raw.selected_output_bound == pytest.approx(expected, rel=1.0e-14)
    actual = abs(
        ridge_selected_output_via_solve(matrix, residual, functional, alpha)
        - ridge_selected_output_via_solve(sparse, residual, functional, alpha)
    )
    validated = validate_ridge_selected_output_certificate(raw, actual)
    assert validated.certificate_holds is True
    assert validated.actual_selected_output_error == actual
    assert validated.tightness_ratio == pytest.approx(
        validated.selected_output_bound / actual
    )


@pytest.mark.parametrize("seed", range(8))
def test_bound_holds_for_multiple_support_perturbations(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(8, 8))
    support = rng.random((8, 8)) > 0.55
    sparse = np.where(support, matrix, 0.0)
    residual = rng.normal(size=8)
    functional = rng.normal(size=8)
    alpha = 0.2 + seed / 10.0
    raw = compute_ridge_selected_output_certificate(
        matrix, sparse, residual, functional, alpha
    )
    actual = abs(
        ridge_selected_output_via_solve(matrix, residual, functional, alpha)
        - ridge_selected_output_via_solve(sparse, residual, functional, alpha)
    )
    assert validate_ridge_selected_output_certificate(raw, actual).certificate_holds


def test_first_order_prediction_is_separate_from_certificate(problem):
    matrix, _sparse, residual, functional, alpha = problem
    perturbation = np.zeros_like(matrix)
    perturbation[2, 3] = 1.0e-5
    candidate = matrix + perturbation
    predicted = ridge_linearized_signed_error(
        matrix, candidate, residual, functional, alpha
    )
    actual = (
        ridge_selected_output_via_solve(candidate, residual, functional, alpha)
        - ridge_selected_output_via_solve(matrix, residual, functional, alpha)
    )
    assert predicted == pytest.approx(actual, rel=2.0e-5, abs=1.0e-12)
    assert not isinstance(predicted, RidgeSelectedOutputCertificate)
