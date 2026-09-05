from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.qsvt_update_workflow import (
    best_real_sign_l2_error,
    compute_ridge_update_reference,
    prepare_weighted_residual_state,
    run_qsvt_update_state_simulation,
)


def test_prepare_weighted_residual_state_returns_unit_state() -> None:
    result = prepare_weighted_residual_state(np.array([3.0, 4.0, 0.0]))

    assert result.residual_norm == pytest.approx(5.0)
    assert np.linalg.norm(result.normalized_residual_state) == pytest.approx(1.0)
    assert result.padded_dimension == 4
    assert result.padding_width == 1


def test_prepare_weighted_residual_state_rejects_zero_residual() -> None:
    with pytest.raises(ValueError, match="residual norm"):
        prepare_weighted_residual_state(np.zeros(2))


def test_ridge_reference_matches_existing_ridge_estimator() -> None:
    H = np.array([[2.0, 0.0], [0.0, 0.5], [1.0, -0.25]])
    r = np.array([1.0, -0.1, 0.2])
    alpha = 1.0e-3
    reference = compute_ridge_update_reference(H, r, alpha)
    estimator_result = RidgeEstimator(alpha).solve(
        WeightedSystem(H_tilde=H, r_tilde=r, x_true=np.zeros(2), metadata={})
    )

    np.testing.assert_allclose(reference.update_vector, estimator_result.x_hat)


def test_qsvt_update_state_aligns_with_ridge_on_tiny_matrix() -> None:
    H = np.array([[0.8, 0.1], [0.0, 0.5]])
    r = np.array([1.0, -0.25])

    result = run_qsvt_update_state_simulation(
        H,
        r,
        alpha=1.0e-2,
        degree=5,
        block_encoding_mode="explicit_dense",
        phase_method="pennylane_poly_to_angles",
        seed=7,
    )

    assert result.simulation_mode == "explicit_dense_block_encoded_qsvt"
    assert result.block_encoding_report["top_left_block_error"] < 1.0e-10
    assert result.phase_aligned_state_l2_error < 0.25
    assert result.normalized_state_overlap_abs > 0.95


def test_best_real_sign_l2_error_handles_sign_flip() -> None:
    reference = np.array([1.0, 0.0])
    candidate = np.array([-1.0, 0.0])

    assert best_real_sign_l2_error(reference, candidate) == pytest.approx(0.0)
