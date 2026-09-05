from __future__ import annotations

import numpy as np

from robust_qsvt_se.estimators.hhl_style_inverse_proxy import HHLStyleInverseProxyEstimator
from robust_qsvt_se.estimators.normal_equation_wls import NormalEquationWLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.qsvt_unregularized_inverse import (
    QSVTUnregularizedInverseEstimator,
)
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.measurement.linear_system import WeightedSystem


def test_estimators_recover_known_full_rank_system() -> None:
    H_tilde = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    x_true = np.array([0.5, -1.0])
    system = WeightedSystem(H_tilde=H_tilde, r_tilde=H_tilde @ x_true, x_true=x_true)

    estimators = [
        PseudoinverseEstimator(rcond=1.0e-14),
        NormalEquationWLSEstimator(),
        RidgeEstimator(alpha=1.0e-12),
        TruncatedSVDEstimator(tau=1.0e-14),
        QSVTSpectralEstimator(alpha=1.0e-12),
        QSVTUnregularizedInverseEstimator(cutoff=1.0e-14),
        HHLStyleInverseProxyEstimator(cutoff=1.0e-14),
    ]

    for estimator in estimators:
        result = estimator.solve(system)
        assert not result.failed
        np.testing.assert_allclose(result.x_hat, x_true, atol=1.0e-9)
        assert result.rmse is not None
        assert result.rmse < 1.0e-9


def test_regularized_estimators_do_not_explode_on_tiny_singular_value() -> None:
    H_tilde = np.diag([1.0, 1.0e-8])
    x_true = np.array([1.0, 1.0])
    r_tilde = H_tilde @ x_true
    r_tilde[1] += 1.0e-4
    system = WeightedSystem(H_tilde=H_tilde, r_tilde=r_tilde, x_true=x_true)

    pinv = PseudoinverseEstimator(rcond=0.0).solve(system)
    ridge = RidgeEstimator(alpha=1.0e-3).solve(system)
    qsvt = QSVTSpectralEstimator(alpha=1.0e-3).solve(system)
    qsvt_unregularized = QSVTUnregularizedInverseEstimator(cutoff=1.0e-12).solve(system)

    assert np.linalg.norm(pinv.x_hat) > 1000.0
    assert np.linalg.norm(qsvt_unregularized.x_hat) > 1000.0
    assert np.linalg.norm(ridge.x_hat) < 2.0
    assert np.linalg.norm(qsvt.x_hat) < 2.0
    assert (
        qsvt.extra_diagnostics["simulation_scope"] == "classical singular-value filter simulation"
    )
    assert qsvt_unregularized.extra_diagnostics["unstable_ablation"]


def test_truncated_svd_removes_unstable_small_direction() -> None:
    H_tilde = np.diag([1.0, 1.0e-8])
    x_true = np.array([1.0, 1.0])
    r_tilde = H_tilde @ x_true
    r_tilde[1] += 1.0e-4
    system = WeightedSystem(H_tilde=H_tilde, r_tilde=r_tilde, x_true=x_true)

    result = TruncatedSVDEstimator(tau=1.0e-6).solve(system)

    np.testing.assert_allclose(result.x_hat, np.array([1.0, 0.0]), atol=1.0e-12)
    assert result.extra_diagnostics["retained_singular_directions"] == 1


def test_normal_equation_records_squared_condition_number() -> None:
    system = WeightedSystem(
        H_tilde=np.diag([1.0, 1.0e-4]),
        r_tilde=np.array([1.0, 1.0e-4]),
        x_true=np.ones(2),
    )

    result = NormalEquationWLSEstimator().solve(system)

    assert not result.failed
    assert result.extra_diagnostics["H_tilde_condition_number"] == np.linalg.cond(system.H_tilde)
    assert result.extra_diagnostics["normal_matrix_condition_number"] >= (
        result.condition_number**2 * 0.99
    )


def test_normal_equation_can_fail_where_svd_pseudoinverse_succeeds() -> None:
    system = WeightedSystem(
        H_tilde=np.diag([1.0, 1.0e-8]),
        r_tilde=np.array([1.0, 1.0e-8]),
        x_true=np.ones(2),
    )

    pinv = PseudoinverseEstimator(rcond=0.0).solve(system)
    normal = NormalEquationWLSEstimator(max_gain_condition_number=1.0e12).solve(system)

    assert not pinv.failed
    assert normal.failed
    assert "condition number" in str(normal.failure_reason)
    assert normal.extra_diagnostics["normal_matrix_condition_number"] > 1.0e12


def test_normal_equation_records_failure_on_singular_gain() -> None:
    system = WeightedSystem(
        H_tilde=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        r_tilde=np.array([1.0, 2.0, 3.0]),
    )

    result = NormalEquationWLSEstimator().solve(system)

    assert result.failed
    assert "singular" in str(result.failure_reason).lower()
    assert result.extra_diagnostics["normal_matrix_condition_number"] > 1.0e15


def test_hhl_style_proxy_reports_resource_and_instability() -> None:
    system = WeightedSystem(
        H_tilde=np.diag([1.0, 1.0e-10]),
        r_tilde=np.array([1.0, 1.0e-6]),
        x_true=np.ones(2),
    )

    result = HHLStyleInverseProxyEstimator(
        cutoff=1.0e-8,
        precision=1.0e-3,
        instability_condition_threshold=1.0e6,
    ).solve(system)

    assert not result.failed
    assert result.extra_diagnostics["hhl_instability_flag"]
    assert result.extra_diagnostics["hhl_effective_condition_number"] == 1.0e8
    assert result.extra_diagnostics["hhl_resource_proxy"] > 1.0e18
    assert "no circuit execution" in result.extra_diagnostics["simulation_scope"]


def test_weighted_system_shape_validation() -> None:
    try:
        WeightedSystem(H_tilde=np.ones((2, 2)), r_tilde=np.ones(3))
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("WeightedSystem accepted invalid shapes")

    # A consistent (m x n) system is accepted and keeps its declared shape.
    valid = WeightedSystem(H_tilde=np.ones((3, 2)), r_tilde=np.ones(3))
    assert valid.H_tilde.shape == (3, 2)
