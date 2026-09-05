from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.experiments.metrics import (
    condition_number,
    residual_norm,
    rmse,
    weighted_residual,
    weighted_residual_norm,
    weighted_residual_quadratic,
)
from robust_qsvt_se.measurement.linear_system import WeightedSystem


def test_rmse() -> None:
    x_hat = np.array([1.0, 3.0])
    x_true = np.array([1.0, 1.0])

    assert rmse(x_hat, x_true) == pytest.approx(np.sqrt(2.0))


def test_residual_metrics_on_weighted_system() -> None:
    system = WeightedSystem(
        H_tilde=np.array([[1.0, 0.0], [0.0, 2.0]]),
        r_tilde=np.array([1.0, 2.0]),
        x_true=np.array([1.0, 1.0]),
    )
    x_hat = np.array([0.0, 1.0])

    assert residual_norm(system, x_hat) == pytest.approx(1.0)
    assert weighted_residual(system, x_hat) == pytest.approx(1.0)
    assert weighted_residual_norm(system, x_hat) == pytest.approx(1.0)
    assert weighted_residual_quadratic(system, x_hat) == pytest.approx(
        weighted_residual_norm(system, x_hat) ** 2
    )


def test_condition_number_metric() -> None:
    system = WeightedSystem(
        H_tilde=np.diag([4.0, 2.0]),
        r_tilde=np.array([0.0, 0.0]),
    )

    assert condition_number(system) == pytest.approx(2.0)


def test_rank_deficient_condition_number_is_infinite() -> None:
    system = WeightedSystem(
        H_tilde=np.diag([1.0, 0.0]),
        r_tilde=np.array([0.0, 0.0]),
    )

    assert np.isinf(condition_number(system))

    # A finite near-singular system has condition number sigma_max / sigma_min exactly.
    near_singular = WeightedSystem(
        H_tilde=np.diag([8.0, 1.0e-3]),
        r_tilde=np.array([0.0, 0.0]),
    )
    assert abs(condition_number(near_singular) - 8000.0) < 1e-6
