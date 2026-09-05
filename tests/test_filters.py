from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.filters import (
    inverse_filter,
    qsvt_regularized_filter,
    qsvt_unregularized_inverse_filter,
    ridge_filter,
    truncated_inverse_filter,
)


def test_qsvt_regularized_filter_matches_target_formula() -> None:
    sigma = np.array([0.0, 0.1, 1.0, 10.0])
    alpha = 0.01

    expected = sigma / (sigma**2 + alpha)

    np.testing.assert_allclose(qsvt_regularized_filter(sigma, alpha), expected)


def test_ridge_filter_controls_small_singular_values() -> None:
    sigma = np.array([1.0e-12, 1.0])

    filtered = ridge_filter(sigma, alpha=1.0e-2)

    assert filtered[0] < 1.0e-9
    assert filtered[1] == pytest.approx(1.0 / 1.01)


def test_truncated_inverse_filter_zeros_values_below_tau() -> None:
    sigma = np.array([0.001, 0.1, 1.0])

    filtered = truncated_inverse_filter(sigma, tau=0.01)

    np.testing.assert_allclose(filtered, np.array([0.0, 10.0, 1.0]))


def test_inverse_filter_uses_absolute_cutoff() -> None:
    sigma = np.array([0.0, 1.0e-4, 1.0])

    filtered = inverse_filter(sigma, eps=1.0e-3)

    np.testing.assert_allclose(filtered, np.array([0.0, 0.0, 1.0]))


def test_qsvt_unregularized_inverse_filter_grows_as_sigma_decreases() -> None:
    sigma = np.array([1.0, 0.1, 0.01])

    filtered = qsvt_unregularized_inverse_filter(sigma, cutoff=1.0e-6)

    assert filtered[0] < filtered[1] < filtered[2]


def test_qsvt_unregularized_inverse_filter_cutoff_prevents_infinities() -> None:
    sigma = np.array([0.0, 1.0e-12, 1.0])

    filtered = qsvt_unregularized_inverse_filter(sigma, cutoff=1.0e-6)

    assert np.all(np.isfinite(filtered))
    np.testing.assert_allclose(filtered, np.array([1.0e6, 1.0e6, 1.0]))


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (ridge_filter, {"alpha": 0.0}),
        (qsvt_regularized_filter, {"alpha": -1.0}),
        (qsvt_unregularized_inverse_filter, {"cutoff": 0.0}),
        (truncated_inverse_filter, {"tau": -1.0}),
        (inverse_filter, {"eps": -1.0}),
    ],
)
def test_invalid_filter_parameters_raise(fn, kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn(np.array([1.0]), **kwargs)


def test_negative_singular_values_raise() -> None:
    with pytest.raises(ValueError):
        ridge_filter(np.array([-1.0]), alpha=0.1)
