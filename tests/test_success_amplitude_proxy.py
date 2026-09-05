from __future__ import annotations

import pytest

from robust_qsvt_se.qsvt.norm_success import (
    estimate_success_probability_by_sampling,
    estimate_success_probability_iterative_proxy,
)


def test_success_sampling_proxy_is_reproducible() -> None:
    first = estimate_success_probability_by_sampling(0.2, 1000, 123)
    second = estimate_success_probability_by_sampling(0.2, 1000, 123)

    assert first == second
    assert first.standard_error == pytest.approx((0.2 * 0.8 / 1000) ** 0.5)


def test_amplitude_estimation_proxy_is_reproducible() -> None:
    first = estimate_success_probability_iterative_proxy(0.2, 1000, 123)
    second = estimate_success_probability_iterative_proxy(0.2, 1000, 123)

    assert first == second
    assert first.standard_error_proxy == pytest.approx((0.2 * 0.8) ** 0.5 / 1000)
    assert "proxy" in first.limitation
