from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.partial_observable_readout import (
    basis_probability,
    estimate_bernoulli_probability,
    estimate_overlap_from_hadamard_proxy,
    linear_functional_overlap,
    normalize_state,
    subset_probability,
)


def test_normalize_state_returns_unit_state_and_original_norm() -> None:
    state, norm = normalize_state(np.array([3.0, 4.0], dtype=np.float64))

    assert norm == pytest.approx(5.0)
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_probability_helpers_match_numpy_calculation() -> None:
    state, _ = normalize_state(np.array([1.0, 2.0, 2.0], dtype=np.float64))

    assert basis_probability(state, 1) == pytest.approx(abs(state[1]) ** 2)
    assert subset_probability(state, [0, 2]) == pytest.approx(
        abs(state[0]) ** 2 + abs(state[2]) ** 2
    )


def test_linear_functional_overlap_normalizes_coefficients() -> None:
    state, _ = normalize_state(np.array([1.0, -1.0, 0.0], dtype=np.float64))
    coeffs = np.array([2.0, -2.0, 0.0], dtype=np.float64)

    overlap = linear_functional_overlap(state, coeffs)

    expected = np.vdot(coeffs / np.linalg.norm(coeffs), state)
    assert overlap == pytest.approx(expected)


def test_bernoulli_estimator_is_reproducible_with_fixed_seed() -> None:
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first = estimate_bernoulli_probability(0.35, 1000, first_rng)
    second = estimate_bernoulli_probability(0.35, 1000, second_rng)

    assert first == second


def test_hadamard_proxy_is_reproducible_and_converges_for_large_shots() -> None:
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)
    overlap = 0.25 + 0.1j

    first = estimate_overlap_from_hadamard_proxy(overlap, 20000, first_rng, component="real")
    second = estimate_overlap_from_hadamard_proxy(overlap, 20000, second_rng, component="real")

    assert first == second
    assert abs(first[0] - overlap.real) < 0.03
    assert first[1] == pytest.approx(np.sqrt((1.0 - overlap.real**2) / 20000.0))
