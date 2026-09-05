"""Tests for the Phase 9 lambda accuracy-realizability tradeoff (pure helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.paper.phase9_lambda_tradeoff import (
    BENCHMARK_LAMBDA,
    LAMBDA_GRID,
    TARGET_FIT_ERROR,
    _block_specs,
    _domain_min,
    _min_feasible_degree_for_fit,
)

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")


def test_lambda_grid_and_benchmark_are_consistent():
    assert min(LAMBDA_GRID) == BENCHMARK_LAMBDA
    assert 6.9e-2 in LAMBDA_GRID  # the lambda-matched anchor value is in the grid
    assert tuple(sorted(LAMBDA_GRID, reverse=True)) == LAMBDA_GRID


def test_block_specs_build_the_four_blocks():
    specs = _block_specs(123)
    ids = [s["block_id"] for s in specs]
    assert "ieee14_4x4_anchor" in ids
    assert "ieee14_8x8_lambda_matched" in ids
    assert "controlled_8x8_kappa_1e4" in ids
    by_id = {s["block_id"]: s for s in specs}
    kappa_8 = float(
        np.linalg.svd(by_id["ieee14_8x8_lambda_matched"]["H_block"], compute_uv=False).max()
        / np.linalg.svd(by_id["ieee14_8x8_lambda_matched"]["H_block"], compute_uv=False).min()
    )
    assert kappa_8 == pytest.approx(132.84, abs=0.5)
    controlled = by_id["controlled_8x8_kappa_1e4"]
    assert controlled["block_kind"] == "controlled_svd_stress"
    assert controlled["H_block"].shape == (8, 8)


def test_min_feasible_degree_is_monotonic_in_lambda():
    """Smaller lambda (sharper filter) needs a higher minimum feasible degree."""

    specs = {s["block_id"]: s for s in _block_specs(123)}
    H = np.asarray(specs["ieee14_4x4_anchor"]["H_block"], dtype=np.float64)
    singular = np.linalg.svd(H, compute_uv=False)
    beta = float(singular.max())
    domain_min = _domain_min(singular, beta)

    degrees = []
    for lam in (1.0e-1, 6.9e-2, 3.0e-2):
        alpha = lam * beta**2
        min_degree, _trace = _min_feasible_degree_for_fit(beta, alpha, domain_min, TARGET_FIT_ERROR)
        assert min_degree > 0  # feasible in the tested range at these lambdas
        degrees.append(min_degree)
    # Non-decreasing minimum degree as lambda shrinks.
    assert degrees[0] <= degrees[1] <= degrees[2]


def test_uniform_proxy_and_fit_are_pure_and_reproducible():
    specs = {s["block_id"]: s for s in _block_specs(123)}
    H = np.asarray(specs["ieee14_8x8_lambda_matched"]["H_block"], dtype=np.float64)
    singular = np.linalg.svd(H, compute_uv=False)
    beta = float(singular.max())
    domain_min = _domain_min(singular, beta)
    first, _ = _min_feasible_degree_for_fit(beta, 0.069 * beta**2, domain_min, TARGET_FIT_ERROR)
    second, _ = _min_feasible_degree_for_fit(beta, 0.069 * beta**2, domain_min, TARGET_FIT_ERROR)
    assert first == second
