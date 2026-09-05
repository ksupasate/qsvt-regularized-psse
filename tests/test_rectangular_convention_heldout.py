"""Tests for held-out rectangular-matrix validation (WP-C).

Two layers:
  1. Artifact test: reads heldout_rectangular_matrix_results.csv and asserts the
     convention passes across all dimensions and spectral families.
  2. Integrity invariant (MUST FAIL if dev matrices are reused): held-out seeds
     must lie in the reserved range [770000, 779999] and must NOT collide with
     documented development seeds, and must follow the preregistered scheme.

Reusing a development-set matrix as held-out evidence is a protocol violation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "heldout_rectangular_matrix_results.csv"
DEV_SEEDS = {0, 1, 2, 10, 42, 123, 2024}
HELDOUT_LO, HELDOUT_HI = 770000, 779999


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


def test_heldout_results_exist(df):
    assert len(df) >= 200


def test_all_pass_within_tolerance(df):
    passed = df[df.status == "pass"]
    assert len(passed) == len(df), "some held-out cases did not convention-pass"
    assert passed.convention_error_vs_encoded.max() < 1e-6


def test_covers_all_dimensions(df):
    expected = {"2x1", "3x2", "4x3", "5x3", "6x4", "8x5", "12x7"}
    assert expected.issubset(set(df.dim.unique()))


def test_covers_spectral_families(df):
    expected = {
        "well_conditioned",
        "moderate_condition",
        "nearly_rank_deficient",
        "exact_zero_singular_values",
        "repeated_singular_values",
        "clustered_singular_values",
        "random_decay",
    }
    assert expected.issubset(set(df.spectral_family.unique()))


def test_heldout_seeds_disjoint_from_dev(df):
    """Integrity invariant: held-out seeds must be disjoint from development seeds."""

    seeds = set(df.seed.unique())
    leaked = seeds & DEV_SEEDS
    assert not leaked, f"held-out seeds collided with development seeds: {leaked}"


def test_heldout_seeds_in_reserved_range(df):
    seeds = set(df.seed.unique())
    out_of_range = {s for s in seeds if not (HELDOUT_LO <= s <= HELDOUT_HI)}
    assert not out_of_range, f"seeds outside reserved range: {out_of_range}"


def test_heldout_seeds_match_preregistered_scheme(df):
    dims = [(2, 1), (3, 2), (4, 3), (5, 3), (6, 4), (8, 5), (12, 7)]
    for di, (m, n) in enumerate(dims):
        sub = df[(df.rows == m) & (df.cols == n)]
        for s in sub.seed.unique():
            assert 770000 + di * 100 <= s < 770000 + (di + 1) * 100
