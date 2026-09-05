"""Rectangular zero-padded workload - equivalence semantics and registered artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.rectangular_padded_workload import (
    RECT_COLS,
    RECT_ROWS,
    WORKLOAD_ID,
    build_rectangular_block,
    classical_padding_equivalence,
    zero_pad,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/rectangular_padded_workload"


def test_extractor_block_is_deterministic_and_rectangular():
    block, residual, rows, cols = build_rectangular_block()
    assert block.shape == (RECT_ROWS, RECT_COLS)
    assert residual.shape == (RECT_ROWS,)
    assert rows == (17, 18, 31, 32, 45, 46, 48, 68)
    assert cols == (2, 3, 16, 17)
    # The 8x4 columns are a subset of the frozen 8x8 selection (nested extractor scores).
    assert set(cols) <= {0, 2, 3, 7, 13, 14, 16, 17}


def test_zero_padding_is_exactly_equivalent_for_any_alpha():
    block, residual, _rows, _cols = build_rectangular_block()
    padded = zero_pad(block)
    assert padded.shape == (8, 8)
    assert np.array_equal(padded[:, :RECT_COLS], block)
    assert not padded[:, RECT_COLS:].any()
    for alpha in (1.0e2, 1.134521e6):
        result = classical_padding_equivalence(block, padded, residual, alpha)
        assert result["equivalent"]
        assert result["max_abs_padded_coordinate"] == 0.0


@pytest.mark.skipif(
    not (OUTPUT_DIR / "workload_registration.csv").is_file(),
    reason="workload not registered",
)
def test_registered_workload_artifacts_are_consistent():
    registration = pd.read_csv(OUTPUT_DIR / "workload_registration.csv").iloc[0]
    assert registration["workload_id"] == WORKLOAD_ID
    assert registration["equivalent"]
    assert registration["rectangular_rank"] == 4
    resource = pd.read_csv(OUTPUT_DIR / "resource_ledger.csv").iloc[0]
    assert resource["workload_id"] == WORKLOAD_ID
    assert int(resource["total_simultaneously_live_qubits"]) == 8
    shots = pd.read_csv(OUTPUT_DIR / "shot_rows.csv")
    assert set(shots["seed"]) == {0, 1, 2}
    assert (shots["shots_attempted"] == 100_000).all()
    # Support never touches the zero-padded columns.
    coords = ast.literal_eval(registration["support_coordinates"])
    assert all(j < RECT_COLS for _i, j in coords)
