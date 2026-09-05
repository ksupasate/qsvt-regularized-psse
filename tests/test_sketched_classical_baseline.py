"""Sketched classical baseline - protocol identity, sanity anchors, and table extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs/sketched_classical_baseline/sketched_baseline_rows.csv"
TABLE_PATH = ROOT / "manuscript/tables/audit_classical_comparison.tex"
FROZEN_PATH = ROOT / (
    "outputs/final_falsification_and_submission/classical_baseline_reproduction.csv"
)


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="sketched baseline not generated")
def test_sketched_rows_structure_and_sanity():
    rows = pd.read_csv(CSV_PATH)
    assert len(rows) == 6  # 3 sketch sizes x {gaussian sketch, row subsample}
    assert rows["is_primary_table_row"].sum() == 1
    primary = rows.loc[rows["is_primary_table_row"]].iloc[0]
    assert primary["method"] == "sketched_Ridge_s54"
    assert (rows["sketch_seed"] == 20260718).all()
    assert (rows["timing_reps"] == 30).all()
    # Full-size uniform subsample (s = m = 82) must reproduce dense Ridge exactly.
    full = rows.loc[rows["method"] == "row_subsampled_Ridge_s82"].iloc[0]
    assert full["output_error_vs_ridge"] < 1.0e-12
    # Sketch quality improves monotonically with sketch size for the Gaussian family.
    gaussian = rows[rows["family"] == "gaussian_sketch_and_solve"].sort_values("sketch_rows")
    errors = gaussian["output_error_vs_ridge"].to_numpy()
    assert (errors[1:] <= errors[:-1]).all()
    # Timing separation: solve-only never slower than the full query.
    assert (rows["per_query_solve_only_s"] <= rows["per_query_s"] + 1e-9).all()


@pytest.mark.skipif(
    not (CSV_PATH.is_file() and TABLE_PATH.is_file() and FROZEN_PATH.is_file()),
    reason="table or inputs missing",
)
def test_table_contains_frozen_rows_and_sketched_row():
    text = TABLE_PATH.read_text(encoding="utf-8")
    frozen = pd.read_csv(FROZEN_PATH)
    for method in frozen["method"]:
        assert method.replace("_", "\\_") in text
    assert "sketched\\_Ridge\\_s54" in text
    assert "identical query protocol" in text
