"""Tests for the Phase 9 leakage-aware bridge audit."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase8_bridge_characterization import _coupling_diagnostics
from robust_qsvt_se.paper.phase9_bridge_leakage_aware import (
    SELECTION_RULES,
    leakage_aware_block,
    run_phase9_bridge_leakage_aware,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system


@pytest.fixture(scope="module")
def ieee30_system():
    system, _ = build_engineering_system(
        {
            "case_name": "ieee30",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    return H, r


def test_leakage_aware_rule_is_deterministic_and_wellformed(ieee30_system):
    H, r = ieee30_system
    _block1, _res1, rows1, cols1 = leakage_aware_block(H, r, 8)
    _block2, _res2, rows2, cols2 = leakage_aware_block(H, r, 8)
    np.testing.assert_array_equal(rows1, rows2)
    np.testing.assert_array_equal(cols1, cols2)
    assert cols1.size == 8 and rows1.size == 8
    assert len(set(cols1.tolist())) == 8
    assert len(set(rows1.tolist())) == 8
    assert int(cols1[0]) == int(np.argmax(np.linalg.norm(H, axis=0)))


def test_leakage_aware_rule_uses_no_post_solve_inputs():
    import inspect

    source = inspect.getsource(leakage_aware_block)
    for banned in ("ridge_svd_solution", "ridge_update", "solve(", "full_update"):
        assert banned not in source


def test_leakage_aware_reduces_functional_column_leakage(ieee30_system):
    """The leakage-aware rule keeps more of its target column than the norm rule."""

    H, r = ieee30_system
    _, _, la_rows, la_cols = leakage_aware_block(H, r, 8)
    _, _, nr_rows, nr_cols = select_deterministic_block(
        H, r, row_count=8, col_count=8, policy="largest_row_col_norms"
    )
    la_leak = _coupling_diagnostics(H, r, la_rows, la_cols)["functional_column_leakage"]
    nr_leak = _coupling_diagnostics(H, r, nr_rows, nr_cols)["functional_column_leakage"]
    assert la_leak <= nr_leak
    assert la_leak < 0.2  # most of the target column energy is retained


def test_end_to_end_small_case_set(tmp_path):
    run = run_phase9_bridge_leakage_aware(
        {"output_dir": str(tmp_path), "cases": ("ieee14", "ieee30"), "command": "test"}
    )
    summary = run["rule_summary"]
    assert set(summary["selection_rule"]) == set(SELECTION_RULES)
    # Leakage-aware achieves the lowest median discrepancy of the four rules.
    ordered = summary.sort_values("median_relative_discrepancy")
    assert ordered.iloc[0]["selection_rule"] == "leakage_aware"
    # Discrepancy correlates positively with functional-column leakage.
    assert run["correlations"]["functional_column_leakage"] > 0.4
    for name in [
        "bridge_leakage_aware_full.csv",
        "bridge_leakage_aware_rule_summary.csv",
        "discrepancy_vs_leakage_scatter.png",
        "README.md",
        "checksums.sha256",
        "manifest.json",
    ]:
        assert (tmp_path / name).is_file(), name
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []


def test_manifest_records_leakage_aware_definition(tmp_path):
    run_phase9_bridge_leakage_aware(
        {"output_dir": str(tmp_path), "cases": ("ieee14",), "command": "test"}
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "leakage_aware" in manifest["selection_rules"]
    assert "pre-solve" in manifest["leakage_aware_rule_definition"]
