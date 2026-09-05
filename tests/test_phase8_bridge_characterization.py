"""Tests for the Phase 8 bridge-discrepancy characterization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.paper.phase8_bridge_characterization import (
    BLOCK_SIZES,
    SELECTION_RULES,
    _bridge_metrics,
    run_phase8_bridge_characterization,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution

pytest.importorskip("pypower")

_CASES = ("ieee14", "ieee30")
_EXISTING_BRIDGE_CSV = Path("outputs/selected_block_bridge/selected_block_full_system_bridge.csv")


@pytest.fixture(scope="module")
def bridge_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase8_bridge")
    return run_phase8_bridge_characterization(
        {
            "output_dir": str(output_dir),
            "table_path": str(output_dir / "phase8_bridge_characterization.tex"),
            "cases": _CASES,
        }
    )


def test_every_case_size_rule_cell_runs_or_records_reason(bridge_run):
    frame = bridge_run["full_frame"]
    sweep = frame[frame["selection_rule"] != "provenance_bridge_row"]
    assert len(sweep) == len(_CASES) * len(BLOCK_SIZES) * len(SELECTION_RULES)
    assert set(sweep["status"].unique()) <= {"computed", "skipped"}
    skipped = sweep[sweep["status"] == "skipped"]
    assert (skipped["skipped_with_reason"].str.len() > 0).all()
    computed = sweep[sweep["status"] == "computed"]
    assert (computed["skipped_with_reason"] == "").all()
    assert np.isfinite(computed["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64)).all()


def test_discrepancy_formula_on_synthetic_example():
    rng = np.random.default_rng(7)
    full = rng.standard_normal((6, 5))
    residual = rng.standard_normal(6)
    sel_rows = np.asarray([0, 2], dtype=np.int64)
    sel_cols = np.asarray([1, 3], dtype=np.int64)
    block = full[np.ix_(sel_rows, sel_cols)]
    block_residual = residual[sel_rows]
    lam = 0.069
    metrics = _bridge_metrics(full, residual, block, block_residual, sel_rows, sel_cols, lam=lam)
    beta = float(np.linalg.svd(block, compute_uv=False)[0])
    alpha = lam * beta**2
    full_update = ridge_svd_solution(full, residual, alpha=alpha)
    block_update = ridge_svd_solution(block, block_residual, alpha=alpha)
    expected_full = float(full_update[sel_cols[0]])
    expected_block = float(block_update[0])
    assert metrics["alpha"] == pytest.approx(alpha, rel=1.0e-12)
    assert metrics["full_selected_functional"] == pytest.approx(expected_full, rel=1.0e-12)
    assert metrics["block_selected_functional"] == pytest.approx(expected_block, rel=1.0e-12)
    assert metrics["absolute_discrepancy_delta_l"] == pytest.approx(
        abs(expected_full - expected_block), rel=1.0e-12
    )
    assert metrics["relative_discrepancy_vs_full"] == pytest.approx(
        abs(expected_full - expected_block) / abs(expected_full), rel=1.0e-12
    )


def test_identical_block_gives_zero_discrepancy():
    rng = np.random.default_rng(11)
    full = rng.standard_normal((4, 4))
    residual = rng.standard_normal(4)
    indices = np.arange(4, dtype=np.int64)
    metrics = _bridge_metrics(full, residual, full, residual, indices, indices, lam=0.069)
    assert metrics["absolute_discrepancy_delta_l"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["out_of_block_coupling_fraction"] == pytest.approx(0.0, abs=1.0e-15)
    assert metrics["functional_column_leakage"] == pytest.approx(0.0, abs=1.0e-15)


def test_existing_bridge_rows_preserved(bridge_run):
    if not _EXISTING_BRIDGE_CSV.is_file():
        pytest.skip("original bridge artifact not present")
    original = pd.read_csv(_EXISTING_BRIDGE_CSV)
    frame = bridge_run["full_frame"]
    provenance = frame[frame["selection_rule"] == "provenance_bridge_row"].reset_index(drop=True)
    assert len(provenance) == len(original)
    np.testing.assert_allclose(
        provenance["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64),
        original["relative_discrepancy_vs_full"].to_numpy(dtype=np.float64),
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        provenance["full_selected_functional"].to_numpy(dtype=np.float64),
        original["full_selected_functional"].to_numpy(dtype=np.float64),
        rtol=1.0e-9,
    )


def test_required_outputs_and_checksums_validate(bridge_run):
    output_dir = bridge_run["output_dir"]
    for name in [
        "README.md",
        "bridge_characterization_summary.csv",
        "bridge_characterization_full.csv",
        "bridge_characterization_by_case_size_rule.csv",
        "bridge_characterization_manifest.json",
        "checksums.sha256",
    ]:
        assert (output_dir / name).is_file(), name
    for line in (output_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = output_dir / name.strip()
        if not path.is_file():
            continue  # the manuscript table lives outside the output directory
        recomputed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert recomputed == digest, name
    manifest = json.loads(
        (output_dir / "bridge_characterization_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["experiment_id"] == "phase8_bridge_characterization"
    assert "diagnostic_spearman_correlations" in manifest


def test_wording_is_claim_safe(bridge_run):
    output_dir = bridge_run["output_dir"]
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    table = (output_dir / "phase8_bridge_characterization.tex").read_text(encoding="utf-8")
    assert forbidden_in(table) == []
    assert "surrogate" in table
