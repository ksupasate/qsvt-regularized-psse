"""Robust companion statistics - Holm adjustment, frozen consistency, and produced rows."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs/robust_companion_statistics/robust_companion_statistics.csv"
CHECKSUMS_PATH = ROOT / "outputs/robust_companion_statistics/checksums.sha256"
TABLE_PATH = ROOT / "manuscript/tables/robust_companion_statistics.tex"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_robust_companion_statistics",
        ROOT / "scripts" / "build_robust_companion_statistics.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_holm_adjustment_is_step_down_monotone():
    module = _load_script()
    adjusted = module.holm_adjust([0.01, 0.04, 0.03, 0.20])
    # Ordered raw: 0.01, 0.03, 0.04, 0.20 -> 4*0.01, 3*0.03, 2*0.04, 1*0.20 with monotonicity.
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert module.holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="companion statistics not generated")
def test_generated_rows_consistent_with_frozen_bootstrap():
    rows = pd.read_csv(CSV_PATH)
    assert len(rows) == 4
    assert (rows["pre_declared_contrast_count"] == 4).all()
    assert (rows["paired_structure_count"] == 12).all()
    assert rows["consistent_with_frozen_mean"].all()
    # Sign bookkeeping: wins + losses + ties partition the 12 structures.
    total = rows["n_positive"] + rows["n_negative"] + rows["n_ties"]
    assert (total == 12).all()
    # Direction agreement with the frozen bootstrap mean.
    assert (
        np.sign(rows["mean_effect"]) == np.sign(rows["frozen_observed_mean_effect"])
    ).all()
    # Holm p-values never fall below their raw counterparts.
    assert (rows["sign_test_p_holm"] >= rows["sign_test_p_exact"] - 1e-15).all()
    assert (rows["wilcoxon_p_holm"] >= rows["wilcoxon_p_exact"] - 1e-15).all()


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="companion statistics not generated")
def test_agreement_labels_match_columns():
    rows = pd.read_csv(CSV_PATH)
    for row in rows.itertuples():
        significant = row.wilcoxon_p_holm < 0.05
        if row.agreement_with_bootstrap == "significant_same_direction_as_bootstrap":
            assert significant and row.frozen_ci_excludes_zero
        elif row.agreement_with_bootstrap == "inconclusive_consistent_with_bootstrap":
            assert (not significant) and (not row.frozen_ci_excludes_zero)


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="companion statistics not generated")
def test_manuscript_only_mode_leaves_legacy_evidence_byte_identical(monkeypatch):
    # Required guard (W6 historical-evidence preservation): regenerating only the
    # publication-facing table must not rewrite the frozen legacy CSV, and the CSV
    # must still match its frozen checksum. The .tex is snapshot-restored so the
    # test leaves the working tree byte-for-byte unchanged.
    module = _load_script()
    csv_before = CSV_PATH.read_bytes()
    tex_before = TABLE_PATH.read_bytes()
    frozen_sha = hashlib.sha256(csv_before).hexdigest()
    try:
        monkeypatch.setattr(
            sys, "argv", ["build_robust_companion_statistics.py", "--manuscript-only"]
        )
        module.main()
        assert CSV_PATH.read_bytes() == csv_before
        assert frozen_sha in CHECKSUMS_PATH.read_text(encoding="utf-8")
    finally:
        TABLE_PATH.write_bytes(tex_before)


def test_frozen_csv_guard_rejects_modified_legacy_evidence(tmp_path):
    # The manuscript-only hash guard must pass on a matching checksum and fail
    # fast (SystemExit) on a drift, without touching the real frozen evidence.
    module = _load_script()
    csv = tmp_path / "robust_companion_statistics.csv"
    checksums = tmp_path / "checksums.sha256"
    csv.write_text("original,content\n1,2\n", encoding="utf-8")

    matching = hashlib.sha256(csv.read_bytes()).hexdigest()
    checksums.write_text(f"{matching}  robust_companion_statistics.csv\n", encoding="utf-8")
    module._assert_legacy_csv_matches_frozen(csv, checksums)  # no raise on match

    checksums.write_text(f"{'0' * 64}  robust_companion_statistics.csv\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        module._assert_legacy_csv_matches_frozen(csv, checksums)
