from __future__ import annotations

import pandas as pd

from robust_qsvt_se.paper.tqe_claim_boundary_audit import (
    CLAIM_PHRASES,
    FLAG_COLUMNS,
    audit_claim_boundaries,
    classify_severity,
    scan_text_file,
    suggested_replacement,
)


def test_claim_phrase_detection(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("QSVT outperforms Ridge on this benchmark.\n", encoding="utf-8")

    rows = scan_text_file(report)

    assert len(rows) == 1
    assert rows[0]["phrase"] == "QSVT outperforms Ridge"
    assert rows[0]["severity"] == "critical"


def test_severity_classification_safe_context():
    phrase = next(item for item in CLAIM_PHRASES if item.phrase == "quantum speedup")

    severity = classify_severity("No quantum speedup is claimed.", phrase)

    assert severity == "low"


def test_suggested_replacement_for_unsupported_claim():
    phrase = next(item for item in CLAIM_PHRASES if item.phrase == "hardware execution")

    replacement = suggested_replacement(phrase, "critical")

    assert "simulation" in replacement


def test_claim_boundary_audit_output_schema(tmp_path):
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    (scan_root / "draft.md").write_text(
        "This is selected-subproblem evidence. No hardware execution is claimed.\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"

    run = audit_claim_boundaries({"output_root": output_root, "scan_roots": [scan_root]})

    assert list(run["flags"].columns) == FLAG_COLUMNS
    assert run["artifacts"]["flags_csv"].exists()
    loaded = pd.read_csv(run["artifacts"]["flags_csv"])
    assert list(loaded.columns) == FLAG_COLUMNS
