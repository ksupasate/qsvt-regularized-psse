"""Tests for Phase 10 WP F consolidated implementation-completion index."""

from __future__ import annotations

import json

from robust_qsvt_se.paper.phase10_implementation_completion import (
    PHASE10_PACKAGES,
    UNRESOLVED_AFTER_PHASE10,
    run_phase10_completion_index,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in


def test_index_lists_all_packages(tmp_path):
    run = run_phase10_completion_index({"output_dir": str(tmp_path)})
    records = {r["work_package"]: r for r in run["package_records"]}
    assert set(records) == {"A", "B", "C", "D", "E"}
    assert len(PHASE10_PACKAGES) == 5


def test_claim_safety_report_present(tmp_path):
    run = run_phase10_completion_index({"output_dir": str(tmp_path)})
    report = (run["output_dir"] / "phase10_claim_safety_report.txt").read_text(encoding="utf-8")
    assert "CLAIM-SAFETY REPORT" in report
    # The report itself must not trip the forbidden scan.
    assert forbidden_in(report) == []


def test_unresolved_limitations_are_honest(tmp_path):
    run = run_phase10_completion_index({"output_dir": str(tmp_path)})
    unresolved = (run["output_dir"] / "phase10_unresolved_after_completion.md").read_text(
        encoding="utf-8"
    )
    # Must explicitly address the mandated unresolved items.
    lowered = unresolved.lower()
    assert "hardware" in lowered
    assert "field pmu/scada" in lowered
    assert "speedup" in lowered
    assert "superiority" in lowered
    assert "competitiveness" in lowered
    assert len(UNRESOLVED_AFTER_PHASE10) >= 8


def test_required_outputs_exist(tmp_path):
    run = run_phase10_completion_index({"output_dir": str(tmp_path)})
    output_dir = run["output_dir"]
    for name in (
        "phase10_index.json",
        "phase10_summary.md",
        "phase10_unresolved_after_completion.md",
        "phase10_claim_safety_report.txt",
        "phase10_all_checksums.sha256",
        "manifest.json",
    ):
        assert (output_dir / name).is_file(), name


def test_index_reflects_package_presence(tmp_path):
    run = run_phase10_completion_index({"output_dir": str(tmp_path)})
    index = json.loads((run["output_dir"] / "phase10_index.json").read_text(encoding="utf-8"))
    # When run after WP A-E, all packages should be present and claim-safe.
    if index["all_packages_present"]:
        assert index["claim_safe"] is True
        assert index["all_packages_have_manifest"] is True
