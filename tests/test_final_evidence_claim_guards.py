import json
from pathlib import Path

from robust_qsvt_se.evidence.canonical_registry import run_claim_guards

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def test_generated_claim_guard_report_passes() -> None:
    report = json.loads((OUT / "excluded_claim_guard_report.json").read_text())
    assert report["status"] == "pass"
    assert report["violation_count"] == 0
    assert report["negative_statements_allowed"] > 0


def test_positive_prohibited_claim_fails_and_negative_statement_passes(tmp_path: Path) -> None:
    positive = tmp_path / "positive"
    positive.mkdir()
    (positive / "summary.md").write_text("This work demonstrates quantum advantage.")
    assert run_claim_guards(ROOT, positive)["status"] == "blocking_failure"
    negative = tmp_path / "negative"
    negative.mkdir()
    (negative / "summary.md").write_text("No quantum advantage is claimed.")
    assert run_claim_guards(ROOT, negative)["status"] == "pass"
