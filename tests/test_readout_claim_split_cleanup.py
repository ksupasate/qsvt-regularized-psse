from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.paper.claim_lint import scan_file
from robust_qsvt_se.paper.full_vector_readout import (
    build_readout_claim_family_rows,
    readout_claim_split_note_markdown,
)


def test_c10_is_parent_not_independent() -> None:
    rows = build_readout_claim_family_rows(full_vector_supported=True)
    by_id = {row["claim_id"]: row for row in rows}
    assert by_id["C10"]["counts_as_independent_claim"] == "no"
    assert by_id["C10"]["claim_role"] == "parent_or_legacy_family"
    assert by_id["C10"]["superseded_by"] == "C10a;C10b"


def test_children_are_independent_with_correct_status() -> None:
    supported = {
        r["claim_id"]: r for r in build_readout_claim_family_rows(full_vector_supported=True)
    }
    absent = {
        r["claim_id"]: r for r in build_readout_claim_family_rows(full_vector_supported=False)
    }
    assert supported["C10a"]["counts_as_independent_claim"] == "yes"
    assert supported["C10b"]["counts_as_independent_claim"] == "yes"
    assert supported["C10a"]["status"] == "supported_with_limitations"
    assert absent["C10a"]["status"] == "assumption_only"
    # The full-scale child is never promoted.
    assert supported["C10b"]["status"] == "assumption_only"
    assert absent["C10b"]["status"] == "assumption_only"


def test_independent_claim_count_excludes_parent() -> None:
    rows = build_readout_claim_family_rows(full_vector_supported=True)
    independent = [r for r in rows if r["counts_as_independent_claim"] == "yes"]
    assert {r["claim_id"] for r in independent} == {"C10a", "C10b"}


def test_split_note_clean_and_lint_safe(tmp_path: Path) -> None:
    note = tmp_path / "readout_claim_split_note.md"
    note.write_text(readout_claim_split_note_markdown(full_vector_supported=True), encoding="utf-8")
    assert not [r for r in scan_file(note) if r["risk_level"] == "high"]


def test_lint_still_flags_full_scale_overclaim(tmp_path: Path) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text("We achieve full-vector readout solved at IEEE scale.", encoding="utf-8")
    high = [r for r in scan_file(draft) if r["risk_level"] == "high"]
    assert any(r["matched_phrase"] == "full-vector readout solved" for r in high)
