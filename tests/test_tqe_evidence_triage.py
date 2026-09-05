from __future__ import annotations

import pandas as pd

from robust_qsvt_se.paper.tqe_evidence_triage import (
    EVIDENCE_COLUMNS,
    REVIEWER_COLUMNS,
    build_evidence_triage,
)


def test_evidence_map_schema(tmp_path):
    run = build_evidence_triage({"output_root": tmp_path})

    assert list(run["main"].columns) == EVIDENCE_COLUMNS
    assert list(run["supplement"].columns) == EVIDENCE_COLUMNS
    assert run["artifacts"]["main_csv"].exists()
    assert run["artifacts"]["supplement_csv"].exists()


def test_main_vs_supplement_assignment_logic(tmp_path):
    run = build_evidence_triage({"output_root": tmp_path})
    evidence = run["evidence"]

    assert "E5" in set(run["main"]["evidence_id"])
    assert "S1" in set(run["supplement"]["evidence_id"])
    assert "N1" not in set(run["main"]["evidence_id"])
    location = evidence.loc[evidence["evidence_id"] == "N1", "recommended_location"].iloc[0]
    assert location == "limitation_only"


def test_reviewer_checklist_schema(tmp_path):
    run = build_evidence_triage({"output_root": tmp_path})

    checklist = run["reviewer_checklist"]
    assert list(checklist.columns) == REVIEWER_COLUMNS
    assert len(checklist) == 20
    assert run["artifacts"]["reviewer_checklist_csv"].exists()
    loaded = pd.read_csv(run["artifacts"]["reviewer_checklist_csv"])
    assert list(loaded.columns) == REVIEWER_COLUMNS
