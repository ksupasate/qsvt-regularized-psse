from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def test_near_zero_audit_retains_every_primary_task() -> None:
    audit = pd.read_csv(OUT / "near_zero_output_audit.csv")
    assert len(audit) == 2880
    assert audit["task_id"].nunique() == 1440
    assert audit["included_in_original_primary_evidence"].all()
    assert audit["near_zero_threshold"].nunique() == 1
    assert audit["near_zero_flag"].equals(
        audit["reference_output"].abs() < audit["near_zero_threshold"]
    )


def test_near_zero_and_non_near_zero_summaries_are_separate() -> None:
    summary = pd.read_csv(OUT / "near_zero_output_summary.csv")
    paired = summary[summary["summary_type"] == "paired_outcome"].set_index("scope")
    assert int(paired.loc["near_zero_outputs", "unique_task_count"]) == 2
    assert int(paired.loc["non_near_zero_outputs", "unique_task_count"]) == 1438
    assert int(paired["original_rows_removed"].sum()) == 0
