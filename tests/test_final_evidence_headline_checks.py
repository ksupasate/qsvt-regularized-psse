import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def _checks() -> pd.DataFrame:
    return pd.read_csv(OUT / "headline_result_checks.csv").set_index("check_id")


def test_every_headline_check_passes() -> None:
    summary = json.loads((OUT / "headline_result_check_summary.json").read_text())
    checks = _checks()
    assert summary["status"] == "pass"
    assert summary["failed"] == 0
    assert len(checks) == summary["checks"]
    assert (checks["status"] == "pass").all()


def test_primary_counts_and_qsvt_subset_recompute() -> None:
    checks = _checks()
    expected = {
        "generalization.primary.win": 13,
        "generalization.primary.tie": 0,
        "generalization.primary.loss": 2,
        "structural.primary.win": 6,
        "structural.primary.tie": 5,
        "structural.primary.loss": 1,
        "structural.qsvt.rows": 72,
        "structural.qsvt.failures": 0,
    }
    for check_id, value in expected.items():
        assert float(checks.loc[check_id, "recomputed_value"]) == value


def test_finite_shot_and_ieee57_aggregate_weakness_remain_visible() -> None:
    checks = _checks()
    assert checks.loc["integrated.finite_shot_1e6", "status"] == "pass"
    assert (
        checks.loc["structural.finite_shot_status", "recomputed_value"]
        == "skipped_under_frozen_cost_ceiling"
    )
    results = pd.read_csv(OUT / "canonical_result_registry.csv")
    row = results[results["result_id"] == "res:structural:ieee57:aggregate:win_tie_loss"].iloc[0]
    assert row["value"] == "0/2/2"
