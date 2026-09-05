"""Utility-framing provenance and wording guards."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"


def test_utility_status_records_selection_dependence() -> None:
    status = json.loads((OUT / "utility_claim_status.json").read_text("utf-8"))
    assert status["threshold"] == 1.25
    assert status["threshold_declared_before_results"] is None
    assert status["selection_data_independent_from_evaluation"] is False
    assert status["held_out_evaluation_used"] is False
    assert status["ieee30_best_row_selected_from_sweep"] is True
    assert status["ieee57_best_row_selected_from_sweep"] is True
    assert status["application_rmse_uses_ground_truth"] is True


def test_only_approved_utility_term_is_authorized() -> None:
    status = json.loads((OUT / "utility_claim_status.json").read_text("utf-8"))
    assert status["approved_term"] == "controlled benchmark useful-overlap criterion"
    prohibited = " ".join(status["prohibited_terms"])
    for phrase in ("operationally useful", "deployment-ready", "application-optimal"):
        assert phrase in prohibited

