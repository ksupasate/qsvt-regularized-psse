"""Schema, JSON, evidence-tier, and traceability guards for readout records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"


def test_readout_schema_is_complete_and_ids_are_unique() -> None:
    frame = pd.read_csv(OUT / "readout_registry.csv", keep_default_na=False)
    required = {
        "configuration_id",
        "polynomial_degree",
        "attempted_shots",
        "postselection_accepted_shots",
        "readout_accepted_shots",
        "postselection_probability",
        "branch_probability",
        "quadrature_probability",
        "selected_output_estimate",
        "standard_error",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "relative_ci_half_width",
        "analytic_variance",
        "empirical_seed_variance",
        "estimator_definition",
        "artifact_paths",
    }
    assert required <= set(frame.columns)
    assert frame["readout_id"].is_unique




def test_configuration_ids_exist_and_tiers_are_not_conflated() -> None:
    frame = pd.read_csv(OUT / "readout_registry.csv", keep_default_na=False)
    evidence = pd.read_csv(OUT / "evidence_registry.csv", keep_default_na=False)
    assert set(frame["configuration_id"]) <= set(evidence["configuration_id"])
    d31 = frame[frame["readout_id"] == "d31_integrated_30seed_distribution"].iloc[0]
    assert d31["evidence_status"] == "sampled_distribution"
    assert "not backend shots at scale" in d31["variant"]
    isolated = frame[frame["readout_id"].str.startswith("d255_isolated_wpj")]
    assert isolated["variant"].str.contains("does not act").all()


def test_json_has_same_records_as_csv() -> None:
    frame = pd.read_csv(OUT / "readout_registry.csv", keep_default_na=False)
    data = json.loads((OUT / "readout_registry.json").read_text("utf-8"))
    assert [row["readout_id"] for row in data] == frame["readout_id"].tolist()

