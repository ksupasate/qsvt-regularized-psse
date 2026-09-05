from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"

REQUIRED = {
    "small_scale_8x8",
    "simulator_only",
    "selected_output_only",
    "controlled_generated_residuals",
    "no_field_data",
    "no_hardware",
    "no_quantum_speedup",
    "no_ieee_scale_oracle",
    "direct_rotation_architecture",
    "finite_shot_structural_campaign_skipped",
    "case_dependence",
    "functional_dependence",
    "aggregate_weakness",
    "ieee57_mixed_result",
    "certificate_looseness",
    "local_sensitivity_model",
    "local_refinement_only",
    "rank_deficient_blocks",
    "fixed_three_topologies",
    "no_iterative_nonlinear_integration",
}


def test_required_limitation_registry_is_complete_and_resolved() -> None:
    limitations = pd.read_csv(OUT / "canonical_limitation_registry.csv")
    results = pd.read_csv(OUT / "canonical_result_registry.csv")
    assert REQUIRED.issubset(set(limitations["limitation_id"]))
    assert limitations["limitation_id"].is_unique
    valid_ids = set(results["result_id"])
    for affected in limitations["affected_result_ids"].dropna():
        assert set(affected.split(";")).issubset(valid_ids)
    assert (limitations["status"] == "active").all()
