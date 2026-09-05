from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"




def test_eligible_results_have_provenance_and_superseded_summary_is_excluded() -> None:
    results = pd.read_csv(OUT / "canonical_result_registry.csv", low_memory=False)
    eligible = results[results["manuscript_eligible"]]
    assert eligible["source_row_locator"].notna().all()
    assert eligible["limitation_code"].notna().all()
    assert eligible["matrix_fingerprint"].notna().all()
    superseded = results[results["result_id"] == "res:excluded:structural_markdown_summary"]
    assert len(superseded) == 1
    assert not bool(superseded.iloc[0]["manuscript_eligible"])
    assert superseded.iloc[0]["evidence_tier"] == "excluded"
