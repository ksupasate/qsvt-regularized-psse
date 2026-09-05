from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"




def test_certificate_case_summary_has_complete_coverage_no_violations() -> None:
    summary = pd.read_csv(OUT / "certificate_case_summary.csv")
    cases = summary[summary["summary_dimension"] == "ieee_case"]
    assert set(cases["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert (cases["coverage"] == 1.0).all()
    assert (cases["violations"] == 0).all()
    assert (cases["median_tightness"] >= 1.0).all()
    assert (cases["worst_tightness"] >= cases["median_tightness"]).all()
