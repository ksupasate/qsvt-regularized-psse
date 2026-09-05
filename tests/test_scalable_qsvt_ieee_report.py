from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.scalable_qsvt_report import build_scalable_qsvt_report


def test_scalable_qsvt_report_generates_claim_support_matrix(tmp_path: Path) -> None:
    run = build_scalable_qsvt_report(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "matrix_free_input_dirs": [],
        }
    )

    claim = pd.read_csv(run["artifacts"]["claim_support_matrix"])
    assert "claim" in claim.columns
    assert "category" in claim.columns
    assert (claim["category"] == "unsupported").any()
    assert run["artifacts"]["scalable_qsvt_ieee_report"].exists()
