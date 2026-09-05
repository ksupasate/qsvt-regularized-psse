from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.engineering_extension_report import build_engineering_extension_summary
from robust_qsvt_se.qsvt.failure_fix import build_failure_fix_summary


def test_failure_fix_summary_boundaries(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    outputs = tmp_path / "outputs"
    phase = outputs / "qsvt_phase_validation_stable_basis"
    pre = outputs / "qsvt_preconditioned_ieee300_estimator"
    residual = outputs / "qsvt_ieee300_residual_weighted_error"
    for path in [phase, pre, residual]:
        path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "passed_1e_minus_3": False,
                "phase_status": "failed_coefficient_dynamic_range",
            }
        ]
    ).to_csv(phase / "candidate_polynomial_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "useful_preconditioned_variant",
                "variant_name": "preconditioned_ridge_column_equilibrated_coordinate_penalty",
            }
        ]
    ).to_csv(pre / "preconditioned_ieee300_estimator_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "ok",
                "interpretation": "Residual-weighted diagnostics do not replace full validation.",
            }
        ]
    ).to_csv(residual / "residual_weighted_error_summary.csv", index=False)

    monkeypatch.chdir(tmp_path)
    run = build_failure_fix_summary({"output_dir": str(outputs / "qsvt_failure_fix_summary")})
    report = (run["output_dir"] / "failure_fix_summary.md").read_text()

    assert "PARTIAL PASS" in report
    assert "not full validation" in report
    assert "Do not claim quantum speedup" in report


def test_claim_matrix_and_docs_include_failure_fix_boundaries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_engineering_extension_summary({"output_dir": str(tmp_path / "engineering")})
    matrix = pd.read_csv(run["output_dir"] / "claim_support_matrix.csv")
    claims = " ".join(matrix["claim"].astype(str))

    assert "Stable phase-synthesis diagnostics were implemented" in claims
    assert "formal preconditioned IEEE300 estimator variant" in claims
    assert "Residual-weighted spectral diagnostics were implemented" in claims

    docs = [
        Path("docs/QSVT_STABLE_PHASE_SYNTHESIS.md"),
        Path("docs/QSVT_PRECONDITIONED_IEEE300_VARIANT.md"),
        Path("docs/QSVT_RESIDUAL_WEIGHTED_SPECTRAL_ERROR.md"),
    ]
    for doc in docs:
        if doc.exists():
            text = doc.read_text()
            assert "Avoid wording" in text or "Claim Boundary" in text
