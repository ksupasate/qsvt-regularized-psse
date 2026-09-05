from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.engineering_extension_report import build_engineering_extension_summary
from robust_qsvt_se.qsvt.nonbruteforce_refinement import (
    build_nonbruteforce_refinement_summary,
)


def test_nonbruteforce_summary_states_boundaries(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    outputs = tmp_path / "outputs"
    phase_dir = outputs / "qsvt_phase_target_failure_diagnostics"
    stable_dir = outputs / "qsvt_stable_phase_validation_attempt"
    spectral_dir = outputs / "qsvt_ieee300_spectral_difficulty"
    spectrum_dir = outputs / "qsvt_spectrum_aware_diagnostics"
    ieee118_dir = outputs / "qsvt_ieee118_targeted_refinement"
    for directory in [phase_dir, stable_dir, spectral_dir, spectrum_dir, ieee118_dir]:
        directory.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "degree": 35,
                "status": "diagnosed_failure",
                "failure_class": "degree_too_low",
                "phase_response_max_error": 0.004,
                "polynomial_approx_max_error": 0.004,
                "recommended_fix": "Use passing polynomial first.",
            }
        ]
    ).to_csv(phase_dir / "phase_target_failure_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "target_type": "ridge_tikhonov_bounded_target",
                "status": "failed_polynomial_approximation",
                "degree": 35,
            }
        ]
    ).to_csv(stable_dir / "stable_phase_validation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "case_name": "ieee300",
                "full_interval_max_error": 0.08,
                "actual_singular_values_max_error": 0.02,
                "error_peak_region": "near_sigma_min",
                "diagnostic_interpretation": "Full and actual errors separated.",
            }
        ]
    ).to_csv(spectral_dir / "spectral_difficulty_summary.csv", index=False)
    pd.DataFrame([{"case_name": "ieee300", "diagnostic_type": "central_95"}]).to_csv(
        spectrum_dir / "spectrum_aware_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "degree": 1201,
                "passed_1e_minus_3": False,
                "max_pointwise_error": 0.0011,
            }
        ]
    ).to_csv(ieee118_dir / "ieee118_refinement_summary.csv", index=False)

    monkeypatch.chdir(tmp_path)
    run = build_nonbruteforce_refinement_summary(
        {"output_dir": str(outputs / "qsvt_nonbruteforce_refinement_summary")}
    )
    report = (run["output_dir"] / "nonbruteforce_refinement_summary.md").read_text()

    assert "No brute-force degree escalation was used" in report
    assert "strict 1e-3 tolerance was not relaxed" in report
    assert "Restricted-interval diagnostics are diagnostic only" in report


def test_claim_support_matrix_includes_nonbruteforce_claims(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_engineering_extension_summary({"output_dir": str(tmp_path / "engineering")})
    matrix = pd.read_csv(run["output_dir"] / "claim_support_matrix.csv")
    claims = " ".join(matrix["claim"].astype(str))

    assert "No brute-force degree escalation was used" in claims
    assert "IEEE300 spectral difficulty was analyzed" in claims
    assert "No tolerance relaxation was used" in claims
