from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.nonbruteforce_refinement import (
    diagnose_phase_target_failure,
    run_stable_phase_validation_attempt,
)


def test_phase_target_diagnostics_required_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = diagnose_phase_target_failure(
        {
            "output_dir": str(tmp_path / "phase_target"),
            "matrix_source": "synthetic",
            "degrees": [5],
            "grid_size": 64,
            "force_dependency_missing": True,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "phase_target_failure_summary.csv")
    coefficients = pd.read_csv(output_dir / "coefficient_diagnostics.csv")

    required = {
        "alpha",
        "degree",
        "approximation_method",
        "polynomial_basis_original",
        "polynomial_basis_passed_to_phase_synthesis",
        "coefficient_order",
        "max_abs_coefficient",
        "min_abs_nonzero_coefficient",
        "coefficient_dynamic_range",
        "bounded_target_max_abs",
        "polynomial_approx_max_abs",
        "polynomial_approx_max_error",
        "phase_response_max_error",
        "phase_response_minus_polynomial_error",
        "parity_error",
        "boundedness_violation",
        "basis_conversion_error",
        "status",
        "failure_reason",
        "recommended_fix",
    }
    assert required.issubset(summary.columns)
    assert np.isfinite(coefficients["coefficient_dynamic_range"]).all()
    assert (output_dir / "basis_conversion_diagnostics.csv").is_file()
    assert (output_dir / "phase_response_error_breakdown.csv").is_file()
    assert (output_dir / "phase_target_failure_report.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_stable_phase_validation_dependency_skip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_stable_phase_validation_attempt(
        {
            "output_dir": str(tmp_path / "stable_phase"),
            "matrix_source": "synthetic",
            "grid_size": 64,
            "force_dependency_missing": True,
        }
    )
    summary = pd.read_csv(run["output_dir"] / "stable_phase_validation_summary.csv")

    assert set(summary["status"]) == {"skipped_dependency_missing"}
    assert {
        "passed",
        "failed_polynomial_approximation",
        "failed_phase_response",
        "skipped_coefficients_unstable",
        "skipped_dependency_missing",
    }.issuperset(set(summary["status"]))
