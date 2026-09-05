from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.failure_fix import run_phase_validation_stable_basis


def test_stable_phase_diagnostics_files_columns_and_pass_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_phase_validation_stable_basis(
        {
            "output_dir": str(tmp_path / "phase"),
            "matrix_source": "synthetic",
            "degrees": [51],
            "grid_size": 80,
            "bound_grid_size": 257,
            "force_dependency_missing": True,
        }
    )
    output_dir = run["output_dir"]
    candidates = pd.read_csv(output_dir / "candidate_polynomial_diagnostics.csv")

    required = {
        "candidate_name",
        "alpha",
        "degree",
        "native_basis",
        "parity",
        "native_approx_max_error",
        "native_approx_mean_error",
        "native_max_abs_value",
        "bounded_in_native_basis",
        "coefficient_basis_for_backend",
        "conversion_method",
        "conversion_precision",
        "conversion_max_error",
        "max_abs_coefficient",
        "min_abs_nonzero_coefficient",
        "coefficient_dynamic_range",
        "bounded_after_conversion",
        "phase_backend",
        "phase_status",
        "phase_response_max_error",
        "phase_response_mean_error",
        "passed_1e_minus_3",
        "failure_reason",
        "recommended_interpretation",
    }
    assert required.issubset(candidates.columns)
    assert (output_dir / "phase_validation_stable_basis_summary.csv").is_file()
    assert (output_dir / "coefficient_stability_diagnostics.csv").is_file()
    assert (output_dir / "phase_response_diagnostics.csv").is_file()
    assert (output_dir / "stable_phase_validation_report.md").is_file()
    assert (output_dir / "manifest.json").is_file()

    passed = candidates[candidates["passed_1e_minus_3"] == True]  # noqa: E712
    for row in passed.itertuples():
        assert row.native_approx_max_error <= 1.0e-3
        assert bool(row.bounded_after_conversion)
        assert row.phase_response_max_error <= 1.0e-3

    unstable = candidates[candidates["coefficient_dynamic_range"].fillna(0.0) > 1.0e12]
    assert not unstable.empty
    assert not unstable["bounded_after_conversion"].all()


def test_stable_phase_dependency_skip_is_explicit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_phase_validation_stable_basis(
        {
            "output_dir": str(tmp_path / "phase_skip"),
            "matrix_source": "synthetic",
            "degrees": [5],
            "grid_size": 64,
            "bound_grid_size": 129,
            "force_dependency_missing": True,
            "coefficient_dynamic_range_limit": 1.0e99,
            "conversion_error_limit": 1.0e99,
        }
    )
    candidates = pd.read_csv(run["output_dir"] / "candidate_polynomial_diagnostics.csv")
    assert "skipped_dependency_missing" in set(candidates["phase_status"].astype(str))
