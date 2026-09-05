from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.stable_phase_candidates import build_stable_phase_candidates
from robust_qsvt_se.qsvt.stable_phase_validation import run_stable_target_phase_validation


def test_stable_target_validation_outputs_and_skips_unsafe_candidates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    candidate_dir = tmp_path / "candidates"
    build_stable_phase_candidates(
        {
            "output_dir": str(candidate_dir),
            "matrix_source": "synthetic",
            "degrees": [5, 35],
            "lambdas": [1.0e-8],
            "approximation_grid_size": 64,
            "boundedness_grid_size": 129,
            "conversion_grid_size": 129,
            "include_decimal_conversion": False,
        }
    )
    run = run_stable_target_phase_validation(
        {
            "output_dir": str(tmp_path / "validation"),
            "candidate_output_dir": str(candidate_dir),
            "sanity_output_dir": str(tmp_path / "sanity"),
            "rebuild_candidates": False,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "stable_target_phase_validation_summary.csv")
    phases = pd.read_csv(output_dir / "phase_angles.csv")

    assert (output_dir / "stable_target_phase_validation_summary.json").is_file()
    assert (output_dir / "phase_response_values.csv").is_file()
    assert (output_dir / "phase_response_error_grid.csv").is_file()
    assert (output_dir / "stable_target_phase_validation_report.md").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert set(summary["status"]) == {"skipped_candidate_safety_gate"}
    assert phases.empty


def test_stable_target_validation_pass_rows_must_meet_all_error_gates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    candidate_dir = tmp_path / "candidates"
    build_stable_phase_candidates(
        {
            "output_dir": str(candidate_dir),
            "matrix_source": "synthetic",
            "degrees": [5],
            "lambdas": [1.0e-8],
            "approximation_grid_size": 64,
            "boundedness_grid_size": 129,
            "conversion_grid_size": 129,
            "include_decimal_conversion": False,
        }
    )
    run = run_stable_target_phase_validation(
        {
            "output_dir": str(tmp_path / "validation"),
            "candidate_output_dir": str(candidate_dir),
            "sanity_output_dir": str(tmp_path / "sanity"),
            "rebuild_candidates": False,
        }
    )
    summary = pd.read_csv(run["output_dir"] / "stable_target_phase_validation_summary.csv")
    passed = summary[summary["passed_1e_minus_3"] == True]  # noqa: E712
    candidates = pd.read_csv(candidate_dir / "stable_phase_candidate_summary.csv")

    for row in passed.itertuples():
        candidate = candidates[candidates["candidate_name"] == row.candidate_name].iloc[0]
        assert candidate["native_max_error"] <= 1.0e-3
        assert bool(candidate["bounded_in_native_basis"])
        assert pd.notna(candidate["coefficient_dynamic_range"])
        assert row.phase_response_max_error <= 1.0e-3
