from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.stable_phase_candidates import (
    SUMMARY_COLUMNS,
    build_stable_phase_candidates,
)


def test_stable_phase_candidates_outputs_required_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_stable_phase_candidates(
        {
            "output_dir": str(tmp_path / "candidates"),
            "matrix_source": "synthetic",
            "degrees": [5, 35],
            "lambdas": [1.0e-8],
            "approximation_grid_size": 64,
            "boundedness_grid_size": 129,
            "conversion_grid_size": 129,
            "include_decimal_conversion": False,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "stable_phase_candidate_summary.csv")

    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert (output_dir / "stable_phase_candidate_summary.json").is_file()
    assert (output_dir / "candidate_coefficients_chebyshev.csv").is_file()
    assert (output_dir / "candidate_coefficients_monomial.csv").is_file()
    assert (output_dir / "candidate_error_grid.csv").is_file()
    assert (output_dir / "candidate_boundedness_grid.csv").is_file()
    assert (output_dir / "candidate_report.md").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_candidate_pass_gate_invariants(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_stable_phase_candidates(
        {
            "output_dir": str(tmp_path / "candidates"),
            "matrix_source": "synthetic",
            "degrees": [5],
            "lambdas": [1.0e-8],
            "approximation_grid_size": 64,
            "boundedness_grid_size": 129,
            "conversion_grid_size": 129,
            "include_decimal_conversion": False,
        }
    )
    summary = pd.read_csv(run["output_dir"] / "stable_phase_candidate_summary.csv")
    passed = summary[summary["safe_for_phase_synthesis"] == True]  # noqa: E712

    for row in passed.itertuples():
        assert row.native_max_error <= 1.0e-3
        assert bool(row.bounded_in_native_basis)
        assert row.parity_error <= 1.0e-10
        assert row.conversion_max_error <= 1.0e-5
        assert row.coefficient_dynamic_range <= 1.0e12
        assert bool(row.bounded_after_conversion)
        assert row.post_conversion_max_abs_value <= 1.0 + 1.0e-5
