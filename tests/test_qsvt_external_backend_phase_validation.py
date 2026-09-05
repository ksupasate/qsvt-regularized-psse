from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.external_backend_phase_validation import (
    run_external_backend_phase_validation,
)


def test_external_backend_phase_validation_outputs_and_pass_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_external_backend_phase_validation(
        {
            "output_dir": str(tmp_path / "external_phase"),
            "sanity_output_dir": str(tmp_path / "external_sanity"),
            "candidate_config": {
                "degrees": [101, 201],
                "lambdas": [1.0e-4],
                "validation_grid_size": 1001,
                "fit_grid_size": 513,
            },
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "external_backend_phase_validation_summary.csv")

    assert (output_dir / "external_backend_phase_validation_summary.json").is_file()
    assert (output_dir / "external_backend_phase_angles.csv").is_file()
    assert (output_dir / "external_backend_phase_response_values.csv").is_file()
    assert (output_dir / "external_backend_phase_error_grid.csv").is_file()
    assert (output_dir / "external_backend_phase_report.md").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert {
        "phase_response_max_error_full_domain",
        "phase_response_max_error_actual_singular_values_if_available",
        "passed_1e_minus_3_full_domain",
        "passed_1e_minus_3_actual_singular_values",
    }.issubset(summary.columns)

    passed = summary[summary["passed_1e_minus_3_full_domain"] == True]  # noqa: E712
    for row in passed.itertuples():
        assert row.phase_response_max_error_full_domain <= 1.0e-3
        assert row.status == "passed"


def test_actual_singular_value_pass_is_labeled_separately(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_external_backend_phase_validation(
        {
            "output_dir": str(tmp_path / "external_phase"),
            "sanity_output_dir": str(tmp_path / "external_sanity"),
            "candidate_config": {
                "degrees": [101],
                "lambdas": [],
                "validation_grid_size": 1001,
                "fit_grid_size": 513,
            },
        }
    )
    summary = pd.read_csv(run["output_dir"] / "external_backend_phase_validation_summary.csv")
    actual_only = summary[
        (summary["passed_1e_minus_3_actual_singular_values"] == True)  # noqa: E712
        & (summary["passed_1e_minus_3_full_domain"] == False)  # noqa: E712
    ]

    for row in actual_only.itertuples():
        assert row.phase_response_max_error_full_domain > 1.0e-3
        assert row.phase_response_max_error_actual_singular_values_if_available <= 1.0e-3
