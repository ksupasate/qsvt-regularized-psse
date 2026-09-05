from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.phase1_finalization import finalize_qsvt_phase1_artifacts


def test_phase1_finalization_includes_pyqsp_pass_row(tmp_path: Path) -> None:
    phase_csv = tmp_path / "external_backend_phase_validation_summary.csv"
    _write_pyqsp_summary(phase_csv)
    table_dir = tmp_path / "paper_ready_qsvt_tables"
    table_dir.mkdir()
    pd.DataFrame([{"target": "bounded_ridge_tikhonov_pyqsp"}]).to_csv(
        table_dir / "table_6_phase_validation_status.csv",
        index=False,
    )

    run = finalize_qsvt_phase1_artifacts(
        {
            "output_dir": str(tmp_path / "phase1"),
            "phase_validation_summary_csv": str(phase_csv),
            "paper_ready_table_dir": str(table_dir),
            "regenerate_upstream": False,
        }
    )
    summary = pd.read_csv(run["output_dir"] / "phase1_finalization_summary.csv")
    text = (run["output_dir"] / "phase1_finalization_summary.md").read_text(encoding="utf-8")

    assert "bounded_ridge_tikhonov_pyqsp" in set(summary["target"])
    assert "passed_scalar_full_domain" in set(summary["status"])
    assert "Phase 1 PASS for scalar full-domain phase-response validation." in text
    assert "hardware execution" in text
    assert (run["output_dir"] / "phase1_claim_delta.csv").is_file()
    assert (run["output_dir"] / "manifest.json").is_file()


def _write_pyqsp_summary(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "backend_name": "pyqsp_sym_qsp",
                "backend_version": "0.2.0",
                "candidate_name": "coefficient_conditioned_chebyshev_degree_201_lambda_1e-04",
                "alpha": 0.01,
                "degree": 201,
                "input_basis": "chebyshev_T_low_to_high",
                "phase_count": 202,
                "phase_response_max_error_full_domain": 4.668e-4,
                "phase_response_max_error_actual_singular_values_if_available": 8.673e-5,
                "passed_1e_minus_3_full_domain": True,
                "passed_1e_minus_3_actual_singular_values": True,
                "status": "passed",
            }
        ]
    ).to_csv(path, index=False)
