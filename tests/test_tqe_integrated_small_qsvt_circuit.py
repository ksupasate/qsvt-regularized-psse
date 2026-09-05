from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import bounded_ridge_normalization_C
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import ridge_update_svd
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    RESULT_COLUMNS,
    qsvt_rescaled_update_from_transform,
    run_integrated_small_qsvt_circuit,
)


def test_sanity_polynomial_circuit_matches_expected_transform(tmp_path: Path) -> None:
    run = run_integrated_small_qsvt_circuit(
        {
            "output_root": str(tmp_path),
            "run_ieee": False,
            "sanity_polynomial_coefficients": [0.0, 0.5],
        }
    )
    row = run["results"].iloc[0]

    assert row["run_type"] == "sanity_check"
    assert row["qsvt_sequence_status"] == "sanity_passed"
    assert float(row["transform_block_error_fro"]) <= 1.0e-9
    assert int(row["num_U_calls"]) == 1


def test_phase_convention_failure_is_recorded_and_ieee_is_skipped(tmp_path: Path) -> None:
    run = run_integrated_small_qsvt_circuit(
        {
            "output_root": str(tmp_path),
            "run_ieee": True,
            "force_sanity_convention_failure": True,
        }
    )
    frame = run["results"]

    assert len(frame) == 2
    assert frame.loc[0, "qsvt_sequence_status"] == "failed_convention_mismatch"
    assert frame.loc[1, "qsvt_sequence_status"] == "skipped_with_convention_mismatch"
    assert "forced sanity" in str(frame.loc[0, "failure_or_skip_reason"])


def test_ridge_rescaling_matches_end_to_end_convention() -> None:
    alpha = 1.0e-2
    A = np.diag(np.array([0.2, 0.5], dtype=np.float64))
    b = np.array([1.0, -0.25], dtype=np.float64)
    C_alpha = bounded_ridge_normalization_C(alpha, beta=1.0)
    physical_filter = np.diag(A) / (np.diag(A) ** 2 + alpha)
    transform = np.diag(physical_filter / C_alpha)

    qsvt_update = qsvt_rescaled_update_from_transform(transform, b, C_alpha=C_alpha)
    ridge_update = ridge_update_svd(A, b, alpha=alpha)

    np.testing.assert_allclose(qsvt_update, ridge_update, atol=1.0e-12, rtol=1.0e-12)


def test_integrated_qsvt_output_schema_contains_required_columns(tmp_path: Path) -> None:
    run = run_integrated_small_qsvt_circuit(
        {
            "output_root": str(tmp_path),
            "run_ieee": False,
        }
    )
    frame = pd.read_csv(run["artifacts"]["results_csv"])

    assert set(RESULT_COLUMNS).issubset(frame.columns)
    assert run["artifacts"]["sanity_check_csv"].is_file()
    assert run["artifacts"]["statevector_probe_details_csv"].is_file()
    assert run["artifacts"]["summary_table_csv"].is_file()


def test_integrated_qsvt_circuit_resource_metadata_is_populated(tmp_path: Path) -> None:
    run = run_integrated_small_qsvt_circuit(
        {
            "output_root": str(tmp_path),
            "run_ieee": False,
        }
    )
    row = run["results"].iloc[0]

    assert int(row["num_U_calls"]) >= 1
    assert int(row["num_phase_rotations"]) >= 1
    assert int(row["num_qubits"]) == 2
    assert row["simulation_status"] == "completed"
    assert row["transpilation_status"] == "completed"
