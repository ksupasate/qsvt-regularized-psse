from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.phase2_qsvt_boundary import (
    FINAL_STATUSES,
    REQUIRED_COLUMNS,
    condition_controlled_variant,
)


def test_condition_controlled_variant_preserves_vectors_and_condition() -> None:
    matrix = np.array([[3.0, 1.0, 0.2], [0.5, 2.0, -0.3], [0.1, 0.4, 1.0]], dtype=float)
    controlled = condition_controlled_variant(matrix, 1.0e3)
    source_u, source_s, source_vt = np.linalg.svd(matrix, full_matrices=False)
    target_u, target_s, target_vt = np.linalg.svd(controlled, full_matrices=False)
    assert np.isclose(target_s[0], source_s[0])
    assert np.isclose(target_s[0] / target_s[-1], 1.0e3, rtol=1.0e-10)
    assert np.allclose(np.abs(target_u.T @ source_u), np.eye(3), atol=1.0e-10)
    assert np.allclose(np.abs(target_vt @ source_vt.T), np.eye(3), atol=1.0e-10)


def test_phase2_output_contract_and_status_vocabulary() -> None:
    assert len(REQUIRED_COLUMNS) == len(set(REQUIRED_COLUMNS))
    assert "failure_reason" in REQUIRED_COLUMNS
    assert "update_relative_error_vs_matched_ridge" in REQUIRED_COLUMNS
    assert {
        "feasible",
        "degree-limited",
        "tolerance-missing",
        "phase-synthesis-failed",
        "statevector-failed",
        "readout-failed",
        "skipped-with-reason",
    } == FINAL_STATUSES


def test_generated_phase2_boundary_and_required_success_metadata() -> None:
    artifact = Path("outputs/phase2_qsvt_boundary/all_attempts.csv")
    assert artifact.is_file()
    frame = pd.read_csv(artifact, keep_default_na=False)
    assert len(frame) == 64
    assert (frame["final_status"] == "feasible").sum() == 32
    assert (frame["final_status"] != "feasible").sum() == 32

    hard_regime = frame.loc[
        (frame["kappa"].astype(float) >= 100.0) & (frame["lambda"].astype(float) <= 1.0e-2)
    ]
    assert len(hard_regime) == 24
    assert not (hard_regime["final_status"] == "feasible").any()

    raw = frame.loc[
        (frame["block_id"] == "ieee30_16x16_raw")
        & np.isclose(frame["lambda"].astype(float), 0.069)
        & (frame["degree_ceiling"].astype(int) == 45)
    ].iloc[0]
    required = [
        "case_name",
        "block_size",
        "selection_rule",
        "selected_rows",
        "selected_cols",
        "matrix_source",
        "kappa",
        "sigma_min",
        "sigma_max",
        "beta",
        "lambda",
        "alpha",
        "degree_attempted",
        "phase_count",
        "max_polynomial_error",
        "phase_synthesis_status",
        "boundedness_pass",
        "parity_pass",
        "postselection_probability",
        "update_relative_error_vs_matched_ridge",
        "selected_functional_errors",
        "finite_shot_readout_status",
        "classical_adjoint_value",
        "classical_adjoint_time",
        "final_status",
    ]
    assert all(str(raw[field]).strip() for field in required)
    assert raw["case_name"] == "ieee30"
    assert int(raw["block_size"]) == 16
    assert int(raw["degree_attempted"]) == 25
    assert int(raw["phase_count"]) == 26
    assert raw["phase_synthesis_status"] == "completed"
    assert raw["final_status"] == "feasible"
