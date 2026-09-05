from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.norm_residual_gap_audit import (
    best_scalar_for_residual,
    run_norm_residual_gap_audit,
)


def test_best_scalar_for_residual_matches_closed_form() -> None:
    H = np.eye(2)
    qsvt = np.array([1.0, 2.0])
    r = np.array([2.0, 1.0])

    expected = float(np.dot(H @ qsvt, r) / np.dot(H @ qsvt, H @ qsvt))

    assert best_scalar_for_residual(H, qsvt, r) == pytest.approx(expected)


def test_norm_residual_gap_audit_outputs_required_columns(tmp_path: Path) -> None:
    run = run_norm_residual_gap_audit(
        {
            "output_dir": str(tmp_path),
            "case": "ieee14",
            "model": "ac_linearized",
            "submatrix_size": 4,
            "alpha": 1.0e-4,
            "degree": 9,
            "seed": 123,
        }
    )
    summary = pd.read_csv(run["artifacts"]["norm_gap_summary"])

    for column in [
        "condition_number",
        "residual_qsvt_raw",
        "residual_qsvt_best_scalar",
        "dominant_gap_source",
    ]:
        assert column in summary.columns
    assert run["artifacts"]["residual_gap_decomposition"].is_file()
    assert run["artifacts"]["norm_recovery_interpretation"].is_file()
