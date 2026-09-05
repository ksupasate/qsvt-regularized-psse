from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import build_full_engineering_report
from robust_qsvt_se.qsvt.hardware_resource_estimator import estimate_hardware_resources


def test_full_engineering_report_fails_clearly_when_inputs_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing required pathway inputs"):
        build_full_engineering_report(
            {
                "output_dir": str(tmp_path / "report"),
                "input_root": str(tmp_path),
                "require_inputs": True,
            }
        )


def test_hardware_resource_estimator_reports_expected_counts() -> None:
    estimate = estimate_hardware_resources(
        np.eye(4),
        qsvt_degree=5,
        phase_count=6,
        readout_shots=100,
        block_encoding_model="sparse_access_oracle",
    )

    assert estimate.logical_index_qubits == 2
    assert estimate.phase_count == 6
    assert estimate.query_count == 11
    assert estimate.readout_shots == 100
