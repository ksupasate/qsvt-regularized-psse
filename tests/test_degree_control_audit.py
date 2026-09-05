from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.degree_control_audit import (
    audit_single_degree_request,
    write_degree_control_outputs,
)


def test_degree_control_audit_reports_requested_constructed_and_effective(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_synthesis(coefficients, **kwargs):
        return {
            "phases": np.zeros(6),
            "phase_count": 6,
            "cache_key": "fake-cache",
            "cache_hit": True,
            "metadata": {},
        }

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.degree_control_audit.synthesize_phases_with_cache",
        fake_synthesis,
    )

    row = audit_single_degree_request(
        alpha=1.0e-4,
        alpha_norm=1.0e-4,
        requested_degree=5,
        domain_min=0.2,
        cache_dir=tmp_path,
    )

    assert row["requested_degree"] == 5
    assert row["constructed_polynomial_degree"] == 5
    assert row["synthesized_phase_degree"] == 5
    assert row["effective_qsvt_degree"] == 5
    assert row["cache_hit"] is True


def test_degree_control_audit_detects_fallback_metadata(monkeypatch, tmp_path: Path) -> None:
    def fake_lower_degree_synthesis(coefficients, **kwargs):
        return {
            "phases": np.zeros(4),
            "phase_count": 4,
            "cache_key": "fallback-cache",
            "cache_hit": False,
            "metadata": {},
        }

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.degree_control_audit.synthesize_phases_with_cache",
        fake_lower_degree_synthesis,
    )

    row = audit_single_degree_request(
        alpha=1.0e-4,
        alpha_norm=1.0e-4,
        requested_degree=5,
        domain_min=0.2,
        cache_dir=tmp_path,
    )

    assert row["constructed_polynomial_degree"] == 5
    assert row["synthesized_phase_degree"] == 3
    assert row["fallback_used"] is True


def test_degree_control_outputs_include_required_audit_files(tmp_path: Path) -> None:
    rows = [
        {
            "alpha": 1.0e-4,
            "requested_degree": 5,
            "constructed_polynomial_degree": 5,
            "synthesized_phase_degree": 5,
            "effective_qsvt_degree": 5,
            "phase_count": 6,
            "cache_key": "abc",
            "cache_hit": False,
            "backend_name": "fake",
            "backend_status": "synthesized",
            "tolerance": 1.0e-5,
            "parity": "odd",
            "fallback_used": False,
            "failure_reason_if_any": "",
        }
    ]

    artifacts = write_degree_control_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)
    frame = pd.read_csv(artifacts["degree_control_audit"])

    assert frame.loc[0, "requested_degree"] == 5
    assert artifacts["phase_cache_audit"].is_file()
    assert artifacts["requested_vs_constructed_degree"].is_file()
    assert artifacts["phase_synthesis_backend_summary"].is_file()
