from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.phase_backend_audit import AUDIT_COLUMNS, audit_phase_backend_options


def test_phase_backend_audit_outputs_required_files_and_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = audit_phase_backend_options({"output_dir": str(tmp_path / "backend_audit")})
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "phase_backend_audit_summary.csv")

    assert set(AUDIT_COLUMNS).issubset(summary.columns)
    assert (output_dir / "phase_backend_audit_summary.json").is_file()
    assert (output_dir / "phase_backend_capabilities.md").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert "pennylane.poly_to_angles" in set(summary["backend_name"])
    assert "pyqsp" in set(summary["backend_name"])


def test_missing_optional_backends_are_reported_or_skipped_gracefully(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = audit_phase_backend_options({"output_dir": str(tmp_path / "backend_audit")})
    summary = run["summary"]
    optional = summary[summary["backend_name"].isin(["pyqsp", "QSPPACK"])]

    assert not optional.empty
    assert set(optional["status"]).issubset(
        {"unavailable_optional_dependency", "available_not_integrated"}
    )
