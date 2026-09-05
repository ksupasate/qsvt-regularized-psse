from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.external_phase_backend_audit import (
    EXTERNAL_BACKEND_COLUMNS,
    install_or_audit_external_phase_backends,
)


def test_external_phase_backend_audit_completes_and_writes_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = install_or_audit_external_phase_backends(
        {"output_dir": str(tmp_path / "external_backend_audit"), "install": False}
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "external_backend_audit_summary.csv")

    assert set(EXTERNAL_BACKEND_COLUMNS).issubset(summary.columns)
    assert (output_dir / "external_backend_audit_summary.json").is_file()
    assert (output_dir / "external_backend_install_log.txt").is_file()
    assert (output_dir / "external_backend_capabilities.md").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert "pyqsp_sym_qsp" in set(summary["backend_name"])
    assert "qsppack_if_available" in set(summary["backend_name"])


def test_external_phase_backend_audit_skips_missing_qsppack_safely(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = install_or_audit_external_phase_backends(
        {"output_dir": str(tmp_path / "external_backend_audit"), "install": False}
    )
    summary = run["summary"]
    qsppack = summary[summary["backend_name"] == "qsppack_if_available"].iloc[0]

    assert qsppack["status"] == "not_directly_usable_from_python"
    assert not bool(qsppack["returns_phase_angles"])


def test_available_external_backends_report_versions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = install_or_audit_external_phase_backends(
        {"output_dir": str(tmp_path / "external_backend_audit"), "install": False}
    )
    summary = run["summary"]
    available = summary[summary["install_success"] == True]  # noqa: E712

    for row in available.itertuples():
        assert str(row.version) != ""
