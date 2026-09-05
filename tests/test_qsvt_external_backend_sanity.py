from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.external_backend_sanity import (
    run_external_backend_sanity_regression,
    sanity_passed_backends,
)


def test_external_backend_sanity_outputs_and_passes_available_backends(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_external_backend_sanity_regression(
        {"output_dir": str(tmp_path / "external_sanity"), "grid_size": 41}
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "external_backend_sanity_summary.csv")

    assert (output_dir / "external_backend_sanity_summary.json").is_file()
    assert (output_dir / "external_backend_sanity_response_values.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert {"backend_name", "target_name", "max_error", "passed", "status"}.issubset(
        summary.columns
    )
    passed_backends = sanity_passed_backends(summary)
    assert "pennylane_poly_to_angles" in passed_backends
    if "pyqsp_sym_qsp" in set(summary["backend_name"]):
        pyqsp_rows = summary[summary["backend_name"] == "pyqsp_sym_qsp"]
        if set(pyqsp_rows["status"]) == {"passed"}:
            assert "pyqsp_sym_qsp" in passed_backends


def test_external_backend_sanity_records_qsppack_skip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_external_backend_sanity_regression(
        {"output_dir": str(tmp_path / "external_sanity"), "grid_size": 21}
    )
    summary = pd.read_csv(run["output_dir"] / "external_backend_sanity_summary.csv")
    qsppack = summary[summary["backend_name"] == "qsppack_if_available"]

    assert not qsppack.empty
    assert set(qsppack["status"]) == {"skipped_backend_unavailable"}
