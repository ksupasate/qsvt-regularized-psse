from __future__ import annotations

import importlib.util

import pandas as pd

from robust_qsvt_se.qsvt.phase_response_conventions import (
    diagnose_phase_response_conventions,
)


def test_phase_response_convention_sanity_polynomials(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dependency_available = importlib.util.find_spec("pennylane") is not None
    run = diagnose_phase_response_conventions(
        {
            "output_dir": str(tmp_path / "phase"),
            "matrix_source": "synthetic",
            "alpha": 1.0e-2,
            "ridge_degree": 5,
            "ridge_grid_size": 64,
            "phase_order": ["original"],
            "phase_sign": ["phi"],
            "phase_offset_rule": ["none"],
            "signal_operator_convention": ["pennylane_rx_pcphase"],
            "response_component": ["real_u00"],
            "write_all_response_values": "false",
            "force_dependency_missing": not dependency_available,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "convention_search_summary.csv")
    sanity = pd.read_csv(output_dir / "sanity_polynomial_results.csv")

    assert (output_dir / "manifest.json").is_file()
    assert {
        "polynomial_name",
        "coefficient_basis_input",
        "coefficient_basis_expected",
        "max_pointwise_error",
        "status",
    }.issubset(summary.columns)

    if dependency_available:
        assert len(sanity) == 4
        assert set(sanity["best_status"]) == {"passed"}
        assert float(sanity["best_max_pointwise_error"].max()) <= 1.0e-6
        canonical = summary[
            (summary["phase_order"] == "original")
            & (summary["phase_sign"] == "phi")
            & (summary["phase_offset_rule"] == "none")
            & (summary["signal_operator_convention"] == "pennylane_rx_pcphase")
            & (summary["response_component"] == "real_u00")
            & (summary["target_type"] == "sanity_polynomial")
        ]
        assert len(canonical) == 4
        assert set(canonical["status"]) == {"passed"}
    else:
        assert set(summary["status"]) == {"skipped_dependency_missing"}
