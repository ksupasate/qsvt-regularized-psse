from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.phase_and_multicase_summary import (
    build_phase_and_multicase_summary,
)


def test_phase_and_multicase_summary_from_minimal_inputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    convention_path = tmp_path / "convention.csv"
    sanity_path = tmp_path / "sanity.csv"
    optional_path = tmp_path / "optional.csv"
    multicase_path = tmp_path / "multicase.csv"

    pd.DataFrame(
        [
            {
                "target_type": "ridge_tikhonov_bounded_target",
                "phase_order": "original",
                "phase_sign": "phi",
                "phase_offset_rule": "none",
                "signal_operator_convention": "pennylane_rx_pcphase",
                "response_component": "real_u00",
                "max_pointwise_error": 0.004,
                "status": "failed_validation",
                "degree": 35,
            }
        ]
    ).to_csv(convention_path, index=False)
    pd.DataFrame(
        [
            {
                "polynomial_name": "x",
                "best_status": "passed",
                "best_max_pointwise_error": 1.0e-12,
            }
        ]
    ).to_csv(sanity_path, index=False)
    pd.DataFrame(
        [
            {
                "alpha": 1.0e-2,
                "status": "failed_validation",
                "max_pointwise_error": 0.004,
                "degree": 35,
                "query_count_estimate": 71,
                "convention": "original/phi/none/pennylane_rx_pcphase/real_u00",
            }
        ]
    ).to_csv(optional_path, index=False)
    pd.DataFrame(
        [
            {
                "case_name": "ieee14",
                "status": "passed",
                "achieved_max_error": 6.0e-4,
                "selected_degree": 101,
                "selected_query_count": 203,
                "failure_reason_if_any": "",
            }
        ]
    ).to_csv(multicase_path, index=False)

    run = build_phase_and_multicase_summary(
        {
            "output_dir": str(tmp_path / "summary"),
            "convention_summary_path": str(convention_path),
            "sanity_results_path": str(sanity_path),
            "optional_phase_summary_path": str(optional_path),
            "adaptive_multicase_summary_path": str(multicase_path),
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "phase_and_multicase_summary.csv")
    markdown = (output_dir / "phase_and_multicase_summary.md").read_text(encoding="utf-8")

    assert {"phase_response_convention", "optional_phase_synthesis"}.issubset(
        set(frame["summary_area"])
    )
    assert "Claims To Avoid" in markdown
    assert "do not imply quantum advantage" in markdown
    assert (output_dir / "manifest.json").is_file()
