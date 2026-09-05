from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.baseline_coverage_extension import (
    ALL_COLUMNS,
    build_baseline_coverage_extension,
)


def test_schema_and_hhl_not_full_hhl(tmp_path: Path) -> None:
    run = build_baseline_coverage_extension(
        {
            "cases": ["ieee14"],
            "stress_types": ["clean_reference", "bad_data_heavy"],
            "estimators": ["lav", "hhl_style_proxy", "ridge_tikhonov", "huber_irls"],
            "seeds": [0],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "baseline"),
        }
    )
    frame = pd.read_csv(run["artifacts"]["baseline_coverage_results"])
    assert list(frame.columns) == ALL_COLUMNS
    # The HHL-style proxy is never described as full HHL.
    hhl = frame[frame["estimator"] == "hhl_style_proxy"]
    assert not hhl.empty
    assert hhl["claim_boundary"].astype(str).str.contains("not full HHL").all()
    interp = Path(run["artifacts"]["baseline_coverage_interpretation"]).read_text(encoding="utf-8")
    assert "not claimed as full HHL" in interp


def test_lav_coverage_grounded_in_run_cases(tmp_path: Path) -> None:
    # LAV coverage must reflect only the cases actually run, never claimed all-case.
    run = build_baseline_coverage_extension(
        {
            "cases": ["ieee14"],
            "stress_types": ["clean_reference"],
            "estimators": ["lav", "ridge_tikhonov"],
            "seeds": [0],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "baseline"),
        }
    )
    lav = run["coverage"]["lav"]
    assert lav["cases"] == ["ieee14"]
    assert "ieee118" not in lav["cases"]
    assert "ieee57" not in lav["cases"]


def test_robust_beats_ridge_under_bad_data(tmp_path: Path) -> None:
    run = build_baseline_coverage_extension(
        {
            "cases": ["ieee14", "ieee57"],
            "stress_types": ["bad_data_heavy"],
            "estimators": ["lav", "ridge_tikhonov", "huber_irls"],
            "seeds": [0, 1],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "baseline"),
        }
    )
    assert run["robust_beats_ridge_bad_data"] is True
    summary = pd.read_csv(run["artifacts"]["baseline_coverage_summary"])
    assert not summary.empty
    assert set(summary["estimator"]) >= {"lav", "ridge_tikhonov", "huber_irls"}
