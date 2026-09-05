from __future__ import annotations

import importlib.util

import pandas as pd

from robust_qsvt_se.qsvt.phase_sanity_regression import run_phase_sanity_regression


def test_phase_sanity_regression_outputs_and_status(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dependency_available = importlib.util.find_spec("pennylane") is not None
    run = run_phase_sanity_regression(
        {
            "output_dir": str(tmp_path / "sanity"),
            "force_dependency_missing": not dependency_available,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "phase_sanity_regression_summary.csv")

    assert (output_dir / "phase_sanity_regression_summary.json").is_file()
    assert (output_dir / "phase_sanity_response_values.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert set(summary["polynomial_name"]) == {
        "x",
        "0.5x",
        "x^3",
        "0.5x_plus_0.25x^3",
    }
    if dependency_available:
        assert set(summary["status"]) == {"passed"}
        assert bool(summary["passed"].all())
    else:
        assert set(summary["status"]) == {"skipped_dependency_missing"}


def test_phase_sanity_regression_dependency_skip_is_explicit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_phase_sanity_regression(
        {"output_dir": str(tmp_path / "sanity_skip"), "force_dependency_missing": True}
    )
    summary = pd.read_csv(run["output_dir"] / "phase_sanity_regression_summary.csv")

    assert set(summary["status"]) == {"skipped_dependency_missing"}
    assert not summary["passed"].any()
