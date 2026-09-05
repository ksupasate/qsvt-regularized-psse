from __future__ import annotations

import warnings

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.nonbruteforce_refinement import run_ieee118_targeted_refinement


def test_ieee118_targeted_refinement_approved_degrees_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        run = run_ieee118_targeted_refinement(
            {
                "output_dir": str(tmp_path / "ieee118"),
                "case_name": "synthetic",
                "matrix_source": "synthetic",
                "degrees": [1201],
                "grid_size": 80,
            }
        )
    summary = pd.read_csv(run["output_dir"] / "ieee118_refinement_summary.csv")

    assert not captured
    assert set(summary["degree"]).issubset({1201, 1501, 2001})
    assert int(summary["degree"].max()) <= 1201
    assert {"diagnostic_status", "numerical_stability_status"}.issubset(summary.columns)
    unstable = summary[summary["numerical_stability_status"] != "ok"]
    assert set(unstable["diagnostic_status"]).issubset({"diagnostic_only"})
    assert not unstable["passed_1e_minus_3"].astype(bool).any()
    assert (run["output_dir"] / "ieee118_refinement_trace.csv").is_file()


def test_ieee118_targeted_refinement_rejects_unapproved_degree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="unapproved"):
        run_ieee118_targeted_refinement(
            {
                "output_dir": str(tmp_path / "bad_ieee118"),
                "case_name": "synthetic",
                "matrix_source": "synthetic",
                "degrees": [1003],
            }
        )
