from __future__ import annotations

import json

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.partial_readout_demo import run_partial_observable_readout_demo


def test_partial_readout_demo_writes_required_outputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_partial_observable_readout_demo(
        {
            "output_dir": str(tmp_path / "partial_readout"),
            "case": "ieee14",
            "case_name": "ieee14",
            "matrix_source": "weighted_jacobian",
            "submatrix_size": 2,
            "alpha": 1.0e-4,
            "degree": 5,
            "max_synthesis_degree": 5,
            "shots": [100, 1000],
            "seed": 123,
        }
    )

    output_dir = run["output_dir"]
    expected = {
        "manifest.json",
        "observable_summary.csv",
        "shot_readout_summary.csv",
        "state_vector_diagnostics.json",
        "qsvt_readout_summary.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    observable_summary = pd.read_csv(output_dir / "observable_summary.csv")
    shot_summary = pd.read_csv(output_dir / "shot_readout_summary.csv")
    diagnostics = json.loads((output_dir / "state_vector_diagnostics.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary_md = (output_dir / "qsvt_readout_summary.md").read_text(encoding="utf-8")

    assert {
        "observable_name",
        "observable_type",
        "indices",
        "ridge_exact_normalized",
        "qsvt_exact_normalized",
        "absolute_error",
        "relative_error",
        "qsvt_minus_ridge",
        "notes",
    }.issubset(observable_summary.columns)
    assert {
        "observable_name",
        "observable_type",
        "shots",
        "seed",
        "true_qsvt_value",
        "shot_estimate",
        "standard_error",
        "absolute_sampling_error",
        "ridge_reference_value",
        "absolute_error_vs_ridge",
        "notes",
    }.issubset(shot_summary.columns)
    assert set(shot_summary["shots"]) == {100, 1000}
    assert diagnostics["qsvt_state_norm_after_normalization"] == pytest.approx(1.0)
    assert (
        "simulator validation checks, not readout claims"
        in diagnostics["full_vector_diagnostic_caveat"]
    )
    assert manifest["synthesized_degree"] == 5
    assert manifest["shot_counts"] == [100, 1000]
    assert "does not demonstrate scalable block encoding" in summary_md


def test_partial_readout_demo_rejects_non_power_of_two_submatrix() -> None:
    with pytest.raises(ValueError, match="power of two"):
        run_partial_observable_readout_demo({"submatrix_size": 3})
