from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.engineering_utils import RESOURCE_CAVEAT
from robust_qsvt_se.qsvt.resource_estimator import run_resource_readout_report


def test_resource_report_required_fields_and_readout_caveat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_resource_readout_report(
        {
            "output_dir": str(tmp_path / "resource"),
            "matrix_source": "synthetic",
            "alpha": 1.0e-2,
            "degrees": [2, 4],
            "epsilon": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "resource_summary.csv")
    shot_frame = pd.read_csv(output_dir / "shot_readout_summary.csv")
    row = frame.loc[0]
    readout = (output_dir / "readout_summary.md").read_text(encoding="utf-8")

    required = {
        "case_name",
        "matrix_source",
        "m",
        "n",
        "sparsity",
        "rank",
        "kappa",
        "alpha",
        "epsilon",
        "beta",
        "qsvt_degree_estimate",
        "query_count_estimate",
        "logical_qubits_estimate",
        "ancilla_qubits_estimate",
        "depth_estimate",
        "state_preparation_model",
        "readout_model",
        "full_vector_readout_required",
        "readout_caveat",
        "claim_strength",
    }
    assert required.issubset(frame.columns)
    assert int(row["qsvt_degree_estimate"]) >= 0
    assert int(row["query_count_estimate"]) >= 1
    assert int(row["logical_qubits_estimate"]) >= 1
    assert RESOURCE_CAVEAT in readout
    assert "Full vector reconstruction" in readout
    assert set(shot_frame["target_observable"]) == {
        "selected_state_component_0",
        "update_vector_norm",
        "residual_norm_proxy",
    }
    assert (shot_frame["shot_count"] == 4096).all()
    assert (shot_frame["estimated_standard_error"] > 0.0).all()
    assert "full_vector_reconstruction_caveat" in shot_frame.columns
    assert (output_dir / "shot_readout_summary.json").is_file()
    assert (output_dir / "resource_assumptions.md").is_file()
    assert (output_dir / "manifest.json").is_file()
