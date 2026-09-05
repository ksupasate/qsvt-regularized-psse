from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.qsvt_resource_phase_summary import build_qsvt_resource_phase_summary


def _make_inputs(root: Path) -> None:
    oracle = root / "qsvt_oracle_model_resources"
    oracle.mkdir(parents=True)
    (oracle / "oracle_model_resource_summary.csv").write_text(
        "case,matrix_rows,matrix_cols,alpha,degree,phase_count,qsvt_query_count,"
        "total_logical_qubits_padded_convention,success_probability_proxy,readout_shots,"
        "implemented_or_estimated\n"
        "ieee14,82,27,0.0001,51,52,103,13,0.4,1000,oracle_model_resource_estimate\n",
        encoding="utf-8",
    )
    full = root / "qsvt_resource_full_ieee"
    full.mkdir(parents=True)
    (full / "qsvt_resource_estimates.csv").write_text(
        "case_name,matrix_shape,matrix_columns,polynomial_degree,phase_count,"
        "estimated_qsvt_query_count,estimated_total_qubits,estimated_circuit_depth_proxy,"
        "estimated_gate_count_proxy,alpha\n"
        "ieee118,726x235,235,35,36,71,18,5000,200000,0.0001\n",
        encoding="utf-8",
    )


def test_resource_summary_separates_selected_from_full_case(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    _make_inputs(input_root)
    run = build_qsvt_resource_phase_summary(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase6")}
    )
    resources = pd.read_csv(run["artifacts"]["paper_table_qsvt_resource_summary"])
    scopes = set(resources["subproblem_or_full"].astype(str))
    assert "selected" in scopes
    assert "full" in scopes
    # The selected-subproblem row is the oracle-model estimate; the full row is resource-model only.
    selected = resources[resources["subproblem_or_full"] == "selected"].iloc[0]
    assert selected["resource_model_type"] == "oracle_model"
    full = resources[resources["subproblem_or_full"] == "full"].iloc[0]
    assert "resource-model only" in str(full["notes"])


def test_qsvt_summary_preserves_unsupported_claims(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    _make_inputs(input_root)
    run = build_qsvt_resource_phase_summary(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase6")}
    )
    claims = pd.read_csv(run["artifacts"]["paper_table_qsvt_claim_boundaries"])
    unsupported = claims[claims["support_status"] == "unsupported_do_not_claim"]
    text = " ".join(unsupported["claim"].astype(str)).lower()
    assert "superiority over ridge" in text
    assert "quantum speedup" in text


def test_missing_qsvt_outputs_records_documented_limitations(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    _make_inputs(input_root)
    run = build_qsvt_resource_phase_summary(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase6")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_qsvt_outputs"])
    blob = " ".join(missing["missing_output"].astype(str)).lower()
    assert "full output-direction" in blob
    assert "full ieee-scale gate-level qsvt execution" in blob
