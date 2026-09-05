from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model import (
    build_hardware_aware_oracle_cost_model,
    write_oracle_cost_outputs,
)


def test_hardware_aware_oracle_cost_model_includes_all_ieee_cases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_system(config):
        scale = int(str(config["case_name"]).replace("ieee", ""))
        matrix = np.eye(3, dtype=np.float64) * (1.0 + scale / 1000.0)
        return SimpleNamespace(H_tilde=matrix), "fake_matrix"

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model.build_engineering_system",
        fake_system,
    )

    run = build_hardware_aware_oracle_cost_model(
        {
            "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
            "value_precision_bits": [8],
            "degrees": [35],
            "observable_readout_shots": [1000],
            "output_dir": str(tmp_path),
        }
    )
    cases = {row["case"] for row in run["rows"]}

    assert cases == {"ieee14", "ieee30", "ieee57", "ieee118", "ieee300"}
    assert all(row["signal_unitary_calls"] == 35 for row in run["rows"])
    assert all(row["projector_phase_operations"] == 36 for row in run["rows"])
    assert all(row["alternating_sequence_length"] == 71 for row in run["rows"])
    assert all(row["qsvt_query_count"] == 35 for row in run["rows"])
    assert run["artifacts"]["oracle_cost_by_case"].is_file()


def test_hardware_aware_oracle_cost_varies_with_precision(monkeypatch, tmp_path: Path) -> None:
    def fake_system(config):
        return SimpleNamespace(H_tilde=np.array([[1.0, 0.0], [0.5, 1.0]])), "fake_matrix"

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model.build_engineering_system",
        fake_system,
    )

    run = build_hardware_aware_oracle_cost_model(
        {
            "cases": ["ieee14"],
            "value_precision_bits": [8, 16],
            "degrees": [35],
            "observable_readout_shots": [1000],
            "output_dir": str(tmp_path),
        }
    )
    frame = pd.DataFrame(run["rows"]).sort_values("value_precision_bits")

    assert (
        frame.iloc[0]["estimated_block_encoding_query_cost"]
        < frame.iloc[1]["estimated_block_encoding_query_cost"]
    )


def test_hardware_aware_oracle_outputs_required_files(tmp_path: Path) -> None:
    row = {
        "case": "ieee14",
        "matrix_shape": "2x2",
        "nnz": 2,
        "density": 0.5,
        "max_row_sparsity": 1,
        "max_col_sparsity": 1,
        "row_qubits": 1,
        "col_qubits": 1,
        "value_precision_bits": 8,
        "work_qubits": 10,
        "ancilla_qubits": 7,
        "index_lookup_model": "table_lookup_or_qrom_proxy",
        "value_loading_model": "fixed_point_value_register",
        "rotation_synthesis_model": "clifford_t_proxy",
        "state_preparation_model": "sparse_residual_loading",
        "estimated_row_lookup_cost": 5.0,
        "estimated_value_loading_cost": 72.0,
        "estimated_rotation_cost": 192.0,
        "estimated_uncompute_cost": 77.0,
        "estimated_block_encoding_query_cost": 346.0,
        "qsvt_degree": 35,
        "qsvt_query_count": 71,
        "success_probability_proxy": 0.1,
        "amplitude_amplification_proxy": 3.16,
        "estimated_total_query_cost": 100.0,
        "estimated_total_gate_cost_proxy": 200.0,
        "observable_readout_shots": 1000,
        "resource_status": "oracle_model_estimate",
        "dominant_cost_source": "rotation",
        "assumption_risk_level": "moderate",
    }

    artifacts = write_oracle_cost_outputs(tmp_path, {"output_dir": str(tmp_path)}, [row])

    assert artifacts["qsvt_total_cost_estimate"].is_file()
    assert artifacts["amplification_adjusted_cost"].is_file()
    assert "Full IEEE-scale QSVT remains" in artifacts["oracle_cost_assumption_summary"].read_text(
        encoding="utf-8"
    )
