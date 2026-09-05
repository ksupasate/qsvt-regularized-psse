from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.block_encoding import build_dense_block_encoding
from robust_qsvt_se.qsvt.tqe_circuit_level_block_encoding import (
    CIRCUIT_RESULTS_COLUMNS,
    BlockEncodingInput,
    build_dense_unitary_circuit,
    run_circuit_level_block_encoding,
    transpile_circuit_if_feasible,
    verify_operator_block_action,
    verify_statevector_actions,
)


def _toy_contraction() -> np.ndarray:
    return np.array([[0.45, 0.10], [0.00, 0.25]], dtype=np.float64)


def _toy_unitary() -> np.ndarray:
    return build_dense_block_encoding(_toy_contraction())


def _toy_input() -> BlockEncodingInput:
    A_bar = _toy_contraction()
    return BlockEncodingInput(
        case_name="toy",
        subproblem_size=2,
        selection_criterion="unit_test",
        A=A_bar.copy(),
        A_bar_padded=A_bar,
        U_A=_toy_unitary(),
        gamma=1.0,
        weighted_status="synthetic_unit_test",
        matrix_path="",
        padded_matrix_path="",
        unitary_path="",
    )


def test_dense_unitary_circuit_has_expected_qubit_count() -> None:
    pytest.importorskip("qiskit")
    bundle = build_dense_unitary_circuit(_toy_unitary())

    assert bundle.circuit is not None
    assert bundle.circuit.num_qubits == 2
    assert bundle.raw_depth == 1
    assert bundle.construction_status == "completed"


def test_operator_top_left_block_matches_normalized_matrix() -> None:
    pytest.importorskip("qiskit")
    A_bar = _toy_contraction()
    bundle = build_dense_unitary_circuit(_toy_unitary())
    checks = verify_operator_block_action(bundle.operator_matrix, A_bar)

    assert checks["block_fro_error"] <= 1.0e-12
    assert checks["block_spectral_error"] <= 1.0e-12
    assert checks["operator_unitarity_fro_error"] <= 1.0e-10


def test_statevector_postselected_action_matches_matrix_action() -> None:
    pytest.importorskip("qiskit")
    A_bar = _toy_contraction()
    bundle = build_dense_unitary_circuit(_toy_unitary())
    details = verify_statevector_actions(
        bundle.circuit,
        bundle.operator_matrix,
        A_bar,
        case_name="toy",
        subproblem_size=2,
        seed=7,
        random_state_count=0,
    )

    assert len(details) == 2
    assert max(row["action_abs_error"] for row in details) <= 1.0e-12
    assert all(np.isfinite(row["postselection_probability"]) for row in details)


def test_transpilation_metadata_is_populated_for_tiny_circuit() -> None:
    pytest.importorskip("qiskit")
    bundle = build_dense_unitary_circuit(_toy_unitary())
    metadata = transpile_circuit_if_feasible(
        bundle.circuit,
        num_qubits=2,
        basis_gates=["rz", "sx", "x", "cx"],
        transpile_qubit_limit=2,
    )

    assert metadata["transpilation_status"] == "completed"
    assert metadata["transpiled_depth"] >= 0
    assert metadata["transpiled_total_ops"] >= 0
    assert metadata["transpiled_cx_count"] >= 0


def test_circuit_level_output_csv_contains_required_columns(tmp_path: Path) -> None:
    A_bar = _toy_contraction()
    U = np.real(_toy_unitary())
    run = run_circuit_level_block_encoding(
        {
            "output_root": str(tmp_path),
            "input_blocks": [
                {
                    "case_name": "toy",
                    "subproblem_size": 2,
                    "selection_criterion": "unit_test",
                    "A": A_bar.tolist(),
                    "A_bar_padded": A_bar.tolist(),
                    "U_A": U.tolist(),
                    "gamma": 1.0,
                }
            ],
            "transpile_qubit_limit": 2,
            "random_state_count": 1,
        }
    )
    frame = pd.read_csv(run["artifacts"]["results_csv"])
    details = pd.read_csv(run["artifacts"]["statevector_details_csv"])

    assert set(CIRCUIT_RESULTS_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "simulation_status"] == "completed"
    assert frame.loc[0, "transpilation_status"] == "completed"
    assert not details.empty
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["action_errors_figure"].is_file()


def test_skipped_transpilation_is_recorded_with_reason() -> None:
    row, _details = run_single_toy_with_transpile_limit(limit=1)

    assert row["transpilation_status"] == "skipped_by_budget"
    assert "exceeds transpile_qubit_limit" in row["failure_or_skip_reason"]


def run_single_toy_with_transpile_limit(limit: int) -> tuple[pd.Series, pd.DataFrame]:
    from robust_qsvt_se.qsvt.tqe_circuit_level_block_encoding import (
        evaluate_circuit_block_encoding,
    )

    row, details = evaluate_circuit_block_encoding(
        _toy_input(),
        basis_gates=["rz", "sx", "x", "cx"],
        transpile_qubit_limit=limit,
        state_seed=10,
        random_state_count=1,
    )
    return pd.Series(row), pd.DataFrame(details)
