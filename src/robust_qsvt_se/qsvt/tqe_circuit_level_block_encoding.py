from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_additional_common import (
    CIRCUIT_BLOCK_ENCODING_DIR,
    CLAIM_BOUNDARY,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import construct_padded_block_encoding
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIRCUIT_RESULTS_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "original_matrix_shape",
    "padded_dimension",
    "dense_unitary_dimension",
    "num_qubits",
    "num_classical_bits",
    "system_qubits",
    "block_encoding_ancilla_qubits",
    "gamma",
    "norm_A_2",
    "norm_A_fro",
    "condition_number",
    "weighted_status",
    "circuit_framework",
    "qiskit_version",
    "circuit_construction_status",
    "circuit_depth_raw",
    "operation_counts_raw",
    "basis_gates",
    "transpilation_status",
    "transpiled_depth",
    "transpiled_total_ops",
    "transpiled_1q_ops",
    "transpiled_2q_ops",
    "transpiled_cx_count",
    "transpilation_seconds",
    "block_fro_error",
    "block_spectral_error",
    "relative_block_fro_error",
    "operator_unitarity_fro_error",
    "max_statevector_action_abs_error",
    "max_statevector_action_rel_error",
    "mean_postselection_probability",
    "min_postselection_probability",
    "max_postselection_probability",
    "simulation_status",
    "failure_or_skip_reason",
    "input_matrix_path",
    "input_padded_matrix_path",
    "input_unitary_path",
    "claim_boundary_note",
]

STATEVECTOR_DETAIL_COLUMNS = [
    "case_name",
    "subproblem_size",
    "state_type",
    "state_index",
    "action_abs_error",
    "action_rel_error",
    "postselection_probability",
]

DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]
SMALL_TOL = 1.0e-14


@dataclass(frozen=True, slots=True)
class BlockEncodingInput:
    case_name: str
    subproblem_size: int
    selection_criterion: str
    A: np.ndarray
    A_bar_padded: np.ndarray
    U_A: np.ndarray
    gamma: float
    weighted_status: str
    matrix_path: str
    padded_matrix_path: str
    unitary_path: str


@dataclass(frozen=True, slots=True)
class CircuitBundle:
    circuit: Any | None
    operator_matrix: np.ndarray
    raw_depth: int | None
    raw_counts: dict[str, int]
    num_classical_bits: int
    framework: str
    qiskit_version: str | None
    construction_status: str
    construction_reason: str


def run_circuit_level_block_encoding(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / CIRCUIT_BLOCK_ENCODING_DIR)
    figures_dir = paths["figures"]
    tables_dir = paths["tables"]
    reports_dir = paths["reports"]

    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    inputs = _load_block_encoding_inputs(resolved)
    for index, block_input in enumerate(inputs):
        try:
            row, details = evaluate_circuit_block_encoding(
                block_input,
                basis_gates=list(resolved["basis_gates"]),
                transpile_qubit_limit=int(resolved["transpile_qubit_limit"]),
                state_seed=int(resolved["seed"]) + index,
                random_state_count=int(resolved["random_state_count"]),
            )
        except Exception as exc:
            row = _failure_row(block_input, exc)
            details = []
        rows.append(row)
        state_rows.extend(details)

    results_frame = pd.DataFrame(rows, columns=CIRCUIT_RESULTS_COLUMNS)
    details_frame = pd.DataFrame(state_rows, columns=STATEVECTOR_DETAIL_COLUMNS)

    results_csv = output_dir / "circuit_level_block_encoding_results.csv"
    details_csv = output_dir / "statevector_action_verification_details.csv"
    metadata_json = output_dir / "circuit_level_block_encoding_metadata.json"
    summary_csv = tables_dir / "table_circuit_level_block_encoding_summary.csv"
    action_figure = figures_dir / "figure_circuit_block_action_errors.png"
    depth_figure = figures_dir / "figure_circuit_block_encoding_depth.png"
    cx_figure = figures_dir / "figure_circuit_block_encoding_cx_counts.png"
    report_path = reports_dir / "circuit_level_block_encoding_report.md"

    results_frame.to_csv(results_csv, index=False)
    details_frame.to_csv(details_csv, index=False)
    _summary_frame(results_frame).to_csv(summary_csv, index=False)
    _plot_action_errors(results_frame, action_figure)
    _plot_depths(results_frame, depth_figure)
    _plot_cx_counts(results_frame, cx_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            results=results_frame,
            details=details_frame,
            results_csv=results_csv,
            details_csv=details_csv,
            summary_csv=summary_csv,
        ),
        encoding="utf-8",
    )

    ended_at = utc_timestamp()
    artifacts = {
        "results_csv": str(results_csv),
        "statevector_details_csv": str(details_csv),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "action_errors_figure": str(action_figure),
        "depth_figure": str(depth_figure),
        "cx_counts_figure": str(cx_figure),
        "report": str(report_path),
    }
    write_json(
        metadata_json,
        reproducibility_metadata(
            config=resolved,
            started_at=started_at,
            ended_at=ended_at,
            status="completed",
            command=current_command(),
            artifacts=artifacts,
        )
        | {
            "input_matrix_file_paths": [item.matrix_path for item in inputs if item.matrix_path],
            "input_unitary_file_paths": [item.unitary_path for item in inputs if item.unitary_path],
            "transpiler_settings": {
                "basis_gates": list(resolved["basis_gates"]),
                "optimization_level": 1,
                "transpile_qubit_limit": int(resolved["transpile_qubit_limit"]),
            },
            "simulation_method": (
                "qiskit.quantum_info.Operator and Statevector when available; "
                "dense-matrix fallback otherwise"
            ),
            "status_counts": _status_counts(results_frame),
        },
    )
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results_frame,
        "statevector_details": details_frame,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def evaluate_circuit_block_encoding(
    block_input: BlockEncodingInput,
    *,
    basis_gates: list[str],
    transpile_qubit_limit: int,
    state_seed: int,
    random_state_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    U = np.asarray(block_input.U_A, dtype=np.complex128)
    A_bar_padded = np.asarray(block_input.A_bar_padded, dtype=np.complex128)
    _validate_unitary_dimensions(U, A_bar_padded)
    dense_dimension = int(U.shape[0])
    padded_dimension = int(A_bar_padded.shape[0])
    num_qubits = int(math.log2(dense_dimension))
    system_qubits = int(math.log2(padded_dimension))
    ancilla_qubits = num_qubits - system_qubits

    bundle = build_dense_unitary_circuit(U)
    operator_matrix = bundle.operator_matrix
    operator_checks = verify_operator_block_action(operator_matrix, A_bar_padded)
    details = verify_statevector_actions(
        bundle.circuit,
        operator_matrix,
        A_bar_padded,
        case_name=block_input.case_name,
        subproblem_size=block_input.subproblem_size,
        seed=state_seed,
        random_state_count=random_state_count,
    )
    transpile_metadata = transpile_circuit_if_feasible(
        bundle.circuit,
        num_qubits=num_qubits,
        basis_gates=basis_gates,
        transpile_qubit_limit=transpile_qubit_limit,
    )
    singular_values = np.linalg.svd(np.asarray(block_input.A, dtype=np.float64), compute_uv=False)
    condition = _condition_number(singular_values)
    failure_or_skip = _combine_reasons(
        [
            bundle.construction_reason,
            transpile_metadata["failure_or_skip_reason"],
        ]
    )

    return (
        {
            "case_name": block_input.case_name,
            "subproblem_size": block_input.subproblem_size,
            "selection_criterion": block_input.selection_criterion,
            "original_matrix_shape": f"{block_input.A.shape[0]}x{block_input.A.shape[1]}",
            "padded_dimension": padded_dimension,
            "dense_unitary_dimension": dense_dimension,
            "num_qubits": num_qubits,
            "num_classical_bits": bundle.num_classical_bits,
            "system_qubits": system_qubits,
            "block_encoding_ancilla_qubits": ancilla_qubits,
            "gamma": float(block_input.gamma),
            "norm_A_2": float(singular_values[0]) if singular_values.size else 0.0,
            "norm_A_fro": float(np.linalg.norm(block_input.A, ord="fro")),
            "condition_number": condition,
            "weighted_status": block_input.weighted_status,
            "circuit_framework": bundle.framework,
            "qiskit_version": bundle.qiskit_version,
            "circuit_construction_status": bundle.construction_status,
            "circuit_depth_raw": bundle.raw_depth,
            "operation_counts_raw": _counts_to_json(bundle.raw_counts),
            "basis_gates": ",".join(basis_gates),
            **transpile_metadata,
            **operator_checks,
            **_statevector_summary(details),
            "simulation_status": (
                "completed"
                if bundle.circuit is not None
                else "matrix_fallback_completed_qiskit_unavailable"
            ),
            "failure_or_skip_reason": failure_or_skip,
            "input_matrix_path": block_input.matrix_path,
            "input_padded_matrix_path": block_input.padded_matrix_path,
            "input_unitary_path": block_input.unitary_path,
            "claim_boundary_note": (
                "Explicit dense selected-subproblem circuit verification; not a scalable "
                "sparse-oracle block encoding."
            ),
        },
        details,
    )


def build_dense_unitary_circuit(U_A: np.ndarray) -> CircuitBundle:
    U = np.asarray(U_A, dtype=np.complex128)
    _validate_square_power_of_two(U)
    num_qubits = int(math.log2(U.shape[0]))
    try:
        import qiskit  # type: ignore[import-not-found]
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.library import UnitaryGate  # type: ignore[import-not-found]
        from qiskit.quantum_info import Operator  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        return CircuitBundle(
            circuit=None,
            operator_matrix=U,
            raw_depth=None,
            raw_counts={},
            num_classical_bits=0,
            framework="dense_matrix_fallback",
            qiskit_version=None,
            construction_status="qiskit_unavailable",
            construction_reason=f"qiskit_unavailable: {type(exc).__name__}: {exc}",
        )

    circuit = QuantumCircuit(num_qubits, name="dense_block_encoding")
    circuit.append(UnitaryGate(U, label="U_block"), range(num_qubits))
    return CircuitBundle(
        circuit=circuit,
        operator_matrix=np.asarray(Operator(circuit).data, dtype=np.complex128),
        raw_depth=int(circuit.depth()),
        raw_counts={str(key): int(value) for key, value in circuit.count_ops().items()},
        num_classical_bits=int(circuit.num_clbits),
        framework="qiskit",
        qiskit_version=getattr(qiskit, "__version__", None),
        construction_status="completed",
        construction_reason="",
    )


def transpile_circuit_if_feasible(
    circuit: Any | None,
    *,
    num_qubits: int,
    basis_gates: list[str],
    transpile_qubit_limit: int,
) -> dict[str, Any]:
    if circuit is None:
        return _empty_transpile_metadata(
            "qiskit_unavailable",
            "transpilation skipped because Qiskit circuit construction was unavailable",
        )
    if num_qubits > int(transpile_qubit_limit):
        return _empty_transpile_metadata(
            "skipped_by_budget",
            f"transpilation skipped: num_qubits={num_qubits} exceeds "
            f"transpile_qubit_limit={transpile_qubit_limit}",
        )
    try:
        from qiskit import transpile  # type: ignore[import-not-found]

        start = time.perf_counter()
        transpiled = transpile(
            circuit,
            basis_gates=list(basis_gates),
            optimization_level=1,
        )
        seconds = time.perf_counter() - start
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        return {
            "transpilation_status": "completed",
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_total_ops": int(sum(counts.values())),
            "transpiled_1q_ops": int(_count_ops(counts, {"rz", "sx", "x"})),
            "transpiled_2q_ops": int(_count_ops(counts, {"cx"})),
            "transpiled_cx_count": int(counts.get("cx", 0)),
            "transpilation_seconds": float(seconds),
            "failure_or_skip_reason": "",
        }
    except Exception as exc:  # pragma: no cover - backend-version dependent
        return _empty_transpile_metadata(
            "failed",
            f"transpilation failed: {type(exc).__name__}: {exc}",
        )


def verify_operator_block_action(
    operator_matrix: np.ndarray,
    A_bar_padded: np.ndarray,
) -> dict[str, float]:
    U = np.asarray(operator_matrix, dtype=np.complex128)
    target = np.asarray(A_bar_padded, dtype=np.complex128)
    _validate_unitary_dimensions(U, target)
    extracted = U[: target.shape[0], : target.shape[1]]
    delta = extracted - target
    unitary_delta = U.conj().T @ U - np.eye(U.shape[0], dtype=np.complex128)
    target_norm = float(np.linalg.norm(target, ord="fro"))
    return {
        "block_fro_error": float(np.linalg.norm(delta, ord="fro")),
        "block_spectral_error": float(np.linalg.norm(delta, ord=2)),
        "relative_block_fro_error": float(
            np.linalg.norm(delta, ord="fro") / max(target_norm, SMALL_TOL)
        ),
        "operator_unitarity_fro_error": float(np.linalg.norm(unitary_delta, ord="fro")),
    }


def verify_statevector_actions(
    circuit: Any | None,
    operator_matrix: np.ndarray,
    A_bar_padded: np.ndarray,
    *,
    case_name: str,
    subproblem_size: int,
    seed: int,
    random_state_count: int,
) -> list[dict[str, Any]]:
    target = np.asarray(A_bar_padded, dtype=np.complex128)
    dimension = int(target.shape[0])
    states: list[tuple[str, int, np.ndarray]] = []
    if dimension <= 16:
        for column in range(dimension):
            basis = np.zeros(dimension, dtype=np.complex128)
            basis[column] = 1.0
            states.append(("basis", column, basis))
    rng = np.random.default_rng(seed)
    for index in range(int(random_state_count)):
        vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        vector = vector / np.linalg.norm(vector)
        states.append(("deterministic_random", index, vector.astype(np.complex128)))

    rows = []
    for state_type, state_index, psi in states:
        full_input = np.zeros(operator_matrix.shape[0], dtype=np.complex128)
        full_input[:dimension] = psi
        if circuit is not None:
            try:
                from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

                evolved = Statevector(full_input).evolve(circuit).data
            except Exception:  # pragma: no cover - fallback for version-specific branches
                evolved = operator_matrix @ full_input
        else:
            evolved = operator_matrix @ full_input
        postselected = np.asarray(evolved[:dimension], dtype=np.complex128)
        expected = target @ psi
        abs_error = float(np.linalg.norm(postselected - expected))
        expected_norm = float(np.linalg.norm(expected))
        rows.append(
            {
                "case_name": case_name,
                "subproblem_size": int(subproblem_size),
                "state_type": state_type,
                "state_index": int(state_index),
                "action_abs_error": abs_error,
                "action_rel_error": float(abs_error / max(expected_norm, SMALL_TOL)),
                "postselection_probability": float(np.linalg.norm(postselected) ** 2),
            }
        )
    return rows


def _load_block_encoding_inputs(config: dict[str, Any]) -> list[BlockEncodingInput]:
    if "input_blocks" in config:
        return [_input_from_config_block(block) for block in config["input_blocks"]]

    input_csv = Path(config["input_results_csv"])
    if not input_csv.exists():
        raise FileNotFoundError(f"explicit block-encoding results CSV not found: {input_csv}")
    frame = pd.read_csv(input_csv)
    if "run_status" in frame:
        frame = frame[frame["run_status"] == "completed"].copy()
    if "max_input_blocks" in config and config["max_input_blocks"] is not None:
        frame = frame.head(int(config["max_input_blocks"]))
    inputs = [_input_from_results_row(row) for row in frame.to_dict(orient="records")]
    if not inputs:
        raise ValueError("no completed explicit block-encoding inputs were available")
    return inputs


def _input_from_config_block(block: dict[str, Any]) -> BlockEncodingInput:
    if "U_A" in block:
        U = np.asarray(block["U_A"], dtype=np.complex128)
    elif "unitary" in block:
        U = np.asarray(block["unitary"], dtype=np.complex128)
    else:
        raise ValueError("input block must contain U_A or unitary")
    A_bar_padded = np.asarray(block["A_bar_padded"], dtype=np.complex128)
    A = np.asarray(block.get("A", A_bar_padded), dtype=np.float64)
    return BlockEncodingInput(
        case_name=str(block.get("case_name", "synthetic")),
        subproblem_size=int(block.get("subproblem_size", A.shape[0])),
        selection_criterion=str(block.get("selection_criterion", "synthetic_fixed")),
        A=A,
        A_bar_padded=A_bar_padded,
        U_A=U,
        gamma=float(block.get("gamma", 1.0)),
        weighted_status=str(block.get("weighted_status", "synthetic_unit_test")),
        matrix_path=str(block.get("matrix_path", "")),
        padded_matrix_path=str(block.get("padded_matrix_path", "")),
        unitary_path=str(block.get("unitary_path", "")),
    )


def _input_from_results_row(row: dict[str, Any]) -> BlockEncodingInput:
    matrix_path = _path_from_value(row.get("matrix_original_path", ""))
    padded_path = _path_from_value(row.get("matrix_padded_path", ""))
    unitary_path = _path_from_value(row.get("unitary_path", ""))
    if not matrix_path.exists():
        raise FileNotFoundError(f"saved original matrix not found: {matrix_path}")
    A = np.asarray(np.load(matrix_path), dtype=np.float64)
    A_bar_padded = (
        np.asarray(np.load(padded_path), dtype=np.complex128)
        if padded_path.exists()
        else construct_padded_block_encoding(A, gamma=float(row["gamma"])).A_bar_padded
    )
    if unitary_path.exists():
        U_A = np.asarray(np.load(unitary_path), dtype=np.complex128)
    else:
        U_A = construct_padded_block_encoding(A, gamma=float(row["gamma"])).U
    return BlockEncodingInput(
        case_name=str(row.get("case_name", "unknown")),
        subproblem_size=int(row.get("subproblem_size", A.shape[0])),
        selection_criterion=str(row.get("selection_criterion", "unknown")),
        A=A,
        A_bar_padded=A_bar_padded,
        U_A=U_A,
        gamma=float(row.get("gamma", 1.0)),
        weighted_status=str(row.get("weighted_status", "weighted_jacobian_R_minus_half_H")),
        matrix_path=str(matrix_path),
        padded_matrix_path=str(padded_path) if padded_path.exists() else "",
        unitary_path=str(unitary_path) if unitary_path.exists() else "",
    )


def _summary_frame(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_name",
        "subproblem_size",
        "selection_criterion",
        "padded_dimension",
        "dense_unitary_dimension",
        "num_qubits",
        "system_qubits",
        "block_encoding_ancilla_qubits",
        "gamma",
        "condition_number",
        "transpilation_status",
        "transpiled_depth",
        "transpiled_cx_count",
        "block_fro_error",
        "block_spectral_error",
        "max_statevector_action_abs_error",
        "operator_unitarity_fro_error",
        "simulation_status",
        "failure_or_skip_reason",
    ]
    return results[[column for column in columns if column in results.columns]].copy()


def _plot_action_errors(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    completed = _completed_rows(results)
    if completed.empty:
        ax.text(0.5, 0.5, "No completed circuit simulations", ha="center", va="center")
    else:
        labels = _case_labels(completed)
        x = np.arange(len(completed))
        width = 0.25
        ax.bar(
            x - width,
            _positive_for_log(completed["block_fro_error"]),
            width,
            label="block Frobenius",
        )
        ax.bar(
            x, _positive_for_log(completed["block_spectral_error"]), width, label="block spectral"
        )
        ax.bar(
            x + width,
            _positive_for_log(completed["max_statevector_action_abs_error"]),
            width,
            label="max statevector action",
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("error")
        ax.set_title("Circuit-Level Dense Block-Encoding Action Errors")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_depths(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    completed = _completed_rows(results)
    if completed.empty:
        ax.text(0.5, 0.5, "No circuit objects constructed", ha="center", va="center")
    else:
        labels = _case_labels(completed)
        x = np.arange(len(completed))
        width = 0.35
        raw_depth = pd.to_numeric(completed["circuit_depth_raw"], errors="coerce").fillna(0.0)
        transpiled_depth = pd.to_numeric(completed["transpiled_depth"], errors="coerce").fillna(0.0)
        ax.bar(x - width / 2, raw_depth, width, label="raw dense-unitary circuit")
        ax.bar(x + width / 2, transpiled_depth, width, label="transpiled when attempted")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("circuit depth")
        skipped = int((completed["transpilation_status"] == "skipped_by_budget").sum())
        ax.set_title(f"Circuit Depth Diagnostics ({skipped} transpilation skips)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_cx_counts(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    completed = (
        results[results["transpilation_status"] == "completed"] if not results.empty else results
    )
    if completed.empty:
        ax.text(0.5, 0.5, "No transpiled circuits", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = _case_labels(completed)
        x = np.arange(len(completed))
        ax.bar(x, completed["transpiled_cx_count"].astype(float))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("CX count")
        ax.set_title("Transpiled Dense Block-Encoding CX Counts")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _report_markdown(
    *,
    config: dict[str, Any],
    results: pd.DataFrame,
    details: pd.DataFrame,
    results_csv: Path,
    details_csv: Path,
    summary_csv: Path,
) -> str:
    completed = _completed_rows(results)
    transpiled = (
        results[results["transpilation_status"] == "completed"] if not results.empty else results
    )
    framework_counts = (
        results["circuit_framework"].value_counts().to_dict() if not results.empty else {}
    )
    simulation_counts = (
        results["simulation_status"].value_counts().to_dict() if not results.empty else {}
    )
    transpilation_counts = (
        results["transpilation_status"].value_counts().to_dict() if not results.empty else {}
    )
    input_lines = [
        "- "
        f"{row.case_name}, size={int(row.subproblem_size)}, "
        f"selection={row.selection_criterion}, U=`{row.input_unitary_path}`"
        for row in results.itertuples(index=False)
    ] or ["- No input matrices were evaluated."]
    metric_lines = _metric_lines(completed, transpiled, details)
    return "\n".join(
        [
            "# Circuit-Level Dense Block-Encoding Verification Report",
            "",
            "## Experiment Goal",
            "",
            "This experiment constructs explicit circuit objects from dense block-encoding "
            "unitaries for selected IEEE-derived weighted-Jacobian blocks and verifies "
            "their simulated block action.",
            "",
            "## Command Used",
            "",
            f"`{current_command()}`",
            "",
            "## Input Matrices Used",
            "",
            *input_lines,
            "",
            "## Circuit Framework Used",
            "",
            "- Preferred framework: Qiskit.",
            "- Dense fallback: direct matrix action only if Qiskit is unavailable.",
            f"- Observed frameworks: {framework_counts}",
            "",
            "## Circuit Construction Method",
            "",
            "For each saved dense block-encoding unitary U_A, the experiment creates a "
            'Qiskit QuantumCircuit and appends U_A as `UnitaryGate(U_A, label="U_block")` '
            "on all qubits. The top-left block is interpreted as the block obtained by "
            "postselecting the block-encoding ancilla convention used by the dense dilation.",
            "",
            "## Transpilation Basis and Settings",
            "",
            f"- Basis gates: {config['basis_gates']}",
            "- Optimization level: 1",
            f"- Transpilation qubit limit: {config['transpile_qubit_limit']}",
            "",
            "## Verification Methods",
            "",
            "- Operator top-left block verification compares the circuit Operator block "
            "with the saved padded normalized matrix A_bar.",
            "- Statevector action verification embeds deterministic system-register test "
            "states in the ancilla-zero sector and compares the postselected action "
            "with A_bar |psi>.",
            "- These are simulator-level checks of circuit objects, not hardware execution.",
            "",
            "## Success, Failure, and Skipped Counts",
            "",
            f"- Simulation statuses: {simulation_counts}",
            f"- Transpilation statuses: {transpilation_counts}",
            "",
            "## Key Numerical Results and Resource Estimates",
            "",
            *metric_lines,
            "",
            "## Claim-Safe Interpretation",
            "",
            "This experiment constructs explicit circuit objects from the dense "
            "block-encoding unitaries for selected IEEE-derived weighted-Jacobian "
            "blocks. Operator-level and statevector-level checks verify that the "
            "circuit block action reproduces the normalized weighted-Jacobian block "
            "within numerical tolerance.",
            "",
            "The circuits are dense selected-subproblem constructions and are not "
            "claimed to be scalable sparse-oracle block encodings for full IEEE-scale "
            "PSSE. Transpiled gate counts and depths are proof-of-concept resource "
            "diagnostics, not optimized hardware implementation costs.",
            "",
            "## Limitations",
            "",
            "- Dense unitary gates are loaded explicitly; this is not a scalable oracle.",
            "- 5-qubit dense decompositions are skipped by default to keep the experiment "
            "audit-friendly and reproducible.",
            "- State preparation, full-vector readout, and full IEEE-scale PSSE circuits "
            "remain outside this experiment.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{results_csv}`",
            f"- Statevector details CSV: `{details_csv}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _metric_lines(
    completed: pd.DataFrame,
    transpiled: pd.DataFrame,
    details: pd.DataFrame,
) -> list[str]:
    if completed.empty:
        return ["- No completed circuit simulations."]
    lines = [
        f"- Circuits constructed: {len(completed)}",
        f"- Maximum block Frobenius error: {completed['block_fro_error'].max():.3e}",
        "- Maximum statevector action absolute error: "
        f"{completed['max_statevector_action_abs_error'].max():.3e}",
        "- Maximum operator unitarity Frobenius error: "
        f"{completed['operator_unitarity_fro_error'].max():.3e}",
        "- Circuit depth raw range: "
        f"{int(completed['circuit_depth_raw'].min())} to "
        f"{int(completed['circuit_depth_raw'].max())}",
    ]
    if not transpiled.empty:
        lines.append(
            "- Transpiled depth range: "
            f"{int(transpiled['transpiled_depth'].min())} to "
            f"{int(transpiled['transpiled_depth'].max())}; CX count range: "
            f"{int(transpiled['transpiled_cx_count'].min())} to "
            f"{int(transpiled['transpiled_cx_count'].max())}."
        )
    else:
        lines.append("- No circuits were transpiled under the configured budget.")
    if not details.empty:
        lines.append(
            "- Postselection probability range across test states: "
            f"{details['postselection_probability'].min():.3e} to "
            f"{details['postselection_probability'].max():.3e}."
        )
    return lines


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 321,
        "input_results_csv": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "basis_gates": DEFAULT_BASIS_GATES,
        "transpile_qubit_limit": 4,
        "random_state_count": 3,
        "max_input_blocks": None,
    }
    if config:
        resolved.update(config)
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    if int(resolved["transpile_qubit_limit"]) < 1:
        raise ValueError("transpile_qubit_limit must be positive")
    if int(resolved["random_state_count"]) < 0:
        raise ValueError("random_state_count must be nonnegative")
    return resolved


def _failure_row(block_input: BlockEncodingInput, exc: Exception) -> dict[str, Any]:
    row = {key: np.nan for key in CIRCUIT_RESULTS_COLUMNS}
    row.update(
        {
            "case_name": block_input.case_name,
            "subproblem_size": block_input.subproblem_size,
            "selection_criterion": block_input.selection_criterion,
            "original_matrix_shape": f"{block_input.A.shape[0]}x{block_input.A.shape[1]}",
            "weighted_status": block_input.weighted_status,
            "circuit_construction_status": "failed",
            "transpilation_status": "not_attempted",
            "simulation_status": "failed",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
            "input_matrix_path": block_input.matrix_path,
            "input_padded_matrix_path": block_input.padded_matrix_path,
            "input_unitary_path": block_input.unitary_path,
            "claim_boundary_note": (
                "Failure recorded explicitly; no circuit-level block-encoding claim for this row."
            ),
        }
    )
    return row


def _empty_transpile_metadata(status: str, reason: str) -> dict[str, Any]:
    return {
        "transpilation_status": status,
        "transpiled_depth": np.nan,
        "transpiled_total_ops": np.nan,
        "transpiled_1q_ops": np.nan,
        "transpiled_2q_ops": np.nan,
        "transpiled_cx_count": np.nan,
        "transpilation_seconds": 0.0,
        "failure_or_skip_reason": reason,
    }


def _statevector_summary(details: list[dict[str, Any]]) -> dict[str, float]:
    if not details:
        return {
            "max_statevector_action_abs_error": np.nan,
            "max_statevector_action_rel_error": np.nan,
            "mean_postselection_probability": np.nan,
            "min_postselection_probability": np.nan,
            "max_postselection_probability": np.nan,
        }
    frame = pd.DataFrame(details)
    return {
        "max_statevector_action_abs_error": float(frame["action_abs_error"].max()),
        "max_statevector_action_rel_error": float(frame["action_rel_error"].max()),
        "mean_postselection_probability": float(frame["postselection_probability"].mean()),
        "min_postselection_probability": float(frame["postselection_probability"].min()),
        "max_postselection_probability": float(frame["postselection_probability"].max()),
    }


def _status_counts(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    if results.empty:
        return {"simulation_status": {}, "transpilation_status": {}}
    return {
        "simulation_status": {
            str(key): int(value)
            for key, value in results["simulation_status"].value_counts(dropna=False).items()
        },
        "transpilation_status": {
            str(key): int(value)
            for key, value in results["transpilation_status"].value_counts(dropna=False).items()
        },
    }


def _count_ops(counts: dict[str, int], names: set[str]) -> int:
    return sum(int(value) for key, value in counts.items() if key in names)


def _counts_to_json(counts: dict[str, int]) -> str:
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def _combine_reasons(reasons: list[str]) -> str:
    return "; ".join(reason for reason in reasons if str(reason).strip())


def _path_from_value(value: Any) -> Path:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return Path("__missing_tqe_circuit_block_path__")
    text = str(value).strip()
    return Path(text) if text else Path("__missing_tqe_circuit_block_path__")


def _case_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{row.case_name}-{int(row.subproblem_size)}" for row in frame.itertuples(index=False)]


def _completed_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "simulation_status" not in results:
        return results
    return results[results["simulation_status"].astype(str).str.contains("completed", na=False)]


def _positive_for_log(values: pd.Series) -> np.ndarray:
    return np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 1.0e-18)


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(np.max(positive) / np.min(positive))


def _validate_unitary_dimensions(U: np.ndarray, A_bar_padded: np.ndarray) -> None:
    _validate_square_power_of_two(U)
    if A_bar_padded.ndim != 2 or A_bar_padded.shape[0] != A_bar_padded.shape[1]:
        raise ValueError("A_bar_padded must be square")
    if not _is_power_of_two(A_bar_padded.shape[0]):
        raise ValueError("A_bar_padded dimension must be a power of two")
    if U.shape[0] != 2 * A_bar_padded.shape[0]:
        raise ValueError("U_A dimension must be twice the padded matrix dimension")


def _validate_square_power_of_two(U: np.ndarray) -> None:
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U_A must be square")
    if not _is_power_of_two(U.shape[0]):
        raise ValueError("U_A dimension must be a power of two")
    if not np.all(np.isfinite(U)):
        raise ValueError("U_A entries must be finite")


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run TQE circuit-level block-encoding verification"
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--transpile-qubit-limit", type=int, default=4)
    args = parser.parse_args(argv)
    run = run_circuit_level_block_encoding(
        {
            "output_root": args.output_root,
            "input_results_csv": str(
                Path(args.output_root)
                / "explicit_block_encoding_demo"
                / "block_encoding_demo_results.csv"
            ),
            "transpile_qubit_limit": args.transpile_qubit_limit,
        }
    )
    print(f"TQE circuit-level block-encoding verification complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
