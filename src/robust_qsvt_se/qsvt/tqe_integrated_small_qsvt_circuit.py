from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    INTEGRATED_QSVT_CIRCUIT_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import load_sweep_subproblem
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import (
    fit_actual_singular_interpolating_polynomial,
    ridge_update_svd,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import construct_padded_block_encoding
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_COLUMNS = [
    "run_type",
    "case_name",
    "subproblem_size",
    "alpha",
    "epsilon_target",
    "degree",
    "phase_count",
    "phase_synthesis_status",
    "phase_convention",
    "qsvt_sequence_status",
    "gamma",
    "C_alpha",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "transform_block_error_fro",
    "transform_block_error_spectral",
    "max_singular_response_error",
    "circuit_vs_polynomial_fro_error",
    "circuit_vs_polynomial_spectral_error",
    "residual_no_update",
    "residual_ridge",
    "residual_qsvt_circuit",
    "ridge_residual_ratio",
    "qsvt_circuit_residual_ratio",
    "residual_gap",
    "relative_update_error",
    "absolute_update_error",
    "max_component_error",
    "cosine_similarity",
    "num_qubits",
    "num_U_calls",
    "num_U_dagger_calls",
    "num_phase_rotations",
    "raw_circuit_depth",
    "transpilation_status",
    "transpiled_depth",
    "transpiled_cx_count",
    "transpiled_total_ops",
    "success_probability_residual_state",
    "simulation_status",
    "failure_or_skip_reason",
]

PROBE_COLUMNS = [
    "run_type",
    "case_name",
    "state_type",
    "state_index",
    "action_abs_error",
    "action_rel_error",
    "success_probability",
]

DEFAULT_PHASE_CONVENTION = (
    "pennylane_poly_to_angles_QSVT; repository structured sequence matching "
    "qml.QSVT; projector-controlled PCPhase; original phase order; real(top-left "
    "block) is interpreted as the odd-polynomial signal transform"
)
DEFAULT_BASIS_GATES = ["rz", "sx", "x", "cx"]
SMALL_TOL = 1.0e-14


@dataclass(frozen=True, slots=True)
class IntegratedEvaluation:
    row: dict[str, Any]
    probes: list[dict[str, Any]]
    transformed_block: np.ndarray | None
    polynomial_block: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PhaseSynthesisResult:
    phases: np.ndarray
    status: str
    failure_reason: str
    convention: str


def run_integrated_small_qsvt_circuit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / INTEGRATED_QSVT_CIRCUIT_DIR)
    figures_dir = paths["figures"]
    tables_dir = paths["tables"]
    reports_dir = paths["reports"]

    rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    sanity = run_sanity_check(resolved)
    rows.append(sanity.row)
    sanity_rows.append(sanity.row)
    probe_rows.extend(sanity.probes)

    if sanity.row["qsvt_sequence_status"] == "sanity_passed" and bool(resolved["run_ieee"]):
        try:
            ieee = run_ieee_selected_block(resolved)
        except Exception as exc:
            ieee = _ieee_exception_row(resolved, exc)
        rows.append(ieee.row)
        probe_rows.extend(ieee.probes)
    elif bool(resolved["run_ieee"]):
        rows.append(_skipped_ieee_row(resolved, str(sanity.row["failure_or_skip_reason"])))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    sanity_frame = pd.DataFrame(sanity_rows, columns=RESULT_COLUMNS)
    probes = pd.DataFrame(probe_rows, columns=PROBE_COLUMNS)
    summary = _summary_frame(results)

    results_csv = output_dir / "integrated_small_qsvt_circuit_results.csv"
    sanity_csv = output_dir / "sanity_check_results.csv"
    probes_csv = output_dir / "statevector_probe_details.csv"
    metadata_json = output_dir / "integrated_small_qsvt_circuit_metadata.json"
    summary_csv = tables_dir / "table_integrated_small_qsvt_circuit_summary.csv"
    update_figure = figures_dir / "figure_integrated_qsvt_circuit_update_error.png"
    transform_figure = figures_dir / "figure_integrated_qsvt_circuit_transform_error.png"
    resource_figure = figures_dir / "figure_integrated_qsvt_circuit_resource_counts.png"
    report_path = reports_dir / "integrated_small_qsvt_circuit_report.md"

    results.to_csv(results_csv, index=False)
    sanity_frame.to_csv(sanity_csv, index=False)
    probes.to_csv(probes_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_update_error(results, update_figure)
    _plot_transform_error(results, transform_figure)
    _plot_resource_counts(results, resource_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            results=results,
            summary=summary,
            results_csv=results_csv,
            sanity_csv=sanity_csv,
            probes_csv=probes_csv,
            summary_csv=summary_csv,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "results_csv": str(results_csv),
        "sanity_check_csv": str(sanity_csv),
        "statevector_probe_details_csv": str(probes_csv),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "update_error_figure": str(update_figure),
        "transform_error_figure": str(transform_figure),
        "resource_counts_figure": str(resource_figure),
        "report": str(report_path),
    }
    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts=artifacts,
    )
    metadata.update(
        {
            "phase_synthesis_settings": {
                "backend": "pennylane_poly_to_angles",
                "angle_solver": resolved["angle_solver"],
                "phase_convention": DEFAULT_PHASE_CONVENTION,
            },
            "transpilation_settings": {
                "basis_gates": list(resolved["basis_gates"]),
                "optimization_level": int(resolved["transpile_optimization_level"]),
                "transpile_qubit_limit": int(resolved["transpile_qubit_limit"]),
            },
            "simulation_settings": {
                "operator_backend": "qiskit.quantum_info.Operator",
                "statevector_backend": "qiskit.quantum_info.Statevector",
                "signal_component": "real_top_left_block",
            },
            "input_matrix_paths": _input_matrix_paths(resolved),
            "status_counts": _status_counts(results),
        }
    )
    write_json(metadata_json, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results,
        "sanity": sanity_frame,
        "probes": probes,
        "summary": summary,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def run_sanity_check(config: dict[str, Any]) -> IntegratedEvaluation:
    matrix = np.diag(np.asarray(config["sanity_singular_values"], dtype=np.float64))
    coefficients = np.asarray(config["sanity_polynomial_coefficients"], dtype=np.float64)
    degree = len(coefficients) - 1
    polynomial = Polynomial(coefficients)
    encoding = construct_padded_block_encoding(matrix, gamma=1.0)
    phase_result = synthesize_qsvt_phases(
        coefficients,
        angle_solver=str(config["angle_solver"]),
    )
    if phase_result.status != "completed":
        return _sanity_failure(
            config=config,
            matrix=matrix,
            degree=degree,
            phase_result=phase_result,
            reason=phase_result.failure_reason,
        )
    try:
        evaluation = evaluate_qsvt_transform(
            run_type="sanity_check",
            case_name="diagonal_sanity",
            subproblem_size=matrix.shape[0],
            A=matrix,
            b=None,
            A_bar_padded=encoding.A_bar_padded,
            U_A=encoding.U,
            gamma=1.0,
            C_alpha=1.0,
            alpha=np.nan,
            epsilon_target=np.nan,
            degree=degree,
            polynomial=polynomial,
            phases=phase_result.phases,
            phase_result=phase_result,
            basis_gates=list(config["basis_gates"]),
            transpile_qubit_limit=int(config["transpile_qubit_limit"]),
            transpile_optimization_level=int(config["transpile_optimization_level"]),
        )
    except Exception as exc:
        return _sanity_failure(
            config=config,
            matrix=matrix,
            degree=degree,
            phase_result=phase_result,
            reason=f"{type(exc).__name__}: {exc}",
        )

    error = float(evaluation.row["transform_block_error_fro"])
    if bool(config["force_sanity_convention_failure"]):
        error = float("inf")
        evaluation.row["transform_block_error_fro"] = error
        evaluation.row["failure_or_skip_reason"] = "forced sanity convention mismatch for testing"
    if error <= float(config["sanity_tolerance"]):
        evaluation.row["qsvt_sequence_status"] = "sanity_passed"
        evaluation.row["simulation_status"] = "completed"
    else:
        evaluation.row["qsvt_sequence_status"] = "failed_convention_mismatch"
        evaluation.row["simulation_status"] = "failed_convention_mismatch"
        if not evaluation.row["failure_or_skip_reason"]:
            evaluation.row["failure_or_skip_reason"] = (
                "sanity transform did not match the known polynomial under the documented "
                "phase convention"
            )
    return evaluation


def run_ieee_selected_block(config: dict[str, Any]) -> IntegratedEvaluation:
    spec = dict(config["subproblem_spec"])
    subproblem = load_sweep_subproblem(spec, seed=int(config["seed"]))
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    b = np.asarray(subproblem.r_tilde, dtype=np.float64)
    artifact = _load_block_artifacts(config, A)
    if artifact["A"] is not None and not np.allclose(
        A,
        artifact["A"],
        rtol=float(config["artifact_match_rtol"]),
        atol=float(config["artifact_match_atol"]),
    ):
        raise ValueError(
            "saved block matrix does not match the reconstructed selected subproblem; "
            "check the seed and selection configuration before running the integrated circuit"
        )
    A_bar_padded = artifact["A_bar_padded"]
    U_A = artifact["U_A"]
    gamma = float(artifact["gamma"])
    degree = _select_ieee_degree(config)
    singular_values = np.linalg.svd(A, compute_uv=False)
    cheb, C_alpha = fit_actual_singular_interpolating_polynomial(
        alpha=float(config["alpha"]),
        gamma=gamma,
        singular_values=singular_values,
        degree=degree,
    )
    coefficients = cheb.convert(kind=Polynomial).coef
    coefficients = _pad_coefficients(coefficients, degree)
    polynomial = Polynomial(coefficients)
    phase_result = synthesize_qsvt_phases(
        coefficients,
        angle_solver=str(config["angle_solver"]),
    )
    if phase_result.status != "completed":
        return _ieee_phase_failure_row(
            config=config,
            A=A,
            gamma=gamma,
            C_alpha=C_alpha,
            degree=degree,
            phase_result=phase_result,
        )
    return evaluate_qsvt_transform(
        run_type="ieee_selected_block",
        case_name=str(subproblem.metadata.get("case_name", spec.get("case_name", "unknown"))),
        subproblem_size=int(subproblem.metadata.get("subproblem_size", min(A.shape))),
        A=A,
        b=b,
        A_bar_padded=A_bar_padded,
        U_A=U_A,
        gamma=gamma,
        C_alpha=C_alpha,
        alpha=float(config["alpha"]),
        epsilon_target=float(config["epsilon_target"]),
        degree=degree,
        polynomial=polynomial,
        phases=phase_result.phases,
        phase_result=phase_result,
        basis_gates=list(config["basis_gates"]),
        transpile_qubit_limit=int(config["transpile_qubit_limit"]),
        transpile_optimization_level=int(config["transpile_optimization_level"]),
    )


def synthesize_qsvt_phases(coefficients: np.ndarray, *, angle_solver: str) -> PhaseSynthesisResult:
    try:
        import pennylane as qml  # type: ignore[import-not-found]

        phases = np.asarray(
            qml.poly_to_angles(
                np.asarray(coefficients, dtype=np.float64),
                "QSVT",
                angle_solver=str(angle_solver),
            ),
            dtype=np.float64,
        )
        return PhaseSynthesisResult(
            phases=phases,
            status="completed",
            failure_reason="",
            convention=DEFAULT_PHASE_CONVENTION,
        )
    except Exception as exc:  # pragma: no cover - optional dependency/version branch
        return PhaseSynthesisResult(
            phases=np.array([], dtype=np.float64),
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            convention=DEFAULT_PHASE_CONVENTION,
        )


def evaluate_qsvt_transform(
    *,
    run_type: str,
    case_name: str,
    subproblem_size: int,
    A: np.ndarray,
    b: np.ndarray | None,
    A_bar_padded: np.ndarray,
    U_A: np.ndarray,
    gamma: float,
    C_alpha: float,
    alpha: float,
    epsilon_target: float,
    degree: int,
    polynomial: Polynomial,
    phases: np.ndarray,
    phase_result: PhaseSynthesisResult,
    basis_gates: list[str],
    transpile_qubit_limit: int,
    transpile_optimization_level: int,
) -> IntegratedEvaluation:
    from qiskit.quantum_info import Operator  # type: ignore[import-not-found]

    A_values = np.asarray(A, dtype=np.float64)
    target = np.asarray(A_bar_padded, dtype=np.complex128)
    U_values = np.asarray(U_A, dtype=np.complex128)
    bundle = build_structured_qsvt_operator_circuit(
        U_values,
        np.asarray(phases, dtype=np.float64),
        encoded_dimension=target.shape[0],
    )
    operator_matrix = np.asarray(Operator(bundle.qsvt_operator_circuit).data, dtype=np.complex128)
    transformed = np.real(operator_matrix[: target.shape[0], : target.shape[1]])
    polynomial_block = polynomial_singular_transform(target, polynomial)
    errors = _transform_errors(transformed, polynomial_block, target, polynomial)
    call_counts = _call_counts(len(phases))
    transpile = _transpile_qsvt_circuit(
        bundle.qsvt_operator_circuit,
        basis_gates=basis_gates,
        optimization_level=transpile_optimization_level,
        qubit_limit=transpile_qubit_limit,
    )
    residual_values = _residual_and_update_metrics(
        A=A_values,
        b=b,
        transformed=transformed[: A_values.shape[0], : A_values.shape[1]],
        C_alpha=float(C_alpha),
        alpha=alpha,
        operator_matrix=operator_matrix,
    )
    singular_values = np.linalg.svd(A_values, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else np.inf
    probes = _statevector_probe_rows(
        run_type=run_type,
        case_name=case_name,
        circuit=bundle.qsvt_operator_circuit,
        operator_matrix=operator_matrix,
        transformed=transformed,
        expected=polynomial_block,
        residual=b,
    )
    row = {
        "run_type": run_type,
        "case_name": case_name,
        "subproblem_size": int(subproblem_size),
        "alpha": alpha,
        "epsilon_target": epsilon_target,
        "degree": int(degree),
        "phase_count": len(phases),
        "phase_synthesis_status": phase_result.status,
        "phase_convention": phase_result.convention,
        "qsvt_sequence_status": "completed",
        "gamma": float(gamma),
        "C_alpha": float(C_alpha),
        "condition_number": condition,
        "sigma_min": float(np.min(positive)) if positive.size else np.nan,
        "sigma_max": float(np.max(positive)) if positive.size else np.nan,
        **errors,
        **residual_values,
        "num_qubits": int(math.log2(operator_matrix.shape[0])),
        "num_U_calls": call_counts["num_U_calls"],
        "num_U_dagger_calls": call_counts["num_U_dagger_calls"],
        "num_phase_rotations": len(phases),
        "raw_circuit_depth": int(bundle.qsvt_operator_circuit.depth()),
        "transpilation_status": transpile["transpilation_status"],
        "transpiled_depth": transpile["transpiled_depth"],
        "transpiled_cx_count": transpile["transpiled_cx_count"],
        "transpiled_total_ops": transpile["transpiled_total_ops"],
        "success_probability_residual_state": residual_values["success_probability_residual_state"],
        "simulation_status": "completed",
        "failure_or_skip_reason": transpile["failure_or_skip_reason"],
    }
    return IntegratedEvaluation(
        row=row,
        probes=probes,
        transformed_block=transformed,
        polynomial_block=polynomial_block,
    )


def polynomial_singular_transform(A_bar: np.ndarray, polynomial: Polynomial) -> np.ndarray:
    matrix = _as_real_matrix(A_bar, name="A_bar")
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=True)
    values = np.asarray(polynomial(singular_values), dtype=np.float64)
    return U @ np.diag(values) @ Vt


def qsvt_rescaled_update_from_transform(
    transformed: np.ndarray,
    b: np.ndarray,
    *,
    C_alpha: float,
) -> np.ndarray:
    return (
        float(C_alpha)
        * np.asarray(transformed, dtype=np.float64).T
        @ np.asarray(b, dtype=np.float64)
    )


def _residual_and_update_metrics(
    *,
    A: np.ndarray,
    b: np.ndarray | None,
    transformed: np.ndarray,
    C_alpha: float,
    alpha: float,
    operator_matrix: np.ndarray,
) -> dict[str, float]:
    if b is None or not np.isfinite(alpha):
        return {
            "residual_no_update": np.nan,
            "residual_ridge": np.nan,
            "residual_qsvt_circuit": np.nan,
            "ridge_residual_ratio": np.nan,
            "qsvt_circuit_residual_ratio": np.nan,
            "residual_gap": np.nan,
            "relative_update_error": np.nan,
            "absolute_update_error": np.nan,
            "max_component_error": np.nan,
            "cosine_similarity": np.nan,
            "success_probability_residual_state": np.nan,
        }
    residual = np.asarray(b, dtype=np.float64)
    ridge_update = ridge_update_svd(A, residual, alpha=float(alpha))
    qsvt_update = qsvt_rescaled_update_from_transform(
        transformed,
        residual,
        C_alpha=float(C_alpha),
    )
    residual_no_update = float(np.linalg.norm(residual))
    residual_ridge = float(np.linalg.norm(A @ ridge_update - residual))
    residual_qsvt = float(np.linalg.norm(A @ qsvt_update - residual))
    ridge_norm = float(np.linalg.norm(ridge_update))
    qsvt_norm = float(np.linalg.norm(qsvt_update))
    delta = qsvt_update - ridge_update
    absolute = float(np.linalg.norm(delta))
    cosine = float(np.dot(qsvt_update, ridge_update) / max(qsvt_norm * ridge_norm, SMALL_TOL))
    success_probability = _residual_adjoint_success_probability(
        operator_matrix=operator_matrix,
        residual=residual,
        encoded_dimension=transformed.shape[0],
    )
    return {
        "residual_no_update": residual_no_update,
        "residual_ridge": residual_ridge,
        "residual_qsvt_circuit": residual_qsvt,
        "ridge_residual_ratio": residual_ridge / max(residual_no_update, SMALL_TOL),
        "qsvt_circuit_residual_ratio": residual_qsvt / max(residual_no_update, SMALL_TOL),
        "residual_gap": abs(
            residual_qsvt / max(residual_no_update, SMALL_TOL)
            - residual_ridge / max(residual_no_update, SMALL_TOL)
        ),
        "relative_update_error": absolute / max(ridge_norm, SMALL_TOL),
        "absolute_update_error": absolute,
        "max_component_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
        "success_probability_residual_state": success_probability,
    }


def _transform_errors(
    transformed: np.ndarray,
    polynomial_block: np.ndarray,
    A_bar: np.ndarray,
    polynomial: Polynomial,
) -> dict[str, float]:
    delta = transformed - polynomial_block
    U, singular_values, Vt = np.linalg.svd(_as_real_matrix(A_bar, name="A_bar"), full_matrices=True)
    response_matrix = U.T @ transformed @ Vt.T
    response = np.diag(response_matrix)
    expected = polynomial(singular_values)
    response_error = float(np.max(np.abs(response - expected))) if expected.size else 0.0
    fro = float(np.linalg.norm(delta, ord="fro"))
    spectral = float(np.linalg.norm(delta, ord=2))
    return {
        "transform_block_error_fro": fro,
        "transform_block_error_spectral": spectral,
        "max_singular_response_error": response_error,
        "circuit_vs_polynomial_fro_error": fro,
        "circuit_vs_polynomial_spectral_error": spectral,
    }


def _statevector_probe_rows(
    *,
    run_type: str,
    case_name: str,
    circuit: Any,
    operator_matrix: np.ndarray,
    transformed: np.ndarray,
    expected: np.ndarray,
    residual: np.ndarray | None,
) -> list[dict[str, Any]]:
    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    dimension = transformed.shape[0]
    rows: list[dict[str, Any]] = []
    for index in range(dimension):
        psi = np.zeros(dimension, dtype=np.complex128)
        psi[index] = 1.0
        full = np.zeros(operator_matrix.shape[0], dtype=np.complex128)
        full[:dimension] = psi
        evolved = np.asarray(Statevector(full).evolve(circuit).data, dtype=np.complex128)
        complex_post = evolved[:dimension]
        post = np.real(complex_post)
        target = expected @ np.real(psi)
        error = float(np.linalg.norm(post - target))
        rows.append(
            {
                "run_type": run_type,
                "case_name": case_name,
                "state_type": "basis_forward",
                "state_index": index,
                "action_abs_error": error,
                "action_rel_error": error / max(float(np.linalg.norm(target)), SMALL_TOL),
                "success_probability": float(np.linalg.norm(complex_post) ** 2),
            }
        )
    if residual is not None:
        normalized = np.asarray(residual, dtype=np.float64)
        normalized = normalized / max(float(np.linalg.norm(normalized)), SMALL_TOL)
        full = np.zeros(operator_matrix.shape[0], dtype=np.complex128)
        full[: normalized.size] = normalized
        evolved = operator_matrix.conj().T @ full
        complex_post = evolved[: transformed.shape[1]]
        post = np.real(complex_post)
        target = transformed.T @ normalized
        error = float(np.linalg.norm(post - target))
        rows.append(
            {
                "run_type": run_type,
                "case_name": case_name,
                "state_type": "residual_adjoint",
                "state_index": 0,
                "action_abs_error": error,
                "action_rel_error": error / max(float(np.linalg.norm(target)), SMALL_TOL),
                "success_probability": float(np.linalg.norm(complex_post) ** 2),
            }
        )
    return rows


def _residual_adjoint_success_probability(
    *,
    operator_matrix: np.ndarray,
    residual: np.ndarray,
    encoded_dimension: int,
) -> float:
    normalized = np.asarray(residual, dtype=np.float64)
    normalized = normalized / max(float(np.linalg.norm(normalized)), SMALL_TOL)
    full = np.zeros(operator_matrix.shape[0], dtype=np.complex128)
    full[: normalized.size] = normalized
    evolved = operator_matrix.conj().T @ full
    return float(np.linalg.norm(evolved[:encoded_dimension]) ** 2)


def _transpile_qsvt_circuit(
    circuit: Any,
    *,
    basis_gates: list[str],
    optimization_level: int,
    qubit_limit: int,
) -> dict[str, Any]:
    if int(circuit.num_qubits) > int(qubit_limit):
        return {
            "transpilation_status": "skipped_by_budget",
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
            "failure_or_skip_reason": (
                f"transpilation skipped: num_qubits={circuit.num_qubits} exceeds "
                f"transpile_qubit_limit={qubit_limit}"
            ),
        }
    try:
        from qiskit import transpile  # type: ignore[import-not-found]

        start = time.perf_counter()
        transpiled = transpile(
            circuit,
            basis_gates=list(basis_gates),
            optimization_level=int(optimization_level),
        )
        _ = time.perf_counter() - start
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        return {
            "transpilation_status": "completed",
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_cx_count": int(counts.get("cx", 0)),
            "transpiled_total_ops": int(sum(counts.values())),
            "failure_or_skip_reason": "",
        }
    except Exception as exc:  # pragma: no cover - backend-version dependent
        return {
            "transpilation_status": "failed",
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
            "failure_or_skip_reason": f"transpilation failed: {type(exc).__name__}: {exc}",
        }


def _load_block_artifacts(config: dict[str, Any], A: np.ndarray) -> dict[str, Any]:
    spec = dict(config["subproblem_spec"])
    label = _safe_label(
        str(spec.get("case_name", "unknown")),
        int(spec.get("subproblem_size", min(A.shape))),
        str(spec.get("selection_mode", "high_leverage")),
    )
    matrices_dir = Path(config["block_matrices_dir"])
    A_path = matrices_dir / f"{label}_A.npy"
    A_bar_path = matrices_dir / f"{label}_A_bar_padded.npy"
    U_path = matrices_dir / f"{label}_U_A.npy"
    block_csv = Path(config["block_results_path"])
    gamma = float(np.linalg.svd(A, compute_uv=False)[0])
    if block_csv.exists():
        frame = pd.read_csv(block_csv)
        match = frame[
            (frame["case_name"] == str(spec.get("case_name")))
            & (frame["subproblem_size"].astype(int) == int(spec.get("subproblem_size")))
            & (frame["selection_criterion"] == str(spec.get("selection_mode", "high_leverage")))
        ]
        if not match.empty:
            gamma = float(match.iloc[0]["gamma"])
    A_saved = np.asarray(np.load(A_path), dtype=np.float64) if A_path.exists() else None
    if A_bar_path.exists() and U_path.exists():
        return {
            "A": A_saved,
            "A_bar_padded": np.asarray(np.load(A_bar_path), dtype=np.complex128),
            "U_A": np.asarray(np.load(U_path), dtype=np.complex128),
            "gamma": gamma,
        }
    encoding = construct_padded_block_encoding(A, gamma=gamma)
    return {
        "A": A_saved,
        "A_bar_padded": encoding.A_bar_padded,
        "U_A": encoding.U,
        "gamma": encoding.gamma,
    }


def _select_ieee_degree(config: dict[str, Any]) -> int:
    if config.get("degree") is not None:
        return int(config["degree"])
    path = Path(config["end_to_end_results_path"])
    spec = dict(config["subproblem_spec"])
    if path.exists():
        frame = pd.read_csv(path)
        matches = frame[
            (frame["case_name"] == str(spec.get("case_name")))
            & (frame["subproblem_size"].astype(int) == int(spec.get("subproblem_size")))
            & (frame["selection_criterion"] == str(spec.get("selection_mode", "high_leverage")))
            & np.isclose(frame["alpha"].astype(float), float(config["alpha"]))
            & np.isclose(frame["epsilon_target"].astype(float), float(config["epsilon_target"]))
            & (frame["target_met"].astype(bool))
        ]
        if not matches.empty:
            return int(matches.sort_values("degree").iloc[0]["degree"])
    return 11


def _pad_coefficients(coefficients: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.size < int(degree) + 1:
        values = np.pad(values, (0, int(degree) + 1 - values.size))
    values = values[: int(degree) + 1].copy()
    values[0::2] = 0.0
    values[np.abs(values) < 1.0e-14] = 0.0
    return values


def _call_counts(phase_count: int) -> dict[str, int]:
    u_calls = 0
    u_dagger_calls = 0
    for _index in range(1, int(phase_count) - 1, 2):
        u_calls += 1
        u_dagger_calls += 1
    if int(phase_count) % 2 == 0:
        u_calls += 1
    return {"num_U_calls": u_calls, "num_U_dagger_calls": u_dagger_calls}


def _summary_frame(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_type",
        "case_name",
        "subproblem_size",
        "alpha",
        "epsilon_target",
        "degree",
        "phase_count",
        "qsvt_sequence_status",
        "simulation_status",
        "transform_block_error_fro",
        "circuit_vs_polynomial_fro_error",
        "relative_update_error",
        "residual_gap",
        "success_probability_residual_state",
        "num_U_calls",
        "num_U_dagger_calls",
        "raw_circuit_depth",
        "transpilation_status",
        "transpiled_depth",
        "transpiled_cx_count",
        "failure_or_skip_reason",
    ]
    return results[[column for column in columns if column in results.columns]].copy()


def _plot_update_error(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    completed = results[results["relative_update_error"].notna()]
    if completed.empty:
        ax.text(0.5, 0.5, "No update comparison rows", ha="center", va="center")
    else:
        labels = _labels(completed)
        values = _positive_for_log(completed["relative_update_error"])
        ax.bar(np.arange(len(completed)), values)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(completed)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("relative update error")
        ax.set_title("Integrated QSVT Circuit vs Matched Ridge Update")
        ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_transform_error(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    completed = results[results["transform_block_error_fro"].notna()]
    if completed.empty:
        ax.text(0.5, 0.5, "No transform rows", ha="center", va="center")
    else:
        labels = _labels(completed)
        x = np.arange(len(completed))
        width = 0.35
        ax.bar(
            x - width / 2,
            _positive_for_log(completed["transform_block_error_fro"]),
            width,
            label="Frobenius",
        )
        ax.bar(
            x + width / 2,
            _positive_for_log(completed["max_singular_response_error"]),
            width,
            label="singular response",
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("error")
        ax.set_title("Integrated QSVT Circuit Transform Errors")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_resource_counts(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    completed = results[results["simulation_status"].astype(str).str.contains("completed")]
    if completed.empty:
        ax.text(0.5, 0.5, "No completed QSVT circuits", ha="center", va="center")
    else:
        labels = _labels(completed)
        x = np.arange(len(completed))
        width = 0.25
        ax.bar(x - width, completed["num_U_calls"].astype(float), width, label="U calls")
        ax.bar(x, completed["raw_circuit_depth"].astype(float), width, label="raw depth")
        transpiled = pd.to_numeric(completed["transpiled_cx_count"], errors="coerce").fillna(0.0)
        ax.bar(x + width, transpiled, width, label="CX count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("count")
        ax.set_title("Integrated QSVT Circuit Resource Diagnostics")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _report_markdown(
    *,
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    results_csv: Path,
    sanity_csv: Path,
    probes_csv: Path,
    summary_csv: Path,
) -> str:
    status_counts = _status_counts(results)
    completed = results[results["simulation_status"].astype(str).str.contains("completed")]
    sanity = results[results["run_type"] == "sanity_check"]
    ieee = results[results["run_type"] == "ieee_selected_block"]
    sanity_counts = (
        sanity["qsvt_sequence_status"].value_counts().to_dict() if not sanity.empty else {}
    )
    ieee_counts = ieee["qsvt_sequence_status"].value_counts().to_dict() if not ieee.empty else {}
    metric_lines = _metric_lines(completed)
    return "\n".join(
        [
            "# Integrated Small QSVT Circuit Sequence Report",
            "",
            "## Goal",
            "",
            "This experiment constructs an integrated small QSVT circuit sequence using "
            "a dense block-encoding unitary, synthesized QSVT phases, and "
            "statevector/operator simulation.",
            "",
            "## QSVT/QSP Convention Used",
            "",
            f"- {DEFAULT_PHASE_CONVENTION}",
            '- Phases are synthesized with PennyLane `poly_to_angles(coefficients, "QSVT")`.',
            "- The sequence uses projector-controlled phase rotations around repeated "
            "dense block-encoding calls and their adjoints.",
            "- The real part of the extracted top-left block is compared with the "
            "odd polynomial singular-value transform.",
            "",
            "## Sanity-Check Result",
            "",
            f"- Status rows: {sanity_counts}",
            "",
            "## IEEE-Derived Block Result",
            "",
            f"- Status rows: {ieee_counts}",
            f"- Selected subproblem: {config['subproblem_spec']}",
            f"- Alpha: {config['alpha']}; epsilon target: {config['epsilon_target']}",
            "",
            "## Phase Synthesis Settings",
            "",
            "- Backend: PennyLane `poly_to_angles`.",
            f"- Angle solver: {config['angle_solver']}",
            "- Coefficients: monomial basis, low-to-high order, converted from the "
            "same odd Chebyshev polynomial convention used by the end-to-end experiment.",
            "",
            "## Sequence Construction",
            "",
            "- A dense selected-subproblem block encoding U_A is loaded from the explicit "
            "block-encoding demo when available.",
            "- The structured QSVT circuit alternates U_A and U_A dagger according to "
            "the repository convention tested against PennyLane `qml.QSVT`.",
            "",
            "## Verification Method",
            "",
            "- Sanity check: diagonal test matrix and known polynomial response.",
            "- IEEE check: real circuit transform block versus matrix-level polynomial "
            "transform; adjoint transform applied to the weighted residual and rescaled "
            "by C_alpha for comparison with matched Ridge/Tikhonov.",
            "",
            "## Resource Counts and Numerical Results",
            "",
            *metric_lines,
            "",
            "## Success, Failure, and Skipped Counts",
            "",
            f"- Status counts: {status_counts}",
            "",
            "## Claim-Safe Interpretation",
            "",
            "This experiment constructs an integrated small QSVT circuit sequence for a "
            "selected IEEE-derived weighted-Jacobian block and verifies its "
            "statevector-level action against the synthesized polynomial and matched "
            "Ridge/Tikhonov target.",
            "",
            "The result is a selected-subproblem circuit-level consistency check. It "
            "does not imply QSVT numerical superiority over Ridge/Tikhonov. The dense "
            "block encoding and QSVT sequence are proof-of-concept simulator "
            "constructions and are not claimed to be scalable sparse-oracle "
            "implementations.",
            "",
            "## Limitations",
            "",
            "- Only a small selected block is used by default.",
            "- Dense unitary gates are loaded directly; no sparse oracle is constructed.",
            "- Full IEEE-scale execution, hardware deployment, and full-vector readout "
            "remain outside scope.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "The integrated small-QSVT circuit sequence reproduces the synthesized "
            "polynomial transform and the matched Ridge/Tikhonov update on a selected "
            "IEEE-derived weighted-Jacobian block within controlled statevector "
            "simulation error. This supports the implementation-pathway claim only; "
            "it is not evidence of quantum speedup or QSVT-over-Ridge superiority.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{results_csv}`",
            f"- Sanity CSV: `{sanity_csv}`",
            f"- Probe details CSV: `{probes_csv}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _metric_lines(completed: pd.DataFrame) -> list[str]:
    if completed.empty:
        return ["- No completed integrated QSVT circuit rows."]
    lines = [
        "- Transform block Frobenius error range: "
        f"{completed['transform_block_error_fro'].min():.3e} to "
        f"{completed['transform_block_error_fro'].max():.3e}.",
        "- Max singular response error range: "
        f"{completed['max_singular_response_error'].min():.3e} to "
        f"{completed['max_singular_response_error'].max():.3e}.",
        "- U-call range: "
        f"{int(completed['num_U_calls'].min())} to {int(completed['num_U_calls'].max())}.",
        "- Raw circuit depth range: "
        f"{int(completed['raw_circuit_depth'].min())} to "
        f"{int(completed['raw_circuit_depth'].max())}.",
    ]
    relative = pd.to_numeric(completed["relative_update_error"], errors="coerce").dropna()
    if not relative.empty:
        lines.append(
            "- Circuit-vs-Ridge relative update error range: "
            f"{relative.min():.3e} to {relative.max():.3e}."
        )
    residual_gap = pd.to_numeric(completed["residual_gap"], errors="coerce").dropna()
    if not residual_gap.empty:
        lines.append(
            f"- Residual-ratio gap range: {residual_gap.min():.3e} to {residual_gap.max():.3e}."
        )
    return lines


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 123,
        "run_ieee": True,
        "subproblem_spec": {
            "case_name": "ieee14",
            "subproblem_size": 4,
            "selection_mode": "high_leverage",
        },
        "alpha": 1.0e-2,
        "epsilon_target": 1.0e-3,
        "degree": None,
        "end_to_end_results_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_results.csv"
        ),
        "block_results_path": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "block_matrices_dir": str(root / "explicit_block_encoding_demo" / "matrices"),
        "angle_solver": "root-finding",
        "basis_gates": DEFAULT_BASIS_GATES,
        "transpile_qubit_limit": 4,
        "transpile_optimization_level": 1,
        "sanity_singular_values": [0.2, 0.5],
        "sanity_polynomial_coefficients": [0.0, 0.5],
        "sanity_tolerance": 1.0e-9,
        "force_sanity_convention_failure": False,
        "artifact_match_rtol": 1.0e-9,
        "artifact_match_atol": 1.0e-8,
    }
    if config:
        resolved.update(config)
    resolved["subproblem_spec"] = dict(resolved["subproblem_spec"])
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(resolved["epsilon_target"]) <= 0.0:
        raise ValueError("epsilon_target must be positive")
    return resolved


def _sanity_failure(
    *,
    config: dict[str, Any],
    matrix: np.ndarray,
    degree: int,
    phase_result: PhaseSynthesisResult,
    reason: str,
) -> IntegratedEvaluation:
    row = _empty_row("sanity_check", "diagonal_sanity", matrix.shape[0])
    row.update(
        {
            "alpha": np.nan,
            "epsilon_target": np.nan,
            "degree": int(degree),
            "phase_count": int(phase_result.phases.size),
            "phase_synthesis_status": phase_result.status,
            "phase_convention": phase_result.convention,
            "qsvt_sequence_status": "failed_convention_mismatch",
            "gamma": 1.0,
            "C_alpha": 1.0,
            "condition_number": float(np.linalg.cond(matrix)),
            "sigma_min": float(np.min(np.diag(matrix))),
            "sigma_max": float(np.max(np.diag(matrix))),
            "simulation_status": "failed_convention_mismatch",
            "failure_or_skip_reason": reason,
        }
    )
    if bool(config["force_sanity_convention_failure"]):
        row["failure_or_skip_reason"] = "forced sanity convention mismatch for testing"
    return IntegratedEvaluation(row=row, probes=[], transformed_block=None, polynomial_block=None)


def _ieee_phase_failure_row(
    *,
    config: dict[str, Any],
    A: np.ndarray,
    gamma: float,
    C_alpha: float,
    degree: int,
    phase_result: PhaseSynthesisResult,
) -> IntegratedEvaluation:
    row = _empty_row(
        "ieee_selected_block",
        str(config["subproblem_spec"].get("case_name", "unknown")),
        int(config["subproblem_spec"].get("subproblem_size", min(A.shape))),
    )
    singular_values = np.linalg.svd(A, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(degree),
            "phase_count": int(phase_result.phases.size),
            "phase_synthesis_status": phase_result.status,
            "phase_convention": phase_result.convention,
            "qsvt_sequence_status": "skipped_phase_synthesis_failed",
            "gamma": float(gamma),
            "C_alpha": float(C_alpha),
            "condition_number": float(np.max(positive) / np.min(positive)),
            "sigma_min": float(np.min(positive)),
            "sigma_max": float(np.max(positive)),
            "simulation_status": "skipped_phase_synthesis_failed",
            "failure_or_skip_reason": phase_result.failure_reason,
        }
    )
    return IntegratedEvaluation(row=row, probes=[], transformed_block=None, polynomial_block=None)


def _ieee_exception_row(config: dict[str, Any], exc: Exception) -> IntegratedEvaluation:
    spec = dict(config["subproblem_spec"])
    row = _empty_row(
        "ieee_selected_block",
        str(spec.get("case_name", "unknown")),
        int(spec.get("subproblem_size", 0)),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "phase_synthesis_status": "not_completed",
            "qsvt_sequence_status": "failed",
            "simulation_status": "failed",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
        }
    )
    return IntegratedEvaluation(row=row, probes=[], transformed_block=None, polynomial_block=None)


def _skipped_ieee_row(config: dict[str, Any], reason: str) -> dict[str, Any]:
    spec = dict(config["subproblem_spec"])
    row = _empty_row(
        "ieee_selected_block",
        str(spec.get("case_name", "unknown")),
        int(spec.get("subproblem_size", 0)),
    )
    row.update(
        {
            "alpha": float(config["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": config.get("degree") if config.get("degree") is not None else np.nan,
            "phase_synthesis_status": "not_attempted",
            "qsvt_sequence_status": "skipped_with_convention_mismatch",
            "simulation_status": "skipped_with_convention_mismatch",
            "failure_or_skip_reason": reason,
        }
    )
    return row


def _empty_row(run_type: str, case_name: str, subproblem_size: int) -> dict[str, Any]:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "run_type": run_type,
            "case_name": case_name,
            "subproblem_size": int(subproblem_size),
            "phase_count": 0,
            "phase_synthesis_status": "not_attempted",
            "phase_convention": DEFAULT_PHASE_CONVENTION,
            "qsvt_sequence_status": "not_attempted",
            "num_U_calls": 0,
            "num_U_dagger_calls": 0,
            "num_phase_rotations": 0,
            "transpilation_status": "not_attempted",
            "simulation_status": "not_attempted",
            "failure_or_skip_reason": "",
        }
    )
    return row


def _status_counts(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "qsvt_sequence_status": _value_counts(results, "qsvt_sequence_status"),
        "phase_synthesis_status": _value_counts(results, "phase_synthesis_status"),
        "simulation_status": _value_counts(results, "simulation_status"),
        "transpilation_status": _value_counts(results, "transpilation_status"),
    }


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def _labels(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row.run_type}-{row.case_name}-{int(row.subproblem_size)}"
        for row in frame.itertuples(index=False)
    ]


def _positive_for_log(values: pd.Series) -> np.ndarray:
    return np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 1.0e-18)


def _as_real_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if np.iscomplexobj(matrix):
        imag_max = float(np.max(np.abs(np.imag(matrix)))) if matrix.size else 0.0
        if imag_max > 1.0e-12:
            raise ValueError(f"{name} has non-negligible imaginary entries: {imag_max}")
        matrix = np.real(matrix)
    return np.asarray(matrix, dtype=np.float64)


def _safe_label(case_name: str, size: int, selection: str) -> str:
    raw = f"{case_name}_{size}x{size}_{selection}"
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_" for character in raw
    )


def _input_matrix_paths(config: dict[str, Any]) -> dict[str, str]:
    spec = dict(config["subproblem_spec"])
    label = _safe_label(
        str(spec.get("case_name", "unknown")),
        int(spec.get("subproblem_size", 0)),
        str(spec.get("selection_mode", "high_leverage")),
    )
    matrices_dir = Path(config["block_matrices_dir"])
    return {
        "block_results_path": str(config["block_results_path"]),
        "end_to_end_results_path": str(config["end_to_end_results_path"]),
        "A": str(matrices_dir / f"{label}_A.npy"),
        "A_bar_padded": str(matrices_dir / f"{label}_A_bar_padded.npy"),
        "U_A": str(matrices_dir / f"{label}_U_A.npy"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TQE integrated small QSVT circuit sequence")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--epsilon-target", type=float, default=1.0e-3)
    parser.add_argument("--degree", type=int, default=None)
    args = parser.parse_args(argv)
    run = run_integrated_small_qsvt_circuit(
        {
            "output_root": args.output_root,
            "alpha": args.alpha,
            "epsilon_target": args.epsilon_target,
            "degree": args.degree,
            "end_to_end_results_path": str(
                Path(args.output_root)
                / "end_to_end_qsvt_vs_ridge"
                / "end_to_end_qsvt_vs_ridge_results.csv"
            ),
            "block_results_path": str(
                Path(args.output_root)
                / "explicit_block_encoding_demo"
                / "block_encoding_demo_results.csv"
            ),
            "block_matrices_dir": str(
                Path(args.output_root) / "explicit_block_encoding_demo" / "matrices"
            ),
        }
    )
    print(f"TQE integrated small QSVT circuit sequence complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
