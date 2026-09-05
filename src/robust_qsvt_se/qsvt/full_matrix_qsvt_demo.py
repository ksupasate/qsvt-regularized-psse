from __future__ import annotations

import argparse
import os
import signal
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.qsvt.polynomial import (
    OddPolynomialApproximation,
    fit_odd_regularized_polynomial,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

SCOPE_STATEMENT = (
    "This experiment demonstrates a small explicit block-encoded matrix-level QSVT "
    "simulation for a normalized weighted state-estimation matrix or submatrix. It "
    "does not demonstrate full IEEE-scale quantum execution or quantum speedup."
)


@dataclass(frozen=True, slots=True)
class SourceMatrix:
    B: np.ndarray
    residual: np.ndarray
    selected_rows: np.ndarray
    selected_columns: np.ndarray
    row_labels: list[str]
    column_labels: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PhaseSequence:
    phases: np.ndarray
    coefficients: np.ndarray
    approximation: OddPolynomialApproximation
    requested_degree: int
    actual_degree: int
    candidate_errors: list[dict[str, Any]]
    selection_reason: str
    qsvt_polynomial_validation: dict[str, Any]


def run_full_matrix_qsvt_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    source = _build_source_matrix(resolved)
    singular_values_B = np.linalg.svd(source.B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = source.B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    alpha = float(resolved["alpha"])
    alpha_norm = alpha / beta**2

    block_encoding = canonical_square_block_encoding(A, tolerance=float(resolved["tolerance"]))
    block_report = validate_block_encoding(
        block_encoding,
        beta=beta,
        tolerance=float(resolved["validation_tolerance"]),
    )

    phase_sequence = _build_phase_sequence(
        singular_values_A=singular_values_A,
        requested_degree=int(resolved["degree"]),
        alpha_norm=alpha_norm,
        grid_size=int(resolved["grid_size"]),
        max_synthesis_degree=int(resolved["max_synthesis_degree"]),
        angle_solver=str(resolved["angle_solver"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
    )

    qsvt = _apply_explicit_block_encoded_qsvt(
        A=A,
        block_unitary=block_encoding.unitary,
        phases=phase_sequence.phases,
    )
    comparison, matrix_metrics = _matrix_comparison_frame(
        B=source.B,
        A=A,
        beta=beta,
        alpha=alpha,
        C=phase_sequence.approximation.scale_factor,
        polynomial_coefficients=phase_sequence.coefficients,
        qsvt_bounded_block=qsvt["transformed_block"],
    )
    state_comparison, state_metrics = _state_comparison_frame(
        source=source,
        ridge_operator=np.asarray(matrix_metrics["ridge_reference_operator"]),
        polynomial_operator=np.asarray(matrix_metrics["polynomial_rescaled_operator"]),
        qsvt_operator=np.asarray(matrix_metrics["qsvt_rescaled_operator"]),
    )
    singular_frame = _singular_values_frame(
        singular_values_B=singular_values_B,
        singular_values_A=singular_values_A,
        alpha=alpha,
        alpha_norm=alpha_norm,
        C=phase_sequence.approximation.scale_factor,
        polynomial_coefficients=phase_sequence.coefficients,
    )

    matrix_metadata = _matrix_metadata(
        resolved=resolved,
        source=source,
        B=source.B,
        A=A,
        beta=beta,
        alpha_norm=alpha_norm,
        singular_values_B=singular_values_B,
        singular_values_A=singular_values_A,
    )
    summary = _summary_dict(
        resolved=resolved,
        source=source,
        beta=beta,
        singular_values_B=singular_values_B,
        singular_values_A=singular_values_A,
        alpha_norm=alpha_norm,
        block_report=block_report,
        phase_sequence=phase_sequence,
        qsvt=qsvt,
        matrix_metrics=matrix_metrics,
        state_metrics=state_metrics,
    )
    artifacts = _write_artifacts(
        output_dir=output_dir,
        resolved=resolved,
        matrix_metadata=matrix_metadata,
        block_report=block_report,
        phase_sequence=phase_sequence,
        singular_frame=singular_frame,
        comparison=comparison,
        state_comparison=state_comparison,
        summary=summary,
    )
    manifest = write_manifest(
        output_dir,
        artifacts={key: str(value) for key, value in artifacts.items()},
        input_config=resolved,
        claim_boundary=SCOPE_STATEMENT,
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": artifacts,
        "matrix_comparison": comparison,
        "state_comparison": state_comparison,
    }


def normalized_bounded_ridge_target(
    normalized_sigma: np.ndarray,
    *,
    alpha_norm: float,
    C: float,
) -> np.ndarray:
    if alpha_norm <= 0.0:
        raise ValueError("alpha_norm must be positive")
    if C <= 0.0:
        raise ValueError("C must be positive")
    values = np.asarray(normalized_sigma, dtype=np.float64)
    return values / (float(C) * (values**2 + float(alpha_norm)))


def rescale_bounded_target_to_original(
    bounded_values: np.ndarray,
    *,
    beta: float,
    C: float,
) -> np.ndarray:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if C <= 0.0:
        raise ValueError("C must be positive")
    return (float(C) / float(beta)) * np.asarray(bounded_values, dtype=np.float64)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "case": "ieee14",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "max_synthesis_degree": 201,
        "grid_size": 2048,
        "angle_solver": "iterative",
        "phase_timeout_seconds": 25,
        "seed": 123,
        "output_dir": "outputs/full_qsvt_small_matrix_demo",
        "tolerance": 1.0e-8,
        "validation_tolerance": 1.0e-7,
    }
    if config:
        resolved.update(config)
    if "case" in resolved and "case_name" not in (config or {}):
        resolved["case_name"] = resolved["case"]
    size = int(resolved["submatrix_size"])
    if size <= 0:
        raise ValueError("submatrix_size must be positive")
    if not _is_power_of_two(size):
        raise ValueError("submatrix_size must be a power of two for this explicit demo")
    alpha = float(resolved["alpha"])
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    degree = int(resolved["degree"])
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    max_degree = int(resolved["max_synthesis_degree"])
    if max_degree < 1:
        raise ValueError("max_synthesis_degree must be positive")
    if max_degree % 2 == 0:
        max_degree -= 1
    resolved["submatrix_size"] = size
    resolved["alpha"] = alpha
    resolved["degree"] = degree
    resolved["max_synthesis_degree"] = max_degree
    resolved["grid_size"] = max(int(resolved["grid_size"]), degree + 2)
    return resolved


def _build_source_matrix(config: dict[str, Any]) -> SourceMatrix:
    system, matrix_source = build_engineering_system(
        {
            "matrix_source": _engineering_matrix_source(str(config["matrix_source"])),
            "case_name": str(config["case_name"]),
            "case_source": str(config["case_source"]),
            "seed": int(config["seed"]),
            "measurement": config.get("measurement", {}),
            "linearization": config.get("linearization", {}),
        }
    )
    B_full = np.asarray(system.H_tilde, dtype=np.float64).T
    size = int(config["submatrix_size"])
    selected_columns = _top_indices(np.linalg.norm(B_full, axis=0), size)
    row_scores = np.linalg.norm(B_full[:, selected_columns], axis=1)
    selected_rows = _top_indices(row_scores, size)
    B = B_full[np.ix_(selected_rows, selected_columns)]
    residual = np.asarray(system.r_tilde, dtype=np.float64)[selected_columns]
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 0.0:
        residual = residual / residual_norm

    state_labels = _state_labels(system.metadata, B_full.shape[0])
    measurement_labels = list(system.metadata.get("measurement_labels", []))
    if len(measurement_labels) < B_full.shape[1]:
        measurement_labels = [f"measurement_{index}" for index in range(B_full.shape[1])]
    return SourceMatrix(
        B=B,
        residual=residual,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
        row_labels=[state_labels[index] for index in selected_rows],
        column_labels=[measurement_labels[index] for index in selected_columns],
        metadata={
            "case_name": system.metadata.get("case_name", config["case_name"]),
            "case_source": config["case_source"],
            "matrix_source": matrix_source,
            "matrix_orientation": "B = H_tilde.T",
            "full_weighted_jacobian_shape": [
                int(system.H_tilde.shape[0]),
                int(system.H_tilde.shape[1]),
            ],
            "full_transpose_shape": [int(B_full.shape[0]), int(B_full.shape[1])],
            "selection_strategy": (
                "largest column norms of B=H_tilde.T, then largest row norms restricted "
                "to those columns"
            ),
            "selected_rows": selected_rows.astype(int).tolist(),
            "selected_columns": selected_columns.astype(int).tolist(),
            "selected_state_labels": [state_labels[index] for index in selected_rows],
            "selected_measurement_labels": [
                measurement_labels[index] for index in selected_columns
            ],
            "residual_source": (
                "weighted residual r_tilde restricted to selected measurement columns and "
                "normalized to unit 2-norm"
            ),
            "residual_norm_before_normalization": residual_norm,
        },
    )


def _build_phase_sequence(
    *,
    singular_values_A: np.ndarray,
    requested_degree: int,
    alpha_norm: float,
    grid_size: int,
    max_synthesis_degree: int,
    angle_solver: str,
    phase_timeout_seconds: int,
) -> PhaseSequence:
    _configure_mpl_cache()
    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for explicit matrix-level QSVT") from exc

    positive = singular_values_A[singular_values_A > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected matrix must have at least one nonzero singular value")
    domain_min = max(1.0e-6, 0.9 * float(np.min(positive)))
    domain_min = min(domain_min, 0.95)
    candidate_errors: list[dict[str, Any]] = []
    for degree in _candidate_degrees(requested_degree, max_synthesis_degree):
        try:
            approximation = fit_odd_regularized_polynomial(
                alpha=alpha_norm,
                block_encoding_normalization=1.0,
                degree=degree,
                domain_min=domain_min,
                domain_max=1.0,
                grid_size=max(grid_size, degree + 2),
            )
            coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
            coefficients = coefficients / approximation.scale_factor
            validation = validate_qsvt_polynomial(
                coefficients,
                parity="odd",
                grid_size=8193,
                bound_tolerance=1.0e-5,
            )
            with _phase_timeout(phase_timeout_seconds):
                phases = np.asarray(
                    qml.poly_to_angles(coefficients, "QSVT", angle_solver=angle_solver),
                    dtype=np.float64,
                )
            if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
                raise RuntimeError("phase synthesis returned invalid phases")
            if degree == requested_degree:
                reason = "requested degree synthesized successfully"
            elif int(max_synthesis_degree) < int(requested_degree):
                reason = (
                    f"requested degree {requested_degree} was capped at "
                    f"{max_synthesis_degree} by max_synthesis_degree"
                )
            else:
                reason = (
                    f"requested degree {requested_degree} failed synthesis or validation; "
                    f"fallback synthesized degree {degree}"
                )
            return PhaseSequence(
                phases=phases,
                coefficients=coefficients,
                approximation=approximation,
                requested_degree=int(requested_degree),
                actual_degree=int(degree),
                candidate_errors=candidate_errors,
                selection_reason=reason,
                qsvt_polynomial_validation=validation,
            )
        except Exception as exc:
            candidate_errors.append(
                {
                    "degree": int(degree),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    raise RuntimeError(f"no stable QSVT phase sequence found: {candidate_errors}")


def _apply_explicit_block_encoded_qsvt(
    *,
    A: np.ndarray,
    block_unitary: np.ndarray,
    phases: np.ndarray,
) -> dict[str, Any]:
    _configure_mpl_cache()
    import pennylane as qml  # type: ignore[import-not-found]

    dimension = int(A.shape[0])
    n_wires = int(np.log2(block_unitary.shape[0]))
    wires = list(range(n_wires))
    block_operator = qml.QubitUnitary(block_unitary, wires=wires)
    projectors = [qml.PCPhase(float(phase), dim=dimension, wires=wires) for phase in phases]
    operator = qml.QSVT(block_operator, projectors)
    qsvt_unitary = np.asarray(qml.matrix(operator, wire_order=wires), dtype=np.complex128)
    transformed_block = qsvt_unitary[:dimension, :dimension]
    return {
        "qsvt_unitary_dimension": int(qsvt_unitary.shape[0]),
        "qsvt_unitary_error": float(
            np.max(
                np.abs(
                    qsvt_unitary.conj().T @ qsvt_unitary
                    - np.eye(qsvt_unitary.shape[0], dtype=np.complex128)
                )
            )
        ),
        "transformed_block": np.real(transformed_block),
        "transformed_block_max_imag": float(np.max(np.abs(np.imag(transformed_block)))),
        "qsvt_operator": "qml.QSVT(QubitUnitary(U_A), PCPhase(phi_i))",
        "phase_backend": f"pennylane-{qml.__version__}",
    }


def _matrix_comparison_frame(
    *,
    B: np.ndarray,
    A: np.ndarray,
    beta: float,
    alpha: float,
    C: float,
    polynomial_coefficients: np.ndarray,
    qsvt_bounded_block: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    polynomial = Polynomial(polynomial_coefficients)
    polynomial_reference = _singular_value_transform(A, polynomial(singular_values_A))
    ridge_reference = _singular_value_transform(B, ridge_filter(singular_values_B, alpha=alpha))
    qsvt_rescaled = rescale_bounded_target_to_original(qsvt_bounded_block, beta=beta, C=C)
    polynomial_rescaled = rescale_bounded_target_to_original(
        polynomial_reference,
        beta=beta,
        C=C,
    )
    rows = []
    for row in range(B.shape[0]):
        for column in range(B.shape[1]):
            rows.append(
                {
                    "row": row,
                    "column": column,
                    "qsvt_bounded_block": float(qsvt_bounded_block[row, column]),
                    "svd_polynomial_reference": float(polynomial_reference[row, column]),
                    "abs_error_vs_polynomial_svd": abs(
                        float(qsvt_bounded_block[row, column] - polynomial_reference[row, column])
                    ),
                    "qsvt_rescaled_ridge_operator": float(qsvt_rescaled[row, column]),
                    "svd_ridge_reference": float(ridge_reference[row, column]),
                    "svd_polynomial_rescaled_reference": float(polynomial_rescaled[row, column]),
                    "abs_error_vs_ridge_svd": abs(
                        float(qsvt_rescaled[row, column] - ridge_reference[row, column])
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    ridge_norm = float(np.linalg.norm(ridge_reference))
    metrics = {
        "matrix_level_max_abs_error_vs_polynomial_svd": float(
            frame["abs_error_vs_polynomial_svd"].max()
        ),
        "matrix_level_fro_error_vs_polynomial_svd": float(
            np.linalg.norm(qsvt_bounded_block - polynomial_reference)
        ),
        "matrix_level_max_abs_error_vs_ridge_svd": float(frame["abs_error_vs_ridge_svd"].max()),
        "matrix_level_relative_fro_error_vs_ridge_svd": (
            float(np.linalg.norm(qsvt_rescaled - ridge_reference) / ridge_norm)
            if ridge_norm > 0.0
            else float(np.linalg.norm(qsvt_rescaled - ridge_reference))
        ),
        "qsvt_rescaled_operator": qsvt_rescaled,
        "ridge_reference_operator": ridge_reference,
        "polynomial_rescaled_operator": polynomial_rescaled,
    }
    return frame, metrics


def _state_comparison_frame(
    *,
    source: SourceMatrix,
    ridge_operator: np.ndarray,
    polynomial_operator: np.ndarray,
    qsvt_operator: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    residual = np.asarray(source.residual, dtype=np.float64)
    ridge_state = ridge_operator @ residual
    polynomial_state = polynomial_operator @ residual
    qsvt_state = qsvt_operator @ residual
    rows = []
    for index, label in enumerate(source.row_labels):
        rows.append(
            {
                "state_index": int(source.selected_rows[index]),
                "state_label": label,
                "normalized_residual_component": float(residual[index]),
                "qsvt_state_update": float(qsvt_state[index]),
                "ridge_svd_state_update": float(ridge_state[index]),
                "polynomial_svd_state_update": float(polynomial_state[index]),
                "abs_error_vs_ridge": abs(float(qsvt_state[index] - ridge_state[index])),
            }
        )
    ridge_norm = float(np.linalg.norm(ridge_state))
    metrics = {
        "state_solution_max_abs_error_vs_ridge_svd": float(
            np.max(np.abs(qsvt_state - ridge_state))
        ),
        "state_solution_relative_error_vs_ridge_svd": (
            float(np.linalg.norm(qsvt_state - ridge_state) / ridge_norm)
            if ridge_norm > 0.0
            else float(np.linalg.norm(qsvt_state - ridge_state))
        ),
        "state_solution_max_abs_error_vs_polynomial_svd": float(
            np.max(np.abs(qsvt_state - polynomial_state))
        ),
    }
    return pd.DataFrame(rows), metrics


def _singular_values_frame(
    *,
    singular_values_B: np.ndarray,
    singular_values_A: np.ndarray,
    alpha: float,
    alpha_norm: float,
    C: float,
    polynomial_coefficients: np.ndarray,
) -> pd.DataFrame:
    bounded_target = normalized_bounded_ridge_target(
        singular_values_A,
        alpha_norm=alpha_norm,
        C=C,
    )
    scaled_back = rescale_bounded_target_to_original(bounded_target, beta=singular_values_B[0], C=C)
    polynomial_values = Polynomial(polynomial_coefficients)(singular_values_A)
    return pd.DataFrame(
        {
            "singular_index": np.arange(singular_values_B.size),
            "sigma_B": singular_values_B,
            "s_A": singular_values_A,
            "ridge_filter_original": ridge_filter(singular_values_B, alpha=alpha),
            "bounded_normalized_target": bounded_target,
            "polynomial_bounded_value": polynomial_values,
            "polynomial_abs_error_on_singular_value": np.abs(polynomial_values - bounded_target),
            "scaled_back_filter_from_bounded_target": scaled_back,
        }
    )


def _write_artifacts(
    *,
    output_dir: Path,
    resolved: dict[str, Any],
    matrix_metadata: dict[str, Any],
    block_report: dict[str, Any],
    phase_sequence: PhaseSequence,
    singular_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    state_comparison: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Path]:
    block_report_path = output_dir / "block_encoding_report.json"
    metadata_path = output_dir / "matrix_metadata.json"
    singular_path = output_dir / "singular_values.csv"
    matrix_comparison_path = output_dir / "qsvt_matrix_level_comparison.csv"
    state_comparison_path = output_dir / "qsvt_state_solution_comparison.csv"
    phase_path = output_dir / "phase_angles.csv"
    coefficient_path = output_dir / "polynomial_coefficients.csv"
    summary_path = output_dir / "summary.md"
    figure_path = output_dir / "full_qsvt_small_matrix_demo.png"

    write_json(block_report_path, block_report)
    write_json(metadata_path, matrix_metadata)
    singular_frame.to_csv(singular_path, index=False)
    comparison.to_csv(matrix_comparison_path, index=False)
    state_comparison.to_csv(state_comparison_path, index=False)
    pd.DataFrame(
        {
            "phase_index": np.arange(phase_sequence.phases.size),
            "phase_angle": phase_sequence.phases,
        }
    ).to_csv(phase_path, index=False)
    pd.DataFrame(
        {
            "coefficient_index": np.arange(phase_sequence.coefficients.size),
            "scaled_power_coefficient": phase_sequence.coefficients,
        }
    ).to_csv(coefficient_path, index=False)
    summary_path.write_text(_summary_markdown(summary), encoding="utf-8")
    _write_summary_figure(
        figure_path=figure_path,
        singular_frame=singular_frame,
        state_comparison=state_comparison,
        comparison=comparison,
    )
    write_json(output_dir / "resolved_config.json", resolved)
    return {
        "block_encoding_report": block_report_path,
        "matrix_metadata": metadata_path,
        "singular_values": singular_path,
        "qsvt_matrix_level_comparison": matrix_comparison_path,
        "qsvt_state_solution_comparison": state_comparison_path,
        "phase_angles": phase_path,
        "polynomial_coefficients": coefficient_path,
        "summary": summary_path,
        "figure": figure_path,
        "resolved_config": output_dir / "resolved_config.json",
    }


def _matrix_metadata(
    *,
    resolved: dict[str, Any],
    source: SourceMatrix,
    B: np.ndarray,
    A: np.ndarray,
    beta: float,
    alpha_norm: float,
    singular_values_B: np.ndarray,
    singular_values_A: np.ndarray,
) -> dict[str, Any]:
    return {
        "case": resolved["case_name"],
        "matrix_source": source.metadata["matrix_source"],
        "matrix_orientation": source.metadata["matrix_orientation"],
        "matrix_size": [int(B.shape[0]), int(B.shape[1])],
        "full_weighted_jacobian_shape": source.metadata["full_weighted_jacobian_shape"],
        "full_transpose_shape": source.metadata["full_transpose_shape"],
        "selected_rows": source.selected_rows.astype(int).tolist(),
        "selected_columns": source.selected_columns.astype(int).tolist(),
        "selected_state_labels": source.row_labels,
        "selected_measurement_labels": source.column_labels,
        "selection_strategy": source.metadata["selection_strategy"],
        "original_matrix_B": B.tolist(),
        "normalization_factor_beta": float(beta),
        "normalized_matrix_A": A.tolist(),
        "alpha_norm": float(alpha_norm),
        "singular_values_B": singular_values_B.tolist(),
        "singular_values_A": singular_values_A.tolist(),
        "residual_vector_normalized": source.residual.tolist(),
        "residual_norm_before_normalization": source.metadata["residual_norm_before_normalization"],
    }


def _summary_dict(
    *,
    resolved: dict[str, Any],
    source: SourceMatrix,
    beta: float,
    singular_values_B: np.ndarray,
    singular_values_A: np.ndarray,
    alpha_norm: float,
    block_report: dict[str, Any],
    phase_sequence: PhaseSequence,
    qsvt: dict[str, Any],
    matrix_metrics: dict[str, Any],
    state_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "matrix_source": source.metadata["matrix_source"],
        "matrix_orientation": source.metadata["matrix_orientation"],
        "matrix_size": [int(source.B.shape[0]), int(source.B.shape[1])],
        "beta": float(beta),
        "spectral_norm_before_normalization": float(singular_values_B[0]),
        "spectral_norm_after_normalization": float(singular_values_A[0]),
        "block_encoding_unitarity_error": block_report["unitarity_error"],
        "top_left_block_reconstruction_error": block_report["top_left_block_error"],
        "requested_degree": int(resolved["degree"]),
        "constructed_polynomial_degree": int(phase_sequence.coefficients.size - 1),
        "synthesized_phase_degree": int(phase_sequence.actual_degree),
        "effective_qsvt_degree": int(phase_sequence.actual_degree),
        "qsp_qsvt_polynomial_degree": phase_sequence.actual_degree,
        "number_of_phases": int(phase_sequence.phases.size),
        "degree_selection_reason": phase_sequence.selection_reason,
        "target_filter": "g_alpha(s)=s/(C*(s^2+alpha_norm))",
        "alpha": float(resolved["alpha"]),
        "alpha_norm": float(alpha_norm),
        "C": float(phase_sequence.approximation.scale_factor),
        "phase_backend": qsvt["phase_backend"],
        "qsvt_operator": qsvt["qsvt_operator"],
        "qsvt_unitary_error": qsvt["qsvt_unitary_error"],
        "qsvt_transformed_block_max_imag": qsvt["transformed_block_max_imag"],
        "matrix_level_max_abs_error_vs_polynomial_svd": matrix_metrics[
            "matrix_level_max_abs_error_vs_polynomial_svd"
        ],
        "matrix_level_fro_error_vs_polynomial_svd": matrix_metrics[
            "matrix_level_fro_error_vs_polynomial_svd"
        ],
        "matrix_level_max_abs_error_vs_ridge_svd": matrix_metrics[
            "matrix_level_max_abs_error_vs_ridge_svd"
        ],
        "matrix_level_relative_fro_error_vs_ridge_svd": matrix_metrics[
            "matrix_level_relative_fro_error_vs_ridge_svd"
        ],
        "state_solution_max_abs_error_vs_ridge_svd": state_metrics[
            "state_solution_max_abs_error_vs_ridge_svd"
        ],
        "state_solution_relative_error_vs_ridge_svd": state_metrics[
            "state_solution_relative_error_vs_ridge_svd"
        ],
        "scope_statement": SCOPE_STATEMENT,
        "demonstrated": (
            "A small weighted-Jacobian-derived matrix is explicitly normalized, embedded "
            "in a dense unitary block encoding, and transformed by a QSVT phase sequence."
        ),
        "not_demonstrated": (
            "No scalable oracle construction, full IEEE-scale circuit execution, readout "
            "complexity analysis, or quantum speedup is demonstrated."
        ),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full QSVT Small Matrix Demo",
            "",
            f"1. Matrix source: {summary['matrix_source']} ({summary['matrix_orientation']}).",
            f"2. Matrix size: {summary['matrix_size'][0]} x {summary['matrix_size'][1]}.",
            f"3. Normalization factor beta: {summary['beta']:.17g}.",
            "4. Spectral norm before and after normalization: "
            f"{summary['spectral_norm_before_normalization']:.17g}, "
            f"{summary['spectral_norm_after_normalization']:.17g}.",
            f"5. Block-encoding unitarity error: {summary['block_encoding_unitarity_error']:.17g}.",
            "6. Top-left block reconstruction error: "
            f"{summary['top_left_block_reconstruction_error']:.17g}.",
            "7. QSP/QSVT polynomial degree: "
            f"{summary['qsp_qsvt_polynomial_degree']} "
            f"(requested {summary['requested_degree']}).",
            f"8. Number of phases: {summary['number_of_phases']}.",
            "9. Target filter and alpha: "
            f"{summary['target_filter']}, alpha={summary['alpha']:.17g}, "
            f"alpha_norm={summary['alpha_norm']:.17g}, C={summary['C']:.17g}.",
            "10. Matrix-level error against SVD reference: "
            f"{summary['matrix_level_max_abs_error_vs_ridge_svd']:.17g} "
            "max absolute error after Ridge rescaling; "
            f"{summary['matrix_level_max_abs_error_vs_polynomial_svd']:.17g} "
            "max absolute error against the exact polynomial SVD reference.",
            f"11. Demonstrated: {summary['demonstrated']}",
            f"12. Not demonstrated: {summary['not_demonstrated']}",
            "",
            f"Degree selection: {summary['degree_selection_reason']}.",
            "",
            f"> {SCOPE_STATEMENT}",
            "",
        ]
    )


def _write_summary_figure(
    *,
    figure_path: Path,
    singular_frame: pd.DataFrame,
    state_comparison: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    _configure_mpl_cache()
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(
        singular_frame["singular_index"],
        singular_frame["sigma_B"],
        marker="o",
        label="B",
    )
    axes[0].plot(
        singular_frame["singular_index"],
        singular_frame["s_A"],
        marker="s",
        label="A",
    )
    axes[0].set_title("Singular values")
    axes[0].set_xlabel("index")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(
        singular_frame["s_A"],
        singular_frame["bounded_normalized_target"],
        marker="o",
        label="target",
    )
    axes[1].plot(
        singular_frame["s_A"],
        singular_frame["polynomial_bounded_value"],
        marker="s",
        label="polynomial",
    )
    axes[1].set_title("Bounded filter")
    axes[1].set_xlabel("normalized singular value")
    axes[1].legend()

    axes[2].bar(
        np.arange(len(state_comparison)),
        state_comparison["abs_error_vs_ridge"],
    )
    axes[2].set_title("State error")
    axes[2].set_xlabel("selected state")
    axes[2].set_ylabel("abs error")
    matrix_error = float(comparison["abs_error_vs_polynomial_svd"].max())
    fig.suptitle(f"Explicit block-encoded matrix QSVT demo, polynomial error {matrix_error:.2e}")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def _singular_value_transform(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ np.diag(np.asarray(values, dtype=np.float64)) @ vh


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:count])


def _candidate_degrees(requested_degree: int, max_synthesis_degree: int) -> list[int]:
    first = min(int(requested_degree), int(max_synthesis_degree))
    if first % 2 == 0:
        first -= 1
    standard = [35, 31, 25, 21, 15, 11, 7, 5, 3, 1]
    candidates = [first]
    candidates.extend(degree for degree in standard if degree < first)
    return list(dict.fromkeys(degree for degree in candidates if degree >= 1))


def _engineering_matrix_source(matrix_source: str) -> str:
    if matrix_source in {"weighted_jacobian", "ieee14_ac_weighted_jacobian"}:
        return "ieee14_ac_weighted_jacobian"
    if matrix_source == "synthetic":
        return "synthetic"
    raise ValueError("matrix_source must be weighted_jacobian or synthetic")


def _state_labels(metadata: dict[str, Any], count: int) -> list[str]:
    angle_buses = list(metadata.get("angle_state_buses", []))
    voltage_buses = list(metadata.get("voltage_state_buses", []))
    labels = [f"theta_{bus}" for bus in angle_buses] + [f"V_{bus}" for bus in voltage_buses]
    if len(labels) < count:
        labels = [f"state_{index}" for index in range(count)]
    return labels


def _configure_mpl_cache() -> None:
    cache = Path(tempfile.gettempdir()) / "robust_qsvt_se_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("MPLBACKEND", "Agg")


class _PhaseTimeoutError(TimeoutError):
    pass


@contextmanager
def _phase_timeout(seconds: int) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise _PhaseTimeoutError(f"phase synthesis exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small explicit matrix-level QSVT demo")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--output-dir", default="outputs/full_qsvt_small_matrix_demo")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_full_matrix_qsvt_demo(
        {
            "case": args.case,
            "case_name": args.case,
            "case_source": args.case_source,
            "matrix_source": args.matrix_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "output_dir": args.output_dir,
            "seed": args.seed,
        }
    )
    summary = run["summary"]
    print(f"Full matrix-level QSVT demo complete: {run['output_dir']}")
    print(
        "matrix_size="
        f"{summary['matrix_size']} beta={summary['beta']:.6g} "
        "unitarity_error="
        f"{summary['block_encoding_unitarity_error']:.3e} "
        "qsvt_vs_svd_error="
        f"{summary['matrix_level_max_abs_error_vs_polynomial_svd']:.3e}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
