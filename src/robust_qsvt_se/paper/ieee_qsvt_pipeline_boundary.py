"""IEEE-derived quantum-pipeline boundary study for QSVT-compatible PSSE updates.

This module instantiates the quantum workload boundary on IEEE/PYPOWER-derived
weighted Jacobians ``H_tilde = R^{-1/2} H``: deterministic selected-block
extraction, block encoding, residual-state preparation, the bounded Tikhonov
QSVT target versus matched-alpha Ridge, selected-observable readout, and
complexity accounting. Selected small blocks are executable evidence; the full
IEEE matrices enter only through spectrum and cost models. Ridge/Tikhonov stays
the matched classical reference: the QSVT target implements the *same*
regularized spectral filter ``sigma / (sigma^2 + alpha)`` at the *same* alpha.
Nothing here claims speedup, QSVT-over-Ridge superiority, full-vector readout,
field-data validation, or execution of full IEEE matrices on quantum devices.

Status taxonomy used throughout the generated reports:

* ``implemented`` - constructed and validated at small simulator/matrix scale,
* ``simulated``   - exact matrix-level state with a finite-shot sampling model,
* ``modeled``     - a formal cost model / symbolic proxy only,
* ``not_implemented`` - explicitly out of scope for this study.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    array_checksum,
    assert_safe,
    fit_codesigned_bounded_polynomial,
    write_demo_manifest,
)
from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_utils import (
    build_engineering_system,
    estimate_degree_and_queries,
    ridge_svd_solution,
)
from robust_qsvt_se.qsvt.gate_state_preparation import (
    build_initialize_circuit,
    normalize_and_pad_for_gate_preparation,
    validate_initialize_circuit,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.io import ensure_directory, write_json

BOUNDARY_DIR = Path("outputs/ieee_qsvt_pipeline_boundary")

# Matched-alpha convention: the benchmark Ridge default from configs/real_ieee*.yaml
# and the manuscript (alpha = 1e-4). The same alpha is used for every block and is
# never tuned per block.
DEFAULT_ALPHA = 1.0e-4
# Degree convention shared with the selected-observable QSVT demonstration package.
DEFAULT_DEGREE = 31
DEFAULT_SEED = 123
DEFAULT_CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("ieee14", (4, 8, 16)),
    ("ieee30", (4, 8)),
)
SELECTION_POLICY = "largest_row_col_norms"
EQUIVALENCE_RELATIVE_TOLERANCE = 1.0e-10
READOUT_CONFIDENCE_DELTA = 0.05
READOUT_PRECISION_TARGETS = (1.0e-2, 1.0e-3)
SIMULATED_READOUT_SHOTS = 10_000
READOUT_EPSILON_PROXY = 1.0e-2

TOMOGRAPHY_NON_CLAIM = (
    "Full-vector recovery would require one queried functional per coordinate or "
    "state tomography and is not claimed; each readout returns one selected functional."
)

SELECTION_RATIONALE = (
    "Deterministic largest-row/col-norm policy: the columns with the largest Euclidean "
    "norms of the full IEEE-derived weighted Jacobian are selected first, then the rows "
    "with the largest norms restricted to those columns (ties broken by index). The rule "
    "is reproducible from (case, seed) alone, does not depend on the update solution, and "
    "also selects ill-conditioned larger blocks, so easy rows are not favored."
)

SIGMA_WEIGHTING_CONVENTION = (
    "Diagonal measurement covariance R_ii = sigma_i^2 per generated measurement row; "
    "H_tilde = R^(-1/2) H and r_tilde = R^(-1/2) r."
)

MEASUREMENT_PROVENANCE = (
    "Measurement rows are generated from IEEE/PYPOWER network equations "
    "(voltage magnitudes, P/Q injections, P/Q branch flows) at an AC-linearized "
    "operating point; they are not field PMU or SCADA records."
)

WORKLOAD_COLUMNS = [
    "case",
    "reference_config",
    "matrix_source",
    "matrix_shape",
    "measurement_rows",
    "state_dimension",
    "rows_generated_from_ieee_pypower",
    "measurement_row_provenance",
    "linearization_mode",
    "sigma_weighting_convention",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "rank",
    "effective_rank",
    "nnz",
    "max_row_nnz",
    "block_shape",
    "selection_policy",
    "selection_rationale",
    "selected_rows",
    "selected_cols",
    "selected_row_labels",
    "selected_row_types",
    "selected_col_state_types",
    "selected_col_bus_ids",
    "block_sigma_min",
    "block_sigma_max",
    "block_condition_number",
    "block_effective_rank",
    "block_residual_norm",
    "block_checksum",
    "residual_checksum",
]


@dataclass(frozen=True, slots=True)
class SelectedWorkload:
    """One deterministic square block of the IEEE-derived weighted system."""

    case: str
    matrix_source: str
    H_block: np.ndarray = field(repr=False)
    r_block: np.ndarray = field(repr=False)
    rows: np.ndarray = field(repr=False)
    cols: np.ndarray = field(repr=False)
    row_labels: list[str]
    row_types: list[str]
    column_labels: list[dict[str, Any]]

    @property
    def size(self) -> int:
        return int(self.H_block.shape[0])

    @property
    def block_shape(self) -> str:
        return f"{self.size}x{self.size}"


def _matrix_sparsity(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> tuple[int, int]:
    mask = np.abs(np.asarray(matrix, dtype=np.float64)) > tolerance
    return int(mask.sum()), int(mask.sum(axis=1).max()) if mask.size else 0


def _safe_condition(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    smallest = float(values.min())
    return math.inf if smallest <= 0.0 else float(values.max()) / smallest


def _effective_rank(singular_values: np.ndarray, *, rtol: float = 1.0e-10) -> int:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.size == 0:
        return 0
    return int(np.count_nonzero(values > values.max() * rtol))


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _column_state_labels(system_metadata: dict[str, Any], cols: np.ndarray) -> list[dict[str, Any]]:
    try:
        metadata = build_state_metadata_from_system_metadata(system_metadata)
    except Exception:
        metadata = None
    labels: list[dict[str, Any]] = []
    for position, full_index in enumerate(cols):
        descriptor: dict[str, Any] = {
            "block_column": int(position),
            "full_state_index": int(full_index),
            "state_type": "unknown",
            "bus_id": None,
        }
        if metadata is not None and 0 <= int(full_index) < metadata.dimension:
            try:
                record = metadata.record_for_index(int(full_index))
                descriptor["state_type"] = str(record.state_type)
                descriptor["bus_id"] = None if record.bus_id is None else int(record.bus_id)
            except Exception:
                pass
        labels.append(descriptor)
    return labels


def build_case_system(case: str, *, seed: int) -> tuple[Any, str]:
    """Build the IEEE/PYPOWER-derived weighted system H_tilde = R^(-1/2) H."""

    return build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )


def select_workload(system: Any, matrix_source: str, *, case: str, size: int) -> SelectedWorkload:
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    H_block, r_block, rows, cols = select_deterministic_block(
        H_full, r_full, row_count=size, col_count=size, policy=SELECTION_POLICY
    )
    labels = list(system.metadata.get("measurement_labels", []))
    types = list(system.metadata.get("measurement_types", []))
    return SelectedWorkload(
        case=case,
        matrix_source=matrix_source,
        H_block=H_block,
        r_block=r_block,
        rows=rows,
        cols=cols,
        row_labels=[str(labels[i]) if i < len(labels) else "unknown" for i in rows],
        row_types=[str(types[i]) if i < len(types) else "unknown" for i in rows],
        column_labels=_column_state_labels(system.metadata, cols),
    )


def workload_summary_row(
    system: Any, workload: SelectedWorkload, *, reference_config: str
) -> dict[str, Any]:
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    full_sv = np.linalg.svd(H_full, compute_uv=False)
    block_sv = np.linalg.svd(workload.H_block, compute_uv=False)
    nnz, max_row_nnz = _matrix_sparsity(H_full)
    return {
        "case": workload.case,
        "reference_config": reference_config,
        "matrix_source": workload.matrix_source,
        "matrix_shape": f"{H_full.shape[0]}x{H_full.shape[1]}",
        "measurement_rows": int(H_full.shape[0]),
        "state_dimension": int(H_full.shape[1]),
        "rows_generated_from_ieee_pypower": bool(system.metadata.get("external_case", False)),
        "measurement_row_provenance": MEASUREMENT_PROVENANCE,
        "linearization_mode": (
            "AC-linearized one-step update at a perturbed operating point "
            "(mode=ac_power_flow_linearized); not a nonlinear snapshot"
        ),
        "sigma_weighting_convention": SIGMA_WEIGHTING_CONVENTION,
        "sigma_min": float(full_sv.min()),
        "sigma_max": float(full_sv.max()),
        "condition_number": _safe_condition(full_sv),
        "rank": int(np.linalg.matrix_rank(H_full)),
        "effective_rank": _effective_rank(full_sv),
        "nnz": nnz,
        "max_row_nnz": max_row_nnz,
        "block_shape": workload.block_shape,
        "selection_policy": SELECTION_POLICY,
        "selection_rationale": SELECTION_RATIONALE,
        "selected_rows": " ".join(str(int(i)) for i in workload.rows),
        "selected_cols": " ".join(str(int(i)) for i in workload.cols),
        "selected_row_labels": " ".join(workload.row_labels),
        "selected_row_types": " ".join(workload.row_types),
        "selected_col_state_types": " ".join(
            str(label["state_type"]) for label in workload.column_labels
        ),
        "selected_col_bus_ids": " ".join(
            "none" if label["bus_id"] is None else str(label["bus_id"])
            for label in workload.column_labels
        ),
        "block_sigma_min": float(block_sv.min()),
        "block_sigma_max": float(block_sv.max()),
        "block_condition_number": _safe_condition(block_sv),
        "block_effective_rank": _effective_rank(block_sv),
        "block_residual_norm": float(np.linalg.norm(workload.r_block)),
        "block_checksum": array_checksum(workload.H_block),
        "residual_checksum": array_checksum(workload.r_block),
    }


def _transpiled_counts(circuit: Any) -> dict[str, Any]:
    from qiskit import transpile

    transpiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
    ops = {str(key): int(value) for key, value in transpiled.count_ops().items()}
    return {
        "gate_count": int(sum(ops.values())),
        "circuit_depth": int(transpiled.depth()),
        "cx_count": int(ops.get("cx", 0)),
    }


def block_encoding_study(
    workload: SelectedWorkload, *, build_circuits: bool = True
) -> dict[str, Any]:
    """Explicit unitary dilation of A = H_B^T / beta for the selected block.

    The residual-to-update orientation ``A = H_B^T / beta`` matches the manuscript
    convention; the transpose leaves the singular values unchanged.
    """

    H = workload.H_block
    n = workload.size
    beta = float(np.linalg.svd(H, compute_uv=False).max())
    A = H.T / beta
    padded_dimension = _next_power_of_two(n)
    if padded_dimension != n:
        padded = np.zeros((padded_dimension, padded_dimension), dtype=np.float64)
        padded[:n, :n] = A
        A_encoded = padded
    else:
        A_encoded = A
    encoding = canonical_square_block_encoding(A_encoded, tolerance=1.0e-8)
    report = validate_block_encoding(encoding, beta=beta, tolerance=1.0e-7)

    circuit_counts: dict[str, Any] = {"gate_count": None, "circuit_depth": None, "cx_count": None}
    circuit_status = "not_built"
    if build_circuits:
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import UnitaryGate

            qubits = int(np.log2(encoding.unitary.shape[0]))
            circuit = QuantumCircuit(qubits, name="dense_block_encoding")
            circuit.append(UnitaryGate(encoding.unitary), list(range(qubits)))
            circuit_counts = _transpiled_counts(circuit)
            circuit_status = "transpiled_u3_cx"
        except Exception as exc:
            circuit_status = f"unavailable ({type(exc).__name__})"

    return {
        "case": workload.case,
        "path": "selected_block_executable",
        "block_shape": workload.block_shape,
        "orientation": "A = H_B^T / beta (residual-to-update)",
        "normalization_beta": beta,
        "padded_dimension": padded_dimension,
        "padding_added": padded_dimension != n,
        "unitary_dimension": int(encoding.unitary.shape[0]),
        "qubits": int(np.log2(encoding.unitary.shape[0])),
        "top_left_block_error": float(report["top_left_block_error"]),
        "unitarity_error": float(report["unitarity_error"]),
        "validation_passed": bool(report["passed"]),
        "gate_count": circuit_counts["gate_count"],
        "circuit_depth": circuit_counts["circuit_depth"],
        "cx_count": circuit_counts["cx_count"],
        "circuit_status": circuit_status,
        "implementation_status": "implemented",
        "implementation_detail": (
            "Dense matrix-level unitary dilation, validated top-left block and unitarity; "
            "Qiskit-transpiled circuit counts for the small selected block only. Not a "
            "scalable oracle compilation."
        ),
    }


def full_matrix_block_encoding_row(system: Any, *, case: str) -> dict[str, Any]:
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    m, n = H_full.shape
    beta = float(np.linalg.svd(H_full, compute_uv=False).max())
    nnz, max_row_nnz = _matrix_sparsity(H_full)
    padded_dimension = _next_power_of_two(max(m, n))
    lookup_t_count = 7 * (m * max_row_nnz + nnz)
    return {
        "case": case,
        "path": "full_matrix_modeled",
        "block_shape": f"{m}x{n}",
        "orientation": "A = H_tilde^T / beta (residual-to-update)",
        "normalization_beta": beta,
        "padded_dimension": padded_dimension,
        "padding_added": padded_dimension != max(m, n),
        "unitary_dimension": 2 * padded_dimension,
        "qubits": int(np.log2(2 * padded_dimension)),
        "top_left_block_error": None,
        "unitarity_error": None,
        "validation_passed": None,
        "gate_count": lookup_t_count,
        "circuit_depth": None,
        "cx_count": None,
        "circuit_status": "not_built",
        "implementation_status": "modeled",
        "implementation_detail": (
            "Formal cost model only: QROM lookup T-count proxy 7(m*s_r + nnz) for the "
            "sparse index/value oracles; no full-matrix block-encoding circuit is "
            "implemented in this study."
        ),
    }


def state_preparation_study(
    workload: SelectedWorkload,
    *,
    target_dimension: int,
    build_circuits: bool = True,
) -> dict[str, Any]:
    """Amplitude preparation of |r_B> on the padded QSVT system register."""

    preparation = normalize_and_pad_for_gate_preparation(
        workload.r_block, target_dimension=target_dimension
    )
    row: dict[str, Any] = {
        "case": workload.case,
        "path": "selected_block_executable",
        "block_shape": workload.block_shape,
        "residual_source": (
            "IEEE-derived weighted residual r_tilde restricted to the selected rows"
        ),
        "residual_norm": preparation.original_norm,
        "input_dimension": preparation.input_dimension,
        "padded_dimension": preparation.padded_dimension,
        "qubits": preparation.n_qubits,
        "state_preparation_fidelity": None,
        "state_preparation_l2_error": None,
        "gate_count": None,
        "circuit_depth": None,
        "implementation_status": "implemented",
        "implementation_detail": (
            "Dense amplitude loading via Qiskit Initialize, statevector-validated; small "
            "simulator evidence only, not an efficient scalable state-preparation proof."
        ),
    }
    if build_circuits:
        try:
            circuit = build_initialize_circuit(preparation.padded_state)
            validation = validate_initialize_circuit(circuit, preparation.padded_state)
            row["state_preparation_fidelity"] = float(validation["state_preparation_fidelity"])
            row["state_preparation_l2_error"] = float(validation["state_preparation_l2_error"])
            row.update(
                {
                    key: value
                    for key, value in _transpiled_counts(circuit).items()
                    if key in {"gate_count", "circuit_depth"}
                }
            )
        except Exception as exc:
            row["implementation_status"] = "validated_matrix_level_only"
            row["implementation_detail"] = (
                f"Circuit backend unavailable ({type(exc).__name__}); normalization and "
                "padding validated at matrix level only."
            )
    else:
        row["implementation_status"] = "validated_matrix_level_only"
        row["implementation_detail"] = (
            "Circuit construction disabled; normalization and padding validated at matrix level."
        )
    return row


def full_matrix_state_preparation_row(system: Any, *, case: str) -> dict[str, Any]:
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    m = int(r_full.size)
    qubits = math.ceil(math.log2(max(m, 2)))
    return {
        "case": case,
        "path": "full_matrix_modeled",
        "block_shape": f"{m}x{np.asarray(system.H_tilde).shape[1]}",
        "residual_source": "IEEE-derived weighted residual r_tilde (full measurement vector)",
        "residual_norm": float(np.linalg.norm(r_full)),
        "input_dimension": m,
        "padded_dimension": _next_power_of_two(m),
        "qubits": qubits,
        "state_preparation_fidelity": None,
        "state_preparation_l2_error": None,
        "gate_count": 2 ** (qubits + 1),
        "circuit_depth": None,
        "implementation_status": "modeled",
        "implementation_detail": (
            "Complexity model only: generic dense amplitude loading costs O(2^q) rotations "
            "for q qubits; efficient preparation would require exploitable structure or "
            "specialized access and is not implemented here."
        ),
    }


def equivalence_study(
    workload: SelectedWorkload, *, alpha: float, degree: int
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Matched-alpha comparison of the bounded QSVT target and Ridge/Tikhonov.

    The primary reference is the repository Ridge estimator path (the SVD spectral
    filter ``sigma/(sigma^2+alpha)`` used throughout the benchmark); the QSVT-target
    solution applies the exact bounded map ``(1/C) s/(s^2+alpha/beta^2)`` in
    normalized units and inverts the recorded ``C/beta`` rescale, so the check
    validates the normalization-convention round trip at matched alpha. Two
    independent classical solvers (augmented least squares and normal equations)
    are reported as cross-checks; their larger differences on numerically singular
    blocks reflect finite-precision conditioning, not a filter mismatch.
    """

    H = workload.H_block
    r = workload.r_block
    n = workload.size
    U, sigma, Vt = np.linalg.svd(H, full_matrices=False)
    beta = float(sigma.max())
    s = sigma / beta
    alpha_normalized = float(alpha) / beta**2

    ridge_solution = ridge_svd_solution(H, r, alpha=float(alpha))
    augmented = np.vstack([H, math.sqrt(float(alpha)) * np.eye(n)])
    augmented_solution = np.linalg.lstsq(augmented, np.concatenate([r, np.zeros(n)]), rcond=None)[0]
    normal_equations_solution = np.linalg.solve(H.T @ H + float(alpha) * np.eye(n), H.T @ r)

    f_bounded = s / (s**2 + alpha_normalized)
    qsvt_target_solution = Vt.T @ ((f_bounded / beta) * (U.T @ r))

    ridge_norm = max(float(np.linalg.norm(ridge_solution)), 1.0e-300)
    norm_difference = float(np.linalg.norm(qsvt_target_solution - ridge_solution))
    relative_difference = norm_difference / ridge_norm
    augmented_relative_difference = (
        float(np.linalg.norm(qsvt_target_solution - augmented_solution)) / ridge_norm
    )
    normal_equations_relative_difference = (
        float(np.linalg.norm(qsvt_target_solution - normal_equations_solution)) / ridge_norm
    )

    domain_min = float(np.clip(0.9 * s.min(), 1.0e-4, 0.999))
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=float(alpha),
        domain_min=domain_min,
        domain_max=1.0,
        degree=int(degree),
    )
    p_values = target.polynomial(s)
    selected_target_error = float(np.max(np.abs(p_values - f_bounded / target.bound_C)))
    poly_solution = target.physical_recovery_factor * (Vt.T @ (p_values * (U.T @ r)))
    poly_relative_error = float(
        np.linalg.norm(poly_solution - ridge_solution)
        / max(float(np.linalg.norm(ridge_solution)), 1.0e-300)
    )

    filtered_direction = Vt.T @ ((f_bounded / target.bound_C) * (U.T @ (r / np.linalg.norm(r))))
    postselection_success_proxy = float(np.linalg.norm(filtered_direction) ** 2)

    row = {
        "case": workload.case,
        "block_shape": workload.block_shape,
        "alpha": float(alpha),
        "alpha_source": "benchmark Ridge default (configs/real_ieee*.yaml); not tuned per block",
        "beta": beta,
        "bound_C": target.bound_C,
        "singular_value_interval": f"[{sigma.min():.6e}, {sigma.max():.6e}]",
        "normalized_domain": f"[{target.domain_min:.6e}, {target.domain_max:.6e}]",
        "polynomial_degree": int(degree),
        "polynomial_fit_max_abs_error": target.fit_max_abs_error,
        "selected_singular_value_target_error": selected_target_error,
        "polynomial_filtered_relative_error_vs_ridge": poly_relative_error,
        "bounded_max_abs": target.bounded_max_abs,
        "polynomial_admissible": bool(target.bounded_max_abs <= 1.0 + 2.0e-3),
        "polynomial_feasibility_note": (
            "Exact-target equivalence is separate from finite-degree feasibility; "
            "blocks whose degree-d fit is inadmissible or inaccurate at the fixed "
            "benchmark alpha are degree-limited, consistent with the recorded boundary."
        ),
        "ridge_solution_norm": float(np.linalg.norm(ridge_solution)),
        "ridge_reference_method": (
            "repository Ridge estimator path (SVD spectral filter sigma/(sigma^2+alpha))"
        ),
        "qsvt_target_solution_norm": float(np.linalg.norm(qsvt_target_solution)),
        "norm_difference": norm_difference,
        "relative_difference": relative_difference,
        "augmented_lstsq_relative_difference": augmented_relative_difference,
        "normal_equations_relative_difference": normal_equations_relative_difference,
        "cross_check_note": (
            "Augmented-LS and normal-equations cross-checks are independent classical "
            "solvers; their differences grow with finite-precision conditioning on "
            "rank-deficient selected blocks and do not indicate filter mismatch."
        ),
        "equivalence_tolerance": EQUIVALENCE_RELATIVE_TOLERANCE,
        "equivalence_status": (
            "pass" if relative_difference <= EQUIVALENCE_RELATIVE_TOLERANCE else "fail"
        ),
        "postselection_success_proxy": postselection_success_proxy,
        "framing": (
            "Matched-alpha equivalence of the same spectral filter; not a comparison of "
            "estimator quality and not evidence that either implementation is superior."
        ),
    }
    extras = {
        "target": target,
        "postselection_success_proxy": postselection_success_proxy,
        "beta": beta,
    }
    return row, ridge_solution, extras


def _deterministic_observables(workload: SelectedWorkload) -> list[dict[str, Any]]:
    n = workload.size
    labels = workload.column_labels

    def descriptor(position: int) -> str:
        if position >= len(labels):
            return "unknown"
        state_type = labels[position].get("state_type", "unknown")
        bus = labels[position].get("bus_id")
        return f"{state_type}" + ("" if bus is None else f"_bus{bus}")

    candidates: list[dict[str, Any]] = []
    e0 = np.zeros(n)
    e0[0] = 1.0
    candidates.append(
        {
            "observable_id": "state_component_0",
            "observable_type": "single_state_component",
            "physical_meaning": f"e_j^T dx for selected state 0 ({descriptor(0)})",
            "vector": e0,
        }
    )
    angle_positions = [
        i for i, label in enumerate(labels) if "angle" in str(label.get("state_type", ""))
    ]
    voltage_positions = [
        i for i, label in enumerate(labels) if "voltage" in str(label.get("state_type", ""))
    ]
    if angle_positions:
        vec = np.zeros(n)
        vec[angle_positions[0]] = 1.0
        candidates.append(
            {
                "observable_id": "angle_component",
                "observable_type": "angle_state_component",
                "physical_meaning": (
                    f"angle-related correction e_j^T dtheta ({descriptor(angle_positions[0])})"
                ),
                "vector": vec,
            }
        )
    if voltage_positions:
        vec = np.zeros(n)
        vec[voltage_positions[0]] = 1.0
        candidates.append(
            {
                "observable_id": "voltage_component",
                "observable_type": "voltage_magnitude_state_component",
                "physical_meaning": (
                    f"voltage-magnitude correction e_j^T dV ({descriptor(voltage_positions[0])})"
                ),
                "vector": vec,
            }
        )
    if n >= 2:
        pair = angle_positions[:2] if len(angle_positions) >= 2 else [0, 1]
        vec = np.zeros(n)
        vec[pair[0]] = 1.0
        vec[pair[1]] = -1.0
        candidates.append(
            {
                "observable_id": "difference_functional",
                "observable_type": "angle_difference_like",
                "physical_meaning": (
                    f"difference functional ({descriptor(pair[0])} - {descriptor(pair[1])})"
                ),
                "vector": vec,
            }
        )
    half = max(1, (n + 1) // 2)
    aggregate = np.zeros(n)
    aggregate[:half] = 1.0
    candidates.append(
        {
            "observable_id": "aggregate_first_half",
            "observable_type": "aggregate_functional",
            "physical_meaning": f"c^T dx aggregate over the first {half} selected states",
            "vector": aggregate,
        }
    )

    unique: dict[bytes, dict[str, Any]] = {}
    for candidate in candidates:
        key = np.asarray(candidate["vector"], dtype=np.float64).tobytes()
        if key not in unique:
            unique[key] = candidate
    return list(unique.values())


def readout_study(
    workload: SelectedWorkload,
    ridge_solution: np.ndarray,
    *,
    postselection_success_proxy: float,
    physical_recovery_factor: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Selected-observable readout accounting with a simulated finite-shot model.

    The output state used here is the exact matrix-level filtered state (identical
    to Ridge at matched alpha); shot noise is simulated with a binomial model of
    the +/-1 Hadamard-test outcome. This is a sampling simulation, not a compiled
    readout circuit and not full-vector recovery.
    """

    residual_norm = float(np.linalg.norm(workload.r_block))
    dx_norm = float(np.linalg.norm(ridge_solution))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for observable in _deterministic_observables(workload):
        vector = np.asarray(observable["vector"], dtype=np.float64)
        vector_norm = float(np.linalg.norm(vector))
        exact_value = float(vector @ ridge_solution)
        scale = (
            physical_recovery_factor
            * residual_norm
            * math.sqrt(max(postselection_success_proxy, 0.0))
            * vector_norm
        )
        overlap = exact_value / scale if scale > 0 else math.nan
        shot_estimates = {}
        for epsilon in READOUT_PRECISION_TARGETS:
            t = epsilon / scale if scale > 0 else math.nan
            shots = (
                math.ceil(2.0 * math.log(2.0 / READOUT_CONFIDENCE_DELTA) / t**2)
                if np.isfinite(t) and t > 0
                else None
            )
            shot_estimates[epsilon] = shots
        if np.isfinite(overlap) and abs(overlap) <= 1.0:
            success_probability = 0.5 * (1.0 + overlap)
            hits = rng.binomial(SIMULATED_READOUT_SHOTS, success_probability)
            overlap_estimate = 2.0 * hits / SIMULATED_READOUT_SHOTS - 1.0
            simulated_value = scale * overlap_estimate
            simulated_error = abs(simulated_value - exact_value)
        else:
            simulated_value = math.nan
            simulated_error = math.nan
        rows.append(
            {
                "case": workload.case,
                "block_shape": workload.block_shape,
                "observable_id": observable["observable_id"],
                "observable_type": observable["observable_type"],
                "physical_meaning": observable["physical_meaning"],
                "observable_support": " ".join(str(i) for i in np.nonzero(vector)[0]),
                "classical_exact_value": exact_value,
                "update_norm": dx_norm,
                "quantum_interpretation": (
                    "y = (C/beta) * ||r_tilde|| * sqrt(p_success) * ||l|| * <l_hat|psi_out>, "
                    "with <l_hat|psi_out> estimated by a Hadamard-test overlap"
                ),
                "readout_scale_factor": scale,
                "overlap_a": overlap,
                "hoeffding_confidence_delta": READOUT_CONFIDENCE_DELTA,
                "shots_for_abs_precision_1e-2": shot_estimates[1.0e-2],
                "shots_for_abs_precision_1e-3": shot_estimates[1.0e-3],
                "simulated_shots": SIMULATED_READOUT_SHOTS,
                "simulated_value_estimate": simulated_value,
                "simulated_abs_error": simulated_error,
                "readout_status": "simulated",
                "readout_detail": (
                    "Binomial finite-shot model of the +/-1 Hadamard-test outcome on the "
                    "exact matrix-level output state; not a compiled readout circuit."
                ),
                "full_vector_tomography_non_claim": TOMOGRAPHY_NON_CLAIM,
            }
        )
    return rows


def complexity_rows(
    *,
    workload: SelectedWorkload | None,
    system: Any,
    case: str,
    alpha: float,
    degree: int,
    block_encoding_row: dict[str, Any],
    state_preparation_row: dict[str, Any],
    postselection_success_proxy: float | None,
    observable_count: int,
) -> dict[str, Any]:
    """One complexity-accounting row (selected executable or full-matrix modeled)."""

    if workload is not None:
        matrix = workload.H_block
        path = "selected_block_executable"
        degree_d = int(degree)
        degree_note = "co-designed bounded-target degree convention (matches the 4x4 demo)"
    else:
        matrix = np.asarray(system.H_tilde, dtype=np.float64)
        path = "full_matrix_modeled"
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        estimate = estimate_degree_and_queries(singular_values, alpha=float(alpha))
        degree_d = int(estimate["qsvt_degree_estimate"])
        degree_note = (
            f"grid-capped degree estimate (max polynomial error "
            f"{float(estimate['max_polynomial_error']):.3e} at target "
            f"{float(estimate['target_error']):.0e}; recommended_degree="
            f"{estimate['recommended_degree']})"
        )
    m, n = matrix.shape
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    nnz, max_row_nnz = _matrix_sparsity(matrix)
    shots_per_observable = math.ceil(
        2.0 * math.log(2.0 / READOUT_CONFIDENCE_DELTA) / READOUT_EPSILON_PROXY**2
    )
    readout_proxy = observable_count * shots_per_observable
    queries_per_attempt = 2 * degree_d + 1
    p_succ = postselection_success_proxy
    total_query_proxy = (
        readout_proxy * queries_per_attempt / p_succ if p_succ and p_succ > 0 else None
    )
    t_be = block_encoding_row.get("gate_count")
    t_prep = state_preparation_row.get("gate_count")
    return {
        "case": case,
        "path": path,
        "m_measurement_rows": int(m),
        "n_state_dimension": int(n),
        "nnz": nnz,
        "max_row_sparsity_s": max_row_nnz,
        "condition_number_kappa": _safe_condition(singular_values),
        "alpha": float(alpha),
        "qsvt_degree_d": degree_d,
        "degree_note": degree_note,
        "T_BE_proxy": t_be,
        "T_BE_detail": block_encoding_row.get("implementation_detail"),
        "T_prep_proxy": t_prep,
        "T_prep_detail": state_preparation_row.get("implementation_detail"),
        "q_selected_observables": observable_count,
        "readout_epsilon": READOUT_EPSILON_PROXY,
        "shots_per_observable": shots_per_observable,
        "T_readout_proxy": readout_proxy,
        "T_readout_formula": "O(q/epsilon^2) simple sampling; amplitude estimation not modeled",
        "postselection_success_proxy": p_succ,
        "queries_per_attempt": queries_per_attempt,
        "total_query_proxy": total_query_proxy,
        "total_query_formula": "N_total = q * N_shots(epsilon) * (2d+1) / p_success",
        "T_QSVT_formula": "T_QSVT = O(T_prep + d * T_BE + T_readout)",
        "classical_dense_svd_proxy_flops": int(m) * int(n) ** 2,
        "classical_dense_svd_formula": "O(m n^2) for m >= n (dense SVD)",
        "classical_normal_equations_proxy_flops": int(m) * int(n) ** 2 + int(n) ** 3,
        "classical_normal_equations_formula": "O(m n^2 + n^3)",
        "asymptotic_comparison_note": (
            "Boundary accounting only; no asymptotic claim is made because coherent "
            "access, scalable preparation, and matched baselines are not all satisfied."
        ),
    }


def _summary_markdown(
    *,
    alpha: float,
    degree: int,
    workload_rows: list[dict[str, Any]],
    encoding_rows: list[dict[str, Any]],
    preparation_rows: list[dict[str, Any]],
    equivalence_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
) -> str:
    executable_encodings = [r for r in encoding_rows if r["path"] == "selected_block_executable"]
    executable_preps = [r for r in preparation_rows if r["path"] == "selected_block_executable"]
    encoding_note = "."
    if executable_encodings:
        worst_block = max(r["top_left_block_error"] for r in executable_encodings)
        worst_unitarity = max(r["unitarity_error"] for r in executable_encodings)
        encoding_note = (
            f"; worst top-left error {worst_block:.2e}, "
            f"worst unitarity error {worst_unitarity:.2e}."
        )
    fidelities = [
        float(r["state_preparation_fidelity"])
        for r in executable_preps
        if r["state_preparation_fidelity"] is not None
    ]
    preparation_note = (
        f" (worst-case fidelity {min(fidelities):.12f})."
        if fidelities
        else " (matrix-level validation only in this run)."
    )
    worst_relative = max(r["relative_difference"] for r in equivalence_rows)
    equivalence_note = (
        f"; worst relative difference {worst_relative:.2e} "
        f"(tolerance {EQUIVALENCE_RELATIVE_TOLERANCE:.0e})."
    )
    admissible = sum(1 for r in equivalence_rows if r["polynomial_admissible"])
    feasibility_line = (
        f"- Degree-{{degree}} bounded-polynomial feasibility recorded per block: "
        f"{admissible}/{len(equivalence_rows)} admissible. Exact-target equivalence is "
        "separate from finite-degree feasibility; ill-conditioned selected blocks remain "
        "degree-limited at the fixed benchmark alpha, with per-block fit errors reported."
    )
    lines = [
        "# IEEE-Derived Quantum Pipeline Boundary",
        "",
        "IEEE/PYPOWER-derived workload-boundary study for the QSVT-compatible bounded "
        "Tikhonov filter `P_alpha(sigma) = sigma/(sigma^2 + alpha)`. Ridge/Tikhonov is the "
        "matched classical reference at the same alpha; the QSVT target is the same filter. "
        "Selected small blocks extracted deterministically from the generated weighted "
        "Jacobians are the executable evidence; full IEEE matrices enter through spectrum "
        "and cost models only.",
        "",
        f"- Matched alpha = {alpha:g} (benchmark Ridge default; never tuned per block).",
        f"- Bounded-target degree convention d = {degree}.",
        "- Cases and blocks: "
        + "; ".join(
            f"{row['case']} {row['block_shape']} (kappa = {row['block_condition_number']:.3g})"
            for row in workload_rows
        )
        + ".",
        "",
        "## What was implemented",
        "",
        "- Deterministic selected-block extraction from the IEEE-derived weighted Jacobian "
        f"(policy: `{SELECTION_POLICY}`), with row/column measurement and state metadata.",
        "- Dense matrix-level block encoding (explicit unitary dilation) of each selected "
        "block, with top-left reconstruction and unitarity checks" + encoding_note,
        "- Residual amplitude preparation for each selected block via dense Initialize, "
        "statevector-validated" + preparation_note,
        "- Matched-alpha equivalence of the exact bounded QSVT target (normalized units "
        "plus the recorded C/beta rescale) against the repository Ridge spectral filter"
        + equivalence_note
        + " Independent augmented-LS and normal-equations cross-checks are reported with "
        "their conditioning-limited differences.",
        feasibility_line.format(degree=degree),
        "",
        "## What was simulated",
        "",
        "- Selected-observable readout: finite-shot binomial sampling of the +/-1 "
        "Hadamard-test outcome on the exact matrix-level output state, plus Hoeffding "
        "shot budgets for absolute precisions "
        + ", ".join(f"{eps:g}" for eps in READOUT_PRECISION_TARGETS)
        + ".",
        "",
        "## What was modeled",
        "",
        "- Full-matrix block encoding: QROM lookup T-count proxy `7(m*s_r + nnz)`; no "
        "full-matrix circuit is constructed.",
        "- Full-matrix residual preparation: generic dense amplitude-loading rotation "
        "count `O(2^q)`; efficient preparation would require structure or special access.",
        "- Complexity accounting: `T_QSVT = O(T_prep + d*T_BE + T_readout)` with "
        "`O(q/epsilon^2)` simple-sampling readout and classical dense-SVD / "
        "normal-equations baselines `O(m n^2)` and `O(m n^2 + n^3)`.",
        "",
        "## What was not implemented",
        "",
        "- No run on any quantum device; every artifact is classical simulation or a model.",
        "- No full-matrix QSVT circuit, no scalable sparse-access block encoding, and no "
        "efficient residual-state preparation beyond the dense small-block primitives.",
        "- No full-vector readout: " + TOMOGRAPHY_NON_CLAIM,
        "- No phase-factor synthesis in this study; the degree entry reuses the recorded "
        "co-designed convention, and circuit-level phase evidence remains in the "
        "selected-observable demonstration package.",
        "",
        "## Safe manuscript wording",
        "",
        "- The selected executable workload is IEEE-derived (rows and columns extracted "
        "from the generated PSSE weighted Jacobian), not a random toy matrix.",
        "- Full IEEE matrices are used for spectrum and resource accounting, not as "
        "executed QSVT circuits.",
        "- Block encoding and residual-state preparation are implemented only for the "
        "selected small blocks; the full-matrix versions are modeled.",
        "- Readout is selected-observable; matched-alpha QSVT-target and Ridge remain "
        "equivalent as spectral filters.",
        "- Complexity results are boundary accounting, not a performance proof.",
        "",
        "## Claims to avoid",
        "",
        "- No speedup claim of any kind and no claim of advantage.",
        "- No claim that the QSVT pathway improves on Ridge/Tikhonov accuracy.",
        "- No claim of executing full IEEE matrices on quantum devices.",
        "- No claim of field PMU or SCADA data validation.",
        "- No full-vector readout claim.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def build_ieee_qsvt_pipeline_boundary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(BOUNDARY_DIR),
        "alpha": DEFAULT_ALPHA,
        "degree": DEFAULT_DEGREE,
        "seed": DEFAULT_SEED,
        "cases": [[case, list(sizes)] for case, sizes in DEFAULT_CASES],
        "build_circuits": True,
        "readout_seed": 20260702,
        "command": "run_ieee_qsvt_pipeline_boundary",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    alpha = float(resolved["alpha"])
    degree = int(resolved["degree"])
    seed = int(resolved["seed"])
    build_circuits = bool(resolved["build_circuits"])

    workload_rows: list[dict[str, Any]] = []
    encoding_rows: list[dict[str, Any]] = []
    preparation_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    complexity: list[dict[str, Any]] = []
    workload_metadata: list[dict[str, Any]] = []
    array_artifacts: dict[str, Path] = {}

    for case, sizes in resolved["cases"]:
        system, matrix_source = build_case_system(str(case), seed=seed)
        reference_config = f"configs/qsvt_full_matrix_{case}.yaml"
        full_encoding_row = full_matrix_block_encoding_row(system, case=str(case))
        full_preparation_row = full_matrix_state_preparation_row(system, case=str(case))

        U_full, full_sv, _ = np.linalg.svd(
            np.asarray(system.H_tilde, dtype=np.float64), full_matrices=False
        )
        s_full = full_sv / full_sv.max()
        alpha_norm_full = alpha / float(full_sv.max()) ** 2
        f_full = s_full / (s_full**2 + alpha_norm_full)
        c_full = float(1.05 * f_full.max())
        r_full = np.asarray(system.r_tilde, dtype=np.float64)
        coefficients = (f_full / c_full) * (U_full.T @ (r_full / np.linalg.norm(r_full)))
        full_postselection_proxy = float(np.linalg.norm(coefficients) ** 2)

        for size in sizes:
            workload = select_workload(system, matrix_source, case=str(case), size=int(size))
            workload_rows.append(
                workload_summary_row(system, workload, reference_config=reference_config)
            )
            encoding_row = block_encoding_study(workload, build_circuits=build_circuits)
            encoding_rows.append(encoding_row)
            preparation_row = state_preparation_study(
                workload,
                target_dimension=int(encoding_row["unitary_dimension"]),
                build_circuits=build_circuits,
            )
            preparation_rows.append(preparation_row)
            equivalence_row, ridge_solution, extras = equivalence_study(
                workload, alpha=alpha, degree=degree
            )
            equivalence_rows.append(equivalence_row)
            block_readout = readout_study(
                workload,
                ridge_solution,
                postselection_success_proxy=extras["postselection_success_proxy"],
                physical_recovery_factor=extras["target"].physical_recovery_factor,
                seed=int(resolved["readout_seed"]),
            )
            readout_rows.extend(block_readout)
            complexity.append(
                complexity_rows(
                    workload=workload,
                    system=system,
                    case=str(case),
                    alpha=alpha,
                    degree=degree,
                    block_encoding_row=encoding_row,
                    state_preparation_row=preparation_row,
                    postselection_success_proxy=extras["postselection_success_proxy"],
                    observable_count=len(block_readout),
                )
            )
            stem = f"{case}_{workload.block_shape}"
            block_path = output_dir / f"selected_block_{stem}.npy"
            residual_path = output_dir / f"selected_residual_{stem}.npy"
            np.save(block_path, workload.H_block)
            np.save(residual_path, workload.r_block)
            array_artifacts[f"selected_block_{stem}"] = block_path
            array_artifacts[f"selected_residual_{stem}"] = residual_path
            workload_metadata.append(
                {
                    "case": str(case),
                    "block_shape": workload.block_shape,
                    "seed": seed,
                    "matrix_source": workload.matrix_source,
                    "selection_policy": SELECTION_POLICY,
                    "selection_rationale": SELECTION_RATIONALE,
                    "selected_rows": [int(i) for i in workload.rows],
                    "selected_cols": [int(i) for i in workload.cols],
                    "row_labels": workload.row_labels,
                    "row_types": workload.row_types,
                    "column_labels": workload.column_labels,
                    "block_file": str(block_path),
                    "residual_file": str(residual_path),
                    "block_checksum": array_checksum(workload.H_block),
                    "residual_checksum": array_checksum(workload.r_block),
                }
            )

        encoding_rows.append(full_encoding_row)
        preparation_rows.append(full_preparation_row)
        complexity.append(
            complexity_rows(
                workload=None,
                system=system,
                case=str(case),
                alpha=alpha,
                degree=degree,
                block_encoding_row=full_encoding_row,
                state_preparation_row=full_preparation_row,
                postselection_success_proxy=full_postselection_proxy,
                observable_count=len(_deterministic_observables_count_proxy()),
            )
        )

    frames = {
        "selected_workload_summary.csv": pd.DataFrame(workload_rows, columns=WORKLOAD_COLUMNS),
        "block_encoding_report.csv": pd.DataFrame(encoding_rows),
        "state_preparation_report.csv": pd.DataFrame(preparation_rows),
        "qsvt_target_equivalence_report.csv": pd.DataFrame(equivalence_rows),
        "readout_report.csv": pd.DataFrame(readout_rows),
        "complexity_report.csv": pd.DataFrame(complexity),
    }
    artifacts: dict[str, Path] = dict(array_artifacts)
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        artifacts[name] = path

    metadata_path = output_dir / "workload_metadata.json"
    write_json(metadata_path, {"workloads": workload_metadata})
    artifacts["workload_metadata.json"] = metadata_path

    summary_text = _summary_markdown(
        alpha=alpha,
        degree=degree,
        workload_rows=workload_rows,
        encoding_rows=encoding_rows,
        preparation_rows=preparation_rows,
        equivalence_rows=equivalence_rows,
        readout_rows=readout_rows,
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    artifacts["summary.md"] = summary_path

    manifest_path = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="ieee_qsvt_pipeline_boundary",
        description=(
            "IEEE-derived quantum-pipeline boundary study: deterministic selected blocks "
            "of the generated PSSE weighted Jacobian, dense block-encoding validation, "
            "residual-state preparation, matched-alpha bounded-target/Ridge equivalence, "
            "selected-observable readout accounting, and complexity boundary accounting. "
            "Selected functionals only; not full-vector readout and no quantum-device run."
        ),
        command=str(resolved["command"]),
        artifacts=artifacts,
        input_files=[
            f"build_engineering_system:{case}:weighted_jacobian:seed={seed}"
            for case, _ in resolved["cases"]
        ],
        extra={
            "alpha": alpha,
            "degree": degree,
            "seed": seed,
            "cases": resolved["cases"],
            "selection_policy": SELECTION_POLICY,
            "equivalence_relative_tolerance": EQUIVALENCE_RELATIVE_TOLERANCE,
            "full_vector_tomography_non_claim": TOMOGRAPHY_NON_CLAIM,
        },
    )
    artifacts["manifest.json"] = manifest_path

    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "workload_rows": workload_rows,
        "encoding_rows": encoding_rows,
        "preparation_rows": preparation_rows,
        "equivalence_rows": equivalence_rows,
        "readout_rows": readout_rows,
        "complexity_rows": complexity,
        "summary_text": summary_text,
    }


def _deterministic_observables_count_proxy() -> tuple[str, ...]:
    """Observable-count proxy for the full-matrix modeled path (q selected functionals)."""

    return ("state_component", "angle_component", "voltage_component", "aggregate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the IEEE-derived QSVT pipeline boundary study"
    )
    parser.add_argument("--output-dir", default=str(BOUNDARY_DIR))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cases",
        default=json.dumps([[case, list(sizes)] for case, sizes in DEFAULT_CASES]),
        help='JSON list of [case, [block sizes]] pairs, e.g. [["ieee14", [4, 8]]]',
    )
    parser.add_argument("--no-circuits", action="store_true")
    args = parser.parse_args(argv)

    run = build_ieee_qsvt_pipeline_boundary(
        {
            "output_dir": args.output_dir,
            "alpha": args.alpha,
            "degree": args.degree,
            "seed": args.seed,
            "cases": json.loads(args.cases),
            "build_circuits": not args.no_circuits,
            "command": "scripts/run_ieee_qsvt_pipeline_boundary.py " + " ".join(argv or []),
        }
    )
    equivalence = run["equivalence_rows"]
    worst = max(row["relative_difference"] for row in equivalence)
    all_pass = all(row["equivalence_status"] == "pass" for row in equivalence)
    print(f"IEEE QSVT pipeline boundary study complete: {run['output_dir']}")
    print(
        f"blocks={len(equivalence)} worst_matched_alpha_relative_difference={worst:.3e} "
        f"status={'pass' if all_pass else 'fail'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
