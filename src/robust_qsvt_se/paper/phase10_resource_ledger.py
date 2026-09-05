"""Phase 10 WP E: end-to-end resource and classical comparator ledger.

Consolidates every Phase 10 workload into a single set of ledgers covering
residual loading, block encoding, QSVT degree and phase count, signal-unitary
calls, sparse lookup calls, postselection probability, selected-output readout,
full-vector recovery, nonlinear-loop repetition, and the classical comparators
(selected-output adjoint solve and full Ridge solve).

Workloads:

1. 4x4 selected-submatrix integrated chain (executed, sampled counts),
2. 8x8 selected-submatrix integrated chain (executed, sampled counts),
3. 8x8 sparse block-encoding wrapper (WP A; executed statevector),
4. full rectangular IEEE 14 selected-output QSVT (WP B; executed statevector),
5. full rectangular IEEE 30 selected-output QSVT (WP B; executed statevector),
6. nonlinear AC IEEE 14 QSVT-in-loop (WP D; executed statevector, per iteration),
7. modeled IEEE 30/57/118/300 full-rectangular rows (resource-estimated only).

Every row carries an explicit execution tier (executed sampled counts /
executed statevector / modeled / excluded).  Units are never merged: quantum
query and T-count estimates live in the quantum ledger, classical wall-clock in
the classical ledger, and the master ledger cross-references both without
claiming competitiveness.  No speedup and no matched-alpha QSVT-over-Ridge
superiority is asserted anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase10_common import (
    assert_safe,
    write_phase10_manifest,
)
from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import (
    apply_qsvt_sequence_to_vector,
    build_padded_dilation,
)
from robust_qsvt_se.paper.phase10_residual_loading import (
    dense_loader_metrics,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import sparsify_block
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/phase10_end_to_end_resource_ledger")
DEGREE = 31
DEGREE_CANDIDATES = (31, 39, 45)
READOUT_EPSILONS = (1.0e-2, 1.0e-3)
CLASSICAL_SOLVE_REPEATS = 30
NONLINEAR_ITERATIONS = 8
EXECUTED_FULL_CASES = ("ieee14", "ieee30")
MODELED_FULL_CASES = ("ieee57", "ieee118", "ieee300")
INTEGRATED_CHAIN_EVIDENCE = {
    4: Path("outputs/phase8_integrated_readout"),
    8: Path("outputs/phase9_integrated_8x8_readout"),
}

CLAIM = (
    "End-to-end Phase 10 resource ledger consolidating residual loading, block encoding, "
    "QSVT degree/phase/query counts, postselection, selected-output readout, full-vector "
    "recovery, nonlinear-loop repetition, and classical comparators (selected-output adjoint "
    "solve and full Ridge solve). Each row is tagged executed / statevector-simulated / "
    "modeled / excluded. Classical wall-clock and quantum query/T-count units are never "
    "merged; no speedup or competitiveness claim is made, and no matched-alpha QSVT-over-Ridge "
    "numerical superiority is claimed."
)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _matvec_p_succ(
    H: np.ndarray, r: np.ndarray, *, alpha: float, beta: float, degree: int, phase_cache_dir: Path
) -> dict[str, Any]:
    m, _ = H.shape
    singular = np.linalg.svd(H, compute_uv=False)
    # Rank-deficient blocks (e.g. the sparsified 8x8) have zero singular values;
    # fit the bounded target on the smallest *positive* normalized singular value,
    # matching the WP A/WP B convention.
    singular_pos = singular[singular > 1.0e-10]
    s_min_normalized = float(singular_pos.min() / beta)
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=float(alpha),
        domain_min=max(1.0e-4, 0.9 * s_min_normalized),
        domain_max=1.0,
        degree=int(degree),
        margin=1.05,
    )
    validate_qsvt_polynomial(np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3)
    cached = synthesize_pennylane_phases_cached(
        np.asarray(target.coefficients),
        angle_solver="iterative",
        cache_dir=phase_cache_dir,
        cache_metadata={"workload": "phase10_ledger", "degree": int(degree)},
    )
    phases = np.asarray(cached.phases, dtype=np.float64)
    dilation = build_padded_dilation(H, beta)
    padded_n = int(dilation["padded_dimension"])
    psi_in = np.zeros(2 * padded_n, dtype=np.complex128)
    psi_in[:m] = r / np.linalg.norm(r)
    psi_out = apply_qsvt_sequence_to_vector(
        dilation["unitary"], phases, encoded_dimension=padded_n, vector=psi_in
    )
    encoded = psi_out[:padded_n]
    return {
        "p_succ": float(np.vdot(encoded, encoded).real),
        "qubits": int(dilation["qubits"]),
        "padded_dimension": padded_n,
        "phase_count": int(phases.size),
        "bound_C": target.bound_C,
    }


def _time_classical(callable_fn: Any, repeats: int) -> float:
    best = math.inf
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        callable_fn()
        best = min(best, time.perf_counter() - start)
    return best


def classical_comparators(
    H: np.ndarray, r: np.ndarray, ell: np.ndarray, alpha: float, repeats: int
) -> dict[str, Any]:
    """Selected-output adjoint solve and full Ridge solve (classical wall-clock)."""

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = H.shape[1]
    gram = H.T @ H + alpha * np.eye(n)
    htr = H.T @ r

    def full_ridge() -> np.ndarray:
        return np.linalg.solve(gram, htr)

    def selected_adjoint() -> float:
        g = np.linalg.solve(gram, ell)  # adjoint system for the functional
        return float(g @ htr)

    full_seconds = _time_classical(full_ridge, repeats)
    adjoint_seconds = _time_classical(selected_adjoint, repeats)
    dx_ridge = full_ridge()
    return {
        "full_ridge_seconds_median_best": full_seconds,
        "selected_adjoint_seconds_median_best": adjoint_seconds,
        "full_ridge_flops_estimate": float(2 * n**3 / 3 + 2 * n**2),
        "selected_adjoint_flops_estimate": float(2 * n**2),
        "selected_output_ell_T_dx_ridge": float(ell @ dx_ridge),
        "n_states": int(n),
        "measurement": "median-of-best wall-clock over repeats on this host (not normalized)",
    }


def _selected_block(case: str, size: int, seed: int, sparsify: bool) -> dict[str, Any]:
    system, matrix_source = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_block, r_block, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=size,
        col_count=size,
        policy="largest_row_col_norms",
    )
    matrix = sparsify_block(H_block, keep_per_row=2) if sparsify else H_block
    return {"H": matrix, "r": r_block, "matrix_source": matrix_source}


def _integrated_chain_evidence(size: int) -> dict[str, Any]:
    """Load the executed Phase 8/9 integrated-chain register and target data."""

    evidence_dir = INTEGRATED_CHAIN_EVIDENCE[size]
    reference = json.loads(
        (evidence_dir / "integrated_readout_reference_values.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (evidence_dir / "integrated_readout_circuit_metadata.json").read_text(encoding="utf-8")
    )
    circuit_rows = list(metadata["per_observable_circuits"].values())
    qubit_counts = {int(row["circuit_qubits"]) for row in circuit_rows}
    if len(qubit_counts) != 1:
        raise ValueError(f"inconsistent integrated-chain qubit counts for {size}x{size}")
    return {
        "alpha": float(reference["alpha"]),
        "p_succ": float(reference["statevector_postselection_probability"]),
        "qubits": qubit_counts.pop(),
        "phase_count": int(metadata["phase_count"]),
        "bound_C": float(reference["bound_C"]),
        "evidence_source": str(evidence_dir),
    }


def build_workload_rows(seed: int, phase_cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    master: list[dict[str, Any]] = []
    quantum: list[dict[str, Any]] = []
    classical: list[dict[str, Any]] = []
    readout: list[dict[str, Any]] = []
    nonlinear: list[dict[str, Any]] = []

    # --- Selected-submatrix integrated chains (executed sampled counts) ---
    for size in (4, 8):
        block = _selected_block("ieee14", size, seed, sparsify=False)
        H, r = block["H"], block["r"]
        integrated = _integrated_chain_evidence(size)
        alpha = integrated["alpha"]
        prob = {
            "p_succ": integrated["p_succ"],
            "qubits": integrated["qubits"],
            "phase_count": integrated["phase_count"],
            "bound_C": integrated["bound_C"],
        }
        p_succ = prob["p_succ"]
        ell = np.zeros(H.shape[1])
        ell[0] = 1.0
        comparators = classical_comparators(H, r, ell, alpha, CLASSICAL_SOLVE_REPEATS)
        workload = f"selected_{size}x{size}_integrated_chain"
        tier = "executed_sampled_counts"
        counts = qsvt_sequence_operation_counts(prob["phase_count"])
        dense = dense_loader_metrics(r, compile_limit=16)
        master.append(
            _master_row(
                workload,
                tier,
                f"{size}x{size} selected submatrix",
                prob["qubits"],
                DEGREE,
                p_succ,
                comparators,
                block["matrix_source"],
                evidence=integrated["evidence_source"],
            )
        )
        quantum.append(
            {
                **_quantum_row(workload, tier, prob, counts, dense, sparse_calls=None),
                "note": "qubit count includes the signed-readout ancilla",
            }
        )
        classical.append(_classical_row(workload, tier, comparators))
        readout.extend(_readout_rows(workload, tier, p_succ, comparators["n_states"]))

    # --- 8x8 sparse block-encoding wrapper (WP A; executed statevector) ---
    # Use the same 6-bit-quantized block as WP A so the ledger p_succ matches the
    # executed sparse-wrapper package it cites.
    from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
        build_quantized_sparse_block,
    )

    raw_block = _selected_block("ieee14", 8, seed, sparsify=False)
    quantized = build_quantized_sparse_block(raw_block["H"], magnitude_bits=6)
    Hs = quantized.quantized
    rs = raw_block["r"]
    mu = quantized.mu
    pattern = np.abs(Hs.T) > 0.0
    slots = int(max(pattern.sum(axis=1).max(), pattern.sum(axis=0).max()))
    beta_sparse = slots * mu
    singular_s = np.linalg.svd(Hs, compute_uv=False)
    alpha_sparse = 4.0 * float(singular_s[singular_s > 1.0e-10].min()) ** 2
    prob_sparse = _matvec_p_succ(
        Hs, rs, alpha=alpha_sparse, beta=beta_sparse, degree=DEGREE, phase_cache_dir=phase_cache_dir
    )
    ell = np.zeros(Hs.shape[1])
    ell[0] = 1.0
    comparators_s = classical_comparators(Hs, rs, ell, alpha_sparse, CLASSICAL_SOLVE_REPEATS)
    counts_s = qsvt_sequence_operation_counts(prob_sparse["phase_count"])
    dense_s = dense_loader_metrics(rs, compile_limit=16)
    workload = "sparse_wrapper_8x8"
    tier = "executed_statevector"
    master.append(
        _master_row(
            workload,
            tier,
            "8x8 sparsified quantized selected block (slots=3)",
            6,
            DEGREE,
            prob_sparse["p_succ"],
            comparators_s,
            raw_block["matrix_source"],
            evidence="outputs/phase10_sparse_wrapper_8x8_complete",
        )
    )
    quantum.append(
        _quantum_row(
            workload,
            tier,
            {**prob_sparse, "qubits": 6},
            counts_s,
            dense_s,
            sparse_calls=2 * counts_s["signal_unitary_calls"],
        )
    )
    classical.append(_classical_row(workload, tier, comparators_s))
    readout.extend(_readout_rows(workload, tier, prob_sparse["p_succ"], comparators_s["n_states"]))

    # --- Full rectangular executed cases (WP B; executed statevector) ---
    for case in EXECUTED_FULL_CASES:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        H = np.asarray(system.H_tilde, dtype=np.float64)
        r = np.asarray(system.r_tilde, dtype=np.float64)
        beta = float(np.linalg.svd(H, compute_uv=False).max())
        alpha = 0.068 * beta**2
        prob = _matvec_p_succ(
            H, r, alpha=alpha, beta=beta, degree=DEGREE, phase_cache_dir=phase_cache_dir
        )
        ell = np.zeros(H.shape[1])
        ell[0] = 1.0
        comparators = classical_comparators(H, r, ell, alpha, CLASSICAL_SOLVE_REPEATS)
        counts = qsvt_sequence_operation_counts(prob["phase_count"])
        dense = dense_loader_metrics(r, compile_limit=16)
        workload = f"full_rectangular_{case}"
        tier = "executed_statevector"
        master.append(
            _master_row(
                workload,
                tier,
                f"full rectangular {H.shape[0]}x{H.shape[1]} (A=H^T/beta)",
                prob["qubits"],
                DEGREE,
                prob["p_succ"],
                comparators,
                matrix_source,
                evidence="outputs/phase10_full_rectangular_selected_output_qsvt",
            )
        )
        quantum.append(_quantum_row(workload, tier, prob, counts, dense, sparse_calls=None))
        classical.append(_classical_row(workload, tier, comparators))
        readout.extend(_readout_rows(workload, tier, prob["p_succ"], comparators["n_states"]))
        if case == "ieee14":
            nonlinear.extend(_nonlinear_rows(workload, prob["p_succ"], DEGREE))

    # --- Modeled larger cases (quantum side not executed) ---
    for case in MODELED_FULL_CASES:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        H = np.asarray(system.H_tilde, dtype=np.float64)
        r = np.asarray(system.r_tilde, dtype=np.float64)
        m, n = H.shape
        beta = float(np.linalg.svd(H, compute_uv=False).max())
        alpha = 0.068 * beta**2
        N = _next_power_of_two(max(m, n))
        certified_degree, fit_error = _certify_polynomial(H, alpha, beta)
        ell = np.zeros(n)
        ell[0] = 1.0
        comparators = classical_comparators(H, r, ell, alpha, CLASSICAL_SOLVE_REPEATS)
        counts = (
            qsvt_sequence_operation_counts(certified_degree + 1)
            if certified_degree is not None
            else {}
        )
        workload = f"full_rectangular_{case}"
        tier = "modeled"
        master.append(
            {
                "workload": workload,
                "execution_tier": tier,
                "matrix_description": f"full rectangular {m}x{n} (A=H^T/beta)",
                "qubits": int(math.log2(2 * N)),
                "degree": certified_degree,
                "polynomial_certified": certified_degree is not None,
                "target_fit_error": fit_error,
                "postselection_probability": None,
                "classical_full_ridge_seconds": comparators["full_ridge_seconds_median_best"],
                "classical_selected_adjoint_seconds": comparators[
                    "selected_adjoint_seconds_median_best"
                ],
                "matrix_source": matrix_source,
                "evidence_source": "resource estimate; quantum side not executed",
            }
        )
        quantum.append(
            {
                "workload": workload,
                "execution_tier": tier,
                "qubits": int(math.log2(2 * N)),
                "degree": certified_degree,
                "signal_unitary_calls_per_attempt": counts.get("signal_unitary_calls"),
                "projector_phase_ops_per_attempt": counts.get("projector_phase_operations"),
                "block_encoding": "padded dense dilation (modeled; not executed)",
                "residual_loading_dense_cx_estimate": max(N - 2, 0),
                "postselection_probability": None,
                "note": "quantum resource estimate only; not executed",
            }
        )
        classical.append(_classical_row(workload, tier, comparators))

    return {
        "master": master,
        "quantum": quantum,
        "classical": classical,
        "readout": readout,
        "nonlinear": nonlinear,
    }


def _certify_polynomial(
    H: np.ndarray, alpha: float, beta: float
) -> tuple[int | None, float | None]:
    singular = np.linalg.svd(H, compute_uv=False)
    for degree in DEGREE_CANDIDATES:
        for margin in (1.05, 1.25):
            target = fit_codesigned_bounded_polynomial(
                beta=beta,
                alpha=alpha,
                domain_min=max(1.0e-4, 0.9 * float(singular.min()) / beta),
                domain_max=1.0,
                degree=degree,
                margin=margin,
            )
            try:
                validate_qsvt_polynomial(
                    np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
                )
            except Exception:
                continue
            return degree, target.fit_max_abs_error
    return None, None


def _master_row(
    workload: str,
    tier: str,
    matrix_description: str,
    qubits: int,
    degree: int,
    p_succ: float,
    comparators: dict[str, Any],
    matrix_source: str,
    *,
    evidence: str,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "execution_tier": tier,
        "matrix_description": matrix_description,
        "qubits": qubits,
        "degree": degree,
        "polynomial_certified": True,
        "postselection_probability": p_succ,
        "classical_full_ridge_seconds": comparators["full_ridge_seconds_median_best"],
        "classical_selected_adjoint_seconds": comparators["selected_adjoint_seconds_median_best"],
        "matrix_source": matrix_source,
        "evidence_source": evidence,
    }


def _quantum_row(
    workload: str,
    tier: str,
    prob: dict[str, Any],
    counts: dict[str, int],
    dense: dict[str, Any],
    *,
    sparse_calls: int | None,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "execution_tier": tier,
        "qubits": prob["qubits"],
        "degree": DEGREE,
        "phase_count": prob["phase_count"],
        "signal_unitary_calls_per_attempt": counts["signal_unitary_calls"],
        "projector_phase_ops_per_attempt": counts["projector_phase_operations"],
        "sparse_lookup_calls_per_attempt": sparse_calls,
        "block_encoding": (
            "compiled sparse query-model wrapper"
            if sparse_calls is not None
            else "padded dense dilation (statevector)"
        ),
        "residual_loading_dense_gates": dense.get("transpiled_gate_count")
        or dense.get("transpiled_cx_count"),
        "postselection_probability": prob["p_succ"],
        "bound_C": prob["bound_C"],
    }


def _classical_row(workload: str, tier: str, comparators: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload": workload,
        "execution_tier": tier,
        "n_states": comparators["n_states"],
        "full_ridge_seconds_median_best": comparators["full_ridge_seconds_median_best"],
        "selected_adjoint_seconds_median_best": comparators["selected_adjoint_seconds_median_best"],
        "full_ridge_flops_estimate": comparators["full_ridge_flops_estimate"],
        "selected_adjoint_flops_estimate": comparators["selected_adjoint_flops_estimate"],
        "selected_output_ell_T_dx_ridge": comparators["selected_output_ell_T_dx_ridge"],
        "unit": "seconds (classical wall-clock, host-specific, not merged with quantum counts)",
    }


def _readout_rows(workload: str, tier: str, p_succ: float, n_states: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon in READOUT_EPSILONS:
        shots = required_shots_for_additive_error(epsilon)
        attempts = math.ceil(shots / p_succ) if p_succ > 0 else None
        rows.append(
            {
                "workload": workload,
                "execution_tier": tier,
                "readout_epsilon": epsilon,
                "shots_per_functional": shots,
                "postselection_probability": p_succ,
                "attempts_per_functional_no_AA": attempts,
                "signal_calls_per_attempt": DEGREE,
                "full_vector_recovery_attempts": attempts * n_states if attempts else None,
                "full_vector_recovery_note": "T_full_vector ~ n * T_selected",
            }
        )
    return rows


def _nonlinear_rows(workload: str, p_succ: float, degree: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon in READOUT_EPSILONS:
        shots = required_shots_for_additive_error(epsilon)
        attempts_per_iter = math.ceil(shots / p_succ) if p_succ > 0 else None
        rows.append(
            {
                "workload": f"nonlinear_{workload}",
                "execution_tier": "executed_statevector_per_iteration",
                "readout_epsilon": epsilon,
                "iterations": NONLINEAR_ITERATIONS,
                "postselection_probability": p_succ,
                "attempts_per_iteration": attempts_per_iter,
                "signal_calls_per_attempt": degree,
                "state_prep_reloads_per_iteration": attempts_per_iter,
                "total_attempts_over_loop": (
                    attempts_per_iter * NONLINEAR_ITERATIONS if attempts_per_iter else None
                ),
                "residual_and_jacobian_rebuilt_per_iteration": True,
            }
        )
    return rows


def run_phase10_resource_ledger(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "command": "scripts/run_phase10_resource_ledger.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")

    rows = build_workload_rows(int(resolved["seed"]), phase_cache_dir)

    master_csv = output_dir / "end_to_end_resource_ledger.csv"
    classical_csv = output_dir / "classical_comparator_ledger.csv"
    quantum_csv = output_dir / "quantum_component_ledger.csv"
    readout_csv = output_dir / "readout_cost_ledger.csv"
    nonlinear_csv = output_dir / "nonlinear_repetition_ledger.csv"
    readme_md = output_dir / "README.md"

    pd.DataFrame(rows["master"]).to_csv(master_csv, index=False)
    pd.DataFrame(rows["classical"]).to_csv(classical_csv, index=False)
    pd.DataFrame(rows["quantum"]).to_csv(quantum_csv, index=False)
    pd.DataFrame(rows["readout"]).to_csv(readout_csv, index=False)
    pd.DataFrame(rows["nonlinear"]).to_csv(nonlinear_csv, index=False)
    readme_md.write_text(_readme(rows["master"]), encoding="utf-8")

    artifacts = {
        "end_to_end_resource_ledger_csv": master_csv,
        "classical_comparator_ledger_csv": classical_csv,
        "quantum_component_ledger_csv": quantum_csv,
        "readout_cost_ledger_csv": readout_csv,
        "nonlinear_repetition_ledger_csv": nonlinear_csv,
        "readme_md": readme_md,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_end_to_end_resource_ledger",
        script_name="scripts/run_phase10_resource_ledger.py",
        command=str(resolved["command"]),
        description=CLAIM,
        artifacts=artifacts,
        seeds={"system_seed": int(resolved["seed"])},
        extra={
            "tier_counts": _tier_counts(rows["master"]),
            "readout_epsilons": list(READOUT_EPSILONS),
        },
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def _tier_counts(master: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in master:
        counts[row["execution_tier"]] = counts.get(row["execution_tier"], 0) + 1
    return counts


def _readme(master: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 10 WP E: End-to-End Resource and Classical Comparator Ledger",
        "",
        CLAIM,
        "",
        "## Execution tiers",
        "",
        "- **executed_sampled_counts**: 4x4 and 8x8 selected-submatrix integrated chains "
        "(sampled Aer/statevector counts; evidence in phase8/phase9 packages).",
        "- **executed_statevector**: 8x8 sparse block-encoding wrapper (WP A), full "
        "rectangular IEEE 14 and IEEE 30 selected-output QSVT (WP B).",
        "- **executed_statevector_per_iteration**: nonlinear AC IEEE 14 QSVT-in-loop (WP D).",
        "- **modeled**: IEEE 57/118/300 full-rectangular rows (quantum side not executed; "
        "polynomial certification and dimensions only).",
        "- **excluded**: fault-tolerant/error-correction overheads are not provided "
        "(logical-level accounting only).",
        "",
        "## Master ledger",
        "",
        "| workload | tier | qubits | degree | p_succ | classical full Ridge (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in master:
        lines.append(
            f"| {row['workload']} | {row['execution_tier']} | {row['qubits']} | {row['degree']} | "
            f"{_fmt(row['postselection_probability'], '{:.4f}')} | "
            f"{_fmt(row['classical_full_ridge_seconds'])} |"
        )
    lines += [
        "",
        "## Units and interpretation",
        "",
        "- The **quantum component ledger** reports qubits, degree, phase count, signal-unitary "
        "calls per attempt, sparse-lookup calls, and residual-loading gate counts.",
        "- Qubit counts report the full register of the executed circuit for each row. The "
        "integrated 4x4 and 8x8 rows therefore include the signed-readout ancilla; "
        "statevector-only rows do not add a readout ancilla that was not executed.",
        "- The **classical comparator ledger** reports the selected-output adjoint solve and the "
        "full Ridge solve in **wall-clock seconds** on this host (not hardware-normalized) plus "
        "order-of-magnitude flop estimates.",
        "- The **readout cost ledger** reports shots = ceil(0.25/eps^2), attempts = shots/p_succ, "
        "and full-vector recovery ~ n * selected.",
        "- The **nonlinear repetition ledger** reports per-iteration attempts with residual and "
        "Jacobian rebuilt each iteration.",
        "",
        "Classical wall-clock seconds and quantum query/T-count estimates are in different units "
        "and are never merged into a single figure. No row asserts speedup, competitiveness, or "
        "matched-alpha QSVT-over-Ridge numerical superiority.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _fmt(value: Any, spec: str = "{:.3e}") -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return spec.format(number)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 WP E: end-to-end resource and classical comparator ledger"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_phase10_resource_ledger(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "command": "scripts/run_phase10_resource_ledger.py " + " ".join(argv or []),
        }
    )
    print(pd.DataFrame(run["rows"]["master"]).to_string(index=False))
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
