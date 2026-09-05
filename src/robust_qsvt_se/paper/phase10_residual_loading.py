"""Phase 10 WP C: residual-state loading implementations and repeat-cost accounting.

Makes residual loading concrete for every QSVT workload rather than leaving it a
vague modeled term, and accounts for its *repeated* use inside sampling and
postselection attempts.

Three loading modes are implemented and compared:

* Mode 1 - exact dense ``Initialize`` (Qiskit ``StatePreparation``): compiled
  and statevector-validated for the small workloads; reported (dimension,
  qubits, gate count, depth, transpilation feasibility) for all.
* Mode 2 - explicit binary-tree / Möttönen real-amplitude loader
  (:mod:`robust_qsvt_se.qsvt.binary_tree_state_loader`): a concrete cascade of
  ``N - 1`` multiplexed RY rotations; angle correctness is exact at every size,
  the circuit is statevector-validated at the small feasible sizes, and rotation
  and qubit counts are reported analytically for the full residuals.
* Mode 3 - QROM / oracle-access loading cost model: an explicit, traceable
  resource model (address qubits, value precision, stored values, Toffoli and
  T-count estimates, uncomputation, per-attempt and nonlinear-loop repetition).

The repetition ledger uses

    T_Q = (q * N_shots / p_succ) * [T_prep + d * T_U + (d + 1) * T_phi + T_read]

with every symbol reported as a primitive-invocation count (never merged with
wall-clock time), and ``T_prep`` counted *once per postselection attempt*
because a failed postselection discards the prepared state.  Full-vector
recovery is reported as ``T_full_vector ~ n * T_selected``.

Postselection probabilities are computed exactly from the fitted bounded
polynomial and the singular-value decomposition (the same value the executed
statevector runs produce), and every source is tagged.  Nothing here claims a
scalable advantage; the binary-tree and QROM costs are explicitly not
asymptotically cheaper than dense loading at these sizes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase10_common import (
    assert_safe,
    json_ready,
    write_phase10_manifest,
)
from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import (
    apply_qsvt_sequence_to_vector,
    build_padded_dilation,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.binary_tree_state_loader import (
    build_binary_tree_circuit,
    compute_binary_tree_plan,
    validate_binary_tree_circuit,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.gate_state_preparation import (
    build_initialize_circuit,
    normalize_and_pad_for_gate_preparation,
    validate_initialize_circuit,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import sparsify_block
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_DIR = Path("outputs/phase10_residual_loading_accounting")
COMPILE_DIMENSION_LIMIT = 16  # statevector-validate compiled loaders up to this dimension
QROM_VALUE_PRECISION_BITS = 8
T_PER_TOFFOLI = 4  # standard Clifford+T Toffoli synthesis
READOUT_EPSILONS = (1.0e-2, 1.0e-3)
NONLINEAR_ITERATIONS = 8  # matches the IEEE 14 nonlinear-AC iteration cap

CLAIM = (
    "Explicit residual-loading implementations (exact dense Initialize, binary-tree "
    "Möttönen real-amplitude loader, and a QROM oracle-access cost model) with per-attempt "
    "and nonlinear-loop repeat-cost accounting for the QSVT workloads. Small loaders are "
    "compiled and statevector-validated; larger residual loaders are reported with analytic "
    "rotation and qubit counts. The binary-tree and QROM costs are NOT claimed to be "
    "asymptotically cheaper than dense loading, and no scalable state-preparation advantage "
    "is claimed."
)


def _qubits_for(dimension: int) -> int:
    return math.ceil(math.log2(max(int(dimension), 2)))


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def build_workloads(seed: int) -> list[dict[str, Any]]:
    """Residual vectors and QSVT parameters for every Phase 10 workload."""

    workloads: list[dict[str, Any]] = []
    system14, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H14 = np.asarray(system14.H_tilde, dtype=np.float64)
    r14 = np.asarray(system14.r_tilde, dtype=np.float64)

    # Selected 4x4 and 8x8 blocks (sparse wrapper workloads).
    for size in (4, 8):
        H_block, r_block, _rows, _cols = select_deterministic_block(
            H14, r14, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        matrix = sparsify_block(H_block, keep_per_row=2) if size == 8 else H_block
        singular = np.linalg.svd(matrix, compute_uv=False)
        singular_pos = singular[singular > 1.0e-10]
        alpha = 4.0 * float(singular_pos.min()) ** 2
        # Sparse wrapper encodes A = H^T/(s*mu); use the same s*mu normalization convention.
        if size == 8:
            mu = float(np.max(np.abs(matrix)))
            pattern = np.abs(matrix.T) > 0.0
            slots = int(max(pattern.sum(axis=1).max(), pattern.sum(axis=0).max()))
            beta = slots * mu
        else:
            beta = float(singular.max())
        workloads.append(
            {
                "workload": f"selected_{size}x{size}_integrated_chain",
                "kind": "selected",
                "H": matrix,
                "residual": r_block,
                "alpha": alpha,
                "beta": beta,
                "degree": 31,
            }
        )

    # Full rectangular residuals (executed cases): A = H^T/beta.
    system30, _ = build_engineering_system(
        {
            "case_name": "ieee30",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    full_cases = (
        ("ieee14", H14, r14),
        ("ieee30", np.asarray(system30.H_tilde, dtype=np.float64), np.asarray(system30.r_tilde)),
    )
    for case, H_full, r_full in full_cases:
        singular = np.linalg.svd(H_full, compute_uv=False)
        beta = float(singular.max())
        workloads.append(
            {
                "workload": f"full_rectangular_{case}",
                "kind": "full_rectangular",
                "H": H_full,
                "residual": r_full,
                "alpha": 0.068 * beta**2,
                "beta": beta,
                "degree": 31,
            }
        )
    return workloads


def exact_postselection_probability(
    H: np.ndarray,
    residual: np.ndarray,
    *,
    alpha: float,
    beta: float,
    degree: int,
    phase_cache_dir: str | Path,
) -> dict[str, Any]:
    """p_succ from the executed QSVT circuit path (matches the WP B convention).

    The postselection probability is the squared norm of the *full complex*
    encoded amplitude ``||(U_phi)[:N,:N] @ r_padded||^2`` (the block-encoding
    ancilla landing in success), matching the repository convention used by the
    integrated-readout and sparse-wrapper QSVT paths.  Computed with the same
    validated matrix-vector QSVT sequence as WP B; the real-part singular-value
    transform is reported separately as a reference.
    """

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    m, _ = H.shape
    singular = np.linalg.svd(H, compute_uv=False)
    s_min_normalized = float(singular.min() / beta)
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=float(alpha),
        domain_min=max(1.0e-4, 0.9 * s_min_normalized),
        domain_max=1.0,
        degree=int(degree),
        margin=1.05,
    )
    status = "bounded"
    try:
        validate_qsvt_polynomial(
            np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
        )
    except Exception:
        status = "degree_limited_bound"

    A = H.T / beta
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    r_hat = r / np.linalg.norm(r)
    real_encoded = target.polynomial(S) * (Vt @ r_hat)
    p_succ_real_svt = float(np.dot(real_encoded, real_encoded))

    cached = synthesize_pennylane_phases_cached(
        np.asarray(target.coefficients),
        angle_solver="iterative",
        cache_dir=phase_cache_dir,
        cache_metadata={"workload": "phase10_residual_loading", "degree": int(degree)},
    )
    phases = np.asarray(cached.phases, dtype=np.float64)
    dilation = build_padded_dilation(H, beta)
    padded_n = int(dilation["padded_dimension"])
    psi_in = np.zeros(2 * padded_n, dtype=np.complex128)
    psi_in[:m] = r_hat
    psi_out = apply_qsvt_sequence_to_vector(
        dilation["unitary"], phases, encoded_dimension=padded_n, vector=psi_in
    )
    encoded = psi_out[:padded_n]
    p_succ = float(np.vdot(encoded, encoded).real)
    return {
        "p_succ": p_succ,
        "p_succ_real_svt_reference": p_succ_real_svt,
        "degree": int(degree),
        "bound_C": target.bound_C,
        "polynomial_status": status,
        "provenance": (
            "executed QSVT matrix-vector sequence on the padded dense dilation; "
            "p_succ = ||full complex encoded amplitude||^2 (repository postselection "
            "convention, matches WP B executed runs)"
        ),
    }


def dense_loader_metrics(residual: np.ndarray, *, compile_limit: int) -> dict[str, Any]:
    prep = normalize_and_pad_for_gate_preparation(np.asarray(residual, dtype=np.float64))
    row: dict[str, Any] = {
        "mode": "exact_dense_initialize",
        "input_dimension": prep.input_dimension,
        "padded_dimension": prep.padded_dimension,
        "qubits": prep.n_qubits,
        "counted_per_attempt": True,
    }
    if prep.padded_dimension <= compile_limit:
        circuit = build_initialize_circuit(prep.padded_state)
        validation = validate_initialize_circuit(circuit, prep.padded_state)
        from qiskit import transpile

        transpiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
        counts = {str(k): int(v) for k, v in transpiled.count_ops().items()}
        row.update(
            {
                "compiled": True,
                "state_preparation_l2_error": validation["state_preparation_l2_error"],
                "state_preparation_fidelity": validation["state_preparation_fidelity"],
                "transpiled_gate_count": int(sum(counts.values())),
                "transpiled_depth": int(transpiled.depth()),
                "transpiled_cx_count": int(counts.get("cx", 0)),
                "transpilation_feasible": True,
            }
        )
    else:
        # Analytic Shende-Bullock-Markov cost for dense preparation of a real state.
        cx_estimate = prep.padded_dimension - 2 if prep.padded_dimension >= 2 else 0
        row.update(
            {
                "compiled": False,
                "state_preparation_l2_error": None,
                "state_preparation_fidelity": None,
                "transpiled_gate_count": None,
                "transpiled_depth": None,
                "transpiled_cx_count": int(cx_estimate),
                "transpilation_feasible": True,
                "note": (
                    "compilation feasible but not transpiled at this dimension; CX estimate "
                    "is the Shende-Bullock-Markov lower bound (2^n - 2 CX) for real states"
                ),
            }
        )
    return row


def binary_tree_loader_metrics(residual: np.ndarray, *, compile_limit: int) -> dict[str, Any]:
    plan = compute_binary_tree_plan(np.asarray(residual, dtype=np.float64))
    row: dict[str, Any] = {
        "mode": "binary_tree_mottonen",
        "input_dimension": plan.input_dimension,
        "padded_dimension": plan.dimension,
        "qubits": plan.n_qubits,
        "rotation_count": plan.rotation_count,
        "nonzero_rotation_count": plan.nonzero_rotation_count,
        "angle_reconstruction_error": plan.reconstruction_error,
        "counted_per_attempt": True,
    }
    if plan.dimension <= compile_limit:
        circuit = build_binary_tree_circuit(plan)
        validation = validate_binary_tree_circuit(circuit, plan)
        row.update(
            {
                "compiled": True,
                "state_preparation_l2_error": validation["state_preparation_l2_error"],
                "state_preparation_fidelity": validation["state_preparation_fidelity"],
                "raw_gate_count": validation["raw_gate_count"],
                "raw_depth": validation["raw_depth"],
                "compilation_feasible": True,
            }
        )
    else:
        row.update(
            {
                "compiled": False,
                "state_preparation_l2_error": plan.reconstruction_error,
                "state_preparation_fidelity": None,
                "raw_gate_count": None,
                "raw_depth": None,
                "compilation_feasible": True,
                "note": (
                    "angle plan is exact (reconstruction error reported); the multiplexed-RY "
                    f"circuit has {plan.rotation_count} rotations but is not transpiled at this "
                    "dimension because multi-controlled expansion is expensive"
                ),
            }
        )
    return row


def qrom_loader_cost(residual: np.ndarray, *, precision_bits: int) -> dict[str, Any]:
    """Explicit QROM/oracle-access loading resource model (traceable, modeled)."""

    values = np.asarray(residual, dtype=np.float64)
    stored_values = int(np.count_nonzero(np.abs(values) > 0.0))
    padded_dimension = _next_power_of_two(values.size)
    address_qubits = _qubits_for(padded_dimension)
    # QROM SELECT via unary iteration: ~ (stored_values - 1) Toffolis to fan out
    # the address, value register loaded with `precision_bits` outputs.
    select_toffolis = max(stored_values - 1, 0)
    value_load_toffolis = stored_values * precision_bits
    total_toffolis = select_toffolis + value_load_toffolis
    # Amplitude synthesis: one controlled RY per precision bit (angle from the value word).
    rotation_gates = precision_bits
    # Uncomputation of the address fan-out is a measurement-based mirror (~ same Toffoli count).
    uncompute_toffolis = select_toffolis
    return {
        "mode": "qrom_oracle_access",
        "input_dimension": int(values.size),
        "padded_dimension": int(padded_dimension),
        "qubits": int(address_qubits + precision_bits + 1),
        "address_qubits": int(address_qubits),
        "value_precision_bits": int(precision_bits),
        "value_register_qubits": int(precision_bits),
        "stored_values": stored_values,
        "select_toffoli_estimate": int(select_toffolis),
        "value_load_toffoli_estimate": int(value_load_toffolis),
        "toffoli_estimate": int(total_toffolis),
        "uncompute_toffoli_estimate": int(uncompute_toffolis),
        "t_count_estimate": int((total_toffolis + uncompute_toffolis) * T_PER_TOFFOLI),
        "amplitude_rotation_gates": int(rotation_gates),
        "counted_per_attempt": True,
        "model": (
            "unary-iteration QROM SELECT + fixed-point value register + amplitude rotation; "
            "T-count = 4 * (SELECT + value-load + uncompute) Toffolis (modeled, not compiled)"
        ),
    }


def repetition_accounting(
    workload: dict[str, Any],
    p_succ: float,
    dense_metrics: dict[str, Any],
    tree_metrics: dict[str, Any],
    qrom_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per-attempt and total primitive counts under the T_Q repetition formula."""

    degree = int(workload["degree"])
    n_states = int(np.asarray(workload["H"]).shape[1])
    signal_calls_per_attempt = degree
    phase_ops_per_attempt = degree + 1
    rows: list[dict[str, Any]] = []
    prep_costs = {
        "dense_initialize": dense_metrics.get("transpiled_gate_count")
        or dense_metrics.get("transpiled_cx_count"),
        "binary_tree": tree_metrics.get("rotation_count"),
        "qrom_toffolis": qrom_metrics.get("toffoli_estimate"),
    }
    for epsilon in READOUT_EPSILONS:
        shots = required_shots_for_additive_error(epsilon)
        for q_label, q in (("single_functional", 1), ("full_vector", n_states)):
            attempts = math.ceil(q * shots / p_succ) if p_succ > 0 else math.inf
            rows.append(
                {
                    "workload": workload["workload"],
                    "kind": workload["kind"],
                    "readout_epsilon": epsilon,
                    "readout_shots_per_functional": shots,
                    "q_functionals": q,
                    "q_label": q_label,
                    "p_succ": p_succ,
                    "degree": degree,
                    "n_states": n_states,
                    "postselection_attempts_no_AA": attempts,
                    "state_prep_invocations": attempts,
                    "signal_unitary_calls_total": attempts * signal_calls_per_attempt,
                    "projector_phase_ops_total": attempts * phase_ops_per_attempt,
                    "readout_measurements_total": attempts,
                    "prep_gates_dense_initialize_per_attempt": prep_costs["dense_initialize"],
                    "prep_rotations_binary_tree_per_attempt": prep_costs["binary_tree"],
                    "prep_toffolis_qrom_per_attempt": prep_costs["qrom_toffolis"],
                    "T_prep_counted_per_attempt": True,
                    "T_Q_formula": (
                        "(q * N_shots / p_succ) * [T_prep + d*T_U + (d+1)*T_phi + T_read]"
                    ),
                }
            )
    return rows


def nonlinear_loop_accounting(
    workload: dict[str, Any], p_succ: float, tree_metrics: dict[str, Any], iterations: int
) -> list[dict[str, Any]]:
    """Per-iteration residual-loading repetition for a nonlinear AC QSVT loop."""

    degree = int(workload["degree"])
    rows: list[dict[str, Any]] = []
    for epsilon in READOUT_EPSILONS:
        shots = required_shots_for_additive_error(epsilon)
        attempts_per_iter = math.ceil(shots / p_succ) if p_succ > 0 else math.inf
        rows.append(
            {
                "workload": workload["workload"],
                "readout_epsilon": epsilon,
                "iterations": iterations,
                "p_succ_per_iteration": p_succ,
                "degree_per_iteration": degree,
                "residual_reload_per_iteration": True,
                "jacobian_rebuild_per_iteration": True,
                "state_prep_invocations_per_iteration": attempts_per_iter,
                "state_prep_invocations_total_loop": attempts_per_iter * iterations,
                "binary_tree_rotations_per_iteration": tree_metrics.get("rotation_count"),
                "binary_tree_rotations_total_loop": (
                    tree_metrics.get("rotation_count", 0) * attempts_per_iter * iterations
                ),
                "note": (
                    "residual and Jacobian are rebuilt each Gauss-Newton iteration, so the "
                    "residual state is reloaded on every postselection attempt of every "
                    "iteration; beta_k and lambda_k are recomputed per iteration"
                ),
            }
        )
    return rows


def run_phase10_residual_loading(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "compile_limit": COMPILE_DIMENSION_LIMIT,
        "qrom_precision_bits": QROM_VALUE_PRECISION_BITS,
        "nonlinear_iterations": NONLINEAR_ITERATIONS,
        "command": "scripts/run_phase10_residual_loading.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")
    compile_limit = int(resolved["compile_limit"])

    workloads = build_workloads(int(resolved["seed"]))
    mode_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for workload in workloads:
        residual = workload["residual"]
        prob = exact_postselection_probability(
            workload["H"],
            residual,
            alpha=workload["alpha"],
            beta=workload["beta"],
            degree=workload["degree"],
            phase_cache_dir=phase_cache_dir,
        )
        p_succ = prob["p_succ"]
        dense = dense_loader_metrics(residual, compile_limit=compile_limit)
        tree = binary_tree_loader_metrics(residual, compile_limit=compile_limit)
        qrom = qrom_loader_cost(residual, precision_bits=int(resolved["qrom_precision_bits"]))
        for metrics in (dense, tree, qrom):
            mode_rows.append(
                {"workload": workload["workload"], "kind": workload["kind"], **metrics}
            )
        repetition = repetition_accounting(workload, p_succ, dense, tree, qrom)
        if workload["kind"] == "selected":
            selected_rows.extend(repetition)
        else:
            full_rows.extend(repetition)
        if workload["workload"] == "full_rectangular_ieee14":
            nonlinear_rows.extend(
                nonlinear_loop_accounting(
                    workload, p_succ, tree, int(resolved["nonlinear_iterations"])
                )
            )
        summary[workload["workload"]] = {
            "p_succ": p_succ,
            "p_succ_provenance": prob["provenance"],
            "degree": prob["degree"],
            "residual_dimension": int(np.asarray(residual).size),
            "padded_dimension": dense["padded_dimension"],
            "dense_compiled": dense["compiled"],
            "binary_tree_compiled": tree["compiled"],
            "binary_tree_rotations": tree["rotation_count"],
            "qrom_toffoli_estimate": qrom["toffoli_estimate"],
            "qrom_t_count_estimate": qrom["t_count_estimate"],
        }

    modes_csv = output_dir / "residual_loading_modes.csv"
    selected_csv = output_dir / "residual_loading_selected_workloads.csv"
    full_csv = output_dir / "residual_loading_full_rectangular.csv"
    nonlinear_csv = output_dir / "residual_loading_nonlinear_loop.csv"
    summary_json = output_dir / "residual_loading_resource_summary.json"
    readme_md = output_dir / "README.md"

    pd.DataFrame(mode_rows).to_csv(modes_csv, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_csv, index=False)
    pd.DataFrame(full_rows).to_csv(full_csv, index=False)
    pd.DataFrame(nonlinear_rows).to_csv(nonlinear_csv, index=False)
    write_json(
        summary_json,
        json_ready(
            {
                "workloads": summary,
                "readout_epsilons": list(READOUT_EPSILONS),
                "qrom_precision_bits": int(resolved["qrom_precision_bits"]),
                "t_per_toffoli": T_PER_TOFFOLI,
                "repetition_formula": (
                    "T_Q = (q * N_shots / p_succ) * [T_prep + d*T_U + (d+1)*T_phi + T_read]; "
                    "T_prep counted once per postselection attempt; "
                    "T_full_vector ~ n * T_selected"
                ),
                "claim_boundary": CLAIM,
            }
        ),
    )
    readme_md.write_text(_readme(mode_rows, summary, nonlinear_rows), encoding="utf-8")

    artifacts = {
        "residual_loading_modes_csv": modes_csv,
        "residual_loading_selected_workloads_csv": selected_csv,
        "residual_loading_full_rectangular_csv": full_csv,
        "residual_loading_nonlinear_loop_csv": nonlinear_csv,
        "residual_loading_resource_summary_json": summary_json,
        "readme_md": readme_md,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_residual_loading_accounting",
        script_name="scripts/run_phase10_residual_loading.py",
        command=str(resolved["command"]),
        description=CLAIM,
        artifacts=artifacts,
        seeds={"system_seed": int(resolved["seed"])},
        extra={"workload_summary": summary},
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "mode_rows": mode_rows,
        "selected_rows": selected_rows,
        "full_rows": full_rows,
        "nonlinear_rows": nonlinear_rows,
        "summary": summary,
        "artifacts": artifacts,
    }


def _readme(
    mode_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    nonlinear_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 10 WP C: Residual Loading and Repeat-Cost Accounting",
        "",
        CLAIM,
        "",
        "## Loading modes",
        "",
        "- **Mode 1 - exact dense Initialize**: Qiskit `StatePreparation`; compiled and "
        "statevector-validated for small workloads, analytic Shende-Bullock-Markov CX bound "
        f"(2^n - 2) beyond dimension {COMPILE_DIMENSION_LIMIT}.",
        "- **Mode 2 - binary-tree / Möttönen loader**: explicit cascade of N-1 multiplexed "
        "RY rotations; the signed `atan2` angle plan is exact at every size (reconstruction "
        "error reported), and the circuit is statevector-validated at small sizes.",
        "- **Mode 3 - QROM oracle-access cost model**: unary-iteration SELECT + fixed-point "
        "value register + amplitude rotation; Toffoli, T-count, and uncomputation are modeled "
        "and traceable, not compiled.",
        "",
        "| workload | mode | dim | qubits | key cost | validated |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in mode_rows:
        if row["mode"] == "exact_dense_initialize":
            cost = (
                f"{row.get('transpiled_gate_count')} gates"
                if row.get("compiled")
                else f"~{row.get('transpiled_cx_count')} CX (est)"
            )
            validated = (
                f"fid {row.get('state_preparation_fidelity'):.4f}"
                if row.get("compiled")
                else "analytic"
            )
        elif row["mode"] == "binary_tree_mottonen":
            cost = f"{row.get('rotation_count')} RY"
            validated = (
                f"fid {row.get('state_preparation_fidelity'):.4f}"
                if row.get("compiled")
                else f"angle err {row.get('angle_reconstruction_error'):.1e}"
            )
        else:
            cost = f"{row.get('toffoli_estimate')} Toffoli / {row.get('t_count_estimate')} T"
            validated = "modeled"
        lines.append(
            f"| {row['workload']} | {row['mode']} | {row['padded_dimension']} | "
            f"{row['qubits']} | {cost} | {validated} |"
        )
    lines += [
        "",
        "## Repetition accounting",
        "",
        "`T_Q = (q * N_shots / p_succ) * [T_prep + d*T_U + (d+1)*T_phi + T_read]`, with "
        "`T_prep` counted **once per postselection attempt** (a failed postselection discards "
        "the prepared state) and `T_full_vector ~ n * T_selected`. Per-workload attempt and "
        "primitive-count breakdowns are in `residual_loading_selected_workloads.csv` and "
        "`residual_loading_full_rectangular.csv`. Postselection probabilities:",
        "",
        "| workload | p_succ | residual dim | binary-tree RY | QROM Toffoli |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, data in summary.items():
        lines.append(
            f"| {name} | {data['p_succ']:.4f} | {data['residual_dimension']} | "
            f"{data['binary_tree_rotations']} | {data['qrom_toffoli_estimate']} |"
        )
    lines += [
        "",
        "## Nonlinear-loop repetition",
        "",
        "The residual and Jacobian are rebuilt every Gauss-Newton iteration, so the residual "
        "state is reloaded on every postselection attempt of every iteration; `beta_k` and "
        f"`lambda_k` are recomputed per iteration. Over {NONLINEAR_ITERATIONS} IEEE 14 "
        "iterations:",
        "",
        "| epsilon | attempts/iter | total prep invocations | binary-tree RY total |",
        "| --- | --- | --- | --- |",
    ]
    for row in nonlinear_rows:
        lines.append(
            f"| {row['readout_epsilon']:g} | {row['state_prep_invocations_per_iteration']} | "
            f"{row['state_prep_invocations_total_loop']} | "
            f"{row['binary_tree_rotations_total_loop']} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "Units are primitive-invocation counts, never merged with wall-clock time. The "
        "binary-tree and QROM loaders are explicit but not asymptotically cheaper than dense "
        "loading at these sizes; no scalable state-preparation advantage is claimed.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 WP C: residual loading and repeat-cost accounting"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_phase10_residual_loading(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "command": "scripts/run_phase10_residual_loading.py " + " ".join(argv or []),
        }
    )
    print(pd.DataFrame(run["mode_rows"]).to_string(index=False))
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
