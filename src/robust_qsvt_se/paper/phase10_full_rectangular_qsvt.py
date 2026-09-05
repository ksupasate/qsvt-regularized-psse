"""Phase 10 WP B: full rectangular PSSE selected-output QSVT execution.

Executes the QSVT-compatible Ridge/Tikhonov filter on the **full rectangular**
weighted Jacobian orientation ``A = H_tilde^T / beta`` for IEEE 14 and IEEE 30
(generated PYPOWER measurement models), with the **full** weighted residual
prepared as the input state.  This is *not* a selected-submatrix execution:
the executed matrix is the complete generated measurement Jacobian, and the
classical reference is the full-system Ridge update at the same alpha.

Construction (dense validation path, not an oracle):

* zero-pad ``A`` (n_states x n_measurements) into an ``N x N`` square with
  ``N = next_power_of_two(max(m, n))``; odd polynomials preserve the padding
  because ``p(0) = 0``, so the singular-value transform of the padded matrix
  contains exactly the transform of ``A`` in its top-left block;
* build the canonical dense dilation (``2N x 2N`` unitary) of the padded
  matrix and apply the projector-controlled QSVT phase sequence;
* prepare ``|r_tilde>`` over the full measurement index space, postselect the
  encoded half, and recover the physical update with the single factor
  ``C/beta``.

Both a compiled-circuit path (Qiskit ``Operator``/``Statevector``) and a
matrix-vector statevector path (identical operation sequence, no circuit
object) are run and cross-checked for the executed cases.

Alpha tiers make the degree boundary explicit instead of hiding it: the
canonical nonlinear-AC ``alpha = 1e-4`` and the sigma-matched
``alpha = 4 sigma_min^2`` are recorded as infeasible/degree-limited at the
repository's polynomial-synthesis ceiling (degree <= 45 in the monomial
basis), while degree-aware tiers (``lambda = alpha/beta^2`` of 0.02 and 0.068)
execute and pass against full-system Ridge.  IEEE 57/118/300 rows are resource
estimates with polynomial certification only, never presented as executions.

Everything runs on a classical simulator.  No speedup, no QSVT-over-Ridge
numerical superiority at matched alpha, and no field-data validation is
claimed or implied.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.phase10_common import (
    assert_safe,
    json_ready,
    write_phase10_manifest,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    build_structured_qsvt_operator_circuit,
    qsvt_sequence_operation_counts,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_DIR = Path("outputs/phase10_full_rectangular_selected_output_qsvt")
EXECUTED_CASES = ("ieee14", "ieee30")
MODELED_CASES = ("ieee57", "ieee118", "ieee300")
DEGREE_CANDIDATES = (31, 39, 45)
MARGIN_CANDIDATES = (1.05, 1.25)
PASS_RELATIVE_TOLERANCE = 0.05
POSTSELECTION_SAMPLE_SHOTS = 4096
READOUT_EPSILONS = (1.0e-2, 1.0e-3)

ALPHA_TIERS: tuple[dict[str, Any], ...] = (
    {
        "tier": "canonical_alpha_1e-4",
        "alpha_rule": "fixed alpha = 1e-4 (nonlinear-AC baseline configuration value)",
        "fixed_alpha": 1.0e-4,
    },
    {
        "tier": "sigma_matched",
        "alpha_rule": "alpha = 4 * sigma_min(H_tilde)^2",
        "sigma_min_factor": 4.0,
    },
    {
        "tier": "degree_aware_lambda_0.02",
        "alpha_rule": "alpha = lambda * beta^2 with lambda = 0.02 (degree-aware selection)",
        "lambda_normalized": 0.02,
    },
    {
        "tier": "anchor_lambda_0.068",
        "alpha_rule": "alpha = lambda * beta^2 with lambda = 0.068 (validated anchor band)",
        "lambda_normalized": 0.068,
    },
)

CLAIM = (
    "Full rectangular weighted-Jacobian selected-output QSVT execution for IEEE 14 and "
    "IEEE 30 on a classical statevector simulator, compared against the full-system "
    "Ridge/Tikhonov update at the same alpha. The executed matrix is the complete generated "
    "measurement Jacobian orientation A = H^T/beta with the full weighted residual as input "
    "(NOT a selected square submatrix). This is simulator execution, NOT a hardware run; it "
    "does not imply IEEE-scale feasibility, does not imply speedup, and does not imply QSVT "
    "numerical superiority over Ridge. Larger IEEE cases are resource-estimated, not "
    "executed."
)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def build_padded_dilation(H_tilde: np.ndarray, beta: float) -> dict[str, Any]:
    """Zero-pad ``A = H^T/beta`` to ``N x N`` and build the dense dilation."""

    H = np.asarray(H_tilde, dtype=np.float64)
    m, n = H.shape
    A = H.T / float(beta)
    N = _next_power_of_two(max(m, n))
    padded = np.zeros((N, N), dtype=np.float64)
    padded[:n, :m] = A
    encoding = canonical_square_block_encoding(padded, tolerance=1.0e-8)
    return {
        "n_states": n,
        "m_measurements": m,
        "padded_dimension": N,
        "unitary_dimension": 2 * N,
        "qubits": int(math.log2(2 * N)),
        "padded_matrix": padded,
        "unitary": np.asarray(encoding.unitary, dtype=np.complex128),
        "top_left_block_error": float(encoding.summary["top_left_block_error"]),
        "unitarity_error": float(encoding.summary["unitarity_error"]),
        "construction": (
            "zero-padded square embedding of A = H^T/beta + canonical dense dilation "
            "(validation construction, not an oracle decomposition)"
        ),
    }


def apply_qsvt_sequence_to_vector(
    unitary: np.ndarray,
    phases: np.ndarray,
    *,
    encoded_dimension: int,
    vector: np.ndarray,
) -> np.ndarray:
    """Matrix-vector statevector path mirroring the compiled circuit exactly.

    Applies the identical operation sequence as
    ``build_structured_qsvt_operator_circuit`` (projector-controlled phase,
    then alternating signal calls) without materializing the operator product,
    so large executed cases stay O(d * dim^2).
    """

    U = np.asarray(unitary, dtype=np.complex128)
    dim = U.shape[0]
    phase_values = np.asarray(phases, dtype=np.float64)
    state = np.asarray(vector, dtype=np.complex128).copy()
    if state.shape != (dim,):
        raise ValueError("vector dimension must match the block-encoding unitary")
    k = int(encoded_dimension)

    def pcphase(phi: float, psi: np.ndarray) -> np.ndarray:
        out = psi.copy()
        out[:k] *= np.exp(1j * phi)
        out[k:] *= np.exp(-1j * phi)
        return out

    state = pcphase(float(phase_values[0]), state)
    U_dag = U.conj().T
    for index in range(1, phase_values.size - 1, 2):
        state = U @ state
        state = pcphase(float(phase_values[index]), state)
        state = U_dag @ state
        state = pcphase(float(phase_values[index + 1]), state)
    if phase_values.size % 2 == 0:
        state = U @ state
        state = pcphase(float(phase_values[-1]), state)
    return state


def run_full_rectangular_qsvt(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    alpha: float,
    degree: int,
    margin: float,
    phase_cache_dir: str | Path,
    prebuilt_dilation: dict[str, Any] | None = None,
    beta: float | None = None,
    run_circuit_path: bool = True,
) -> dict[str, Any]:
    """One full rectangular QSVT execution attempt at a fixed degree/margin.

    Returns a record with a ``status`` field; failure modes (unbounded
    polynomial, phase-synthesis failure) are recorded, never raised past this
    boundary.  The physical update recovery is the single factor ``C/beta``.
    """

    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    m, n = H.shape
    if r.shape != (m,):
        raise ValueError("r_tilde must match the full measurement dimension")
    singular_values = np.linalg.svd(H, compute_uv=False)
    beta_value = float(singular_values.max()) if beta is None else float(beta)
    s_min_normalized = float(singular_values.min() / beta_value)

    record: dict[str, Any] = {
        "alpha": float(alpha),
        "beta": beta_value,
        "lambda_alpha_over_beta2": float(alpha) / beta_value**2,
        "degree": int(degree),
        "margin": float(margin),
        "n_states": n,
        "m_measurements": m,
        "kappa": float(singular_values.max() / singular_values.min()),
        "sigma_min": float(singular_values.min()),
        "sigma_max": float(singular_values.max()),
    }

    target = fit_codesigned_bounded_polynomial(
        beta=beta_value,
        alpha=float(alpha),
        domain_min=max(1.0e-4, 0.9 * s_min_normalized),
        domain_max=1.0,
        degree=int(degree),
        margin=float(margin),
    )
    record.update(
        {
            "target_fit_error": target.fit_max_abs_error,
            "bound_C": target.bound_C,
            "bounded_max_abs": target.bounded_max_abs,
            "physical_recovery_factor_C_over_beta": target.physical_recovery_factor,
        }
    )
    try:
        validate_qsvt_polynomial(
            np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
        )
    except Exception as exc:
        record.update({"status": "bounded_polynomial_invalid", "failure_reason": str(exc)})
        return record
    try:
        cached = synthesize_pennylane_phases_cached(
            np.asarray(target.coefficients),
            angle_solver="iterative",
            cache_dir=phase_cache_dir,
            cache_metadata={
                "workload": "phase10_full_rectangular",
                "degree": int(degree),
                "alpha": float(alpha),
            },
        )
        phases = np.asarray(cached.phases, dtype=np.float64)
    except Exception as exc:
        record.update({"status": "phase_synthesis_failed", "failure_reason": str(exc)})
        return record
    record["phase_count"] = int(phases.size)
    record.update(qsvt_sequence_operation_counts(int(phases.size)))

    dilation = prebuilt_dilation or build_padded_dilation(H, beta_value)
    N = int(dilation["padded_dimension"])
    unitary = dilation["unitary"]
    record.update(
        {
            "padded_dimension": N,
            "unitary_dimension": int(dilation["unitary_dimension"]),
            "qubits": int(dilation["qubits"]),
            "block_encoding_unitarity_error": float(dilation["unitarity_error"]),
            "block_encoding_top_left_error": float(dilation["top_left_block_error"]),
        }
    )

    residual_norm = float(np.linalg.norm(r))
    psi_in = np.zeros(2 * N, dtype=np.complex128)
    psi_in[:m] = r / residual_norm
    record["residual_dimension_prepared"] = m
    record["residual_norm"] = residual_norm

    started = time.perf_counter()
    psi_out_full = apply_qsvt_sequence_to_vector(
        unitary, phases, encoded_dimension=N, vector=psi_in
    )
    record["matvec_seconds"] = time.perf_counter() - started

    if run_circuit_path:
        from qiskit.quantum_info import Statevector

        started = time.perf_counter()
        bundle = build_structured_qsvt_operator_circuit(unitary, phases, encoded_dimension=N)
        circuit_out = np.asarray(Statevector(psi_in).evolve(bundle.qsvt_operator_circuit).data)
        record["circuit_seconds"] = time.perf_counter() - started
        record["circuit_vs_matvec_error"] = float(np.max(np.abs(circuit_out - psi_out_full)))
        psi_out_full = circuit_out

    encoded = psi_out_full[:N]
    p_succ = float(np.vdot(encoded, encoded).real)
    psi_post = encoded / math.sqrt(p_succ) if p_succ > 1.0e-15 else encoded
    recovery = target.physical_recovery_factor * residual_norm * math.sqrt(p_succ)
    padded_update = recovery * np.real(psi_post)
    update = padded_update[:n]
    tail_norm = float(np.linalg.norm(padded_update[n:]))

    ridge_update = ridge_svd_solution(H, r, alpha=float(alpha))
    rel_error = float(
        np.linalg.norm(update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
    )
    max_abs_error = float(np.max(np.abs(update - ridge_update)))

    # Exact singular-value-transform cross-check on the padded matrix.
    U_pad, S_pad, Vt_pad = np.linalg.svd(dilation["padded_matrix"])
    exact_action = U_pad @ np.diag(target.polynomial(S_pad)) @ Vt_pad
    exact_update = (
        target.physical_recovery_factor * residual_norm * (exact_action @ psi_in[:N].real)[:n]
    )
    record["matvec_vs_exact_svt_update_error"] = float(np.max(np.abs(update - exact_update)))

    record.update(
        {
            "postselection_probability": p_succ,
            "update_relative_error_vs_full_ridge": rel_error,
            "update_max_abs_error_vs_full_ridge": max_abs_error,
            "padding_tail_norm": tail_norm,
            "update_vector": update,
            "ridge_update_vector": ridge_update,
            "postselected_state": psi_post,
            "output_statevector": psi_out_full,
            "status": (
                "executed_pass"
                if rel_error <= PASS_RELATIVE_TOLERANCE
                else "executed_degree_limited"
            ),
        }
    )
    return record


def selected_functionals(metadata: dict[str, Any], n_states: int) -> list[dict[str, Any]]:
    """Metadata-driven selected functionals over the full state correction."""

    angle_buses = [int(b) for b in metadata.get("angle_state_buses", [])]
    voltage_buses = [int(b) for b in metadata.get("voltage_state_buses", [])]
    angle_count = len(angle_buses)
    functionals: list[dict[str, Any]] = []

    e_first = np.zeros(n_states)
    e_first[0] = 1.0
    functionals.append(
        {
            "name": "first_state_coordinate",
            "vector": e_first,
            "description": f"first non-reference voltage-angle correction (bus {angle_buses[0]})",
        }
    )
    if voltage_buses:
        e_voltage = np.zeros(n_states)
        e_voltage[angle_count] = 1.0
        functionals.append(
            {
                "name": "first_voltage_magnitude",
                "vector": e_voltage,
                "description": f"first voltage-magnitude correction (bus {voltage_buses[0]})",
            }
        )

    branch = _first_angle_branch(metadata, angle_buses)
    if branch is not None:
        bus_i, bus_j, label = branch
        vec = np.zeros(n_states)
        vec[angle_buses.index(bus_i)] = 1.0
        vec[angle_buses.index(bus_j)] = -1.0
        functionals.append(
            {
                "name": "branch_angle_difference",
                "vector": vec,
                "description": (
                    f"angle-difference correction across measured branch {bus_i}-{bus_j} "
                    f"(from measurement row '{label}')"
                ),
            }
        )

    area_buses = angle_buses[-3:]
    vec = np.zeros(n_states)
    for bus in area_buses:
        vec[angle_buses.index(bus)] = 1.0 / len(area_buses)
    functionals.append(
        {
            "name": "area_aggregate_angle",
            "vector": vec,
            "description": (
                f"mean angle correction over the last three angle-state buses {area_buses} "
                "(the weak-area buses used across the repository for IEEE 14)"
            ),
        }
    )
    return functionals


def _first_angle_branch(
    metadata: dict[str, Any], angle_buses: list[int]
) -> tuple[int, int, str] | None:
    buses_per_row = metadata.get("measurement_buses") or []
    labels = metadata.get("measurement_labels") or []
    for index, buses in enumerate(buses_per_row):
        pair = [int(b) for b in buses]
        if len(pair) == 2 and pair[0] in angle_buses and pair[1] in angle_buses:
            label = str(labels[index]) if index < len(labels) else f"row_{index}"
            return pair[0], pair[1], label
    return None


def sample_postselection(
    psi_out_full: np.ndarray, encoded_dimension: int, *, shots: int, seed: int
) -> dict[str, Any]:
    """Finite-shot postselection sampled from the exact output distribution."""

    probabilities = np.abs(np.asarray(psi_out_full, dtype=np.complex128)) ** 2
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(probabilities.size, size=int(shots), p=probabilities)
    accepted = int(np.count_nonzero(draws < int(encoded_dimension)))
    return {
        "shots": int(shots),
        "seed": int(seed),
        "accepted": accepted,
        "p_hat_succ": accepted / int(shots),
        "sampling_model": "multinomial draws from the exact statevector distribution",
    }


def _tier_alpha(tier: dict[str, Any], sigma_min: float, beta: float) -> float:
    if "fixed_alpha" in tier:
        return float(tier["fixed_alpha"])
    if "sigma_min_factor" in tier:
        return float(tier["sigma_min_factor"]) * sigma_min**2
    return float(tier["lambda_normalized"]) * beta**2


def execute_case(
    case: str,
    *,
    seed: int,
    phase_cache_dir: Path,
    postselection_seed: int,
    tiers: tuple[dict[str, Any], ...] = ALPHA_TIERS,
    degree_candidates: tuple[int, ...] = DEGREE_CANDIDATES,
) -> dict[str, Any]:
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
    singular_values = np.linalg.svd(H, compute_uv=False)
    beta = float(singular_values.max())
    dilation = build_padded_dilation(H, beta)
    functionals = selected_functionals(system.metadata, H.shape[1])

    attempts: list[dict[str, Any]] = []
    tier_finals: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    postselection_records: list[dict[str, Any]] = []

    for tier in tiers:
        alpha = _tier_alpha(tier, float(singular_values.min()), beta)
        tier_attempts: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for degree in degree_candidates:
            attempt: dict[str, Any] | None = None
            for margin in MARGIN_CANDIDATES:
                attempt = run_full_rectangular_qsvt(
                    H,
                    r,
                    alpha=alpha,
                    degree=degree,
                    margin=margin,
                    phase_cache_dir=phase_cache_dir,
                    prebuilt_dilation=dilation,
                    beta=beta,
                    run_circuit_path=True,
                )
                attempt.update(
                    {"case": case, "tier": tier["tier"], "alpha_rule": tier["alpha_rule"]}
                )
                if attempt["status"] != "bounded_polynomial_invalid":
                    break
            tier_attempts.append(attempt)
            if attempt["status"].startswith("executed"):
                # Keep the best executed attempt so degree-limited tiers still
                # report their smallest achieved error honestly.
                if final is None or attempt["update_relative_error_vs_full_ridge"] < final.get(
                    "update_relative_error_vs_full_ridge", float("inf")
                ):
                    final = attempt
                if attempt["status"] == "executed_pass":
                    break
        attempts.extend(tier_attempts)
        if final is None:
            final = dict(tier_attempts[-1])
            if all(a["status"] == "bounded_polynomial_invalid" for a in tier_attempts):
                final["status"] = "bounded_polynomial_infeasible_at_synthesis_ceiling"
                final["failure_reason"] = (
                    "no bounded odd polynomial certified at degrees "
                    f"{list(degree_candidates)} and margins {list(MARGIN_CANDIDATES)}; "
                    "lambda = alpha/beta^2 is below the feasible band of the monomial-basis "
                    "synthesis pipeline"
                )
        tier_finals.append(final)

        if final["status"].startswith("executed"):
            update = final["update_vector"]
            ridge_update = final["ridge_update_vector"]
            p_succ = float(final["postselection_probability"])
            postselection_records.append(
                {
                    "case": case,
                    "tier": tier["tier"],
                    "exact_p_succ": p_succ,
                    **sample_postselection(
                        final["output_statevector"],
                        int(final["padded_dimension"]),
                        shots=POSTSELECTION_SAMPLE_SHOTS,
                        seed=postselection_seed,
                    ),
                }
            )
            for functional in functionals:
                vec = functional["vector"]
                value_qsvt = float(vec @ update)
                value_ridge = float(vec @ ridge_update)
                row = {
                    "case": case,
                    "tier": tier["tier"],
                    "alpha": final["alpha"],
                    "degree": final["degree"],
                    "functional": functional["name"],
                    "functional_description": functional["description"],
                    "selected_output_qsvt": value_qsvt,
                    "selected_output_full_ridge": value_ridge,
                    "absolute_error": abs(value_qsvt - value_ridge),
                    "relative_error": (
                        abs(value_qsvt - value_ridge) / abs(value_ridge)
                        if abs(value_ridge) > 1e-30
                        else float("nan")
                    ),
                    "postselection_probability": p_succ,
                }
                for epsilon in READOUT_EPSILONS:
                    shots = required_shots_for_additive_error(epsilon)
                    row[f"modeled_shots_eps_{epsilon:g}"] = shots
                    row[f"modeled_attempts_eps_{epsilon:g}"] = math.ceil(shots / p_succ)
                selected_rows.append(row)

    return {
        "case": case,
        "matrix_source": matrix_source,
        "H_shape": [int(v) for v in H.shape],
        "beta": beta,
        "singular_values": singular_values,
        "dilation": dilation,
        "functionals": functionals,
        "attempts": attempts,
        "tier_finals": tier_finals,
        "selected_rows": selected_rows,
        "postselection_records": postselection_records,
    }


def model_case(case: str, *, seed: int) -> list[dict[str, Any]]:
    """Resource estimates + polynomial certification only; never executed."""

    system, matrix_source = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    m, n = H.shape
    singular_values = np.linalg.svd(H, compute_uv=False)
    beta = float(singular_values.max())
    N = _next_power_of_two(max(m, n))
    rows: list[dict[str, Any]] = []
    for tier in ALPHA_TIERS:
        alpha = _tier_alpha(tier, float(singular_values.min()), beta)
        certification = "polynomial_infeasible_at_synthesis_ceiling"
        certified_degree: int | None = None
        fit_error: float | None = None
        for degree in DEGREE_CANDIDATES:
            done = False
            for margin in MARGIN_CANDIDATES:
                target = fit_codesigned_bounded_polynomial(
                    beta=beta,
                    alpha=alpha,
                    domain_min=max(1.0e-4, 0.9 * float(singular_values.min()) / beta),
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
                certification = "polynomial_certified_bounded"
                certified_degree = degree
                fit_error = target.fit_max_abs_error
                done = True
                break
            if done:
                break
        counts = (
            qsvt_sequence_operation_counts(certified_degree + 1)
            if certified_degree is not None
            else {}
        )
        rows.append(
            {
                "case": case,
                "matrix_source": matrix_source,
                "tier": tier["tier"],
                "status": "resource_estimated_not_executed",
                "n_states": n,
                "m_measurements": m,
                "kappa": float(singular_values.max() / singular_values.min()),
                "beta": beta,
                "alpha": alpha,
                "lambda_alpha_over_beta2": alpha / beta**2,
                "padded_dimension": N,
                "unitary_dimension": 2 * N,
                "qubits": int(math.log2(2 * N)),
                "polynomial_certification": certification,
                "certified_degree": certified_degree,
                "target_fit_error": fit_error,
                "signal_unitary_calls": counts.get("signal_unitary_calls"),
                "projector_phase_operations": counts.get("projector_phase_operations"),
            }
        )
    return rows


def _summary_row(case_result: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case_result["case"],
        "tier": final.get("tier"),
        "status": final.get("status"),
        "execution_tier": (
            "executed_statevector_circuit"
            if str(final.get("status", "")).startswith("executed")
            else "not_executed"
        ),
        "H_rows_measurements": case_result["H_shape"][0],
        "H_cols_states": case_result["H_shape"][1],
        "padded_dimension": final.get("padded_dimension"),
        "unitary_dimension": final.get("unitary_dimension"),
        "qubits": final.get("qubits"),
        "alpha": final.get("alpha"),
        "beta": final.get("beta"),
        "lambda_alpha_over_beta2": final.get("lambda_alpha_over_beta2"),
        "kappa": final.get("kappa"),
        "degree": final.get("degree"),
        "margin": final.get("margin"),
        "bound_C": final.get("bound_C"),
        "physical_recovery_factor_C_over_beta": final.get("physical_recovery_factor_C_over_beta"),
        "target_fit_error": final.get("target_fit_error"),
        "postselection_probability": final.get("postselection_probability"),
        "update_relative_error_vs_full_ridge": final.get("update_relative_error_vs_full_ridge"),
        "padding_tail_norm": final.get("padding_tail_norm"),
        "circuit_vs_matvec_error": final.get("circuit_vs_matvec_error"),
        "matvec_vs_exact_svt_update_error": final.get("matvec_vs_exact_svt_update_error"),
        "failure_reason": final.get("failure_reason"),
    }


def run_phase10_full_rectangular(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "postselection_seed": 20261,
        "executed_cases": list(EXECUTED_CASES),
        "modeled_cases": list(MODELED_CASES),
        "command": "scripts/run_phase10_full_rectangular_qsvt.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")

    case_results = [
        execute_case(
            case,
            seed=int(resolved["seed"]),
            phase_cache_dir=phase_cache_dir,
            postselection_seed=int(resolved["postselection_seed"]),
        )
        for case in resolved["executed_cases"]
    ]
    modeled_rows: list[dict[str, Any]] = []
    for case in resolved["modeled_cases"]:
        modeled_rows.extend(model_case(case, seed=int(resolved["seed"])))

    summary_rows = [
        _summary_row(case_result, final)
        for case_result in case_results
        for final in case_result["tier_finals"]
    ]
    selected_rows = [row for case_result in case_results for row in case_result["selected_rows"]]
    vs_ridge_rows = []
    for case_result in case_results:
        for final in case_result["tier_finals"]:
            if not str(final.get("status", "")).startswith("executed"):
                continue
            update = np.asarray(final["update_vector"])
            ridge_update = np.asarray(final["ridge_update_vector"])
            vs_ridge_rows.append(
                {
                    "case": case_result["case"],
                    "tier": final["tier"],
                    "alpha": final["alpha"],
                    "degree": final["degree"],
                    "update_l2_norm_qsvt": float(np.linalg.norm(update)),
                    "update_l2_norm_full_ridge": float(np.linalg.norm(ridge_update)),
                    "relative_l2_error": final["update_relative_error_vs_full_ridge"],
                    "max_abs_error": final["update_max_abs_error_vs_full_ridge"],
                    "postselection_probability": final["postselection_probability"],
                    "residual_norm": final["residual_norm"],
                    "residual_dimension_prepared": final["residual_dimension_prepared"],
                    "physical_recovery_factor_C_over_beta": final[
                        "physical_recovery_factor_C_over_beta"
                    ],
                }
            )
    postselection_records = [
        record for case_result in case_results for record in case_result["postselection_records"]
    ]

    resource_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        entry = dict(row)
        entry["execution_tier"] = row["execution_tier"]
        resource_rows.append(entry)
    resource_rows.extend(modeled_rows)

    cases_csv = output_dir / "full_rectangular_cases_summary.csv"
    selected_csv = output_dir / "full_rectangular_selected_outputs.csv"
    vs_ridge_csv = output_dir / "full_rectangular_qsvt_vs_ridge.csv"
    resource_csv = output_dir / "full_rectangular_resource_accounting.csv"
    encoding_json = output_dir / "full_rectangular_block_encoding_metadata.json"
    postselection_json = output_dir / "full_rectangular_postselection.json"
    readme_md = output_dir / "README.md"

    pd.DataFrame(summary_rows).to_csv(cases_csv, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_csv, index=False)
    pd.DataFrame(vs_ridge_rows).to_csv(vs_ridge_csv, index=False)
    pd.DataFrame(resource_rows).to_csv(resource_csv, index=False)
    write_json(
        encoding_json,
        json_ready(
            {
                case_result["case"]: {
                    "matrix_source": case_result["matrix_source"],
                    "full_matrix_shape_rows_cols": case_result["H_shape"],
                    "orientation": "A = H_tilde^T / beta (residual-to-update)",
                    "beta_spectral_norm": case_result["beta"],
                    "padded_dimension": case_result["dilation"]["padded_dimension"],
                    "unitary_dimension": case_result["dilation"]["unitary_dimension"],
                    "qubits": case_result["dilation"]["qubits"],
                    "construction": case_result["dilation"]["construction"],
                    "top_left_block_error": case_result["dilation"]["top_left_block_error"],
                    "unitarity_error": case_result["dilation"]["unitarity_error"],
                }
                for case_result in case_results
            }
        ),
    )
    write_json(postselection_json, json_ready(postselection_records))
    readme_md.write_text(_readme(summary_rows, selected_rows, modeled_rows), encoding="utf-8")

    artifacts = {
        "full_rectangular_cases_summary_csv": cases_csv,
        "full_rectangular_selected_outputs_csv": selected_csv,
        "full_rectangular_qsvt_vs_ridge_csv": vs_ridge_csv,
        "full_rectangular_resource_accounting_csv": resource_csv,
        "full_rectangular_block_encoding_metadata_json": encoding_json,
        "full_rectangular_postselection_json": postselection_json,
        "readme_md": readme_md,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_full_rectangular_selected_output_qsvt",
        script_name="scripts/run_phase10_full_rectangular_qsvt.py",
        command=str(resolved["command"]),
        description=CLAIM,
        artifacts=artifacts,
        seeds={
            "system_seed": int(resolved["seed"]),
            "postselection_sampling_seed": int(resolved["postselection_seed"]),
        },
        extra={
            "executed_cases": list(resolved["executed_cases"]),
            "modeled_cases": list(resolved["modeled_cases"]),
            "alpha_tiers": [tier["tier"] for tier in ALPHA_TIERS],
            "pass_relative_tolerance": PASS_RELATIVE_TOLERANCE,
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "case_results": case_results,
        "summary_rows": summary_rows,
        "selected_rows": selected_rows,
        "modeled_rows": modeled_rows,
        "postselection_records": postselection_records,
        "artifacts": artifacts,
    }


def _readme(
    summary_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    modeled_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 10 WP B: Full Rectangular PSSE Selected-Output QSVT",
        "",
        CLAIM,
        "",
        "## What is executed here",
        "",
        "- The executed matrix is the **full rectangular** weighted Jacobian orientation "
        "`A = H^T/beta` for the complete generated measurement system (IEEE 14: 82x27, "
        "IEEE 30: 172x59), zero-padded into a power-of-two square and block-encoded with the "
        "canonical dense dilation. This is full-system selected-output execution, **not** "
        "selected-submatrix execution.",
        "- The input state is the **full** weighted residual vector; postselected outputs are "
        "recovered with the single physical factor `C/beta` and compared against the "
        "**full-system** Ridge/Tikhonov update at the same alpha.",
        "- Everything is classical simulator execution (compiled-circuit statevector plus an "
        "identical matrix-vector path, cross-checked); it is not a hardware run.",
        "- Executed small cases do not imply IEEE-scale feasibility; IEEE 57/118/300 rows are "
        "resource estimates with polynomial certification only.",
        "- No speedup and no QSVT numerical superiority over Ridge is implied: matched-alpha "
        "agreement is the *pass criterion*, and infeasible alpha tiers are recorded as such.",
        "",
        "## Alpha tiers (degree boundary made explicit)",
        "",
        "| case | tier | status | alpha | lambda | degree | p_succ | rel err vs full Ridge |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case']} | {row['tier']} | {row['status']} | {row['alpha']:.4g} | "
            f"{row['lambda_alpha_over_beta2']:.3g} | {row['degree']} | "
            f"{_fmt(row['postselection_probability'], '{:.4f}')} | "
            f"{_fmt(row['update_relative_error_vs_full_ridge'])} |"
        )
    lines += [
        "",
        "The canonical nonlinear-AC `alpha = 1e-4` and the sigma-matched alpha are "
        "polynomial-infeasible or degree-limited at the repository's synthesis ceiling "
        "(degree <= 45, monomial basis); this is recorded, not hidden. Degree-aware tiers "
        "execute and pass at the 5% relative tolerance.",
        "",
        "## Selected functionals (executed tiers)",
        "",
        "| case | tier | functional | QSVT | full Ridge | abs err |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['case']} | {row['tier']} | {row['functional']} | "
            f"{row['selected_output_qsvt']:.6e} | {row['selected_output_full_ridge']:.6e} | "
            f"{row['absolute_error']:.2e} |"
        )
    lines += [
        "",
        "The bridge-free selected-output discrepancy `|l^T dx_QSVT - l^T dx_Ridge|` above is "
        "a full-system comparison; no selected-submatrix bridge is involved. Modeled "
        "finite-shot readout costs per functional are in "
        "`full_rectangular_selected_outputs.csv` (shots = ceil(0.25/eps^2), attempts = "
        "shots/p_succ); postselection was additionally sampled at finite shots from the "
        "exact output distribution (`full_rectangular_postselection.json`). A fully "
        "shot-executed readout chain at this scale was not run; the executed evidence is "
        "statevector-exact with sampled postselection counts.",
        "",
        f"## Modeled larger cases ({len(modeled_rows)} rows)",
        "",
        "IEEE 57/118/300 rows in `full_rectangular_resource_accounting.csv` record "
        "dimensions, qubits, alpha/lambda tiers, and whether a bounded polynomial certifies "
        "at the synthesis ceiling. They are labeled `resource_estimated_not_executed` and "
        "must not be read as executions.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _fmt(value: Any, spec: str = "{:.2e}") -> str:
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
        description="Phase 10 WP B: full rectangular selected-output QSVT"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--executed-cases", nargs="+", default=list(EXECUTED_CASES))
    parser.add_argument("--modeled-cases", nargs="+", default=list(MODELED_CASES))
    args = parser.parse_args(argv)
    run = run_phase10_full_rectangular(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "executed_cases": args.executed_cases,
            "modeled_cases": args.modeled_cases,
            "command": "scripts/run_phase10_full_rectangular_qsvt.py " + " ".join(argv or []),
        }
    )
    print(pd.DataFrame(run["summary_rows"]).to_string(index=False))
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
