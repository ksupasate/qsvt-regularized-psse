"""Closed-loop nonlinear sparse-QSVT state-update experiment (TQE reviewer-blocking).

Unlike :mod:`.nonlinear_circuit_loop` (Workstream B), which advances the nonlinear state with
the matched *full* Ridge update while the QSVT circuit is executed only as a per-iteration
diagnostic, this study *closes the loop*: the update produced by the selected experimental arm
actually drives the next nonlinear PSSE iteration,

    x_{k+1} = x_k + eta_k * dx_k^{method},

so support truncation, value quantization, the bounded-polynomial approximation, explicit QSVT
circuit execution, and finite-shot signed coordinate readout each feed back into the nonlinear
trajectory.  Seven matched arms are run, each advancing *its own* state from the identical
initial state / measurement realization / block-selection, support, regularization, damping, and
stopping policy:

* ``full_system_exact_ridge``          - full weighted Jacobian, exact Ridge (classical reference)
* ``block_full_support_ridge``         - dense selected block, Ridge (block-truncation isolation)
* ``sparse_exact_ridge``               - sparse support, exact matrix, Ridge (support isolation)
* ``sparse_quantized_ridge``           - sparse support, quantized matrix, Ridge (quantization)
* ``sparse_exact_polynomial``          - bounded polynomial matrix action (polynomial isolation)
* ``sparse_qsvt_statevector_closed_loop`` - explicit QSVT statevector circuit drives the loop
* ``sparse_qsvt_finite_shot_closed_loop`` - finite-shot signed per-coordinate readout drives it

For block-based arms only the selected block state coordinates are advanced; non-selected
coordinates are frozen.  No arm ever falls back to a different arm's update or to a classical
solve: circuit-construction, boundedness, phase-synthesis, slot-overflow, zero-postselection,
finite-shot cost-ceiling and divergence failures are retained as structured rows and stop that
arm's trajectory.

The clean stage-by-stage error decomposition (full -> block -> sparse -> quantized -> polynomial
-> statevector -> finite shots) is computed separately along the full-system reference trajectory
so every stage is evaluated at a single shared operating point.

Reuses the frozen primitives by import only: ``build_ac_nonlinear_problem`` /
``_linearized_update_system`` (per-iteration residual + weighted Jacobian rebuild),
``select_deterministic_block``, ``build_quantized_sparse_block`` / ``validate_complete_wrapper``
(sparsify + quantize + sparse block-encoding wrapper), ``fit_codesigned_bounded_polynomial`` +
the timeout-guarded phase synthesis, ``build_structured_qsvt_operator_circuit`` (statevector
QSVT), ``build_integrated_sparse_selected_output_circuit`` + ``estimate_signed_selected_output``
(signed finite-shot readout), and ``ridge_svd_solution``.  No estimator or QSVT definition is
modified.  This is a small-scale simulator feasibility experiment: no hardware execution, no
full-state quantum PSSE, and no quantum speedup or advantage is claimed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.experiments.iterative_ac import (
    _linearized_update_system,
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    build_quantized_sparse_block,
    validate_complete_wrapper,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.physical_alignment.nonlinear_ac import build_problem_config
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint
from robust_qsvt_se.reviewer_blocking.common import (
    atomic_write_csv,
    atomic_write_json,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.tqe_extensions.common import CLAIM_BOUNDARY, EVIDENCE_TIERS, load_yaml_config
from robust_qsvt_se.tqe_extensions.degree_lambda_scaling import _synthesize_guarded

STUDY_ID = "tqe_closed_loop_nonlinear_update_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/nonlinear_closed_loop_qsvt")
DEFAULT_CONFIG_PATH = Path("configs/tqe_closed_loop_nonlinear_update.yaml")
PHASE_CONVENTION = "pennylane_qsvt_pcphase_u_udagger_real_top_left"

ARM_FULL = "full_system_exact_ridge"
ARM_BLOCK = "block_full_support_ridge"
ARM_SPARSE = "sparse_exact_ridge"
ARM_QUANT = "sparse_quantized_ridge"
ARM_POLY = "sparse_exact_polynomial"
ARM_STATEVECTOR = "sparse_qsvt_statevector_closed_loop"
ARM_FINITE_SHOT = "sparse_qsvt_finite_shot_closed_loop"

BLOCK_ARMS = (ARM_BLOCK, ARM_SPARSE, ARM_QUANT, ARM_POLY, ARM_STATEVECTOR, ARM_FINITE_SHOT)
CIRCUIT_ARMS = (ARM_STATEVECTOR, ARM_FINITE_SHOT)

# Claim-boundary run classification (task section 9.5); never collapsed to pass/fail.
CLASS_CONVERGED_ACCURATE = "closed_loop_converged_accurate"
CLASS_CONVERGED_INACCURATE = "closed_loop_converged_inaccurate"
CLASS_STALLED = "closed_loop_stalled"
CLASS_DIVERGED = "closed_loop_diverged"
CLASS_CIRCUIT_FAILED = "circuit_construction_failed"
CLASS_PHASE_FAILED = "phase_synthesis_failed"
CLASS_FINITE_SHOT_CEILING = "finite_shot_cost_ceiling"
CLASS_ZERO_POSTSELECTION = "zero_postselection_acceptance"
CLASS_UNSUPPORTED = "unsupported_configuration"


# --------------------------------------------------------------------------- operating point


@dataclass(slots=True)
class BlockOperatingPoint:
    """The matched sparse-QSVT operating point for one (state, block) at one iteration."""

    rows: np.ndarray
    cols: np.ndarray
    block_dense: np.ndarray
    block_sparsified: np.ndarray
    block_quantized: np.ndarray
    residual_block: np.ndarray
    residual_block_norm: float
    beta: float
    mu: float
    slots: int
    alpha_k: float
    normalized_lambda: float
    contraction_c: float
    physical_scale: float
    degree: int
    coefficients: np.ndarray
    domain_min: float
    uniform_fit_error: float
    bounded_max_abs: float
    bounded_ok: bool
    block_rank: int
    kappa_block: float
    sparsification_fro_error: float
    max_quantization_error: float
    support_nnz: int
    wrapper_unitary: np.ndarray
    wrapper_reconstruction_error: float
    phases: np.ndarray | None = None
    phase_status: str = "not_attempted"
    failure_code: str = ""
    failure_stage: str = ""
    failure_message: str = ""


def build_block_operating_point(
    matrix: np.ndarray,
    residual: np.ndarray,
    settings: dict[str, Any],
    cache_dir: Path,
    *,
    need_phases: bool,
) -> BlockOperatingPoint:
    """Extract the selected block and build its matched quantized sparse-QSVT operating point.

    Any structural failure (rank collapse, wrapper reconstruction, unbounded target, phase
    synthesis) is recorded in ``failure_code``/``failure_stage`` and returned; the caller retains
    it and stops that arm.  No fallback is ever substituted.
    """

    block_size = int(settings["block_size"])
    magnitude_bits = int(settings["magnitude_bits"])
    lambda_target = float(settings["lambda_target"])
    degree = int(settings["degree"])
    margin = float(settings.get("margin", 1.05))
    bound_tol = float(settings.get("bound_tolerance", 2.0e-3))
    domain_floor = float(settings.get("domain_min_floor", 1.0e-4))
    timeout_s = float(settings.get("phase_synthesis_timeout_seconds", 60.0))

    block_dense, rblock, rows, cols = select_deterministic_block(
        matrix, residual, row_count=block_size, col_count=block_size
    )
    rblock = np.asarray(rblock, dtype=np.float64)
    rblock_norm = float(np.linalg.norm(rblock))

    def failed(stage: str, message: str) -> BlockOperatingPoint:
        return BlockOperatingPoint(
            rows=np.asarray(rows, dtype=np.int64),
            cols=np.asarray(cols, dtype=np.int64),
            block_dense=np.asarray(block_dense, dtype=np.float64),
            block_sparsified=np.zeros_like(block_dense),
            block_quantized=np.zeros_like(block_dense),
            residual_block=rblock,
            residual_block_norm=rblock_norm,
            beta=float("nan"),
            mu=float("nan"),
            slots=-1,
            alpha_k=float("nan"),
            normalized_lambda=float("nan"),
            contraction_c=float("nan"),
            physical_scale=float("nan"),
            degree=degree,
            coefficients=np.zeros(degree + 1),
            domain_min=float("nan"),
            uniform_fit_error=float("nan"),
            bounded_max_abs=float("nan"),
            bounded_ok=False,
            block_rank=-1,
            kappa_block=float("nan"),
            sparsification_fro_error=float("nan"),
            max_quantization_error=float("nan"),
            support_nnz=-1,
            wrapper_unitary=np.zeros((1, 1), dtype=np.complex128),
            wrapper_reconstruction_error=float("nan"),
            failure_code=stage,
            failure_stage=stage,
            failure_message=message[:200],
        )

    if rblock_norm <= 0.0:
        return failed("nonzero_residual_norm_failure", "selected block residual is zero")

    qblock = build_quantized_sparse_block(block_dense, magnitude_bits=magnitude_bits)
    quantized = np.asarray(qblock.quantized, dtype=np.float64)
    if np.count_nonzero(quantized) == 0:
        return failed("empty_quantized_support_failure", "quantized support is empty")

    try:
        wrapper = validate_complete_wrapper(
            qblock, encode_transpose=True, transpile_circuit=False
        )
    except Exception as exc:  # slot overflow / reconstruction / unitarity failure
        return failed(CLASS_CIRCUIT_FAILED, f"wrapper build failed: {exc}")
    if wrapper.top_left_reconstruction_error > 1.0e-6:
        return failed(
            CLASS_CIRCUIT_FAILED,
            f"wrapper reconstruction error {wrapper.top_left_reconstruction_error:.3e}",
        )

    beta = float(wrapper.normalization_factor)
    mu = float(qblock.mu)
    slots = int(wrapper.slots)
    singular = np.linalg.svd(quantized, compute_uv=False)
    positive = singular[singular > 1.0e-10]
    rank = int(positive.size)
    if rank == 0:
        return failed("rank_failure", "quantized block has no positive singular values")
    kappa = float(positive.max() / positive.min())
    domain_min = float(np.clip(0.9 * positive.min() / beta, domain_floor, 0.999))
    alpha_k = lambda_target * beta**2

    try:
        target = fit_codesigned_bounded_polynomial(
            beta=beta,
            alpha=alpha_k,
            domain_min=domain_min,
            domain_max=1.0,
            degree=degree,
            margin=margin,
        )
    except Exception as exc:
        return failed("polynomial_fit_failure", f"bounded fit failed: {exc}")

    bounded_max = float(target.bounded_max_abs)
    bounded_ok = bool(bounded_max <= 1.0 + bound_tol)
    contraction_c = float(target.bound_C)
    coefficients = np.asarray(target.coefficients, dtype=np.float64)

    op = BlockOperatingPoint(
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        block_dense=np.asarray(block_dense, dtype=np.float64),
        block_sparsified=np.asarray(qblock.sparsified, dtype=np.float64),
        block_quantized=quantized,
        residual_block=rblock,
        residual_block_norm=rblock_norm,
        beta=beta,
        mu=mu,
        slots=slots,
        alpha_k=alpha_k,
        normalized_lambda=alpha_k / beta**2,
        contraction_c=contraction_c,
        physical_scale=contraction_c / beta * rblock_norm,
        degree=degree,
        coefficients=coefficients,
        domain_min=domain_min,
        uniform_fit_error=float(target.fit_max_abs_error),
        bounded_max_abs=bounded_max,
        bounded_ok=bounded_ok,
        block_rank=rank,
        kappa_block=kappa,
        sparsification_fro_error=float(qblock.sparsification_fro_error),
        max_quantization_error=float(qblock.max_quantization_error),
        support_nnz=int(np.count_nonzero(quantized)),
        wrapper_unitary=np.asarray(wrapper.unitary, dtype=np.complex128),
        wrapper_reconstruction_error=float(wrapper.top_left_reconstruction_error),
    )

    if need_phases:
        if not bounded_ok:
            op.failure_code = "boundedness_failure"
            op.failure_stage = "boundedness"
            op.failure_message = f"bounded_max_abs {bounded_max:.4g} exceeds 1+{bound_tol}"
            op.phase_status = "skipped_unbounded"
            return op
        phases, status = _synthesize_guarded(
            coefficients,
            angle_solver="iterative",
            cache_dir=cache_dir,
            meta={"study_id": STUDY_ID, "degree": degree, "beta": round(beta, 6)},
            timeout_s=timeout_s,
        )
        op.phase_status = status
        if phases is None or status != "synthesized":
            op.failure_code = CLASS_PHASE_FAILED
            op.failure_stage = "phase_synthesis"
            op.failure_message = f"phase synthesis status {status}"
            return op
        if phases.size != degree + 1:
            op.failure_code = "phase_count_mismatch"
            op.failure_stage = "phase_synthesis"
            op.failure_message = f"phase count {phases.size} != degree+1 {degree + 1}"
            return op
        op.phases = phases
    return op


# --------------------------------------------------------------------------- stage updates


def full_ridge_update(matrix: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    return ridge_svd_solution(matrix, residual, alpha=alpha)


def dense_block_ridge_update(op: BlockOperatingPoint, alpha: float) -> np.ndarray:
    return ridge_svd_solution(op.block_dense, op.residual_block, alpha=alpha)


def sparse_ridge_update(op: BlockOperatingPoint, alpha: float) -> np.ndarray:
    return ridge_svd_solution(op.block_sparsified, op.residual_block, alpha=alpha)


def quantized_ridge_update(op: BlockOperatingPoint, alpha: float) -> np.ndarray:
    return ridge_svd_solution(op.block_quantized, op.residual_block, alpha=alpha)


def exact_polynomial_update(op: BlockOperatingPoint) -> np.ndarray:
    """Bounded polynomial matrix action on the quantized block (no circuit).

    Uses the ``A = H_q^T / beta`` block-encoding SVD convention identical to the statevector
    path, then the single physical rescale ``C/beta`` and the residual norm.
    """

    normalized = op.block_quantized.T / op.beta
    left, singular, right_t = np.linalg.svd(normalized, full_matrices=False)
    poly = Polynomial(op.coefficients)
    residual_unit = op.residual_block / op.residual_block_norm
    action = left @ (poly(singular) * (right_t @ residual_unit))
    return op.physical_scale * action


TRANSPILE_BASIS = ("rz", "ry", "rx", "cx")
TRANSPILE_OPT_LEVEL = 1
TRANSPILE_SEED = 20260722
TRANSPILE_COUPLING_ASSUMPTION = "all_to_all_no_coupling_map"
TRANSPILE_ROUTING_INCLUDED = False


def circuit_resource_levels(
    circuit: Any,
    *,
    label: str,
    qsvt_degree: int | None = None,
    signal_applications: int | None = None,
    phase_applications: int | None = None,
    includes_residual_state_preparation: bool = False,
    includes_functional_preparation: bool = False,
    includes_interference_readout: bool = False,
    includes_direct_postselection: bool = False,
) -> dict[str, Any]:
    """Report resource counts at every abstraction level for one circuit (Issue B).

    The QSVT circuits are assembled from opaque ``unitary`` blocks (the sparse block-encoding
    wrapper and the projector-controlled phases).  Under ``transpile(..., optimization_level=0)``
    to the Aer *statevector* target those blocks are preserved, so the "logical" operation count
    is NOT a primitive gate count.  This routine additionally decomposes and transpiles to a
    concrete single/two-qubit basis so the primitive basis-operation cost and any residual opaque
    instructions are made explicit.  No backend target or coupling map is supplied, so routing is
    not included and these are not device-specific hardware gate counts.
    """

    from qiskit import transpile

    raw = {str(k): int(v) for k, v in circuit.count_ops().items()}
    decomposed = circuit.decompose()
    dec = {str(k): int(v) for k, v in decomposed.count_ops().items()}
    transpiled = transpile(
        circuit,
        basis_gates=list(TRANSPILE_BASIS),
        optimization_level=TRANSPILE_OPT_LEVEL,
        seed_transpiler=TRANSPILE_SEED,
    )
    tb = {str(k): int(v) for k, v in transpiled.count_ops().items()}
    allowed = set(TRANSPILE_BASIS) | {"measure", "barrier"}
    opaque_remaining = {k: v for k, v in tb.items() if k not in allowed}
    return {
        "circuit_type": label,
        "n_qubits": int(circuit.num_qubits),
        "n_clbits": int(circuit.num_clbits),
        "qsvt_degree": int(qsvt_degree) if qsvt_degree is not None else -1,
        "signal_unitary_applications": (
            int(signal_applications) if signal_applications is not None else -1
        ),
        "projector_phase_applications": (
            int(phase_applications) if phase_applications is not None else -1
        ),
        "logical_operations": int(sum(raw.values())),
        "untranspiled_sdk_operations": int(sum(raw.values())),
        "logical_depth": int(circuit.depth()),
        "logical_op_breakdown": ";".join(f"{k}:{v}" for k, v in sorted(raw.items())),
        "untranspiled_measurements": int(raw.get("measure", 0)),
        "untranspiled_controlled_phase_operations": int(
            sum(v for k, v in raw.items() if "PCPhase" in k or "pcphase" in k)
        ),
        "untranspiled_multi_controlled_operations": int(
            sum(v for k, v in raw.items() if k in {"ccx", "mcx"})
        ),
        "untranspiled_custom_or_opaque_operations": int(
            sum(
                v
                for k, v in raw.items()
                if k == "unitary" or "PCPhase" in k or "pcphase" in k
                or "sparse_BE_wrapper" in k
            )
        ),
        "decomposed_operations": int(sum(dec.values())),
        "transpiled_basis_gates": int(sum(tb.values())),
        "transpiled_depth": int(transpiled.depth()),
        "transpiled_cx": int(tb.get("cx", 0)),
        "transpiled_basis": "+".join(TRANSPILE_BASIS),
        "transpiler_opt_level": TRANSPILE_OPT_LEVEL,
        "transpiler_seed": TRANSPILE_SEED,
        "coupling_map_assumption": TRANSPILE_COUPLING_ASSUMPTION,
        "routing_included": TRANSPILE_ROUTING_INCLUDED,
        "backend_target": "none",
        "opaque_instructions_remaining": int(sum(opaque_remaining.values())),
        "includes_residual_state_preparation": bool(includes_residual_state_preparation),
        "includes_sparse_block_encoding": True,
        "includes_qsvt_signal_phase_sequence": True,
        "includes_functional_preparation": bool(includes_functional_preparation),
        "includes_interference_readout": bool(includes_interference_readout),
        "includes_postselection_registers": True,
        "includes_direct_postselection": bool(includes_direct_postselection),
    }


def statevector_update(op: BlockOperatingPoint) -> dict[str, Any]:
    """Execute the explicit QSVT statevector circuit; recover the block update + resources.

    The reported ``logical_operations`` is the count of high-level ``unitary`` blocks in the QSVT
    operator circuit (one projector-controlled phase or one signal-unitary application per block),
    NOT a decomposed primitive-gate count.  See :func:`circuit_resource_levels` for the transpiled
    all-to-all basis-decomposition characterisation.
    """

    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
    from robust_qsvt_se.qsvt.sparse_integrated_chain import _resource_counts, compile_for_aer

    n = op.block_quantized.shape[1]
    bundle = build_structured_qsvt_operator_circuit(
        op.wrapper_unitary, op.phases, encoded_dimension=n
    )
    initial = np.zeros(op.wrapper_unitary.shape[0], dtype=np.complex128)
    initial[:n] = op.residual_block / op.residual_block_norm
    evolved = Statevector(initial).evolve(bundle.qsvt_operator_circuit).data
    encoded = np.asarray(evolved[:n], dtype=np.complex128)
    p_post = float(np.vdot(encoded, encoded).real)
    dx_block = op.physical_scale * np.real(encoded)
    compiled, _sim = compile_for_aer(bundle.qsvt_operator_circuit)
    res = _resource_counts(compiled)
    return {
        "dx_block": dx_block,
        "postselection_probability": p_post,
        "total_qubits": round(math.log2(op.wrapper_unitary.shape[0])),
        # ``logical_operations`` == high-level unitary-block count (== _resource_counts gate_count
        # at the statevector abstraction), explicitly NOT primitive gates.
        "logical_operations": int(res["gate_count"]),
        "logical_depth": int(res["depth"]),
        "gate_count": int(res["gate_count"]),  # legacy alias (logical level)
        "depth": int(res["depth"]),
        "signal_unitary_calls": int(bundle.block_encoding_gate_count),
        "projector_phase_calls": int(bundle.phase_gate_count),
        "statevector_dim": int(op.wrapper_unitary.shape[0]),
    }


def _finite_shot_config(op: BlockOperatingPoint, settings: dict[str, Any]) -> Any:
    from robust_qsvt_se.qsvt.sparse_integrated_chain import SparseIntegratedQSVTConfig

    n = op.block_quantized.shape[1]
    return SparseIntegratedQSVTConfig(
        configuration_id=f"{STUDY_ID}_block4x4",
        case_name=str(settings["case_name"]),
        case_source=str(settings["case_source"]),
        matrix_source="per_iteration_selected_block_quantized",
        matrix_path=Path("in_memory"),
        residual_path=Path("in_memory"),
        phase_path=Path("in_memory"),
        matrix_fingerprint=stable_array_fingerprint(op.block_quantized),
        residual_fingerprint=stable_array_fingerprint(
            np.ascontiguousarray(op.residual_block, dtype=np.float64)
        ),
        matrix_shape=(n, n),
        matrix_value_bits=int(settings["magnitude_bits"]),
        alpha=op.alpha_k,
        beta=op.beta,
        normalized_lambda=op.normalized_lambda,
        contraction_c=op.contraction_c,
        polynomial_degree=op.degree,
        phase_convention=PHASE_CONVENTION,
        selected_output_name="coordinate_readout",
        selected_output_vector=tuple(float(x) for x in np.eye(n)[0]),
        shot_counts=(int(settings["finite_shot_budget"]),),
        seeds=(int(settings["finite_shot_sampler_seed"]),),
        selected_rows=tuple(int(x) for x in op.rows),
        selected_columns=tuple(int(x) for x in op.cols),
    )


def finite_shot_update(
    op: BlockOperatingPoint, settings: dict[str, Any], *, shots: int, sampler_seed: int
) -> dict[str, Any]:
    """Reconstruct the block update coordinate-by-coordinate via signed finite-shot readout.

    One functional-specific circuit per block coordinate ``ell_j = e_j``; the assembled vector of
    signed estimates drives the next iteration.  Statevector amplitudes, exact Ridge, and exact
    polynomial values are never substituted for a noisy or failed coordinate estimate.
    """

    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        _direct_postselection_estimate,
        build_integrated_sparse_selected_output_circuit,
        compile_for_aer,
        estimate_signed_selected_output,
        sample_aer_counts,
    )

    n = op.block_quantized.shape[1]
    config = _finite_shot_config(op, settings)
    coordinates: list[dict[str, Any]] = []
    dx_block = np.zeros(n, dtype=np.float64)
    # Non-overlapping accounting (Issue A).  One *functional query* per block coordinate drives the
    # state update through the readout/interference branch; a *direct-postselection diagnostic*
    # circuit is also sampled per coordinate but never enters dx_block.  Both branches are physical
    # sampling calls of `shots` shots each, so the total-attempted invariant is
    # total_attempted_shots == sampling_calls * shots_per_sampling_call.
    readout_signal_attempted = 0
    diagnostic_attempted = 0
    interference_accepted = 0
    postselection_accepted = 0
    sampling_calls = 0
    circuit_hashes: list[str] = []
    for j in range(n):
        functional = np.eye(n)[j]
        circuit_digest = hashlib.sha256()
        for values in (
            op.block_quantized,
            op.residual_block,
            np.asarray(op.phases, dtype=np.float64),
            functional,
        ):
            array = np.ascontiguousarray(values, dtype=np.float64)
            circuit_digest.update(str(array.shape).encode("ascii"))
            circuit_digest.update(array.tobytes())
        functional_circuit_fingerprint = circuit_digest.hexdigest()
        bundle = build_integrated_sparse_selected_output_circuit(
            config,
            matrix=op.block_quantized,
            residual=op.residual_block,
            selected_functional=functional,
            phases=op.phases,
        )
        compiled, simulator = compile_for_aer(bundle.circuit)
        counts = sample_aer_counts(
            compiled, simulator, shots=int(shots), seed=int(sampler_seed) + 101 * j
        )
        physical_scale = op.contraction_c / op.beta * op.residual_block_norm  # ell_norm = 1
        estimate = estimate_signed_selected_output(counts, physical_scale=physical_scale)
        direct_compiled, direct_sim = compile_for_aer(bundle.direct_postselection_circuit)
        direct_counts = sample_aer_counts(
            direct_compiled, direct_sim, shots=int(shots), seed=int(sampler_seed) + 101 * j + 7
        )
        post_accepted, post_prob = _direct_postselection_estimate(direct_counts)
        readout_shots = int(sum(counts.values()))
        diagnostic_shots = int(sum(direct_counts.values()))
        readout_accepted = int(estimate["readout_accepted"])
        dx_block[j] = float(estimate["selected_output_estimate"])
        readout_signal_attempted += readout_shots
        diagnostic_attempted += diagnostic_shots
        interference_accepted += readout_accepted
        postselection_accepted += int(post_accepted)
        sampling_calls += 2  # one readout-branch call + one direct-postselection call
        circuit_hashes.append(stable_array_fingerprint(functional))
        coordinates.append(
            {
                "coordinate": int(j),
                "global_column": int(op.cols[j]),
                "readout_sampler_seed": int(sampler_seed) + 101 * j,
                "diagnostic_sampler_seed": int(sampler_seed) + 101 * j + 7,
                "readout_attempted_shots": readout_shots,
                "diagnostic_attempted_shots": diagnostic_shots,
                "attempted_shots": readout_shots,  # signal branch (kept for continuity)
                "postselection_accepted_shots": int(post_accepted),
                "interference_branch_accepted_shots": readout_accepted,
                "interference_acceptance_probability": float(
                    estimate["interference_acceptance_probability"]
                ),
                "direct_postselection_probability": float(post_prob),
                "signed_raw_estimate": float(estimate["signed_overlap_estimate"]),
                "recovery_scale": float(physical_scale),
                "recovered_coordinate_update": float(estimate["selected_output_estimate"]),
                "sampling_standard_error": float(estimate["analytic_standard_error"]),
                "functional_fingerprint": circuit_hashes[-1],
                "functional_circuit_fingerprint": functional_circuit_fingerprint,
            }
        )
    total_attempted = readout_signal_attempted + diagnostic_attempted
    # Invariant: every sampling call drew exactly `shots` shots.
    if total_attempted != sampling_calls * int(shots):
        raise RuntimeError(
            f"shot-accounting invariant violated: {total_attempted} != "
            f"{sampling_calls} * {shots}"
        )
    return {
        "dx_block": dx_block,
        "coordinates": coordinates,
        "functional_queries": int(n),
        "unique_functional_circuits": int(
            len({c["functional_circuit_fingerprint"] for c in coordinates})
        ),
        "physical_circuit_executions": int(sampling_calls),
        "sampling_calls": int(sampling_calls),
        "shots_per_sampling_call": int(shots),
        "readout_signal_attempted_shots": int(readout_signal_attempted),
        "diagnostic_postselection_attempted_shots": int(diagnostic_attempted),
        "total_attempted_shots": int(total_attempted),
        "interference_accepted_shots": int(interference_accepted),
        "postselection_accepted_shots": int(postselection_accepted),
        # Legacy aliases kept so downstream readers do not silently break.
        "coordinate_query_count": int(n),
        "total_accepted_shots": int(interference_accepted),
        "total_circuit_executions": int(sampling_calls),
        "min_interference_acceptance": float(
            min(c["interference_acceptance_probability"] for c in coordinates)
        ),
    }


# --------------------------------------------------------------------------- closed loop driver


def _state_metrics(problem: Any, state: np.ndarray, angle_count: int) -> dict[str, float]:
    error = np.asarray(state, dtype=np.float64) - np.asarray(problem.true_state, dtype=np.float64)
    n_true = float(np.linalg.norm(problem.true_state))
    return {
        "state_rmse": float(np.linalg.norm(error) / max(n_true, 1.0e-30)),
        "angle_rmse": float(np.sqrt(np.mean(error[:angle_count] ** 2))),
        "voltage_rmse": float(np.sqrt(np.mean(error[angle_count:] ** 2))),
        "max_state_error": float(np.max(np.abs(error))),
    }


def _partition_state_metrics(
    problem: Any, state: np.ndarray, selected_columns: np.ndarray
) -> dict[str, float]:
    """Return true coordinate-wise RMSE for the selected and frozen state partitions."""

    error = np.asarray(state, dtype=np.float64) - np.asarray(problem.true_state, dtype=np.float64)
    selected = np.asarray(selected_columns, dtype=np.int64)
    frozen = np.setdiff1d(np.arange(error.size, dtype=np.int64), selected, assume_unique=False)
    return {
        "selected_coordinate_rmse": (
            float(np.sqrt(np.mean(error[selected] ** 2))) if selected.size else float("nan")
        ),
        "frozen_coordinate_rmse": (
            float(np.sqrt(np.mean(error[frozen] ** 2))) if frozen.size else float("nan")
        ),
    }


def _array_json(values: np.ndarray) -> str:
    return json.dumps(
        [float(value) for value in np.asarray(values, dtype=np.float64)], separators=(",", ":")
    )


def _apply_block_update(
    state: np.ndarray, cols: np.ndarray, dx_block: np.ndarray, damping: float, min_voltage: float,
    angle_count: int,
) -> np.ndarray:
    updated = state.copy()
    updated[cols] = updated[cols] + damping * np.asarray(dx_block, dtype=np.float64)
    updated[angle_count:] = np.maximum(updated[angle_count:], min_voltage)
    return updated


def drive_arm(
    arm: str,
    problem: Any,
    settings: dict[str, Any],
    block_settings: dict[str, Any],
    loop_settings: dict[str, Any],
    cache_dir: Path,
    *,
    scenario_id: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run one closed-loop arm; return (iteration rows, coordinate rows, run summary)."""

    iteration_cfg = settings["iteration"]
    max_iterations = int(iteration_cfg["max_iterations"])
    update_tol = float(iteration_cfg["update_tolerance"])
    residual_tol = float(iteration_cfg["residual_tolerance"])
    damping = float(iteration_cfg["damping"])
    max_update_norm = float(iteration_cfg.get("max_update_norm", 1.0e6))
    residual_growth_limit = float(iteration_cfg.get("residual_growth_limit", 1.0e6))
    min_voltage = float(settings["linearization"]["min_voltage_magnitude"])
    alpha_full = float(settings["fixed_alpha"])
    angle_count = len(problem.case.angle_state_buses)
    shots = int(loop_settings["finite_shot_budget"])
    sampler_seed = int(loop_settings["finite_shot_sampler_seed"])
    shot_ceiling = int(loop_settings.get("max_total_shots_per_run", 10**9))
    accurate_threshold = float(loop_settings.get("accurate_state_rmse_threshold", 2.0e-3))
    stalled_update_norm = float(loop_settings.get("stalled_update_norm", 1.0e-6))

    state = problem.initial_state.copy()
    rows: list[dict[str, Any]] = []
    coord_rows: list[dict[str, Any]] = []
    converged = False
    failed = False
    failure_code = ""
    failure_stage = ""
    failure_message = ""
    cumulative_shots = 0
    cumulative_circuits = 0
    functional_circuit_fingerprints: set[str] = set()
    last_selected_columns = np.asarray([], dtype=np.int64)
    started = time.perf_counter()

    for iteration in range(max_iterations):
        system, weighted_residual_before = _linearized_update_system(problem, state)
        matrix = np.asarray(system.H_tilde, dtype=np.float64)
        residual = np.asarray(system.r_tilde, dtype=np.float64)
        metrics = _state_metrics(problem, state, angle_count)
        raw_residual_norm = float(np.linalg.norm(residual))

        row: dict[str, Any] = {
            "arm": arm,
            "case_name": settings["case_name"],
            "scenario": scenario_id,
            "seed": int(seed),
            "iteration": int(iteration),
            "state_fingerprint": stable_array_fingerprint(state),
            "state_vector_json": _array_json(state),
            "state_after_update_json": "",
            "applied_update_vector_json": "",
            "residual_norm": raw_residual_norm,
            "weighted_residual_norm": float(weighted_residual_before),
            "damping": damping,
            "alpha_full": alpha_full,
            **metrics,
            "selected_global_columns": "",
            "selected_measurement_rows": "",
            "selected_coordinate_rmse": float("nan"),
            "frozen_coordinate_rmse": float("nan"),
            "block_rank": -1,
            "kappa_block": float("nan"),
            "alpha_k": float("nan"),
            "beta_k": float("nan"),
            "contraction_C_k": float("nan"),
            "normalized_lambda": float("nan"),
            "degree": int(block_settings["degree"]),
            "support_nnz": -1,
            "polynomial_fit_error": float("nan"),
            "bounded_max_abs": float("nan"),
            "bounded_ok": None,
            "phase_synthesis_status": "not_applicable",
            "phase_count": 0,
            "step_norm": float("nan"),
            "full_ridge_agreement_error": float("nan"),
            "full_ridge_agreement_abs_error": float("nan"),
            "block_matched_ridge_error": float("nan"),
            "block_matched_ridge_abs_error": float("nan"),
            "selected_output_value": float("nan"),
            "selected_output_benchmark_error": float("nan"),
            "postselection_probability": float("nan"),
            # Non-overlapping finite-shot accounting (Issue A).  A functional query is one signed
            # coordinate readout; it issues two physical sampling calls (readout branch + direct-
            # postselection diagnostic), each of `shots_per_sampling_call` shots.
            "functional_queries_iter": 0,
            "unique_functional_circuits_iter": 0,
            "physical_circuit_executions_iter": 0,
            "sampling_calls_iter": 0,
            "shots_per_sampling_call": 0,
            "readout_signal_attempted_shots_iter": 0,
            "diagnostic_attempted_shots_iter": 0,
            "total_attempted_shots_iter": 0,
            "interference_accepted_shots_iter": 0,
            "postselection_accepted_shots_iter": 0,
            "coordinate_query_count": 0,  # legacy alias of functional_queries_iter
            "shots_attempted_iter": 0,  # legacy alias of total_attempted_shots_iter (both branches)
            "shots_accepted_iter": 0,  # legacy alias of interference_accepted_shots_iter
            "cumulative_shots": cumulative_shots,  # cumulative TOTAL attempted (both branches)
            "cumulative_circuit_executions": cumulative_circuits,  # cumulative physical executions
            "min_interference_acceptance": float("nan"),
            "circuit_qubits": -1,
            # Logical (high-level unitary-block) circuit resources; NOT primitive gates.  The
            # transpiled all-to-all basis characterisation lives in circuit_resource_levels.csv.
            "circuit_logical_operations": -1,
            "circuit_logical_depth": -1,
            "circuit_signal_unitary_calls": -1,
            "circuit_projector_phase_calls": -1,
            "statevector_dim": -1,
            "evidence_tier": "",
            "convergence_status": "running",
            "failure_code": "",
            "failure_stage": "",
            "failure_message": "",
        }

        truth_update_block = None  # benchmark reference on block coords
        method_update_full = np.zeros_like(state, dtype=np.float64)
        if arm == ARM_FULL:
            _block, _residual, diagnostic_rows, diagnostic_cols = select_deterministic_block(
                matrix,
                residual,
                row_count=int(block_settings["block_size"]),
                col_count=int(block_settings["block_size"]),
            )
            last_selected_columns = np.asarray(diagnostic_cols, dtype=np.int64)
            row["selected_global_columns"] = ",".join(
                str(int(c)) for c in last_selected_columns
            )
            row["selected_measurement_rows"] = ",".join(
                str(int(r)) for r in np.asarray(diagnostic_rows, dtype=np.int64)
            )
            row.update(_partition_state_metrics(problem, state, last_selected_columns))
            dx_full = full_ridge_update(matrix, residual, alpha_full)
            method_update_full = np.asarray(dx_full, dtype=np.float64)
            step_norm = float(np.linalg.norm(dx_full))
            row["step_norm"] = step_norm
            row["evidence_tier"] = "classical_full_system_ridge"
            next_state = state + damping * dx_full
            next_state[angle_count:] = np.maximum(next_state[angle_count:], min_voltage)
        else:
            need_phases = arm in CIRCUIT_ARMS
            op = build_block_operating_point(
                matrix, residual, block_settings, cache_dir, need_phases=need_phases
            )
            row["selected_global_columns"] = ",".join(str(int(c)) for c in op.cols)
            row["selected_measurement_rows"] = ",".join(str(int(r)) for r in op.rows)
            last_selected_columns = np.asarray(op.cols, dtype=np.int64)
            row.update(_partition_state_metrics(problem, state, last_selected_columns))
            row["block_rank"] = op.block_rank
            row["kappa_block"] = op.kappa_block
            row["alpha_k"] = op.alpha_k
            row["beta_k"] = op.beta
            row["contraction_C_k"] = op.contraction_c
            row["normalized_lambda"] = op.normalized_lambda
            row["support_nnz"] = op.support_nnz
            row["polynomial_fit_error"] = op.uniform_fit_error
            row["bounded_max_abs"] = op.bounded_max_abs
            row["bounded_ok"] = op.bounded_ok
            row["phase_synthesis_status"] = op.phase_status
            row["phase_count"] = int(op.phases.size) if op.phases is not None else 0

            if op.failure_code:
                failed = True
                failure_code = op.failure_code
                failure_stage = op.failure_stage
                failure_message = op.failure_message
                row["failure_code"] = failure_code
                row["failure_stage"] = failure_stage
                row["failure_message"] = failure_message
                row["convergence_status"] = "failed"
                row["evidence_tier"] = EVIDENCE_TIERS["EXACT_MATRIX_ACTION"]
                rows.append(row)
                break

            # Benchmark reference: functional value on the (true - current) block coords.
            truth_update_block = (
                np.asarray(problem.true_state, dtype=np.float64)[op.cols]
                - np.asarray(state, dtype=np.float64)[op.cols]
            )
            dx_matched_quantized = quantized_ridge_update(op, op.alpha_k)

            if arm == ARM_BLOCK:
                dx_block = dense_block_ridge_update(op, op.alpha_k)
                row["evidence_tier"] = EVIDENCE_TIERS["EXACT_MATRIX_ACTION"]
            elif arm == ARM_SPARSE:
                dx_block = sparse_ridge_update(op, op.alpha_k)
                row["evidence_tier"] = EVIDENCE_TIERS["SUPPORT_ERROR"]
            elif arm == ARM_QUANT:
                dx_block = dx_matched_quantized
                row["evidence_tier"] = EVIDENCE_TIERS["EXACT_MATRIX_ACTION"]
            elif arm == ARM_POLY:
                dx_block = exact_polynomial_update(op)
                row["evidence_tier"] = EVIDENCE_TIERS["EXACT_MATRIX_ACTION"]
            elif arm == ARM_STATEVECTOR:
                sv = statevector_update(op)
                dx_block = sv["dx_block"]
                row["postselection_probability"] = sv["postselection_probability"]
                row["circuit_qubits"] = sv["total_qubits"]
                row["circuit_logical_operations"] = sv["logical_operations"]
                row["circuit_logical_depth"] = sv["logical_depth"]
                row["circuit_signal_unitary_calls"] = sv["signal_unitary_calls"]
                row["circuit_projector_phase_calls"] = sv["projector_phase_calls"]
                row["statevector_dim"] = sv["statevector_dim"]
                row["evidence_tier"] = EVIDENCE_TIERS["STATEVECTOR"]
                # The statevector arm evolves the operator exactly (no sampling): one physical
                # circuit execution, zero sampling calls, zero attempted shots.
                cumulative_circuits += 1
                if sv["postselection_probability"] <= 0.0:
                    failed = True
                    failure_code = CLASS_ZERO_POSTSELECTION
                    failure_stage = "postselection"
                    failure_message = "statevector postselection probability is zero"
            elif arm == ARM_FINITE_SHOT:
                # Ceiling is checked against the TOTAL physical shots (both sampling branches).
                projected = cumulative_shots + 2 * shots * op.block_quantized.shape[1]
                if projected > shot_ceiling:
                    failed = True
                    failure_code = CLASS_FINITE_SHOT_CEILING
                    failure_stage = "finite_shot_budget"
                    failure_message = (
                        f"cumulative shots would exceed ceiling {shot_ceiling}"
                    )
                    row["failure_code"] = failure_code
                    row["failure_stage"] = failure_stage
                    row["failure_message"] = failure_message
                    row["convergence_status"] = "failed"
                    row["evidence_tier"] = EVIDENCE_TIERS["FINITE_SHOT"]
                    rows.append(row)
                    break
                statevector_reference = statevector_update(op)
                fs = finite_shot_update(
                    op, block_settings, shots=shots, sampler_seed=sampler_seed
                )
                dx_block = fs["dx_block"]
                row["functional_queries_iter"] = fs["functional_queries"]
                row["unique_functional_circuits_iter"] = fs["unique_functional_circuits"]
                row["physical_circuit_executions_iter"] = fs["physical_circuit_executions"]
                row["sampling_calls_iter"] = fs["sampling_calls"]
                row["shots_per_sampling_call"] = fs["shots_per_sampling_call"]
                row["readout_signal_attempted_shots_iter"] = fs["readout_signal_attempted_shots"]
                row["diagnostic_attempted_shots_iter"] = fs[
                    "diagnostic_postselection_attempted_shots"
                ]
                row["total_attempted_shots_iter"] = fs["total_attempted_shots"]
                row["interference_accepted_shots_iter"] = fs["interference_accepted_shots"]
                row["postselection_accepted_shots_iter"] = fs["postselection_accepted_shots"]
                row["coordinate_query_count"] = fs["functional_queries"]
                row["shots_attempted_iter"] = fs["total_attempted_shots"]
                row["shots_accepted_iter"] = fs["interference_accepted_shots"]
                row["min_interference_acceptance"] = fs["min_interference_acceptance"]
                row["evidence_tier"] = EVIDENCE_TIERS["FINITE_SHOT"]
                cumulative_shots += fs["total_attempted_shots"]
                cumulative_circuits += fs["physical_circuit_executions"]
                for coord in fs["coordinates"]:
                    coordinate_index = int(coord["coordinate"])
                    reference_coordinate = float(
                        statevector_reference["dx_block"][coordinate_index]
                    )
                    functional_circuit_fingerprints.add(
                        str(coord["functional_circuit_fingerprint"])
                    )
                    coord_rows.append(
                        {
                            "arm": arm,
                            "scenario": scenario_id,
                            "seed": int(seed),
                            "iteration": int(iteration),
                            "shots_per_coordinate": int(shots),
                            "statevector_coordinate_update": reference_coordinate,
                            "coordinate_readout_error": float(
                                coord["recovered_coordinate_update"] - reference_coordinate
                            ),
                            "coordinate_readout_abs_error": float(
                                abs(coord["recovered_coordinate_update"] - reference_coordinate)
                            ),
                            **coord,
                        }
                    )
            else:
                raise ValueError(f"unknown arm {arm!r}")

            step_norm = float(np.linalg.norm(dx_block))
            row["step_norm"] = step_norm
            row["block_matched_ridge_abs_error"] = float(
                np.linalg.norm(dx_block - dx_matched_quantized)
            )
            row["block_matched_ridge_error"] = float(
                np.linalg.norm(dx_block - dx_matched_quantized)
                / max(np.linalg.norm(dx_matched_quantized), 1.0e-30)
            )
            dx_full_local = full_ridge_update(matrix, residual, op.alpha_k)
            row["full_ridge_agreement_abs_error"] = float(
                np.linalg.norm(dx_block - dx_full_local[op.cols])
            )
            row["full_ridge_agreement_error"] = float(
                np.linalg.norm(dx_block - dx_full_local[op.cols])
                / max(np.linalg.norm(dx_full_local[op.cols]), 1.0e-30)
            )
            row["selected_output_value"] = float(dx_block[0])
            if truth_update_block is not None:
                row["selected_output_benchmark_error"] = float(
                    abs(dx_block[0] - truth_update_block[0])
                )
            row["cumulative_shots"] = cumulative_shots
            row["cumulative_circuit_executions"] = cumulative_circuits

            if not np.all(np.isfinite(dx_block)):
                failed = True
                failure_code = "nonfinite_update"
                failure_stage = "update"
                failure_message = "block update contains non-finite values"

            if failed:
                row["failure_code"] = failure_code
                row["failure_stage"] = failure_stage
                row["failure_message"] = failure_message
                row["convergence_status"] = "failed"
                rows.append(row)
                break

            next_state = _apply_block_update(
                state, op.cols, dx_block, damping, min_voltage, angle_count
            )
            method_update_full[op.cols] = np.asarray(dx_block, dtype=np.float64)
            step_norm = float(np.linalg.norm(dx_block))

        # Shared advance + stopping checks (both full and block arms reach here on success).
        if step_norm > max_update_norm:
            failed = True
            failure_code = "update_norm_exceeded"
            failure_stage = "update_norm"
            failure_message = f"step norm {step_norm:.3e} exceeds {max_update_norm}"
            row["failure_code"] = failure_code
            row["failure_stage"] = failure_stage
            row["failure_message"] = failure_message
            row["convergence_status"] = "failed"
            rows.append(row)
            break

        weighted_residual_after = _weighted_residual_norm(problem, next_state)
        if not np.isfinite(weighted_residual_after):
            failed = True
            failure_code = "nonfinite_residual"
            failure_stage = "residual"
            failure_message = "weighted residual became non-finite"
        elif (
            weighted_residual_before > 0.0
            and weighted_residual_after / weighted_residual_before > residual_growth_limit
        ):
            failed = True
            failure_code = "residual_growth"
            failure_stage = "residual"
            failure_message = (
                f"residual growth {weighted_residual_after / weighted_residual_before:.3e} "
                f"exceeds {residual_growth_limit}"
            )

        row["state_after_update_json"] = _array_json(next_state)
        row["applied_update_vector_json"] = _array_json(method_update_full)

        converged = bool(
            not failed
            and (step_norm <= update_tol or weighted_residual_after <= residual_tol)
        )
        if failed:
            row["failure_code"] = failure_code
            row["failure_stage"] = failure_stage
            row["failure_message"] = failure_message
            row["convergence_status"] = "failed"
        else:
            row["convergence_status"] = "converged" if converged else "running"
        row["cumulative_shots"] = cumulative_shots
        row["cumulative_circuit_executions"] = cumulative_circuits
        rows.append(row)
        if failed:
            break
        state = next_state
        if converged:
            break

    final_metrics = _state_metrics(problem, state, angle_count) if rows else {}
    final_partition_metrics = (
        _partition_state_metrics(problem, state, last_selected_columns)
        if rows and last_selected_columns.size
        else {
            "selected_coordinate_rmse": float("nan"),
            "frozen_coordinate_rmse": float("nan"),
        }
    )
    final_residual = float(_weighted_residual_norm(problem, state)) if rows else float("nan")
    classification = _classify_run(
        rows,
        converged=converged,
        failed=failed,
        failure_code=failure_code,
        final_state_rmse=final_metrics.get("state_rmse", float("nan")),
        accurate_threshold=accurate_threshold,
        stalled_update_norm=stalled_update_norm,
        max_iterations=max_iterations,
    )
    summary = {
        "arm": arm,
        "case_name": settings["case_name"],
        "scenario": scenario_id,
        "seed": int(seed),
        "iterations": len(rows),
        "converged": bool(converged),
        "failed": bool(failed),
        "classification": classification,
        "final_state_rmse": final_metrics.get("state_rmse", float("nan")),
        "final_angle_rmse": final_metrics.get("angle_rmse", float("nan")),
        "final_voltage_rmse": final_metrics.get("voltage_rmse", float("nan")),
        "final_max_state_error": final_metrics.get("max_state_error", float("nan")),
        "final_selected_coordinate_rmse": final_partition_metrics["selected_coordinate_rmse"],
        "final_frozen_coordinate_rmse": final_partition_metrics["frozen_coordinate_rmse"],
        "final_state_vector_json": _array_json(state) if rows else "",
        "final_weighted_residual": final_residual,
        "final_selected_output_error": (
            float(rows[-1]["selected_output_benchmark_error"]) if rows else float("nan")
        ),
        "final_step_norm": float(rows[-1]["step_norm"]) if rows else float("nan"),
        # Non-overlapping resource accounting (Issue A).  For the finite-shot arm a functional
        # query is one signed coordinate readout; each issues two physical sampling calls
        # (readout branch + direct-postselection diagnostic) of `shots` shots.  The statevector
        # arm evolves exactly: physical executions count circuits, sampling calls and shots are 0.
        "functional_queries": int(sum(int(r["functional_queries_iter"]) for r in rows)),
        "unique_functional_circuits": int(len(functional_circuit_fingerprints)),
        "physical_circuit_executions": cumulative_circuits,
        "sampling_calls": int(sum(int(r["sampling_calls_iter"]) for r in rows)),
        "readout_signal_attempted_shots": int(
            sum(int(r["readout_signal_attempted_shots_iter"]) for r in rows)
        ),
        "diagnostic_attempted_shots": int(
            sum(int(r["diagnostic_attempted_shots_iter"]) for r in rows)
        ),
        "total_attempted_shots": cumulative_shots,
        "interference_accepted_shots": int(
            sum(int(r["interference_accepted_shots_iter"]) for r in rows)
        ),
        "postselection_accepted_shots": int(
            sum(int(r["postselection_accepted_shots_iter"]) for r in rows)
        ),
        # Legacy aliases retained for continuity.
        "total_circuit_queries": cumulative_circuits,
        "total_accepted_shots": int(sum(int(r["shots_accepted_iter"]) for r in rows)),
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "runtime_seconds": time.perf_counter() - started,
    }
    for row in rows:
        row["run_classification"] = classification
    return rows, coord_rows, summary


def _classify_run(
    rows: list[dict[str, Any]],
    *,
    converged: bool,
    failed: bool,
    failure_code: str,
    final_state_rmse: float,
    accurate_threshold: float,
    stalled_update_norm: float,
    max_iterations: int,
) -> str:
    if failed:
        if failure_code == CLASS_CIRCUIT_FAILED:
            return CLASS_CIRCUIT_FAILED
        if failure_code == CLASS_PHASE_FAILED or failure_code in {
            "boundedness_failure",
            "phase_count_mismatch",
        }:
            return CLASS_PHASE_FAILED
        if failure_code == CLASS_FINITE_SHOT_CEILING:
            return CLASS_FINITE_SHOT_CEILING
        if failure_code == CLASS_ZERO_POSTSELECTION:
            return CLASS_ZERO_POSTSELECTION
        if failure_code in {
            "rank_failure",
            "empty_quantized_support_failure",
            "nonzero_residual_norm_failure",
            "polynomial_fit_failure",
        }:
            return CLASS_UNSUPPORTED
        return CLASS_DIVERGED
    if converged:
        if math.isfinite(final_state_rmse) and final_state_rmse <= accurate_threshold:
            return CLASS_CONVERGED_ACCURATE
        return CLASS_CONVERGED_INACCURATE
    return CLASS_STALLED


# --------------------------------------------------------------- error decomposition trace


def compute_decomposition_trace(
    problem: Any,
    anchor_states: list[np.ndarray],
    settings: dict[str, Any],
    block_settings: dict[str, Any],
    loop_settings: dict[str, Any],
    cache_dir: Path,
    *,
    scenario_id: str,
    seed: int,
    include_finite_shots: bool,
) -> list[dict[str, Any]]:
    """Evaluate every modeling stage at the shared operating point of each anchor state.

    Anchor states are the full-system reference (Arm A) trajectory, so each row isolates one
    error source at a single (state, block) operating point: regularization, block truncation,
    support removal, quantization, polynomial approximation, statevector circuit, finite shots.
    """

    alpha_full = float(settings["fixed_alpha"])
    shots = int(loop_settings["finite_shot_budget"])
    sampler_seed = int(loop_settings["finite_shot_sampler_seed"])
    rows: list[dict[str, Any]] = []
    for iteration, state in enumerate(anchor_states):
        system, _ = _linearized_update_system(problem, np.asarray(state, dtype=np.float64))
        matrix = np.asarray(system.H_tilde, dtype=np.float64)
        residual = np.asarray(system.r_tilde, dtype=np.float64)
        op = build_block_operating_point(
            matrix, residual, block_settings, cache_dir, need_phases=include_finite_shots
        )
        base = {
            "scenario": scenario_id,
            "seed": int(seed),
            "iteration": int(iteration),
            "anchor": "full_system_exact_ridge_trajectory",
            "selected_global_columns": ",".join(str(int(c)) for c in op.cols),
            "failure_code": op.failure_code,
        }
        if op.failure_code and op.failure_stage in {"phase_synthesis", "boundedness"}:
            # Non-circuit stages are still valid; only the circuit stages are unavailable.
            include_circuit = False
        elif op.failure_code:
            rows.append({**base, "stage_available": False})
            continue
        else:
            include_circuit = True

        dx_full_alpha_full = full_ridge_update(matrix, residual, alpha_full)
        dx_full_alpha_k = full_ridge_update(matrix, residual, op.alpha_k)
        dx_block_dense = dense_block_ridge_update(op, op.alpha_k)
        dx_sparse = sparse_ridge_update(op, op.alpha_k)
        dx_quantized = quantized_ridge_update(op, op.alpha_k)
        dx_poly = exact_polynomial_update(op)
        cols = op.cols

        def abs_err(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        def rel(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))

        record = {
            **base,
            "stage_available": True,
            "beta_k": op.beta,
            "alpha_k": op.alpha_k,
            "normalized_lambda": op.normalized_lambda,
            "contraction_C_k": op.contraction_c,
            "block_rank": op.block_rank,
            "kappa_block": op.kappa_block,
            "support_nnz": op.support_nnz,
            "sparsification_fro_error": op.sparsification_fro_error,
            "max_quantization_error": op.max_quantization_error,
            "polynomial_fit_error": op.uniform_fit_error,
            "bounded_max_abs": op.bounded_max_abs,
            # Reference update norms so relative errors near zero are interpretable.
            "full_update_alpha_full_norm": float(np.linalg.norm(dx_full_alpha_full)),
            "full_update_alpha_k_block_norm": float(np.linalg.norm(dx_full_alpha_k[cols])),
            "block_dense_update_norm": float(np.linalg.norm(dx_block_dense)),
            "quantized_update_norm": float(np.linalg.norm(dx_quantized)),
            # Stage-to-stage errors (each isolates ONE source); absolute + relative.
            "regularization_gap_abs_error": abs_err(dx_full_alpha_k, dx_full_alpha_full),
            "regularization_gap_error": rel(dx_full_alpha_k, dx_full_alpha_full),
            "block_truncation_abs_error": abs_err(dx_block_dense, dx_full_alpha_k[cols]),
            "block_truncation_error": rel(dx_block_dense, dx_full_alpha_k[cols]),
            "support_removal_abs_error": abs_err(dx_sparse, dx_block_dense),
            "support_removal_error": rel(dx_sparse, dx_block_dense),
            "quantization_abs_error": abs_err(dx_quantized, dx_sparse),
            "quantization_error": rel(dx_quantized, dx_sparse),
            "polynomial_abs_error": abs_err(dx_poly, dx_quantized),
            "polynomial_error": rel(dx_poly, dx_quantized),
            "statevector_circuit_abs_error": float("nan"),
            "statevector_circuit_error": float("nan"),
            "finite_shot_readout_abs_error": float("nan"),
            "finite_shot_readout_error": float("nan"),
            "postselection_probability": float("nan"),
            "coordinate_query_count": 0,
            "shots_per_coordinate": 0,
        }
        if include_circuit and op.phases is not None:
            sv = statevector_update(op)
            dx_sv = sv["dx_block"]
            record["statevector_circuit_abs_error"] = abs_err(dx_sv, dx_poly)
            record["statevector_circuit_error"] = rel(dx_sv, dx_poly)
            record["postselection_probability"] = sv["postselection_probability"]
            if include_finite_shots:
                fs = finite_shot_update(
                    op, block_settings, shots=shots, sampler_seed=sampler_seed
                )
                record["finite_shot_readout_abs_error"] = abs_err(fs["dx_block"], dx_sv)
                record["finite_shot_readout_error"] = rel(fs["dx_block"], dx_sv)
                record["coordinate_query_count"] = fs["coordinate_query_count"]
                record["shots_per_coordinate"] = int(shots)
        rows.append(record)
    return rows


# ----------------------------------------------------------------- circuit-resource audit (B)


def build_circuit_resource_audit(
    problem_iter: Any,
    settings: dict[str, Any],
    block_settings: dict[str, Any],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Characterise every closed-loop circuit at each abstraction level (Issue B).

    For each distinct block-encoding dimension encountered at the first iteration of the primary
    scenarios/seeds, the QSVT statevector operator circuit, the integrated finite-shot readout
    circuit, and its direct-postselection twin are built once and reported at logical, decomposed,
    and transpiled-basis levels.  The logical count (e.g. 63) is the number of opaque ``unitary``
    blocks and is explicitly labelled as such; the transpiled count is a primitive basis-operation
    count under all-to-all connectivity, without device routing.  Structure-determined, so one
    representative per dimension is exact.
    """

    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        build_integrated_sparse_selected_output_circuit,
    )

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for state, scenario_id, seed in problem_iter:
        system, _ = _linearized_update_system(problem_iter.problem, np.asarray(state, float))
        op = build_block_operating_point(
            np.asarray(system.H_tilde, dtype=np.float64),
            np.asarray(system.r_tilde, dtype=np.float64),
            block_settings,
            cache_dir,
            need_phases=True,
        )
        if op.failure_code or op.phases is None:
            continue
        dim = int(op.wrapper_unitary.shape[0])
        if dim in seen:
            continue
        seen.add(dim)
        n = op.block_quantized.shape[1]
        provenance = {
            "scenario": scenario_id,
            "seed": int(seed),
            "block_encoding_dimension": dim,
            "degree": int(op.degree),
        }
        sv_bundle = build_structured_qsvt_operator_circuit(
            op.wrapper_unitary, op.phases, encoded_dimension=n
        )
        rows.append({**provenance, **circuit_resource_levels(
            sv_bundle.qsvt_operator_circuit,
            label="qsvt_statevector_operator",
            qsvt_degree=int(op.degree),
            signal_applications=int(sv_bundle.block_encoding_gate_count),
            phase_applications=int(sv_bundle.phase_gate_count),
        )})
        fs_config = _finite_shot_config(op, block_settings)
        integ = build_integrated_sparse_selected_output_circuit(
            fs_config,
            matrix=op.block_quantized,
            residual=op.residual_block,
            selected_functional=np.eye(n)[0],
            phases=op.phases,
        )
        rows.append({**provenance, **circuit_resource_levels(
            integ.circuit,
            label="finite_shot_selected_output_readout",
            qsvt_degree=int(op.degree),
            signal_applications=int(sv_bundle.block_encoding_gate_count),
            phase_applications=int(sv_bundle.phase_gate_count),
            includes_residual_state_preparation=True,
            includes_functional_preparation=True,
            includes_interference_readout=True,
        )})
        rows.append({**provenance, **circuit_resource_levels(
            integ.direct_postselection_circuit,
            label="finite_shot_direct_postselection",
            qsvt_degree=int(op.degree),
            signal_applications=int(sv_bundle.block_encoding_gate_count),
            phase_applications=int(sv_bundle.phase_gate_count),
            includes_residual_state_preparation=True,
            includes_direct_postselection=True,
        )})
    return rows


class _FirstIterationStates:
    """Yield (initial_state, scenario_id, seed) for the primary scenarios/seeds (audit input)."""

    def __init__(self, problem: Any, scenario_id: str, seed: int) -> None:
        self.problem = problem
        self._items = [(problem.initial_state.copy(), scenario_id, seed)]

    def __iter__(self):
        return iter(self._items)


# --------------------------------------------------- extended-horizon convergence diagnostic (C)


def classify_trajectory(
    rmse: list[float],
    step_norm: list[float],
    *,
    converged: bool,
    failed: bool,
    accurate_threshold: float,
    plateau_window: int,
    plateau_rmse_rel_tol: float,
    plateau_step_norm: float,
    still_improving_rel_tol: float,
    oscillation_rel_tol: float,
) -> dict[str, Any]:
    """Classify one extended-horizon trajectory using the predeclared plateau rule (Issue C).

    Categories: converged, plateaued, still_improving_at_horizon, oscillatory, diverged,
    maximum_iterations_without_plateau.  The rule is fixed before inspecting the extended runs.
    """

    if len(rmse) != len(step_norm):
        raise ValueError("rmse and step_norm must have equal length")
    finite = [float(v) for v in rmse if math.isfinite(v)]
    result = {
        "final_state_rmse": float(rmse[-1]) if rmse else float("nan"),
        "final_step_norm": float(step_norm[-1]) if step_norm else float("nan"),
        "plateau_above_accurate_threshold": bool(
            rmse and math.isfinite(rmse[-1]) and rmse[-1] > accurate_threshold
        ),
        "plateau_onset_iteration": -1,
        "rmse_floor_entry_iteration": -1,
        "first_qualifying_window_end_iteration": -1,
        "convergence_iteration": -1,
        "plateau_rule_satisfied_at_horizon": False,
        "plateau_rule_remains_satisfied_from_onset": False,
        "window_relative_rmse_change": float("nan"),
        "window_max_relative_swing": float("nan"),
    }
    if finite:
        floor = finite[-1]
        result["rmse_floor_entry_iteration"] = int(
            next(
                (
                    idx
                    for idx in range(len(finite))
                    if all(
                        abs(value - floor) / max(abs(floor), 1.0e-30)
                        <= plateau_rmse_rel_tol
                        for value in finite[idx:]
                    )
                ),
                len(finite) - 1,
            )
        )
    if failed:
        result["classification"] = "diverged"
        return result
    if converged:
        result["classification"] = "converged"
        result["convergence_iteration"] = len(rmse) - 1
        return result
    if len(finite) != len(rmse):
        result["classification"] = "diverged"
        return result
    if len(finite) < plateau_window:
        result["classification"] = "maximum_iterations_without_plateau"
        return result

    qualifying_windows: list[tuple[int, bool, float, float]] = []
    for end in range(plateau_window - 1, len(finite)):
        candidate = finite[end - plateau_window + 1 : end + 1]
        candidate_ref = max(abs(candidate[-1]), 1.0e-30)
        candidate_change = abs(candidate[-1] - candidate[0]) / candidate_ref
        candidate_swing = (max(candidate) - min(candidate)) / candidate_ref
        candidate_settled = (
            math.isfinite(step_norm[end]) and step_norm[end] <= plateau_step_norm
        )
        qualifies = bool(
            candidate_change <= plateau_rmse_rel_tol
            and candidate_swing <= oscillation_rel_tol
            and candidate_settled
        )
        qualifying_windows.append((end, qualifies, candidate_change, candidate_swing))

    window = finite[-plateau_window:]
    ref = max(abs(window[-1]), 1.0e-30)
    rel_change = abs(window[-1] - window[0]) / ref
    swing = (max(window) - min(window)) / ref
    result["window_relative_rmse_change"] = float(rel_change)
    result["window_max_relative_swing"] = float(swing)
    settled_step = math.isfinite(step_norm[-1]) and step_norm[-1] <= plateau_step_norm
    result["first_qualifying_window_end_iteration"] = int(
        next((end for end, qualifies, _change, _swing in qualifying_windows if qualifies), -1)
    )
    result["plateau_rule_satisfied_at_horizon"] = bool(
        qualifying_windows and qualifying_windows[-1][1]
    )

    if swing > oscillation_rel_tol:
        result["classification"] = "oscillatory"
        return result
    # Plateau: the trailing RMSE window is flat and the update has settled.
    if rel_change <= plateau_rmse_rel_tol and settled_step:
        # Onset is the END of the earliest diagnostic window from which every future window
        # continues to satisfy the same joint RMSE-flatness, non-oscillation, and settled-step
        # rule.  The separate rmse_floor_entry_iteration records the earlier RMSE-only quantity.
        onset = qualifying_windows[-1][0]
        for position, (end, qualifies, _change, _swing) in enumerate(qualifying_windows):
            if qualifies and all(item[1] for item in qualifying_windows[position:]):
                onset = end
                break
        result["plateau_onset_iteration"] = int(onset)
        result["plateau_rule_remains_satisfied_from_onset"] = bool(
            all(
                item[1]
                for item in qualifying_windows
                if item[0] >= result["plateau_onset_iteration"]
            )
        )
        result["classification"] = "plateaued"
        return result
    if window[0] - window[-1] > still_improving_rel_tol * ref:
        result["classification"] = "still_improving_at_horizon"
        return result
    result["classification"] = "maximum_iterations_without_plateau"
    return result


def run_extended_horizon_diagnostic(
    config: dict[str, Any], cache_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the labelled extended-horizon diagnostic for the inexpensive arms (Issue C)."""

    ext = config["extended_horizon"]
    settings = copy_settings_with_horizon(config["nonlinear_settings"], int(ext["max_iterations"]))
    block_settings = config["block_qsvt"]
    loop_settings = config["closed_loop"]
    scenarios = config["scenarios"]
    seeds = [int(s) for s in config["seeds"]]
    arms = list(ext["arms"])
    accurate_threshold = float(loop_settings.get("accurate_state_rmse_threshold", 2.0e-3))

    iteration_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            problem = build_ac_nonlinear_problem(build_problem_config(settings, scenario, seed))
            for arm in arms:
                rows, _coords, summary = drive_arm(
                    arm, problem, settings, block_settings, loop_settings, cache_dir,
                    scenario_id=scenario_id, seed=seed,
                )
                rmse = [float(r["state_rmse"]) for r in rows]
                step = [float(r["step_norm"]) for r in rows]
                for r in rows:
                    iteration_rows.append(
                        {
                            "arm": arm,
                            "scenario": scenario_id,
                            "seed": int(seed),
                            "iteration": int(r["iteration"]),
                            "state_rmse": float(r["state_rmse"]),
                            "angle_rmse": float(r["angle_rmse"]),
                            "voltage_rmse": float(r["voltage_rmse"]),
                            "weighted_residual_norm": float(r["weighted_residual_norm"]),
                            "step_norm": float(r["step_norm"]),
                            "selected_coordinate_rmse": float(r["selected_coordinate_rmse"]),
                            "frozen_coordinate_rmse": float(r["frozen_coordinate_rmse"]),
                            "selected_output_benchmark_error": float(
                                r["selected_output_benchmark_error"]
                            ),
                            "convergence_status": r["convergence_status"],
                        }
                    )
                cls = classify_trajectory(
                    rmse, step,
                    converged=bool(summary["converged"]),
                    failed=bool(summary["failed"]),
                    accurate_threshold=accurate_threshold,
                    plateau_window=int(ext["plateau_window"]),
                    plateau_rmse_rel_tol=float(ext["plateau_rmse_rel_tol"]),
                    plateau_step_norm=float(ext["plateau_step_norm"]),
                    still_improving_rel_tol=float(ext["still_improving_rel_tol"]),
                    oscillation_rel_tol=float(ext["oscillation_rel_tol"]),
                )
                classification_rows.append(
                    {
                        "arm": arm,
                        "scenario": scenario_id,
                        "seed": int(seed),
                        "executed_iterations": len(rows),
                        "max_iterations": int(ext["max_iterations"]),
                        "primary_protocol_iterations": int(
                            settings["iteration"].get("primary_max_iterations", 8)
                        ),
                        "final_weighted_residual": float(summary["final_weighted_residual"]),
                        "final_selected_coordinate_rmse": float(
                            summary["final_selected_coordinate_rmse"]
                        ),
                        "final_frozen_coordinate_rmse": float(
                            summary["final_frozen_coordinate_rmse"]
                        ),
                        "final_selected_output_error": float(
                            summary["final_selected_output_error"]
                        ),
                        **cls,
                    }
                )
    return iteration_rows, classification_rows


def copy_settings_with_horizon(settings: dict[str, Any], max_iterations: int) -> dict[str, Any]:
    import copy as _copy

    new = _copy.deepcopy(settings)
    new["iteration"]["primary_max_iterations"] = int(settings["iteration"]["max_iterations"])
    new["iteration"]["max_iterations"] = int(max_iterations)
    return new


# ------------------------------------------------- finite-shot vs statevector comparison (D)


def build_finite_shot_statevector_comparison(
    iteration_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    coordinate_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Paired finite-shot vs statevector trajectory differences per matched run (Issue D)."""

    columns = [
        "scenario", "seed", "matched", "statevector_final_rmse", "finite_shot_final_rmse",
        "delta_final_rmse", "absolute_final_rmse_difference", "max_state_rmse_difference",
        "final_state_norm_difference", "max_trajectory_state_norm_difference",
        "max_selected_coordinate_state_norm_difference", "max_selected_output_difference",
        "final_selected_output_difference", "max_step_norm_difference",
        "mean_step_norm_difference", "max_update_vector_norm_difference",
        "mean_update_vector_norm_difference", "median_coordinate_readout_abs_error",
        "max_coordinate_readout_abs_error", "interference_acceptance_mean",
        "interference_acceptance_std", "direct_postselection_mean",
        "direct_postselection_std", "iterations",
    ]
    if iteration_frame.empty:
        return pd.DataFrame(columns=columns)
    sv = iteration_frame[iteration_frame["arm"] == ARM_STATEVECTOR]
    fs = iteration_frame[iteration_frame["arm"] == ARM_FINITE_SHOT]
    records = []
    for (scenario, seed), fs_run in fs.groupby(["scenario", "seed"]):
        sv_run = sv[(sv["scenario"] == scenario) & (sv["seed"] == seed)]
        if sv_run.empty:
            continue
        f = fs_run.sort_values("iteration").reset_index(drop=True)
        s = sv_run.sort_values("iteration").reset_index(drop=True)
        k = min(len(f), len(s))
        f, s = f.iloc[:k], s.iloc[:k]
        rmse_diff = np.abs(f["state_rmse"].to_numpy() - s["state_rmse"].to_numpy())
        sel_diff = np.abs(
            f["selected_output_value"].to_numpy() - s["selected_output_value"].to_numpy()
        )
        step_diff = np.abs(f["step_norm"].to_numpy() - s["step_norm"].to_numpy())
        state_norm_differences = []
        selected_state_norm_differences = []
        update_norm_differences = []
        for index in range(k):
            f_state = np.asarray(json.loads(f["state_after_update_json"].iloc[index]), dtype=float)
            s_state = np.asarray(json.loads(s["state_after_update_json"].iloc[index]), dtype=float)
            state_norm_differences.append(float(np.linalg.norm(f_state - s_state)))
            selected_columns = np.asarray(
                [int(value) for value in str(f["selected_global_columns"].iloc[index]).split(",")],
                dtype=np.int64,
            )
            selected_state_norm_differences.append(
                float(np.linalg.norm(f_state[selected_columns] - s_state[selected_columns]))
            )
            f_update = np.asarray(
                json.loads(f["applied_update_vector_json"].iloc[index]), dtype=float
            )
            s_update = np.asarray(
                json.loads(s["applied_update_vector_json"].iloc[index]), dtype=float
            )
            update_norm_differences.append(float(np.linalg.norm(f_update - s_update)))

        sv_summary = summary_frame[
            (summary_frame["arm"] == ARM_STATEVECTOR)
            & (summary_frame["scenario"] == scenario)
            & (summary_frame["seed"] == seed)
        ].iloc[0]
        fs_summary = summary_frame[
            (summary_frame["arm"] == ARM_FINITE_SHOT)
            & (summary_frame["scenario"] == scenario)
            & (summary_frame["seed"] == seed)
        ].iloc[0]
        statevector_final_rmse = float(sv_summary["final_state_rmse"])
        finite_shot_final_rmse = float(fs_summary["final_state_rmse"])
        delta_final_rmse = finite_shot_final_rmse - statevector_final_rmse
        final_sv_state = np.asarray(json.loads(sv_summary["final_state_vector_json"]), dtype=float)
        final_fs_state = np.asarray(json.loads(fs_summary["final_state_vector_json"]), dtype=float)

        run_coordinates = (
            coordinate_frame[
                (coordinate_frame["scenario"] == scenario)
                & (coordinate_frame["seed"] == seed)
            ]
            if coordinate_frame is not None and not coordinate_frame.empty
            else pd.DataFrame()
        )
        coordinate_errors = (
            run_coordinates["coordinate_readout_abs_error"].to_numpy(dtype=float)
            if not run_coordinates.empty
            else np.asarray([], dtype=float)
        )
        interference = (
            run_coordinates["interference_acceptance_probability"].to_numpy(dtype=float)
            if not run_coordinates.empty
            else np.asarray([], dtype=float)
        )
        direct = (
            run_coordinates["direct_postselection_probability"].to_numpy(dtype=float)
            if not run_coordinates.empty
            else np.asarray([], dtype=float)
        )

        def statistic(values: np.ndarray, operation: str) -> float:
            if values.size == 0 or not np.any(np.isfinite(values)):
                return float("nan")
            if operation == "max":
                return float(np.nanmax(values))
            if operation == "median":
                return float(np.nanmedian(values))
            if operation == "mean":
                return float(np.nanmean(values))
            if operation == "std":
                return float(np.nanstd(values))
            raise ValueError(operation)

        records.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "matched": True,
                "statevector_final_rmse": statevector_final_rmse,
                "finite_shot_final_rmse": finite_shot_final_rmse,
                "delta_final_rmse": delta_final_rmse,
                "absolute_final_rmse_difference": abs(delta_final_rmse),
                "max_state_rmse_difference": float(np.nanmax(rmse_diff)),
                "final_state_norm_difference": float(
                    np.linalg.norm(final_fs_state - final_sv_state)
                ),
                "max_trajectory_state_norm_difference": float(
                    np.max(state_norm_differences)
                ),
                "max_selected_coordinate_state_norm_difference": float(
                    np.max(selected_state_norm_differences)
                ),
                "max_selected_output_difference": float(np.nanmax(sel_diff)),
                "final_selected_output_difference": float(sel_diff[-1]),
                "max_step_norm_difference": float(np.nanmax(step_diff)),
                "mean_step_norm_difference": float(np.nanmean(step_diff)),
                "max_update_vector_norm_difference": float(
                    np.max(update_norm_differences)
                ),
                "mean_update_vector_norm_difference": float(
                    np.mean(update_norm_differences)
                ),
                "median_coordinate_readout_abs_error": statistic(
                    coordinate_errors, "median"
                ),
                "max_coordinate_readout_abs_error": statistic(coordinate_errors, "max"),
                "interference_acceptance_mean": statistic(interference, "mean"),
                "interference_acceptance_std": statistic(interference, "std"),
                "direct_postselection_mean": statistic(direct, "mean"),
                "direct_postselection_std": statistic(direct, "std"),
                "iterations": int(k),
            }
        )
    return pd.DataFrame(records, columns=columns)


def sampler_seed_variability(
    op: BlockOperatingPoint, block_settings: dict[str, Any], shots: int, seeds: list[int]
) -> dict[str, Any]:
    """Spread of the finite-shot update and postselection across sampler seeds (Issue D)."""

    updates = []
    interference = []
    postselection = []
    for s in seeds:
        fs = finite_shot_update(op, block_settings, shots=shots, sampler_seed=int(s))
        updates.append(fs["dx_block"])
        interference.append(
            fs["interference_accepted_shots"] / max(fs["readout_signal_attempted_shots"], 1)
        )
        postselection.append(
            fs["postselection_accepted_shots"]
            / max(fs["diagnostic_postselection_attempted_shots"], 1)
        )
    stack = np.vstack(updates)
    return {
        "sampler_seeds": seeds,
        "update_coordinate_std_max": float(np.max(np.std(stack, axis=0))),
        "update_norm_std": float(np.std([np.linalg.norm(u) for u in updates])),
        "interference_acceptance_std": float(np.std(interference)),
        "postselection_acceptance_std": float(np.std(postselection)),
    }


# --------------------------------------------------------------------------- orchestrator


def run_closed_loop(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"config study_id mismatch: {config.get('study_id')!r} != {STUDY_ID!r}")
    destination = Path(output_dir)
    for sub in (
        "configs",
        "iteration_ledgers",
        "run_summaries",
        "error_decomposition",
        "resource_ledgers",
        "failure_ledgers",
        "figures",
        "tables",
        "manifests",
    ):
        (destination / sub).mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = config["nonlinear_settings"]
    block_settings = config["block_qsvt"]
    loop_settings = config["closed_loop"]
    scenarios = config["scenarios"]
    seeds = [int(s) for s in config["seeds"]]
    finite_shot_seed = int(config["finite_shot_seed"])
    arms = list(loop_settings["arms"])
    finite_shot_seed_only = bool(loop_settings.get("finite_shot_arm_seed_only", True))

    all_iteration_rows: list[dict[str, Any]] = []
    all_coord_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    anchor_states_by_key: dict[tuple[str, int], list[np.ndarray]] = {}
    resource_audit_rows: list[dict[str, Any]] = []
    seen_resource_dims: set[int] = set()

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        for seed in seeds:
            problem = build_ac_nonlinear_problem(build_problem_config(settings, scenario, seed))
            # Circuit-resource audit (Issue B): characterise each distinct block-encoding size once.
            audit_rows = build_circuit_resource_audit(
                _FirstIterationStates(problem, scenario_id, seed),
                settings, block_settings, cache_dir,
            )
            new_dims = {
                int(r["block_encoding_dimension"])
                for r in audit_rows
                if int(r["block_encoding_dimension"]) not in seen_resource_dims
            }
            if new_dims:
                seen_resource_dims.update(new_dims)
                resource_audit_rows.extend(
                    r for r in audit_rows if int(r["block_encoding_dimension"]) in new_dims
                )
            for arm in arms:
                if arm == ARM_FINITE_SHOT and finite_shot_seed_only and seed != finite_shot_seed:
                    continue
                rows, coords, summary = drive_arm(
                    arm,
                    problem,
                    settings,
                    block_settings,
                    loop_settings,
                    cache_dir,
                    scenario_id=scenario_id,
                    seed=seed,
                )
                all_iteration_rows.extend(rows)
                all_coord_rows.extend(coords)
                all_summaries.append(summary)
                if arm == ARM_FULL:
                    anchor_states_by_key[(scenario_id, seed)] = full_reference_trajectory(
                        problem, settings
                    )
                if progress:
                    print(
                        f"[closed_loop] {scenario_id} seed={seed} arm={arm} "
                        f"class={summary['classification']} rmse={summary['final_state_rmse']:.3e}",
                        flush=True,
                    )
            # Error-decomposition trace along the full-system reference trajectory.
            anchor_states = anchor_states_by_key.get((scenario_id, seed), [])
            include_fs = seed == finite_shot_seed
            all_decomposition_rows.extend(
                compute_decomposition_trace(
                    problem,
                    anchor_states,
                    settings,
                    block_settings,
                    loop_settings,
                    cache_dir,
                    scenario_id=scenario_id,
                    seed=seed,
                    include_finite_shots=include_fs,
                )
            )

    # Extended-horizon convergence diagnostic (Issue C), clearly separated from the primary run.
    extended_iteration_rows: list[dict[str, Any]] = []
    extended_classification_rows: list[dict[str, Any]] = []
    if bool(config.get("extended_horizon", {}).get("enabled", False)):
        extended_iteration_rows, extended_classification_rows = run_extended_horizon_diagnostic(
            config, cache_dir
        )

    # Sampler-seed variability probe (Issue D): spread across independent sampling seeds at one
    # representative operating point (the first scenario/seed, first iteration).
    sampler_variability: dict[str, Any] = {}
    try:
        probe_problem = build_ac_nonlinear_problem(
            build_problem_config(settings, scenarios[0], seeds[0])
        )
        system0, _ = _linearized_update_system(probe_problem, probe_problem.initial_state.copy())
        op0 = build_block_operating_point(
            np.asarray(system0.H_tilde, dtype=np.float64),
            np.asarray(system0.r_tilde, dtype=np.float64),
            block_settings, cache_dir, need_phases=True,
        )
        if not op0.failure_code and op0.phases is not None:
            base = int(block_settings["finite_shot_sampler_seed"])
            sampler_variability = sampler_seed_variability(
                op0, block_settings, int(block_settings["finite_shot_budget"]),
                [base, base + 1, base + 2, base + 3, base + 4],
            )
    except Exception as exc:  # pragma: no cover - probe is best-effort, never blocks the run
        sampler_variability = {"error": str(exc)[:200]}

    _write_all_outputs(
        destination,
        config,
        config_path,
        all_iteration_rows,
        all_coord_rows,
        all_summaries,
        all_decomposition_rows,
        resource_audit_rows,
        extended_iteration_rows,
        extended_classification_rows,
        sampler_variability,
    )
    return {
        "arms": arms,
        "iteration_rows": len(all_iteration_rows),
        "runs": len(all_summaries),
        "statevector_runs": sum(
            1 for s in all_summaries if s["arm"] == ARM_STATEVECTOR
        ),
        "finite_shot_runs": sum(1 for s in all_summaries if s["arm"] == ARM_FINITE_SHOT),
        "converged_runs": sum(1 for s in all_summaries if s["converged"]),
        "failed_runs": sum(1 for s in all_summaries if s["failed"]),
        "decomposition_rows": len(all_decomposition_rows),
        "coordinate_rows": len(all_coord_rows),
        "circuit_resource_audit_rows": len(resource_audit_rows),
        "extended_horizon_rows": len(extended_iteration_rows),
        "extended_horizon_runs": len(extended_classification_rows),
    }


def full_reference_trajectory(problem: Any, settings: dict[str, Any]) -> list[np.ndarray]:
    """Deterministically replay the full-system exact-Ridge arm and return its anchor states.

    Each returned state is the state at the *start* of one full-system iteration - the operating
    point from which the shared block is extracted for the error-decomposition trace.  Mirrors the
    ``ARM_FULL`` branch of :func:`drive_arm` exactly (same fixed alpha, damping, and stopping).
    """

    iteration_cfg = settings["iteration"]
    max_iterations = int(iteration_cfg["max_iterations"])
    update_tol = float(iteration_cfg["update_tolerance"])
    residual_tol = float(iteration_cfg["residual_tolerance"])
    damping = float(iteration_cfg["damping"])
    max_update_norm = float(iteration_cfg.get("max_update_norm", 1.0e6))
    residual_growth_limit = float(iteration_cfg.get("residual_growth_limit", 1.0e6))
    min_voltage = float(settings["linearization"]["min_voltage_magnitude"])
    alpha_full = float(settings["fixed_alpha"])
    angle_count = len(problem.case.angle_state_buses)

    state = problem.initial_state.copy()
    anchors: list[np.ndarray] = []
    for _ in range(max_iterations):
        anchors.append(state.copy())
        system, weighted_before = _linearized_update_system(problem, state)
        dx_full = full_ridge_update(
            np.asarray(system.H_tilde, dtype=np.float64),
            np.asarray(system.r_tilde, dtype=np.float64),
            alpha_full,
        )
        step_norm = float(np.linalg.norm(dx_full))
        if step_norm > max_update_norm:
            break
        next_state = state + damping * dx_full
        next_state[angle_count:] = np.maximum(next_state[angle_count:], min_voltage)
        weighted_after = _weighted_residual_norm(problem, next_state)
        if not np.isfinite(weighted_after):
            break
        if (
            weighted_before > 0.0
            and weighted_after / weighted_before > residual_growth_limit
        ):
            break
        if step_norm <= update_tol or weighted_after <= residual_tol:
            break
        state = next_state
    return anchors


def _write_all_outputs(
    destination: Path,
    config: dict[str, Any],
    config_path: str | Path,
    iteration_rows: list[dict[str, Any]],
    coord_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    resource_audit_rows: list[dict[str, Any]] | None = None,
    extended_iteration_rows: list[dict[str, Any]] | None = None,
    extended_classification_rows: list[dict[str, Any]] | None = None,
    sampler_variability: dict[str, Any] | None = None,
) -> None:
    for sub in ("audits", "extended_horizon"):
        (destination / sub).mkdir(parents=True, exist_ok=True)
    iteration_frame = pd.DataFrame(iteration_rows)
    summary_frame = pd.DataFrame(summaries)
    coord_frame = (
        pd.DataFrame(coord_rows)
        if coord_rows
        else pd.DataFrame(columns=["arm", "scenario", "seed", "iteration", "coordinate"])
    )
    decomposition_frame = (
        pd.DataFrame(decomposition_rows)
        if decomposition_rows
        else pd.DataFrame(columns=["scenario", "seed", "iteration", "stage_available"])
    )

    atomic_write_csv(
        destination / "iteration_ledgers" / "closed_loop_iterations.csv", iteration_frame
    )
    atomic_write_csv(destination / "run_summaries" / "solver_outcomes.csv", summary_frame)
    atomic_write_csv(
        destination / "resource_ledgers" / "finite_shot_coordinate_readout.csv", coord_frame
    )
    atomic_write_csv(
        destination / "error_decomposition" / "stage_error_decomposition.csv",
        decomposition_frame,
    )

    _write_resource_summary(destination, iteration_frame, coord_frame)
    _write_failure_ledger(destination, iteration_frame, summary_frame)
    _write_claim_boundary(destination, summary_frame)
    _write_readout_cost(destination, coord_frame, summary_frame)
    _write_query_execution_accounting(destination, summary_frame, coord_frame, config)
    _write_circuit_resource_audit(destination, resource_audit_rows or [])
    _write_extended_horizon(
        destination,
        extended_iteration_rows or [],
        extended_classification_rows or [],
        config,
    )
    comparison = build_finite_shot_statevector_comparison(
        iteration_frame, summary_frame, coord_frame
    )
    atomic_write_csv(
        destination / "audits" / "finite_shot_statevector_comparison.csv", comparison
    )
    atomic_write_json(
        destination / "audits" / "sampler_seed_variability.json", sampler_variability or {}
    )

    provenance = provenance_block(config_path, config)
    # Preserve environment identity without embedding an absolute repository path.  Package
    # versions and the exact invocation are recorded separately in the reproduction report.
    provenance["environment"]["executable"] = Path(
        str(provenance["environment"]["executable"])
    ).name
    atomic_write_json(
        destination / "manifests" / "run_manifest.json",
        provenance
        | {"study_id": STUDY_ID, "runs": len(summaries), "iterations": len(iteration_rows)},
    )
    _write_audit_manifest(
        destination, summary_frame, coord_frame, comparison, resource_audit_rows or [],
        extended_classification_rows or [], sampler_variability or {},
    )
    _write_resolved_config(destination, config)
    _write_readme(destination, summary_frame, decomposition_frame, coord_frame)
    write_manifest_and_checksums(
        destination,
        study_id=STUDY_ID,
        extra={"runs": len(summaries), "iterations": len(iteration_rows)},
    )


def _write_circuit_resource_audit(
    destination: Path, resource_audit_rows: list[dict[str, Any]]
) -> None:
    columns = [
        "scenario", "seed", "block_encoding_dimension", "degree", "circuit_type", "n_qubits",
        "n_clbits", "qsvt_degree", "signal_unitary_applications",
        "projector_phase_applications", "logical_operations", "untranspiled_sdk_operations",
        "logical_depth", "logical_op_breakdown", "untranspiled_measurements",
        "untranspiled_controlled_phase_operations", "untranspiled_multi_controlled_operations",
        "untranspiled_custom_or_opaque_operations", "decomposed_operations",
        "transpiled_basis_gates", "transpiled_depth", "transpiled_cx", "transpiled_basis",
        "transpiler_opt_level", "transpiler_seed", "coupling_map_assumption",
        "routing_included", "backend_target", "opaque_instructions_remaining",
        "includes_residual_state_preparation", "includes_sparse_block_encoding",
        "includes_qsvt_signal_phase_sequence", "includes_functional_preparation",
        "includes_interference_readout", "includes_postselection_registers",
        "includes_direct_postselection",
    ]
    frame = (
        pd.DataFrame(resource_audit_rows)
        if resource_audit_rows
        else pd.DataFrame(columns=columns)
    )
    frame = frame[[c for c in columns if c in frame.columns]]
    atomic_write_csv(destination / "resource_ledgers" / "circuit_resource_levels.csv", frame)
    atomic_write_csv(destination / "audits" / "circuit_resource_audit.csv", frame)


def _write_extended_horizon(
    destination: Path,
    extended_iteration_rows: list[dict[str, Any]],
    extended_classification_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    for sub in ("iteration_ledgers", "run_summaries", "figures"):
        (destination / "extended_horizon" / sub).mkdir(parents=True, exist_ok=True)
    iter_frame = (
        pd.DataFrame(extended_iteration_rows)
        if extended_iteration_rows
        else pd.DataFrame(columns=["arm", "scenario", "seed", "iteration", "state_rmse"])
    )
    cls_frame = (
        pd.DataFrame(extended_classification_rows)
        if extended_classification_rows
        else pd.DataFrame(columns=["arm", "scenario", "seed", "classification"])
    )
    atomic_write_csv(
        destination / "extended_horizon" / "iteration_ledgers" / "extended_iterations.csv",
        iter_frame,
    )
    atomic_write_csv(
        destination / "extended_horizon" / "run_summaries" / "trajectory_classification.csv",
        cls_frame,
    )
    audit_rows: list[dict[str, Any]] = []
    for _, classification in cls_frame.iterrows():
        run = iter_frame[
            (iter_frame["arm"] == classification["arm"])
            & (iter_frame["scenario"] == classification["scenario"])
            & (iter_frame["seed"] == classification["seed"])
        ].sort_values("iteration")
        sorted_unique = bool(
            run["iteration"].is_monotonic_increasing and not run["iteration"].duplicated().any()
        )
        initial_rmse = float(run["state_rmse"].iloc[0]) if not run.empty else float("nan")
        final_rmse = float(classification["final_state_rmse"])
        initial_residual = (
            float(run["weighted_residual_norm"].iloc[0]) if not run.empty else float("nan")
        )
        final_residual = float(classification["final_weighted_residual"])
        audit_rows.append(
            {
                "arm": classification["arm"],
                "scenario": classification["scenario"],
                "seed": int(classification["seed"]),
                "classification": classification["classification"],
                "executed_iterations": int(classification["executed_iterations"]),
                "iteration_order_sorted_unique": sorted_unique,
                "iteration_0_state_rmse": initial_rmse,
                "final_state_rmse": final_rmse,
                "iteration_0_relative_gap_to_final_floor": (
                    abs(initial_rmse - final_rmse) / max(abs(final_rmse), 1.0e-30)
                ),
                "iteration_0_weighted_residual": initial_residual,
                "final_weighted_residual": final_residual,
                "relative_residual_reduction": (
                    (initial_residual - final_residual) / max(abs(initial_residual), 1.0e-30)
                ),
                "final_update_norm": float(classification["final_step_norm"]),
                "terminal_window_relative_rmse_change": float(
                    classification["window_relative_rmse_change"]
                ),
                "terminal_window_max_relative_swing": float(
                    classification["window_max_relative_swing"]
                ),
                "rmse_floor_entry_iteration": int(
                    classification["rmse_floor_entry_iteration"]
                ),
                "first_qualifying_window_end_iteration": int(
                    classification["first_qualifying_window_end_iteration"]
                ),
                "plateau_onset_iteration": int(
                    classification["plateau_onset_iteration"]
                ),
                "plateau_rule_satisfied_at_horizon": bool(
                    classification["plateau_rule_satisfied_at_horizon"]
                ),
                "plateau_rule_remains_satisfied_from_onset": bool(
                    classification["plateau_rule_remains_satisfied_from_onset"]
                ),
                "plateau_window": int(config["extended_horizon"]["plateau_window"]),
                "plateau_rmse_rel_tol": float(
                    config["extended_horizon"]["plateau_rmse_rel_tol"]
                ),
                "plateau_step_norm_threshold": float(
                    config["extended_horizon"]["plateau_step_norm"]
                ),
            }
        )
    atomic_write_csv(
        destination / "audits" / "trajectory_plateau_onset_audit.csv",
        pd.DataFrame(audit_rows),
    )

    summary_rows: list[dict[str, Any]] = []
    for arm, rows in cls_frame.groupby("arm", sort=True):
        plateau_onsets = rows.loc[
            rows["classification"] == "plateaued", "plateau_onset_iteration"
        ].to_numpy(dtype=float)
        selected_output_values = rows["final_selected_output_error"].to_numpy(dtype=float)
        selected_output_values = selected_output_values[np.isfinite(selected_output_values)]
        summary_rows.append(
            {
                "arm": arm,
                "runs": len(rows),
                "convergence_count": int((rows["classification"] == "converged").sum()),
                "plateau_count": int((rows["classification"] == "plateaued").sum()),
                "still_improving_count": int(
                    (rows["classification"] == "still_improving_at_horizon").sum()
                ),
                "oscillation_count": int((rows["classification"] == "oscillatory").sum()),
                "divergence_count": int((rows["classification"] == "diverged").sum()),
                "maximum_iterations_without_plateau_count": int(
                    (rows["classification"] == "maximum_iterations_without_plateau").sum()
                ),
                "median_final_state_rmse": float(np.nanmedian(rows["final_state_rmse"])),
                "median_final_weighted_residual": float(
                    np.nanmedian(rows["final_weighted_residual"])
                ),
                "median_plateau_onset_iteration": (
                    float(np.nanmedian(plateau_onsets))
                    if plateau_onsets.size
                    else float("nan")
                ),
                "median_final_selected_coordinate_rmse": float(
                    np.nanmedian(rows["final_selected_coordinate_rmse"])
                ),
                "median_final_frozen_coordinate_rmse": float(
                    np.nanmedian(rows["final_frozen_coordinate_rmse"])
                ),
                "median_final_selected_output_error": float(
                    np.median(selected_output_values)
                ) if selected_output_values.size else float("nan"),
                "median_final_update_norm": float(np.nanmedian(rows["final_step_norm"])),
            }
        )
    atomic_write_csv(
        destination / "audits" / "extended_horizon_arm_summary.csv",
        pd.DataFrame(summary_rows),
    )


def _write_audit_manifest(
    destination: Path,
    summary_frame: pd.DataFrame,
    coordinate_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    resource_audit_rows: list[dict[str, Any]],
    extended_classification_rows: list[dict[str, Any]],
    sampler_variability: dict[str, Any],
) -> None:
    finite = (
        summary_frame[summary_frame["arm"] == ARM_FINITE_SHOT]
        if not summary_frame.empty
        else pd.DataFrame()
    )
    statevector = (
        summary_frame[summary_frame["arm"] == ARM_STATEVECTOR]
        if not summary_frame.empty
        else pd.DataFrame()
    )
    ext_counts: dict[str, dict[str, int]] = {}
    for r in extended_classification_rows:
        ext_counts.setdefault(str(r["arm"]), {})
        c = str(r["classification"])
        ext_counts[str(r["arm"])][c] = ext_counts[str(r["arm"])].get(c, 0) + 1
    comparison_abs = (
        comparison["absolute_final_rmse_difference"].to_numpy(dtype=float)
        if not comparison.empty
        else np.asarray([], dtype=float)
    )
    finite_comparison_abs = comparison_abs[np.isfinite(comparison_abs)]
    plateau_onsets = np.asarray(
        [
            float(row["plateau_onset_iteration"])
            for row in extended_classification_rows
            if row["classification"] == "plateaued"
        ],
        dtype=float,
    )
    manifest = {
        "study_id": STUDY_ID,
        "audit": "closed_loop_nonlinear_qsvt_verification_2026_07_22",
        "issue_a_query_shot_accounting": {
            "finite_shot_runs": len(finite),
            "functional_queries_total": int(finite["functional_queries"].sum())
            if not finite.empty
            else 0,
            "physical_circuit_executions_total": int(finite["physical_circuit_executions"].sum())
            if not finite.empty
            else 0,
            "sampling_calls_total": int(finite["sampling_calls"].sum())
            if not finite.empty
            else 0,
            "readout_signal_attempted_shots_total": int(
                finite["readout_signal_attempted_shots"].sum()
            )
            if not finite.empty
            else 0,
            "diagnostic_attempted_shots_total": int(finite["diagnostic_attempted_shots"].sum())
            if not finite.empty
            else 0,
            "total_attempted_shots_total": int(finite["total_attempted_shots"].sum())
            if not finite.empty
            else 0,
            "unique_functional_circuits_total": int(
                coordinate_frame["functional_circuit_fingerprint"].nunique()
            )
            if not coordinate_frame.empty
            and "functional_circuit_fingerprint" in coordinate_frame
            else 0,
        },
        "issue_b_circuit_resource_levels": {
            "rows": len(resource_audit_rows),
            "note": (
                "logical_operations counts opaque unitary blocks (not primitive gates); "
                "transpiled_basis_gates is the all-to-all basis-operation count without routing"
            ),
            "basis": "+".join(TRANSPILE_BASIS),
            "optimization_level": TRANSPILE_OPT_LEVEL,
            "transpiler_seed": TRANSPILE_SEED,
            "coupling_map_assumption": TRANSPILE_COUPLING_ASSUMPTION,
            "routing_included": TRANSPILE_ROUTING_INCLUDED,
        },
        "issue_c_extended_horizon": {
            "max_iterations": int(extended_classification_rows[0]["max_iterations"])
            if extended_classification_rows
            else 0,
            "classification_counts": ext_counts,
            "plateau_onset_min": (
                int(np.min(plateau_onsets)) if plateau_onsets.size else None
            ),
            "plateau_onset_median": (
                float(np.median(plateau_onsets)) if plateau_onsets.size else None
            ),
            "plateau_onset_max": (
                int(np.max(plateau_onsets)) if plateau_onsets.size else None
            ),
        },
        "issue_d_finite_shot_evidence": {
            "finite_shot_runs": len(finite),
            "statevector_runs": len(statevector),
            "matched_comparison_rows": len(comparison),
            "max_absolute_final_rmse_difference": (
                float(np.max(finite_comparison_abs))
                if finite_comparison_abs.size
                else None
            ),
            "max_trajectory_state_norm_difference": (
                float(comparison["max_trajectory_state_norm_difference"].max())
                if not comparison.empty
                else None
            ),
            "median_coordinate_readout_abs_error": (
                float(coordinate_frame["coordinate_readout_abs_error"].median())
                if not coordinate_frame.empty
                and "coordinate_readout_abs_error" in coordinate_frame
                else None
            ),
            "max_coordinate_readout_abs_error": (
                float(coordinate_frame["coordinate_readout_abs_error"].max())
                if not coordinate_frame.empty
                and "coordinate_readout_abs_error" in coordinate_frame
                else None
            ),
            "sampler_seed_variability": sampler_variability,
        },
    }
    atomic_write_json(destination / "manifests" / "audit_manifest.json", manifest)


def _write_resource_summary(
    destination: Path, iteration_frame: pd.DataFrame, coord_frame: pd.DataFrame
) -> None:
    if iteration_frame.empty:
        atomic_write_csv(
            destination / "resource_ledgers" / "resource_summary.csv",
            pd.DataFrame(columns=["arm", "scenario", "seed"]),
        )
        return
    circuit = iteration_frame[iteration_frame["circuit_logical_operations"].astype(int) >= 0]
    cols = [
        "arm",
        "scenario",
        "seed",
        "iteration",
        "circuit_qubits",
        "circuit_logical_operations",
        "circuit_logical_depth",
        "circuit_signal_unitary_calls",
        "circuit_projector_phase_calls",
        "statevector_dim",
        "postselection_probability",
        "functional_queries_iter",
        "physical_circuit_executions_iter",
        "sampling_calls_iter",
        "shots_per_sampling_call",
        "readout_signal_attempted_shots_iter",
        "diagnostic_attempted_shots_iter",
        "total_attempted_shots_iter",
        "cumulative_shots",
        "cumulative_circuit_executions",
    ]
    present = [c for c in cols if c in circuit.columns]
    atomic_write_csv(
        destination / "resource_ledgers" / "resource_summary.csv", circuit[present].copy()
    )


def _write_failure_ledger(
    destination: Path, iteration_frame: pd.DataFrame, summary_frame: pd.DataFrame
) -> None:
    if iteration_frame.empty:
        failures = pd.DataFrame(columns=["arm", "scenario", "seed", "iteration", "failure_code"])
    else:
        failures = iteration_frame[iteration_frame["failure_code"].astype(str) != ""][
            [
                "arm",
                "scenario",
                "seed",
                "iteration",
                "failure_code",
                "failure_stage",
                "failure_message",
                "phase_synthesis_status",
                "run_classification",
            ]
        ].copy()
    atomic_write_csv(destination / "failure_ledgers" / "structured_failures.csv", failures)


def _write_claim_boundary(destination: Path, summary_frame: pd.DataFrame) -> None:
    if summary_frame.empty:
        table = pd.DataFrame(columns=["classification", "run_count"])
    else:
        table = (
            summary_frame.groupby(["arm", "classification"], sort=True)
            .size()
            .rename("run_count")
            .reset_index()
        )
    atomic_write_csv(destination / "run_summaries" / "claim_boundary_summary.csv", table)


def _write_readout_cost(
    destination: Path, coord_frame: pd.DataFrame, summary_frame: pd.DataFrame
) -> None:
    if coord_frame.empty:
        cost = pd.DataFrame(columns=["scenario", "seed", "block_dimension"])
    else:
        grouped = coord_frame.groupby(["scenario", "seed"], sort=True)
        records = []
        for (scenario, seed), block in grouped:
            n_iters = int(block["iteration"].nunique())
            per_iter_queries = int(block.groupby("iteration").size().max())
            functional_queries = len(block)
            shots_each = int(block["shots_per_coordinate"].iloc[0])
            readout_shots = int(block["readout_attempted_shots"].sum())
            diagnostic_shots = int(block["diagnostic_attempted_shots"].sum())
            records.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "block_dimension": per_iter_queries,
                    "coordinate_queries_per_iteration": per_iter_queries,
                    "shots_per_functional_query": 2 * shots_each,
                    "shots_per_sampling_call": shots_each,
                    "mean_interference_acceptance": float(
                        block["interference_acceptance_probability"].mean()
                    ),
                    "mean_direct_postselection": float(
                        block["direct_postselection_probability"].mean()
                    ),
                    "iterations": n_iters,
                    "functional_queries": functional_queries,
                    "unique_functional_circuits": functional_queries,
                    "physical_circuit_executions": 2 * functional_queries,
                    "sampling_calls": 2 * functional_queries,
                    "readout_signal_attempted_shots": readout_shots,
                    "diagnostic_attempted_shots": diagnostic_shots,
                    "total_attempted_shots": readout_shots + diagnostic_shots,
                    "total_interference_accepted_shots": int(
                        block["interference_branch_accepted_shots"].sum()
                    ),
                    "total_postselection_accepted_shots": int(
                        block["postselection_accepted_shots"].sum()
                    ),
                    "mean_sampling_standard_error": float(
                        block["sampling_standard_error"].mean()
                    ),
                }
            )
        cost = pd.DataFrame(records)
    atomic_write_csv(destination / "resource_ledgers" / "readout_cost.csv", cost)


def _write_query_execution_accounting(
    destination: Path,
    summary_frame: pd.DataFrame,
    coord_frame: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Explicit non-overlapping query/execution/shot accounting per run (Issue A)."""

    columns = [
        "arm",
        "scenario",
        "seed",
        "executed_iterations",
        "coordinates_per_iteration",
        "functional_queries",
        "unique_functional_circuits",
        "physical_circuit_executions",
        "sampling_calls",
        "shots_per_sampling_call",
        "readout_signal_attempted_shots",
        "diagnostic_attempted_shots",
        "total_attempted_shots",
        "interference_accepted_shots",
        "postselection_accepted_shots",
        "invariant_total_equals_calls_times_shots",
        "invariant_queries_equals_iters_times_coords",
        "invariant_unique_circuits_not_above_queries",
    ]
    if summary_frame.empty:
        atomic_write_csv(
            destination / "resource_ledgers" / "query_execution_accounting.csv",
            pd.DataFrame(columns=columns),
        )
        return
    shots_each = int(config["block_qsvt"]["finite_shot_budget"])
    coords = int(config["block_qsvt"]["block_size"])
    finite = summary_frame[summary_frame["arm"] == ARM_FINITE_SHOT]
    records = []
    for _, r in finite.iterrows():
        iters = int(r["iterations"])
        fq = int(r["functional_queries"])
        calls = int(r["sampling_calls"])
        total = int(r["total_attempted_shots"])
        run_coordinates = coord_frame[
            (coord_frame["scenario"] == r["scenario"])
            & (coord_frame["seed"] == r["seed"])
        ]
        unique_functional_circuits = int(
            run_coordinates["functional_circuit_fingerprint"].nunique()
            if "functional_circuit_fingerprint" in run_coordinates
            else 0
        )
        records.append(
            {
                "arm": r["arm"],
                "scenario": r["scenario"],
                "seed": int(r["seed"]),
                "executed_iterations": iters,
                "coordinates_per_iteration": coords,
                "functional_queries": fq,
                "unique_functional_circuits": unique_functional_circuits,
                "physical_circuit_executions": int(r["physical_circuit_executions"]),
                "sampling_calls": calls,
                "shots_per_sampling_call": shots_each,
                "readout_signal_attempted_shots": int(r["readout_signal_attempted_shots"]),
                "diagnostic_attempted_shots": int(r["diagnostic_attempted_shots"]),
                "total_attempted_shots": total,
                "interference_accepted_shots": int(r["interference_accepted_shots"]),
                "postselection_accepted_shots": int(r["postselection_accepted_shots"]),
                "invariant_total_equals_calls_times_shots": bool(total == calls * shots_each),
                "invariant_queries_equals_iters_times_coords": bool(fq == iters * coords),
                "invariant_unique_circuits_not_above_queries": bool(
                    unique_functional_circuits <= fq
                ),
            }
        )
    atomic_write_csv(
        destination / "resource_ledgers" / "query_execution_accounting.csv",
        pd.DataFrame(records, columns=columns),
    )
    global_unique = int(
        coord_frame["functional_circuit_fingerprint"].nunique()
        if not coord_frame.empty and "functional_circuit_fingerprint" in coord_frame
        else 0
    )
    reconciliation = [
        {
            "quantity": "finite_shot_runs",
            "verified_value": len(records),
            "definition": "executed finite-shot scenario x seed runs",
        },
        {
            "quantity": "executed_iterations",
            "verified_value": int(sum(item["executed_iterations"] for item in records)),
            "definition": "sum of executed nonlinear iterations over finite-shot runs",
        },
        {
            "quantity": "coordinates_per_iteration",
            "verified_value": coords,
            "definition": "selected block coordinates reconstructed each iteration",
        },
        {
            "quantity": "functional_queries",
            "verified_value": int(sum(item["functional_queries"] for item in records)),
            "definition": "one signed coordinate readout request that drives the update",
        },
        {
            "quantity": "unique_functional_circuits",
            "verified_value": global_unique,
            "definition": (
                "distinct matrix/residual/phase/coordinate readout parameterizations across "
                "the evidence; repeated executions are deduplicated by SHA-256"
            ),
        },
        {
            "quantity": "physical_circuit_executions",
            "verified_value": int(
                sum(item["physical_circuit_executions"] for item in records)
            ),
            "definition": "executed readout plus direct-postselection circuit branches",
        },
        {
            "quantity": "sampling_calls",
            "verified_value": int(sum(item["sampling_calls"] for item in records)),
            "definition": "Aer run invocations; one per physical execution",
        },
        {
            "quantity": "shots_per_sampling_call",
            "verified_value": shots_each,
            "definition": "attempted shots in each Aer sampling call",
        },
        {
            "quantity": "shots_per_functional_query",
            "verified_value": 2 * shots_each,
            "definition": "two sampling branches per functional query",
        },
        {
            "quantity": "total_attempted_shots",
            "verified_value": int(sum(item["total_attempted_shots"] for item in records)),
            "definition": "sampling calls x shots per call",
        },
        {
            "quantity": "postselection_accepted_shots",
            "verified_value": int(
                sum(item["postselection_accepted_shots"] for item in records)
            ),
            "definition": "accepted shots in the diagnostic direct-postselection branch",
        },
        {
            "quantity": "interference_accepted_shots",
            "verified_value": int(
                sum(item["interference_accepted_shots"] for item in records)
            ),
            "definition": "accepted shots in the readout/interference branch",
        },
    ]
    atomic_write_csv(
        destination / "resource_ledgers" / "query_execution_reconciliation.csv",
        pd.DataFrame(reconciliation),
    )


def _write_resolved_config(destination: Path, config: dict[str, Any]) -> None:
    import yaml

    (destination / "configs" / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )


def _write_readme(
    destination: Path,
    summary_frame: pd.DataFrame,
    decomposition_frame: pd.DataFrame,
    coord_frame: pd.DataFrame,
) -> None:
    lines = [
        "# Closed-Loop Nonlinear Sparse-QSVT State-Update Experiment",
        "",
        CLAIM_BOUNDARY,
        "",
        "The selected experimental arm's update actually advances the nonlinear PSSE state:",
        "`x_{k+1} = x_k + eta_k * dx_k^{method}`.  For block-based arms only the selected block",
        "state coordinates are advanced; non-selected coordinates are frozen.  No arm falls back",
        "to a different arm or to a classical solve; failures are retained as structured rows.",
        "This is a small-scale simulator feasibility experiment: no hardware execution, no",
        "full-state quantum PSSE, and no quantum speedup or advantage is claimed.",
        "",
        f"- runs: {len(summary_frame)}; decomposition rows: {len(decomposition_frame)}; "
        f"finite-shot coordinate rows: {len(coord_frame)}",
        "",
        "## Evidence directories",
        "- `iteration_ledgers/closed_loop_iterations.csv` - per-arm per-iteration ledger.",
        "- `run_summaries/solver_outcomes.csv` - solver outcome table (task 9.1).",
        "- `run_summaries/claim_boundary_summary.csv` - run classification counts (task 9.5).",
        "- `error_decomposition/stage_error_decomposition.csv` - full->block->sparse->quantized->"
        "polynomial->statevector->finite-shot stage errors along the full-system trajectory (9.2).",
        "- `resource_ledgers/finite_shot_coordinate_readout.csv` - per-coordinate signed readout.",
        "- `resource_ledgers/resource_summary.csv`, `readout_cost.csv` - circuit + readout cost.",
        "- `resource_ledgers/query_execution_accounting.csv` - non-overlapping query/shot counts.",
        "- `resource_ledgers/query_execution_reconciliation.csv` - aggregate counter definitions.",
        "- `resource_ledgers/circuit_resource_levels.csv` - logical/decomposed/transpiled levels.",
        "- `failure_ledgers/structured_failures.csv` - retained structured failures.",
        "- `extended_horizon/` - 30-iteration convergence diagnostic + plateau classification (C).",
        "- `audits/` - circuit_resource_audit, finite_shot_statevector_comparison, sampler_seed_"
        "variability, trajectory_plateau_onset_audit, and extended_horizon_arm_summary; "
        "`manifests/audit_manifest.json`.",
        "- `figures/` - convergence trajectories.  `tables/` - manuscript LaTeX tables.",
        "",
        "## Counter definitions (all non-overlapping; Issue A)",
        "- `functional_queries`: one signed selected-output (coordinate) query = executed "
        "iterations x block coordinates.  Drives the state update.",
        "- `unique_functional_circuits`: distinct matrix/residual/phase/coordinate readout "
        "parameterizations after SHA-256 deduplication; repeated executions remain counted below.",
        "- `physical_circuit_executions` = `sampling_calls`: Aer .run() invocations = 2 per "
        "functional query (readout/interference branch + direct-postselection diagnostic branch).",
        "- `shots_per_sampling_call`: shots drawn per Aer run (10^5).",
        "- `readout_signal_attempted_shots`: shots on the readout branch (signal; drives update).",
        "- `diagnostic_attempted_shots`: shots on the direct-postselection diagnostic branch "
        "(reported only; NEVER enters the update).",
        "- `total_attempted_shots` = readout + diagnostic = `sampling_calls` x "
        "`shots_per_sampling_call` (enforced invariant).",
        "- `interference_accepted_shots` / `postselection_accepted_shots`: accepted subsets "
        "(<= attempted) of the two branches.",
        "",
        "## Circuit-resource levels (Issue B)",
        "- `logical_operations`: count of high-level opaque `unitary` blocks (one projector-"
        "controlled phase or one signal-unitary application each).  NOT primitive gates.",
        "- `decomposed_operations`: after one `.decompose()`.",
        "- `transpiled_basis_gates` / `transpiled_depth`: after transpile to {rz,ry,rx,cx} "
        "(opt level 1, seed 20260722); a primitive basis-operation count.  No coupling map or "
        "backend target is supplied, so routing is not included and the result is not a device-"
        "specific hardware gate count. `opaque_instructions_remaining` is 0 when no undecomposed "
        "custom instruction remains.",
        "",
        "## Evidence tiers (kept separate, never merged)",
        "sparse support error; exact polynomial matrix action; explicit statevector circuit "
        "execution; finite-shot circuit simulation.",
        "",
        "## Reproduce",
        "```",
        "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 \\",
        "  .venv/bin/python scripts/run_tqe_closed_loop_nonlinear_update.py",
        "```",
    ]
    (destination / "manifests" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
