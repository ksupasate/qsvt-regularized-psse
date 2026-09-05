"""Phase 9: integrated 8x8 finite-shot selected-submatrix QSVT chain.

Extends the Phase 8 integrated 4x4 anchor chain to the deterministic
IEEE-14-derived ``8x8`` lambda-matched selected-submatrix anchor.  The 8x8 block
uses the *same normalized regularization* ``lambda = alpha/beta^2`` as the 4x4
correctness anchor (``alpha = lambda_anchor * beta_8^2``); at degree 31 this is a
statevector *pass* (update relative error ~1e-3 vs matched Ridge), so the same
integrated finite-shot chain construction applies at the larger block.

One shot-executed Qiskit circuit composes: (1) dense residual initialization (a
unitary ``StatePreparation`` of the padded normalized residual -- the *input*,
never the output), (2) the synthesized degree-31 QSVT sequence, (3) measured
ancilla postselection (the block-encoding flag qubit is measured every shot),
and (4) a Hadamard-type signed-overlap readout for predetermined selected
functionals.  The physical signed functional is recovered from *measured*
statistics only via ``y_hat = (C/beta) ||r_B|| ||l|| * f_hat * Xbar_acc``.  The
classically computed postselected output state is used **only** as an external
validation reference; no gate in the sampled circuit prepares it.

All heavy circuit/estimator machinery is imported verbatim from the Phase 8
module so the 4x4 and 8x8 chains are bit-for-bit the same construction; only the
anchor (block, alpha, phases) differs.

Scope / claim boundary: this remains a dense selected-submatrix demonstration on
a statevector-based shot simulator with dense controlled-unitary gates (now 5
qubits).  It does not imply scalable residual loading, IEEE-scale sparse block
encoding, full selected-output PSSE execution, or quantum competitiveness.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.circuit_signed_readout import estimate_overlap
from robust_qsvt_se.paper.phase8_integrated_readout import (
    INTEGRATED_READOUT_MODEL,
    PER_SEED_COLUMNS,
    RECOVERY_CONVENTION,
    AnchorContext,
    _chain_output_statevector,
    _classical_adjoint_reference,
    _direct_chain_p_hat,
    _exact_clbit_distribution,
    _sample_counts,
    _sha256,
    _summarize,
    build_direct_chain_circuit,
    build_integrated_readout_circuit,
    estimate_from_integrated_counts,
)
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    _state_labels_for_cols,
    run_demo_for_block,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    assert_safe,
    forbidden_in,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.qsvt.gate_state_preparation import normalize_and_pad_for_gate_preparation
from robust_qsvt_se.qsvt.phase_synthesis import synthesize_pennylane_phases_cached
from robust_qsvt_se.utils.io import ensure_directory, write_json

PHASE9_READOUT_DIR = Path("outputs/phase9_integrated_8x8_readout")
PHASE9_PHASE_CACHE = PHASE9_READOUT_DIR / "phase_cache"
MANUSCRIPT_TABLE_PATH = Path("manuscript/tables/phase9_integrated_8x8_readout.tex")

FULL_SHOTS = (1_000, 10_000, 100_000, 1_000_000)
QUICK_SHOTS = (1_000, 10_000)
PRIMARY_OBSERVABLE = "state_correction_0"

# The anchor is a statevector pass, so no additional overclaim beyond the shared
# Phase 8 recovery convention is introduced here.
BLOCK_SIZE = 8
ANCHOR_DEGREE = 31


def _anchor_lambda(H_full: np.ndarray, r_full: np.ndarray) -> float:
    """Normalized regularization ``lambda = 4 sigma_min^2 / sigma_max^2`` of the 4x4 anchor."""

    h4, _r4, _rows, _cols = select_deterministic_block(
        H_full, r_full, row_count=4, col_count=4, policy="largest_row_col_norms"
    )
    singular = np.linalg.svd(h4, compute_uv=False)
    return 4.0 * float(singular.min()) ** 2 / float(singular.max()) ** 2


def build_anchor_context_8x8(
    *,
    case: str = "ieee14",
    case_source: str = "pypower",
    system_seed: int = 123,
    degree: int = ANCHOR_DEGREE,
    phase_cache_dir: str | Path = PHASE9_PHASE_CACHE,
) -> AnchorContext:
    """Rebuild the passing 8x8 lambda-matched anchor and re-derive its circuit pieces.

    Mirrors :func:`phase8_integrated_readout.build_anchor_context` but selects the
    deterministic IEEE-14-derived 8x8 block at ``alpha = lambda_anchor * beta_8^2``
    (lambda matched to the 4x4 anchor).  The rebuilt plain chain is validated
    against the demo pipeline's postselected statevector output before any
    sampling; that demo output state is never loaded by any sampled circuit.
    """

    system, matrix_source = build_engineering_system(
        {
            "case_name": case,
            "case_source": case_source,
            "matrix_source": "weighted_jacobian",
            "seed": int(system_seed),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)

    lambda_anchor = _anchor_lambda(H_full, r_full)
    H_block, r_block, rows, cols = select_deterministic_block(
        H_full, r_full, row_count=BLOCK_SIZE, col_count=BLOCK_SIZE, policy="largest_row_col_norms"
    )
    beta_8 = float(np.linalg.svd(H_block, compute_uv=False).max())
    alpha = lambda_anchor * beta_8**2
    column_labels = _state_labels_for_cols(system.metadata, cols)

    result = run_demo_for_block(
        case=case,
        matrix_source=matrix_source,
        H_block=H_block,
        r_block=r_block,
        selected_rows=rows,
        selected_cols=cols,
        column_labels=column_labels,
        alpha=alpha,
        degree=int(degree),
        angle_solver="iterative",
        margin=1.05,
        domain_low_factor=0.9,
        pass_relative_tolerance=0.05,
        phase_cache_dir=phase_cache_dir,
    )
    if result.status_label != "pass":
        raise RuntimeError(
            f"8x8 lambda-matched anchor status is '{result.status_label}', expected 'pass'"
        )

    meta = result.pipeline_metadata
    beta = float(meta["beta"])
    n = int(result.H_block.shape[0])
    A = np.asarray(result.H_block, dtype=np.float64).T / beta
    encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    coefficients = np.asarray(meta["polynomial_coefficients_bounded"], dtype=np.float64)
    cached = synthesize_pennylane_phases_cached(
        coefficients,
        angle_solver=str(meta["angle_solver"]),
        cache_dir=phase_cache_dir,
        cache_metadata={
            "case": case,
            "block": f"{n}x{n}",
            "alpha": alpha,
            "degree": int(degree),
        },
    )
    phases = np.asarray(cached.phases, dtype=np.float64)
    if int(phases.size) != int(meta["phase_count"]):
        raise RuntimeError("re-synthesized phase count does not match the anchor metadata")

    preparation = normalize_and_pad_for_gate_preparation(
        np.asarray(result.r_block, dtype=np.float64),
        target_dimension=encoding.unitary.shape[0],
    )
    p_statevector = float(result.row_common["postselection_probability"])

    x_full = _chain_output_statevector(
        block_unitary=encoding.unitary,
        phases=phases,
        padded_residual=preparation.padded_state,
    )
    encoded = x_full[:n]
    reference = math.sqrt(p_statevector) * np.asarray(result.output_state, dtype=np.complex128)
    chain_state_error = float(np.linalg.norm(encoded - reference))
    chain_probability_error = abs(float(np.vdot(encoded, encoded).real) - p_statevector)
    if chain_state_error > 1.0e-8 or chain_probability_error > 1.0e-10:
        raise RuntimeError(
            "rebuilt 8x8 plain chain disagrees with the demo pipeline: "
            f"state error {chain_state_error:.3e}, p error {chain_probability_error:.3e}"
        )

    return AnchorContext(
        result=result,
        matrix_source=matrix_source,
        block_unitary=np.asarray(encoding.unitary, dtype=np.complex128),
        phases=phases,
        padded_residual=np.asarray(preparation.padded_state, dtype=np.complex128),
        residual_norm=float(preparation.original_norm),
        encoded_dimension=n,
        alpha=alpha,
        beta=beta,
        bound_c=float(meta["bound_C"]),
        degree=int(degree),
        statevector_postselection_probability=p_statevector,
        validation={
            "plain_chain_encoded_state_error": chain_state_error,
            "plain_chain_probability_error": chain_probability_error,
            "phase_cache_hit": float(bool(cached.cache_hit)),
            "lambda_anchor_matched": lambda_anchor,
        },
    )


def _isolated_baseline_frame(
    context: AnchorContext, signed: list[Any], shots_grid: list[int], seeds: list[int]
) -> pd.DataFrame:
    """Isolated Hadamard-overlap baseline (assumed output-state preparation) for the 8x8 anchor.

    Reproduces the previous ``4x4`` isolated-readout comparison for the 8x8 block by
    directly preparing the classically computed postselected output state and
    Hadamard-overlap sampling it -- the assumption the integrated chain removes.
    Returns per-``(observable, shots)`` mean relative error vs Ridge across seeds.
    """

    p_sv = context.statevector_postselection_probability
    psi = np.asarray(context.result.output_state, dtype=np.complex128)
    ridge_update = np.asarray(context.result.ridge_update, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for observable in signed:
        ell = np.asarray(observable.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        scale = context.bound_c / context.beta * context.residual_norm * math.sqrt(p_sv) * ell_norm
        ridge_value = float(ell @ ridge_update)
        for shots in shots_grid:
            rels: list[float] = []
            for seed in seeds:
                estimate = estimate_overlap(ell, psi, shots=int(shots), seed=int(seed))
                physical = scale * estimate.overlap_estimate
                if abs(ridge_value) > 1.0e-15:
                    rels.append(abs(physical - ridge_value) / abs(ridge_value))
            rows.append(
                {
                    "observable_label": observable.observable_id,
                    "shots": int(shots),
                    "mean_relative_error_vs_ridge": float(np.mean(rels)) if rels else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=["observable_label", "shots", "mean_relative_error_vs_ridge"])


def run_phase9_integrated_8x8_readout(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(Path(resolved["phase_cache_dir"]))
    shots_grid = [int(value) for value in resolved["shots_grid"]]
    seeds = list(range(int(resolved["num_seeds"])))

    context = build_anchor_context_8x8(
        case=str(resolved["case"]),
        case_source=str(resolved["case_source"]),
        system_seed=int(resolved["seed"]),
        degree=int(resolved["degree"]),
        phase_cache_dir=phase_cache_dir,
    )
    result = context.result
    signed = [obs for obs in result.observables if obs.sign_aware_required]
    counts_meta = qsvt_sequence_operation_counts(int(context.phases.size))

    direct_circuit = build_direct_chain_circuit(
        block_unitary=context.block_unitary,
        phases=context.phases,
        padded_residual=context.padded_residual,
    )
    # n_chain_qubits = log2(2*encoded_dimension); flag qubit index = n_chain_qubits - 1.
    n_chain_qubits = int(np.log2(context.block_unitary.shape[0]))
    direct_flag_qubit = n_chain_qubits - 1

    isolated = _isolated_baseline_frame(context, signed, shots_grid, seeds)
    p_sv = context.statevector_postselection_probability
    ridge_update = np.asarray(result.ridge_update, dtype=np.float64)
    x_encoded = math.sqrt(p_sv) * np.asarray(result.output_state, dtype=np.complex128)

    per_seed_rows: list[dict[str, Any]] = []
    circuit_records: dict[str, dict[str, Any]] = {}
    counts_examples: dict[str, Any] = {}
    warnings: list[str] = []

    for observable in signed:
        ell = np.asarray(observable.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        ell_hat = ell / ell_norm
        physical_scale = context.bound_c / context.beta * context.residual_norm * ell_norm
        ridge_value = float(ell @ ridge_update)
        z_exact = float(np.real(np.vdot(ell_hat, x_encoded)))
        statevector_value = physical_scale * z_exact

        circuit, accounting = build_integrated_readout_circuit(
            block_unitary=context.block_unitary,
            phases=context.phases,
            padded_residual=context.padded_residual,
            functional_unit=ell_hat,
        )
        flag_qubit = accounting["circuit_qubits"] - 2
        readout_qubit = accounting["circuit_qubits"] - 1
        exact_distribution = _exact_clbit_distribution(circuit, [flag_qubit, readout_qubit])
        exact_accept = sum(prob for key, prob in exact_distribution.items() if key[-1] == "0")
        exact_z = exact_distribution.get("00", 0.0) - exact_distribution.get("10", 0.0)
        accept_error = abs(exact_accept - (1.0 + p_sv) / 2.0)
        z_error = abs(exact_z - z_exact)
        if accept_error > 1.0e-9 or z_error > 1.0e-9:
            raise RuntimeError(
                "integrated 8x8 circuit disagrees with the demo pipeline: "
                f"acceptance error {accept_error:.3e}, overlap error {z_error:.3e}"
            )
        label_counts: dict[str, int] = {}
        for instruction in circuit.data:
            key = str(instruction.operation.label or instruction.operation.name)
            label_counts[key] = label_counts.get(key, 0) + 1
        circuit_records[observable.observable_id] = {
            **accounting,
            "exact_acceptance_probability": exact_accept,
            "exact_signed_overlap_z": exact_z,
            "acceptance_probability_validation_error": accept_error,
            "signed_overlap_validation_error": z_error,
            "gate_counts": {str(k): int(v) for k, v in circuit.count_ops().items()},
            "gate_label_counts": label_counts,
            "circuit_depth": int(circuit.depth()),
        }

        for shots in shots_grid:
            for seed in seeds:
                counts, backend = _sample_counts(
                    circuit, shots=shots, seed=seed, measured_qubits=[flag_qubit, readout_qubit]
                )
                direct_counts, _ = _sample_counts(
                    direct_circuit, shots=shots, seed=seed, measured_qubits=[direct_flag_qubit]
                )
                estimate = estimate_from_integrated_counts(counts, physical_scale=physical_scale)
                recovered = estimate["recovered_physical_functional"]
                abs_ridge = abs(recovered - ridge_value)
                abs_sv = abs(recovered - statevector_value)
                status = (
                    "executed_integrated_chain"
                    if backend.startswith("aer")
                    else "executed_integrated_chain_exact_distribution_proxy"
                )
                if (
                    backend == "exact_distribution_multinomial_proxy"
                    and "aer_unavailable" not in warnings
                ):
                    warnings.append("aer_unavailable")
                per_seed_rows.append(
                    {
                        "observable_label": observable.observable_id,
                        "shots": int(shots),
                        "seed": int(seed),
                        "backend": backend,
                        "status": status,
                        **{
                            key: estimate[key]
                            for key in (
                                "total_attempts",
                                "accepted_attempts",
                                "effective_shots_after_postselection",
                                "acceptance_frequency",
                                "measured_postselection_probability",
                                "measured_postselection_probability_standard_error",
                                "readout_sign_mean_accepted",
                                "signed_overlap_estimate_z",
                                "signed_overlap_standard_error",
                                "recovered_physical_functional",
                                "recovered_physical_functional_standard_error",
                            )
                        },
                        "direct_chain_measured_postselection_probability": _direct_chain_p_hat(
                            direct_counts
                        ),
                        "physical_recovery_scale": physical_scale,
                        "exact_ridge_functional": ridge_value,
                        "exact_qsvt_statevector_functional": statevector_value,
                        "absolute_error_vs_ridge": abs_ridge,
                        "relative_error_vs_ridge": abs_ridge / abs(ridge_value)
                        if abs(ridge_value) > 1.0e-15
                        else float("nan"),
                        "absolute_error_vs_statevector_qsvt": abs_sv,
                        "relative_error_vs_statevector_qsvt": abs_sv / abs(statevector_value)
                        if abs(statevector_value) > 1.0e-15
                        else float("nan"),
                        "circuit_qubits": accounting["circuit_qubits"],
                        "signal_unitary_calls_per_attempt": accounting[
                            "signal_unitary_calls_per_attempt"
                        ],
                        "projector_phase_operations_per_attempt": accounting[
                            "projector_phase_operations_per_attempt"
                        ],
                        "alternating_sequence_length": accounting["alternating_sequence_length"],
                        "output_state_used_for_preparation": False,
                    }
                )
                if seed == 0 and shots == max(shots_grid):
                    counts_examples[observable.observable_id] = {
                        "shots": int(shots),
                        "seed": int(seed),
                        "backend": backend,
                        "integrated_counts_c1c0": counts,
                        "direct_chain_counts_c0": direct_counts,
                        "bit_convention": (
                            "rightmost bit = postselection flag (0 accepts); leftmost bit = "
                            "readout ancilla after the closing Hadamard (0 -> +1, 1 -> -1)"
                        ),
                    }

    per_seed = pd.DataFrame(per_seed_rows, columns=PER_SEED_COLUMNS)
    summary = _summarize(per_seed, isolated, p_sv)

    adjoint = _classical_adjoint_reference(
        np.asarray(result.H_block, dtype=np.float64),
        np.asarray(result.r_block, dtype=np.float64),
        np.asarray(signed[0].vector, dtype=np.float64),
        alpha=context.alpha,
    )
    references = _reference_values(context, signed, ridge_update, x_encoded, adjoint, isolated)
    circuit_metadata = _circuit_metadata(context, counts_meta, circuit_records, resolved)

    artifacts = _write_outputs(
        output_dir=output_dir,
        per_seed=per_seed,
        summary=summary,
        references=references,
        circuit_metadata=circuit_metadata,
        counts_examples=counts_examples,
        context=context,
    )
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="phase9_integrated_8x8_readout",
        script_name="scripts/run_phase9_integrated_8x8_readout.py",
        command=str(resolved["command"]),
        description=(
            "Integrated 8x8 finite-shot selected-submatrix chain on the deterministic "
            "IEEE-14-derived lambda-matched anchor: dense residual initialization, the "
            "synthesized degree-31 QSVT sequence, measured ancilla postselection, and "
            "Hadamard-type signed-functional readout composed in one shot-executed 5-qubit "
            "circuit. The classically computed postselected output state is used only as a "
            "validation reference and is never prepared by the sampled circuit."
        ),
        artifacts=artifacts,
        inputs_used=[
            f"build_engineering_system:{resolved['case']}:weighted_jacobian:seed{resolved['seed']}",
        ],
        random_seeds={"shot_seeds": seeds, "system_seed": int(resolved["seed"])},
        warnings=warnings,
        failures=[],
        interpretation_boundary=(
            "One integrated 8x8 finite-shot selected-submatrix chain on a statevector-based "
            "shot simulator with dense controlled-unitary gates (5 qubits). It does not imply "
            "scalable residual loading, IEEE-scale sparse block encoding, full selected-output "
            "PSSE execution, or quantum competitiveness; larger blocks and IEEE-scale "
            "composition remain modeled."
        ),
        extra={
            "integrated_readout_model": INTEGRATED_READOUT_MODEL,
            "recovery_convention": RECOVERY_CONVENTION,
            "output_state_used_for_preparation": False,
            "signed_observables": [obs.observable_id for obs in signed],
            "shots_grid": shots_grid,
            "num_shot_seeds": len(seeds),
            "block": "ieee14_8x8_lambda_matched",
            "lambda_normalized": context.result.row_common["alpha_normalized"],
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "per_seed": per_seed,
        "summary": summary,
        "references": references,
        "circuit_metadata": circuit_metadata,
        "artifacts": artifacts,
    }


def _reference_values(
    context: AnchorContext,
    signed: list[Any],
    ridge_update: np.ndarray,
    x_encoded: np.ndarray,
    adjoint: dict[str, float],
    isolated: pd.DataFrame,
) -> dict[str, Any]:
    per_observable: dict[str, Any] = {}
    for observable in signed:
        ell = np.asarray(observable.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        scale = context.bound_c / context.beta * context.residual_norm * ell_norm
        z_exact = float(np.real(np.vdot(ell / ell_norm, x_encoded)))
        iso = isolated[isolated["observable_label"] == observable.observable_id]
        per_observable[observable.observable_id] = {
            "ridge_functional": float(ell @ ridge_update),
            "statevector_qsvt_functional": scale * z_exact,
            "exact_signed_overlap_z": z_exact,
            "physical_recovery_scale": scale,
            "isolated_readout_mean_relative_error_vs_ridge_by_shots": {
                str(int(row["shots"])): float(row["mean_relative_error_vs_ridge"])
                for _, row in iso.iterrows()
            },
        }
    common = context.result.row_common
    return {
        "case": common["case"],
        "matrix_source": context.matrix_source,
        "block_shape": common["block_shape"],
        "block_checksum": common["block_checksum"],
        "residual_checksum": common["residual_checksum"],
        "selected_rows": common["selected_rows"],
        "selected_cols": common["selected_cols"],
        "condition_number": common["condition_number"],
        "alpha": context.alpha,
        "beta": context.beta,
        "lambda_normalized": common["alpha_normalized"],
        "bound_C": context.bound_c,
        "physical_recovery_factor_C_over_beta": context.bound_c / context.beta,
        "residual_norm": context.residual_norm,
        "degree": context.degree,
        "statevector_postselection_probability": context.statevector_postselection_probability,
        "statevector_update_relative_error_vs_ridge": common["update_relative_error_vs_ridge"],
        "classical_adjoint_reference": {
            **adjoint,
            "timing_note": "Python wall-clock; diagnostic and environment-specific",
        },
        "observables": per_observable,
        "recovery_convention": RECOVERY_CONVENTION,
    }


def _circuit_metadata(
    context: AnchorContext,
    counts_meta: dict[str, int],
    circuit_records: dict[str, dict[str, Any]],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    return {
        "integrated_readout_model": INTEGRATED_READOUT_MODEL,
        "block": "ieee14_8x8_lambda_matched",
        "output_state_used_for_preparation": False,
        "residual_state_preparation": (
            "unitary StatePreparation of the padded normalized residual (input state)"
        ),
        "controlled_operation_realization": (
            "explicit dense controlled-unitary gates (block-diagonal construction on the "
            "readout ancilla); same dense small-scale idiom as the canonical block encoding"
        ),
        "postselection_measurement": (
            "block-encoding flag qubit measured every shot; acceptance = flag 0; failures "
            "retained in the shot record"
        ),
        "signed_readout_measurement": (
            "readout ancilla Hadamard interference between the chain branch and the "
            "functional branch; sign from the ancilla bit on accepted shots"
        ),
        "signal_unitary_calls_per_attempt": counts_meta["signal_unitary_calls"],
        "projector_phase_operations_per_attempt": counts_meta["projector_phase_operations"],
        "alternating_sequence_length": counts_meta["alternating_sequence_length"],
        "degree": context.degree,
        "phase_count": int(context.phases.size),
        "anchor_validation": context.validation,
        "per_observable_circuits": circuit_records,
        "config": {
            key: resolved[key] for key in ("case", "case_source", "seed", "degree", "shots_grid")
        },
    }


def _write_outputs(
    *,
    output_dir: Path,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    references: dict[str, Any],
    circuit_metadata: dict[str, Any],
    counts_examples: dict[str, Any],
    context: AnchorContext,
) -> dict[str, Path]:
    per_seed_csv = output_dir / "integrated_readout_per_seed.csv"
    summary_csv = output_dir / "integrated_readout_summary.csv"
    references_json = output_dir / "integrated_readout_reference_values.json"
    metadata_json = output_dir / "integrated_readout_circuit_metadata.json"
    counts_json = output_dir / "integrated_readout_counts_summary.json"
    readme_md = output_dir / "README.md"

    per_seed.to_csv(per_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_json(references_json, references)
    write_json(metadata_json, circuit_metadata)
    write_json(counts_json, counts_examples)
    readme_md.write_text(_readme(summary, references, context), encoding="utf-8")

    artifacts = {
        "integrated_readout_per_seed_csv": per_seed_csv,
        "integrated_readout_summary_csv": summary_csv,
        "integrated_readout_reference_values_json": references_json,
        "integrated_readout_circuit_metadata_json": metadata_json,
        "integrated_readout_counts_summary_json": counts_json,
        "readme_md": readme_md,
    }
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for _, path in sorted(artifacts.items(), key=lambda item: item[1].name)
        ),
        encoding="utf-8",
    )
    artifacts["checksums_sha256"] = checksum_path
    return artifacts


def _readme(summary: pd.DataFrame, references: dict[str, Any], context: AnchorContext) -> str:
    primary = summary[summary["observable_label"] == PRIMARY_OBSERVABLE].sort_values("shots")
    common = context.result.row_common
    lines = [
        "# Phase 9: Integrated 8x8 Finite-Shot Selected-Submatrix Chain",
        "",
        "One shot-executed 5-qubit circuit composes dense residual initialization, the "
        "synthesized degree-31 QSVT sequence, measured ancilla postselection, and "
        "Hadamard-type signed-functional readout for the deterministic IEEE-14-derived 8x8 "
        "lambda-matched selected-submatrix anchor. The 8x8 block uses the same normalized "
        "regularization lambda = alpha/beta^2 as the 4x4 correctness anchor "
        f"(lambda = {common['alpha_normalized']:.6f}); at degree 31 it is a statevector pass "
        f"(update relative error {common['update_relative_error_vs_ridge']:.3e} vs matched "
        "Ridge). The classically computed postselected output state is used only as a "
        "validation reference; no gate in the sampled circuit prepares it "
        "(`output_state_used_for_preparation = false`).",
        "",
        f"- block condition number kappa = {common['condition_number']:.4f}, "
        f"alpha = {context.alpha:.6g}, beta = {context.beta:.6g}, bounded scale C = "
        f"{context.bound_c:.4f}",
        f"- physical recovery: `{RECOVERY_CONVENTION}`",
        f"- statevector postselection probability (reference): "
        f"{context.statevector_postselection_probability:.4f}",
        f"- signal-unitary calls per attempt N_U = d = {context.degree}; projector phases "
        f"N_phi = d+1 = {context.degree + 1}; alternating length 2d+1 = "
        f"{2 * context.degree + 1}; circuit qubits = "
        f"{int(primary['circuit_qubits'].iloc[0]) if not primary.empty else 5}",
        "",
        f"## Primary functional `{PRIMARY_OBSERVABLE}` (first selected coordinate)",
        "",
        "| Shots | measured p_succ | recovered mean | rel err vs Ridge | rel err vs "
        "statevector QSVT | isolated rel err vs Ridge | seeds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {int(row['shots'])} | {row['mean_measured_postselection_probability']:.4f} | "
            f"{row['mean_recovered_physical_functional']:.4e} | "
            f"{row['mean_relative_error_vs_ridge']:.3e} | "
            f"{row['mean_relative_error_vs_statevector_qsvt']:.3e} | "
            f"{row['isolated_readout_mean_relative_error_vs_ridge']:.3e} | "
            f"{int(row['num_seeds'])} |"
        )
    lines += [
        "",
        "The isolated column repeats the assumed-output-state-preparation Hadamard-overlap "
        "experiment for the same functional on this 8x8 block; the integrated chain replaces "
        "that assumption with the measured chain itself.",
        "",
        "## Interpretation boundary",
        "",
        "This is an 8x8 dense selected-submatrix demonstration on a statevector-based shot "
        "simulator with dense controlled-unitary gates (5 qubits). It does not imply scalable "
        "residual loading, IEEE-scale sparse block encoding, full selected-output PSSE "
        "execution, or quantum competitiveness. Larger blocks and IEEE-scale composition "
        "remain modeled. Ridge/Tikhonov is the matched reference; the QSVT sequence "
        "implements the same regularized filter at the same alpha.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _sci_tex(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def build_manuscript_table(
    summary_csv: str | Path = PHASE9_READOUT_DIR / "integrated_readout_summary.csv",
    table_path: str | Path = MANUSCRIPT_TABLE_PATH,
    references_json: str | Path = PHASE9_READOUT_DIR / "integrated_readout_reference_values.json",
) -> Path:
    """Write the compact manuscript table for the primary integrated 8x8 functional."""

    import json

    summary = pd.read_csv(summary_csv)
    primary = summary[summary["observable_label"] == PRIMARY_OBSERVABLE].sort_values("shots")
    if primary.empty:
        raise RuntimeError(f"summary CSV has no {PRIMARY_OBSERVABLE} rows")
    references = json.loads(Path(references_json).read_text(encoding="utf-8"))
    ridge = float(primary["exact_ridge_functional"].iloc[0])
    statevector = float(primary["exact_qsvt_statevector_functional"].iloc[0])
    p_sv = float(primary["statevector_postselection_probability"].iloc[0])
    seeds = int(primary["num_seeds"].iloc[0])
    kappa = float(references["condition_number"])
    lam = float(references["lambda_normalized"])
    lines = [
        "% Source: outputs/phase9_integrated_8x8_readout/integrated_readout_summary.csv",
        "% Regenerate: .venv/bin/python scripts/run_phase9_integrated_8x8_readout.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Integrated $8\times8$ finite-shot selected-submatrix chain on the "
        r"deterministic IEEE-14-derived $\lambda$-matched anchor "
        f"($\\kappa{{=}}{kappa:.1f}$, $\\lambda{{=}}{lam:.3f}$): dense residual "
        r"initialization, the synthesized degree-31 QSVT sequence, measured ancilla "
        r"postselection, and Hadamard-type signed readout composed in one shot-executed "
        r"circuit (5 qubits; per attempt $N_U{=}31$ signal-unitary calls, $N_\phi{=}32$ "
        r"projector phases), for the primary functional (first selected coordinate), "
        f"mean over {seeds} shot seeds per cell. "
        r"Recovery uses the measured acceptance frequency and the recorded $C/\beta$; the "
        r"classically computed output state is never prepared by the sampled circuit. "
        r"References: matched Ridge "
        + _sci_tex(ridge, 4)
        + r", statevector QSVT "
        + _sci_tex(statevector, 4)
        + r", statevector $p_{\rm succ}="
        + f"{p_sv:.4f}"
        + r"$. "
        r"``Isolated'' repeats the assumed-output-state-preparation overlap experiment for "
        r"the same functional. This remains a selected-submatrix demonstration; it does not "
        r"imply scalable residual loading, IEEE-scale sparse block encoding, full "
        r"selected-output PSSE execution, or quantum competitiveness.}",
        r"\label{tab:phase9_integrated_8x8_readout}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{rcccccc}",
        r"\hline",
        r"Shots & meas.\ $p_{\rm succ}$ & accepted & recovered mean & rel.\ err vs Ridge & "
        r"rel.\ err vs SV & isolated \\",
        r"\hline",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"$10^{{{int(np.log10(row['shots']))}}}$ & "
            f"{float(row['mean_measured_postselection_probability']):.4f} & "
            f"{_sci_tex(float(row['mean_accepted_attempts']), 2)} & "
            f"{_sci_tex(float(row['mean_recovered_physical_functional']), 4)} & "
            f"{_sci_tex(float(row['mean_relative_error_vs_ridge']))} & "
            f"{_sci_tex(float(row['mean_relative_error_vs_statevector_qsvt']))} & "
            f"{_sci_tex(float(row['isolated_readout_mean_relative_error_vs_ridge']))} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    text = "\n".join(lines)
    assert_safe(text)
    destination = Path(table_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(PHASE9_READOUT_DIR),
        "case": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "degree": ANCHOR_DEGREE,
        "num_seeds": 30,
        "shots_grid": list(FULL_SHOTS),
        "phase_cache_dir": str(PHASE9_PHASE_CACHE),
        "command": "run_phase9_integrated_8x8_readout",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    resolved["num_seeds"] = int(resolved["num_seeds"])
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 9: integrated 8x8 finite-shot QSVT chain")
    parser.add_argument("--output-dir", default=str(PHASE9_READOUT_DIR))
    parser.add_argument("--quick", action="store_true", help="fast smoke run (6 seeds, 2 shots)")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--table-path", default=str(MANUSCRIPT_TABLE_PATH))
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the manuscript table from the existing summary CSV",
    )
    args = parser.parse_args(argv)

    if args.table_only:
        table = build_manuscript_table(
            Path(args.output_dir) / "integrated_readout_summary.csv",
            args.table_path,
            Path(args.output_dir) / "integrated_readout_reference_values.json",
        )
        print(f"Manuscript table rebuilt: {table}")
        return

    if args.quick:
        num_seeds, shots = 6, list(QUICK_SHOTS)
    else:
        num_seeds, shots = 30, list(FULL_SHOTS)
    if args.num_seeds is not None:
        num_seeds = int(args.num_seeds)

    started = time.perf_counter()
    run = run_phase9_integrated_8x8_readout(
        {
            "output_dir": args.output_dir,
            "num_seeds": num_seeds,
            "shots_grid": shots,
            "command": "scripts/run_phase9_integrated_8x8_readout.py " + " ".join(argv or []),
        }
    )
    if not args.quick:
        table = build_manuscript_table(
            Path(args.output_dir) / "integrated_readout_summary.csv",
            args.table_path,
            Path(args.output_dir) / "integrated_readout_reference_values.json",
        )
        print(f"Manuscript table: {table}")
    readme = (Path(args.output_dir) / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == [], "generated README contains forbidden wording"
    print(
        f"Integrated 8x8 readout complete in {time.perf_counter() - started:.1f}s: "
        f"{run['artifacts']['integrated_readout_summary_csv']}"
    )
    print(run["summary"].to_string(index=False, max_colwidth=40))


if __name__ == "__main__":  # pragma: no cover
    main()
