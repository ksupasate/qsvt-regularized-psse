"""Phase 3 access-matched resource and reproducibility audit.

Centered on the raw IEEE-30 ``16x16`` selected-block workload from Phase 2,
compared against the ``4x4`` primary anchor, the lambda-matched ``8x8``
secondary anchor, and the controlled ``8x8`` kappa=1e4 stress row.  The audit

* instantiates the direct-sampling selected-submatrix cost model
  ``T_Q = (q N_shots / p_succ) [T_prep + d T_U + (d+1) T_phase + T_read]`` as explicit
  per-workload repetition, signal-unitary-call, and phase-operation counts;
* re-measures the classical selected-output adjoint comparator
  ``(H^T H + alpha I) w = ell``, ``y = w^T H^T r`` on the identical
  deterministic blocks and cross-checks the recorded Phase 2 values;
* records residual-loading, block-encoding, and access-model status ledgers
  with measured-versus-modeled labels;
* writes a traceability manifest with SHA-256 checksums mapping manuscript
  tables to scripts, configurations, and generated artifacts.

Audit-only: every quantum feasibility number is read from the Phase 2 ledger
and never recomputed or replaced here.  Amplitude estimation enters as a
modeled variant only; it is not executed.  No speedup, numerical-superiority,
IEEE-scale-execution, or nonlinear-loop claim is made or implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.paper.phase2_qsvt_boundary import (
    ISOLATED_READOUT_STATUS,
    SelectedBlock,
    _build_raw_block,
    build_selected_blocks,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import array_checksum
from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_CONFIG = Path("configs/qsvt_phase3_resource_reproducibility.yaml")
ANCHOR_BLOCK_ID = "ieee14_4x4_anchor"
ADJOINT_OBSERVABLE_ID = "state_correction_0"
AE_STATUS = "modeled_only_not_executed"
UNITS_NOTE = (
    "quantum side: repetition and signal-unitary-call counts with T_U modeled unless compiled; "
    "classical side: wall-clock seconds; different units, never merged"
)

OUTPUT_FILES = (
    "README.md",
    "phase3_resource_ledger.csv",
    "phase3_resource_ledger.json",
    "classical_selected_output_timings.csv",
    "residual_loading_models.csv",
    "access_status.csv",
    "readout_scaling.csv",
    "traceability_manifest.json",
    "checksums.sha256",
)

MANUSCRIPT_TABLE_FILES = (
    "phase3_access_matched_resource.tex",
    "phase3_residual_loading_models.tex",
    "phase3_reproducibility_trace.tex",
)


@dataclass(frozen=True, slots=True)
class DirectSamplingCounts:
    """Unit counts for the modeled direct-sampling composition."""

    q: int
    shots: float
    p_succ: float
    degree: int
    signal_unitary_calls_per_attempt: int
    phase_operations_per_attempt: int
    alternating_sequence_length_per_attempt: int
    expected_attempts: float
    prep_repetitions: float
    readout_repetitions: float
    signal_unitary_calls: float


@dataclass(frozen=True, slots=True)
class ClassicalTiming:
    method: str
    value: float
    median_seconds: float
    q1_seconds: float
    q3_seconds: float
    min_seconds: float
    max_seconds: float
    repeats: int
    warmups: int


def direct_sampling_counts(
    q: int, shots: float, p_succ: float, degree: int
) -> DirectSamplingCounts:
    """Expected repetition and operation counts without amplitude amplification.

    Residual preparation and signed readout each run once per modeled attempt. The
    implemented degree-``d`` sequence contains ``d`` calls to ``U_A`` or
    ``U_A^dagger`` and ``d+1`` projector-phase operations. Thus ``2d+1`` is the
    alternating sequence length, not the block-encoding query count.
    """

    if not 0.0 < p_succ <= 1.0:
        raise ValueError("p_succ must lie in (0, 1]")
    if q < 1 or degree < 1 or shots <= 0:
        raise ValueError("q, degree must be >= 1 and shots must be positive")
    operation_counts = qsvt_sequence_operation_counts(int(degree) + 1)
    expected_attempts = float(q) * float(shots) / float(p_succ)
    return DirectSamplingCounts(
        q=int(q),
        shots=float(shots),
        p_succ=float(p_succ),
        degree=int(degree),
        signal_unitary_calls_per_attempt=operation_counts["signal_unitary_calls"],
        phase_operations_per_attempt=operation_counts["projector_phase_operations"],
        alternating_sequence_length_per_attempt=operation_counts["alternating_sequence_length"],
        expected_attempts=expected_attempts,
        prep_repetitions=expected_attempts,
        readout_repetitions=expected_attempts,
        signal_unitary_calls=expected_attempts * operation_counts["signal_unitary_calls"],
    )


def extrapolated_shots(shots: float, measured_error: float, target_error: float) -> float:
    """N^{-1/2}-only shot extrapolation; excludes preparation and postselection overhead."""

    if min(shots, measured_error, target_error) <= 0:
        raise ValueError("shots and errors must be positive")
    return float(shots) * (float(measured_error) / float(target_error)) ** 2


def _time_callable(
    method: str, func: Callable[[], float], repeats: int, warmups: int
) -> ClassicalTiming:
    for _ in range(warmups):
        func()
    timings = []
    value = math.nan
    for _ in range(repeats):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)
    samples = np.asarray(timings, dtype=np.float64)
    return ClassicalTiming(
        method=method,
        value=float(value),
        median_seconds=float(np.median(samples)),
        q1_seconds=float(np.percentile(samples, 25)),
        q3_seconds=float(np.percentile(samples, 75)),
        min_seconds=float(samples.min()),
        max_seconds=float(samples.max()),
        repeats=int(repeats),
        warmups=int(warmups),
    )


def measure_classical_selected_output(
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    ell: np.ndarray,
    repeats: int,
    warmups: int,
) -> tuple[ClassicalTiming, ClassicalTiming]:
    """Time the adjoint solve and the dense direct full-update solve for one functional.

    Adjoint form of y_ell = ell^T (H^T H + alpha I)^{-1} H^T r:
    solve (H^T H + alpha I) w = ell once, then evaluate y_ell = w^T (H^T r).
    Gram/RHS formation stays inside the timed region, matching the Phase 2 measurement.
    """

    matrix = np.asarray(H, dtype=np.float64)
    residual = np.asarray(r, dtype=np.float64)
    loading = np.asarray(ell, dtype=np.float64)

    def adjoint() -> float:
        gram = matrix.T @ matrix + float(alpha) * np.eye(matrix.shape[1])
        rhs = matrix.T @ residual
        return float(np.linalg.solve(gram, loading) @ rhs)

    def direct() -> float:
        gram = matrix.T @ matrix + float(alpha) * np.eye(matrix.shape[1])
        rhs = matrix.T @ residual
        return float(loading @ np.linalg.solve(gram, rhs))

    return (
        _time_callable("dense_adjoint_selected_output", adjoint, repeats, warmups),
        _time_callable("dense_direct_full_update", direct, repeats, warmups),
    )


def rebuild_comparison_blocks(phase2_config: dict[str, Any], seed: int) -> dict[str, SelectedBlock]:
    """Rebuild the four comparison blocks through the exact Phase 2 construction path."""

    blocks = {block.block_id: block for block in build_selected_blocks(phase2_config)}
    anchor = _build_raw_block(
        {"case_name": "ieee14", "case_source": "pypower", "block_size": 4}, seed
    )
    blocks[ANCHOR_BLOCK_ID] = anchor
    return blocks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"traceability source does not exist: {relative}")
    return path


def _load_representative_rows(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path, keep_default_na=False)
    return {str(row["run_id"]): row.to_dict() for _, row in frame.iterrows()}


def _readout_fields(rep: dict[str, Any]) -> tuple[str, float, float]:
    status = str(rep["finite_shot_readout_status"])
    if status in {ISOLATED_READOUT_STATUS, "completed", "see_existing_output"}:
        return (
            "measured",
            float(rep["finite_shot_shots"]),
            float(rep["finite_shot_mean_relative_error"]),
        )
    return "not_measured", math.nan, math.nan


def _verify_block_against_ledger(
    block: SelectedBlock,
    rep: dict[str, Any],
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    rebuilt_block = array_checksum(block.H)
    rebuilt_residual = array_checksum(block.r)
    recorded_block = str(rep["block_checksum"])
    if rebuilt_block != recorded_block:
        raise RuntimeError(
            f"block checksum mismatch for {block.block_id}: "
            f"recorded {recorded_block}, rebuilt {rebuilt_block}"
        )
    recorded_residual = str(rep["residual_checksum"])
    if recorded_residual == "see_existing_output":
        metadata = metadata_by_id.get(block.block_id)
        recorded_residual = str(metadata["residual_checksum"]) if metadata else "not_recorded"
    if recorded_residual not in {"not_recorded", rebuilt_residual}:
        raise RuntimeError(
            f"residual checksum mismatch for {block.block_id}: "
            f"recorded {recorded_residual}, rebuilt {rebuilt_residual}"
        )
    return {
        "block_checksum_recorded": str(rep["block_checksum"]),
        "block_checksum_rebuilt": rebuilt_block,
        "residual_checksum_recorded": recorded_residual,
        "residual_checksum_rebuilt": rebuilt_residual,
    }


def _audit_workload(
    spec: dict[str, Any],
    rep: dict[str, Any],
    block: SelectedBlock,
    config: dict[str, Any],
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if str(rep["classical_adjoint_observable"]) != ADJOINT_OBSERVABLE_ID:
        raise RuntimeError(
            f"unexpected recorded adjoint observable for {rep['run_id']}: "
            f"{rep['classical_adjoint_observable']}"
        )
    checks = _verify_block_against_ledger(block, rep, metadata_by_id)

    alpha = float(rep["alpha"])
    degree = int(float(rep["degree_min_feasible"]))
    p_succ = float(rep["postselection_probability"])
    counts = direct_sampling_counts(
        q=int(config["functional_count_q"]),
        shots=float(config["reference_shot_budget"]),
        p_succ=p_succ,
        degree=degree,
    )

    timing_cfg = config["adjoint_timing"]
    ell = np.zeros(block.H.shape[1], dtype=np.float64)
    ell[0] = 1.0
    adjoint, direct = measure_classical_selected_output(
        block.H,
        block.r,
        alpha,
        ell,
        repeats=int(timing_cfg["timing_repeats"]),
        warmups=int(timing_cfg["warmup_repeats"]),
    )
    recorded_value = float(rep["classical_adjoint_value"])
    value_rel_diff = abs(adjoint.value - recorded_value) / max(abs(recorded_value), 1e-300)
    rtol = float(timing_cfg["value_agreement_rtol"])
    if value_rel_diff > rtol:
        raise RuntimeError(
            f"classical adjoint value mismatch for {rep['run_id']}: fresh {adjoint.value!r} "
            f"vs recorded {recorded_value!r} (rel diff {value_rel_diff:.3e} > {rtol:.1e})"
        )
    adjoint_vs_direct = abs(adjoint.value - direct.value)

    readout_status, readout_shots, readout_error = _readout_fields(rep)
    target = float(config["target_relative_error"])
    if readout_status == "measured":
        shots_to_target = extrapolated_shots(readout_shots, readout_error, target)
        signal_calls_at_target = shots_to_target / p_succ * counts.signal_unitary_calls_per_attempt
    else:
        shots_to_target = math.nan
        signal_calls_at_target = math.nan

    return {
        "workload_id": spec["workload_id"],
        "representative_run_id": rep["run_id"],
        "case_name": rep["case_name"],
        "block_size": int(rep["block_size"]),
        "block_kind": block.block_kind,
        "kappa": float(rep["kappa"]),
        "lambda": float(rep["lambda"]),
        "alpha": alpha,
        "degree": degree,
        "phase_count": int(rep["phase_count"]),
        "postselection_probability_statevector": p_succ,
        "functional_count_q": counts.q,
        "reference_shot_budget": counts.shots,
        "signal_unitary_calls_per_attempt": counts.signal_unitary_calls_per_attempt,
        "phase_operations_per_attempt": counts.phase_operations_per_attempt,
        "alternating_sequence_length_per_attempt": (counts.alternating_sequence_length_per_attempt),
        "expected_attempts_modeled": counts.expected_attempts,
        "prep_repetitions_modeled": counts.prep_repetitions,
        "readout_repetitions_modeled": counts.readout_repetitions,
        "signal_unitary_calls_modeled": counts.signal_unitary_calls,
        "readout_status": readout_status,
        "readout_shots_measured": readout_shots,
        "readout_mean_relative_error_measured": readout_error,
        "shots_to_target_error_nhalf_extrapolation_modeled": shots_to_target,
        "signal_unitary_calls_at_extrapolated_budget_modeled": signal_calls_at_target,
        "target_relative_error": target,
        "amplitude_estimation_status": AE_STATUS,
        "classical_adjoint_median_seconds_measured": adjoint.median_seconds,
        "classical_adjoint_value_fresh": adjoint.value,
        "classical_adjoint_value_recorded_phase2": recorded_value,
        "classical_adjoint_value_rel_diff": value_rel_diff,
        "classical_direct_median_seconds_measured": direct.median_seconds,
        "classical_adjoint_vs_direct_abs_diff": adjoint_vs_direct,
        "units_note": UNITS_NOTE,
        "interpretation": spec["interpretation"],
        **checks,
        "_adjoint_timing": adjoint,
        "_direct_timing": direct,
    }


def _classical_timing_rows(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for audit in audits:
        for timing in (audit["_adjoint_timing"], audit["_direct_timing"]):
            record = asdict(timing)
            record.update(
                {
                    "workload_id": audit["workload_id"],
                    "representative_run_id": audit["representative_run_id"],
                    "case_name": audit["case_name"],
                    "block_size": audit["block_size"],
                    "alpha": audit["alpha"],
                    "observable_id": ADJOINT_OBSERVABLE_ID,
                    "classical_adjoint_value_recorded_phase2": audit[
                        "classical_adjoint_value_recorded_phase2"
                    ],
                    "abs_diff_vs_recorded_phase2": abs(
                        timing.value - audit["classical_adjoint_value_recorded_phase2"]
                    ),
                    "timing_note": (
                        "Python wall-clock; host-specific diagnostic, not a "
                        "hardware-normalized performance claim"
                    ),
                }
            )
            rows.append(record)
    return rows


def _residual_loading_rows(root: Path) -> list[dict[str, Any]]:
    repeat_note = (
        "repeats with every sampling/postselection attempt; in nonlinear AC it would "
        "also repeat after every residual/Jacobian rebuild if QSVT were placed inside "
        "the loop (it is not in this work)"
    )
    rows = [
        {
            "loading_model": "dense_selected_block_initialization",
            "implemented_status": (
                "implemented + statevector validated for small selected blocks (4x4-16x16) only"
            ),
            "cost_model": "exact dense initialization, O(2^{q_r}) rotations",
            "source_artifact": "outputs/ieee_qsvt_pipeline_boundary/state_preparation_report.csv",
        },
        {
            "loading_model": "generic_amplitude_loading",
            "implemented_status": "modeled only; no compiled loader",
            "cost_model": "O(2^{q_r}) elementary rotations, q_r = ceil(log2 m)",
            "source_artifact": (
                "outputs/tqe_revision_experiments/end_to_end_resource_case/"
                "fixed_case_resource_ledger.csv"
            ),
        },
        {
            "loading_model": "qram_coherent_oracle_loading",
            "implemented_status": "assumed access model only; hardware not synthesized",
            "cost_model": "idealized O(polylog) query; data-loading hardware not compiled",
            "source_artifact": "manuscript/tables/residual_loading_ledger.tex",
        },
        {
            "loading_model": "structured_residual_generation",
            "implemented_status": "not implemented; future work",
            "cost_model": "requires exploitable residual structure or specialized access",
            "source_artifact": "manuscript/tables/residual_loading_ledger.tex",
        },
    ]
    for row in rows:
        _require_file(root, row["source_artifact"])
        row["repeats_per_attempt"] = "yes"
        row["repeats_per_nonlinear_iteration"] = "yes (hypothetical; QSVT not in loop)"
        row["not_one_time_preprocessing"] = repeat_note
    return rows


def _access_status_rows(root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "component": "dense_block_encoding_selected_blocks",
            "status": "executed_statevector_validated",
            "scale": "4x4-16x16 selected blocks",
            "source_artifact": "outputs/ieee_qsvt_pipeline_boundary/block_encoding_report.csv",
            "note": "exact dense dilation; correctness instrument, not scalable access",
        },
        {
            "component": "phase2_qsvt_selected_block_circuits",
            "status": "executed_statevector_validated",
            "scale": "8x8 and 16x16 selected blocks (feasible rows)",
            "source_artifact": "outputs/phase2_qsvt_boundary/representative_rows.csv",
            "note": "raw IEEE-30 16x16 and controlled kappa=1e4 rows; Phase 2 evidence",
        },
        {
            "component": "sparse_lookup_oracles_Ocol_Oval",
            "status": "compiled_statevector_validated",
            "scale": "4x4 and 8x8 quantized blocks",
            "source_artifact": (
                "outputs/tqe_revision_experiments/sparse_block_encoding_demo/"
                "compiled_circuit_summary.json"
            ),
            "note": "reversible lookup only; not a complete block encoding",
        },
        {
            "component": "toy_sparse_block_encoding_wrapper",
            "status": "compiled_statevector_validated",
            "scale": "one sparsified quantized 4x4 block",
            "source_artifact": (
                "outputs/sparse_block_encoding_wrapper_demo/wrapper_demo_results.csv"
            ),
            "note": "Koenig-colorable pattern only; wrapper completeness at toy scale",
        },
        {
            "component": "ieee_scale_sparse_block_encoding",
            "status": "modeled_only",
            "scale": "full IEEE matrices",
            "source_artifact": "outputs/hardware_aware_oracle_cost_model/oracle_cost_by_case.csv",
            "note": "unary-iteration QROM lookup model; T_U is modeled, not compiled",
        },
        {
            "component": "residual_state_loading",
            "status": "implemented_small_scale_else_modeled",
            "scale": "dense init 4x4-16x16; generic/qRAM models beyond",
            "source_artifact": (
                "outputs/phase3_resource_reproducibility/residual_loading_models.csv"
            ),
            "note": "repeats per attempt; see residual-loading model table",
        },
        {
            "component": "isolated_finite_shot_signed_overlap",
            "status": "executed_with_assumed_output_state_preparation_where_stated",
            "scale": "4x4 grid (1e3-1e6 shots); 8x8 anchors at 1e5; 16x16 not measured",
            "source_artifact": (
                "outputs/tqe_revision_experiments/readout_statistics/"
                "readout_shot_scaling_summary.csv"
            ),
            "note": (
                "direct StatePreparation of classically computed postselected output; "
                "per-functional shot-noise experiment, not integrated execution"
            ),
        },
        {
            "component": "amplitude_estimation_variant",
            "status": AE_STATUS,
            "scale": "none (model only)",
            "source_artifact": "manuscript/main.tex",
            "note": "O(q/(eps sqrt(p_succ))) model; needs coherent controlled prep/QSVT/readout",
        },
    ]
    for row in rows:
        if row["source_artifact"].startswith("outputs/phase3_resource_reproducibility"):
            continue
        _require_file(root, row["source_artifact"])
    return rows


def _readout_scaling_rows(
    root: Path, audits: list[dict[str, Any]], target: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_path = _require_file(
        root,
        "outputs/tqe_revision_experiments/readout_statistics/readout_shot_scaling_summary.csv",
    )
    grid = pd.read_csv(grid_path)
    pooled = grid[grid["observable_label"] == "__all_signed_pooled__"]
    rows.extend(
        {
            "workload_id": "ieee14_4x4_primary_anchor",
            "kind": "measured_grid_pooled_signed",
            "shots": float(row["shots"]),
            "mean_relative_error": float(row["mean_relative_error_vs_ridge"]),
            "basis": "30 seeds x 3 signed functionals, Hadamard test",
            "source_artifact": (
                "outputs/tqe_revision_experiments/readout_statistics/"
                "readout_shot_scaling_summary.csv"
            ),
        }
        for _, row in pooled.iterrows()
    )
    for audit in audits:
        if audit["readout_status"] == "measured":
            rows.append(
                {
                    "workload_id": audit["workload_id"],
                    "kind": "measured_representative_row",
                    "shots": audit["readout_shots_measured"],
                    "mean_relative_error": audit["readout_mean_relative_error_measured"],
                    "basis": "mean over signed functionals at the recorded budget",
                    "source_artifact": "outputs/phase2_qsvt_boundary/representative_rows.csv",
                }
            )
            rows.append(
                {
                    "workload_id": audit["workload_id"],
                    "kind": f"modeled_nhalf_extrapolation_to_{target:g}",
                    "shots": audit["shots_to_target_error_nhalf_extrapolation_modeled"],
                    "mean_relative_error": target,
                    "basis": (
                        "N^{-1/2}-only scaling estimate; excludes state preparation and "
                        "postselection overhead; not a measured result"
                    ),
                    "source_artifact": (
                        "outputs/phase3_resource_reproducibility/phase3_resource_ledger.csv"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "workload_id": audit["workload_id"],
                    "kind": "not_measured",
                    "shots": math.nan,
                    "mean_relative_error": math.nan,
                    "basis": (
                        "isolated finite-shot signed-overlap experiment not run for this "
                        "row; budget-modeled only"
                    ),
                    "source_artifact": "outputs/phase2_qsvt_boundary/representative_rows.csv",
                }
            )
    return rows


def _sci_body(value: float, digits: int = 2) -> str:
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    rounded = round(mantissa, digits)
    if abs(rounded) >= 10.0:
        rounded /= 10.0
        exponent += 1
    return f"{rounded:.{digits}f}{{\\times}}10^{{{exponent}}}"


def _sci_tex(value: float, digits: int = 2) -> str:
    if not math.isfinite(value) or value == 0:
        return "--"
    return f"${_sci_body(value, digits)}$"


def _kappa_tex(audit: dict[str, Any]) -> str:
    kappa = audit["kappa"]
    if audit["block_kind"] == "ieee_derived_condition_controlled_stress_block":
        return f"$10^{{{round(math.log10(kappa))}}}$"
    return f"{kappa:.2f}"


_TEX_ROW_LABELS = {
    "ieee14_4x4_primary_anchor": ("$4\\times4$", "IEEE-14 anchor", "primary anchor"),
    "ieee14_8x8_lambda_matched_anchor": (
        "$8\\times8$",
        "IEEE-14 $\\lambda$-matched",
        "secondary anchor",
    ),
    "ieee14_8x8_condition_controlled_k1e4": (
        "$8\\times8$",
        "IEEE-14 controlled",
        "smooth-$\\lambda$ stress; readout-limited",
    ),
    "ieee30_16x16_raw": (
        "$16\\times16$",
        "IEEE-30 raw",
        "strongest IEEE-derived row",
    ),
}


def _write_access_matched_table(path: Path, audits: list[dict[str, Any]]) -> None:
    controlled = next(
        audit for audit in audits if audit["workload_id"] == "ieee14_8x8_condition_controlled_k1e4"
    )
    rounded_extrapolation = extrapolated_shots(
        controlled["readout_shots_measured"],
        round(controlled["readout_mean_relative_error_measured"], 3),
        controlled["target_relative_error"],
    )
    be_at_extrapolation = (
        rounded_extrapolation
        / controlled["postselection_probability_statevector"]
        * controlled["signal_unitary_calls_per_attempt"]
    )
    lines = [
        "% Source: outputs/phase3_resource_reproducibility/phase3_resource_ledger.csv",
        "% Regenerate: .venv/bin/python scripts/run_phase3_resource_reproducibility.py"
        " --config configs/qsvt_phase3_resource_reproducibility.yaml",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Access-matched selected-submatrix resource accounting from"
        " \\eqref{eq:cost_direct_sampling} with $q{=}1$ functional and reference budget"
        " $N_{\\rm shots}{=}10^{5}$ accepted shots. Attempts"
        " ${=}\\,qN_{\\rm shots}/p_{\\rm post}$; every attempt repeats residual-state"
        " preparation, $d$ signal-unitary calls, $d{+}1$ projector phases, and one"
        " signed-overlap readout. The attempt and call totals are modeled counts built"
        " from each row's"
        " statevector-estimated $p_{\\rm post}$ and synthesized degree. Block encodings"
        " are compiled dense only at these selected-block scales; per-call $T_U$"
        " at larger scale remains the sparse-lookup model. Readout reports the measured"
        " $10^{5}$-shot mean relative signed-functional error where available. The"
        " classical column is the measured median wall-clock time of the matched adjoint"
        " solve \\eqref{eq:classical_adjoint} on the same block, $\\alpha$, and"
        " functional (30 repeats after 3 warmups); quantum signal-unitary-call counts"
        " and classical"
        " seconds are different units and are never merged. For these fixed"
        " selected-block workloads the accounting does not support quantum"
        " competitiveness; its value is identifying the dominant bottlenecks:"
        " postselection-inflated sampling, per-attempt residual loading, and"
        " signed-functional readout. The shot result directly prepares the classically"
        " computed postselected output; this table models, but does not execute, its"
        " composition with residual loading and QSVT.}",
        "\\label{tab:phase3_access_matched_resource}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{tabular}{llcccccccccl}",
        "\\hline",
        "Block & Source/type & $\\kappa$ & $\\lambda$ & $d$ & $p_{\\rm post}$ &"
        " $N_U$ & $N_\\phi$ & Readout err.\\ @$10^{5}$ & Attempts & $U_A$ calls &"
        " Adjoint (s) / interpretation \\\\",
        "\\hline",
    ]
    for audit in audits:
        block_tex, source_tex, interp_tex = _TEX_ROW_LABELS[audit["workload_id"]]
        readout = (
            _sci_tex(audit["readout_mean_relative_error_measured"])
            if audit["readout_status"] == "measured"
            else "not meas."
        )
        lines.append(
            f"{block_tex} & {source_tex} & {_kappa_tex(audit)} &"
            f" {_sci_tex(audit['lambda'])} & {audit['degree']} &"
            f" {audit['postselection_probability_statevector']:.3f} &"
            f" {audit['signal_unitary_calls_per_attempt']} &"
            f" {audit['phase_operations_per_attempt']} & {readout} &"
            f" {_sci_tex(audit['expected_attempts_modeled'])} &"
            f" {_sci_tex(audit['signal_unitary_calls_modeled'])} &"
            f" {_sci_tex(audit['classical_adjoint_median_seconds_measured'], 1)} /"
            f" {interp_tex} \\\\"
        )
    footnote = (
        "\\multicolumn{12}{@{}p{0.98\\textwidth}@{}}{\\footnotesize Readout entries are"
        " measured isolated $10^{5}$-shot mean relative signed-functional errors under"
        " assumed output-state preparation; ``not meas.'' means the isolated shot"
        " experiment was not run and the totals are budget-modeled."
        " Reaching $10^{-2}$ readout error on the controlled row under $N^{-1/2}$-only"
        " scaling needs"
        f" $10^{{5}}(0.154/0.01)^{{2}}\\approx{_sci_body(rounded_extrapolation)}$ shots"
        f" (${_sci_body(be_at_extrapolation)}$ signal-unitary calls), a modeled"
        " extrapolation that excludes state-preparation and postselection overhead."
        " The IEEE-30 $16\\times16$ row keeps selected-block assumptions and is not"
        " IEEE-scale sparse block-encoded execution.} \\\\"
    )
    lines.extend(
        [
            "\\hline",
            footnote,
            "\\hline",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_residual_loading_table(path: Path, rows: list[dict[str, Any]]) -> None:
    display = {
        "dense_selected_block_initialization": (
            "Dense selected-block initialization",
            "implemented + statevector validated;"
            " small selected blocks ($4\\times4$--$16\\times16$) only",
            "exact $O(2^{q_r})$ rotations",
        ),
        "generic_amplitude_loading": (
            "Generic amplitude loading",
            "modeled only; no compiled loader",
            "$O(2^{q_r})$, $q_r=\\lceil\\log_2 m\\rceil$",
        ),
        "qram_coherent_oracle_loading": (
            "qRAM/coherent-oracle loading",
            "assumed access model only; hardware not synthesized",
            "idealized $O(\\mathrm{polylog})$ query",
        ),
        "structured_residual_generation": (
            "Structured residual generation",
            "not implemented; future work",
            "requires exploitable residual structure",
        ),
    }
    lines = [
        "% Source: outputs/phase3_resource_reproducibility/residual_loading_models.csv",
        "% Regenerate: .venv/bin/python scripts/run_phase3_resource_reproducibility.py"
        " --config configs/qsvt_phase3_resource_reproducibility.yaml",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Residual-loading models in the access-matched accounting. Residual"
        " loading is not a one-time preprocessing cost: $T_{\\rm prep}$ repeats inside"
        " every sampling and postselection attempt in \\eqref{eq:cost_direct_sampling},"
        " and in nonlinear AC it would also repeat after every residual/Jacobian rebuild"
        " if QSVT were placed inside the loop (it is not in this work).}",
        "\\label{tab:phase3_residual_loading_models}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}p{0.30\\columnwidth}p{0.34\\columnwidth}p{0.26\\columnwidth}@{}}",
        "\\toprule",
        "Loading model & Status & Cost model \\\\",
        "\\midrule",
    ]
    lines.extend(
        f"{display[row['loading_model']][0]} & {display[row['loading_model']][1]} &"
        f" {display[row['loading_model']][2]} \\\\"
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _paths_tex(*names: str) -> str:
    """Comma-joined \\path{} entries; \\path breaks at / _ . so p-columns can wrap."""

    return ", ".join(f"\\path{{{name}}}" for name in names)


def _write_reproducibility_trace_table(path: Path) -> None:
    rows = [
        (
            "Selected-block boundary ledger",
            _paths_tex("run_phase2_qsvt_boundary.py", "qsvt_phase2_boundary.yaml"),
            _paths_tex(
                "phase2_qsvt_boundary/all_attempts.csv",
                "representative_rows.csv",
                "selected_block_metadata.json",
            ),
            "seed 123; tolerance $10^{-2}$; degree ceilings 45--201",
        ),
        (
            "Access-matched resources",
            _paths_tex(
                "run_phase3_resource_reproducibility.py",
                "qsvt_phase3_resource_reproducibility.yaml",
            ),
            _paths_tex("phase3_resource_reproducibility/phase3_resource_ledger.csv"),
            "$q{=}1$; $N_{\\rm shots}{=}10^{5}$; block checksums re-verified",
        ),
        (
            "Classical selected-functional adjoint timings on the matched submatrix",
            _paths_tex(
                "run_phase3_resource_reproducibility.py",
                "qsvt_phase3_resource_reproducibility.yaml",
            ),
            _paths_tex("phase3_resource_reproducibility/classical_selected_output_timings.csv"),
            "30 repeats, 3 warmups; values cross-checked to the recorded boundary ledger",
        ),
        (
            "Residual loading and access status",
            _paths_tex(
                "run_phase3_resource_reproducibility.py",
                "qsvt_phase3_resource_reproducibility.yaml",
            ),
            _paths_tex(
                "phase3_resource_reproducibility/residual_loading_models.csv",
                "access_status.csv",
            ),
            "status audit of existing artifacts; no new experiment",
        ),
        (
            "Isolated finite-shot signed-overlap scaling",
            _paths_tex("run_tqe_revision_readout_statistics.py") + " (script defaults)",
            _paths_tex(
                "tqe_revision_experiments/readout_statistics/readout_shot_scaling_summary.csv"
            ),
            (
                "30 seeds; $10^{3}$--$10^{6}$ shots; direct preparation of the "
                "classically computed output state"
            ),
        ),
        (
            "Fixed-case $4\\times4$ ledger",
            _paths_tex("run_tqe_revision_resource_ledger.py") + " (script defaults)",
            _paths_tex(
                "tqe_revision_experiments/end_to_end_resource_case/fixed_case_resource_ledger.csv"
            ),
            "one signed functional; measured circuit $p_{\\rm post}$",
        ),
        (
            "Full-system classical baselines",
            _paths_tex("run_classical_selected_observable_baseline.py") + " (script defaults)",
            _paths_tex("classical_selected_observable_baseline/baseline_summary.csv"),
            "30 timed runs after 3 warmups; IEEE 14--300",
        ),
        (
            "Selected-submatrix/full-system bridge",
            _paths_tex("run_selected_block_bridge.py") + " (script defaults)",
            _paths_tex("selected_block_bridge/selected_block_full_system_bridge.csv"),
            "same alpha and corresponding first state coordinate; seed 123",
        ),
        (
            "PSSE parameter and rank audit",
            _paths_tex("run_psse_assumption_audit.py") + " (script defaults)",
            _paths_tex(
                "psse_assumption_audit/psse_experiment_parameters.csv",
                "psse_rank_diagnostics.csv",
            ),
            "generated measurement rows; per-unit basis; explicit rank cutoff",
        ),
    ]
    lines = [
        "% Source: outputs/phase3_resource_reproducibility/traceability_manifest.json",
        "% Regenerate: .venv/bin/python scripts/run_phase3_resource_reproducibility.py"
        " --config configs/qsvt_phase3_resource_reproducibility.yaml",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Reproducibility trace for the access-matched audit. Each row maps a"
        " manuscript result to its generating script and configuration (under"
        " \\texttt{scripts/} and \\texttt{configs/}) and primary artifacts (under"
        " \\texttt{outputs/}). The machine-readable"
        " \\texttt{traceability\\_manifest.json} in"
        " \\texttt{outputs/phase3\\_resource\\_reproducibility/} records the same mapping"
        " with SHA-256 checksums and lists only files that exist;"
        " \\texttt{checksums.sha256} covers the audit directory. Functional values and"
        " block checksums are deterministic (seed 123); wall-clock timings are"
        " host-specific diagnostics.}",
        "\\label{tab:phase3_reproducibility_trace}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}p{0.25\\textwidth}p{0.26\\textwidth}p{0.27\\textwidth}"
        "p{0.16\\textwidth}@{}}",
        "\\toprule",
        "Result & Script and config & Primary artifacts & Settings \\\\",
        "\\midrule",
    ]
    lines.extend(
        f"{result} & {script_config} & {artifact} & {settings} \\\\"
        for result, script_config, artifact, settings in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _environment_record() -> dict[str, str]:
    import scipy

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def _build_manifest(
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    audits: list[dict[str, Any]],
    output_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()

    referenced: list[str] = [
        "scripts/run_phase3_resource_reproducibility.py",
        "scripts/run_phase2_qsvt_boundary.py",
        "scripts/run_tqe_revision_readout_statistics.py",
        "scripts/run_tqe_revision_resource_ledger.py",
        "scripts/run_classical_selected_observable_baseline.py",
        "scripts/run_selected_workload_extension.py",
        "src/robust_qsvt_se/paper/phase3_resource_reproducibility.py",
        "src/robust_qsvt_se/paper/phase2_qsvt_boundary.py",
        rel(config_path),
        str(config["phase2_config"]),
        "outputs/phase2_qsvt_boundary/all_attempts.csv",
        "outputs/phase2_qsvt_boundary/representative_rows.csv",
        "outputs/phase2_qsvt_boundary/selected_block_metadata.json",
        "outputs/phase2_qsvt_boundary/phase2_summary.json",
        "outputs/tqe_revision_experiments/readout_statistics/readout_shot_scaling_summary.csv",
        "outputs/tqe_revision_experiments/end_to_end_resource_case/fixed_case_resource_ledger.csv",
        (
            "outputs/tqe_revision_experiments/end_to_end_resource_case/"
            "classical_adjoint_baseline.csv"
        ),
        "outputs/classical_selected_observable_baseline/baseline_summary.csv",
        "outputs/qsvt_selected_workload_extension/selected_workload_results.csv",
        "outputs/ieee_qsvt_pipeline_boundary/state_preparation_report.csv",
        "outputs/ieee_qsvt_pipeline_boundary/block_encoding_report.csv",
        "outputs/tqe_revision_experiments/sparse_block_encoding_demo/compiled_circuit_summary.json",
        "outputs/sparse_block_encoding_wrapper_demo/wrapper_demo_results.csv",
        "outputs/hardware_aware_oracle_cost_model/oracle_cost_by_case.csv",
        "manuscript/main.tex",
        "manuscript/tables/phase2_qsvt_boundary_evidence.tex",
        "manuscript/tables/residual_loading_ledger.tex",
    ]
    referenced += [rel(output_dir / name) for name in OUTPUT_FILES[:-2]]
    referenced += [rel(tables_dir / name) for name in MANUSCRIPT_TABLE_FILES]
    checksums = {
        relative: _sha256_file(_require_file(root, relative))
        for relative in sorted(set(referenced))
    }

    tables = [
        {
            "manuscript_table": rel(tables_dir / "phase3_access_matched_resource.tex"),
            "label": "tab:phase3_access_matched_resource",
            "script": "scripts/run_phase3_resource_reproducibility.py",
            "config": rel(config_path),
            "inputs": [
                "outputs/phase2_qsvt_boundary/representative_rows.csv",
                "outputs/phase2_qsvt_boundary/selected_block_metadata.json",
            ],
            "primary_artifact": rel(output_dir / "phase3_resource_ledger.csv"),
            "settings": {
                "seed": int(config["seed"]),
                "functional_count_q": int(config["functional_count_q"]),
                "reference_shot_budget": float(config["reference_shot_budget"]),
                "target_relative_error": float(config["target_relative_error"]),
                "adjoint_timing": dict(config["adjoint_timing"]),
            },
        },
        {
            "manuscript_table": rel(tables_dir / "phase3_residual_loading_models.tex"),
            "label": "tab:phase3_residual_loading_models",
            "script": "scripts/run_phase3_resource_reproducibility.py",
            "config": rel(config_path),
            "inputs": [
                "outputs/ieee_qsvt_pipeline_boundary/state_preparation_report.csv",
                "manuscript/tables/residual_loading_ledger.tex",
            ],
            "primary_artifact": rel(output_dir / "residual_loading_models.csv"),
            "settings": {"kind": "status audit; no new experiment"},
        },
        {
            "manuscript_table": rel(tables_dir / "phase3_reproducibility_trace.tex"),
            "label": "tab:phase3_reproducibility_trace",
            "script": "scripts/run_phase3_resource_reproducibility.py",
            "config": rel(config_path),
            "inputs": sorted(set(referenced)),
            "primary_artifact": rel(output_dir / "traceability_manifest.json"),
            "settings": {"kind": "checksummed mapping of results to artifacts"},
        },
        {
            "manuscript_table": "manuscript/tables/phase2_qsvt_boundary_evidence.tex",
            "label": "tab:phase2_qsvt_boundary",
            "script": "scripts/run_phase2_qsvt_boundary.py",
            "config": str(config["phase2_config"]),
            "inputs": ["outputs/phase2_qsvt_boundary/representative_rows.csv"],
            "primary_artifact": "outputs/phase2_qsvt_boundary/all_attempts.csv",
            "settings": {"seed": 123, "note": "Phase 2 evidence; unchanged by this audit"},
        },
    ]

    workloads = [
        {
            "workload_id": audit["workload_id"],
            "representative_run_id": audit["representative_run_id"],
            "block_kind": audit["block_kind"],
            "seed": int(config["seed"]),
            "block_checksum_recorded": audit["block_checksum_recorded"],
            "block_checksum_rebuilt": audit["block_checksum_rebuilt"],
            "residual_checksum_recorded": audit["residual_checksum_recorded"],
            "residual_checksum_rebuilt": audit["residual_checksum_rebuilt"],
            "classical_adjoint_value_rel_diff": audit["classical_adjoint_value_rel_diff"],
        }
        for audit in audits
    ]

    return {
        "generated_by": "scripts/run_phase3_resource_reproducibility.py",
        "config": rel(config_path),
        "environment": _environment_record(),
        "claim_boundary": (
            "Access-matched selected-submatrix resource and reproducibility audit for the "
            "same matched-alpha Ridge/Tikhonov spectral filter. Quantum totals are "
            "repetition and signal-unitary-call counts with modeled T_U unless compiled; classical "
            "timings are wall-clock seconds; the units are different and never merged. "
            "The accounting does not support quantum competitiveness for these fixed "
            "selected-block workloads and makes no speedup, numerical-superiority, "
            "IEEE-scale-execution, full-vector, field-data, or nonlinear-loop claim."
        ),
        "units_note": UNITS_NOTE,
        "amplitude_estimation_status": AE_STATUS,
        "manuscript_tables": tables,
        "workloads": workloads,
        "file_checksums_sha256": checksums,
    }


def _write_checksums_file(output_dir: Path) -> None:
    lines = [
        f"{_sha256_file(output_dir / name)}  {name}"
        for name in OUTPUT_FILES
        if name != "checksums.sha256" and (output_dir / name).is_file()
    ]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_traceability_checksums(output_dir: Path, root: Path | None = None) -> dict[str, Any]:
    """Refresh only the hashes in the existing manuscript-facing trace manifest.

    This maintenance path deliberately does not rebuild blocks, rerun timings, or
    rewrite scientific result rows. It is used after an intentional manuscript
    revision so the provenance ledger follows the current reader-facing sources.
    """

    repository_root = Path.cwd() if root is None else root
    manifest_path = output_dir / "traceability_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    recorded = manifest.get("file_checksums_sha256", {})
    if not recorded:
        raise ValueError(f"empty or missing checksum mapping: {manifest_path}")
    missing = [relative for relative in recorded if not (repository_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "traceability manifest lists missing file(s): " + ", ".join(sorted(missing))
        )
    manifest["file_checksums_sha256"] = {
        relative: _sha256_file(repository_root / relative) for relative in sorted(recorded)
    }
    manifest["traceability_refresh"] = {
        "mode": "metadata_only",
        "scientific_rows_recomputed": False,
        "reason": "intentional evidence-locked manuscript revision",
    }
    write_json(manifest_path, manifest)
    _write_checksums_file(output_dir)
    return manifest


def _write_readme(output_dir: Path, config: dict[str, Any]) -> None:
    text = f"""# Phase 3 access-matched resource and reproducibility audit

Generated by:

```bash
.venv/bin/python scripts/run_phase3_resource_reproducibility.py \\
    --config configs/qsvt_phase3_resource_reproducibility.yaml
```

## Scope

Access-matched selected-submatrix accounting centered on the raw IEEE-30 16x16
workload from Phase 2, with the 4x4 primary anchor, the lambda-matched 8x8
secondary anchor, and the controlled kappa=1e4 8x8 stress row as comparison
rows. Quantum feasibility numbers (degree, phase count, postselection
probability, update error, and isolated finite-shot signed-overlap error) are read from
`outputs/phase2_qsvt_boundary/` and are never recomputed here.

## Resource model

Direct sampling (no amplitude amplification):
`T_Q = (q N_shots / p_succ) [T_prep + d T_U + (d+1) T_phase + T_read]`.
Residual preparation repeats inside every sampling/postselection attempt;
T_U is modeled unless the block encoding is actually compiled at that scale;
quantum signal-unitary-call counts and classical wall-clock timings are different units and
are never merged. Amplitude estimation is included as a modeled variant only
(`O(q/(eps sqrt(p_succ))) [T_prep + d T_U + T_read]`); it is not executed.

The finite-shot evidence is an isolated overlap experiment that directly loads
the classically computed postselected output state. The complete direct-sampling
formula is therefore a modeled composition, not an integrated shot execution.

## Classical comparator

The classical selected-output adjoint solve evaluates the same functional
directly: solve `(H^T H + alpha I) w = ell`, then `y = w^T H^T r`, with
Gram/RHS formation inside the timed region ({config["adjoint_timing"]["timing_repeats"]}
repeats after {config["adjoint_timing"]["warmup_repeats"]} warmups). Blocks are rebuilt through the
exact Phase 2 deterministic construction (seed {config["seed"]}), block and residual
checksums are verified against the recorded ledger, and the fresh functional
values are cross-checked against the recorded Phase 2 values. Timings vary run
to run; functional values and checksums are deterministic.

## Files

- `phase3_resource_ledger.csv`: per-workload access-matched resource rows.
- `phase3_resource_ledger.json`: the same per-workload rows in JSON form, with
  the units note and amplitude-estimation status attached.
- `classical_selected_output_timings.csv`: adjoint and dense-direct timings.
- `residual_loading_models.csv`: loading models with implemented/modeled/assumed
  status. Residual loading is not one-time preprocessing: it repeats per
  attempt, and in nonlinear AC it would repeat after every residual/Jacobian
  rebuild if QSVT were placed inside the loop (it is not in this work).
- `access_status.csv`: block-encoding/access component status.
- `readout_scaling.csv`: measured readout errors and labeled N^-1/2
  extrapolations.
- `traceability_manifest.json`: manuscript-table-to-artifact mapping with
  SHA-256 checksums; lists only files that exist.
- `checksums.sha256`: checksums of the files in this directory.

## Claim boundary

For the fixed selected-block workloads here, the accounting does not support
quantum competitiveness; its value is identifying the dominant bottlenecks.
The raw IEEE-30 16x16 row is the strongest IEEE-derived selected-block
success, but it still uses selected-block assumptions and is not IEEE-scale
sparse block-encoded execution. No speedup, numerical-superiority,
full-vector, field-data, or nonlinear-loop claim.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_phase3_audit(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = Path.cwd()
    output_dir = ensure_directory(config["output_dir"])
    tables_dir = ensure_directory(config["manuscript_tables_dir"])
    phase2_dir = Path(config["phase2_output_dir"])

    with Path(config["phase2_config"]).open(encoding="utf-8") as handle:
        phase2_config = yaml.safe_load(handle)
    rep_rows = _load_representative_rows(phase2_dir / "representative_rows.csv")
    with (phase2_dir / "selected_block_metadata.json").open(encoding="utf-8") as handle:
        metadata_by_id = {entry["block_id"]: entry for entry in json.load(handle)}

    blocks = rebuild_comparison_blocks(phase2_config, int(config["seed"]))
    audits = [
        _audit_workload(
            spec,
            rep_rows[str(spec["representative_run_id"])],
            blocks[str(spec["block_id"])],
            config,
            metadata_by_id,
        )
        for spec in config["workloads"]
    ]

    ledger_rows = [
        {key: value for key, value in audit.items() if not key.startswith("_")} for audit in audits
    ]
    pd.DataFrame(ledger_rows).to_csv(output_dir / "phase3_resource_ledger.csv", index=False)
    write_json(
        output_dir / "phase3_resource_ledger.json",
        {
            "generated_by": "scripts/run_phase3_resource_reproducibility.py",
            "units_note": UNITS_NOTE,
            "amplitude_estimation_status": AE_STATUS,
            "workloads": [
                {
                    key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in row.items()
                }
                for row in ledger_rows
            ],
        },
    )
    pd.DataFrame(_classical_timing_rows(audits)).to_csv(
        output_dir / "classical_selected_output_timings.csv", index=False
    )
    loading_rows = _residual_loading_rows(root)
    pd.DataFrame(loading_rows).to_csv(output_dir / "residual_loading_models.csv", index=False)
    pd.DataFrame(_access_status_rows(root)).to_csv(output_dir / "access_status.csv", index=False)
    pd.DataFrame(
        _readout_scaling_rows(root, audits, float(config["target_relative_error"]))
    ).to_csv(output_dir / "readout_scaling.csv", index=False)
    _write_readme(output_dir, config)

    _write_access_matched_table(tables_dir / "phase3_access_matched_resource.tex", audits)
    _write_residual_loading_table(tables_dir / "phase3_residual_loading_models.tex", loading_rows)
    _write_reproducibility_trace_table(tables_dir / "phase3_reproducibility_trace.tex")

    manifest = _build_manifest(root, config, config_path, audits, output_dir, tables_dir)
    write_json(output_dir / "traceability_manifest.json", manifest)
    _write_checksums_file(output_dir)
    return manifest


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--refresh-traceability-only",
        action="store_true",
        help="refresh existing manifest/checksum metadata without recomputing scientific rows",
    )
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    if arguments.refresh_traceability_only:
        manifest = refresh_traceability_checksums(Path(config["output_dir"]))
    else:
        manifest = run_phase3_audit(config, arguments.config)
    summary = {
        "output_dir": str(config["output_dir"]),
        "workloads": [entry["workload_id"] for entry in manifest["workloads"]],
        "checksummed_files": len(manifest["file_checksums_sha256"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
