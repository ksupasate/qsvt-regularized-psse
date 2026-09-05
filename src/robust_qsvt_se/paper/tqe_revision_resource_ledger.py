"""Experiment C: fixed-case end-to-end selected-observable resource ledger.

One concrete, fixed case (the passing IEEE-14 4x4 weighted-Jacobian block from the
selected-observable QSVT demonstration) is accounted end-to-end and compared,
numerically, against the classical selected-observable adjoint baseline that
computes the *same* regularized functional at the *same* alpha:

    y_l = l^T (H~^T H~ + alpha I)^{-1} H~^T r~ ,   adjoint: (H~^T H~ + alpha I) w = l,
                                                            y_l = w^T (H~^T r~).

The ledger deliberately separates *implemented* circuit/statevector quantities,
*finite-shot* sampling quantities, *proxy* small-simulator stand-ins, and
*modeled* symbolic factors. Python wall-clock timing is diagnostic and
environment-specific. The honest conclusion the numbers support is that the
selected-observable QSVT pipeline is not competitive with the classical adjoint
baseline under the stated assumptions -- it is an implementation study, not a
speed comparison. Ridge/Tikhonov is the matched reference; the QSVT-target filter
computes the same value at the same alpha.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from robust_qsvt_se.paper.circuit_signed_readout import estimate_overlap
from robust_qsvt_se.paper.reversible_sparse_oracle import formal_cost_model
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    EXPERIMENTS_CLAIM_BOUNDARY,
    RESOURCE_DIR,
    assert_safe,
    get_pyplot,
    write_experiment_manifest,
)
from robust_qsvt_se.paper.tqe_revision_readout_statistics import _build_headline
from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.utils.io import ensure_directory

DEFAULT_TARGET_RELATIVE_ERROR = 1.0e-2
DEFAULT_TIMING_REPEATS = 30
QSVT_QUERY_CONVENTION = (
    "degree signal-unitary calls; degree+1 projector-phase operations are reported "
    "separately; 2*degree+1 is total alternating sequence length"
)

LEDGER_COLUMNS = [
    "field",
    "value",
    "tier",
    "units",
    "note",
]

CLASSICAL_COLUMNS = [
    "case",
    "matrix_shape",
    "observable_label",
    "method",
    "alpha",
    "median_runtime_seconds",
    "iqr_low_seconds",
    "iqr_high_seconds",
    "selected_functional_value",
    "abs_difference_from_ridge_reference",
    "num_timing_repeats",
    "timing_note",
]


def _ancilla_stats(ell: np.ndarray, psi: np.ndarray) -> tuple[float, float]:
    """Return ``(overlap_exact, ancilla_p0)`` for the Hadamard-test readout circuit."""

    estimate = estimate_overlap(ell, psi, shots=1, seed=0)
    return float(estimate.overlap_exact), float(estimate.success_probability_ancilla)


def _shots_for_target(overlap_exact: float, ancilla_p0: float, target_rel: float) -> float:
    """Shots to reach ``target_rel`` relative error on the signed functional.

    The Hadamard-test overlap estimator has variance ``4 p0 (1-p0) / N`` on ``mu``;
    the physical rescale cancels in the *relative* error, so
    ``rel_err = 2 sqrt(p0(1-p0)/N) / |mu|`` and ``N = 4 p0 (1-p0) / (rel^2 mu^2)``.
    """

    if abs(overlap_exact) <= 1.0e-15 or target_rel <= 0.0:
        return float("nan")
    variance = max(ancilla_p0 * (1.0 - ancilla_p0), 0.0)
    return 4.0 * variance / (target_rel**2 * overlap_exact**2)


def build_quantum_ledger(
    *,
    result: Any,
    matrix_source: str,
    observable_label: str,
    target_rel: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = result.pipeline_metadata
    row = result.row_common
    beta = float(meta["beta"])
    bound_c = float(meta["bound_C"])
    alpha = float(meta["alpha"])
    lam = float(meta["alpha_normalized"])
    degree = int(meta["degree"])
    phase_count = int(meta["phase_count"])
    n = int(result.H_block.shape[0])
    p_success = float(row["postselection_probability"])

    obs = next(o for o in result.observables if o.observable_id == observable_label)
    ell = np.asarray(obs.vector, dtype=np.float64)
    psi = np.asarray(result.output_state, dtype=np.complex128)
    overlap_exact, ancilla_p0 = _ancilla_stats(ell, psi)
    shots_for_target = _shots_for_target(overlap_exact, ancilla_p0, target_rel)

    operation_counts = qsvt_sequence_operation_counts(phase_count)
    signal_calls_per_attempt = operation_counts["signal_unitary_calls"]
    alternating_sequence_length = operation_counts["alternating_sequence_length"]
    attempts_without_aa = 1.0 / p_success if p_success > 0 else float("inf")
    attempts_with_aa = 1.0 / math.sqrt(p_success) if p_success > 0 else float("inf")
    total_signal_calls_no_aa = signal_calls_per_attempt * shots_for_target * attempts_without_aa
    total_signal_calls_aa = signal_calls_per_attempt * shots_for_target * attempts_with_aa

    # Sparse-access oracle T-count proxy for this block (per O_col+O_val pair, modeled).
    nonzero_mask = np.abs(result.H_block) > 1.0e-12
    cost_model = formal_cost_model(
        case="ieee14_4x4",
        rows=n,
        cols=n,
        nnz=int(nonzero_mask.sum()),
        max_nonzeros_per_row=int(nonzero_mask.sum(axis=1).max()),
        value_bits=6,
    )
    t_count_per_pair = int(cost_model["total_t_count_qrom"])
    oracle_calls_proxy = 2.0 * total_signal_calls_no_aa

    ledger = [
        _f("matrix_source", matrix_source, "implemented", "-", "IEEE-14 weighted Jacobian block"),
        _f("matrix_shape", f"{n}x{n}", "implemented", "-", "fixed selected block"),
        _f("state_dimension", n, "implemented", "states", "block column dimension"),
        _f("measurement_rows", n, "implemented", "rows", "selected weighted-Jacobian rows"),
        _f("observable_label", observable_label, "implemented", "-", obs.physical_meaning),
        _f("alpha", alpha, "implemented", "-", "matched Ridge alpha (= 4 sigma_min^2)"),
        _f("beta", beta, "implemented", "-", "sigma_max normalization"),
        _f("C_bound", bound_c, "implemented", "-", "co-designed bounded-target scale"),
        _f("lambda_alpha_over_beta2", lam, "implemented", "-", "normalized regularization"),
        _f("degree", degree, "implemented", "-", "odd QSVT polynomial degree"),
        _f("phase_count", phase_count, "implemented", "phases", "synthesized PennyLane phases"),
        _f(
            "signal_unitary_calls_per_attempt",
            signal_calls_per_attempt,
            "implemented",
            "signal-unitary calls",
            QSVT_QUERY_CONVENTION,
        ),
        _f(
            "alternating_sequence_length_per_attempt",
            alternating_sequence_length,
            "implemented",
            "operations",
            "signal-unitary calls plus projector-phase operations; not a query count",
        ),
        _f(
            "block_encoding_status",
            "implemented_dense_small_scale",
            "implemented",
            "-",
            "canonical dense block-encoding gate (non-scalable primitive)",
        ),
        _f(
            "block_encoding_unitarity_error",
            float(row["block_encoding_unitarity_error"]),
            "implemented",
            "-",
            "statevector-validated",
        ),
        _f(
            "svt_circuit_vs_exact_max_error",
            float(row["svt_circuit_vs_exact_max_error"]),
            "statevector",
            "-",
            "circuit SVT vs dense SVT",
        ),
        _f(
            "state_preparation_status",
            "proxy_dense_amplitude_loading",
            "proxy",
            "-",
            "dense amplitude loading; not an efficient-preparation proof",
        ),
        _f(
            "postselection_probability",
            p_success,
            "statevector",
            "-",
            "norm^2 of the encoded success branch for the residual input",
        ),
        _f(
            "attempts_without_amplitude_amplification",
            attempts_without_aa,
            "modeled",
            "repeats",
            "expected 1/p_success postselection repeats",
        ),
        _f(
            "attempts_with_amplitude_amplification_proxy",
            attempts_with_aa,
            "modeled",
            "repeats",
            "modeled O(1/sqrt(p_success)) amplified repeats (not synthesized)",
        ),
        _f(
            "target_observable_error",
            target_rel,
            "finite_shot",
            "relative",
            "target relative error on the signed functional",
        ),
        _f(
            "readout_overlap_exact",
            overlap_exact,
            "statevector",
            "-",
            "noiseless Hadamard-test overlap",
        ),
        _f(
            "shots_for_target_error",
            shots_for_target,
            "finite_shot",
            "shots",
            "N = 4 p0(1-p0)/(rel^2 mu^2)",
        ),
        _f(
            "total_signal_unitary_calls_without_AA",
            total_signal_calls_no_aa,
            "modeled",
            "signal-unitary calls",
            "signal-unitary calls/attempt * shots * (1/p_success)",
        ),
        _f(
            "total_signal_unitary_calls_with_AA_proxy",
            total_signal_calls_aa,
            "modeled",
            "signal-unitary calls",
            "signal-unitary calls/attempt * shots * (1/sqrt(p_success)); AA is modeled",
        ),
        _f(
            "oracle_call_proxy",
            oracle_calls_proxy,
            "modeled",
            "oracle calls",
            "~1 O_col + 1 O_val call per signal-unitary call (modeled)",
        ),
        _f(
            "T_count_proxy_per_oracle_pair",
            t_count_per_pair,
            "modeled",
            "T gates",
            "unary-iteration (QROM) O_col+O_val T-count for the block (modeled)",
        ),
        _f(
            "depth_proxy_qsvt_operator",
            _num_or_nan(row["raw_circuit_depth"]),
            "implemented",
            "gate layers",
            "raw QSVT operator circuit depth (small-simulator)",
        ),
        _f(
            "qubit_proxy",
            int(row["num_qubits"]) + 1,
            "implemented",
            "qubits",
            "block-encoding qubits + 1 readout ancilla",
        ),
    ]
    context = {
        "overlap_exact": overlap_exact,
        "ancilla_p0": ancilla_p0,
        "shots_for_target": shots_for_target,
        "signal_calls_per_attempt": signal_calls_per_attempt,
        "phase_operations_per_attempt": operation_counts["projector_phase_operations"],
        "alternating_sequence_length": alternating_sequence_length,
        "attempts_without_aa": attempts_without_aa,
        "total_signal_calls_no_aa": total_signal_calls_no_aa,
        "p_success": p_success,
        "degree": degree,
        "alpha": alpha,
        "t_count_per_pair": t_count_per_pair,
        "target_rel": target_rel,
    }
    return ledger, context


def _f(field: str, value: Any, tier: str, units: str, note: str) -> dict[str, Any]:
    return {"field": field, "value": value, "tier": tier, "units": units, "note": note}


def _num_or_nan(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return float("nan")


def _classical_baseline(
    H: np.ndarray,
    r: np.ndarray,
    ell: np.ndarray,
    *,
    alpha: float,
    observable_label: str,
    repeats: int,
    matrix_shape: str,
) -> tuple[list[dict[str, Any]], float]:
    """Time the classical selected-observable methods with ``repeats`` trials each."""

    n = H.shape[1]
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    ridge_update = Vt.T @ ((s / (s**2 + alpha)) * (U.T @ r))
    ridge_value = float(ell @ ridge_update)
    htr = H.T @ r

    def dense_direct() -> float:
        gram = H.T @ H + alpha * np.eye(n)
        x = np.linalg.solve(gram, htr)
        return float(ell @ x)

    def dense_adjoint() -> float:
        gram = H.T @ H + alpha * np.eye(n)
        w = np.linalg.solve(gram, ell)
        return float(w @ htr)

    H_sparse = sp.csr_matrix(H)

    def sparse_adjoint() -> float:
        gram = (H_sparse.T @ H_sparse).tocsc() + alpha * sp.identity(n, format="csc")
        lu = spla.splu(gram)
        w = lu.solve(np.asarray(ell, dtype=np.float64))
        return float(w @ np.asarray(H_sparse.T @ r).ravel())

    methods = {
        "dense_direct_full_update": dense_direct,
        "dense_adjoint_selected_observable": dense_adjoint,
        "sparse_adjoint_selected_observable": sparse_adjoint,
    }
    rows: list[dict[str, Any]] = []
    best_median = float("inf")
    for name, fn in methods.items():
        times: list[float] = []
        value = float("nan")
        for _ in range(int(repeats)):
            start = time.perf_counter()
            value = fn()
            times.append(time.perf_counter() - start)
        arr = np.asarray(times, dtype=np.float64)
        median = float(np.median(arr))
        best_median = min(best_median, median)
        rows.append(
            {
                "case": "ieee14",
                "matrix_shape": matrix_shape,
                "observable_label": observable_label,
                "method": name,
                "alpha": float(alpha),
                "median_runtime_seconds": median,
                "iqr_low_seconds": float(np.percentile(arr, 25)),
                "iqr_high_seconds": float(np.percentile(arr, 75)),
                "selected_functional_value": value,
                "abs_difference_from_ridge_reference": abs(value - ridge_value),
                "num_timing_repeats": int(repeats),
                "timing_note": "Python wall-clock; diagnostic and environment-specific",
            }
        )
    return rows, best_median


def run_resource_ledger(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    target_rel = float(resolved["target_relative_error"])
    observable_label = resolved["observable_label"]

    result, matrix_source = _build_headline(
        resolved["case"],
        resolved["case_source"],
        int(resolved["seed"]),
        int(resolved["degree"]),
        float(resolved["alpha_mult"]),
    )
    warnings: list[str] = []
    if result.status_label != "pass":
        warnings.append(f"headline block status '{result.status_label}' is not 'pass'")

    ledger, context = build_quantum_ledger(
        result=result,
        matrix_source=matrix_source,
        observable_label=observable_label,
        target_rel=target_rel,
    )

    ell = np.asarray(
        next(o for o in result.observables if o.observable_id == observable_label).vector,
        dtype=np.float64,
    )
    classical_rows, best_classical_median = _classical_baseline(
        result.H_block,
        result.r_block,
        ell,
        alpha=float(result.pipeline_metadata["alpha"]),
        observable_label=observable_label,
        repeats=int(resolved["timing_repeats"]),
        matrix_shape=str(result.row_common["block_shape"]),
    )

    boundary_rows = _boundary_rows(context, best_classical_median, observable_label)

    artifacts = _write_outputs(
        output_dir=output_dir,
        ledger=ledger,
        classical_rows=classical_rows,
        boundary_rows=boundary_rows,
        context=context,
        result=result,
        observable_label=observable_label,
        target_rel=target_rel,
    )
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="C_end_to_end_resource_case",
        script_name="scripts/run_tqe_revision_resource_ledger.py",
        command=resolved["command"],
        description=(
            "Fixed-case (IEEE-14 4x4) end-to-end selected-observable QSVT resource ledger "
            "compared numerically against the classical selected-observable adjoint baseline "
            "at the same alpha. Separates implemented / statevector / finite-shot / proxy / "
            "modeled quantities explicitly."
        ),
        artifacts=artifacts,
        inputs_used=[f"build_engineering_system:{resolved['case']}:weighted_jacobian"],
        random_seeds={"demo_system_seed": int(resolved["seed"]), "ancilla_readout_seed": 0},
        warnings=warnings,
        failures=[],
        interpretation_boundary=(
            "Implemented rows are synthesized/statevector-validated at small simulator scale; "
            "finite-shot rows are shot-sampling counts; proxy rows are small-simulator "
            "stand-ins (dense state preparation); modeled rows are symbolic factors "
            "(amplitude amplification, sparse-oracle T-count, aggregate query totals). Python "
            "wall-clock timing is diagnostic. Under these assumptions the selected-observable "
            "QSVT pipeline is not competitive with the classical adjoint baseline, which "
            "returns the same functional at the same alpha."
        ),
        extra={
            "observable_label": observable_label,
            "target_relative_error": target_rel,
            "best_classical_adjoint_median_seconds": best_classical_median,
            "total_signal_unitary_calls_without_AA": context["total_signal_calls_no_aa"],
            "shots_for_target_error": context["shots_for_target"],
            "resource_conclusion_type": 1,
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "ledger": pd.DataFrame(ledger, columns=LEDGER_COLUMNS),
        "classical": pd.DataFrame(classical_rows, columns=CLASSICAL_COLUMNS),
        "boundary": pd.DataFrame(boundary_rows),
        "artifacts": artifacts,
    }


def _boundary_rows(
    context: dict[str, Any], best_classical_median: float, observable_label: str
) -> list[dict[str, Any]]:
    return [
        {
            "observable_label": observable_label,
            "side": "classical_adjoint",
            "quantity": "wall_clock_seconds_per_functional (median)",
            "value": best_classical_median,
            "tier": "measured",
            "note": "single 4x4 adjoint solve returning the same value at the same alpha",
        },
        {
            "observable_label": observable_label,
            "side": "quantum_qsvt",
            "quantity": "signal_unitary_calls_for_target_error (no AA)",
            "value": context["total_signal_calls_no_aa"],
            "tier": "modeled",
            "note": "signal-unitary calls/attempt * shots_for_target * (1/p_success)",
        },
        {
            "observable_label": observable_label,
            "side": "quantum_qsvt",
            "quantity": "finite_shots_for_target_error",
            "value": context["shots_for_target"],
            "tier": "finite_shot",
            "note": "Hadamard-test shots for the target relative error",
        },
        {
            "observable_label": observable_label,
            "side": "quantum_qsvt",
            "quantity": "modeled_T_count_per_oracle_pair",
            "value": context["t_count_per_pair"],
            "tier": "modeled",
            "note": "sparse-access oracle T-count (QROM) per block query pair",
        },
        {
            "observable_label": observable_label,
            "side": "conclusion",
            "quantity": "resource_conclusion",
            "value": (
                "selected-observable QSVT pipeline not competitive with the classical adjoint "
                "baseline under the stated assumptions"
            ),
            "tier": "interpretation",
            "note": "same functional, same alpha; no speed comparison is claimed",
        },
    ]


def _write_outputs(
    *,
    output_dir: Path,
    ledger: list[dict[str, Any]],
    classical_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    context: dict[str, Any],
    result: Any,
    observable_label: str,
    target_rel: float,
) -> dict[str, Path]:
    ledger_frame = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    classical_frame = pd.DataFrame(classical_rows, columns=CLASSICAL_COLUMNS)
    boundary_frame = pd.DataFrame(boundary_rows)

    ledger_csv = output_dir / "fixed_case_resource_ledger.csv"
    classical_csv = output_dir / "classical_adjoint_baseline.csv"
    boundary_csv = output_dir / "quantum_vs_classical_boundary.csv"
    ledger_frame.to_csv(ledger_csv, index=False)
    classical_frame.to_csv(classical_csv, index=False)
    boundary_frame.to_csv(boundary_csv, index=False)

    waterfall_pdf = output_dir / "resource_waterfall.pdf"
    waterfall_png = output_dir / "resource_waterfall.png"
    _plot_waterfall(context, waterfall_pdf, waterfall_png)

    resource_tex = output_dir / "resource_table.tex"
    classical_tex = output_dir / "classical_baseline_table.tex"
    resource_tex.write_text(_resource_tex(ledger_frame), encoding="utf-8")
    classical_tex.write_text(_classical_tex(classical_frame), encoding="utf-8")

    assumptions = output_dir / "assumptions.md"
    assumptions.write_text(_assumptions_md(), encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(
        _readme(
            ledger_frame,
            classical_frame,
            boundary_frame,
            context,
            result,
            observable_label,
            target_rel,
        ),
        encoding="utf-8",
    )

    return {
        "fixed_case_resource_ledger_csv": ledger_csv,
        "classical_adjoint_baseline_csv": classical_csv,
        "quantum_vs_classical_boundary_csv": boundary_csv,
        "resource_waterfall_pdf": waterfall_pdf,
        "resource_waterfall_png": waterfall_png,
        "resource_table_tex": resource_tex,
        "classical_baseline_table_tex": classical_tex,
        "assumptions_md": assumptions,
        "readme_md": readme,
    }


def _plot_waterfall(context: dict[str, Any], pdf_path: Path, png_path: Path) -> None:
    plt = get_pyplot()
    stages = [
        "signal calls/attempt\n(d)",
        "x shots for\ntarget error",
        "x postselection\nrepeats (1/p_s)",
    ]
    per_attempt = context["signal_calls_per_attempt"]
    after_shots = per_attempt * context["shots_for_target"]
    after_post = after_shots * context["attempts_without_aa"]
    cumulative = [per_attempt, after_shots, after_post]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#4c72b0", "#dd8452", "#c44e52"]
    ax.bar(range(len(stages)), cumulative, color=colors)
    for i, value in enumerate(cumulative):
        ax.text(i, value, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylabel("cumulative signal-unitary calls (log)")
    ax.set_title(
        "Selected-observable QSVT query budget for one functional\n"
        f"(IEEE-14 4x4, degree {context['degree']}, target rel err "
        f"{context['target_rel']:.0e})"
    )
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def _resource_tex(ledger: pd.DataFrame) -> str:
    lines = [
        "% Fixed-case (IEEE-14 4x4) selected-observable QSVT resource ledger.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Field & Value & Tier & Units \\",
        r"\midrule",
    ]

    def esc(text: Any) -> str:
        return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    for _, row in ledger.iterrows():
        value = row["value"]
        if isinstance(value, float):
            value = f"{value:.4g}"
        lines.append(
            f"{esc(row['field'])} & {esc(value)} & {esc(row['tier'])} & {esc(row['units'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{End-to-end selected-observable resource ledger for the fixed IEEE-14 4x4 "
        r"block. Tiers separate implemented circuit/statevector quantities from finite-shot, "
        r"proxy (small-simulator), and modeled (symbolic) factors. The QSVT-target filter "
        r"returns the same functional at the same $\alpha$ as Ridge/Tikhonov.}",
        r"\label{tab:fixed_case_resource_ledger}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _classical_tex(classical: pd.DataFrame) -> str:
    lines = [
        "% Classical selected-observable adjoint baseline (30 timing repeats).",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & median (s) & IQR (s) & functional value \\",
        r"\midrule",
    ]
    for _, row in classical.iterrows():
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{method} & {row['median_runtime_seconds']:.3e} & "
            f"[{row['iqr_low_seconds']:.3e}, {row['iqr_high_seconds']:.3e}] & "
            f"{row['selected_functional_value']:+.5e} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Classical selected-observable baseline for the fixed IEEE-14 4x4 block: "
        r"median and interquartile wall-clock over 30 repeats. Wall-clock is diagnostic and "
        r"environment-specific. The adjoint solve returns the same regularized functional at "
        r"the same $\alpha$ as the QSVT pathway.}",
        r"\label{tab:classical_adjoint_baseline}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _assumptions_md() -> str:
    text = "\n".join(
        [
            "# Resource Ledger Assumptions and Tiers",
            "",
            EXPERIMENTS_CLAIM_BOUNDARY,
            "",
            "Every ledger row carries a `tier`. The tiers are, from most to least concrete:",
            "",
            "- **implemented**: synthesized and validated at small simulator scale (dense "
            "block-encoding gate, QSVT operator circuit, phase count, qubit/depth counts).",
            "- **statevector**: exact noiseless simulator quantities (postselection probability, "
            "circuit-vs-dense SVT error, noiseless Hadamard-test overlap).",
            "- **finite_shot**: shot-sampling counts (shots for the target relative error).",
            "- **proxy**: small-simulator stand-ins that are *not* efficiency proofs (dense "
            "amplitude-loading state preparation).",
            "- **modeled**: symbolic / asymptotic factors that are *not* compiled circuits "
            "(amplitude amplification `O(1/sqrt(p_success))`, sparse-access oracle T-count, "
            "aggregate signal-unitary call totals, per-call oracle proxy).",
            "",
            "## What is and is not implemented",
            "",
            "- The dense block encoding, QSVT operator circuit, phase synthesis, and "
            "statevector postselection are implemented and simulator-validated.",
            "- The finite-shot result is an isolated Hadamard-overlap circuit that directly "
            "prepares the classically computed postselected output state; integrated "
            "residual-loading--QSVT--postselection--readout execution remains future work.",
            "- State preparation is a dense amplitude-loading **proxy**; scalable preparation is "
            "not proven here.",
            "- Sparse block encoding is **not** compiled end-to-end; the sparse-access oracle "
            "T-count is a **modeled** unary-iteration (QROM) bound (see the sparse-oracle demo).",
            "- Amplitude amplification is a **modeled** factor, not a synthesized circuit.",
            "",
            "## Conclusion the numbers support",
            "",
            "Under these assumptions the selected-observable QSVT pipeline is **not competitive** "
            "with the classical adjoint baseline: recovering one signed functional to the target "
            "relative error needs a large finite-shot budget times the QSVT query count times the "
            "postselection-repeat factor (plus a modeled sparse-oracle T-count), whereas the "
            "classical adjoint solve returns the same functional at the same alpha in microseconds "
            "on this block. This is an implementation and boundary study; it is not a speed "
            "comparison and asserts no advantage.",
            "",
        ]
    )
    assert_safe(text)
    return text


def _readme(
    ledger: pd.DataFrame,
    classical: pd.DataFrame,
    boundary: pd.DataFrame,
    context: dict[str, Any],
    result: Any,
    observable_label: str,
    target_rel: float,
) -> str:
    best = classical.loc[classical["method"] == "dense_adjoint_selected_observable"].iloc[0]
    meaning = next(
        o.physical_meaning for o in result.observables if o.observable_id == observable_label
    )
    lines = [
        "# Experiment C: Fixed-Case End-to-End Resource Ledger (IEEE-14 4x4)",
        "",
        EXPERIMENTS_CLAIM_BOUNDARY,
        "",
        f"Fixed case: IEEE-14 4x4 weighted-Jacobian block, observable `{observable_label}` "
        f"({meaning}). "
        f"QSVT degree {context['degree']}, matched alpha {context['alpha']:.4g}, postselection "
        f"probability {context['p_success']:.4f}.",
        "",
        "## Quantum-side accounting (tiers separated)",
        "",
        "| Field | Value | Tier | Units |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in ledger.iterrows():
        value = row["value"]
        if isinstance(value, float):
            value = f"{value:.4g}"
        lines.append(f"| {row['field']} | {value} | `{row['tier']}` | {row['units']} |")
    lines += [
        "",
        "## Classical selected-observable adjoint baseline (30 repeats)",
        "",
        "| Method | median (s) | IQR (s) | functional value | diff vs Ridge |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in classical.iterrows():
        lines.append(
            f"| {row['method']} | {row['median_runtime_seconds']:.3e} | "
            f"[{row['iqr_low_seconds']:.3e}, {row['iqr_high_seconds']:.3e}] | "
            f"{row['selected_functional_value']:+.5e} | "
            f"{row['abs_difference_from_ridge_reference']:.2e} |"
        )
    lines += [
        "",
        "## Main resource conclusion",
        "",
        f"To reach a {target_rel:.0e} relative error on one signed functional the QSVT readout "
        f"needs about **{context['shots_for_target']:.2e} shots**, and with the "
        f"{context['signal_calls_per_attempt']} signal-unitary calls per attempt and the "
        f"{context['attempts_without_aa']:.2f}x postselection-repeat factor this is about "
        f"**{context['total_signal_calls_no_aa']:.2e} signal-unitary calls** (plus a modeled "
        f"sparse-oracle T-count of {context['t_count_per_pair']} per query pair). The classical "
        f"adjoint solve returns the *same* functional at the *same* alpha in a median of "
        f"**{best['median_runtime_seconds']:.2e} s** on this block.",
        "",
        "**Conclusion (type 1):** the selected-observable QSVT pipeline is not competitive with "
        "the classical adjoint baseline under the stated assumptions. Python wall-clock timing "
        "is diagnostic and environment-specific; the comparison is a resource-accounting "
        "boundary, not a head-to-head speed claim, and asserts no advantage. See "
        "`assumptions.md` for the tier definitions.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(RESOURCE_DIR),
        "case": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "degree": 31,
        "alpha_mult": 4.0,
        "observable_label": "state_correction_0",
        "target_relative_error": DEFAULT_TARGET_RELATIVE_ERROR,
        "timing_repeats": DEFAULT_TIMING_REPEATS,
        "command": "run_tqe_revision_resource_ledger",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment C: fixed-case resource ledger")
    parser.add_argument("--output-dir", default=str(RESOURCE_DIR))
    parser.add_argument("--observable-label", default="state_correction_0")
    parser.add_argument(
        "--target-relative-error", type=float, default=DEFAULT_TARGET_RELATIVE_ERROR
    )
    parser.add_argument("--timing-repeats", type=int, default=DEFAULT_TIMING_REPEATS)
    parser.add_argument("--quick", action="store_true", help="fewer timing repeats")
    parser.add_argument("--full", action="store_true", help="full timing repeats")
    args = parser.parse_args(argv)
    repeats = 10 if args.quick else int(args.timing_repeats)
    run = run_resource_ledger(
        {
            "output_dir": args.output_dir,
            "observable_label": args.observable_label,
            "target_relative_error": args.target_relative_error,
            "timing_repeats": repeats,
            "command": "scripts/run_tqe_revision_resource_ledger.py " + " ".join(argv or []),
        }
    )
    print(f"Resource ledger complete: {run['artifacts']['fixed_case_resource_ledger_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
