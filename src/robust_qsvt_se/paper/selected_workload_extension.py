"""Goal A: larger selected-output QSVT workload attempts beyond the 4x4 anchor.

Extends the frozen selected-observable demonstration with explicitly recorded
attempts on the deterministic IEEE-14-derived 8x8 block:

1. regenerate and verify the 4x4 correctness anchor (never replaces the frozen
   output directory; a `matches_frozen_anchor` field records agreement),
2. re-attempt the 8x8 block at the benchmark co-design alpha = 4 sigma_min^2
   (lambda ~ 2.3e-4) at degree 31 and at the tested degree-45 synthesis ceiling,
3. attempt the 8x8 block at a lambda-matched co-designed alpha (same normalized
   regularization lambda = alpha/beta^2 as the 4x4 anchor) with a minimal odd
   degree search, testing whether block dimension or regularization strength is
   the binding constraint.

Every attempt row keeps the matched Ridge/Tikhonov reference at the *same*
alpha, a classical adjoint evaluation (value + median timing) of the same
functionals, and an explicit boundary status in
{feasible, degree_limited, tolerance_missing, phase_failed, dimension_infeasible}.
No speedup, no QSVT-over-Ridge superiority, and no hardware execution is claimed;
failed attempts are reported as boundary evidence.
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
from robust_qsvt_se.paper.circuit_signed_readout import circuit_signed_readout_rows
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    assert_safe,
    write_demo_manifest,
)
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    BlockDemoResult,
    run_demo_for_block,
)
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    _state_labels_for_cols as state_labels_for_cols,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/qsvt_selected_workload_extension")
FROZEN_ANCHOR_JSON = Path("outputs/selected_observable_qsvt_demo/demo_summary.json")
PASS_RELATIVE_TOLERANCE = 0.05
READOUT_SHOTS = 100_000
READOUT_SEED = 20260702
ADJOINT_TIMING_REPEATS = 30

BLOCK_SELECTION_RULE = "largest_row_col_norms deterministic (case+seed only, pre-solve)"
MEASUREMENT_PROVENANCE = (
    "PYPOWER-generated IEEE-14 AC measurement rows (weighted Jacobian), seed 123; "
    "not field PMU/SCADA data"
)

STATUS_MAP = {
    "pass": "feasible",
    "degree_limited": "degree_limited",
    "phase_unavailable": "phase_failed",
    "failed_boundedness": "tolerance_missing",
}

RESULT_COLUMNS = [
    "workload_id",
    "case",
    "block_size",
    "block_selection_rule",
    "measurement_state_provenance",
    "selected_rows",
    "selected_cols",
    "block_checksum",
    "kappa_block",
    "singular_values",
    "alpha",
    "alpha_rule",
    "beta",
    "lambda_alpha_over_beta2",
    "bound_C",
    "degree_attempted",
    "phase_synthesis_status",
    "phase_count",
    "target_fit_error",
    "actual_singular_value_error",
    "dense_block_encoding_unitarity_error",
    "circuit_vs_exact_svt_error",
    "postselection_probability",
    "update_relative_error_vs_ridge",
    "pass_relative_tolerance",
    "observable_id",
    "selected_functional_definition",
    "ridge_reference_value",
    "qsvt_statevector_value",
    "statevector_selected_output_relative_error",
    "finite_shot_shots",
    "finite_shot_estimate",
    "finite_shot_relative_error",
    "classical_adjoint_value",
    "classical_adjoint_median_seconds",
    "status",
    "matches_frozen_anchor",
    "notes",
]


def actual_singular_value_error(result: BlockDemoResult) -> float:
    """Max |p(s_i) - f(s_i)/C| over the block's own normalized singular values."""

    meta = result.pipeline_metadata
    coefficients = np.asarray(meta["polynomial_coefficients_bounded"], dtype=np.float64)
    polynomial = np.polynomial.Polynomial(coefficients)
    sigmas = np.asarray(meta["singular_values"], dtype=np.float64)
    beta = float(meta["beta"])
    lam = float(meta["alpha_normalized"])
    bound_c = float(meta["bound_C"])
    s = sigmas / beta
    target = (s / (s**2 + lam)) / bound_c
    return float(np.max(np.abs(polynomial(s) - target)))


def classical_adjoint_functional(
    H: np.ndarray, r: np.ndarray, ell: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Adjoint evaluation y = ell^T (H^T H + alpha I)^{-1} H^T r with median timing."""

    timings: list[float] = []
    value = math.nan
    for _ in range(ADJOINT_TIMING_REPEATS):
        start = time.perf_counter()
        gram = H.T @ H + alpha * np.eye(H.shape[1])
        rhs = H.T @ r
        w = np.linalg.solve(gram, ell)
        value = float(w @ rhs)
        timings.append(time.perf_counter() - start)
    return value, float(np.median(timings))


def _load_frozen_anchor_reference() -> dict[str, Any] | None:
    if not FROZEN_ANCHOR_JSON.is_file():
        return None
    records = json.loads(FROZEN_ANCHOR_JSON.read_text(encoding="utf-8"))
    for record in records:
        if record.get("block_shape") == "4x4":
            return record
    return None


def _anchor_agreement(result: BlockDemoResult, frozen: dict[str, Any] | None) -> str:
    if frozen is None:
        return "frozen_anchor_not_found"
    checks = {
        "block_checksum": result.row_common["block_checksum"] == frozen["block_checksum"],
        "degree": int(result.row_common["degree"]) == int(frozen["degree"]),
        "update_relative_error": math.isclose(
            float(result.row_common["update_relative_error_vs_ridge"]),
            float(frozen["update_relative_error_vs_ridge"]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ),
        "postselection_probability": math.isclose(
            float(result.row_common["postselection_probability"]),
            float(frozen["postselection_probability"]),
            rel_tol=1.0e-6,
        ),
    }
    if all(checks.values()):
        return "verified_match"
    failed = sorted(name for name, ok in checks.items() if not ok)
    return "mismatch:" + ",".join(failed)


def _finite_shot_errors(result: BlockDemoResult) -> dict[str, dict[str, float]]:
    """Hadamard-test finite-shot estimates for signed functionals of a pass block."""

    if result.output_state is None or result.status_label != "pass":
        return {}
    signed = [
        {
            "observable_id": obs.observable_id,
            "observable_type": obs.observable_type,
            "vector": obs.vector,
        }
        for obs in result.observables
        if obs.sign_aware_required
    ]
    ridge_reference = {
        obs.observable_id: obs.exact_value(result.ridge_update) for obs in result.observables
    }
    rows = circuit_signed_readout_rows(
        observables=signed,
        output_state=result.output_state,
        physical_recovery_scale=result.physical_recovery_scale,
        ridge_reference=ridge_reference,
        shots_grid=(READOUT_SHOTS,),
        seed=READOUT_SEED,
    )
    estimates: dict[str, dict[str, float]] = {}
    for row in rows:
        estimate = float(row["physical_signed_value_estimate"])
        reference = float(row["ridge_reference_value"])
        rel = abs(estimate - reference) / abs(reference) if abs(reference) > 1e-15 else math.nan
        estimates[str(row["observable_id"])] = {
            "shots": float(row["shots"]),
            "estimate": estimate,
            "relative_error": rel,
        }
    return estimates


def run_attempt(
    *,
    workload_id: str,
    case: str,
    matrix_source: str,
    H_full: np.ndarray,
    r_full: np.ndarray,
    system_metadata: dict[str, Any],
    size: int,
    alpha: float,
    alpha_rule: str,
    degree: int,
    phase_cache_dir: Path,
    notes: str,
    frozen_anchor: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], BlockDemoResult]:
    """One (block, alpha, degree) attempt; one output row per signed functional."""

    H_block, r_block, rows, cols = select_deterministic_block(
        H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
    )
    column_labels = state_labels_for_cols(system_metadata, cols)
    result = run_demo_for_block(
        case=case,
        matrix_source=matrix_source,
        H_block=H_block,
        r_block=r_block,
        selected_rows=rows,
        selected_cols=cols,
        column_labels=column_labels,
        alpha=float(alpha),
        degree=int(degree),
        angle_solver="iterative",
        margin=1.05,
        domain_low_factor=0.9,
        pass_relative_tolerance=PASS_RELATIVE_TOLERANCE,
        phase_cache_dir=phase_cache_dir,
    )
    status = STATUS_MAP.get(result.status_label, result.status_label)
    action_error = actual_singular_value_error(result)
    finite_shot = _finite_shot_errors(result)
    anchor_check = (
        _anchor_agreement(result, frozen_anchor) if frozen_anchor is not None else "not_applicable"
    )

    common = result.row_common
    out_rows: list[dict[str, Any]] = []
    signed_ids = {obs.observable_id for obs in result.observables if obs.sign_aware_required}
    for obs_row in result.observable_rows:
        # Adjoint evaluation targets linear signed functionals only; the quadratic
        # energy-style observable has no matching l^T dx form.
        if obs_row["observable_id"] not in signed_ids:
            continue
        observable = next(
            obs for obs in result.observables if obs.observable_id == obs_row["observable_id"]
        )
        adjoint_value, adjoint_seconds = classical_adjoint_functional(
            result.H_block, result.r_block, observable.vector, float(alpha)
        )
        shot_info = finite_shot.get(obs_row["observable_id"], {})
        out_rows.append(
            {
                "workload_id": workload_id,
                "case": case,
                "block_size": common["block_shape"],
                "block_selection_rule": BLOCK_SELECTION_RULE,
                "measurement_state_provenance": MEASUREMENT_PROVENANCE,
                "selected_rows": common["selected_rows"],
                "selected_cols": common["selected_cols"],
                "block_checksum": common["block_checksum"],
                "kappa_block": common["condition_number"],
                "singular_values": " ".join(
                    f"{value:.6f}" for value in result.pipeline_metadata["singular_values"]
                ),
                "alpha": common["alpha"],
                "alpha_rule": alpha_rule,
                "beta": common["beta"],
                "lambda_alpha_over_beta2": common["alpha_normalized"],
                "bound_C": common["bound_C"],
                "degree_attempted": common["degree"],
                "phase_synthesis_status": common["phase_synthesis_status"],
                "phase_count": common["phase_count"],
                "target_fit_error": common["polynomial_fit_max_abs_error"],
                "actual_singular_value_error": action_error,
                "dense_block_encoding_unitarity_error": common["block_encoding_unitarity_error"],
                "circuit_vs_exact_svt_error": common["svt_circuit_vs_exact_max_error"],
                "postselection_probability": common["postselection_probability"],
                "update_relative_error_vs_ridge": common["update_relative_error_vs_ridge"],
                "pass_relative_tolerance": PASS_RELATIVE_TOLERANCE,
                "observable_id": obs_row["observable_id"],
                "selected_functional_definition": obs_row["observable_physical_meaning"],
                "ridge_reference_value": obs_row["ridge_reference_value"],
                "qsvt_statevector_value": obs_row["qsvt_statevector_value"],
                "statevector_selected_output_relative_error": obs_row["relative_error"],
                "finite_shot_shots": shot_info.get("shots", math.nan),
                "finite_shot_estimate": shot_info.get("estimate", math.nan),
                "finite_shot_relative_error": shot_info.get("relative_error", math.nan),
                "classical_adjoint_value": adjoint_value,
                "classical_adjoint_median_seconds": adjoint_seconds,
                "status": status,
                "matches_frozen_anchor": anchor_check,
                "notes": notes,
            }
        )
    return out_rows, result


def run_selected_workload_extension(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": str(OUTPUT_DIR),
        "case": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "codesigned_degrees": [31, 39, 45],
        "benchmark_degrees_8x8": [31, 45],
        "command": "scripts/run_selected_workload_extension.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})

    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")

    system, matrix_source = build_engineering_system(
        {
            "case_name": resolved["case"],
            "case_source": resolved["case_source"],
            "matrix_source": "weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    frozen_anchor = _load_frozen_anchor_reference()

    def sigma_min_for(size: int) -> float:
        H_block, _, _, _ = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        return float(np.linalg.svd(H_block, compute_uv=False).min())

    def sigma_max_for(size: int) -> float:
        H_block, _, _, _ = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        return float(np.linalg.svd(H_block, compute_uv=False).max())

    all_rows: list[dict[str, Any]] = []
    attempt_records: list[dict[str, Any]] = []

    # Attempt 1: regenerate + verify the frozen 4x4 anchor (in this new directory).
    alpha_anchor = 4.0 * sigma_min_for(4) ** 2
    rows_4x4, result_4x4 = run_attempt(
        workload_id="anchor_4x4_benchmark_alpha_d31",
        case=resolved["case"],
        matrix_source=matrix_source,
        H_full=H_full,
        r_full=r_full,
        system_metadata=system.metadata,
        size=4,
        alpha=alpha_anchor,
        alpha_rule="alpha = 4 sigma_min^2 (benchmark co-design rule of the frozen anchor)",
        degree=31,
        phase_cache_dir=phase_cache_dir,
        notes="Regeneration of the frozen 4x4 correctness anchor; frozen output untouched.",
        frozen_anchor=frozen_anchor,
    )
    all_rows.extend(rows_4x4)
    attempt_records.append({"workload_id": rows_4x4[0]["workload_id"], "result": result_4x4})
    lambda_anchor = float(result_4x4.row_common["alpha_normalized"])

    # Attempt set 2: 8x8 at the benchmark alpha rule (light lambda) up to the degree-45 ceiling.
    alpha_benchmark_8 = 4.0 * sigma_min_for(8) ** 2
    for degree in [int(d) for d in resolved["benchmark_degrees_8x8"]]:
        rows_8, result_8 = run_attempt(
            workload_id=f"8x8_benchmark_alpha_d{degree}",
            case=resolved["case"],
            matrix_source=matrix_source,
            H_full=H_full,
            r_full=r_full,
            system_metadata=system.metadata,
            size=8,
            alpha=alpha_benchmark_8,
            alpha_rule="alpha = 4 sigma_min^2 (same benchmark rule as the 4x4 anchor)",
            degree=degree,
            phase_cache_dir=phase_cache_dir,
            notes=(
                "Benchmark-alpha 8x8 attempt; lambda ~ 2.3e-4 places the bounded target in "
                "the sharp-response regime. Reported as boundary evidence, not hidden."
            ),
        )
        all_rows.extend(rows_8)
        attempt_records.append({"workload_id": rows_8[0]["workload_id"], "result": result_8})

    # Attempt set 3: 8x8 at the lambda-matched co-designed alpha (minimal degree search).
    beta_8 = sigma_max_for(8)
    alpha_codesigned_8 = lambda_anchor * beta_8**2
    codesigned_result: BlockDemoResult | None = None
    for degree in [int(d) for d in resolved["codesigned_degrees"]]:
        rows_8c, result_8c = run_attempt(
            workload_id=f"8x8_codesigned_lambda_matched_d{degree}",
            case=resolved["case"],
            matrix_source=matrix_source,
            H_full=H_full,
            r_full=r_full,
            system_metadata=system.metadata,
            size=8,
            alpha=alpha_codesigned_8,
            alpha_rule=(
                "alpha = lambda_anchor * beta^2 (lambda matched to the 4x4 anchor's "
                f"normalized regularization {lambda_anchor:.6f}; separate recorded row, "
                "not a silent tolerance change)"
            ),
            degree=degree,
            phase_cache_dir=phase_cache_dir,
            notes=(
                "Co-designed 8x8 attempt at the anchor's lambda; Ridge reference matched at "
                "the same alpha. Tests whether dimension or regularization strength binds."
            ),
        )
        all_rows.extend(rows_8c)
        attempt_records.append({"workload_id": rows_8c[0]["workload_id"], "result": result_8c})
        codesigned_result = result_8c
        if result_8c.status_label == "pass":
            break

    frame = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    results_csv = output_dir / "selected_workload_results.csv"
    results_json = output_dir / "selected_workload_results.json"
    frame.to_csv(results_csv, index=False)
    frame.to_json(results_json, orient="records", indent=2)

    readme = output_dir / "README.md"
    readme.write_text(_readme_markdown(frame, lambda_anchor), encoding="utf-8")

    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="qsvt_selected_workload_extension",
        description=(
            "Larger selected-submatrix QSVT workload attempts (8x8 IEEE-14-derived block) "
            "extending the frozen 4x4 correctness anchor. Benchmark-alpha rows are "
            "degree-ceiling boundary evidence; the lambda-matched co-designed row tests "
            "whether block dimension or regularization strength binds. Ridge/Tikhonov is "
            "the matched reference at the same alpha throughout; no speedup and no "
            "QSVT-over-Ridge superiority is claimed."
        ),
        command=str(resolved["command"]),
        artifacts={
            "selected_workload_results_csv": results_csv,
            "selected_workload_results_json": results_json,
            "readme_md": readme,
        },
        input_files=[f"build_engineering_system:{resolved['case']}:weighted_jacobian:seed123"],
        extra={
            "pass_relative_tolerance": PASS_RELATIVE_TOLERANCE,
            "readout_shots": READOUT_SHOTS,
            "readout_seed": READOUT_SEED,
            "seed_provenance": {
                "status": "recorded",
                "seeds": {"system_seed": 123, "readout_seed": READOUT_SEED},
                "source": "script constants and deterministic IEEE block construction",
                "applies_to": "generated system and isolated overlap-shot sampling",
            },
            "adjoint_timing_repeats": ADJOINT_TIMING_REPEATS,
            "lambda_anchor": lambda_anchor,
            "statuses_by_workload": {
                record["workload_id"]: STATUS_MAP.get(
                    record["result"].status_label, record["result"].status_label
                )
                for record in attempt_records
            },
        },
    )

    return {
        "output_dir": output_dir,
        "results": frame,
        "codesigned_result": codesigned_result,
        "manifest": manifest,
    }


def _readme_markdown(frame: pd.DataFrame, lambda_anchor: float) -> str:
    per_attempt = frame.drop_duplicates("workload_id")
    lines = [
        "# Selected-Output QSVT Workload Extension (Goal A)",
        "",
        "Larger selected-output workload attempts on the deterministic IEEE-14-derived",
        "8x8 weighted-Jacobian block, extending the frozen 4x4 correctness anchor",
        "(`outputs/selected_observable_qsvt_demo/`, untouched). Every attempt keeps the",
        "matched Ridge/Tikhonov reference at the same alpha and a classical adjoint",
        "evaluation of the same functionals. Statuses use the boundary vocabulary",
        "{feasible, degree_limited, tolerance_missing, phase_failed, dimension_infeasible}.",
        "",
        f"- Anchor normalized regularization: lambda = {lambda_anchor:.6f}",
        f"- Pass tolerance (statevector update vs matched Ridge): {PASS_RELATIVE_TOLERANCE}",
        f"- Finite-shot readout: {READOUT_SHOTS} Hadamard-test shots (feasible rows only)",
        "",
        "| workload | block | kappa | alpha | lambda | degree | phase status | "
        "update rel. err | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in per_attempt.iterrows():
        lines.append(
            f"| {row['workload_id']} | {row['block_size']} | {row['kappa_block']:.2f} | "
            f"{row['alpha']:.4g} | {row['lambda_alpha_over_beta2']:.3g} | "
            f"{row['degree_attempted']} | {row['phase_synthesis_status']} | "
            f"{row['update_relative_error_vs_ridge']:.3g} | **{row['status']}** |"
        )
    lines += [
        "",
        "## Reading the rows",
        "",
        "- `8x8_benchmark_alpha_*`: the same alpha rule as the anchor gives lambda ~ 2.3e-4 on",
        "  the 8x8 block (kappa ~ 133); the bounded target is in the sharp-response regime and",
        "  the tested degree range cannot meet the update tolerance. These rows are the",
        "  degree-ceiling boundary evidence.",
        "- `8x8_codesigned_lambda_matched_*`: matching the anchor's lambda tests whether the",
        "  block dimension itself is the obstruction. A feasible row here is a secondary",
        "  correctness anchor at heavier physical regularization (larger alpha), recorded as a",
        "  separate row; it does not loosen any tolerance and does not change the benchmark rows.",
        "- The classical adjoint columns give the matched selected-output value and its median",
        "  wall-clock evaluation time; wall-clock seconds and quantum query counts are different",
        "  units and are never merged.",
        "",
        "This package is implementation-boundary evidence for the same Ridge/Tikhonov spectral",
        "filter expressed as a QSVT workload. It does not claim speedup, QSVT-over-Ridge",
        "numerical superiority, validation on field PMU/SCADA records, or full IEEE-scale",
        "execution on quantum devices.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Goal A selected-workload extension")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_selected_workload_extension(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "seed": args.seed,
            "command": "scripts/run_selected_workload_extension.py " + " ".join(argv or []),
        }
    )
    summary = run["results"].drop_duplicates("workload_id")[
        ["workload_id", "block_size", "degree_attempted", "status"]
    ]
    print(summary.to_string(index=False))
    print(f"Results: {run['output_dir']}/selected_workload_results.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
