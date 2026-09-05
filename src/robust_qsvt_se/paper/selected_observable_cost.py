"""Selected-observable component-status and repetition-cost ledger.

Shots and postselection repetitions multiply state preparation, access, and the
degree-derived unitary-call count. Rows remain a resource-boundary ledger: sparse
access and preparation are modeled, postselection and readout are proxies, and
full-vector recovery is excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.paper.degree_aware_alpha import _alpha_degree_profile, _degree_for_target
from robust_qsvt_se.paper.selected_observable_common import (
    MANDATORY_BOUNDARY_STATEMENTS,
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
    write_workload_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.readout_diagnostics import (
    observable_shot_sweep,
    shots_for_relative_error,
)
from robust_qsvt_se.qsvt.selected_observables import (
    BASIS_SAMPLING_MODEL,
    build_selected_observables,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.io import ensure_directory

DEFAULT_CASES = ("ieee14", "ieee57")
DEFAULT_ALPHA = 1.0e-4
DEFAULT_TOLERANCE = 1.0e-2

# Per-term status labels (consistent with the existing component-level cost model).
T_ACCESS_STATUS = "modeled"
T_PREP_STATUS = "modeled"
T_U_STATUS = "spectrum_point_action_only"
T_POST_STATUS = "proxy"
T_AMP_STATUS = "modeled"
T_READOUT_STATUS = "proxy"
CLASSICAL_BASELINE_STATUS = "proxy"

# Existing hardware-aware cost model reused as an optional cross-reference.
HARDWARE_AWARE_CSV = "outputs/hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv"

PER_ROW_CLAIM_BOUNDARY = (
    "selected-observable accounting; modeled/proxy access, preparation, postselection, "
    "amplification; not full-vector readout; not a quantum-hardware run; not a speedup result"
)

COST_COLUMNS = [
    "case",
    "observable_id",
    "alpha",
    "tolerance",
    "degree",
    "shots",
    "success_probability_proxy",
    "unitary_queries_per_attempt",
    "expected_attempts_no_aa",
    "expected_unitary_queries_no_aa",
    "expected_attempts_with_aa_proxy",
    "expected_unitary_queries_with_aa_proxy",
    "state_preparations_no_aa",
    "state_preparations_with_aa_proxy",
    "access_status",
    "prep_status",
    "postselection_status",
    "readout_status",
    "full_vector_recovery_included",
    "dominant_cost_risk",
    "claim_boundary",
    "matrix_source",
    "observable_type",
    "qsvt_query_count",
    "unitary_query_count_status",
    "T_access_status",
    "T_prep_status",
    "T_U_status",
    "T_post_status",
    "T_amp_status",
    "T_readout_status",
    "readout_shots",
    "classical_sparse_baseline_status",
    "classical_sparse_baseline_flops_proxy",
    "bounded_scale_C",
    "readout_relative_error_target",
    "readout_relative_error_at_selected_shots",
    "readout_target_met",
]


def _classical_sparse_baseline_flops(nnz: int, condition_number: float) -> float:
    """Conjugate-gradient cost proxy for sparse Ridge: ~ nnz * sqrt(cond) iterations.

    Each CG iteration costs one sparse mat-vec (~2*nnz flops); the iteration count
    for the regularized normal equations scales as ``O(sqrt(cond))``. This is a
    proxy for context, not a tuned solver count.
    """

    iterations = float(np.sqrt(max(condition_number, 1.0)))
    return float(2.0 * int(nnz) * iterations)


def run_selected_observable_cost_accounting(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    alpha = float(resolved["alpha"])
    tolerance = float(resolved["tolerance"])
    rel_target = float(resolved["readout_relative_error_target"])
    hardware = read_csv(HARDWARE_AWARE_CSV)
    rows: list[dict[str, Any]] = []

    for case in resolved["cases"]:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["seed"]),
            }
        )
        singular_values = system.singular_values()
        condition_number = float(system.condition_number())
        nnz = int(np.count_nonzero(np.abs(system.H_tilde) > 1.0e-12))
        c_alpha, profile = _alpha_degree_profile(singular_values, alpha=alpha)
        degree, _approx = _degree_for_target(profile, tolerance=tolerance)

        update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=alpha)
        success_proxy = _success_probability_proxy(
            case, hardware, update=update, residual=system.r_tilde, c_alpha=c_alpha
        )
        baseline_flops = _classical_sparse_baseline_flops(nnz, condition_number)

        metadata = build_state_metadata_from_system_metadata(system.metadata)
        observables = build_selected_observables(metadata, update)
        for observable in observables:
            sweep = observable_shot_sweep(
                observable,
                update,
                shots_grid=tuple(resolved["shots"]),
                trials=int(resolved["trials"]),
                base_seed=int(resolved["base_seed"]),
            )
            shots = shots_for_relative_error(sweep, target_relative_error=rel_target)
            shots_value = shots if shots is not None else int(max(resolved["shots"]))
            selected_sweep = next(row for row in sweep if int(row["shots"]) == shots_value)
            selected_relative_error = float(selected_sweep["relative_error"])
            target_met = bool(
                np.isfinite(selected_relative_error) and selected_relative_error <= rel_target
            )
            degree_value = "" if degree is None else int(degree)
            query_count = "" if degree is None else int(2 * degree + 1)
            if success_proxy <= 0.0:
                attempts_no_aa = float("inf")
                attempts_aa = float("inf")
            else:
                attempts_no_aa = float(shots_value) / float(success_proxy)
                attempts_aa = float(shots_value) / float(np.sqrt(success_proxy))
            total_queries_no_aa = "" if query_count == "" else attempts_no_aa * int(query_count)
            total_queries_aa = "" if query_count == "" else attempts_aa * int(query_count)
            rows.append(
                {
                    "case": case,
                    "observable_id": observable.observable_id,
                    "alpha": alpha,
                    "tolerance": tolerance,
                    "degree": degree_value,
                    "shots": int(shots_value),
                    "success_probability_proxy": float(success_proxy),
                    "unitary_queries_per_attempt": query_count,
                    "expected_attempts_no_aa": attempts_no_aa,
                    "expected_unitary_queries_no_aa": total_queries_no_aa,
                    "expected_attempts_with_aa_proxy": attempts_aa,
                    "expected_unitary_queries_with_aa_proxy": total_queries_aa,
                    "state_preparations_no_aa": attempts_no_aa,
                    "state_preparations_with_aa_proxy": attempts_aa,
                    "access_status": T_ACCESS_STATUS,
                    "prep_status": T_PREP_STATUS,
                    "postselection_status": T_POST_STATUS,
                    "readout_status": T_READOUT_STATUS,
                    "full_vector_recovery_included": False,
                    "dominant_cost_risk": _dominant_cost_risk(observable, c_alpha),
                    "claim_boundary": PER_ROW_CLAIM_BOUNDARY,
                    "matrix_source": matrix_source,
                    "observable_type": observable.observable_type,
                    "qsvt_query_count": query_count,
                    "unitary_query_count_status": "degree_derived_query_count",
                    "T_access_status": T_ACCESS_STATUS,
                    "T_prep_status": T_PREP_STATUS,
                    "T_U_status": T_U_STATUS,
                    "T_post_status": T_POST_STATUS,
                    "T_amp_status": T_AMP_STATUS,
                    "T_readout_status": T_READOUT_STATUS,
                    "readout_shots": int(shots_value),
                    "classical_sparse_baseline_status": CLASSICAL_BASELINE_STATUS,
                    "classical_sparse_baseline_flops_proxy": baseline_flops,
                    "bounded_scale_C": float(c_alpha),
                    "readout_relative_error_target": (
                        rel_target if shots is not None else f">{rel_target} at max shots"
                    ),
                    "readout_relative_error_at_selected_shots": selected_relative_error,
                    "readout_target_met": target_met,
                }
            )

    cost_frame = pd.DataFrame(rows, columns=COST_COLUMNS)
    cost_csv = output_dir / "selected_observable_cost.csv"
    summary_md = output_dir / "selected_observable_cost_summary.md"
    revised_csv = output_dir / "revised_cost_composition.csv"
    revised_md = output_dir / "revised_cost_composition.md"
    rows_to_table(rows, cost_csv, COST_COLUMNS)
    rows_to_table(rows, revised_csv, COST_COLUMNS)
    summary_text = _summary_markdown(cost_frame, resolved)
    assert_safe(summary_text)
    summary_md.write_text(summary_text, encoding="utf-8")
    revised_md.write_text(summary_text, encoding="utf-8")

    artifacts = {
        "selected_observable_cost_csv": cost_csv,
        "selected_observable_cost_summary_md": summary_md,
        "revised_cost_composition_csv": revised_csv,
        "revised_cost_composition_md": revised_md,
    }
    manifest = write_workload_manifest(
        output_dir=output_dir,
        artifact_name="selected_observable_cost_accounting",
        description=(
            "Selected-observable component-status and repetition-cost ledger with modeled/"
            "proxy/excluded term labels. Not full-vector readout "
            "and not a quantum-hardware run."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[
            *[f"build_engineering_system:{case}:weighted_jacobian" for case in resolved["cases"]],
            HARDWARE_AWARE_CSV,
        ],
        reran_long_experiments=False,
        aggregated_from_existing=not hardware.empty,
        extra={
            "cases": list(resolved["cases"]),
            "alpha": alpha,
            "tolerance": tolerance,
            "excluded_components": ["full_vector_recovery"],
            "full_vector_recovery_included": False,
            "hardware_aware_cross_reference_present": not hardware.empty,
        },
        manifest_name="selected_observable_cost_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "cost": cost_frame, "artifacts": artifacts}


def _success_probability_proxy(
    case: str,
    hardware: pd.DataFrame,
    *,
    update: np.ndarray,
    residual: np.ndarray,
    c_alpha: float,
) -> float:
    """Postselection success-probability proxy in [0, 1].

    Prefers the value from the existing hardware-aware cost model when present;
    otherwise uses ``||dx_alpha||^2 / (C^2 ||r~||^2)``, the squared amplitude of the
    bounded QSVT output relative to the prepared residual state.
    """

    if not hardware.empty and "case" in hardware.columns:
        matched = hardware[hardware["case"].astype(str) == case]
        if not matched.empty and "success_probability_proxy" in matched.columns:
            value = matched.iloc[0]["success_probability_proxy"]
            if pd.notna(value):
                return float(np.clip(float(value), 0.0, 1.0))
    residual_norm_sq = float(np.asarray(residual, dtype=np.float64) @ np.asarray(residual))
    if residual_norm_sq <= 0.0 or c_alpha <= 0.0:
        return 0.0
    update_norm_sq = float(np.asarray(update, dtype=np.float64) @ np.asarray(update))
    return float(np.clip(update_norm_sq / (c_alpha**2 * residual_norm_sq), 0.0, 1.0))


def _dominant_cost_risk(observable: Any, c_alpha: float) -> str:
    if observable.readout_model == BASIS_SAMPLING_MODEL:
        readout = "energy-style basis-sampling shots (proxy)"
    else:
        readout = "sign-aware readout shots (proxy)"
    prep = "state preparation (modeled)"
    if c_alpha > 10.0:
        return f"{prep} and postselection (large bounded C); {readout}"
    return f"{readout}; {prep}"


def _summary_markdown(cost: pd.DataFrame, resolved: dict[str, Any]) -> str:
    lines = [
        "# Selected-Observable Resource-Boundary Ledger",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "## Repetition Cost Composition",
        "",
        "```text",
        "N_U = 2d + 1",
        "N_attempts(no AA) = N_shots / p_succ",
        "N_U,total(no AA) = N_shots * (1/p_succ) * (2d+1)",
        "N_attempts(AA proxy) = N_shots / sqrt(p_succ)",
        "N_U,total(AA proxy) = N_shots * (1/sqrt(p_succ)) * (2d+1)",
        "```",
        "",
        "## Boundary (read first)",
        "",
        *[f"- {statement}" for statement in MANDATORY_BOUNDARY_STATEMENTS],
        "",
        "## Per-Term Status",
        "",
        "| Term | Status | Meaning |",
        "| --- | --- | --- |",
        f"| `T_access` | {T_ACCESS_STATUS} | sparse index/value oracle access (validated "
        "lookup emulator; reversible circuit not synthesized) |",
        f"| `T_prep` | {T_PREP_STATUS} | residual-state preparation loader assumed |",
        f"| `(2d+1)` per attempt | {T_U_STATUS} | degree-derived count from a "
        "spectrum-point action fit; not a realizable uniform-admissible count |",
        f"| `T_post` | {T_POST_STATUS} | postselection success-probability proxy |",
        f"| `T_amp` | {T_AMP_STATUS} | optional amplitude amplification (modeled O(1/sqrt(p))) |",
        f"| `T_readout` | {T_READOUT_STATUS} | selected-observable shot budget |",
        "| `T_full_vector_recovery` | excluded | one readout per state component; out of scope |",
        "",
        "## Per-Observable Accounting",
        "",
        "| Case | Observable | d | Shots | p proxy | Attempts (no AA) | U calls "
        "(no AA) | U calls (AA proxy) | Target |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in cost.iterrows():
        degree_text = "-" if row["degree"] == "" else str(int(row["degree"]))
        lines.append(
            f"| {row['case']} | `{row['observable_id']}` | {degree_text} | "
            f"{int(row['shots'])} | {float(row['success_probability_proxy']):.3e} | "
            f"{float(row['expected_attempts_no_aa']):.3e} | "
            f"{float(row['expected_unitary_queries_no_aa']):.3e} | "
            f"{float(row['expected_unitary_queries_with_aa_proxy']):.3e} | "
            f"{'met' if bool(row['readout_target_met']) else 'MISS'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Shots and postselection repetitions multiply access, residual-state preparation, "
        "and degree-derived unitary calls; they are not additive overheads.",
        "- `success_probability_proxy` is reported for postselection; small values inflate "
        "expected repeats as `1/p`, and the bounded constant `C` directly controls it.",
        "- The classical sparse Ridge/Tikhonov proxy is included for context only; the same "
        "alpha is used by both, so no QSVT-over-Ridge superiority is implied.",
        "- The IEEE-57 predetermined branch-angle functional remains above the 5% relative-"
        "error target at the largest tested 100000-shot budget and is retained as a target miss.",
        "- Full-vector recovery is excluded: it would require one selected-observable readout "
        "per state component.",
        "",
    ]
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(WORKLOAD_DIR),
        "cases": list(DEFAULT_CASES),
        "case_source": "pypower",
        "seed": 123,
        "alpha": DEFAULT_ALPHA,
        "tolerance": DEFAULT_TOLERANCE,
        "shots": [100, 1_000, 10_000, 100_000],
        "trials": 200,
        "base_seed": 20240601,
        "readout_relative_error_target": 0.05,
        "command": "run_selected_observable_cost_accounting",
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["shots"] = [int(s) for s in resolved["shots"]]
    resolved["alpha"] = float(resolved["alpha"])
    resolved["tolerance"] = float(resolved["tolerance"])
    if resolved["alpha"] <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run selected-observable cost accounting")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    run = run_selected_observable_cost_accounting(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "seed": args.seed,
            "alpha": args.alpha,
            "tolerance": args.tolerance,
            "command": "scripts/run_selected_observable_cost_accounting.py " + " ".join(argv or []),
        }
    )
    cost_path = run["artifacts"]["selected_observable_cost_csv"]
    print(f"Selected-observable cost accounting complete: {cost_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
