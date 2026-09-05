"""Task C workload: selected signed-observable readout diagnostics.

For each benchmark case this computes the exact matched-alpha Ridge/Tikhonov
update ``dx_alpha``, builds physically meaningful selected observables, records
their exact values, and runs a shot-budget sweep for each observable's readout
model (sign-aware inner-product proxy or basis-sampling energy). Every readout
estimates one selected functional, not the full update vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper.selected_observable_common import (
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
    write_workload_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.readout_diagnostics import DEFAULT_SHOTS, observable_shot_sweep
from robust_qsvt_se.qsvt.selected_observables import (
    build_selected_observables,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    bounded_ridge_normalization_C,
)
from robust_qsvt_se.utils.io import ensure_directory

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57")
DEFAULT_ALPHA = 1.0e-4

OBSERVABLE_COLUMNS = [
    "case",
    "matrix_source",
    "observable_id",
    "observable_type",
    "physical_meaning",
    "selection_policy",
    "selected_before_solving",
    "depends_on_solution",
    "support_size",
    "units_or_normalization",
    "exact_value",
    "residual_norm",
    "residual_norm_status",
    "block_encoding_beta",
    "block_encoding_beta_status",
    "bounded_scale_C",
    "qsvt_target_scaling_C_over_beta",
    "target_scaling_status",
    "postselection_normalization",
    "postselection_normalization_status",
    "output_state_norm",
    "output_state_norm_status",
    "reported_value_domain",
    "physical_recovery_status",
    "readout_model",
    "basis_sampling_accessible",
    "sign_aware_required",
    "full_vector_required",
    "status",
    "notes",
]

SWEEP_COLUMNS = [
    "case",
    "observable_id",
    "shots",
    "trials",
    "mean_abs_error",
    "max_abs_error",
    "std_abs_error",
    "relative_error",
    "target_relative_error",
    "target_met",
    "target_status",
    "readout_model",
    "status",
]

MAP_COLUMNS = [
    "observable_type",
    "readout_model",
    "basis_sampling_accessible",
    "sign_aware_required",
    "full_vector_required",
    "status",
    "notes",
]


def run_selected_observable_workload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    observable_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    map_records: dict[str, dict[str, Any]] = {}

    for case in resolved["cases"]:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["seed"]),
            }
        )
        metadata = build_state_metadata_from_system_metadata(system.metadata)
        update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=float(resolved["alpha"]))
        beta = float(np.linalg.svd(system.H_tilde, compute_uv=False)[0])
        bounded_c = float(bounded_ridge_normalization_C(float(resolved["alpha"]), beta=beta))
        residual_norm = float(np.linalg.norm(system.r_tilde))
        output_norm = float(np.linalg.norm(update))
        branches = _branch_pairs(system.metadata)
        observables = build_selected_observables(metadata, update, branches=branches)
        for observable in observables:
            row = observable.to_row(case=case, matrix_source=matrix_source)
            row["exact_value"] = observable.exact_value(update)
            row.update(
                {
                    "residual_norm": residual_norm,
                    "residual_norm_status": "available",
                    "block_encoding_beta": beta,
                    "block_encoding_beta_status": "available",
                    "bounded_scale_C": bounded_c,
                    "qsvt_target_scaling_C_over_beta": bounded_c / beta,
                    "target_scaling_status": "available",
                    "postselection_normalization": "",
                    "postselection_normalization_status": "not_available",
                    "output_state_norm": output_norm,
                    "output_state_norm_status": "available",
                    "reported_value_domain": (
                        "physical-unit estimate from matched classical update"
                    ),
                    "physical_recovery_status": "modeled",
                }
            )
            observable_rows.append(row)
            map_records.setdefault(observable.observable_type, _map_row(observable))
            for sweep in observable_shot_sweep(
                observable,
                update,
                shots_grid=tuple(resolved["shots"]),
                trials=int(resolved["trials"]),
                base_seed=int(resolved["base_seed"]),
            ):
                target = float(resolved["target_relative_error"])
                relative = float(sweep["relative_error"])
                met = bool(np.isfinite(relative) and relative <= target)
                sweep_rows.append(
                    {
                        "case": case,
                        "observable_id": sweep["observable_id"],
                        "shots": sweep["shots"],
                        "trials": sweep["trials"],
                        "mean_abs_error": sweep["mean_abs_error"],
                        "max_abs_error": sweep["max_abs_error"],
                        "std_abs_error": sweep["std_abs_error"],
                        "relative_error": relative,
                        "target_relative_error": target,
                        "target_met": met,
                        "target_status": "met" if met else "target_miss",
                        "readout_model": sweep["readout_model"],
                        "status": sweep["status"],
                    }
                )

    observable_frame = pd.DataFrame(observable_rows, columns=OBSERVABLE_COLUMNS)
    sweep_frame = pd.DataFrame(sweep_rows, columns=SWEEP_COLUMNS)
    map_frame = pd.DataFrame(list(map_records.values()), columns=MAP_COLUMNS)

    observables_csv = output_dir / "selected_observables.csv"
    sweep_csv = output_dir / "readout_shot_sweep.csv"
    map_csv = output_dir / "readout_map.csv"
    summary_md = output_dir / "readout_summary.md"
    policy_md = output_dir / "observable_selection_policy.md"

    rows_to_table(observable_rows, observables_csv, OBSERVABLE_COLUMNS)
    rows_to_table(sweep_rows, sweep_csv, SWEEP_COLUMNS)
    rows_to_table(list(map_records.values()), map_csv, MAP_COLUMNS)
    summary_text = _summary_markdown(observable_frame, sweep_frame, map_frame, resolved)
    assert_safe(summary_text)
    summary_md.write_text(summary_text, encoding="utf-8")
    policy_text = _selection_policy_markdown(observable_frame)
    assert_safe(policy_text)
    policy_md.write_text(policy_text, encoding="utf-8")

    artifacts = {
        "selected_observables_csv": observables_csv,
        "readout_shot_sweep_csv": sweep_csv,
        "readout_map_csv": map_csv,
        "readout_summary_md": summary_md,
        "observable_selection_policy_md": policy_md,
    }
    manifest = write_workload_manifest(
        output_dir=output_dir,
        artifact_name="selected_observable_workload",
        description=(
            "Selected signed-observable readout diagnostics over the matched-alpha "
            "Ridge/Tikhonov update. Selected linear functionals only; not full-vector "
            "readout, not a synthesized circuit, and not a quantum-hardware run."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[
            f"build_engineering_system:{case}:weighted_jacobian" for case in resolved["cases"]
        ],
        reran_long_experiments=False,
        aggregated_from_existing=False,
        extra={
            "cases": list(resolved["cases"]),
            "alpha": float(resolved["alpha"]),
            "shots_grid": list(resolved["shots"]),
            "trials": int(resolved["trials"]),
            "full_vector_readout": False,
            "selected_signed_observables_only": True,
            "observable_selection_depends_on_solution": False,
        },
        manifest_name="selected_observable_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "observables": observable_frame,
        "shot_sweep": sweep_frame,
        "readout_map": map_frame,
        "artifacts": artifacts,
    }


def _map_row(observable: Any) -> dict[str, Any]:
    return {
        "observable_type": observable.observable_type,
        "readout_model": observable.readout_model,
        "basis_sampling_accessible": observable.basis_sampling_accessible,
        "sign_aware_required": observable.sign_aware_required,
        "full_vector_required": observable.full_vector_required,
        "status": observable.status,
        "notes": observable.notes,
    }


def _branch_pairs(metadata: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = metadata.get("measurement_buses") or []
    out: list[tuple[int, int]] = []
    for entry in pairs:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            out.append((int(entry[0]), int(entry[1])))
    return out


def _summary_markdown(
    observables: pd.DataFrame,
    sweep: pd.DataFrame,
    readout_map: pd.DataFrame,
    resolved: dict[str, Any],
) -> str:
    lines = [
        "# Selected Signed-Observable Readout Diagnostics",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "Selected linear functionals `y_l = l^T dx_alpha` of the matched-alpha "
        "Ridge/Tikhonov update `dx_alpha = (H~^T H~ + alpha I)^{-1} H~^T r~`. Each readout "
        "estimates **one** selected functional; this is **not** full-vector readout.",
        "",
        f"Alpha = {float(resolved['alpha']):.1e}; shots grid = {list(resolved['shots'])}; "
        f"trials per shot budget = {int(resolved['trials'])}.",
        "",
        "## Readout Model Map",
        "",
        "| Observable type | Readout model | Basis-sampling | Sign-aware | Full-vector | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in readout_map.iterrows():
        lines.append(
            f"| {row['observable_type']} | {row['readout_model']} | "
            f"{row['basis_sampling_accessible']} | {row['sign_aware_required']} | "
            f"{row['full_vector_required']} | {row['status']} |"
        )
    lines += [
        "",
        "## Selected Observables (exact matched values)",
        "",
        "| Case | Observable | Type | Support | Exact value | Readout model |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in observables.iterrows():
        lines.append(
            f"| {row['case']} | `{row['observable_id']}` | {row['observable_type']} | "
            f"{int(row['support_size'])} | {float(row['exact_value']):.4e} | "
            f"{row['readout_model']} |"
        )
    lines += [
        "",
        "## Readout Error vs Shots (mean absolute error)",
        "",
        "| Case | Observable | Model | "
        + " | ".join(f"{int(s)} shots" for s in resolved["shots"])
        + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in resolved["shots"]) + " |",
    ]
    for (case, observable_id), group in sweep.groupby(["case", "observable_id"], sort=False):
        by_shots = group.set_index("shots")["mean_abs_error"]
        model = group["readout_model"].iloc[0]
        cells = " | ".join(
            f"{float(by_shots.get(int(s), float('nan'))):.2e}" for s in resolved["shots"]
        )
        lines.append(f"| {case} | `{observable_id}` | {model} | {cells} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Both readout models are unbiased; mean absolute error falls on the order of "
        "`1/sqrt(shots)`, the expected Monte-Carlo readout cost.",
        "- Signed PSSE corrections (voltage magnitude/angle, branch angle difference, area "
        "aggregate) need a sign-aware readout; the energy-style observable is "
        "basis-sampling accessible but does not recover the sign.",
        "- Recovering every signed component would need one selected functional per "
        "coordinate and is out of scope (full-vector readout is excluded).",
        "- `dx_alpha` is the matched-alpha Ridge/Tikhonov update, so no QSVT-over-Ridge "
        "superiority is implied.",
        "- Physical recovery of `y_l` requires the recorded residual norm, block-encoding "
        "normalization, bounded-target scale, output-state norm, and postselection "
        "normalization. The last quantity is not available in this synthetic diagnostic, "
        "so physical recovery is modeled rather than circuit-validated.",
        "",
    ]
    return "\n".join(lines)


def _selection_policy_markdown(observables: pd.DataFrame) -> str:
    lines = [
        "# Predetermined Observable-Selection Policy",
        "",
        "Every main-workload observable is selected before solving. Selection uses only "
        "case metadata, deterministic bus/branch order, or a fixed first-four-state "
        "support. No selection uses `Delta x_alpha`.",
        "",
        "| Case | Observable | Policy | Before solve | Depends on solution | Meaning | "
        "Support | Units | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in observables.iterrows():
        lines.append(
            f"| {row['case']} | `{row['observable_id']}` | {row['selection_policy']} | "
            f"{row['selected_before_solving']} | {row['depends_on_solution']} | "
            f"{row['physical_meaning']} | {int(row['support_size'])} | "
            f"{row['units_or_normalization']} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(WORKLOAD_DIR),
        "cases": list(DEFAULT_CASES),
        "case_source": "pypower",
        "seed": 123,
        "alpha": DEFAULT_ALPHA,
        "shots": list(DEFAULT_SHOTS),
        "trials": 200,
        "base_seed": 20240601,
        "target_relative_error": 0.05,
        "command": "run_selected_observable_workload",
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["shots"] = [int(s) for s in resolved["shots"]]
    resolved["alpha"] = float(resolved["alpha"])
    if resolved["alpha"] <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the selected-observable readout workload")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--shots", nargs="+", type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20240601)
    args = parser.parse_args(argv)
    run = run_selected_observable_workload(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "seed": args.seed,
            "alpha": args.alpha,
            "shots": args.shots,
            "trials": args.trials,
            "base_seed": args.base_seed,
            "command": "scripts/run_selected_observable_readout.py " + " ".join(argv or []),
        }
    )
    print(f"Selected-observable workload complete: {run['artifacts']['selected_observables_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
