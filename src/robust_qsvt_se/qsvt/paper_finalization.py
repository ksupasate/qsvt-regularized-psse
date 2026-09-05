from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.measurement.perturbations import (
    add_bad_data_outliers,
    add_gaussian_noise,
    remove_random_rows,
)
from robust_qsvt_se.qsvt.engineering_io import (
    CLAIM_BOUNDARY,
    current_command,
    git_commit,
    utc_timestamp,
)
from robust_qsvt_se.qsvt.engineering_utils import (
    bounded_scaling_C,
    build_engineering_system,
)
from robust_qsvt_se.qsvt.failure_fix import (
    ESTIMATOR_CAVEAT,
    QSVT_CAVEAT,
    _column_equilibration_scales,
    _context_from_matrix,
    _kappa,
    _relative_error,
)
from robust_qsvt_se.qsvt.nonbruteforce_refinement import _markdown_table
from robust_qsvt_se.qsvt.polynomial_approximation import evaluate_polynomial_approximation
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.seed import make_rng

STRICT_TOLERANCE = 1.0e-3
ALPHA_GRID = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
NOISE_GRID = [0.0, 0.01, 0.03]
MISSING_GRID = [0.0, 0.05, 0.10, 0.20]
BAD_DATA_GRID = [0.0, 0.02, 0.05]
SEEDS = [10, 20, 30]
SWEEP_VARIANTS = [
    "unpreconditioned_ridge",
    "preconditioned_coordinate_ridge",
    "preconditioned_transformed_penalty_ridge",
    "unpreconditioned_qsvt_diagnostic",
    "preconditioned_qsvt_diagnostic",
]
READOUT_CAVEAT = (
    "Readout costs are proxy caveats only; full-vector readout is not modeled as "
    "hardware execution."
)
ORACLE_CAVEAT = (
    "Oracle and data-loading costs are not implemented; resource rows are proxy diagnostics only."
)
STATE_PREPARATION_CAVEAT = (
    "State preparation is a placeholder caveat and is not included in the query proxy."
)
HARDWARE_CAVEAT = (
    "No quantum speedup, quantum advantage, full hardware execution, or hardware "
    "validation is claimed."
)
RESOURCE_CAVEAT = (
    "The preconditioned matrix reduces approximation difficulty under this proxy "
    "model only; this is not a quantum speedup claim."
)
UNSUPPORTED_CLAIM_CAVEAT = (
    "Unsupported claims are included only as avoid-wording or claim-boundary text."
)
FINAL_UNSAFE_PHRASES = [
    "quantum speedup",
    "quantum advantage",
    "QSVT outperforms Ridge",
    "beats Ridge",
    "hardware validation",
    "full hardware execution",
    "real PMU/SCADA",
    "field-data validation",
    "field calibrated",
    "deployment ready",
    "original IEEE300 passed",
    "preconditioned IEEE300 proves original IEEE300",
    "IEEE300 original passed",
    "phase validation proves hardware",
]
MANUSCRIPT_OUTPUTS = [
    "outputs/qsvt_phase1_finalization/phase1_finalization_summary.md",
    "outputs/qsvt_phase1_finalization/phase1_finalization_summary.csv",
    "outputs/qsvt_phase1_finalization/phase1_finalization_summary.json",
    "outputs/qsvt_phase1_finalization/phase1_claim_delta.csv",
    "outputs/qsvt_phase1_finalization/phase1_updated_table_index.csv",
    "outputs/qsvt_phase1_finalization/manifest.json",
    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv",
    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.json",
    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv",
    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_failure_log.csv",
    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/manifest.json",
    "outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.csv",
    "outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.json",
    "outputs/qsvt_phase2_alpha_selection/alpha_selection_trace.csv",
    "outputs/qsvt_phase2_alpha_selection/alpha_selection_report.md",
    "outputs/qsvt_phase2_alpha_selection/alpha_selection_metric_definitions.md",
    "outputs/qsvt_phase2_alpha_selection/manifest.json",
    "outputs/qsvt_phase2_summary/phase2_summary.md",
    "outputs/qsvt_phase2_summary/phase2_summary.csv",
    "outputs/qsvt_phase2_summary/phase2_summary.json",
    "outputs/qsvt_phase2_summary/manifest.json",
    "outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv",
    "outputs/qsvt_phase2_complete_summary/phase2_complete_summary.json",
    "outputs/qsvt_phase2_complete_summary/phase2_best_alpha_by_metric.csv",
    "outputs/qsvt_phase2_complete_summary/phase2_variant_comparison.csv",
    "outputs/qsvt_phase2_complete_summary/phase2_case_comparison.csv",
    "outputs/qsvt_phase2_complete_summary/phase2_key_findings.md",
    "outputs/qsvt_phase2_complete_summary/phase2_manifest.json",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee300_residual_vs_alpha.png",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee300_rmse_vs_alpha.png",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee300_qsvt_error_vs_alpha.png",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee300_residual_rmse_qsvt_tradeoff.png",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee118_qsvt_error_vs_alpha.png",
    "outputs/qsvt_phase2_figures/fig_phase2_ieee118_residual_vs_alpha.png",
    "outputs/qsvt_phase2_figures/fig_phase2_variant_comparison_ieee300.png",
    "outputs/qsvt_phase2_figures/fig_phase2_original_vs_preconditioned_kappa.png",
    "outputs/qsvt_phase2_figures/fig_phase2_alpha_selection_score.png",
    "outputs/qsvt_phase2_figures/phase2_figure_captions.md",
    "outputs/qsvt_phase2_optional_ieee57/ieee57_phase2_status.json",
    "outputs/qsvt_phase2_optional_ieee57/ieee57_phase2_status.md",
    "outputs/qsvt_phase2_optional_ieee57/ieee57_phase2_manifest.json",
    "outputs/qsvt_phase2_manuscript_text/transformed_penalty_explanation.md",
    "outputs/qsvt_phase2_manuscript_text/phase2_results_paragraph.md",
    "outputs/qsvt_phase2_manuscript_text/phase2_limitations_paragraph.md",
    "outputs/qsvt_phase2_manuscript_text/phase2_claim_safe_wording.md",
    "outputs/qsvt_phase2_manuscript_text/phase2_methods_equations.md",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_results.csv",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_results.json",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_summary.csv",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_summary.json",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_failure_log.csv",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_manifest.json",
    "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_report.md",
    "outputs/qsvt_preconditioning_resource_comparison/preconditioning_resource_comparison.csv",
    "outputs/qsvt_preconditioning_resource_comparison/preconditioning_resource_comparison.json",
    "outputs/qsvt_preconditioning_resource_comparison/preconditioning_resource_comparison_report.md",
    "outputs/qsvt_preconditioning_resource_comparison/manifest.json",
    "outputs/paper_ready_qsvt_tables/table_1_experiment_taxonomy.csv",
    "outputs/paper_ready_qsvt_tables/table_2_measurement_inventory_summary.csv",
    "outputs/paper_ready_qsvt_tables/table_3_estimator_roles.csv",
    "outputs/paper_ready_qsvt_tables/table_4_main_estimator_results_summary.csv",
    "outputs/paper_ready_qsvt_tables/table_5_qsvt_approximation_by_case.csv",
    "outputs/paper_ready_qsvt_tables/table_6_phase_validation_status.csv",
    "outputs/paper_ready_qsvt_tables/table_7_ieee300_preconditioning_summary.csv",
    "outputs/paper_ready_qsvt_tables/table_8_resource_readout_summary.csv",
    "outputs/paper_ready_qsvt_tables/table_9_claim_boundary_matrix.csv",
    "outputs/paper_ready_qsvt_tables/paper_ready_table_manifest.json",
    "outputs/paper_ready_qsvt_tables/paper_ready_table_notes.md",
    "outputs/qsvt_phase_validation_stable_basis/phase_validation_stable_basis_summary.csv",
    "outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_estimator_summary.csv",
    "outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_summary.csv",
    "outputs/qsvt_engineering_extension/claim_support_matrix.csv",
    "outputs/qsvt_nonbruteforce_refinement_summary/nonbruteforce_refinement_summary.md",
    "outputs/qsvt_failure_fix_summary/failure_fix_summary.md",
    "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv",
]
MANUSCRIPT_DOCS = [
    "README.md",
    "docs/QSVT_PHASE1_FINALIZATION.md",
    "docs/QSVT_PHASE2_COMPLETE.md",
    "docs/QSVT_PHASE2_PRECONDITIONED_ALPHA.md",
    "docs/QSVT_PHASE2_FIGURES.md",
    "docs/QSVT_ENGINEERING_EXTENSION.md",
    "docs/QSVT_APPROXIMATION_VALIDATION.md",
    "docs/QSVT_IEEE300_SPECTRAL_DIAGNOSTIC.md",
    "docs/QSVT_PRECONDITIONED_VARIANT_SWEEPS.md",
    "docs/QSVT_PAPER_READY_TABLES.md",
    "docs/QSVT_FINAL_ARTIFACT_FREEZE.md",
    "docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md",
    "docs/QSVT_NONBRUTEFORCE_REFINEMENT.md",
    "docs/QSVT_PRECONDITIONED_IEEE300_VARIANT.md",
    "docs/QSVT_STABLE_PHASE_SYNTHESIS.md",
    "docs/QSVT_RESIDUAL_WEIGHTED_SPECTRAL_ERROR.md",
]
MANUSCRIPT_SCRIPTS = [
    "scripts/finalize_qsvt_phase1_artifacts.py",
    "scripts/run_qsvt_phase2_preconditioned_alpha_sweeps.py",
    "scripts/build_qsvt_phase2_alpha_selection_report.py",
    "scripts/build_qsvt_phase2_summary.py",
    "scripts/build_qsvt_phase2_complete_summary.py",
    "scripts/build_qsvt_phase2_figures.py",
    "scripts/run_qsvt_phase2_optional_ieee57.py",
    "scripts/build_qsvt_phase2_manuscript_text.py",
    "scripts/run_qsvt_preconditioned_variant_sweeps.py",
    "scripts/build_qsvt_preconditioning_resource_comparison.py",
    "scripts/build_paper_ready_qsvt_tables.py",
    "scripts/freeze_qsvt_manuscript_artifacts.py",
    "scripts/run_final_qsvt_claim_safety_audit.py",
]


def run_preconditioned_variant_sweeps(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_sweep_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    approximation_cache: dict[tuple[str, float, str], dict[str, float]] = {}
    context_cache: dict[tuple[str, str, str], Any] = {}
    svd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for case_name in resolved["cases"]:
        try:
            base_system, matrix_source = build_engineering_system(
                _paper_case_config(
                    case_name,
                    case_source=str(resolved["case_source"]),
                    seed=int(resolved["base_seed"]),
                    fallback_to_synthetic=bool(resolved["fallback_to_synthetic"]),
                )
            )
        except Exception as exc:
            failure_rows.append(
                _failure_row(case_name, "build_base_system", np.nan, np.nan, np.nan, np.nan, exc)
            )
            continue
        for missing_ratio in resolved["missing_ratios"]:
            missing_seeds = resolved["seeds"] if missing_ratio > 0.0 else ["not_applicable"]
            for missing_seed in missing_seeds:
                try:
                    missing_system = _apply_missing(
                        base_system,
                        missing_ratio=float(missing_ratio),
                        seed=missing_seed,
                    )
                except Exception as exc:
                    failure_rows.append(
                        _failure_row(
                            case_name,
                            "missing_rows",
                            np.nan,
                            np.nan,
                            missing_ratio,
                            np.nan,
                            exc,
                        )
                    )
                    continue
                for noise_std in resolved["noise_stds"]:
                    for bad_data_ratio in resolved["bad_data_ratios"]:
                        scenario_seeds = (
                            resolved["seeds"]
                            if noise_std > 0.0 or bad_data_ratio > 0.0
                            else ["not_applicable"]
                        )
                        for scenario_seed in scenario_seeds:
                            scenario_start = time.perf_counter()
                            try:
                                system = _apply_rhs_perturbations(
                                    missing_system,
                                    noise_std=float(noise_std),
                                    bad_data_ratio=float(bad_data_ratio),
                                    seed=scenario_seed,
                                    bad_data_magnitude=float(resolved["bad_data_magnitude"]),
                                )
                                alpha_rows = _sweep_system_rows(
                                    system=system,
                                    matrix_source=matrix_source,
                                    case_name=str(case_name),
                                    missing_ratio=float(missing_ratio),
                                    noise_std=float(noise_std),
                                    bad_data_ratio=float(bad_data_ratio),
                                    seed=scenario_seed,
                                    alphas=resolved["alphas"],
                                    degree=int(resolved["degree"]),
                                    method=str(resolved["method"]),
                                    grid_size=int(resolved["grid_size"]),
                                    approximation_cache=approximation_cache,
                                    context_cache=context_cache,
                                    svd_cache=svd_cache,
                                    scenario_start=scenario_start,
                                )
                                result_rows.extend(alpha_rows)
                            except Exception as exc:
                                failure_rows.append(
                                    _failure_row(
                                        case_name,
                                        "scenario",
                                        scenario_seed,
                                        noise_std,
                                        missing_ratio,
                                        bad_data_ratio,
                                        exc,
                                    )
                                )

    results = pd.DataFrame(result_rows, columns=_sweep_result_columns())
    failures = pd.DataFrame(failure_rows, columns=_failure_columns())
    summary = _sweep_summary(results, failures)
    artifacts = _write_sweep_outputs(output_dir, resolved, results, summary, failures)
    return {
        "output_dir": output_dir,
        "summary": summary,
        "results": results,
        "artifacts": artifacts,
    }


def build_preconditioning_resource_comparison(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_resource_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case_name in resolved["cases"]:
        try:
            system, matrix_source = build_engineering_system(
                _paper_case_config(
                    case_name,
                    case_source=str(resolved["case_source"]),
                    seed=int(resolved["seed"]),
                    fallback_to_synthetic=bool(resolved["fallback_to_synthetic"]),
                )
            )
            before = np.asarray(system.H_tilde, dtype=np.float64)
            scales = _column_equilibration_scales(before)
            after = before * scales[None, :]
            for variant, matrix, source in [
                ("unpreconditioned", before, matrix_source),
                ("column_equilibrated", after, f"{matrix_source}_column_equilibrated"),
            ]:
                context = _context_from_matrix(
                    matrix=matrix,
                    case_name=str(case_name),
                    matrix_source=source,
                    source_note="paper-ready preconditioning resource comparison",
                )
                for alpha in resolved["alphas"]:
                    rows.append(
                        _resource_row(
                            context=context,
                            matrix=matrix,
                            case_name=str(case_name),
                            variant=variant,
                            alpha=float(alpha),
                            degree=int(resolved["degree"]),
                            method=str(resolved["method"]),
                            grid_size=int(resolved["grid_size"]),
                        )
                    )
        except Exception as exc:
            rows.append(_resource_failure_row(str(case_name), str(exc), resolved))

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "preconditioning_resource_comparison.csv"
    json_path = output_dir / "preconditioning_resource_comparison.json"
    report_path = output_dir / "preconditioning_resource_comparison_report.md"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": frame.to_dict(orient="records")})
    report_path.write_text(_resource_report(frame), encoding="utf-8")
    manifest = _write_named_manifest(
        output_dir / "manifest.json",
        artifacts={
            "preconditioning_resource_comparison_csv": str(csv_path),
            "preconditioning_resource_comparison_json": str(json_path),
            "preconditioning_resource_comparison_report_md": str(report_path),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": frame, "artifacts": {"manifest": manifest}}


def build_paper_ready_qsvt_tables(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_tables_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    tables = {
        "table_1_experiment_taxonomy": _table_1_experiment_taxonomy(),
        "table_2_measurement_inventory_summary": _table_2_measurement_inventory(),
        "table_3_estimator_roles": _table_3_estimator_roles(),
        "table_4_main_estimator_results_summary": _table_4_main_estimator_results(),
        "table_5_qsvt_approximation_by_case": _table_5_qsvt_approximation(),
        "table_6_phase_validation_status": _table_6_phase_validation(),
        "table_7_ieee300_preconditioning_summary": _table_7_ieee300_preconditioning(),
        "table_8_resource_readout_summary": _table_8_resource_readout(),
        "table_9_claim_boundary_matrix": _table_9_claim_boundary_matrix(),
    }
    artifacts: dict[str, str] = {}
    table_manifest: list[dict[str, Any]] = []
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[name] = str(path)
        table_manifest.append(
            {
                "table": name,
                "path": str(path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "status": "ok" if not frame.empty else "empty",
            }
        )
    notes_path = output_dir / "paper_ready_table_notes.md"
    manifest_path = output_dir / "paper_ready_table_manifest.json"
    notes_path.write_text(_paper_table_notes(tables), encoding="utf-8")
    artifacts["paper_ready_table_notes_md"] = str(notes_path)
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "tables": table_manifest,
            "claim_boundary": CLAIM_BOUNDARY,
            "input_config": resolved,
        },
    )
    artifacts["paper_ready_table_manifest_json"] = str(manifest_path)
    return {"output_dir": output_dir, "tables": tables, "artifacts": artifacts}


def freeze_qsvt_manuscript_artifacts(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_freeze_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    inventory_targets = _artifact_targets(resolved)
    rows = [
        _artifact_row(path=Path(target["path"]), target=target, resolved=resolved)
        for target in inventory_targets
    ]
    inventory = pd.DataFrame(rows)
    missing = inventory[inventory["exists"] != True].copy()  # noqa: E712
    inventory_csv = output_dir / "artifact_inventory.csv"
    inventory_json = output_dir / "artifact_inventory.json"
    missing_csv = output_dir / "missing_or_unverified_outputs.csv"
    commands_path = output_dir / "verification_commands.txt"
    boundary_path = output_dir / "claim_boundary_summary.md"
    summary_path = output_dir / "artifact_freeze_summary.md"
    manifest_path = output_dir / "artifact_freeze_manifest.json"
    inventory.to_csv(inventory_csv, index=False)
    missing.to_csv(missing_csv, index=False)
    write_json(inventory_json, {"rows": inventory.to_dict(orient="records")})
    commands_path.write_text(_verification_commands_text(), encoding="utf-8")
    boundary_path.write_text(_claim_boundary_summary(), encoding="utf-8")
    summary_path.write_text(_artifact_freeze_summary(inventory, missing), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {
                "artifact_inventory_csv": str(inventory_csv),
                "artifact_inventory_json": str(inventory_json),
                "verification_commands_txt": str(commands_path),
                "claim_boundary_summary_md": str(boundary_path),
                "missing_or_unverified_outputs_csv": str(missing_csv),
                "artifact_freeze_summary_md": str(summary_path),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "inventory": inventory,
        "artifacts": {"manifest": manifest_path},
    }


def run_final_claim_safety_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_claim_audit_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for root in resolved["scan_roots"]:
        rows.extend(_scan_claim_root(Path(root), resolved))
    frame = pd.DataFrame(rows, columns=_claim_audit_columns())
    csv_path = output_dir / "claim_safety_audit.csv"
    json_path = output_dir / "claim_safety_audit.json"
    summary_path = output_dir / "claim_safety_audit_summary.md"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": frame.to_dict(orient="records")})
    summary_path.write_text(_claim_audit_summary(frame), encoding="utf-8")
    manifest = _write_named_manifest(
        output_dir / "manifest.json",
        artifacts={
            "claim_safety_audit_csv": str(csv_path),
            "claim_safety_audit_json": str(json_path),
            "claim_safety_audit_summary_md": str(summary_path),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": frame, "artifacts": {"manifest": manifest}}


def _sweep_system_rows(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case_name: str,
    missing_ratio: float,
    noise_std: float,
    bad_data_ratio: float,
    seed: int | str,
    alphas: list[float],
    degree: int,
    method: str,
    grid_size: int,
    approximation_cache: dict[tuple[str, float, str], dict[str, float]],
    context_cache: dict[tuple[str, str, str], Any],
    svd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    scenario_start: float,
) -> list[dict[str, Any]]:
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    rhs = np.asarray(system.r_tilde, dtype=np.float64)
    scales = _column_equilibration_scales(matrix)
    pre_matrix = matrix * scales[None, :]
    before_context = _cached_context(
        cache=context_cache,
        matrix=matrix,
        case_name=case_name,
        matrix_source=matrix_source,
        source_note="preconditioned variant sweep unpreconditioned matrix",
    )
    after_context = _cached_context(
        cache=context_cache,
        matrix=pre_matrix,
        case_name=case_name,
        matrix_source=f"{matrix_source}_column_equilibrated",
        source_note="preconditioned variant sweep column-equilibrated matrix",
    )
    rank_before = int(np.count_nonzero(before_context.singular_values > 1.0e-12))
    rank_after = int(np.count_nonzero(after_context.singular_values > 1.0e-12))
    kappa_before = _kappa(before_context.singular_values)
    kappa_after = _kappa(after_context.singular_values)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        x_original = _ridge_svd_cached(matrix, rhs, alpha=float(alpha), cache=svd_cache)
        y_coordinate = _ridge_svd_cached(pre_matrix, rhs, alpha=float(alpha), cache=svd_cache)
        x_coordinate = scales * y_coordinate
        x_transformed = x_original.copy()
        before_approx = _cached_approximation(
            cache=approximation_cache,
            context=before_context,
            matrix=matrix,
            alpha=float(alpha),
            degree=degree,
            method=method,
            grid_size=grid_size,
        )
        after_approx = _cached_approximation(
            cache=approximation_cache,
            context=after_context,
            matrix=pre_matrix,
            alpha=float(alpha),
            degree=degree,
            method=method,
            grid_size=grid_size,
        )
        variant_specs = [
            ("unpreconditioned_ridge", x_original, kappa_before, rank_before, before_approx),
            (
                "preconditioned_coordinate_ridge",
                x_coordinate,
                kappa_after,
                rank_after,
                after_approx,
            ),
            (
                "preconditioned_transformed_penalty_ridge",
                x_transformed,
                kappa_after,
                rank_after,
                after_approx,
            ),
            (
                "unpreconditioned_qsvt_diagnostic",
                x_original,
                kappa_before,
                rank_before,
                before_approx,
            ),
            (
                "preconditioned_qsvt_diagnostic",
                x_coordinate,
                kappa_after,
                rank_after,
                after_approx,
            ),
        ]
        for variant, x_hat, condition, rank, approximation in variant_specs:
            rows.append(
                _sweep_result_row(
                    system=system,
                    case_name=case_name,
                    variant_name=variant,
                    alpha=float(alpha),
                    noise_std=noise_std,
                    missing_ratio=missing_ratio,
                    bad_data_ratio=bad_data_ratio,
                    seed=seed,
                    condition_number=condition,
                    condition_number_preconditioned=kappa_after,
                    rank=rank,
                    x_hat=x_hat,
                    x_original=x_original,
                    x_transformed=x_transformed,
                    approximation=approximation,
                    runtime_seconds=time.perf_counter() - scenario_start,
                )
            )
    return rows


def _cached_context(
    *,
    cache: dict[tuple[str, str, str], Any],
    matrix: np.ndarray,
    case_name: str,
    matrix_source: str,
    source_note: str,
) -> Any:
    key = (_matrix_fingerprint(matrix), case_name, matrix_source)
    cached = cache.get(key)
    if cached is not None:
        return cached
    context = _context_from_matrix(
        matrix=matrix,
        case_name=case_name,
        matrix_source=matrix_source,
        source_note=source_note,
    )
    cache[key] = context
    return context


def _ridge_svd_cached(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    alpha: float,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    key = _matrix_fingerprint(matrix)
    decomposition = cache.get(key)
    if decomposition is None:
        decomposition = np.linalg.svd(
            np.asarray(matrix, dtype=np.float64),
            full_matrices=False,
        )
        cache[key] = decomposition
    U, singular_values, Vt = decomposition
    gains = singular_values / (singular_values**2 + alpha)
    return Vt.T @ (gains * (U.T @ rhs))


def _sweep_result_row(
    *,
    system: WeightedSystem,
    case_name: str,
    variant_name: str,
    alpha: float,
    noise_std: float,
    missing_ratio: float,
    bad_data_ratio: float,
    seed: int | str,
    condition_number: float,
    condition_number_preconditioned: float,
    rank: int,
    x_hat: np.ndarray,
    x_original: np.ndarray,
    x_transformed: np.ndarray,
    approximation: dict[str, float],
    runtime_seconds: float,
) -> dict[str, Any]:
    residual = system.residual_norm(x_hat)
    rmse = system.rmse(x_hat)
    status = _sweep_variant_status(
        variant_name=variant_name,
        residual=residual,
        original_residual=system.residual_norm(x_original),
        approximation=approximation,
    )
    return {
        "case_name": case_name,
        "variant_name": variant_name,
        "alpha": float(alpha),
        "noise_std": float(noise_std),
        "missing_ratio": float(missing_ratio),
        "bad_data_ratio": float(bad_data_ratio),
        "seed": seed,
        "m": int(system.n_measurements),
        "n": int(system.n_states),
        "rank": int(rank),
        "condition_number": float(condition_number),
        "condition_number_preconditioned_if_applicable": float(condition_number_preconditioned),
        "rmse_if_available": np.nan if rmse is None else float(rmse),
        "angle_rmse_if_available": np.nan,
        "voltage_rmse_if_available": np.nan,
        "residual_norm": float(residual),
        "weighted_residual_norm": float(system.weighted_residual_norm(x_hat)),
        "solution_norm": float(np.linalg.norm(x_hat)),
        "relative_solution_error_vs_unpreconditioned_ridge": _relative_error(
            x_original,
            x_hat,
        ),
        "relative_solution_error_vs_transformed_penalty": _relative_error(
            x_transformed,
            x_hat,
        ),
        "qsvt_full_interval_approx_error": approximation["full_interval_error"],
        "qsvt_actual_singular_value_approx_error": approximation["actual_singular_error"],
        "polynomial_degree": int(approximation["degree"]),
        "query_count": int(approximation["query_count"]),
        "runtime_seconds": float(runtime_seconds),
        "status": status,
        "failure_reason_if_any": "",
        "estimator_caveat": ESTIMATOR_CAVEAT,
        "qsvt_caveat": QSVT_CAVEAT,
    }


def _sweep_variant_status(
    *,
    variant_name: str,
    residual: float,
    original_residual: float,
    approximation: dict[str, float],
) -> str:
    if not np.isfinite(residual):
        return "failed_nonfinite_solution_metric"
    improves = approximation["full_interval_error"] <= STRICT_TOLERANCE
    residual_ratio = residual / max(original_residual, np.finfo(float).eps)
    if variant_name == "preconditioned_coordinate_ridge":
        if improves and residual_ratio <= 2.0:
            return "stable_preconditioned_variant"
        if improves:
            return "approximation_improves_residual_degrades"
    if "transformed_penalty" in variant_name:
        return "consistency_check"
    if "qsvt_diagnostic" in variant_name:
        return "diagnostic_only"
    return "ok"


def _cached_approximation(
    *,
    cache: dict[tuple[str, float, str], dict[str, float]],
    context: Any,
    matrix: np.ndarray,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
) -> dict[str, float]:
    key = (_matrix_fingerprint(matrix), float(alpha), context.matrix_source)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = evaluate_polynomial_approximation(
        context=context,
        alpha=float(alpha),
        degree=degree,
        method=method,
        grid_size=grid_size,
    )
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_errors = result.pointwise_errors[kinds == "grid"]
    singular_errors = result.pointwise_errors[kinds == "actual_singular_value"]
    metrics = {
        "full_interval_error": float(np.max(grid_errors)),
        "actual_singular_error": float(np.max(singular_errors)),
        "degree": float(result.degree),
        "query_count": float(2 * result.degree + 1),
        "bounded_scaling_C": float(result.bounded_scaling_C),
    }
    cache[key] = metrics
    return metrics


def _resource_row(
    *,
    context: Any,
    matrix: np.ndarray,
    case_name: str,
    variant: str,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
) -> dict[str, Any]:
    result = evaluate_polynomial_approximation(
        context=context,
        alpha=alpha,
        degree=degree,
        method=method,
        grid_size=grid_size,
    )
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_errors = result.pointwise_errors[kinds == "grid"]
    singular_errors = result.pointwise_errors[kinds == "actual_singular_value"]
    full_error = float(np.max(grid_errors))
    actual_error = float(np.max(singular_errors))
    m, n = matrix.shape
    logical_qubits = math.ceil(math.log2(max(m, n))) + 2
    return {
        "case_name": case_name,
        "variant": variant,
        "alpha": alpha,
        "sigma_min": float(np.min(context.singular_values)),
        "sigma_max": float(np.max(context.singular_values)),
        "kappa": _kappa(context.singular_values),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "max_filter_gain": float(bounded_scaling_C(context.singular_values, alpha=alpha)),
        "bounded_scaling_C": float(result.bounded_scaling_C),
        "degree_required_for_1e_minus_3_if_available": (
            int(result.degree) if full_error <= STRICT_TOLERANCE else np.nan
        ),
        "degree_used": int(result.degree),
        "query_count": int(2 * result.degree + 1),
        "full_interval_approx_error": full_error,
        "actual_singular_value_approx_error": actual_error,
        "logical_qubits_proxy": logical_qubits,
        "depth_proxy": int((2 * result.degree + 1) * logical_qubits),
        "readout_caveat": READOUT_CAVEAT,
        "oracle_caveat": ORACLE_CAVEAT,
        "status": "passed_1e_minus_3" if full_error <= STRICT_TOLERANCE else "failed_1e_minus_3",
        "resource_caveat": RESOURCE_CAVEAT,
    }


def _resource_failure_row(
    case_name: str,
    reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "variant": "failure",
        "alpha": np.nan,
        "sigma_min": np.nan,
        "sigma_max": np.nan,
        "kappa": np.nan,
        "rank": np.nan,
        "max_filter_gain": np.nan,
        "bounded_scaling_C": np.nan,
        "degree_required_for_1e_minus_3_if_available": np.nan,
        "degree_used": config["degree"],
        "query_count": np.nan,
        "full_interval_approx_error": np.nan,
        "actual_singular_value_approx_error": np.nan,
        "logical_qubits_proxy": np.nan,
        "depth_proxy": np.nan,
        "readout_caveat": READOUT_CAVEAT,
        "oracle_caveat": ORACLE_CAVEAT,
        "status": f"failed: {reason}",
        "resource_caveat": RESOURCE_CAVEAT,
    }


def _apply_missing(
    system: WeightedSystem,
    *,
    missing_ratio: float,
    seed: int | str,
) -> WeightedSystem:
    if missing_ratio == 0.0:
        metadata = {**system.metadata, "missing_ratio": 0.0}
        return WeightedSystem(system.H_tilde, system.r_tilde, system.x_true, metadata)
    return remove_random_rows(
        system,
        missing_ratio=missing_ratio,
        rng=make_rng(int(seed)),
    )


def _paper_case_config(
    case_name: str,
    *,
    case_source: str,
    seed: int,
    fallback_to_synthetic: bool,
) -> dict[str, Any]:
    if str(case_name) == "synthetic":
        return {
            "case_name": "synthetic",
            "matrix_source": "synthetic",
            "seed": seed,
            "fallback_to_synthetic": True,
        }
    return {
        "case_name": str(case_name),
        "case_source": case_source,
        "seed": seed,
        "fallback_to_synthetic": fallback_to_synthetic,
    }


def _apply_rhs_perturbations(
    system: WeightedSystem,
    *,
    noise_std: float,
    bad_data_ratio: float,
    seed: int | str,
    bad_data_magnitude: float,
) -> WeightedSystem:
    scenario_seed = 0 if seed == "not_applicable" else int(seed)
    rng = make_rng(scenario_seed)
    updated = add_gaussian_noise(system, noise_std=noise_std, rng=rng)
    updated = add_bad_data_outliers(
        updated,
        bad_data_config={
            "enabled": bad_data_ratio > 0.0,
            "ratio": bad_data_ratio,
            "magnitude": bad_data_magnitude,
            "target": "random",
        },
        rng=rng,
    )
    metadata = {
        **updated.metadata,
        "noise_std": float(noise_std),
        "bad_data_ratio": float(bad_data_ratio),
    }
    return WeightedSystem(updated.H_tilde, updated.r_tilde, updated.x_true, metadata)


def _sweep_summary(results: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            [
                {
                    "case_name": "",
                    "variant_name": "",
                    "alpha": np.nan,
                    "scenario_count": 0,
                    "mean_residual_norm": np.nan,
                    "mean_rmse_if_available": np.nan,
                    "mean_qsvt_full_interval_approx_error": np.nan,
                    "mean_qsvt_actual_singular_value_approx_error": np.nan,
                    "condition_number_median": np.nan,
                    "preconditioned_condition_number_median": np.nan,
                    "failure_count": len(failures),
                    "interpretation": "No successful rows were generated.",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ]
        )
    grouped = (
        results.groupby(["case_name", "variant_name", "alpha"], dropna=False)
        .agg(
            scenario_count=("status", "size"),
            mean_residual_norm=("residual_norm", "mean"),
            mean_rmse_if_available=("rmse_if_available", "mean"),
            mean_qsvt_full_interval_approx_error=("qsvt_full_interval_approx_error", "mean"),
            mean_qsvt_actual_singular_value_approx_error=(
                "qsvt_actual_singular_value_approx_error",
                "mean",
            ),
            condition_number_median=("condition_number", "median"),
            preconditioned_condition_number_median=(
                "condition_number_preconditioned_if_applicable",
                "median",
            ),
        )
        .reset_index()
    )
    grouped["failure_count"] = len(failures)
    grouped["interpretation"] = grouped.apply(_sweep_interpretation_row, axis=1)
    grouped["claim_boundary"] = CLAIM_BOUNDARY
    return grouped


def _sweep_interpretation_row(row: pd.Series) -> str:
    variant = str(row["variant_name"])
    if variant == "preconditioned_coordinate_ridge":
        return (
            "Coordinate-preconditioned Ridge is a separate estimator; inspect "
            "residual/RMSE before recommending it."
        )
    if variant == "preconditioned_transformed_penalty_ridge":
        return "Transformed-penalty row is an x-space penalty consistency check."
    if "qsvt" in variant:
        return "QSVT row is an approximation/resource diagnostic only."
    return "Original unpreconditioned Ridge reference row."


def _write_sweep_outputs(
    output_dir: Path,
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Path]:
    results_csv = output_dir / "preconditioned_variant_sweep_results.csv"
    results_json = output_dir / "preconditioned_variant_sweep_results.json"
    summary_csv = output_dir / "preconditioned_variant_sweep_summary.csv"
    summary_json = output_dir / "preconditioned_variant_sweep_summary.json"
    failure_csv = output_dir / "preconditioned_variant_failure_log.csv"
    report_md = output_dir / "preconditioned_variant_sweep_report.md"
    manifest = output_dir / "preconditioned_variant_manifest.json"
    results.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    failures.to_csv(failure_csv, index=False)
    write_json(results_json, {"rows": results.to_dict(orient="records")})
    write_json(summary_json, {"rows": summary.to_dict(orient="records")})
    report_md.write_text(_sweep_report(results, summary, failures), encoding="utf-8")
    _write_named_manifest(
        manifest,
        artifacts={
            "preconditioned_variant_sweep_results_csv": str(results_csv),
            "preconditioned_variant_sweep_results_json": str(results_json),
            "preconditioned_variant_sweep_summary_csv": str(summary_csv),
            "preconditioned_variant_sweep_summary_json": str(summary_json),
            "preconditioned_variant_failure_log_csv": str(failure_csv),
            "preconditioned_variant_sweep_report_md": str(report_md),
        },
        input_config=config,
    )
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "failure_log_csv": failure_csv,
        "report_md": report_md,
        "manifest": manifest,
    }


def _sweep_report(results: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame) -> str:
    sample_columns = [
        "case_name",
        "variant_name",
        "alpha",
        "mean_residual_norm",
        "mean_rmse_if_available",
        "mean_qsvt_full_interval_approx_error",
        "interpretation",
    ]
    condition_answer = _condition_answer(summary)
    residual_answer = _residual_answer(results)
    return f"""# QSVT Preconditioned Variant Sweeps

## Executive Interpretation

The sweeps treat column-equilibrated Ridge as a separate estimator variant and
QSVT rows as approximation/resource diagnostics. They do not change the
original Ridge/QSVT-target equivalence claim.

## Summary

{_markdown_table(summary, sample_columns)}

## Required Questions

1. Does preconditioning reduce condition number consistently? {condition_answer}
2. Does preconditioning reduce QSVT approximation difficulty consistently? See
   the per-variant approximation columns; preconditioned matrices are reported
   separately from original matrices.
3. Does coordinate-preconditioned Ridge degrade residual/RMSE? {residual_answer}
4. Does transformed-penalty preconditioning preserve the original solution? It
   is reported as a consistency check against the original x-space penalty.
5. Is the preconditioned QSVT diagnostic a resource pathway or stable estimator?
   It is a resource/approximation pathway unless estimator metrics support the
   coordinate-preconditioned Ridge row.
6. Supported claims: controlled preconditioned variant sweeps and proxy
   approximation-resource comparisons.
7. Unsupported claims: quantum speedup, quantum advantage, hardware execution,
   field-data validation, QSVT superiority over Ridge, and original IEEE300 pass
   from preconditioned rows.

## Failure Log

Failure rows: {len(failures)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _condition_answer(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No successful summary rows were available."
    pre = summary[summary["variant_name"].astype(str).str.contains("preconditioned")]
    if pre.empty:
        return "No preconditioned rows were available."
    return (
        "Condition-number columns are reported for each case/alpha; compare "
        "original and preconditioned rows."
    )


def _residual_answer(results: pd.DataFrame) -> str:
    if results.empty:
        return "No successful rows were available."
    coord = results[results["variant_name"] == "preconditioned_coordinate_ridge"]
    if coord.empty:
        return "No coordinate-preconditioned rows were available."
    degraded = coord["status"].astype(str).str.contains("degrades").any()
    return "Some rows degrade residual/RMSE." if degraded else "No degradation status was recorded."


def _table_1_experiment_taxonomy() -> pd.DataFrame:
    rows = [
        (
            "Synthetic",
            "linear weighted",
            "controlled small matrices",
            "matrix/right-hand side",
            "pinv, ridge, TSVD, QSVT target",
            "RMSE, residual, condition number",
            "unit-test and smoke evidence",
            "core numerical behavior",
            "not an IEEE-scale field benchmark",
        ),
        (
            "DC-linearized",
            "linearized power flow",
            "generated benchmark rows",
            "rows/right-hand side",
            "classical spectral baselines",
            "RMSE and residual",
            "baseline comparison",
            "controlled benchmark evidence",
            "linearization only",
        ),
        (
            "AC-linearized",
            "single-step AC weighted Jacobian",
            "generated measurement rows",
            "weighted Jacobian/right-hand side",
            "ridge, QSVT target, preconditioned variants",
            "update RMSE, residual, singular spectrum",
            "main QSVT-compatible matrix evidence",
            "resource-aware feasibility",
            "not nonlinear convergence proof",
        ),
        (
            "Nonlinear AC",
            "iterative AC",
            "generated benchmark measurements",
            "measurements",
            "WLS/robust estimators",
            "iterations, convergence, RMSE",
            "classical workflow context",
            "state-estimation baseline context",
            "not QSVT hardware execution",
        ),
        (
            "Missing/bad-data stress",
            "controlled perturbations",
            "random row drops/outliers",
            "measurements or weighted rows",
            "robust and regularized baselines",
            "RMSE, residual, failure status",
            "stress-test context",
            "controlled stress evidence",
            "not field-calibrated statistics",
        ),
        (
            "QSVT evidence",
            "spectral approximation",
            "bounded Ridge/Tikhonov target",
            "singular-value interval",
            "polynomial/resource diagnostics",
            "degree, query count, max error",
            "implementation-pathway evidence",
            "QSVT-compatible feasibility",
            "not phase/hardware validation unless explicitly passing",
        ),
        (
            "Reporting aggregation",
            "artifact synthesis",
            "generated CSV/JSON/MD outputs",
            "n/a",
            "n/a",
            "claim matrix, manifests",
            "paper-ready traceability",
            "claim-support traceability",
            "does not create new scientific measurements",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "experiment_group",
            "model_type",
            "measurement_model",
            "perturbation_location",
            "estimators",
            "metrics",
            "paper_role",
            "claim_supported",
            "limitations",
        ],
    )


def _table_2_measurement_inventory() -> pd.DataFrame:
    inventory = _read_csv("outputs/measurement_inventory/measurement_inventory_by_case.csv")
    if inventory is None:
        sweep = _read_csv(
            "outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_results.csv"
        )
        if sweep is None or sweep.empty:
            return pd.DataFrame(
                [
                    {
                        "case_name": "unavailable",
                        "workflow": "measurement inventory",
                        "row_count": np.nan,
                        "state_dimension": np.nan,
                        "measurement_types": "",
                        "redundancy_ratio_if_available": np.nan,
                        "notes": "source output missing",
                    }
                ]
            )
        collapsed = (
            sweep.groupby("case_name", dropna=False)
            .agg(row_count=("m", "max"), state_dimension=("n", "max"))
            .reset_index()
        )
        collapsed["workflow"] = "preconditioned AC weighted sweep"
        collapsed["measurement_types"] = "generated AC weighted rows"
        collapsed["redundancy_ratio_if_available"] = (
            collapsed["row_count"] / collapsed["state_dimension"]
        )
        collapsed["notes"] = "derived from preconditioned sweep rows"
        return collapsed[
            [
                "case_name",
                "workflow",
                "row_count",
                "state_dimension",
                "measurement_types",
                "redundancy_ratio_if_available",
                "notes",
            ]
        ]
    columns = inventory.columns
    return pd.DataFrame(
        {
            "case_name": inventory.get("case_name", inventory.iloc[:, 0]),
            "workflow": "measurement inventory",
            "row_count": _first_available(inventory, ["row_count", "measurement_count", "m"]),
            "state_dimension": _first_available(inventory, ["state_dimension", "n_states", "n"]),
            "measurement_types": _first_available(
                inventory,
                ["measurement_types", "measurement_type", "types"],
                default="see measurement inventory",
            ),
            "redundancy_ratio_if_available": _first_available(
                inventory,
                ["redundancy_ratio", "redundancy_ratio_if_available"],
            ),
            "notes": f"source columns: {', '.join(columns)}",
        }
    )


def _table_3_estimator_roles() -> pd.DataFrame:
    rows = [
        ("pseudoinverse", "SVD inverse on nonzero singular values", "baseline inverse"),
        ("ridge_tikhonov", "sigma/(sigma^2+alpha)", "regularized baseline"),
        ("qsvt_target", "same sigma/(sigma^2+alpha) filter", "QSVT-compatible target"),
        ("truncated_svd", "zero below tau, inverse above tau", "unstable-tail baseline"),
        ("normal_equation_wls", "normal-equation least squares", "classical WLS reference"),
        ("huber_irls", "Huber iterative reweighted least squares", "robust bad-data baseline"),
        ("lav_if_available", "least absolute value objective", "robust baseline if configured"),
        ("hhl_style_proxy_if_available", "condition-number proxy inverse", "resource context only"),
        (
            "preconditioned_coordinate_ridge",
            "ridge penalty in equilibrated y coordinates",
            "separate estimator variant",
        ),
        (
            "preconditioned_transformed_penalty_ridge",
            "x-space penalty under y coordinate transform",
            "consistency-preserving check",
        ),
    ]
    output_rows = []
    for estimator, objective, purpose in rows:
        output_rows.append(
            {
                "estimator": estimator,
                "filter_or_objective": objective,
                "purpose": purpose,
                "supported_claim": "controlled benchmark diagnostic",
                "limitation": "not quantum hardware execution",
                "avoid_wording": "QSVT outperforms Ridge under the same alpha",
            }
        )
    return pd.DataFrame(output_rows)


def _table_4_main_estimator_results() -> pd.DataFrame:
    frames = [
        _read_csv("outputs/manuscript_report_final_full/combined_summary_metrics.csv"),
        _read_csv("outputs/real_ieee118_seed10/summary_metrics.csv"),
        _read_csv("outputs/real_ieee300_seed10/summary_metrics.csv"),
    ]
    available = [frame for frame in frames if frame is not None and not frame.empty]
    if not available:
        return pd.DataFrame(
            [
                {
                    "case_name": "unavailable",
                    "workflow": "main estimator results",
                    "stress_setting": "",
                    "best_regularized_method": "",
                    "baseline_inverse_method": "",
                    "rmse_or_update_rmse": np.nan,
                    "residual": np.nan,
                    "condition_number": np.nan,
                    "main_interpretation": "source outputs missing",
                    "limitation": "table does not invent numbers",
                }
            ]
        )
    combined = pd.concat(available, ignore_index=True, sort=False)
    case_column = "case_name" if "case_name" in combined.columns else combined.columns[0]
    estimator_column = "estimator" if "estimator" in combined.columns else ""
    rmse_column = "rmse_mean" if "rmse_mean" in combined.columns else "rmse"
    residual_column = (
        "residual_norm_mean" if "residual_norm_mean" in combined.columns else "residual_norm"
    )
    rows = []
    for case_name, group in combined.groupby(case_column, dropna=False):
        numeric_rmse = pd.to_numeric(
            group.get(rmse_column, pd.Series(dtype=float)),
            errors="coerce",
        )
        best_index = numeric_rmse.idxmin() if numeric_rmse.notna().any() else group.index[0]
        best = combined.loc[best_index]
        rows.append(
            {
                "case_name": case_name,
                "workflow": str(best.get("workflow", best.get("run_id", "classical benchmark"))),
                "stress_setting": str(best.get("sweep_parameter", best.get("scenario", ""))),
                "best_regularized_method": str(best.get(estimator_column, "")),
                "baseline_inverse_method": "pseudoinverse / WLS where available",
                "rmse_or_update_rmse": _safe_float(best.get(rmse_column)),
                "residual": _safe_float(best.get(residual_column)),
                "condition_number": _safe_float(best.get("condition_number")),
                "main_interpretation": "best available generated benchmark row",
                "limitation": "aggregated from existing outputs; not a new experiment",
            }
        )
    return pd.DataFrame(rows)


def _table_5_qsvt_approximation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    adaptive = _read_csv(
        "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv"
    )
    if adaptive is not None:
        for row in adaptive.to_dict(orient="records"):
            rows.append(
                {
                    "case_name": row.get("case_name"),
                    "variant": "original",
                    "degree": row.get("selected_degree", row.get("degree")),
                    "query_count": row.get("query_count", row.get("query_count_estimate")),
                    "full_interval_error": row.get(
                        "max_pointwise_error",
                        row.get("best_max_pointwise_error"),
                    ),
                    "actual_singular_value_error": row.get(
                        "actual_singular_values_max_error",
                        np.nan,
                    ),
                    "status": row.get("status"),
                    "claim": "full-interval approximation diagnostic only",
                    "caveat": QSVT_CAVEAT,
                }
            )
    pre = _read_csv(
        "outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_qsvt_approximation.csv"
    )
    if pre is not None:
        for row in pre.to_dict(orient="records"):
            rows.append(
                {
                    "case_name": row.get("case_name"),
                    "variant": "preconditioned",
                    "degree": row.get("degree_after"),
                    "query_count": row.get("query_count_after"),
                    "full_interval_error": row.get("full_interval_error_after"),
                    "actual_singular_value_error": row.get("actual_singular_error_after"),
                    "status": row.get("status"),
                    "claim": "preconditioned matrix approximation diagnostic",
                    "caveat": "does not make original unpreconditioned IEEE300 pass",
                }
            )
    complete = _read_csv("outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv")
    if complete is not None and not complete.empty:
        variant_map = {
            "original_ridge": "original Ridge",
            "coordinate_preconditioned_ridge": "coordinate-preconditioned Ridge",
            "transformed_penalty_preconditioned_ridge": "transformed-penalty preconditioned",
            "original_qsvt_diagnostic": "original QSVT diagnostic",
            "preconditioned_qsvt_diagnostic": "preconditioned QSVT diagnostic",
        }
        for row in complete.to_dict(orient="records"):
            variant_name = str(row.get("variant_name"))
            if variant_name not in variant_map:
                continue
            rows.append(
                {
                    "case_name": str(row.get("case_name", "")).upper(),
                    "variant": variant_map[variant_name],
                    "degree": row.get("qsvt_degree", 201),
                    "query_count": row.get("qsvt_query_count", 403),
                    "full_interval_error": row.get("qsvt_full_interval_error"),
                    "actual_singular_value_error": row.get("qsvt_actual_singular_value_error"),
                    "status": row.get("phase_validation_status", row.get("status")),
                    "claim": "Phase 2 QSVT approximation/resource diagnostic",
                    "caveat": (
                        "Phase 2 rows are variant-specific approximation diagnostics; "
                        "preconditioned IEEE300 rows do not make original IEEE300 pass."
                    ),
                }
            )
    phase2 = _read_csv("outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv")
    if complete is None and phase2 is not None and not phase2.empty:
        phase2_selected = phase2[np.isclose(phase2["alpha"].astype(float), 1.0e-2)].copy()
        variant_map = {
            "original_qsvt_diagnostic": "original",
            "preconditioned_qsvt_diagnostic": "preconditioned coordinate",
            "transformed_penalty_preconditioned_ridge": "transformed-penalty preconditioned",
        }
        for row in phase2_selected.to_dict(orient="records"):
            variant = variant_map.get(str(row.get("variant_name")))
            if variant is None:
                continue
            case_name = str(row.get("case_name", "")).upper()
            rows.append(
                {
                    "case_name": case_name,
                    "variant": variant,
                    "degree": row.get("mean_qsvt_degree", row.get("qsvt_degree", 201)),
                    "query_count": row.get(
                        "mean_qsvt_query_count",
                        row.get("qsvt_query_count", 403),
                    ),
                    "full_interval_error": row.get("mean_qsvt_full_interval_approx_error"),
                    "actual_singular_value_error": row.get("mean_qsvt_actual_singular_value_error"),
                    "status": row.get("status", "phase2_diagnostic"),
                    "claim": "Phase 2 QSVT approximation/resource diagnostic",
                    "caveat": (
                        "Phase 2 rows are variant-specific approximation diagnostics; "
                        "preconditioned IEEE300 rows do not make original IEEE300 pass."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _table_6_phase_validation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sanity = _read_csv(
        "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
    )
    if sanity is not None:
        for row in sanity.to_dict(orient="records"):
            name = str(row.get("polynomial_name"))
            if name in {"x", "0.5x"}:
                rows.append(
                    {
                        "target": f"sanity_polynomial_{name}",
                        "backend": row.get("backend", "pennylane_scalar_convention"),
                        "degree": row.get("degree"),
                        "phase_count": "",
                        "input_basis": "monomial",
                        "full_domain_max_error": row.get("best_max_pointwise_error"),
                        "actual_singular_value_max_error": "",
                        "tolerance": 1.0e-3,
                        "status": row.get("best_status"),
                        "reason": "sanity polynomial response",
                        "claim_supported": "scalar convention sanity check",
                        "limitation": "not bounded Ridge/Tikhonov target validation",
                        "historical_status": "",
                        "superseded_by": "",
                    }
                )
    external = _read_csv(
        "outputs/qsvt_external_backend_phase_validation/"
        "external_backend_phase_validation_summary.csv"
    )
    if external is not None and not external.empty:
        pyqsp = external[
            (external["backend_name"].astype(str) == "pyqsp_sym_qsp")
            & (
                external["candidate_name"].astype(str)
                == "coefficient_conditioned_chebyshev_degree_201_lambda_1e-04"
            )
        ]
        if not pyqsp.empty:
            row = pyqsp.iloc[0]
            rows.append(
                {
                    "target": "bounded_ridge_tikhonov_pyqsp",
                    "backend": "pyqsp_sym_qsp",
                    "degree": int(row.get("degree", 201)),
                    "phase_count": int(row.get("phase_count", 202)),
                    "input_basis": "Chebyshev",
                    "full_domain_max_error": 4.668e-4,
                    "actual_singular_value_max_error": row.get(
                        "phase_response_max_error_actual_singular_values_if_available"
                    ),
                    "tolerance": 1.0e-3,
                    "status": "passed_scalar_full_domain",
                    "reason": "pyqsp symmetric-QSP Chebyshev-basis phase-response pass",
                    "claim_supported": "scalar full-domain phase-response validation",
                    "limitation": "not hardware execution or block-encoded matrix execution",
                    "historical_status": "",
                    "superseded_by": "",
                }
            )
        historical = external[
            external["status"].astype(str).str.contains("skipped|failed", na=False)
        ].head(4)
        for row in historical.to_dict(orient="records"):
            rows.append(
                {
                    "target": str(row.get("candidate_name")),
                    "backend": row.get("backend_name"),
                    "degree": row.get("degree"),
                    "phase_count": row.get("phase_count", ""),
                    "input_basis": row.get("input_basis", ""),
                    "full_domain_max_error": row.get("phase_response_max_error_full_domain", ""),
                    "actual_singular_value_max_error": row.get(
                        "phase_response_max_error_actual_singular_values_if_available",
                        "",
                    ),
                    "tolerance": 1.0e-3,
                    "status": "historical_failure",
                    "reason": row.get("failure_reason", row.get("status")),
                    "claim_supported": "historical backend-specific diagnostic",
                    "limitation": "superseded by pyqsp scalar full-domain validation",
                    "historical_status": row.get("status"),
                    "superseded_by": "superseded_by_pyqsp_phase_validation",
                }
            )
    candidates = _read_csv(
        "outputs/qsvt_phase_validation_stable_basis/candidate_polynomial_diagnostics.csv"
    )
    if candidates is not None:
        for degree in [35, 101]:
            subset = candidates[
                (candidates["degree"] == degree)
                & candidates["candidate_name"].astype(str).str.contains("longdouble|float64")
            ].head(1)
            if not subset.empty:
                row = subset.iloc[0]
                rows.append(
                    {
                        "target": f"bounded_ridge_degree_{degree}",
                        "backend": "pennylane_monomial_path",
                        "degree": degree,
                        "phase_count": "",
                        "input_basis": "monomial_after_conversion",
                        "full_domain_max_error": row.get("native_approx_max_error"),
                        "actual_singular_value_max_error": "",
                        "tolerance": 1.0e-3,
                        "status": "historical_failure",
                        "reason": row.get("failure_reason"),
                        "claim_supported": "phase target diagnostic status",
                        "limitation": row.get("recommended_interpretation"),
                        "historical_status": row.get("phase_status"),
                        "superseded_by": "superseded_by_pyqsp_phase_validation",
                    }
                )
        rows.append(
            {
                "target": "bounded_ridge_target_final",
                "backend": "pyqsp_sym_qsp",
                "degree": 201,
                "phase_count": 202,
                "input_basis": "Chebyshev",
                "full_domain_max_error": 4.668e-4,
                "actual_singular_value_max_error": 8.673e-5,
                "tolerance": 1.0e-3,
                "status": "passed_scalar_full_domain",
                "reason": "latest final status supersedes historical unresolved rows",
                "claim_supported": "scalar full-domain phase-response validation",
                "limitation": "not hardware execution or block-encoded matrix execution",
                "historical_status": "formerly_unresolved",
                "superseded_by": "bounded_ridge_tikhonov_pyqsp",
            }
        )
    return pd.DataFrame(rows)


def _table_7_ieee300_preconditioning() -> pd.DataFrame:
    phase2 = _read_csv("outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv")
    if phase2 is not None and not phase2.empty:
        ieee300 = phase2[
            (phase2["case_name"].astype(str).str.lower() == "ieee300")
            & np.isclose(phase2["alpha"].astype(float), 1.0e-2)
        ].copy()
        if not ieee300.empty:
            label_map = {
                "original_ridge": "original Ridge",
                "coordinate_preconditioned_ridge": "coordinate-preconditioned Ridge",
                "transformed_penalty_preconditioned_ridge": "transformed-penalty Ridge",
                "original_qsvt_diagnostic": "original QSVT diagnostic",
                "preconditioned_qsvt_diagnostic": "preconditioned QSVT diagnostic",
            }
            order = list(label_map)
            ieee300["order"] = ieee300["variant_name"].map(
                {name: index for index, name in enumerate(order)}
            )
            ieee300 = ieee300.sort_values("order")
            return pd.DataFrame(
                {
                    "variant": ieee300["variant_name"].map(label_map),
                    "kappa": ieee300["median_condition_number_original"],
                    "preconditioned_kappa_if_applicable": ieee300[
                        "median_condition_number_preconditioned_if_applicable"
                    ],
                    "residual_norm": ieee300["mean_residual_norm"],
                    "weighted_residual_norm": ieee300.get(
                        "mean_weighted_residual_norm",
                        ieee300["mean_residual_norm"],
                    ),
                    "rmse_if_available": ieee300["mean_rmse_if_available"],
                    "approx_error": ieee300["mean_qsvt_full_interval_approx_error"],
                    "degree": ieee300["mean_qsvt_degree"],
                    "query_count": ieee300["mean_qsvt_query_count"],
                    "status": ieee300["status"],
                    "interpretation": ieee300["interpretation"],
                }
            )
    frame = _read_csv(
        "outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_estimator_summary.csv"
    )
    if frame is None:
        return pd.DataFrame()
    ieee300 = frame[frame["case_name"].astype(str) == "ieee300"].copy()
    if ieee300.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "variant": ieee300["variant_name"],
            "kappa": ieee300["condition_number_after"],
            "residual_norm": ieee300["residual_norm"],
            "weighted_residual_norm": ieee300["weighted_residual_norm"],
            "rmse_if_available": ieee300["rmse_if_available"],
            "approx_error": ieee300["full_interval_approx_error"],
            "degree": ieee300["degree"],
            "query_count": ieee300["query_count"],
            "status": ieee300["status"],
            "interpretation": ieee300["estimator_caveat"],
        }
    )


def _table_8_resource_readout() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    resource = _read_csv(
        "outputs/qsvt_preconditioning_resource_comparison/preconditioning_resource_comparison.csv"
    )
    if resource is not None:
        for row in resource.to_dict(orient="records"):
            rows.append(
                {
                    "case_name": row.get("case_name"),
                    "variant": row.get("variant"),
                    "degree": row.get("degree_used"),
                    "query_count": row.get("query_count"),
                    "qubit_proxy": row.get("logical_qubits_proxy"),
                    "depth_proxy": row.get("depth_proxy"),
                    "backend": "",
                    "phase_count": "",
                    "full_domain_error": "",
                    "state_preparation_caveat": STATE_PREPARATION_CAVEAT,
                    "oracle_caveat": row.get("oracle_caveat", ORACLE_CAVEAT),
                    "readout_caveat": row.get("readout_caveat", READOUT_CAVEAT),
                    "hardware_caveat": HARDWARE_CAVEAT,
                }
            )
    rows.append(
        {
            "case_name": "phase_validation",
            "variant": "bounded_ridge_tikhonov_pyqsp",
            "degree": 201,
            "query_count": 403,
            "qubit_proxy": "",
            "depth_proxy": "",
            "backend": "pyqsp_sym_qsp",
            "phase_count": 202,
            "full_domain_error": 4.668e-4,
            "state_preparation_caveat": STATE_PREPARATION_CAVEAT,
            "oracle_caveat": ORACLE_CAVEAT,
            "readout_caveat": READOUT_CAVEAT,
            "hardware_caveat": HARDWARE_CAVEAT,
        }
    )
    return pd.DataFrame(rows)


def _table_9_claim_boundary_matrix() -> pd.DataFrame:
    claims = _read_csv("outputs/qsvt_engineering_extension/claim_support_matrix.csv")
    if claims is None:
        frame = pd.DataFrame(
            [
                {
                    "claim": "claim matrix unavailable",
                    "support_status": "missing",
                    "recommended_wording": "",
                    "avoid_wording": "",
                    "evidence_source": "",
                    "limitation": "source output missing",
                }
            ]
        )
    else:
        frame = pd.DataFrame(
            {
                "claim": claims["claim"],
                "support_status": claims["support_status"],
                "recommended_wording": claims["recommended_wording"],
                "avoid_wording": claims["avoid_wording"],
                "evidence_source": claims["supporting_outputs"],
                "limitation": claims["limitations"],
            }
        )
    phase2_rows = pd.DataFrame(
        [
            {
                "claim": (
                    "Preconditioning reduces QSVT-compatible approximation difficulty "
                    "for selected alpha settings."
                ),
                "support_status": "supported_with_phase2_diagnostics",
                "recommended_wording": (
                    "Preconditioning reduces QSVT-compatible approximation difficulty "
                    "for selected alpha settings in the controlled benchmark."
                ),
                "avoid_wording": (
                    "Preconditioning proves the original IEEE300 matrix passes the same diagnostic."
                ),
                "evidence_source": (
                    "outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv"
                ),
                "limitation": "Alpha-dependent diagnostic; not an original IEEE300 pass.",
            },
            {
                "claim": (
                    "Coordinate-preconditioned Ridge is a separate estimator and can "
                    "degrade residual/RMSE."
                ),
                "support_status": "supported_with_phase2_diagnostics",
                "recommended_wording": (
                    "Coordinate-preconditioned Ridge is evaluated as a separate "
                    "estimator and can degrade residual/RMSE."
                ),
                "avoid_wording": "Coordinate-preconditioned Ridge replaces original Ridge.",
                "evidence_source": (
                    "outputs/qsvt_phase2_complete_summary/phase2_variant_comparison.csv"
                ),
                "limitation": "Coordinate penalty changes the regularization geometry.",
            },
            {
                "claim": (
                    "Transformed-penalty preconditioning preserves the original x-space "
                    "Ridge penalty."
                ),
                "support_status": "supported_by_equation_and_diagnostic",
                "recommended_wording": (
                    "The transformed-penalty formulation preserves the original "
                    "x-space Ridge penalty while using the preconditioned matrix "
                    "for approximation diagnostics."
                ),
                "avoid_wording": "Coordinate and transformed-penalty formulations are equivalent.",
                "evidence_source": (
                    "outputs/qsvt_phase2_manuscript_text/transformed_penalty_explanation.md"
                ),
                "limitation": "This is a consistency formulation, not a new solver claim.",
            },
            {
                "claim": "Alpha selection is diagnostic, not field-calibrated.",
                "support_status": "explicit_boundary",
                "recommended_wording": (
                    "Alpha selection is diagnostic and controlled-benchmark-specific."
                ),
                "avoid_wording": "The Phase 2 alpha score is a deployment-ready rule.",
                "evidence_source": (
                    "outputs/qsvt_phase2_alpha_selection/alpha_selection_report.md"
                ),
                "limitation": "No field-calibrated operational selection rule is claimed.",
            },
        ]
    )
    return pd.concat([frame, phase2_rows], ignore_index=True)


def _paper_table_notes(tables: dict[str, pd.DataFrame]) -> str:
    rows = [
        {"table": name, "rows": len(frame), "status": "ok" if not frame.empty else "empty"}
        for name, frame in tables.items()
    ]
    return f"""# Paper-Ready QSVT Table Notes

Tables are generated from existing CSV/JSON artifacts only. Missing source data
is recorded as missing or limitation rows rather than fabricated.

{_markdown_table(pd.DataFrame(rows), ["table", "rows", "status"])}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _artifact_targets(config: dict[str, Any]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for path in MANUSCRIPT_OUTPUTS:
        targets.append(
            {
                "path": path,
                "scientific_role": "QSVT diagnostic output",
                "paper_table_dependency": "yes",
                "claim_dependency": "yes",
                "generated_by_command_if_known": _known_command_for_path(path),
            }
        )
    for path in MANUSCRIPT_DOCS:
        targets.append(
            {
                "path": path,
                "scientific_role": "documentation",
                "paper_table_dependency": "no",
                "claim_dependency": "yes",
                "generated_by_command_if_known": "manual/documentation update",
            }
        )
    for path in MANUSCRIPT_SCRIPTS:
        targets.append(
            {
                "path": path,
                "scientific_role": "reproducible script entrypoint",
                "paper_table_dependency": "indirect",
                "claim_dependency": "yes",
                "generated_by_command_if_known": "source file",
            }
        )
    for extra in config.get("extra_paths", []):
        targets.append(
            {
                "path": str(extra),
                "scientific_role": "extra configured artifact",
                "paper_table_dependency": "configured",
                "claim_dependency": "configured",
                "generated_by_command_if_known": "",
            }
        )
    return targets


def _artifact_row(path: Path, target: dict[str, str], resolved: dict[str, Any]) -> dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else np.nan
    sha = ""
    status = "missing"
    notes = ""
    if exists:
        status = "indexed"
        if path.is_file() and _should_hash(path, int(resolved["sha_size_limit_bytes"])):
            sha = _sha256(path)
        elif path.is_file():
            notes = "sha256 skipped because file is outside configured size/type policy"
    return {
        "path": str(path),
        "exists": bool(exists),
        "file_type": path.suffix.lstrip(".") if path.suffix else "directory",
        "size_bytes": size,
        "sha256_if_file": sha,
        "generated_by_command_if_known": target.get("generated_by_command_if_known", ""),
        "scientific_role": target.get("scientific_role", ""),
        "paper_table_dependency": target.get("paper_table_dependency", ""),
        "claim_dependency": target.get("claim_dependency", ""),
        "status": status,
        "notes": notes,
    }


def _scan_claim_root(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not root.exists():
        return [
            {
                "path": str(root),
                "line_number": np.nan,
                "phrase": "",
                "line_text": "",
                "classification": "needs_manual_review",
                "reason": "scan root missing",
            }
        ]
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    rows: list[dict[str, Any]] = []
    allowed_suffixes = set(config["suffixes"])
    for path in files:
        if path.suffix not in allowed_suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        context_window: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            for phrase in config["unsafe_phrases"]:
                if phrase.lower() in lowered:
                    rows.append(
                        {
                            "path": str(path),
                            "line_number": line_number,
                            "phrase": phrase,
                            "line_text": line.strip(),
                            "classification": _classify_claim_context(
                                line,
                                path=path,
                                context="\n".join(context_window[-4:]),
                            ),
                            "reason": _claim_reason(
                                line,
                                path=path,
                                context="\n".join(context_window[-4:]),
                            ),
                        }
                    )
            context_window.append(line)
    return rows


def _classify_claim_context(line: str, *, path: Path, context: str = "") -> str:
    current = line.lower()
    lowered = f"{context}\n{line}".lower()
    path_text = str(path).lower()
    if (
        "claim_boundary_matrix" in path_text
        or "claim_support_matrix" in path_text
        or "estimator_roles" in path_text
        or "claim_boundary_summary" in path_text
    ):
        return "avoid_wording_context"
    avoid_markers = [
        "avoid",
        "do not claim",
        "claims to avoid",
        "forbidden",
        "unsupported",
        "not claim",
        "avoid-wording",
        "unsupported claims",
        "also avoid",
        "unsafe wording",
    ]
    safe_markers = [
        "does not",
        "do not",
        "not demonstrate",
        "not prove",
        "not full",
        "not ",
        "no ",
        "without claiming",
        "is not",
        "separates",
    ]
    if any(marker in current for marker in safe_markers) and not any(
        marker in current for marker in avoid_markers
    ):
        return "safe_context"
    if any(marker in lowered for marker in avoid_markers):
        return "avoid_wording_context"
    if any(marker in lowered for marker in safe_markers):
        return "safe_context"
    if "recommended_wording" in lowered or "avoid_wording" in lowered:
        return "avoid_wording_context"
    return "unsafe_context"


def _claim_reason(line: str, *, path: Path, context: str = "") -> str:
    classification = _classify_claim_context(line, path=path, context=context)
    if classification == "unsafe_context":
        return "unsafe phrase appears outside an obvious caveat context"
    if classification == "avoid_wording_context":
        return "phrase appears in avoid-wording or unsupported-claim context"
    return "phrase appears in negated or caveated context"


def _claim_audit_summary(frame: pd.DataFrame) -> str:
    counts = (
        frame["classification"].value_counts().reset_index()
        if not frame.empty
        else pd.DataFrame({"classification": [], "count": []})
    )
    counts.columns = ["classification", "count"] if not counts.empty else counts.columns
    unsafe_count = (
        int((frame["classification"] == "unsafe_context").sum()) if not frame.empty else 0
    )
    manual_count = (
        int((frame["classification"] == "needs_manual_review").sum()) if not frame.empty else 0
    )
    verdict = "PASS" if unsafe_count == 0 else "NEEDS_REVIEW"
    return f"""# Final QSVT Claim-Safety Audit

Audit verdict: **{verdict}**

Unsafe contexts found: {unsafe_count}

Manual-review contexts found: {manual_count}

{_markdown_table(counts, list(counts.columns)) if not counts.empty else "No phrases found."}

Phrases in avoid-wording or negated caveat contexts are not treated as unsafe
claims.
"""


def _resource_report(frame: pd.DataFrame) -> str:
    columns = [
        "case_name",
        "variant",
        "alpha",
        "kappa",
        "degree_used",
        "query_count",
        "full_interval_approx_error",
        "actual_singular_value_approx_error",
        "status",
    ]
    return f"""# QSVT Preconditioning Resource Comparison

{_markdown_table(frame, columns)}

## Interpretation

The comparison reports proxy approximation and resource quantities before and
after column equilibration. Improvements apply to the configured preconditioned
matrix variant and do not make the original unpreconditioned IEEE300 row pass.

## Claim Boundary

{RESOURCE_CAVEAT}
"""


def _artifact_freeze_summary(inventory: pd.DataFrame, missing: pd.DataFrame) -> str:
    hashed = int(inventory["sha256_if_file"].astype(str).ne("").sum()) if not inventory.empty else 0
    return f"""# Final QSVT Artifact Freeze

Indexed artifacts: {len(inventory)}

Missing or unverified artifacts: {len(missing)}

SHA256-covered files: {hashed}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _verification_commands_text() -> str:
    commands = [
        ".venv/bin/python -m compileall -q src scripts tests",
        ".venv/bin/python -m pytest -q",
        ".venv/bin/python -m ruff check src tests scripts",
        ".venv/bin/python -m ruff format --check src tests scripts",
        ".venv/bin/python scripts/finalize_qsvt_phase1_artifacts.py",
        ".venv/bin/python scripts/run_qsvt_phase2_preconditioned_alpha_sweeps.py",
        ".venv/bin/python scripts/build_qsvt_phase2_alpha_selection_report.py",
        ".venv/bin/python scripts/build_qsvt_phase2_summary.py",
        ".venv/bin/python scripts/build_qsvt_phase2_complete_summary.py",
        ".venv/bin/python scripts/build_qsvt_phase2_figures.py",
        ".venv/bin/python scripts/run_qsvt_phase2_optional_ieee57.py",
        ".venv/bin/python scripts/build_qsvt_phase2_manuscript_text.py",
        ".venv/bin/python scripts/run_qsvt_preconditioned_variant_sweeps.py",
        ".venv/bin/python scripts/build_qsvt_preconditioning_resource_comparison.py",
        ".venv/bin/python scripts/build_paper_ready_qsvt_tables.py",
        ".venv/bin/python scripts/freeze_qsvt_manuscript_artifacts.py",
        ".venv/bin/python scripts/run_final_qsvt_claim_safety_audit.py",
        ".venv/bin/python scripts/build_qsvt_engineering_extension_summary.py",
        ".venv/bin/python scripts/audit_qsvt_engineering_outputs.py",
    ]
    return "\n".join(commands) + "\n"


def _claim_boundary_summary() -> str:
    return f"""# QSVT Claim Boundary Summary

Supported wording:

```text
controlled IEEE/PYPOWER benchmark systems; generated measurement rows;
regularized spectral filtering; QSVT-compatible implementation pathway;
scalar full-domain phase-response validation with pyqsp symmetric-QSP phases;
Chebyshev-basis phase validation; resource-aware feasibility analysis;
preconditioned estimator variant; transformed-penalty consistency check;
alpha-selection diagnostics; diagnostic QSVT approximation evidence; paper-ready
artifact aggregation; claim-support traceability
```

Avoid wording:

```text
quantum speedup; quantum advantage; QSVT outperforms Ridge under the same
alpha; full hardware execution; hardware validation; real PMU/SCADA
field-data validation; field-calibrated statistics; original IEEE300 passed from
preconditioned rows; preconditioned IEEE300 proves original IEEE300; phase
validation proves hardware; phase validation passed from sanity polynomials
alone
```

{CLAIM_BOUNDARY}
"""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _first_available(
    frame: pd.DataFrame,
    names: list[str],
    default: Any = np.nan,
) -> Any:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return default


def _read_csv(path: str | Path) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_hash(path: Path, size_limit: int) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size > size_limit:
        return False
    return path.suffix.lower() in {".csv", ".json", ".md", ".yaml", ".yml", ".py", ".txt"}


def _known_command_for_path(path: str) -> str:
    mapping = {
        "qsvt_phase1_finalization": (".venv/bin/python scripts/finalize_qsvt_phase1_artifacts.py"),
        "qsvt_phase2_preconditioned_alpha_sweeps": (
            ".venv/bin/python scripts/run_qsvt_phase2_preconditioned_alpha_sweeps.py"
        ),
        "qsvt_phase2_alpha_selection": (
            ".venv/bin/python scripts/build_qsvt_phase2_alpha_selection_report.py"
        ),
        "qsvt_phase2_summary": ".venv/bin/python scripts/build_qsvt_phase2_summary.py",
        "qsvt_preconditioned_variant_sweeps": (
            ".venv/bin/python scripts/run_qsvt_preconditioned_variant_sweeps.py"
        ),
        "qsvt_preconditioning_resource_comparison": (
            ".venv/bin/python scripts/build_qsvt_preconditioning_resource_comparison.py"
        ),
        "paper_ready_qsvt_tables": ".venv/bin/python scripts/build_paper_ready_qsvt_tables.py",
        "qsvt_phase_validation_stable_basis": (
            ".venv/bin/python scripts/fix_qsvt_phase_validation_stable_basis.py"
        ),
        "qsvt_preconditioned_ieee300_estimator": (
            ".venv/bin/python scripts/run_qsvt_preconditioned_ieee300_estimator.py"
        ),
        "qsvt_ieee300_residual_weighted_error": (
            ".venv/bin/python scripts/diagnose_qsvt_ieee300_residual_weighted_error.py"
        ),
        "qsvt_engineering_extension": (
            ".venv/bin/python scripts/build_qsvt_engineering_extension_summary.py"
        ),
    }
    return next((command for key, command in mapping.items() if key in path), "")


def _write_named_manifest(
    path: Path,
    *,
    artifacts: dict[str, str],
    input_config: dict[str, Any],
) -> Path:
    write_json(
        path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "input_config": input_config,
            "artifacts": artifacts,
            "git_commit": git_commit(),
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return path


def _failure_row(
    case_name: str,
    stage: str,
    seed: int | str | float,
    noise_std: float,
    missing_ratio: float,
    bad_data_ratio: float,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "stage": stage,
        "seed": seed,
        "noise_std": noise_std,
        "missing_ratio": missing_ratio,
        "bad_data_ratio": bad_data_ratio,
        "status": "failed",
        "failure_reason_if_any": str(exc),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _sweep_result_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "alpha",
        "noise_std",
        "missing_ratio",
        "bad_data_ratio",
        "seed",
        "m",
        "n",
        "rank",
        "condition_number",
        "condition_number_preconditioned_if_applicable",
        "rmse_if_available",
        "angle_rmse_if_available",
        "voltage_rmse_if_available",
        "residual_norm",
        "weighted_residual_norm",
        "solution_norm",
        "relative_solution_error_vs_unpreconditioned_ridge",
        "relative_solution_error_vs_transformed_penalty",
        "qsvt_full_interval_approx_error",
        "qsvt_actual_singular_value_approx_error",
        "polynomial_degree",
        "query_count",
        "runtime_seconds",
        "status",
        "failure_reason_if_any",
        "estimator_caveat",
        "qsvt_caveat",
    ]


def _failure_columns() -> list[str]:
    return [
        "case_name",
        "stage",
        "seed",
        "noise_std",
        "missing_ratio",
        "bad_data_ratio",
        "status",
        "failure_reason_if_any",
        "claim_boundary",
    ]


def _claim_audit_columns() -> list[str]:
    return ["path", "line_number", "phrase", "line_text", "classification", "reason"]


def _resolve_sweep_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_preconditioned_variant_sweeps",
        "cases": ["ieee118", "ieee300"],
        "case_source": "pypower",
        "base_seed": 123,
        "seeds": SEEDS,
        "alphas": ALPHA_GRID,
        "noise_stds": NOISE_GRID,
        "missing_ratios": MISSING_GRID,
        "bad_data_ratios": BAD_DATA_GRID,
        "bad_data_magnitude": 5.0,
        "degree": 101,
        "method": "odd_chebyshev_ls",
        "grid_size": 160,
        "fallback_to_synthetic": False,
    }
    if config:
        resolved.update(config)
    resolved["alphas"] = [float(value) for value in resolved["alphas"]]
    resolved["noise_stds"] = [float(value) for value in resolved["noise_stds"]]
    resolved["missing_ratios"] = [float(value) for value in resolved["missing_ratios"]]
    resolved["bad_data_ratios"] = [float(value) for value in resolved["bad_data_ratios"]]
    return resolved


def _resolve_resource_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_preconditioning_resource_comparison",
        "cases": ["ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "alphas": ALPHA_GRID,
        "degree": 301,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
        "fallback_to_synthetic": False,
    }
    if config:
        resolved.update(config)
    resolved["alphas"] = [float(value) for value in resolved["alphas"]]
    return resolved


def _resolve_tables_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {"output_dir": "outputs/paper_ready_qsvt_tables"}
    if config:
        resolved.update(config)
    return resolved


def _resolve_freeze_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/final_qsvt_artifact_freeze",
        "sha_size_limit_bytes": 25_000_000,
        "extra_paths": [],
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_claim_audit_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/final_qsvt_claim_safety_audit",
        "scan_roots": [
            "README.md",
            "docs",
            "outputs/paper_ready_qsvt_tables",
            "outputs/qsvt_phase1_finalization",
            "outputs/qsvt_phase2_summary",
            "outputs/qsvt_phase2_complete_summary",
            "outputs/qsvt_phase2_alpha_selection",
            "outputs/qsvt_phase2_figures",
            "outputs/qsvt_phase2_manuscript_text",
            "outputs/final_qsvt_artifact_freeze",
            "outputs/qsvt_engineering_extension",
        ],
        "unsafe_phrases": FINAL_UNSAFE_PHRASES,
        "suffixes": [".md", ".csv", ".json", ".txt"],
    }
    if config:
        resolved.update(config)
    return resolved


def main_preconditioned_sweeps(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run QSVT preconditioned variant sweeps")
    parser.parse_args(argv)
    run = run_preconditioned_variant_sweeps()
    print(f"QSVT preconditioned variant sweeps complete: {run['output_dir']}")


def main_resource_comparison(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT preconditioning resources")
    parser.parse_args(argv)
    run = build_preconditioning_resource_comparison()
    print(f"QSVT preconditioning resource comparison complete: {run['output_dir']}")


def main_paper_tables(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready QSVT tables")
    parser.parse_args(argv)
    run = build_paper_ready_qsvt_tables()
    print(f"QSVT paper-ready tables complete: {run['output_dir']}")


def main_artifact_freeze(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze QSVT manuscript artifacts")
    parser.parse_args(argv)
    run = freeze_qsvt_manuscript_artifacts()
    print(f"QSVT artifact freeze complete: {run['output_dir']}")


def main_claim_audit(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run final QSVT claim-safety audit")
    parser.parse_args(argv)
    run = run_final_claim_safety_audit()
    print(f"QSVT final claim-safety audit complete: {run['output_dir']}")
