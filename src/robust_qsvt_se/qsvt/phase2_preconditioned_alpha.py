from __future__ import annotations

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
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.failure_fix import (
    ESTIMATOR_CAVEAT,
    QSVT_CAVEAT,
    _column_equilibration_scales,
    _context_from_matrix,
    _kappa,
    _relative_error,
)
from robust_qsvt_se.qsvt.phase1_finalization import PYQSP_BACKEND, PYQSP_TOLERANCE
from robust_qsvt_se.qsvt.polynomial_approximation import evaluate_polynomial_approximation
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.seed import make_rng

PHASE2_ALPHAS = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
PHASE2_NOISE_STDS = [0.0, 0.01, 0.03]
PHASE2_MISSING_RATIOS = [0.0, 0.05, 0.10]
PHASE2_BAD_DATA_RATIOS = [0.0, 0.02]
PHASE2_SEEDS = [10, 20, 30]
PHASE2_VARIANTS = [
    "original_ridge",
    "coordinate_preconditioned_ridge",
    "transformed_penalty_preconditioned_ridge",
    "original_qsvt_diagnostic",
    "preconditioned_qsvt_diagnostic",
]
PHASE_PASS_ALPHA = 1.0e-2
PHASE_PASS_DEGREE = 201
PHASE_PASS_QUERY_COUNT = 202
PHASE_FULL_DOMAIN_ERROR = 4.668e-4
PHASE_ACTUAL_SV_ERROR = 8.673e-5
ALPHA_SELECTION_WEIGHTS = {"residual_or_rmse": 0.5, "qsvt_error": 0.3, "query_count": 0.2}


def run_phase2_preconditioned_alpha_sweeps(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_sweep_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    context_cache: dict[tuple[str, str], Any] = {}
    svd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    approximation_cache: dict[tuple[str, float, str], dict[str, float]] = {}

    for case_name in resolved["cases"]:
        try:
            base_system, matrix_source = build_engineering_system(
                {
                    "case_name": str(case_name),
                    "case_source": str(resolved["case_source"]),
                    "seed": int(resolved["base_seed"]),
                    "fallback_to_synthetic": bool(resolved["fallback_to_synthetic"]),
                    "matrix_source": "synthetic" if str(case_name) == "synthetic" else "",
                }
            )
        except Exception as exc:
            failures.append(_failure_row(str(case_name), "build_system", "", 0.0, 0.0, 0.0, exc))
            continue
        for seed in resolved["seeds"]:
            for missing_ratio in resolved["missing_ratios"]:
                try:
                    missing_system = _apply_missing(
                        base_system,
                        missing_ratio=float(missing_ratio),
                        seed=int(seed),
                    )
                except Exception as exc:
                    failures.append(
                        _failure_row(
                            str(case_name),
                            "missing_measurements",
                            seed,
                            0.0,
                            missing_ratio,
                            0.0,
                            exc,
                        )
                    )
                    continue
                for noise_std in resolved["noise_stds"]:
                    for bad_data_ratio in resolved["bad_data_ratios"]:
                        started = time.perf_counter()
                        try:
                            system = _apply_rhs_perturbations(
                                missing_system,
                                noise_std=float(noise_std),
                                bad_data_ratio=float(bad_data_ratio),
                                seed=int(seed),
                                bad_data_magnitude=float(resolved["bad_data_magnitude"]),
                            )
                            results.extend(
                                _scenario_rows(
                                    system=system,
                                    matrix_source=matrix_source,
                                    case_name=str(case_name),
                                    seed=int(seed),
                                    noise_std=float(noise_std),
                                    missing_ratio=float(missing_ratio),
                                    bad_data_ratio=float(bad_data_ratio),
                                    alphas=resolved["alphas"],
                                    qsvt_degree=int(resolved["qsvt_degree"]),
                                    method=str(resolved["method"]),
                                    grid_size=int(resolved["grid_size"]),
                                    context_cache=context_cache,
                                    svd_cache=svd_cache,
                                    approximation_cache=approximation_cache,
                                    started=started,
                                )
                            )
                        except Exception as exc:
                            failures.append(
                                _failure_row(
                                    str(case_name),
                                    "scenario",
                                    seed,
                                    noise_std,
                                    missing_ratio,
                                    bad_data_ratio,
                                    exc,
                                )
                            )

    result_frame = pd.DataFrame(results, columns=_result_columns())
    failure_frame = pd.DataFrame(failures, columns=_failure_columns())
    summary_frame = _sweep_summary(result_frame, failure_frame)
    artifacts = _write_sweep_outputs(
        output_dir,
        resolved,
        result_frame,
        summary_frame,
        failure_frame,
    )
    return {
        "output_dir": output_dir,
        "results": result_frame,
        "summary": summary_frame,
        "failures": failure_frame,
        "artifacts": artifacts,
    }


def build_phase2_alpha_selection_report(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_alpha_selection_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    results_path = Path(resolved["sweep_results_csv"])
    if not results_path.is_file():
        run_phase2_preconditioned_alpha_sweeps(
            {
                "output_dir": str(results_path.parent),
                "cases": resolved["cases"],
            }
        )
    results = pd.read_csv(results_path)
    summary, trace = select_alpha_diagnostics(results)

    summary_csv = output_dir / "alpha_selection_summary.csv"
    summary_json = output_dir / "alpha_selection_summary.json"
    trace_csv = output_dir / "alpha_selection_trace.csv"
    report_md = output_dir / "alpha_selection_report.md"
    definitions_md = output_dir / "alpha_selection_metric_definitions.md"
    manifest_path = output_dir / "manifest.json"

    summary.to_csv(summary_csv, index=False)
    trace.to_csv(trace_csv, index=False)
    write_json(summary_json, {"rows": summary.to_dict(orient="records")})
    report_md.write_text(_alpha_selection_report(summary, trace), encoding="utf-8")
    definitions_md.write_text(_alpha_selection_metric_definitions(), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {
                "alpha_selection_summary_csv": str(summary_csv),
                "alpha_selection_summary_json": str(summary_json),
                "alpha_selection_trace_csv": str(trace_csv),
                "alpha_selection_report_md": str(report_md),
                "alpha_selection_metric_definitions_md": str(definitions_md),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "trace": trace,
        "artifacts": {"manifest": manifest_path},
    }


def build_phase2_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_summary_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sweep_path = Path(resolved["sweep_summary_csv"])
    alpha_path = Path(resolved["alpha_selection_summary_csv"])
    if not sweep_path.is_file():
        run_phase2_preconditioned_alpha_sweeps({"output_dir": str(sweep_path.parent)})
    if not alpha_path.is_file():
        build_phase2_alpha_selection_report({"output_dir": str(alpha_path.parent)})

    sweep = pd.read_csv(sweep_path)
    alpha = pd.read_csv(alpha_path)
    summary = _phase2_summary_frame(sweep, alpha)

    csv_path = output_dir / "phase2_summary.csv"
    json_path = output_dir / "phase2_summary.json"
    md_path = output_dir / "phase2_summary.md"
    manifest_path = output_dir / "manifest.json"

    summary.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": summary.to_dict(orient="records")})
    md_path.write_text(_phase2_summary_markdown(summary, alpha), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {
                "phase2_summary_md": str(md_path),
                "phase2_summary_csv": str(csv_path),
                "phase2_summary_json": str(json_path),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {"manifest": manifest_path},
    }


def select_alpha_diagnostics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        empty_summary = pd.DataFrame(columns=_alpha_summary_columns())
        empty_trace = pd.DataFrame(columns=_alpha_trace_columns())
        return empty_summary, empty_trace
    grouped = (
        results.groupby(["case_name", "variant_name", "alpha"], dropna=False)
        .agg(
            mean_residual_norm=("residual_norm", "mean"),
            mean_rmse=("rmse_if_available", "mean"),
            mean_qsvt_error=("qsvt_full_interval_approx_error", "mean"),
            mean_actual_sv_error=("qsvt_actual_singular_value_error", "mean"),
            mean_query_count=("qsvt_query_count", "mean"),
            mean_degree=("qsvt_degree", "mean"),
            scenario_count=("status", "size"),
        )
        .reset_index()
    )
    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (case_name, variant_name), group in grouped.groupby(["case_name", "variant_name"]):
        scored = _score_alpha_group(group)
        trace_rows.extend(scored.to_dict(orient="records"))
        summary_rows.extend(_selection_rows(case_name, variant_name, scored))
    return (
        pd.DataFrame(summary_rows, columns=_alpha_summary_columns()),
        pd.DataFrame(trace_rows, columns=_alpha_trace_columns()),
    )


def _scenario_rows(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case_name: str,
    seed: int,
    noise_std: float,
    missing_ratio: float,
    bad_data_ratio: float,
    alphas: list[float],
    qsvt_degree: int,
    method: str,
    grid_size: int,
    context_cache: dict[tuple[str, str], Any],
    svd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    approximation_cache: dict[tuple[str, float, str], dict[str, float]],
    started: float,
) -> list[dict[str, Any]]:
    A = np.asarray(system.H_tilde, dtype=np.float64)
    b = np.asarray(system.r_tilde, dtype=np.float64)
    scales = _column_equilibration_scales(A)
    A_p = A * scales[None, :]
    context_original = _cached_context(context_cache, A, case_name, matrix_source)
    context_pre = _cached_context(
        context_cache,
        A_p,
        case_name,
        f"{matrix_source}_column_equilibrated",
    )
    kappa_original = _kappa(context_original.singular_values)
    kappa_pre = _kappa(context_pre.singular_values)
    rank_original = int(np.linalg.matrix_rank(A))
    rank_pre = int(np.linalg.matrix_rank(A_p))
    rows = []
    for alpha in alphas:
        x_original = _ridge_svd_cached(A, b, alpha=float(alpha), cache=svd_cache)
        y_coordinate = _ridge_svd_cached(A_p, b, alpha=float(alpha), cache=svd_cache)
        x_coordinate = scales * y_coordinate
        x_transformed = _transformed_penalty_solution(A_p, b, scales, alpha=float(alpha))
        approx_original = _cached_approximation(
            approximation_cache,
            context_original,
            A,
            alpha=float(alpha),
            degree=qsvt_degree,
            method=method,
            grid_size=grid_size,
        )
        approx_pre = _cached_approximation(
            approximation_cache,
            context_pre,
            A_p,
            alpha=float(alpha),
            degree=qsvt_degree,
            method=method,
            grid_size=grid_size,
        )
        specs = [
            ("original_ridge", x_original, rank_original, kappa_original, np.nan, approx_original),
            (
                "coordinate_preconditioned_ridge",
                x_coordinate,
                rank_pre,
                kappa_original,
                kappa_pre,
                approx_pre,
            ),
            (
                "transformed_penalty_preconditioned_ridge",
                x_transformed,
                rank_pre,
                kappa_original,
                kappa_pre,
                approx_pre,
            ),
            (
                "original_qsvt_diagnostic",
                x_original,
                rank_original,
                kappa_original,
                np.nan,
                approx_original,
            ),
            (
                "preconditioned_qsvt_diagnostic",
                x_coordinate,
                rank_pre,
                kappa_original,
                kappa_pre,
                approx_pre,
            ),
        ]
        for variant, x_hat, rank, kappa_before, kappa_after, approximation in specs:
            rows.append(
                _result_row(
                    system=system,
                    case_name=case_name,
                    variant_name=variant,
                    alpha=float(alpha),
                    noise_std=noise_std,
                    missing_ratio=missing_ratio,
                    bad_data_ratio=bad_data_ratio,
                    seed=seed,
                    rank=rank,
                    condition_number_original=kappa_before,
                    condition_number_preconditioned=kappa_after,
                    x_hat=x_hat,
                    x_original=x_original,
                    x_transformed=x_transformed,
                    approximation=approximation,
                    runtime_seconds=time.perf_counter() - started,
                )
            )
    return rows


def _result_row(
    *,
    system: WeightedSystem,
    case_name: str,
    variant_name: str,
    alpha: float,
    noise_std: float,
    missing_ratio: float,
    bad_data_ratio: float,
    seed: int,
    rank: int,
    condition_number_original: float,
    condition_number_preconditioned: float,
    x_hat: np.ndarray,
    x_original: np.ndarray,
    x_transformed: np.ndarray,
    approximation: dict[str, float],
    runtime_seconds: float,
) -> dict[str, Any]:
    rmse = system.rmse(x_hat)
    phase_available = bool(abs(alpha - PHASE_PASS_ALPHA) <= 1.0e-15)
    status = _variant_status(
        variant_name,
        system.residual_norm(x_hat),
        system.residual_norm(x_original),
    )
    return {
        "case_name": case_name,
        "variant_name": variant_name,
        "alpha": alpha,
        "noise_std": noise_std,
        "missing_ratio": missing_ratio,
        "bad_data_ratio": bad_data_ratio,
        "seed": seed,
        "m": int(system.n_measurements),
        "n": int(system.n_states),
        "rank": int(rank),
        "condition_number_original": float(condition_number_original),
        "condition_number_preconditioned_if_applicable": float(condition_number_preconditioned),
        "rmse_if_available": "not_available" if rmse is None else float(rmse),
        "angle_rmse_if_available": "not_available",
        "voltage_rmse_if_available": "not_available",
        "residual_norm": float(system.residual_norm(x_hat)),
        "weighted_residual_norm": float(system.weighted_residual_norm(x_hat)),
        "solution_norm": float(np.linalg.norm(x_hat)),
        "relative_solution_error_vs_original_ridge": _relative_error(x_original, x_hat),
        "relative_solution_error_vs_transformed_penalty": _relative_error(x_transformed, x_hat),
        "qsvt_full_interval_approx_error": approximation["full_interval_error"],
        "qsvt_actual_singular_value_error": approximation["actual_singular_error"],
        "qsvt_degree": int(approximation["degree"]),
        "qsvt_query_count": int(approximation["query_count"]),
        "pyqsp_phase_available_for_this_target": phase_available,
        "phase_validation_status": (
            "passed_scalar_full_domain" if phase_available else "not_validated_for_this_alpha"
        ),
        "runtime_seconds": runtime_seconds,
        "status": status,
        "failure_reason_if_any": "",
        "estimator_caveat": _phase2_estimator_caveat(variant_name),
        "qsvt_caveat": _phase2_qsvt_caveat(phase_available),
    }


def _variant_status(variant_name: str, residual: float, original_residual: float) -> str:
    if not np.isfinite(residual):
        return "failed_nonfinite_metric"
    if variant_name == "coordinate_preconditioned_ridge":
        ratio = residual / max(original_residual, np.finfo(float).eps)
        return "residual_degraded" if ratio > 2.0 else "ok"
    if variant_name == "transformed_penalty_preconditioned_ridge":
        return "consistency_check"
    if "qsvt" in variant_name:
        return "diagnostic_only"
    return "ok"


def _transformed_penalty_solution(
    A_p: np.ndarray,
    b: np.ndarray,
    scales: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    penalty = scales**2
    lhs = A_p.T @ A_p + alpha * np.diag(penalty)
    rhs = A_p.T @ b
    try:
        y = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return scales * y


def _cached_context(
    cache: dict[tuple[str, str], Any],
    matrix: np.ndarray,
    case_name: str,
    matrix_source: str,
) -> Any:
    key = (_matrix_fingerprint(matrix), matrix_source)
    if key not in cache:
        cache[key] = _context_from_matrix(
            matrix=matrix,
            case_name=case_name,
            matrix_source=matrix_source,
            source_note="Phase 2 preconditioned alpha diagnostics",
        )
    return cache[key]


def _ridge_svd_cached(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    alpha: float,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    key = _matrix_fingerprint(matrix)
    if key not in cache:
        cache[key] = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    U, singular_values, Vt = cache[key]
    gains = singular_values / (singular_values**2 + alpha)
    return Vt.T @ (gains * (U.T @ rhs))


def _cached_approximation(
    cache: dict[tuple[str, float, str], dict[str, float]],
    context: Any,
    matrix: np.ndarray,
    *,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
) -> dict[str, float]:
    key = (_matrix_fingerprint(matrix), float(alpha), context.matrix_source)
    if key not in cache:
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(alpha),
            degree=int(degree),
            method=method,
            grid_size=int(grid_size),
        )
        kinds = np.asarray(result.evaluation_kind, dtype=object)
        grid_errors = result.pointwise_errors[kinds == "grid"]
        singular_errors = result.pointwise_errors[kinds == "actual_singular_value"]
        cache[key] = {
            "full_interval_error": float(np.max(grid_errors)),
            "actual_singular_error": float(np.max(singular_errors)),
            "degree": float(result.degree),
            "query_count": float(2 * result.degree + 1),
        }
    return cache[key]


def _apply_missing(system: WeightedSystem, *, missing_ratio: float, seed: int) -> WeightedSystem:
    if missing_ratio == 0.0:
        return WeightedSystem(
            system.H_tilde,
            system.r_tilde,
            system.x_true,
            {**system.metadata, "missing_ratio": 0.0},
        )
    return remove_random_rows(system, missing_ratio=missing_ratio, rng=make_rng(seed))


def _apply_rhs_perturbations(
    system: WeightedSystem,
    *,
    noise_std: float,
    bad_data_ratio: float,
    seed: int,
    bad_data_magnitude: float,
) -> WeightedSystem:
    rng = make_rng(seed)
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
    return WeightedSystem(
        updated.H_tilde,
        updated.r_tilde,
        updated.x_true,
        {
            **updated.metadata,
            "noise_std": noise_std,
            "bad_data_ratio": bad_data_ratio,
        },
    )


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
                    "mean_qsvt_actual_singular_value_error": np.nan,
                    "mean_qsvt_degree": np.nan,
                    "mean_qsvt_query_count": np.nan,
                    "median_condition_number_original": np.nan,
                    "median_condition_number_preconditioned_if_applicable": np.nan,
                    "failure_count": len(failures),
                    "status": "empty",
                    "interpretation": "No successful Phase 2 rows were generated.",
                }
            ]
        )
    numeric = results.copy()
    numeric["rmse_numeric"] = pd.to_numeric(numeric["rmse_if_available"], errors="coerce")
    grouped = (
        numeric.groupby(["case_name", "variant_name", "alpha"], dropna=False)
        .agg(
            scenario_count=("status", "size"),
            mean_residual_norm=("residual_norm", "mean"),
            mean_weighted_residual_norm=("weighted_residual_norm", "mean"),
            mean_rmse_if_available=("rmse_numeric", "mean"),
            mean_qsvt_full_interval_approx_error=("qsvt_full_interval_approx_error", "mean"),
            mean_qsvt_actual_singular_value_error=("qsvt_actual_singular_value_error", "mean"),
            mean_qsvt_degree=("qsvt_degree", "mean"),
            mean_qsvt_query_count=("qsvt_query_count", "mean"),
            median_condition_number_original=("condition_number_original", "median"),
            median_condition_number_preconditioned_if_applicable=(
                "condition_number_preconditioned_if_applicable",
                "median",
            ),
        )
        .reset_index()
    )
    grouped["failure_count"] = len(failures)
    grouped["status"] = "ok"
    grouped["interpretation"] = grouped["variant_name"].map(_summary_interpretation)
    return grouped


def _score_alpha_group(group: pd.DataFrame) -> pd.DataFrame:
    scored = group.copy()
    rmse_available = scored["mean_rmse"].notna().any()
    scored["selection_metric"] = (
        scored["mean_rmse"] if rmse_available else scored["mean_residual_norm"]
    )
    scored["residual_norm_score"] = _normalize(scored["mean_residual_norm"])
    scored["rmse_norm_score"] = _normalize(scored["mean_rmse"])
    scored["qsvt_error_norm_score"] = _normalize(scored["mean_qsvt_error"])
    scored["query_norm_score"] = _normalize(scored["mean_query_count"])
    performance_score = (
        scored["rmse_norm_score"] if rmse_available else scored["residual_norm_score"]
    )
    scored["joint_score"] = (
        ALPHA_SELECTION_WEIGHTS["residual_or_rmse"] * performance_score
        + ALPHA_SELECTION_WEIGHTS["qsvt_error"] * scored["qsvt_error_norm_score"]
        + ALPHA_SELECTION_WEIGHTS["query_count"] * scored["query_norm_score"]
    )
    scored["score_metric_used"] = "rmse" if rmse_available else "residual"
    return scored


def _selection_rows(
    case_name: str,
    variant_name: str,
    scored: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    criteria = [
        ("residual_minimizing_alpha", "mean_residual_norm", "residual norm"),
        ("rmse_minimizing_alpha", "mean_rmse", "RMSE"),
        ("qsvt_error_minimizing_alpha", "mean_qsvt_error", "QSVT full-interval error"),
        ("query_or_degree_minimizing_alpha", "qsvt_query_degree_score", "query/degree proxy"),
        ("qsvt_resource_friendly_alpha", "qsvt_resource_score", "degree/query proxy"),
        ("joint_score_alpha", "joint_score", "diagnostic joint score"),
    ]
    resource_scored = scored.copy()
    resource_scored["qsvt_query_degree_score"] = resource_scored["query_norm_score"] + _normalize(
        resource_scored["mean_degree"]
    )
    resource_scored["qsvt_resource_score"] = (
        resource_scored["query_norm_score"] + resource_scored["qsvt_error_norm_score"]
    )
    for criterion, column, label in criteria:
        candidate = resource_scored.dropna(subset=[column])
        if candidate.empty:
            rows.append(
                {
                    "case_name": case_name,
                    "variant_name": variant_name,
                    "selected_alpha": np.nan,
                    "selection_criterion": criterion,
                    "score": np.nan,
                    "metric_used": label,
                    "caveat": "criterion not available for this group",
                }
            )
            continue
        row = candidate.sort_values([column, "alpha"], kind="mergesort").iloc[0]
        rows.append(
            {
                "case_name": case_name,
                "variant_name": variant_name,
                "selected_alpha": float(row["alpha"]),
                "selection_criterion": criterion,
                "score": float(row[column]),
                "metric_used": label,
                "caveat": (
                    "Diagnostic alpha-selection rule only; not field-calibrated "
                    "and not a replacement for estimator validation."
                ),
            }
        )
    return rows


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    min_value = float(numeric.min(skipna=True))
    max_value = float(numeric.max(skipna=True))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (numeric - min_value) / (max_value - min_value)


def _phase2_summary_frame(sweep: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_name, variant_name), group in sweep.groupby(["case_name", "variant_name"]):
        rows.append(
            {
                "case_name": case_name,
                "variant_name": variant_name,
                "alpha_count": int(group["alpha"].nunique()),
                "scenario_rows": int(group["scenario_count"].sum()),
                "best_mean_residual_norm": float(group["mean_residual_norm"].min()),
                "best_mean_rmse_if_available": float(group["mean_rmse_if_available"].min()),
                "best_mean_qsvt_full_interval_approx_error": float(
                    group["mean_qsvt_full_interval_approx_error"].min()
                ),
                "phase_validation_status": (
                    "pyqsp_pass_available_for_alpha_1e_minus_2"
                    if 1.0e-2 in set(np.round(group["alpha"].astype(float), 12))
                    else "not_validated"
                ),
                "alpha_selection_rows": len(
                    alpha[
                        (alpha["case_name"].astype(str) == str(case_name))
                        & (alpha["variant_name"].astype(str) == str(variant_name))
                    ]
                ),
                "status": "ok",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _write_sweep_outputs(
    output_dir: Path,
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Path]:
    results_csv = output_dir / "phase2_sweep_results.csv"
    results_json = output_dir / "phase2_sweep_results.json"
    summary_csv = output_dir / "phase2_sweep_summary.csv"
    failure_csv = output_dir / "phase2_failure_log.csv"
    manifest_path = output_dir / "manifest.json"
    results.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    failures.to_csv(failure_csv, index=False)
    write_json(results_json, {"rows": results.to_dict(orient="records")})
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": config,
            "artifacts": {
                "phase2_sweep_results_csv": str(results_csv),
                "phase2_sweep_results_json": str(results_json),
                "phase2_sweep_summary_csv": str(summary_csv),
                "phase2_failure_log_csv": str(failure_csv),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "results_csv": results_csv,
        "results_json": results_json,
        "summary_csv": summary_csv,
        "failure_log_csv": failure_csv,
        "manifest": manifest_path,
    }


def _alpha_selection_report(summary: pd.DataFrame, trace: pd.DataFrame) -> str:
    return f"""# QSVT Phase 2 Alpha-Selection Diagnostics

Alpha selection is diagnostic and controlled-benchmark-specific. It is not a
field-calibrated operational rule. The criteria do not change the original
solver behavior.

Default joint score weights:

```text
w_r = {ALPHA_SELECTION_WEIGHTS["residual_or_rmse"]}
w_e = {ALPHA_SELECTION_WEIGHTS["qsvt_error"]}
w_q = {ALPHA_SELECTION_WEIGHTS["query_count"]}
```

## Selection Summary

{_markdown_table(summary.head(40), _alpha_report_columns())}

## Trace Rows

Rows in trace: {len(trace)}

## Additional Criteria

The report includes residual-minimizing, RMSE-minimizing, QSVT-error-minimizing,
query/degree-minimizing, legacy resource-friendly, and joint-score alpha rows.
GCV and L-curve criteria are not implemented in this Phase 2 artifact because
the current objective is a traceable controlled-benchmark diagnostic over the
existing alpha grid rather than a new estimator-selection rule.

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _alpha_selection_metric_definitions() -> str:
    return """# Alpha-Selection Metric Definitions

Alpha selection is diagnostic and controlled-benchmark-specific. It is not a
field-calibrated operational rule.

- `residual_minimizing_alpha`: alpha with the smallest mean residual norm for a
  case and variant.
- `rmse_minimizing_alpha`: alpha with the smallest mean RMSE when benchmark
  reference states are available.
- `qsvt_error_minimizing_alpha`: alpha with the smallest mean full-interval QSVT
  approximation diagnostic error.
- `query_or_degree_minimizing_alpha`: alpha with the smallest normalized
  query/degree proxy. In the current fixed-degree sweep this can be tied across
  alphas, so the deterministic tie break is the smallest alpha.
- `qsvt_resource_friendly_alpha`: legacy combined query/error proxy retained for
  backward compatibility with earlier Phase 2 outputs.
- `joint_score_alpha`: alpha minimizing the configured weighted combination of
  estimator metric, QSVT approximation diagnostic error, and query-count proxy.
- `gcv_alpha`: not implemented in Phase 2 because no GCV trace was part of the
  existing controlled sweep.
- `l_curve_alpha`: not implemented in Phase 2 because no L-curve curvature trace
  was part of the existing controlled sweep.
"""


def _phase2_summary_markdown(summary: pd.DataFrame, alpha: pd.DataFrame) -> str:
    return f"""# QSVT Phase 2 Preconditioned Alpha Summary

Phase 2 evaluates original and preconditioned estimator variants separately for
IEEE118 and IEEE300, with alpha sensitivity and QSVT approximation/resource
diagnostics.

Coordinate-preconditioned Ridge is a separate estimator and may degrade
residual/RMSE. Transformed-penalty preconditioned Ridge preserves the original
x-space penalty and is reported as a consistency check.

## Variant Summary

{_markdown_table(summary, _phase2_report_columns())}

## Alpha Selection

Alpha-selection rows: {len(alpha)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(no rows)"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(subset.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [header, separator]
    for row in subset.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


def _alpha_report_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "selected_alpha",
        "selection_criterion",
        "score",
    ]


def _phase2_report_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "alpha_count",
        "scenario_rows",
        "best_mean_residual_norm",
        "best_mean_qsvt_full_interval_approx_error",
        "status",
    ]


def _summary_interpretation(variant_name: str) -> str:
    if variant_name == "coordinate_preconditioned_ridge":
        return "Separate coordinate-penalty estimator; inspect residual/RMSE."
    if variant_name == "transformed_penalty_preconditioned_ridge":
        return "Consistency-preserving x-space penalty formulation."
    if "qsvt" in variant_name:
        return "QSVT approximation/resource diagnostic only."
    return "Original Ridge/Tikhonov reference."


def _phase2_estimator_caveat(variant_name: str) -> str:
    if variant_name == "coordinate_preconditioned_ridge":
        return (
            "Coordinate-preconditioned Ridge is a separate estimator; it does not "
            "replace original Ridge if residual/RMSE degrade."
        )
    if variant_name == "transformed_penalty_preconditioned_ridge":
        return (
            "Transformed-penalty preconditioning preserves the original x-space "
            "penalty and is a consistency check."
        )
    if "qsvt" in variant_name:
        return "QSVT diagnostic rows report approximation/resource metrics only."
    return ESTIMATOR_CAVEAT


def _phase2_qsvt_caveat(phase_available: bool) -> str:
    phase_text = (
        "A pyqsp scalar phase pass is available for alpha=1e-2 only; "
        if phase_available
        else "No pyqsp scalar phase row is available for this alpha; "
    )
    return phase_text + QSVT_CAVEAT


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype=np.float64)
    return f"{values.shape}:{hash(values.tobytes())}"


def _failure_row(
    case_name: str,
    stage: str,
    seed: int | str,
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


def _result_columns() -> list[str]:
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
        "condition_number_original",
        "condition_number_preconditioned_if_applicable",
        "rmse_if_available",
        "angle_rmse_if_available",
        "voltage_rmse_if_available",
        "residual_norm",
        "weighted_residual_norm",
        "solution_norm",
        "relative_solution_error_vs_original_ridge",
        "relative_solution_error_vs_transformed_penalty",
        "qsvt_full_interval_approx_error",
        "qsvt_actual_singular_value_error",
        "qsvt_degree",
        "qsvt_query_count",
        "pyqsp_phase_available_for_this_target",
        "phase_validation_status",
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


def _alpha_summary_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "selected_alpha",
        "selection_criterion",
        "score",
        "metric_used",
        "caveat",
    ]


def _alpha_trace_columns() -> list[str]:
    return [
        "case_name",
        "variant_name",
        "alpha",
        "mean_residual_norm",
        "mean_rmse",
        "mean_qsvt_error",
        "mean_actual_sv_error",
        "mean_query_count",
        "mean_degree",
        "scenario_count",
        "selection_metric",
        "residual_norm_score",
        "rmse_norm_score",
        "qsvt_error_norm_score",
        "query_norm_score",
        "joint_score",
        "score_metric_used",
    ]


def _resolve_sweep_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_preconditioned_alpha_sweeps",
        "cases": ["ieee118", "ieee300"],
        "case_source": "pypower",
        "base_seed": 123,
        "seeds": PHASE2_SEEDS,
        "alphas": PHASE2_ALPHAS,
        "noise_stds": PHASE2_NOISE_STDS,
        "missing_ratios": PHASE2_MISSING_RATIOS,
        "bad_data_ratios": PHASE2_BAD_DATA_RATIOS,
        "bad_data_magnitude": 5.0,
        "qsvt_degree": PHASE_PASS_DEGREE,
        "method": "odd_chebyshev_ls",
        "grid_size": 220,
        "fallback_to_synthetic": False,
        "phase_backend": PYQSP_BACKEND,
        "phase_pass_tolerance": PYQSP_TOLERANCE,
        "phase_pass_query_count": PHASE_PASS_QUERY_COUNT,
        "phase_pass_full_domain_error": PHASE_FULL_DOMAIN_ERROR,
        "phase_pass_actual_singular_value_error": PHASE_ACTUAL_SV_ERROR,
    }
    if config:
        resolved.update(config)
    resolved["alphas"] = [float(value) for value in resolved["alphas"]]
    resolved["noise_stds"] = [float(value) for value in resolved["noise_stds"]]
    resolved["missing_ratios"] = [float(value) for value in resolved["missing_ratios"]]
    resolved["bad_data_ratios"] = [float(value) for value in resolved["bad_data_ratios"]]
    resolved["seeds"] = [int(value) for value in resolved["seeds"]]
    return resolved


def _resolve_alpha_selection_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_alpha_selection",
        "sweep_results_csv": (
            "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv"
        ),
        "cases": ["ieee118", "ieee300"],
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_summary_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase2_summary",
        "sweep_summary_csv": (
            "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv"
        ),
        "alpha_selection_summary_csv": (
            "outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.csv"
        ),
    }
    if config:
        resolved.update(config)
    return resolved
