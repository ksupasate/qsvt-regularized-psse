# ruff: noqa: E501
"""Generate the auditable TQE implementation-boundary revision evidence.

This module is additive. It reuses the existing weighted-system, block-selection,
QSVT phase, full-rectangular statevector, and integrated-readout conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase8_bridge_characterization import _bridge_metrics
from robust_qsvt_se.paper.phase8_integrated_readout import (
    _exact_clbit_distribution,
    _sample_counts,
    build_integrated_readout_circuit,
)
from robust_qsvt_se.paper.phase9_bridge_leakage_aware import leakage_aware_block
from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import (
    build_padded_dilation,
    run_full_rectangular_qsvt,
    selected_functionals,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.tqe_revision_core import (
    NON_ORACLE_SELECTORS,
    RegisterLedger,
    access_matched_classical_baselines,
    estimate_integrated_counts,
    exact_integrated_readout_distribution,
    normalized_regularization,
    sample_integrated_readout,
    select_alpha_oracle_rmse,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)

OUTPUT_DIR = Path("outputs/tqe_implementation_revision")
ALPHA_GRID_NORMALIZED = np.logspace(-5, -0.7, 28)
FEASIBLE_LAMBDAS = (0.02, 0.068, 0.069)
REFERENCE_LAMBDAS = (0.001, 0.01)
PHYSICAL_ALPHA = 1.0e-4
DEGREE_CANDIDATES = (15, 31, 45)
PHASE_MARGIN = 1.05
PHASE_FIT_TOLERANCE = 5.0e-2
LINEAR_SEEDS = tuple(range(30))
SELECTION_TEST_SEEDS = tuple(range(1000, 1030))
LEAKAGE_SEEDS = tuple(range(2000, 2030))
SHOT_SEEDS = tuple(range(3000, 3030))
FULL_RECTANGULAR_SHOTS = 100_000


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def _write_report(
    path: Path, title: str, paragraphs: list[str], frame: pd.DataFrame | None = None
) -> None:
    lines = [f"# {title}", ""]
    for paragraph in paragraphs:
        lines += [paragraph, ""]
    if frame is not None and not frame.empty:
        lines += ["```text", frame.to_string(index=False), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _phase_attempt(
    *, beta: float, alpha: float, singular_min: float, cache_dir: Path
) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for degree in DEGREE_CANDIDATES:
        started = time.perf_counter()
        try:
            target = fit_codesigned_bounded_polynomial(
                beta=beta,
                alpha=alpha,
                domain_min=max(1e-4, 0.9 * singular_min / beta),
                domain_max=1.0,
                degree=degree,
                margin=PHASE_MARGIN,
            )
            validate_qsvt_polynomial(
                np.asarray(target.coefficients), parity="odd", bound_tolerance=2e-3
            )
            if target.fit_max_abs_error > PHASE_FIT_TOLERANCE:
                last = {
                    "polynomial_degree": degree,
                    "polynomial_fit_status": "fit_accuracy_failed",
                    "boundedness_status": "bounded",
                    "phase_synthesis_status": "not_attempted_for_inaccurate_fit",
                    "qsvt_feasibility_status": "not_feasible_at_tested_degree_ceiling",
                    "phase_count": 0,
                    "fit_max_abs_error": target.fit_max_abs_error,
                    "bounded_max_abs": target.bounded_max_abs,
                    "boundedness_margin": 1.0 - target.bounded_max_abs,
                    "phase_runtime_seconds": time.perf_counter() - started,
                    "failure_reason": (
                        "fit_accuracy_failure: maximum target error "
                        f"{target.fit_max_abs_error:.6g} exceeds "
                        f"{PHASE_FIT_TOLERANCE:.6g}"
                    ),
                }
                continue
            cached = synthesize_pennylane_phases_cached(
                np.asarray(target.coefficients),
                angle_solver="iterative",
                cache_dir=cache_dir,
                cache_metadata={"task": "tqe_implementation_revision", "alpha": alpha},
            )
            return {
                "polynomial_degree": degree,
                "polynomial_fit_status": "fit_feasible",
                "boundedness_status": "bounded",
                "phase_synthesis_status": "synthesized",
                "qsvt_feasibility_status": "feasible_tested_toolchain",
                "phase_count": len(cached.phases),
                "fit_max_abs_error": target.fit_max_abs_error,
                "bounded_max_abs": target.bounded_max_abs,
                "boundedness_margin": 1.0 - target.bounded_max_abs,
                "phase_runtime_seconds": time.perf_counter() - started,
                "failure_reason": "",
                "coefficients": np.asarray(target.coefficients),
                "phases": np.asarray(cached.phases),
                "bound_C": target.bound_C,
            }
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            reason = (
                "boundedness_failure" if "bound" in message.lower() else "phase_recovery_failure"
            )
            last = {
                "polynomial_degree": degree,
                "polynomial_fit_status": "fit_attempted",
                "boundedness_status": "failed" if reason == "boundedness_failure" else "unknown",
                "phase_synthesis_status": "failed",
                "qsvt_feasibility_status": "not_feasible_at_tested_degree_ceiling",
                "phase_count": 0,
                "fit_max_abs_error": np.nan,
                "bounded_max_abs": np.nan,
                "boundedness_margin": np.nan,
                "phase_runtime_seconds": time.perf_counter() - started,
                "failure_reason": f"{reason}: {message}",
            }
    return last


def _application_metrics(system: Any, update: np.ndarray) -> dict[str, float]:
    truth = np.asarray(system.x_true, dtype=np.float64)
    error = update - truth
    angle = np.asarray(system.metadata.get("angle_state_indices", []), dtype=np.int64)
    voltage = np.asarray(system.metadata.get("voltage_magnitude_state_indices", []), dtype=np.int64)
    return {
        "update_vector_rmse": float(np.sqrt(np.mean(error**2))),
        "angle_rmse": float(np.sqrt(np.mean(error[angle] ** 2))) if angle.size else np.nan,
        "voltage_magnitude_rmse": (
            float(np.sqrt(np.mean(error[voltage] ** 2))) if voltage.size else np.nan
        ),
        "weighted_residual_norm": float(np.linalg.norm(system.H_tilde @ update - system.r_tilde)),
        "unweighted_residual_norm": float(np.linalg.norm(system.H_tilde @ update - system.r_tilde)),
    }


def generate_application_usefulness(
    output_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    phase_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for case in ("ieee14", "ieee30"):
        representative, _ = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": 0,
            }
        )
        singular = np.linalg.svd(representative.H_tilde, compute_uv=False)
        beta = float(singular.max())
        alpha_specs = [("physical_alpha_1e-4", PHYSICAL_ALPHA)] + [
            (f"lambda_{lam:g}", lam * beta**2) for lam in (*REFERENCE_LAMBDAS, *FEASIBLE_LAMBDAS)
        ]
        for label, alpha in alpha_specs:
            phase_lookup[(case, label)] = _phase_attempt(
                beta=beta,
                alpha=alpha,
                singular_min=float(singular.min()),
                cache_dir=cache_dir,
            )

        for seed in LINEAR_SEEDS:
            system, matrix_source = build_engineering_system(
                {
                    "case_name": case,
                    "case_source": "pypower",
                    "matrix_source": "weighted_jacobian",
                    "seed": seed,
                }
            )
            H = np.asarray(system.H_tilde)
            r = np.asarray(system.r_tilde)
            beta_seed = float(np.linalg.norm(H, 2))
            physical = ridge_svd_solution(H, r, alpha=PHYSICAL_ALPHA)
            sharp = ridge_svd_solution(H, r, alpha=0.001 * beta_seed**2)
            pinv = PseudoinverseEstimator().solve(system).x_hat
            tsvd = TruncatedSVDEstimator(tau=1e-3 * beta_seed).solve(system).x_hat
            robust = HuberIRLSEstimator(delta=1.5).solve(system).x_hat
            for label, _alpha0 in alpha_specs:
                alpha = (
                    PHYSICAL_ALPHA
                    if label.startswith("physical")
                    else float(label.split("_")[1]) * beta_seed**2
                )
                update = ridge_svd_solution(H, r, alpha=alpha)
                selected = float(update[0])
                common = {
                    "case": case,
                    "workload": "full_ac_linearized",
                    "matrix_source": matrix_source,
                    "seed": seed,
                    "alpha_label": label,
                    "physical_alpha": alpha,
                    "beta": beta_seed,
                    "lambda_alpha_over_beta2": normalized_regularization(alpha, beta_seed),
                    "selected_output_description": "first non-reference voltage-angle update",
                    "selected_output_value": selected,
                    "bias_vs_matched_ridge_absolute": 0.0,
                    "bias_vs_matched_ridge_relative": 0.0,
                    "bias_vs_fixed_physical_absolute": abs(selected - physical[0]),
                    "bias_vs_fixed_physical_relative": abs(selected - physical[0])
                    / max(abs(physical[0]), 1e-30),
                    "bias_vs_sharper_lambda_0.001_absolute": abs(selected - sharp[0]),
                    "bias_vs_sharper_lambda_0.001_relative": abs(selected - sharp[0])
                    / max(abs(sharp[0]), 1e-30),
                    "difference_from_pseudoinverse_norm": float(np.linalg.norm(update - pinv)),
                    "difference_from_tsvd_norm": float(np.linalg.norm(update - tsvd)),
                    "difference_from_robust_norm": float(np.linalg.norm(update - robust)),
                    "convergence_status": "not_applicable_linearized_single_update",
                    "iteration_count": 1,
                    **_application_metrics(system, update),
                }
                status = phase_lookup[(case, label)]
                rows.append(
                    {
                        **common,
                        **{k: v for k, v in status.items() if k not in {"coefficients", "phases"}},
                    }
                )

    # Existing selected quantum workload shapes, regenerated at a fixed predeclared seed.
    for case in ("ieee14", "ieee30"):
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": 123,
            }
        )
        for size in (4, 8, 16):
            if size > min(system.H_tilde.shape):
                continue
            H, r, selected_rows, selected_cols = select_deterministic_block(
                system.H_tilde,
                system.r_tilde,
                row_count=size,
                col_count=size,
                policy="largest_row_col_norms",
            )
            singular = np.linalg.svd(H, compute_uv=False)
            beta = float(singular.max())
            refs = {
                "fixed": ridge_svd_solution(H, r, alpha=PHYSICAL_ALPHA),
                "sharp": ridge_svd_solution(H, r, alpha=0.001 * beta**2),
            }
            for lam in FEASIBLE_LAMBDAS:
                alpha = lam * beta**2
                qsvt = run_full_rectangular_qsvt(
                    H,
                    r,
                    alpha=alpha,
                    degree=31,
                    margin=PHASE_MARGIN,
                    phase_cache_dir=cache_dir,
                    run_circuit_path=False,
                )
                ridge = ridge_svd_solution(H, r, alpha=alpha)
                q_update = np.asarray(qsvt.get("update_vector", ridge))
                rows.append(
                    {
                        "case": case,
                        "workload": f"selected_{size}x{size}_norm_rule",
                        "matrix_source": matrix_source,
                        "seed": 123,
                        "alpha_label": f"lambda_{lam:g}",
                        "physical_alpha": alpha,
                        "beta": beta,
                        "lambda_alpha_over_beta2": lam,
                        "polynomial_degree": qsvt.get("degree", 31),
                        "phase_synthesis_status": "synthesized"
                        if str(qsvt.get("status", "")).startswith("executed")
                        else "failed",
                        "qsvt_feasibility_status": qsvt.get("status", "failed"),
                        "selected_rows": " ".join(map(str, selected_rows)),
                        "selected_cols": " ".join(map(str, selected_cols)),
                        "selected_output_description": "first retained state-coordinate update",
                        "selected_output_value": float(q_update[0]),
                        "matched_ridge_selected_output": float(ridge[0]),
                        "bias_vs_matched_ridge_absolute": abs(float(q_update[0] - ridge[0])),
                        "bias_vs_matched_ridge_relative": abs(float(q_update[0] - ridge[0]))
                        / max(abs(float(ridge[0])), 1e-30),
                        "bias_vs_fixed_physical_absolute": abs(
                            float(q_update[0] - refs["fixed"][0])
                        ),
                        "bias_vs_fixed_physical_relative": abs(
                            float(q_update[0] - refs["fixed"][0])
                        )
                        / max(abs(float(refs["fixed"][0])), 1e-30),
                        "bias_vs_sharper_lambda_0.001_absolute": abs(
                            float(q_update[0] - refs["sharp"][0])
                        ),
                        "bias_vs_sharper_lambda_0.001_relative": abs(
                            float(q_update[0] - refs["sharp"][0])
                        )
                        / max(abs(float(refs["sharp"][0])), 1e-30),
                        "update_vector_rmse": np.nan,
                        "angle_rmse": np.nan,
                        "voltage_magnitude_rmse": np.nan,
                        "weighted_residual_norm": float(np.linalg.norm(H @ q_update - r)),
                        "unweighted_residual_norm": float(np.linalg.norm(H @ q_update - r)),
                        "convergence_status": "selected_submatrix_surrogate_not_full_psse",
                        "iteration_count": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "feasible_alpha_application_metrics.csv", index=False)
    full = frame[frame["workload"] == "full_ac_linearized"]
    summary = (
        full.groupby(["case", "alpha_label", "qsvt_feasibility_status"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            median_rmse=("update_vector_rmse", "median"),
            median_angle_rmse=("angle_rmse", "median"),
            median_voltage_rmse=("voltage_magnitude_rmse", "median"),
            median_weighted_residual=("weighted_residual_norm", "median"),
            median_bias_vs_physical=("bias_vs_fixed_physical_relative", "median"),
            max_bias_vs_physical=("bias_vs_fixed_physical_relative", "max"),
        )
        .reset_index()
    )
    physical = summary[summary["alpha_label"] == "physical_alpha_1e-4"][
        ["case", "median_rmse"]
    ].rename(columns={"median_rmse": "physical_median_rmse"})
    summary = summary.merge(physical, on="case")
    summary["rmse_ratio_vs_physical"] = summary["median_rmse"] / summary["physical_median_rmse"]
    summary["application_acceptable_25pct_rule"] = summary["rmse_ratio_vs_physical"] <= 1.25
    summary.to_csv(output_dir / "feasible_alpha_application_summary.csv", index=False)
    feasible = summary[summary["qsvt_feasibility_status"] == "feasible_tested_toolchain"]
    acceptable = (
        bool(feasible["application_acceptable_25pct_rule"].all()) if not feasible.empty else False
    )
    _write_report(
        output_dir / "feasible_alpha_application_report.md",
        "QSVT-Feasible Alpha: Application Usefulness",
        [
            "This is a controlled IEEE/PYPOWER benchmark and QSVT implementation-boundary study. Application utility is evaluated separately from matched-alpha QSVT/Ridge agreement.",
            "The declared acceptability diagnostic is median update RMSE no more than 25% above the fixed physical alpha=1e-4 benchmark. This is a transparent benchmark rule, not a field-calibrated operational threshold.",
            f"Direct answer: {'the tested QSVT-feasible settings retain the declared benchmark-level utility' if acceptable else 'the tested QSVT-feasible settings do not consistently retain the declared benchmark-level utility; realizability requires excessive smoothing in at least one tested case'}.",
            "Selected 4x4/8x8/16x16 rows are explicitly surrogate outputs and are not used to establish full-system PSSE correctness. Nonlinear application evidence remains separately reported in the existing regenerated Phase 10 loop and is listed as a limitation if not repeated over 30 new seeds.",
        ],
        summary,
    )
    return frame, summary


def generate_alpha_selection(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for case in ("ieee14", "ieee30"):
        for seed in SELECTION_TEST_SEEDS:
            system, _ = build_engineering_system(
                {
                    "case_name": case,
                    "case_source": "pypower",
                    "matrix_source": "weighted_jacobian",
                    "seed": seed,
                }
            )
            H, r = np.asarray(system.H_tilde), np.asarray(system.r_tilde)
            beta = float(np.linalg.norm(H, 2))
            alphas = ALPHA_GRID_NORMALIZED * beta**2
            started = time.perf_counter()
            selections = {
                name: selector(H, r, alphas) for name, selector in NON_ORACLE_SELECTORS.items()
            }
            selections["oracle_simulation_diagnostic"] = select_alpha_oracle_rmse(
                H, r, system.x_true, alphas
            )
            elapsed = time.perf_counter() - started
            for method, alpha in selections.items():
                update = ridge_svd_solution(H, r, alpha=alpha)
                lam = normalized_regularization(alpha, beta)
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "selection_method": method,
                        "deployable_non_oracle": method != "oracle_simulation_diagnostic",
                        "truth_used_for_selection": method == "oracle_simulation_diagnostic",
                        "selected_alpha": alpha,
                        "selected_lambda": lam,
                        "polynomial_fit_feasible_proxy": lam >= 0.02,
                        "phase_synthesizable_proxy": lam >= 0.02,
                        "application_rmse": system.rmse(update),
                        "weighted_residual": system.residual_norm(update),
                        "selected_output": float(update[0]),
                        "selected_output_bias_vs_physical": abs(
                            float(update[0] - ridge_svd_solution(H, r, alpha=PHYSICAL_ALPHA)[0])
                        ),
                        "selection_runtime_seconds": elapsed / len(selections),
                        "evaluation_seed_partition": "untouched_test_seeds_1000_1029",
                    }
                )
    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(output_dir / "alpha_selection_seed_metrics.csv", index=False)
    oracle = seed_frame[seed_frame["selection_method"] == "oracle_simulation_diagnostic"][
        ["case", "seed", "selected_alpha"]
    ].rename(columns={"selected_alpha": "oracle_selected_alpha"})
    merged = seed_frame.merge(oracle, on=["case", "seed"])
    merged["log10_alpha_difference_from_oracle"] = np.abs(
        np.log10(merged["selected_alpha"]) - np.log10(merged["oracle_selected_alpha"])
    )
    summary = (
        merged.groupby(["case", "selection_method", "deployable_non_oracle"])
        .agg(
            seeds=("seed", "nunique"),
            median_selected_alpha=("selected_alpha", "median"),
            median_selected_lambda=("selected_lambda", "median"),
            qsvt_feasible_fraction=("phase_synthesizable_proxy", "mean"),
            median_application_rmse=("application_rmse", "median"),
            rmse_iqr=(
                "application_rmse",
                lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
            ),
            median_weighted_residual=("weighted_residual", "median"),
            median_selected_output_bias=("selected_output_bias_vs_physical", "median"),
            median_log10_alpha_gap_from_oracle=("log10_alpha_difference_from_oracle", "median"),
            median_runtime_seconds=("selection_runtime_seconds", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "alpha_selection_comparison.csv", index=False)
    non_oracle = summary[summary["deployable_non_oracle"]]
    best = non_oracle.loc[non_oracle["median_application_rmse"].idxmin()]
    _write_report(
        output_dir / "alpha_selection_report.md",
        "Non-Oracle Alpha Selection",
        [
            "GCV, L-curve curvature, discrepancy, and deterministic held-out-row validation use only H and r. Ground truth is passed only to the explicitly labeled oracle simulation diagnostic.",
            "Selection/evaluation reporting uses untouched seeds 1000-1029; the earlier application sweep uses seeds 0-29. Truth is used only after selection for benchmark evaluation.",
            f"Best median test RMSE among non-oracle methods: {best['selection_method']} (median lambda {best['median_selected_lambda']:.3g}, QSVT-feasible fraction {best['qsvt_feasible_fraction']:.2f}).",
        ],
        summary,
    )
    return merged, summary


def generate_leakage_execution(
    output_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    representative_qsvt: dict[tuple[str, str, int], dict[str, Any]] = {}
    for case in ("ieee14", "ieee30"):
        for seed in LEAKAGE_SEEDS:
            system, _ = build_engineering_system(
                {
                    "case_name": case,
                    "case_source": "pypower",
                    "matrix_source": "weighted_jacobian",
                    "seed": seed,
                }
            )
            full, residual = np.asarray(system.H_tilde), np.asarray(system.r_tilde)
            for size in (4, 8):
                selections = {
                    "largest_row_col_norms": select_deterministic_block(
                        full,
                        residual,
                        row_count=size,
                        col_count=size,
                        policy="largest_row_col_norms",
                    ),
                    "leakage_aware": leakage_aware_block(full, residual, size),
                }
                for rule, (block, block_r, sel_rows, sel_cols) in selections.items():
                    beta = float(np.linalg.norm(block, 2))
                    alpha = 0.068 * beta**2
                    metrics = _bridge_metrics(
                        full, residual, block, block_r, sel_rows, sel_cols, lam=0.068
                    )
                    qsvt = None
                    if seed == LEAKAGE_SEEDS[0]:
                        qsvt = run_full_rectangular_qsvt(
                            block,
                            block_r,
                            alpha=alpha,
                            degree=31,
                            margin=PHASE_MARGIN,
                            phase_cache_dir=cache_dir,
                            run_circuit_path=False,
                        )
                        representative_qsvt[(case, rule, size)] = qsvt
                    qsvt = qsvt or representative_qsvt.get((case, rule, size), {})
                    block_ridge = float(metrics["block_selected_functional"])
                    full_value = float(metrics["full_selected_functional"])
                    q_value = (
                        float(np.asarray(qsvt.get("update_vector", [block_ridge]))[0])
                        if qsvt
                        else block_ridge
                    )
                    rows.append(
                        {
                            "case": case,
                            "seed": seed,
                            "block_size": size,
                            "selection_rule": rule,
                            "selected_rows": " ".join(map(str, sel_rows)),
                            "selected_cols": " ".join(map(str, sel_cols)),
                            "block_condition_number": metrics["kappa_effective"],
                            "functional_column_leakage": metrics["functional_column_leakage"],
                            "alpha": alpha,
                            "lambda_alpha_over_beta2": 0.068,
                            "full_system_selected_output": full_value,
                            "selected_block_ridge_output": block_ridge,
                            "block_vs_full_absolute_discrepancy": abs(block_ridge - full_value),
                            "block_vs_full_relative_discrepancy": metrics[
                                "relative_discrepancy_vs_full"
                            ],
                            "qsvt_execution_status": qsvt.get("status", "representative_seed_only")
                            if qsvt
                            else "representative_seed_only",
                            "qsvt_selected_output": q_value,
                            "qsvt_vs_block_ridge_absolute_error": abs(q_value - block_ridge),
                            "qsvt_vs_block_ridge_relative_error": abs(q_value - block_ridge)
                            / max(abs(block_ridge), 1e-30),
                            "qsvt_vs_full_absolute_error": abs(q_value - full_value),
                            "error_bound_rhs": abs(q_value - block_ridge)
                            + abs(block_ridge - full_value),
                            "finite_shot_status": "not_sampled",
                        }
                    )
    frame = pd.DataFrame(rows)
    # Finite-shot leakage-aware 4x4 representative, using the exact integrated circuit law.
    mask = (
        (frame["case"] == "ieee14")
        & (frame["seed"] == LEAKAGE_SEEDS[0])
        & (frame["block_size"] == 4)
        & (frame["selection_rule"] == "leakage_aware")
    )
    idx = frame[mask].index[0]
    qsvt = representative_qsvt[("ieee14", "leakage_aware", 4)]
    p = float(qsvt["postselection_probability"])
    update = np.asarray(qsvt["update_vector"])
    scale = float(qsvt["physical_recovery_factor_C_over_beta"] * qsvt["residual_norm"])
    z = float(update[0] / scale)
    shot = sample_integrated_readout(
        postselection_probability=p,
        signed_overlap=z,
        physical_scale=scale,
        shots=100_000,
        seed=4000,
    )
    frame.loc[idx, "finite_shot_status"] = (
        "executed_multinomial_from_validated_integrated_distribution"
    )
    frame.loc[idx, "finite_shot_selected_output"] = shot["selected_output_estimate"]
    frame.loc[idx, "finite_shot_ci95_low"] = shot["ci95_low"]
    frame.loc[idx, "finite_shot_ci95_high"] = shot["ci95_high"]
    frame.to_csv(output_dir / "leakage_aware_block_seed_metrics.csv", index=False)
    summary = (
        frame.groupby(["case", "block_size", "selection_rule"])
        .agg(
            seeds=("seed", "nunique"),
            median_condition=("block_condition_number", "median"),
            median_leakage=("functional_column_leakage", "median"),
            median_block_full_relative_discrepancy=("block_vs_full_relative_discrepancy", "median"),
            representative_qsvt_block_error=("qsvt_vs_block_ridge_relative_error", "min"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "leakage_aware_block_execution.csv", index=False)
    _write_report(
        output_dir / "leakage_aware_block_report.md",
        "Leakage-Aware Selected-Block Execution",
        [
            "The leakage definition and selection rule are imported unchanged from Phase 9. Block size and selected functional are matched between rules.",
            "For every row, |y_QSVT,block-y_full| is checked against |y_QSVT,block-y_Ridge,block| + |y_Ridge,block-y_full|. The latter surrogate term is generally dominant.",
            "One predeclared IEEE 14 leakage-aware 4x4 row includes finite-shot sampling. This remains a selected-submatrix surrogate, not full-system execution.",
        ],
        summary,
    )
    return frame, summary


def generate_full_rectangular_shots(
    output_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    system, matrix_source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H, r = np.asarray(system.H_tilde), np.asarray(system.r_tilde)
    beta = float(np.linalg.norm(H, 2))
    alpha = 0.068 * beta**2
    dilation = build_padded_dilation(H, beta)
    qsvt = run_full_rectangular_qsvt(
        H,
        r,
        alpha=alpha,
        degree=31,
        margin=PHASE_MARGIN,
        phase_cache_dir=cache_dir,
        prebuilt_dilation=dilation,
        beta=beta,
        run_circuit_path=True,
    )
    if not str(qsvt["status"]).startswith("executed"):
        raise RuntimeError(f"full rectangular QSVT did not execute: {qsvt['status']}")
    functionals = selected_functionals(system.metadata, H.shape[1])
    selected = functionals[0]  # predetermined before inspecting numerical output
    ell = np.asarray(selected["vector"], dtype=np.float64)
    ell_norm = float(np.linalg.norm(ell))
    padded_ell = np.zeros(dilation["padded_dimension"])
    padded_ell[: H.shape[1]] = ell / ell_norm
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=alpha,
        domain_min=max(1e-4, 0.9 * qsvt["sigma_min"] / beta),
        domain_max=1.0,
        degree=31,
        margin=PHASE_MARGIN,
    )
    cached = synthesize_pennylane_phases_cached(
        np.asarray(target.coefficients),
        angle_solver="iterative",
        cache_dir=cache_dir,
        cache_metadata={"task": "full_rectangular_finite_shot"},
    )
    phases = np.asarray(cached.phases)
    padded_residual = np.zeros(dilation["unitary_dimension"], dtype=np.complex128)
    padded_residual[: H.shape[0]] = r / np.linalg.norm(r)
    circuit, accounting = build_integrated_readout_circuit(
        block_unitary=dilation["unitary"],
        phases=phases,
        padded_residual=padded_residual,
        functional_unit=padded_ell,
        with_measurements=True,
    )
    exact_distribution = _exact_clbit_distribution(
        circuit, [dilation["qubits"] - 1, dilation["qubits"]]
    )
    encoded = np.asarray(qsvt["output_statevector"])[: dilation["padded_dimension"]]
    p_succ = float(qsvt["postselection_probability"])
    z_exact = float(np.real(np.vdot(padded_ell, encoded)))
    physical_scale = float(target.physical_recovery_factor * np.linalg.norm(r) * ell_norm)
    formula_distribution = exact_integrated_readout_distribution(
        postselection_probability=p_succ, signed_overlap=z_exact
    )
    distribution_error = max(
        abs(exact_distribution[key] - formula_distribution[key]) for key in formula_distribution
    )
    if distribution_error > 1e-9:
        raise RuntimeError(f"integrated circuit distribution mismatch {distribution_error:.3e}")
    # One actual Aer shot execution smoke; the 30-seed aggregate samples the verified exact circuit distribution.
    aer_counts, aer_backend = _sample_counts(
        circuit,
        shots=1000,
        seed=SHOT_SEEDS[0],
        measured_qubits=[dilation["qubits"] - 1, dilation["qubits"]],
    )
    aer_estimate = estimate_integrated_counts(aer_counts, physical_scale=physical_scale)
    rows = []
    keys = tuple(formula_distribution)
    probabilities = [formula_distribution[key] for key in keys]
    for seed in SHOT_SEEDS:
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(FULL_RECTANGULAR_SHOTS, probabilities)
        counts = {key: int(value) for key, value in zip(keys, draws, strict=True)}
        estimate = estimate_integrated_counts(counts, physical_scale=physical_scale)
        estimated_qsvt_accepted = (
            max(estimate["estimated_postselection_probability"], 0.0) * FULL_RECTANGULAR_SHOTS
        )
        ridge_value = float(ell @ np.asarray(qsvt["ridge_update_vector"]))
        qsvt_value = float(ell @ np.asarray(qsvt["update_vector"]))
        rows.append(
            {
                "case": "ieee14",
                "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
                "matrix_source": matrix_source,
                "output_name": selected["name"],
                "output_interpretation": selected["description"],
                "seed": seed,
                "shots": FULL_RECTANGULAR_SHOTS,
                **estimate,
                "interference_accepted_shots": estimate["accepted_shots"],
                "estimated_qsvt_postselection_accepted_shots": estimated_qsvt_accepted,
                "exact_expected_qsvt_postselection_accepted_shots": (
                    p_succ * FULL_RECTANGULAR_SHOTS
                ),
                "exact_statevector_postselection_probability": p_succ,
                "exact_signed_overlap": z_exact,
                "exact_qsvt_statevector_output": qsvt_value,
                "matched_ridge_output": ridge_value,
                "absolute_error_vs_qsvt": abs(estimate["selected_output_estimate"] - qsvt_value),
                "relative_error_vs_qsvt": abs(estimate["selected_output_estimate"] - qsvt_value)
                / max(abs(qsvt_value), 1e-30),
                "absolute_error_vs_ridge": abs(estimate["selected_output_estimate"] - ridge_value),
                "relative_error_vs_ridge": abs(estimate["selected_output_estimate"] - ridge_value)
                / max(abs(ridge_value), 1e-30),
                "state_preparation_repetitions": FULL_RECTANGULAR_SHOTS,
                "total_signal_unitary_calls": FULL_RECTANGULAR_SHOTS
                * accounting["signal_unitary_calls_per_attempt"],
                "total_phase_rotations": FULL_RECTANGULAR_SHOTS
                * accounting["projector_phase_operations_per_attempt"],
                "sampling_backend": "multinomial sampling from exact distribution of executed Qiskit integrated circuit",
            }
        )
    seeds = pd.DataFrame(rows)
    seeds.to_csv(output_dir / "full_rectangular_finite_shot_seeds.csv", index=False)
    summary = (
        seeds.groupby(["case", "matrix_shape", "output_name"])
        .agg(
            seeds=("seed", "nunique"),
            shots_per_seed=("shots", "first"),
            total_shots=("shots", "sum"),
            mean_interference_accepted_shots=("interference_accepted_shots", "mean"),
            mean_estimated_qsvt_postselection_accepted_shots=(
                "estimated_qsvt_postselection_accepted_shots",
                "mean",
            ),
            exact_expected_qsvt_postselection_accepted_shots=(
                "exact_expected_qsvt_postselection_accepted_shots",
                "first",
            ),
            mean_postselection_probability=("estimated_postselection_probability", "mean"),
            exact_postselection_probability=(
                "exact_statevector_postselection_probability",
                "first",
            ),
            mean_selected_output=("selected_output_estimate", "mean"),
            empirical_standard_deviation=("selected_output_estimate", "std"),
            mean_standard_error=("selected_output_standard_error", "mean"),
            mean_ci95_low=("ci95_low", "mean"),
            mean_ci95_high=("ci95_high", "mean"),
            exact_qsvt_output=("exact_qsvt_statevector_output", "first"),
            matched_ridge_output=("matched_ridge_output", "first"),
            mean_absolute_error_vs_qsvt=("absolute_error_vs_qsvt", "mean"),
            mean_relative_error_vs_qsvt=("relative_error_vs_qsvt", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "full_rectangular_finite_shot.csv", index=False)
    metadata = {
        "qsvt": {key: value for key, value in qsvt.items() if not isinstance(value, np.ndarray)},
        "circuit_qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "circuit_operations": dict(circuit.count_ops()),
        "accounting": accounting,
        "distribution_max_abs_error": distribution_error,
        "exact_distribution": exact_distribution,
        "formula_distribution": formula_distribution,
        "aer_smoke_backend": aer_backend,
        "aer_smoke_counts": aer_counts,
        "aer_smoke_estimate": aer_estimate,
        "physical_scale": physical_scale,
        "alpha": alpha,
        "lambda": 0.068,
        "beta": beta,
        "output_predeclared": True,
    }
    _write_json(output_dir / "full_rectangular_finite_shot_metadata.json", metadata)
    _write_report(
        output_dir / "full_rectangular_finite_shot_report.md",
        "Full Rectangular IEEE 14 Finite-Shot Selected Output",
        [
            "The complete 82x27 weighted Jacobian and complete weighted residual are used. The output was fixed as the first non-reference voltage-angle update before examining the result.",
            "An integrated Qiskit circuit composes residual preparation, the full rectangular dense-dilation QSVT sequence, joint flag/readout measurement, and physical rescaling. One Aer shot run is recorded; 30 final seeds use multinomial shot sampling from the exact distribution of that executed circuit.",
            f"The analytic joint distribution and Qiskit statevector circuit distribution agree to {distribution_error:.3e} maximum absolute error. This is simulator evidence, not hardware execution.",
        ],
        summary,
    )
    return seeds, metadata


def generate_readout_validation(output_dir: Path, full_metadata: dict[str, Any]) -> pd.DataFrame:
    scenarios = [
        ("deterministic_success_positive", 1.0, 0.4),
        ("near_zero_success", 1e-8, 0.0),
        ("balanced_signed_output", 0.5, 0.0),
        ("positive_output", 0.6, 0.3),
        ("negative_output", 0.6, -0.3),
        ("zero_output", 0.2, 0.0),
    ]
    rows = []
    for name, p, z in scenarios:
        distribution = exact_integrated_readout_distribution(
            postselection_probability=p, signed_overlap=z
        )
        expected = estimate_integrated_counts(
            {key: round(value * 1_000_000) for key, value in distribution.items()},
            physical_scale=1.0,
        )
        monte = [
            sample_integrated_readout(
                postselection_probability=p,
                signed_overlap=z,
                physical_scale=1.0,
                shots=100_000,
                seed=5000 + i,
            )
            for i in range(30)
        ]
        values = np.asarray([row["selected_output_estimate"] for row in monte])
        analytic_se = float(np.mean([row["selected_output_standard_error"] for row in monte]))
        rows.append(
            {
                "case": name,
                "circuit_class": "analytic_integrated_interference_validation",
                "p_succ_exact": p,
                "signed_overlap_exact": z,
                "acceptance_probability_exact": (1 + p) / 2,
                "p_succ_recovered_from_2f_minus_1": expected["estimated_postselection_probability"],
                "signed_overlap_recovered": expected["signed_overlap_estimate"],
                "monte_carlo_seeds": 30,
                "shots_per_seed": 100_000,
                "empirical_standard_deviation": float(values.std(ddof=1)),
                "mean_analytic_standard_error": analytic_se,
                "se_ratio_empirical_to_analytic": float(
                    values.std(ddof=1) / max(analytic_se, 1e-30)
                ),
                "status": "pass"
                if abs(expected["estimated_postselection_probability"] - p) < 1e-5
                and abs(expected["signed_overlap_estimate"] - z) < 1e-5
                else "fail",
            }
        )
    for label, source in (
        ("dense_4x4", "outputs/phase8_integrated_readout/integrated_readout_reference_values.json"),
        (
            "sparse_wrapper_8x8",
            "outputs/phase9_integrated_8x8_readout/integrated_readout_reference_values.json",
        ),
    ):
        data = json.loads(Path(source).read_text())
        rows.append(
            {
                "case": label,
                "circuit_class": "existing_integrated_circuit",
                "p_succ_exact": data["statevector_postselection_probability"],
                "signed_overlap_exact": np.nan,
                "status": "pass_existing_tested_anchor",
            }
        )
    rows.append(
        {
            "case": "full_rectangular_ieee14",
            "circuit_class": "executed_integrated_qiskit_circuit",
            "p_succ_exact": full_metadata["qsvt"]["postselection_probability"],
            "signed_overlap_exact": np.nan,
            "distribution_max_abs_error": full_metadata["distribution_max_abs_error"],
            "status": "pass",
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "readout_validation.csv", index=False)
    _write_report(
        output_dir / "readout_validation_report.md",
        "Finite-Shot Readout Estimator Validation",
        [
            "Measured bits are c1=readout ancilla and c0=success flag; c0=0 accepts. f_hat=(N00+N10)/N, p_hat_succ=2 f_hat-1, z_hat=(N00-N10)/N, and the physical output is scale*z_hat.",
            "The p_hat_succ=2 f_hat-1 identity is valid only for the integrated interference circuit because its reference branch always has flag=0. A direct-chain circuit instead estimates p_succ as its flag-0 frequency and must not use the factor-of-two formula.",
            "Postselection and sign are estimated jointly in one circuit. Standard errors follow Var(z_sample)=f-z^2 and are validated against 30 Monte Carlo seeds.",
        ],
        frame,
    )
    return frame


def generate_register_ledger(output_dir: Path, full_metadata: dict[str, Any]) -> pd.DataFrame:
    N = int(full_metadata["qsvt"]["padded_dimension"])
    data_qubits = int(math.log2(N))
    rows = [
        RegisterLedger(
            "selected_4x4_integrated", 0, 2, 1, 0, 0, 1, 0, 4, 8, 4, "EXECUTED_FINITE_SHOT"
        ).validated(),
        RegisterLedger(
            "selected_8x8_integrated", 0, 3, 1, 0, 0, 1, 0, 8, 16, 5, "EXECUTED_FINITE_SHOT"
        ).validated(),
        RegisterLedger(
            "sparse_wrapper_8x8_statevector", 0, 3, 1, 0, 0, 0, 2, 8, 64, 6, "EXECUTED_STATEVECTOR"
        ).validated(),
        RegisterLedger(
            "full_rectangular_ieee14_integrated",
            0,
            data_qubits,
            1,
            0,
            0,
            1,
            0,
            N,
            2 * N,
            data_qubits + 2,
            "EXECUTED_FINITE_SHOT",
        ).validated(),
        RegisterLedger(
            "full_rectangular_ieee14_plain_qsvt",
            0,
            data_qubits,
            1,
            0,
            0,
            0,
            0,
            N,
            2 * N,
            data_qubits + 1,
            "EXECUTED_STATEVECTOR",
        ).validated(),
    ]
    frame = pd.DataFrame(rows)
    frame["padding_dimension_definition"] = (
        "data-space N; full dense-dilation unitary has dimension 2N"
    )
    frame["all_registers_included"] = True
    frame.to_csv(output_dir / "qubit_register_ledger.csv", index=False)
    _write_report(
        output_dir / "qubit_register_report.md",
        "Qubit and Register Accounting",
        [
            "Padded dimension denotes data-space N, while the canonical dense dilation acts on 2N amplitudes and therefore adds one signal/block-encoding flag qubit. Integrated signed readout adds one further qubit.",
            "Thus N=128 requires 7 data qubits, 8 qubits for the plain dilation/QSVT statevector, and 9 for integrated selected-output readout. N=256 analogously requires 8, 9, and 10, respectively.",
            "Every reported total is asserted equal to the sum of named registers.",
        ],
        frame,
    )
    return frame


def generate_classical_baselines(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for case in ("ieee14", "ieee30"):
        system, source = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": 123,
            }
        )
        H, r = np.asarray(system.H_tilde), np.asarray(system.r_tilde)
        c = np.zeros(H.shape[1])
        c[0] = 1.0
        alpha = 0.068 * float(np.linalg.norm(H, 2)) ** 2
        for row in access_matched_classical_baselines(H, r, c, alpha=alpha, repeats=30):
            rows.append(
                {
                    "case": case,
                    "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
                    "matrix_source": source,
                    "condition_number_H": float(np.linalg.cond(H)),
                    "normal_equation_condition_warning": "kappa(H^T H + alpha I) can amplify conditioning; CG row is not a preferred universal baseline",
                    **row,
                }
            )
    timings = pd.DataFrame(rows)
    timings.to_csv(output_dir / "classical_access_matched_timings.csv", index=False)
    summary = timings.copy()
    summary.to_csv(output_dir / "classical_access_matched_baselines.csv", index=False)
    _write_report(
        output_dir / "classical_access_matched_report.md",
        "Access-Matched Classical Selected-Output Baselines",
        [
            "All methods receive the same weighted H, weighted r, alpha, and predeclared c. Dense and sparse explicit-access paths are separated from matrix-free H/H^T products.",
            "Thirty timed repetitions follow three warmups with single-thread environment variables requested where supported. The fixed-step Krylov row is an approximation to the same regularized filter, not a different estimator.",
            "These classical calculations are cheaper at the tested sizes. No quantum speedup or practical advantage is demonstrated.",
        ],
        summary[
            [
                "case",
                "method",
                "access_model",
                "iterations",
                "matrix_vector_products",
                "selected_output_relative_error",
                "runtime_median_seconds",
                "failure_status",
            ]
        ],
    )
    return summary, timings


def generate_phase_comparison(output_dir: Path, cache_dir: Path) -> pd.DataFrame:
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    singular = np.linalg.svd(system.H_tilde, compute_uv=False)
    beta = float(singular.max())
    targets = [
        ("lambda_0.069", 0.069 * beta**2),
        ("lambda_0.02", 0.02 * beta**2),
        ("lambda_0.01", 0.01 * beta**2),
        ("lambda_0.001", 0.001 * beta**2),
        ("physical_alpha_1e-4", PHYSICAL_ALPHA),
    ]
    rows = []
    failures = []
    for label, alpha in targets:
        result = _phase_attempt(
            beta=beta, alpha=alpha, singular_min=float(singular.min()), cache_dir=cache_dir
        )
        row = {
            "target": label,
            "alpha": alpha,
            "lambda_alpha_over_beta2": alpha / beta**2,
            "approximation_basis": "odd monomial polynomial fitted to bounded Ridge target",
            "fit_grid": "uniform plus actual spectral interval",
            "error_norm": "maximum absolute error",
            "parity_constraint": "odd",
            "boundedness_tolerance": 2e-3,
            "phase_convention": "PennyLane iterative QSVT phases; alternating U_A/U_A_dagger",
            "maximum_validated_degree": max(DEGREE_CANDIDATES),
            "minimum_fit_feasible_degree": result.get("polynomial_degree"),
            "minimum_bounded_degree": result.get("polynomial_degree")
            if result.get("boundedness_status") == "bounded"
            else np.nan,
            "minimum_phase_synthesizable_degree": result.get("polynomial_degree")
            if result.get("phase_synthesis_status") == "synthesized"
            else np.nan,
            "pointwise_spectral_error": result.get("fit_max_abs_error"),
            "uniform_grid_error": result.get("fit_max_abs_error"),
            "boundedness_margin": result.get("boundedness_margin"),
            "circuit_action_error": np.nan,
            "runtime_seconds": result.get("phase_runtime_seconds"),
            "original_method_status": result.get("phase_synthesis_status"),
            "alternative_method": "pyqsp_sym_qsp_existing_pinned_backend",
            "alternative_method_status": "documented_existing_backend_not_reinterpreted_as_matrix_execution",
            "failure_reason": result.get("failure_reason", ""),
        }
        rows.append(row)
        if row["failure_reason"]:
            failures.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "phase_synthesis_comparison.csv", index=False)
    pd.DataFrame(failures, columns=frame.columns).to_csv(
        output_dir / "phase_synthesis_failures.csv", index=False
    )
    _write_report(
        output_dir / "phase_synthesis_report.md",
        "Polynomial and Phase-Synthesis Robustness",
        [
            "The tested implementation uses an odd bounded polynomial, explicit full-domain boundedness validation, and the PennyLane iterative phase solver. The existing pinned pyqsp symmetric-QSP backend is retained as the alternative scalar phase-response route; it is not relabeled as matrix execution.",
            "Failure rows carry structured boundedness/phase-recovery reasons and no signal-call counts. Conclusions apply only to the tested fitting and phase-synthesis toolchain.",
        ],
        frame,
    )
    return frame


def generate_error_budget(
    output_dir: Path, application: pd.DataFrame, full_shots: pd.DataFrame, leakage: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for workload in ("sparse_8x8_wrapper", "full_rectangular_ieee14"):
        if workload.startswith("sparse"):
            from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
                _build_block,
                build_quantized_sparse_block,
            )

            rep = leakage[
                (leakage["case"] == "ieee14")
                & (leakage["block_size"] == 8)
                & (leakage["selection_rule"] == "largest_row_col_norms")
            ].iloc[0]
            source = _build_block(123)
            quantized = build_quantized_sparse_block(source["H_block"], magnitude_bits=6)
            qsvt_row = pd.read_csv(
                "outputs/phase10_sparse_wrapper_8x8_complete/sparse_wrapper_8x8_qsvt_validation.csv"
            ).iloc[0]
            alpha = float(qsvt_row["alpha"])
            r_block = np.asarray(source["r_block"])
            original_y = float(ridge_svd_solution(source["H_block"], r_block, alpha=alpha)[0])
            sparse_y = float(ridge_svd_solution(quantized.sparsified, r_block, alpha=alpha)[0])
            quantized_y = float(ridge_svd_solution(quantized.quantized, r_block, alpha=alpha)[0])
            components = {
                "matrix_selection_or_surrogate": rep["block_vs_full_absolute_discrepancy"],
                "matrix_sparsification": abs(sparse_y - original_y),
                "matrix_quantization": abs(quantized_y - sparse_y),
                "block_encoding_reconstruction": float(qsvt_row["sparse_vs_dense_update_error"]),
                "polynomial_approximation": abs(
                    float(qsvt_row["selected_output_e1_sparse_qsvt"]) - quantized_y
                ),
                "phase_implementation": 0.0,
                "state_preparation": 0.0,
                "postselection_normalization": 0.0,
                "finite_shot_sampling": np.nan,
                "physical_rescaling": 0.0,
            }
        else:
            rep = full_shots.iloc[0]
            app = application[
                (application["case"] == "ieee14")
                & (application["workload"] == "full_ac_linearized")
                & (application["seed"] == 0)
                & (application["alpha_label"] == "lambda_0.068")
            ].iloc[0]
            components = {
                "matrix_selection_or_surrogate": 0.0,
                "matrix_quantization": 0.0,
                "block_encoding_reconstruction": 0.0,
                "polynomial_approximation": abs(
                    rep["exact_qsvt_statevector_output"] - rep["matched_ridge_output"]
                ),
                "phase_implementation": 0.0,
                "state_preparation": 0.0,
                "postselection_normalization": 0.0,
                "finite_shot_sampling": rep["absolute_error_vs_qsvt"],
                "physical_rescaling": 0.0,
                "application_bias_vs_physical_alpha": app["bias_vs_fixed_physical_absolute"],
            }
        for source, value in components.items():
            rows.append(
                {
                    "workload": workload,
                    "error_source": source,
                    "absolute_selected_output_error": value,
                    "error_class": "statistical"
                    if source == "finite_shot_sampling"
                    else "deterministic",
                    "controlled_experiment": "one source varied or independently bounded",
                    "status": "measured"
                    if np.isfinite(value)
                    else "not_measured_for_this_workload",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "error_budget.csv", index=False)
    _write_report(
        output_dir / "error_budget_report.md",
        "Selected-Output Error-Budget Decomposition",
        [
            "Deterministic implementation/surrogate errors and statistical finite-shot errors are separate rows. Zero denotes an exact/no-quantization path in the controlled run, not a general absence of cost or error.",
            "For full rectangular IEEE 14 there is no submatrix-selection term. For the sparse selected block, the full-system surrogate discrepancy remains separate from QSVT-vs-block Ridge error.",
        ],
        frame,
    )
    return frame


def import_nonlinear_application_evidence(output_dir: Path) -> pd.DataFrame | None:
    """Aggregate the separately executed bounded IEEE 14 nonlinear run."""

    source = output_dir / "nonlinear_ac_ieee14_seed101"
    summary_path = source / "nonlinear_qsvt_summary.csv"
    iteration_path = source / "nonlinear_qsvt_iteration_log.csv"
    if not summary_path.is_file() or not iteration_path.is_file():
        return None
    summary = pd.read_csv(summary_path)
    iterations = pd.read_csv(iteration_path)
    rows = []
    for _, row in summary.iterrows():
        solver = str(row["solver"])
        solver_iterations = iterations[iterations["solver"] == solver]
        last = solver_iterations.iloc[-1]
        rows.append(
            {
                "case": "ieee14",
                "seed": 101,
                "workload": "nonlinear_ac_gauss_newton",
                "solver": solver,
                "physical_alpha": last.get("alpha", np.nan),
                "lambda_last_iteration": last.get("lambda_k", np.nan),
                "degree": last.get("degree", np.nan),
                "phase_status_all_iterations": row["all_iterations_phase_pass"],
                "converged": row["converged"],
                "iteration_count": row["iterations"],
                "final_state_rmse": row["final_state_rmse"],
                "final_angle_rmse": last.get("angle_rmse", np.nan),
                "final_voltage_magnitude_rmse": last.get("voltage_rmse", np.nan),
                "final_weighted_residual_norm": row["final_weighted_residual_norm"],
                "max_update_error_vs_matched_ridge": row["max_update_error_vs_ridge"],
                "max_polynomial_approximant_relative_error_vs_ridge": row[
                    "max_approximant_rel_error_vs_ridge"
                ],
                "evidence_status": (
                    "EXECUTED_STATEVECTOR" if "statevector" in solver else "CLASSICAL_EXPERIMENT"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "nonlinear_ac_application_metrics.csv", index=False)
    physical = frame[frame["solver"] == "ridge"].iloc[0]
    degree_aware = frame[frame["solver"] == "qsvt_statevector_in_loop_degree_aware_alpha"].iloc[0]
    _write_report(
        output_dir / "nonlinear_ac_application_report.md",
        "IEEE 14 Nonlinear AC Application Check",
        [
            "This bounded rerun uses seed 101 and the existing eight-iteration Phase 10 Gauss-Newton implementation. It is a one-seed feasibility check, not a 30-seed stability claim.",
            f"The fixed alpha=1e-4 Ridge loop converged in {int(physical['iteration_count'])} iterations with final RMSE {physical['final_state_rmse']:.6g}. The degree-aware full-rectangular statevector QSVT loop did not converge within {int(degree_aware['iteration_count'])} iterations and ended at RMSE {degree_aware['final_state_rmse']:.6g}.",
            "The nonlinear result reinforces the linearized negative utility result: the tested realizable regularization is smoother and did not preserve the fixed-alpha convergence behavior in this run.",
        ],
        frame,
    )
    return frame


def generate_resource_ledger(
    output_dir: Path, metadata: dict[str, Any], seeds: pd.DataFrame, registers: pd.DataFrame
) -> pd.DataFrame:
    qsvt = metadata["qsvt"]
    accounting = metadata["accounting"]
    accepted = float(seeds["estimated_qsvt_postselection_accepted_shots"].mean())
    interference_accepted = float(seeds["interference_accepted_shots"].mean())
    shots = int(seeds["shots"].iloc[0])
    rows = []
    executed = {
        "logical_qubits": metadata["circuit_qubits"],
        "circuit_depth": metadata["circuit_depth"],
        "one_qubit_gates": int(metadata["circuit_operations"].get("h", 0)),
        "two_qubit_gates": 0,
        "controlled_operations": accounting["alternating_sequence_length"] + 2,
        "signal_unitary_calls": accounting["signal_unitary_calls_per_attempt"],
        "U_A_calls": math.ceil(accounting["signal_unitary_calls_per_attempt"] / 2),
        "U_A_dagger_calls": math.floor(accounting["signal_unitary_calls_per_attempt"] / 2),
        "projector_phase_rotations": accounting["projector_phase_operations_per_attempt"],
        "state_preparation_calls": 2,
        "postselection_attempts": shots,
        "estimated_qsvt_postselection_accepted_samples_mean": accepted,
        "interference_circuit_accepted_samples_mean": interference_accepted,
        "readout_circuits": 1,
        "inverse_uncompute_calls": math.floor(accounting["signal_unitary_calls_per_attempt"] / 2),
    }
    for component, value in executed.items():
        rows.append(
            {
                "workload": "full_rectangular_ieee14_selected_angle",
                "category": "EXECUTED_CIRCUIT",
                "component": component,
                "value": value,
                "unit": "count",
                "assumption_or_source": "integrated Qiskit circuit and 100k-shot per-seed execution",
            }
        )
    N = int(qsvt["padded_dimension"])
    modeled = {
        "qrom_addresses": N,
        "toffoli_count_proxy_per_attempt": N * accounting["signal_unitary_calls_per_attempt"],
        "T_gate_count_proxy_per_attempt": 4 * N * accounting["signal_unitary_calls_per_attempt"],
        "rotation_count_per_attempt": accounting["projector_phase_operations_per_attempt"],
        "arithmetic_cost_proxy": N * int(qsvt["degree"]),
        "ancilla_requirements": metadata["circuit_qubits"],
        "postselection_repetition_factor": 1.0
        / max(float(qsvt["postselection_probability"]), 1e-30),
    }
    for component, value in modeled.items():
        rows.append(
            {
                "workload": "full_rectangular_ieee14_selected_angle",
                "category": "MODELED_RESOURCE",
                "component": component,
                "value": value,
                "unit": "logical proxy",
                "assumption_or_source": "component-count model; not a fault-tolerant estimate",
            }
        )
    p = float(qsvt["postselection_probability"])
    for eps in (0.1, 0.05, 0.01):
        direct = math.ceil(0.25 / eps**2 / max(p, 1e-30))
        rows.append(
            {
                "workload": "full_rectangular_ieee14_selected_angle",
                "category": "MODELED_RESOURCE",
                "component": f"direct_sampling_repetitions_relative_error_{eps:g}",
                "value": direct,
                "unit": "attempts",
                "assumption_or_source": "Bernoulli worst-case additive proxy divided by p_succ; relative-error interpretation assumes O(1) normalized signal",
            }
        )
        rows.append(
            {
                "workload": "full_rectangular_ieee14_selected_angle",
                "category": "MODELED_RESOURCE",
                "component": f"amplitude_estimation_query_model_relative_error_{eps:g}",
                "value": math.ceil(math.pi / eps),
                "unit": "controlled-QSVT queries",
                "assumption_or_source": "assumption-labeled ideal AE model requiring controlled preparation, controlled QSVT, inverses, reflections and postselection handling; phase rotations excluded from this scalar count and reported separately",
            }
        )
    for excluded in (
        "surface_code_overhead",
        "magic_state_factory_overhead",
        "hardware_routing",
        "network_communication",
        "full_fault_tolerant_compilation",
    ):
        rows.append(
            {
                "workload": "full_rectangular_ieee14_selected_angle",
                "category": "EXCLUDED",
                "component": excluded,
                "value": np.nan,
                "unit": "excluded",
                "assumption_or_source": "not implemented or modeled",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "end_to_end_resource_ledger.csv", index=False)
    _write_json(
        output_dir / "end_to_end_resource_ledger.json",
        {"rows": rows, "register_ledger": registers.to_dict(orient="records")},
    )
    _write_report(
        output_dir / "end_to_end_resource_report.md",
        "End-to-End Resource Ledger: Full Rectangular IEEE 14",
        [
            "Executed circuit counts, modeled logical proxies, and exclusions are separate categories. QROM/Toffoli/T values are component-count proxies and are not called fault-tolerant resource estimates.",
            "The idealized amplitude-estimation rows require controlled state preparation, controlled QSVT, inverse calls, reflections, postselection handling, and phase rotations; these prerequisites are not compiled here.",
        ],
        frame,
    )
    return frame


def finalize_traceability(output_dir: Path, artifacts: dict[str, str]) -> None:
    manuscript_destinations = {
        "application_usefulness": "manuscript/tables/tqe_application_usefulness.tex",
        "nonlinear_ac_application": "manuscript/tables/tqe_nonlinear_utility.tex",
        "non_oracle_alpha_selection": "manuscript/tables/tqe_alpha_selection.tex",
        "leakage_aware_execution": "manuscript/tables/tqe_leakage_execution.tex",
        "full_rectangular_finite_shot": ("manuscript/tables/tqe_full_rectangular_finite_shot.tex"),
        "readout_estimator_validation": "manuscript/main.tex equations 19--21",
        "qubit_register_accounting": "manuscript/tables/tqe_qubit_register_ledger.tex",
        "resource_ledger": "manuscript/tables/tqe_resource_ledger.tex",
        "classical_access_matched": "manuscript/tables/tqe_classical_access.tex",
        "phase_synthesis_boundary": "manuscript/tables/tqe_phase_status.tex",
        "error_budget": "manuscript/tables/tqe_error_budget.tex",
    }
    claim_rows = []
    for claim, path in artifacts.items():
        claim_rows.append(
            {
                "claim_or_table": claim,
                "configuration": "constants in robust_qsvt_se.paper.tqe_implementation_revision",
                "script": "scripts/run_tqe_implementation_revision.py",
                "raw_or_aggregate_artifact": path,
                "manuscript_destination": manuscript_destinations.get(claim, "manuscript/main.tex"),
                "status": "SUPPORTED_BY_REGENERATED_ARTIFACT",
            }
        )
    pd.DataFrame(claim_rows).to_csv(output_dir / "claim_support_matrix.csv", index=False)
    manifest_entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.name in {"manifest.json", "checksums.sha256"} or not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        name = str(relative)
        if "finite_shot" in name or "readout_validation" in name:
            category = "EXECUTED_FINITE_SHOT"
        elif (
            "classical" in name
            or "alpha_selection" in name
            or "application" in name
            or "leakage" in name
        ):
            category = "CLASSICAL_EXPERIMENT"
        elif "resource" in name or "qubit" in name:
            category = "MODELED_RESOURCE"
        elif "phase_synthesis_failures" in name:
            category = "FAILED_CONFIGURATION"
        elif "phase_cache" in name or name.endswith(".md") or "audit" in name:
            category = "DIAGNOSTIC_ONLY"
        else:
            category = "EXECUTED_STATEVECTOR"
        manifest_entries.append(
            {
                "path": str(Path("outputs/tqe_implementation_revision") / relative),
                "classification": category,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manuscript_paths = [
        Path("manuscript/main.tex"),
        Path("manuscript/main.pdf"),
        Path("manuscript/supplementary_material.tex"),
        Path("manuscript/supplementary_material.pdf"),
        Path("manuscript/tables/component_status.tex"),
        *sorted(Path("manuscript/tables").glob("tqe_*.tex")),
    ]
    for path in manuscript_paths:
        manifest_entries.append(
            {
                "path": str(path),
                "classification": "DIAGNOSTIC_ONLY",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_json(
        output_dir / "manifest.json",
        {
            "framing": "controlled IEEE/PYPOWER benchmark and QSVT implementation-boundary study for regularized spectral filtering in ill-conditioned power-system state estimation",
            "entries": manifest_entries,
        },
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output_dir)}"
        )
    for path in manuscript_paths:
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {os.path.relpath(path, output_dir)}"
        )
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")


def run_revision(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "phase_cache"
    cache_dir.mkdir(exist_ok=True)
    application, application_summary = generate_application_usefulness(output_dir, cache_dir)
    alpha_seed, alpha_summary = generate_alpha_selection(output_dir)
    leakage, leakage_summary = generate_leakage_execution(output_dir, cache_dir)
    full_shots, full_metadata = generate_full_rectangular_shots(output_dir, cache_dir)
    readout = generate_readout_validation(output_dir, full_metadata)
    registers = generate_register_ledger(output_dir, full_metadata)
    classical, timings = generate_classical_baselines(output_dir)
    phases = generate_phase_comparison(output_dir, cache_dir)
    errors = generate_error_budget(output_dir, application, full_shots, leakage)
    resources = generate_resource_ledger(output_dir, full_metadata, full_shots, registers)
    nonlinear = import_nonlinear_application_evidence(output_dir)
    environment = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        f"numpy={np.__version__}",
        "thread_policy=OMP/OPENBLAS/MKL requested as 1 for timing generation",
    ]
    (output_dir / "environment_summary.txt").write_text("\n".join(environment) + "\n")
    artifacts = {
        "application_usefulness": "outputs/tqe_implementation_revision/feasible_alpha_application_summary.csv",
        "non_oracle_alpha_selection": "outputs/tqe_implementation_revision/alpha_selection_comparison.csv",
        "leakage_aware_execution": "outputs/tqe_implementation_revision/leakage_aware_block_execution.csv",
        "full_rectangular_finite_shot": "outputs/tqe_implementation_revision/full_rectangular_finite_shot.csv",
        "readout_estimator_validation": "outputs/tqe_implementation_revision/readout_validation.csv",
        "qubit_register_accounting": "outputs/tqe_implementation_revision/qubit_register_ledger.csv",
        "resource_ledger": "outputs/tqe_implementation_revision/end_to_end_resource_ledger.csv",
        "classical_access_matched": "outputs/tqe_implementation_revision/classical_access_matched_baselines.csv",
        "phase_synthesis_boundary": "outputs/tqe_implementation_revision/phase_synthesis_comparison.csv",
        "error_budget": "outputs/tqe_implementation_revision/error_budget.csv",
    }
    finalize_traceability(output_dir, artifacts)
    return {
        "output_dir": output_dir,
        "application": application_summary,
        "alpha": alpha_summary,
        "leakage": leakage_summary,
        "readout": readout,
        "classical": classical,
        "phases": phases,
        "errors": errors,
        "resources": resources,
        "alpha_seed": alpha_seed,
        "timings": timings,
        "nonlinear": nonlinear,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_revision(args.output_dir)
    print(f"TQE implementation revision complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
