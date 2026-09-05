from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    FULL_GATE_LEVEL_COVERAGE_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    fit_bounded_ridge_polynomial,
    load_sweep_subproblem,
)
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import (
    fit_actual_singular_interpolating_polynomial,
    ridge_update_svd,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import construct_padded_block_encoding
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    DEFAULT_BASIS_GATES,
    DEFAULT_PHASE_CONVENTION,
    PhaseSynthesisResult,
    evaluate_qsvt_transform,
    qsvt_rescaled_update_from_transform,
    synthesize_qsvt_phases,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE_RESCUE_DIR = "phase_synthesis_8x8_rescue"
SMALL_TOL = 1.0e-14

ATTEMPT_COLUMNS = [
    "case_name",
    "subproblem_size",
    "alpha",
    "epsilon_target",
    "degree",
    "gamma",
    "gamma_multiplier",
    "target_scale_safety",
    "approximation_mode",
    "polynomial_parity",
    "parity_check_status",
    "parity_violation_norm",
    "max_abs_target_dense_grid",
    "admissibility_margin",
    "dense_grid_error",
    "actual_singular_value_error",
    "backend_name",
    "backend_parameters",
    "phase_synthesis_status",
    "phase_count",
    "synthesis_runtime_seconds",
    "qsvt_circuit_status",
    "simulation_status",
    "transpilation_status",
    "num_qubits",
    "raw_depth",
    "transpiled_depth",
    "transpiled_cx_count",
    "circuit_vs_polynomial_fro_error",
    "circuit_vs_polynomial_spectral_error",
    "circuit_vs_ridge_relative_update_error",
    "absolute_update_error",
    "ridge_update_norm",
    "qsvt_update_norm",
    "residual_gap",
    "ridge_residual_norm",
    "qsvt_residual_norm",
    "success_probability",
    "rescue_status",
    "failure_mode",
    "failure_message_short",
    "runtime_seconds",
]

SUMMARY_COLUMNS = [
    "case_name",
    "subproblem_size",
    "rescued",
    "best_degree",
    "best_gamma_multiplier",
    "best_target_scale_safety",
    "best_backend",
    "best_relative_update_error",
    "best_absolute_update_error",
    "best_residual_gap",
    "best_success_probability",
    "best_phase_count",
    "best_failure_mode_if_unrescued",
    "recommended_interpretation",
]


@dataclass(frozen=True, slots=True)
class RescueTarget:
    case_name: str
    subproblem_size: int
    selection_criterion: str
    alpha: float
    epsilon_target: float
    original_degree: int


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    degree: int
    gamma_multiplier: float
    target_scale_safety: float
    approximation_mode: str
    stage: str


@dataclass(frozen=True, slots=True)
class PolynomialBuild:
    polynomial: Polynomial
    coefficients: np.ndarray
    C_alpha: float
    effective_C_alpha: float
    coefficient_scale: float
    raw_max_abs_dense: float
    max_abs_dense: float
    dense_grid_error: float
    actual_singular_value_error: float
    parity_violation_norm: float
    parity_check_status: str


def run_phase_synthesis_8x8_rescue(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["full_gate_level_coverage"] / PHASE_RESCUE_DIR)
    reports_dir = paths["reports"]
    figures_dir = paths["figures"]

    targets = _load_targets(resolved)
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.extend(_rescue_one_target(target, resolved))

    attempts = pd.DataFrame(rows, columns=ATTEMPT_COLUMNS)
    summary = summarize_rescue_attempts(attempts)
    artifacts = _write_outputs(
        config=resolved,
        output_dir=output_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        attempts=attempts,
        summary=summary,
        started_at=started_at,
    )

    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts={key: str(value) for key, value in artifacts.items()},
    )
    metadata.update(
        {
            "attempt_count": len(attempts),
            "rescued_rows": int(summary["rescued"].astype(bool).sum()) if not summary.empty else 0,
            "target_rows": [
                {
                    "case_name": target.case_name,
                    "subproblem_size": target.subproblem_size,
                    "selection_criterion": target.selection_criterion,
                    "alpha": target.alpha,
                    "epsilon_target": target.epsilon_target,
                    "original_degree": target.original_degree,
                }
                for target in targets
            ],
            "phase_convention": DEFAULT_PHASE_CONVENTION,
            "claim_boundary": (
                "Targeted 8x8 phase-synthesis rescue only; dense selected-subproblem "
                "circuit validation, not full IEEE-scale QSVT execution."
            ),
        }
    )
    write_json(artifacts["metadata_json"], metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: path for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "attempts": attempts,
        "summary": summary,
        "artifacts": artifacts,
    }


def _rescue_one_target(target: RescueTarget, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plan = build_attempt_plan(target, config)
    for attempt_index, attempt in enumerate(plan):
        if attempt_index >= int(config["max_attempts_per_case"]):
            rows.append(
                _skipped_budget_row(
                    target,
                    plan[attempt_index],
                    reason=(
                        "remaining rescue attempts skipped_by_budget after "
                        f"max_attempts_per_case={config['max_attempts_per_case']}"
                    ),
                )
            )
            break
        row = evaluate_rescue_attempt(target, attempt, config)
        rows.append(row)
        if attempt_meets_rescue_criteria(row, config):
            break
    return rows


def build_attempt_plan(target: RescueTarget, config: dict[str, Any]) -> list[AttemptSpec]:
    degree_grid = [int(value) for value in config["degree_grid"]]
    degree_grid = [degree for degree in degree_grid if degree % 2 == 1]
    if int(target.original_degree) not in degree_grid:
        degree_grid = [int(target.original_degree), *degree_grid]

    specs: list[AttemptSpec] = []

    def add(
        *,
        degrees: list[int],
        gamma_multipliers: list[float],
        safeties: list[float],
        mode: str,
        stage: str,
    ) -> None:
        for degree in degrees:
            for gamma_multiplier in gamma_multipliers:
                for safety in safeties:
                    specs.append(
                        AttemptSpec(
                            degree=int(degree),
                            gamma_multiplier=float(gamma_multiplier),
                            target_scale_safety=float(safety),
                            approximation_mode=str(mode),
                            stage=stage,
                        )
                    )

    original = int(target.original_degree)
    low_degrees = sorted(set([original, 11]))
    add(
        degrees=[original],
        gamma_multipliers=[1.0],
        safeties=[1.0],
        mode="actual_singular_weighted",
        stage="baseline_reproduction",
    )
    add(
        degrees=low_degrees,
        gamma_multipliers=[1.0],
        safeties=[0.95, 0.90, 0.85],
        mode="actual_singular_weighted",
        stage="low_risk_scaling_rescue",
    )
    add(
        degrees=[11],
        gamma_multipliers=[1.1, 1.25, 1.5, 2.0],
        safeties=[0.95, 0.90],
        mode="actual_singular_weighted",
        stage="gamma_over_normalized_rescue",
    )
    add(
        degrees=[11, 15],
        gamma_multipliers=[1.0],
        safeties=[0.95, 0.90],
        mode="dense_grid_uniform",
        stage="dense_grid_uniform_probe",
    )
    add(
        degrees=[degree for degree in degree_grid if degree <= 201],
        gamma_multipliers=[1.0],
        safeties=[0.95, 0.90, 0.85],
        mode="actual_singular_weighted",
        stage="higher_degree_rescue",
    )
    add(
        degrees=[degree for degree in degree_grid if degree > 201],
        gamma_multipliers=[1.0],
        safeties=[0.80, 0.70, 0.60, 0.50],
        mode="actual_singular_weighted",
        stage="aggressive_rescue",
    )
    add(
        degrees=[15, 21, 25, 31, 35],
        gamma_multipliers=[1.1, 1.25, 1.5, 2.0],
        safeties=[0.95, 0.90, 0.80],
        mode="actual_singular_weighted",
        stage="actual_singular_weighted_gamma_search",
    )
    return _deduplicate_attempts(specs)


def evaluate_rescue_attempt(
    target: RescueTarget,
    attempt: AttemptSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    row = _empty_attempt_row(target, attempt)
    try:
        subproblem = load_sweep_subproblem(
            {
                "case_name": target.case_name,
                "subproblem_size": target.subproblem_size,
                "selection_mode": target.selection_criterion,
            },
            seed=int(config["seed"]),
        )
        A = np.asarray(subproblem.H_tilde, dtype=np.float64)
        b = np.asarray(subproblem.r_tilde, dtype=np.float64)
        singular_values = np.linalg.svd(A, compute_uv=False)
        gamma_base = float(np.max(singular_values[singular_values > SMALL_TOL]))
        gamma = gamma_base * float(attempt.gamma_multiplier)
        build = build_rescue_polynomial(
            singular_values=singular_values,
            alpha=target.alpha,
            gamma=gamma,
            degree=attempt.degree,
            target_scale_safety=attempt.target_scale_safety,
            approximation_mode=attempt.approximation_mode,
            dense_grid_size=int(config["dense_grid_size"]),
        )
        row.update(
            {
                "gamma": gamma,
                "parity_check_status": build.parity_check_status,
                "parity_violation_norm": build.parity_violation_norm,
                "max_abs_target_dense_grid": build.max_abs_dense,
                "admissibility_margin": 1.0 - build.max_abs_dense,
                "dense_grid_error": build.dense_grid_error,
                "actual_singular_value_error": build.actual_singular_value_error,
                "backend_parameters": _backend_parameters(
                    config=config,
                    attempt=attempt,
                    C_alpha=build.C_alpha,
                    effective_C_alpha=build.effective_C_alpha,
                    coefficient_scale=build.coefficient_scale,
                    raw_max_abs_dense=build.raw_max_abs_dense,
                ),
            }
        )
        if build.max_abs_dense > 1.0 + float(config["admissibility_tolerance"]):
            row.update(
                {
                    "phase_synthesis_status": "skipped_admissibility_failure",
                    "qsvt_circuit_status": "skipped_admissibility_failure",
                    "simulation_status": "skipped_admissibility_failure",
                    "rescue_status": "attempt_failed",
                    "failure_mode": "admissibility_failure",
                    "failure_message_short": (
                        "polynomial exceeds unit-domain bound after target scaling"
                    ),
                }
            )
            return _finish_row(row, started)

        phase_started = time.perf_counter()
        phase_result = synthesize_qsvt_phases(
            build.coefficients,
            angle_solver=str(config["angle_solver"]),
        )
        synthesis_runtime = time.perf_counter() - phase_started
        row.update(
            {
                "phase_synthesis_status": phase_result.status,
                "phase_count": int(phase_result.phases.size),
                "synthesis_runtime_seconds": float(synthesis_runtime),
            }
        )
        if phase_result.status != "completed":
            row.update(
                {
                    "qsvt_circuit_status": "skipped_phase_synthesis_failed",
                    "simulation_status": "skipped_phase_synthesis_failed",
                    "rescue_status": "attempt_failed",
                    "failure_mode": _failure_mode(phase_result),
                    "failure_message_short": _short_message(phase_result.failure_reason),
                }
            )
            return _finish_row(row, started)

        encoding = construct_padded_block_encoding(A, gamma=gamma)
        evaluation = evaluate_qsvt_transform(
            run_type="phase_synthesis_8x8_rescue",
            case_name=target.case_name,
            subproblem_size=target.subproblem_size,
            A=A,
            b=b,
            A_bar_padded=encoding.A_bar_padded,
            U_A=encoding.U,
            gamma=gamma,
            C_alpha=build.effective_C_alpha,
            alpha=target.alpha,
            epsilon_target=target.epsilon_target,
            degree=attempt.degree,
            polynomial=build.polynomial,
            phases=phase_result.phases,
            phase_result=phase_result,
            basis_gates=list(config["basis_gates"]),
            transpile_qubit_limit=int(config["transpile_qubit_limit"]),
            transpile_optimization_level=int(config["transpile_optimization_level"]),
        )
        _update_from_evaluation(
            row=row,
            evaluation_row=evaluation.row,
            A=A,
            b=b,
            transformed=evaluation.transformed_block,
            C_alpha=build.effective_C_alpha,
            alpha=target.alpha,
        )
    except Exception as exc:  # pragma: no cover - defensive/resource branch
        row.update(
            {
                "phase_synthesis_status": (
                    row["phase_synthesis_status"]
                    if row["phase_synthesis_status"] != "not_attempted"
                    else "not_completed"
                ),
                "qsvt_circuit_status": "failed",
                "simulation_status": "failed",
                "rescue_status": "attempt_failed",
                "failure_mode": f"{type(exc).__name__}",
                "failure_message_short": _short_message(str(exc)),
            }
        )
    row["rescue_status"] = (
        "rescued" if attempt_meets_rescue_criteria(row, config) else "attempt_failed"
    )
    if row["rescue_status"] == "attempt_failed" and not str(row["failure_mode"]):
        row["failure_mode"] = "criteria_not_met"
    return _finish_row(row, started)


def build_rescue_polynomial(
    *,
    singular_values: np.ndarray,
    alpha: float,
    gamma: float,
    degree: int,
    target_scale_safety: float,
    approximation_mode: str,
    dense_grid_size: int,
) -> PolynomialBuild:
    if not (0.0 < float(target_scale_safety) <= 1.0):
        raise ValueError("target_scale_safety must be in (0, 1]")
    if approximation_mode == "dense_grid_uniform":
        cheb, _coefficients, C_alpha = fit_bounded_ridge_polynomial(
            alpha=float(alpha),
            beta=float(gamma),
            degree=int(degree),
        )
    elif approximation_mode == "actual_singular_weighted":
        cheb, C_alpha = fit_actual_singular_interpolating_polynomial(
            alpha=float(alpha),
            gamma=float(gamma),
            singular_values=np.asarray(singular_values, dtype=np.float64),
            degree=int(degree),
        )
    else:
        raise ValueError(f"unknown approximation_mode: {approximation_mode}")

    monomial = cheb.convert(kind=Polynomial).coef
    coefficients, parity_violation = enforce_odd_parity(monomial, int(degree))
    raw_polynomial = Polynomial(coefficients)
    unit_grid = np.linspace(-1.0, 1.0, max(int(dense_grid_size), int(degree) * 16 + 1))
    raw_max_abs = float(np.max(np.abs(raw_polynomial(unit_grid))))
    coefficient_scale = target_scaling_factor(
        raw_max_abs_dense=raw_max_abs,
        target_scale_safety=float(target_scale_safety),
    )
    scaled_coefficients = coefficients * coefficient_scale
    polynomial = Polynomial(scaled_coefficients)
    max_abs = float(np.max(np.abs(polynomial(unit_grid))))
    effective_C_alpha = float(C_alpha) / max(float(coefficient_scale), SMALL_TOL)
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > SMALL_TOL]
    normalized = positive / float(gamma)
    physical_target = positive / (positive**2 + float(alpha))
    actual_values = effective_C_alpha * polynomial(normalized)
    actual_error = (
        float(np.max(np.abs(actual_values - physical_target))) if physical_target.size else 0.0
    )
    dense_positive = np.linspace(0.0, 1.0, max(int(dense_grid_size), int(degree) * 16 + 1))
    dense_physical_target = (float(gamma) * dense_positive) / (
        (float(gamma) * dense_positive) ** 2 + float(alpha)
    )
    dense_values = effective_C_alpha * polynomial(dense_positive)
    dense_error = float(np.max(np.abs(dense_values - dense_physical_target)))
    parity_status = "passed" if parity_violation <= 1.0e-12 else "repaired_to_odd"
    return PolynomialBuild(
        polynomial=polynomial,
        coefficients=np.asarray(polynomial.coef, dtype=np.float64),
        C_alpha=float(C_alpha),
        effective_C_alpha=effective_C_alpha,
        coefficient_scale=float(coefficient_scale),
        raw_max_abs_dense=raw_max_abs,
        max_abs_dense=max_abs,
        dense_grid_error=dense_error,
        actual_singular_value_error=actual_error,
        parity_violation_norm=float(parity_violation),
        parity_check_status=parity_status,
    )


def enforce_odd_parity(coefficients: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.size < int(degree) + 1:
        values = np.pad(values, (0, int(degree) + 1 - values.size))
    values = values[: int(degree) + 1].copy()
    even = values[0::2].copy()
    values[0::2] = 0.0
    values[np.abs(values) < 1.0e-14] = 0.0
    return values, float(np.linalg.norm(even))


def target_scaling_factor(*, raw_max_abs_dense: float, target_scale_safety: float) -> float:
    raw = float(raw_max_abs_dense)
    safety = float(target_scale_safety)
    if raw <= 0.0 or not np.isfinite(raw):
        return safety
    return float(safety / max(raw, 1.0))


def normalized_singular_values_with_gamma(
    singular_values: np.ndarray,
    *,
    gamma_base: float,
    gamma_multiplier: float,
) -> np.ndarray:
    gamma = float(gamma_base) * float(gamma_multiplier)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    return np.asarray(singular_values, dtype=np.float64) / gamma


def attempt_meets_rescue_criteria(row: dict[str, Any], config: dict[str, Any]) -> bool:
    phase_ok = str(row.get("phase_synthesis_status")) == "completed"
    circuit_ok = str(row.get("qsvt_circuit_status")) == "completed"
    simulation_ok = str(row.get("simulation_status")) == "completed"
    transform_ok = _finite_leq(
        row.get("circuit_vs_polynomial_fro_error"),
        float(config["transform_tolerance"]),
    )
    relative_ok = _finite_leq(
        row.get("circuit_vs_ridge_relative_update_error"),
        float(config["relative_update_tolerance"]),
    )
    absolute_ok = _finite_leq(
        row.get("absolute_update_error"),
        float(config["absolute_update_tolerance"]),
    ) and _finite_leq(row.get("residual_gap"), float(config["residual_gap_tolerance"]))
    return bool(
        phase_ok and circuit_ok and simulation_ok and transform_ok and (relative_ok or absolute_ok)
    )


def summarize_rescue_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (case_name, size), group in attempts.groupby(["case_name", "subproblem_size"]):
        rescued = group[group["rescue_status"] == "rescued"]
        if not rescued.empty:
            best = rescued.iloc[0]
            interpretation = (
                "remediated_by_admissible_target_scaling; failure was not a "
                "circuit-convention or normalization error"
            )
            failure_mode = ""
            rescued_flag = True
        else:
            best = _best_failed_attempt(group)
            interpretation = (
                "still an 8x8 dense-circuit phase-synthesis/admissibility boundary "
                "under the tested rescue configurations"
            )
            failure_mode = str(best.get("failure_mode", "criteria_not_met"))
            rescued_flag = False
        rows.append(
            {
                "case_name": str(case_name),
                "subproblem_size": int(size),
                "rescued": bool(rescued_flag),
                "best_degree": int(best.get("degree", 0)),
                "best_gamma_multiplier": float(best.get("gamma_multiplier", np.nan)),
                "best_target_scale_safety": float(best.get("target_scale_safety", np.nan)),
                "best_backend": str(best.get("backend_name", "")),
                "best_relative_update_error": _numeric(
                    best.get("circuit_vs_ridge_relative_update_error")
                ),
                "best_absolute_update_error": _numeric(best.get("absolute_update_error")),
                "best_residual_gap": _numeric(best.get("residual_gap")),
                "best_success_probability": _numeric(best.get("success_probability")),
                "best_phase_count": int(_numeric(best.get("phase_count")) or 0),
                "best_failure_mode_if_unrescued": failure_mode,
                "recommended_interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _best_failed_attempt(group: pd.DataFrame) -> pd.Series:
    frame = group.copy()
    for column in [
        "circuit_vs_ridge_relative_update_error",
        "residual_gap",
        "actual_singular_value_error",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = frame[frame["circuit_vs_ridge_relative_update_error"].notna()]
    if finite.empty:
        return frame.iloc[-1]
    return finite.sort_values(
        ["circuit_vs_ridge_relative_update_error", "residual_gap", "actual_singular_value_error"],
        na_position="last",
    ).iloc[0]


def _update_from_evaluation(
    *,
    row: dict[str, Any],
    evaluation_row: dict[str, Any],
    A: np.ndarray,
    b: np.ndarray,
    transformed: np.ndarray | None,
    C_alpha: float,
    alpha: float,
) -> None:
    ridge_update = ridge_update_svd(A, b, alpha=float(alpha))
    if transformed is None:
        qsvt_update = np.full_like(ridge_update, np.nan)
    else:
        qsvt_update = qsvt_rescaled_update_from_transform(
            np.asarray(transformed[: A.shape[0], : A.shape[1]], dtype=np.float64),
            b,
            C_alpha=float(C_alpha),
        )
    residual_ridge = float(np.linalg.norm(A @ ridge_update - b))
    residual_qsvt = float(np.linalg.norm(A @ qsvt_update - b))
    residual_base = max(float(np.linalg.norm(b)), SMALL_TOL)
    row.update(
        {
            "qsvt_circuit_status": evaluation_row.get("qsvt_sequence_status", "not_completed"),
            "simulation_status": evaluation_row.get("simulation_status", "not_completed"),
            "transpilation_status": evaluation_row.get("transpilation_status", "not_attempted"),
            "num_qubits": evaluation_row.get("num_qubits", np.nan),
            "raw_depth": evaluation_row.get("raw_circuit_depth", np.nan),
            "transpiled_depth": evaluation_row.get("transpiled_depth", np.nan),
            "transpiled_cx_count": evaluation_row.get("transpiled_cx_count", np.nan),
            "circuit_vs_polynomial_fro_error": evaluation_row.get(
                "circuit_vs_polynomial_fro_error",
                np.nan,
            ),
            "circuit_vs_polynomial_spectral_error": evaluation_row.get(
                "circuit_vs_polynomial_spectral_error",
                np.nan,
            ),
            "circuit_vs_ridge_relative_update_error": evaluation_row.get(
                "relative_update_error",
                np.nan,
            ),
            "absolute_update_error": evaluation_row.get("absolute_update_error", np.nan),
            "ridge_update_norm": float(np.linalg.norm(ridge_update)),
            "qsvt_update_norm": float(np.linalg.norm(qsvt_update)),
            "residual_gap": abs(residual_qsvt / residual_base - residual_ridge / residual_base),
            "ridge_residual_norm": residual_ridge,
            "qsvt_residual_norm": residual_qsvt,
            "success_probability": evaluation_row.get(
                "success_probability_residual_state",
                np.nan,
            ),
            "failure_message_short": _short_message(
                str(evaluation_row.get("failure_or_skip_reason", ""))
            ),
        }
    )


def _load_targets(config: dict[str, Any]) -> list[RescueTarget]:
    if config.get("targets"):
        return [_target_from_mapping(mapping) for mapping in config["targets"]]
    forensic_path = Path(config["forensic_rows_path"])
    if forensic_path.exists():
        frame = pd.read_csv(forensic_path)
        wanted = {("ieee14", 8), ("ieee57", 8)}
        targets = []
        for row in frame.itertuples(index=False):
            key = (str(row.case_name), int(row.subproblem_size))
            if key in wanted:
                targets.append(
                    RescueTarget(
                        case_name=str(row.case_name),
                        subproblem_size=int(row.subproblem_size),
                        selection_criterion=str(row.selection_criterion),
                        alpha=float(row.alpha),
                        epsilon_target=float(row.epsilon_target),
                        original_degree=int(row.degree),
                    )
                )
        if targets:
            return targets
    return [
        RescueTarget("ieee14", 8, "high_leverage", 1.0e-2, 1.0e-2, 5),
        RescueTarget("ieee57", 8, "high_leverage", 1.0e-2, 1.0e-2, 5),
    ]


def _target_from_mapping(mapping: dict[str, Any]) -> RescueTarget:
    return RescueTarget(
        case_name=str(mapping["case_name"]),
        subproblem_size=int(mapping["subproblem_size"]),
        selection_criterion=str(mapping.get("selection_criterion", "high_leverage")),
        alpha=float(mapping.get("alpha", 1.0e-2)),
        epsilon_target=float(mapping.get("epsilon_target", 1.0e-2)),
        original_degree=int(mapping.get("original_degree", mapping.get("degree", 5))),
    )


def _empty_attempt_row(target: RescueTarget, attempt: AttemptSpec) -> dict[str, Any]:
    row = {column: np.nan for column in ATTEMPT_COLUMNS}
    row.update(
        {
            "case_name": target.case_name,
            "subproblem_size": int(target.subproblem_size),
            "alpha": float(target.alpha),
            "epsilon_target": float(target.epsilon_target),
            "degree": int(attempt.degree),
            "gamma_multiplier": float(attempt.gamma_multiplier),
            "target_scale_safety": float(attempt.target_scale_safety),
            "approximation_mode": str(attempt.approximation_mode),
            "polynomial_parity": "odd",
            "parity_check_status": "not_checked",
            "backend_name": "pennylane_poly_to_angles",
            "backend_parameters": json.dumps(
                {
                    "stage": attempt.stage,
                    "angle_solver": "root-finding",
                    "phase_convention": DEFAULT_PHASE_CONVENTION,
                },
                sort_keys=True,
            ),
            "phase_synthesis_status": "not_attempted",
            "phase_count": 0,
            "synthesis_runtime_seconds": 0.0,
            "qsvt_circuit_status": "not_attempted",
            "simulation_status": "not_attempted",
            "transpilation_status": "not_attempted",
            "rescue_status": "not_attempted",
            "failure_mode": "",
            "failure_message_short": "",
            "runtime_seconds": 0.0,
        }
    )
    return row


def _skipped_budget_row(
    target: RescueTarget,
    attempt: AttemptSpec,
    *,
    reason: str,
) -> dict[str, Any]:
    row = _empty_attempt_row(target, attempt)
    row.update(
        {
            "phase_synthesis_status": "not_attempted",
            "qsvt_circuit_status": "skipped_by_budget",
            "simulation_status": "skipped_by_budget",
            "transpilation_status": "skipped_by_budget",
            "rescue_status": "skipped_by_budget",
            "failure_mode": "skipped_by_budget",
            "failure_message_short": reason,
        }
    )
    return row


def _finish_row(row: dict[str, Any], started: float) -> dict[str, Any]:
    row["runtime_seconds"] = float(time.perf_counter() - started)
    return row


def _failure_mode(phase_result: PhaseSynthesisResult) -> str:
    message = str(phase_result.failure_reason)
    if "polynomial must satisfy" in message or "|P(x)|" in message:
        return "admissibility_failure"
    if "parity" in message.lower():
        return "parity_mismatch"
    if "timeout" in message.lower():
        return "timeout"
    return "numerical_failure"


def _backend_parameters(
    *,
    config: dict[str, Any],
    attempt: AttemptSpec,
    C_alpha: float,
    effective_C_alpha: float,
    coefficient_scale: float,
    raw_max_abs_dense: float,
) -> str:
    return json.dumps(
        {
            "stage": attempt.stage,
            "angle_solver": config["angle_solver"],
            "phase_convention": DEFAULT_PHASE_CONVENTION,
            "C_alpha_before_rescaling": float(C_alpha),
            "effective_C_alpha_after_target_contraction": float(effective_C_alpha),
            "coefficient_contraction_factor": float(coefficient_scale),
            "raw_max_abs_before_contraction": float(raw_max_abs_dense),
            "transpile_qubit_limit": int(config["transpile_qubit_limit"]),
        },
        sort_keys=True,
    )


def _write_outputs(
    *,
    config: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    attempts: pd.DataFrame,
    summary: pd.DataFrame,
    started_at: str,
) -> dict[str, Path]:
    artifacts = {
        "attempts_csv": output_dir / "phase_synthesis_8x8_rescue_attempts.csv",
        "summary_csv": output_dir / "phase_synthesis_8x8_rescue_summary.csv",
        "metadata_json": output_dir / "phase_synthesis_8x8_rescue_metadata.json",
        "report": reports_dir / "phase_synthesis_8x8_rescue_report.md",
        "error_figure": figures_dir / "figure_8x8_rescue_error_vs_degree.png",
        "admissibility_figure": figures_dir / "figure_8x8_rescue_admissibility_margin.png",
        "success_probability_figure": figures_dir / "figure_8x8_rescue_success_probability.png",
    }
    attempts.to_csv(artifacts["attempts_csv"], index=False)
    summary.to_csv(artifacts["summary_csv"], index=False)
    _plot_error_vs_degree(attempts, artifacts["error_figure"])
    _plot_admissibility(attempts, artifacts["admissibility_figure"])
    _plot_success_probability(attempts, artifacts["success_probability_figure"])
    artifacts["report"].write_text(
        _report_markdown(
            config=config,
            attempts=attempts,
            summary=summary,
            artifacts=artifacts,
            started_at=started_at,
        ),
        encoding="utf-8",
    )
    return artifacts


def _plot_error_vs_degree(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    completed = frame[
        pd.to_numeric(frame["circuit_vs_ridge_relative_update_error"], errors="coerce").notna()
    ]
    if completed.empty:
        ax.text(0.5, 0.5, "No completed rescue attempts", ha="center", va="center")
    else:
        for case_name, group in completed.groupby("case_name"):
            ax.plot(
                group["degree"].astype(float),
                _positive_for_log(group["circuit_vs_ridge_relative_update_error"]),
                marker="o",
                linestyle="-",
                label=str(case_name),
            )
        ax.axhline(1.0e-2, color="k", linestyle="--", linewidth=1.0, label="1e-2 criterion")
        ax.set_yscale("log")
        ax.set_xlabel("degree")
        ax.set_ylabel("relative update error")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.legend(frameon=False)
    ax.set_title("8x8 Rescue Relative Update Error vs Degree")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_admissibility(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if frame.empty:
        ax.text(0.5, 0.5, "No rescue attempts", ha="center", va="center")
    else:
        labels = [f"{row.case_name}-{int(row.degree)}" for row in frame.itertuples(index=False)]
        x = np.arange(len(frame))
        ax.bar(x, pd.to_numeric(frame["admissibility_margin"], errors="coerce").fillna(0.0))
        ax.axhline(0.0, color="k", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("1 - max |p(x)| on dense unit grid")
        ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("8x8 Rescue Admissibility Margin")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_success_probability(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    completed = frame[pd.to_numeric(frame["success_probability"], errors="coerce").notna()]
    if completed.empty:
        ax.text(0.5, 0.5, "No simulated rescue attempts", ha="center", va="center")
    else:
        labels = [f"{row.case_name}-{int(row.degree)}" for row in completed.itertuples(index=False)]
        x = np.arange(len(completed))
        ax.bar(x, pd.to_numeric(completed["success_probability"], errors="coerce"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("residual-state success probability")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("8x8 Rescue Success Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _report_markdown(
    *,
    config: dict[str, Any],
    attempts: pd.DataFrame,
    summary: pd.DataFrame,
    artifacts: dict[str, Path],
    started_at: str,
) -> str:
    phase_success = int((attempts["phase_synthesis_status"] == "completed").sum())
    simulations = int((attempts["simulation_status"] == "completed").sum())
    rescued = int(summary["rescued"].astype(bool).sum()) if not summary.empty else 0
    status_lines = []
    for row in summary.itertuples(index=False):
        status = "rescued" if bool(row.rescued) else "unrescued"
        status_lines.append(
            f"- {row.case_name} {int(row.subproblem_size)}x{int(row.subproblem_size)}: "
            f"{status}; best degree={int(row.best_degree)}, "
            f"relative update error={_format_float(row.best_relative_update_error)}, "
            f"residual gap={_format_float(row.best_residual_gap)}."
        )
    if not status_lines:
        status_lines = ["- No target rows were evaluated."]
    return "\n".join(
        [
            "# 8x8 Phase-Synthesis Rescue Report",
            "",
            "## Goal",
            "",
            "This targeted pass attempts to rescue the unresolved IEEE14 8x8 and IEEE57 "
            "8x8 integrated gate-level QSVT rows by improving target admissibility, "
            "gamma normalization, odd-parity polynomial construction, and phase-synthesis "
            "settings while keeping the matched Ridge/Tikhonov alpha fixed.",
            "",
            "## Configuration",
            "",
            f"- Command: `{current_command()}`",
            f"- Started at: `{started_at}`",
            f"- Degree grid: `{config['degree_grid']}`",
            f"- Target scale safety grid: `{config['target_scale_safety_grid']}`",
            f"- Gamma multiplier grid: `{config['gamma_multiplier_grid']}`",
            f"- Angle solver: `{config['angle_solver']}`",
            f"- Transpile qubit limit: `{config['transpile_qubit_limit']}`",
            "",
            "## Method",
            "",
            "- The physical alpha is not changed for primary rescue attempts.",
            "- The odd polynomial is fitted in the repository's existing convention.",
            "- If the fitted polynomial exceeds the QSVT unit-domain bound, coefficients "
            "are contracted to the requested safety cap and the final update is rescaled "
            "by the inverse contraction. This records an implementation overhead rather "
            "than changing the matched Ridge/Tikhonov reference.",
            "- Dense block encodings are reconstructed for each gamma multiplier.",
            "",
            "## Results",
            "",
            f"- Attempts recorded: {len(attempts)}",
            f"- Phase synthesis successes: {phase_success}",
            f"- Simulations completed: {simulations}",
            f"- Rescued rows: {rescued} of {len(summary)}",
            *status_lines,
            "",
            "## Claim-Safe Interpretation",
            "",
            "The previous 8x8 gate-level mismatches were audited with admissible target "
            "contraction and phase-synthesis retries. If a row is rescued, this indicates "
            "that the failure was not a circuit-convention or normalization error. "
            "Unrescued rows remain dense-circuit phase-synthesis/admissibility boundaries "
            "under the tested configurations.",
            "",
            "This does not demonstrate full IEEE-scale QSVT execution, quantum speedup, "
            "hardware execution, scalable sparse-oracle implementation, or QSVT numerical "
            "superiority over Ridge/Tikhonov.",
            "",
            "## Artifacts",
            "",
            f"- Attempts CSV: `{artifacts['attempts_csv']}`",
            f"- Summary CSV: `{artifacts['summary_csv']}`",
            f"- Error figure: `{artifacts['error_figure']}`",
            f"- Admissibility figure: `{artifacts['admissibility_figure']}`",
            f"- Success probability figure: `{artifacts['success_probability_figure']}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    root = Path(OUTPUT_ROOT)
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "seed": 123,
        "forensic_rows_path": str(
            root / FULL_GATE_LEVEL_COVERAGE_DIR / "full_gate_level_qsvt_forensic_flagged_rows.csv"
        ),
        "degree_grid": [
            11,
            15,
            21,
            25,
            31,
            35,
            41,
            51,
            61,
            75,
            91,
            101,
            121,
            151,
            201,
            251,
            301,
            401,
            501,
        ],
        "target_scale_safety_grid": [0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50],
        "gamma_multiplier_grid": [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0],
        "dense_grid_size": 4097,
        "angle_solver": "root-finding",
        "basis_gates": DEFAULT_BASIS_GATES,
        "transpile_qubit_limit": 3,
        "transpile_optimization_level": 1,
        "admissibility_tolerance": 1.0e-9,
        "transform_tolerance": 1.0e-10,
        "relative_update_tolerance": 1.0e-2,
        "absolute_update_tolerance": 1.0e-6,
        "residual_gap_tolerance": 1.0e-2,
        "max_attempts_per_case": 80,
        "targets": None,
    }
    if config:
        resolved.update(config)
    resolved["degree_grid"] = [int(value) for value in resolved["degree_grid"]]
    resolved["target_scale_safety_grid"] = [
        float(value) for value in resolved["target_scale_safety_grid"]
    ]
    resolved["gamma_multiplier_grid"] = [
        float(value) for value in resolved["gamma_multiplier_grid"]
    ]
    resolved["basis_gates"] = [str(value) for value in resolved["basis_gates"]]
    return resolved


def _deduplicate_attempts(specs: list[AttemptSpec]) -> list[AttemptSpec]:
    seen: set[tuple[int, float, float, str]] = set()
    out: list[AttemptSpec] = []
    for spec in specs:
        key = (
            int(spec.degree),
            round(float(spec.gamma_multiplier), 12),
            round(float(spec.target_scale_safety), 12),
            str(spec.approximation_mode),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _finite_leq(value: Any, threshold: float) -> bool:
    numeric = _numeric(value)
    return bool(np.isfinite(numeric) and numeric <= float(threshold))


def _numeric(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if np.isfinite(numeric) else np.nan


def _positive_for_log(values: pd.Series) -> np.ndarray:
    return np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 1.0e-18)


def _short_message(message: str, limit: int = 220) -> str:
    text = " ".join(str(message).split())
    return text[:limit]


def _format_float(value: Any) -> str:
    numeric = _numeric(value)
    return "n/a" if not np.isfinite(numeric) else f"{numeric:.3e}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run 8x8 phase-synthesis rescue audit")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--max-attempts-per-case", type=int, default=None)
    args = parser.parse_args(argv)
    overrides: dict[str, Any] = {"output_root": args.output_root}
    if args.max_attempts_per_case is not None:
        overrides["max_attempts_per_case"] = int(args.max_attempts_per_case)
    run = run_phase_synthesis_8x8_rescue(overrides)
    print(f"Wrote 8x8 phase-synthesis rescue outputs to {run['output_dir']}")


if __name__ == "__main__":
    main()
