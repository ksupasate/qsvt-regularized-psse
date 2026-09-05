from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.bounded_target_designs import _build_normalized_problem
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_recovery_from_amplitude import (
    DEPLOYABLE_NORM_METHODS,
    DIAGNOSTIC_NORM_METHODS,
    METHOD_ALIASES,
    apply_norm_recovery,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.residual_feasible_config_search import (
    _build_design_map,
    _design_action,
)
from robust_qsvt_se.utils.io import ensure_directory

CLAIM = (
    "Residual-feasibility test of QSVT update recovery using actual amplitude/norm-estimation "
    "routines on a selected IEEE14-derived subproblem. Direction comes from the bounded "
    "polynomial-action output per target design; the scale comes from amplitude estimation. "
    "Deployable feasibility uses only shot-based and iterative-QAE recovery; exact-success and "
    "best-scalar rows are non-deployable diagnostics flagged as diagnostic_feasible. "
    "Ridge/Tikhonov remains the reference; no QSVT superiority over Ridge/Tikhonov, quantum "
    "speedup, quantum advantage, or full IEEE-scale gate-level solver is claimed."
)

RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
DIRECTION_ERROR_FEASIBLE_MAX = 0.1

DEFAULT_TARGET_DESIGNS = (
    "current_global",
    "support_scaled",
    "margin_1p05",
    "margin_1p10",
    "clipped_inverse_like",
    "degree_adaptive",
)
DEFAULT_METHODS = (
    "exact_success_amplitude_diagnostic",
    "bernoulli_success_amplitude",
    "iterative_qae_if_available",
    "hybrid_known_C_amplitude",
    "best_scalar_upper_bound",
)
TARGET_DESIGN_ALIASES = {
    "current_global": "global_safe",
    "support_scaled": "support_scaled",
    "margin_1p05": "margin_1.05",
    "margin_1p10": "margin_1.1",
    "clipped_inverse_like": "clipped_inverse_like",
    "degree_adaptive": "degree_adaptive",
}
MARGIN_FACTORS = (1.05, 1.10)

RESULT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "norm_recovery_method",
    "shots",
    "condition_number",
    "residual_after_recovery",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "direction_error_vs_ridge",
    "norm_error_vs_ridge",
    "scale_factor_error_vs_best_scalar",
    "success_probability_estimate",
    "standard_error",
    "qsvt_query_count",
    "readout_query_cost",
    "total_query_cost_proxy",
    "residual_feasible",
    "diagnostic_feasible",
    "gate_validation_recommended",
    "rejection_reason",
]


def run_residual_feasible_with_amplitude_recovery(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "target_designs": list(DEFAULT_TARGET_DESIGNS),
        "norm_recovery_methods": list(DEFAULT_METHODS),
        "shots": [100, 1000, 10000],
        "grover_powers": [0, 1, 2, 4, 8],
        "grid_size": 4096,
        "gain_cap_factor": 0.5,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_residual_feasible_with_amplitude_recovery",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    rows = evaluate_amplitude_recovery_search(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        target_designs=[str(value) for value in resolved["target_designs"]],
        methods=[str(value) for value in resolved["norm_recovery_methods"]],
        shot_levels=[int(value) for value in resolved["shots"]],
        grover_powers=tuple(int(value) for value in resolved["grover_powers"]),
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        grid_size=int(resolved["grid_size"]),
        gain_cap_factor=float(resolved["gain_cap_factor"]),
        seed=int(resolved["seed"]),
    )
    rows = _mark_gate_validation_recommended(rows)
    artifacts = write_amplitude_recovery_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_amplitude_recovery_search(
    *,
    subproblem: Any,
    alphas: list[float],
    degrees: list[int],
    target_designs: list[str],
    methods: list[str],
    shot_levels: list[int],
    grover_powers: tuple[int, ...] = (0, 1, 2, 4, 8),
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
    gain_cap_factor: float = 0.5,
    seed: int = 123,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    condition_number = _condition_number(H)
    no_update = float(np.linalg.norm(r))
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
        ridge_norm = float(np.linalg.norm(ridge_update))
        ridge_residual = float(np.linalg.norm(H @ ridge_update - r))
        try:
            problem = _build_normalized_problem(subproblem, alpha=float(alpha))
        except Exception as exc:
            rows.extend(
                _failure_rows(
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    alpha=float(alpha),
                    degrees=degrees,
                    target_designs=target_designs,
                    methods=methods,
                    shot_levels=shot_levels,
                    condition_number=condition_number,
                    reason=f"normalization_failed:{type(exc).__name__}",
                )
            )
            continue
        for degree in degrees:
            design_map = _build_design_map(
                problem=problem,
                degree=int(degree),
                grid_size=int(grid_size),
                gain_cap_factor=float(gain_cap_factor),
            )
            for target_design in target_designs:
                design = design_map.get(TARGET_DESIGN_ALIASES.get(target_design, target_design))
                if design is None:
                    rows.extend(
                        _failure_rows(
                            case=case,
                            model=model,
                            subproblem_id=subproblem_id,
                            alpha=float(alpha),
                            degrees=[int(degree)],
                            target_designs=[target_design],
                            methods=methods,
                            shot_levels=shot_levels,
                            condition_number=condition_number,
                            reason="unknown_target_design",
                        )
                    )
                    continue
                action = _design_action(problem=problem, design=design)
                unit = np.asarray(action.unit_direction, dtype=np.float64)
                direction_error = _direction_error(action.physical_known_c_update, ridge_update)
                best_scalar = best_scalar_for_residual(H, unit, r)
                for shots in shot_levels:
                    for raw_method in methods:
                        method = METHOD_ALIASES.get(str(raw_method), str(raw_method))
                        rows.append(
                            _evaluate_row(
                                method=method,
                                action=action,
                                unit=unit,
                                best_scalar=best_scalar,
                                direction_error=direction_error,
                                H=H,
                                r=r,
                                ridge_update=ridge_update,
                                ridge_norm=ridge_norm,
                                ridge_residual=ridge_residual,
                                no_update=no_update,
                                condition_number=condition_number,
                                alpha=float(alpha),
                                degree=int(degree),
                                target_design=target_design,
                                shots=int(shots),
                                grover_powers=grover_powers,
                                case=case,
                                model=model,
                                subproblem_id=subproblem_id,
                                seed=int(seed),
                            )
                        )
    return rows


def _evaluate_row(
    *,
    method: str,
    action: Any,
    unit: np.ndarray,
    best_scalar: float,
    direction_error: float,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
    ridge_norm: float,
    ridge_residual: float,
    no_update: float,
    condition_number: float,
    alpha: float,
    degree: int,
    target_design: str,
    shots: int,
    grover_powers: tuple[int, ...],
    case: str,
    model: str,
    subproblem_id: str,
    seed: int,
) -> dict[str, Any]:
    recovery = apply_norm_recovery(
        method,
        unit=unit,
        direction_norm=action.direction_norm,
        scale_factor_C=action.C_design,
        beta=action.beta,
        H=H,
        r=r,
        ridge_update=ridge_update,
        shots=int(shots),
        seed=int(seed) + int(shots) + int(degree),
        grover_powers=grover_powers,
    )
    update = recovery.estimated_update
    if update.size and np.all(np.isfinite(update)):
        residual = float(np.linalg.norm(H @ update - r))
    else:
        residual = float("nan")
    residual_ratio_no_update = residual / no_update if no_update > 0.0 else float("nan")
    residual_ratio_ridge = residual / ridge_residual if ridge_residual > 0.0 else float("nan")
    norm_error = (
        abs(recovery.estimated_update_norm - ridge_norm) / max(ridge_norm, 1.0e-15)
        if math.isfinite(recovery.estimated_update_norm)
        else float("nan")
    )
    scale_error = (
        abs(recovery.scale_factor - best_scalar)
        if math.isfinite(recovery.scale_factor)
        else float("nan")
    )
    qsvt_query_count = 2 * int(degree) + 1
    grover_multiplier = _grover_multiplier(method, recovery.uses_qae, grover_powers)
    readout_query_cost = float(int(shots) * grover_multiplier) if recovery.uses_shots else 0.0
    total_query_cost = (
        float(qsvt_query_count) * max(readout_query_cost, 1.0)
        if recovery.uses_shots
        else float(qsvt_query_count)
    )

    residual_feasible, diagnostic_feasible, rejection = _decide_feasibility(
        method=method,
        residual_ratio_no_update=residual_ratio_no_update,
        direction_error=direction_error,
        routine_status=recovery.routine_status,
    )
    return {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_design": target_design,
        "norm_recovery_method": method,
        "shots": int(shots),
        "condition_number": float(condition_number),
        "residual_after_recovery": float(residual),
        "residual_ratio_vs_no_update": float(residual_ratio_no_update),
        "residual_ratio_vs_ridge": float(residual_ratio_ridge),
        "direction_error_vs_ridge": float(direction_error),
        "norm_error_vs_ridge": float(norm_error),
        "scale_factor_error_vs_best_scalar": float(scale_error),
        "success_probability_estimate": float(recovery.probability_estimate),
        "standard_error": float(recovery.probability_standard_error),
        "qsvt_query_count": int(qsvt_query_count),
        "readout_query_cost": float(readout_query_cost),
        "total_query_cost_proxy": float(total_query_cost),
        "residual_feasible": bool(residual_feasible),
        "diagnostic_feasible": bool(diagnostic_feasible),
        "gate_validation_recommended": False,
        "rejection_reason": rejection,
    }


def _decide_feasibility(
    *,
    method: str,
    residual_ratio_no_update: float,
    direction_error: float,
    routine_status: str,
) -> tuple[bool, bool, str]:
    if routine_status != "succeeded":
        return False, False, f"routine_{routine_status}"
    if not math.isfinite(residual_ratio_no_update):
        return False, False, "non_finite_residual_ratio"
    if not math.isfinite(direction_error):
        return False, False, "non_finite_direction_error"
    passes = (
        residual_ratio_no_update <= RESIDUAL_RATIO_FEASIBLE_MAX
        and direction_error <= DIRECTION_ERROR_FEASIBLE_MAX
    )
    if not passes:
        if residual_ratio_no_update > RESIDUAL_RATIO_FEASIBLE_MAX:
            reason = (
                f"residual_ratio_vs_no_update_{residual_ratio_no_update:.3g}_exceeds_"
                f"{RESIDUAL_RATIO_FEASIBLE_MAX:g}"
            )
        else:
            reason = (
                f"direction_error_vs_ridge_{direction_error:.3g}_exceeds_"
                f"{DIRECTION_ERROR_FEASIBLE_MAX:g}"
            )
        return False, False, reason
    if method in DEPLOYABLE_NORM_METHODS:
        return True, False, ""
    if method in DIAGNOSTIC_NORM_METHODS:
        return False, True, "diagnostic_only_method_not_deployable"
    return False, False, "unknown_method"


def _mark_gate_validation_recommended(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in rows if bool(row.get("residual_feasible"))]
    feasible.sort(
        key=lambda row: (
            float(row.get("residual_ratio_vs_no_update", float("inf"))),
            float(row.get("direction_error_vs_ridge", float("inf"))),
            int(row.get("degree", 0)),
        )
    )
    for row in feasible[:3]:
        row["gate_validation_recommended"] = True
    return rows


def write_amplitude_recovery_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, RESULT_COLUMNS)
    feasible = frame[frame["residual_feasible"] == True].copy()  # noqa: E712
    diagnostic = frame[frame["diagnostic_feasible"] == True].copy()  # noqa: E712
    rejected = frame[
        (frame["residual_feasible"] != True)  # noqa: E712
        & (frame["diagnostic_feasible"] != True)  # noqa: E712
    ].copy()

    all_path = output_dir / "all_amplitude_recovery_configs.csv"
    feasible_path = output_dir / "residual_feasible_configs.csv"
    diagnostic_path = output_dir / "diagnostic_feasible_configs.csv"
    rejected_path = output_dir / "rejected_configs.csv"
    interpretation_path = output_dir / "residual_feasible_with_amplitude_interpretation.md"

    frame.to_csv(all_path, index=False)
    feasible.to_csv(feasible_path, index=False)
    diagnostic.to_csv(diagnostic_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    interpretation_path.write_text(amplitude_recovery_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "all_amplitude_recovery_configs": str(all_path),
            "residual_feasible_configs": str(feasible_path),
            "diagnostic_feasible_configs": str(diagnostic_path),
            "rejected_configs": str(rejected_path),
            "residual_feasible_with_amplitude_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "all_amplitude_recovery_configs": all_path,
        "residual_feasible_configs": feasible_path,
        "diagnostic_feasible_configs": diagnostic_path,
        "rejected_configs": rejected_path,
        "residual_feasible_with_amplitude_interpretation": interpretation_path,
    }


def amplitude_recovery_interpretation(frame: pd.DataFrame) -> str:
    total = len(frame)
    feasible = int((frame["residual_feasible"] == True).sum())  # noqa: E712
    diagnostic = int((frame["diagnostic_feasible"] == True).sum())  # noqa: E712
    rejected = total - feasible - diagnostic
    deployable = frame[frame["norm_recovery_method"].isin(DEPLOYABLE_NORM_METHODS)]
    best_deployable_ratio = (
        float(deployable["residual_ratio_vs_no_update"].astype(float).min())
        if not deployable.empty
        else float("nan")
    )
    best_direction = float(frame["direction_error_vs_ridge"].astype(float).min())
    scale_limited = best_direction <= DIRECTION_ERROR_FEASIBLE_MAX and feasible == 0

    if feasible > 0:
        conclusion = (
            "Actual amplitude/norm recovery produces at least one deployable residual-feasible "
            "configuration on the selected IEEE14-derived subproblem."
        )
    elif diagnostic > 0:
        conclusion = (
            "Only non-deployable diagnostic methods reach the feasibility thresholds; no "
            "deployable amplitude-recovered configuration is residual-feasible."
        )
    else:
        conclusion = (
            "No configuration reaches the residual-feasibility thresholds, even with diagnostics."
        )
    if scale_limited:
        bottleneck = (
            "scale recovery is not the binding constraint magnitude-wise once direction agrees; "
            "the residual gap is set by the polynomial target/direction, not amplitude sampling"
        )
    elif best_direction > DIRECTION_ERROR_FEASIBLE_MAX:
        bottleneck = "direction error vs Ridge (the bounded QSVT output points away from Ridge)"
    else:
        bottleneck = "readout/scale sampling under deployable amplitude estimation"

    reasons = (
        frame[frame["residual_feasible"] != True]["rejection_reason"]  # noqa: E712
        .astype(str)
        .replace("", np.nan)
        .dropna()
    )
    top_reasons = sorted(
        reasons.value_counts().to_dict().items(), key=lambda item: (-item[1], item[0])
    )[:5]
    no_feasible_verdict = (
        "yes (none deployable or diagnostic)" if feasible == 0 and diagnostic == 0 else "no"
    )
    return "\n".join(
        [
            "# Residual-Feasible Search With Amplitude Recovery",
            "",
            CLAIM,
            "",
            "## Counts",
            f"- Configurations tested: {total}",
            f"- Deployable residual-feasible configurations: {feasible}",
            f"- Diagnostic-feasible configurations: {diagnostic}",
            f"- Rejected configurations: {rejected}",
            "",
            "## Main Rejection Reasons",
            *(
                [f"- {reason}: {count}" for reason, count in top_reasons]
                if top_reasons
                else ["- None recorded."]
            ),
            "",
            "## Required Answers",
            f"1. Does actual amplitude/norm recovery create any deployable residual-feasible "
            f"configuration? {'yes' if feasible > 0 else 'no'}.",
            f"2. Does it only create diagnostic-feasible configurations? "
            f"{'yes' if feasible == 0 and diagnostic > 0 else 'no'}.",
            f"3. Do no feasible configurations remain? {no_feasible_verdict}.",
            f"4. Bottleneck: {bottleneck} "
            f"(best deployable residual_ratio_vs_no_update = {best_deployable_ratio:.3g}, "
            f"best direction error vs Ridge = {best_direction:.3g}).",
            "",
            "## Conclusion",
            f"- {conclusion}",
            "",
        ]
    )


def _failure_rows(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    alpha: float,
    degrees: list[int],
    target_designs: list[str],
    methods: list[str],
    shot_levels: list[int],
    condition_number: float,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for degree in degrees:
        for target_design in target_designs:
            for shots in shot_levels:
                for raw_method in methods:
                    method = METHOD_ALIASES.get(str(raw_method), str(raw_method))
                    row = {column: np.nan for column in RESULT_COLUMNS}
                    row.update(
                        {
                            "case": case,
                            "model": model,
                            "subproblem_id": subproblem_id,
                            "alpha": float(alpha),
                            "degree": int(degree),
                            "target_design": target_design,
                            "norm_recovery_method": method,
                            "shots": int(shots),
                            "condition_number": float(condition_number),
                            "qsvt_query_count": 2 * int(degree) + 1,
                            "readout_query_cost": 0.0,
                            "total_query_cost_proxy": float(2 * int(degree) + 1),
                            "residual_feasible": False,
                            "diagnostic_feasible": False,
                            "gate_validation_recommended": False,
                            "rejection_reason": reason,
                        }
                    )
                    rows.append(row)
    return rows


def _grover_multiplier(method: str, uses_qae: bool, powers: tuple[int, ...]) -> int:
    if uses_qae:
        return int(sum(2 * int(power) + 1 for power in powers))
    if method in {"bernoulli_success_amplitude", "hybrid_known_C_amplitude"}:
        return 1
    return 1


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _condition_number(H: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(H, dtype=np.float64), compute_uv=False)
    if singular_values.size == 0:
        return float("nan")
    sigma_min = float(np.min(singular_values))
    return float(np.max(singular_values) / sigma_min) if sigma_min > 1.0e-14 else float("inf")


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if candidate_norm <= 1.0e-30 or reference_norm <= 1.0e-30:
        return float("nan")
    return float(np.linalg.norm(candidate / candidate_norm - reference / reference_norm))
