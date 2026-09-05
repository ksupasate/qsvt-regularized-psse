from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.bounded_target_designs import (
    NormalizedProblem,
    TargetDesign,
    _build_normalized_problem,
    _build_target_designs,
    _fit_raw_polynomial,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.norm_success import estimate_success_probability_iterative_proxy
from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    build_selected_subproblem_from_policy_row,
    generate_candidate_subproblems,
    score_candidate_subproblems,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

CLAIM = (
    "Residual-feasible QSVT configuration search over a criteria-selected "
    "IEEE-derived subproblem set. Subproblems are chosen ONLY by the numerical/"
    "metadata selection policy; gate-level QSVT performance is never used as a "
    "selection or feasibility criterion. Feasibility is decided purely from "
    "polynomial-action residual and direction agreement with the Ridge/Tikhonov "
    "reference under deployable or proxy scale-recovery. Ridge/Tikhonov is the "
    "reference filter. Optional ieee30/ieee57 rows are matrix-free condition/"
    "polynomial-residual diagnostics only. No QSVT superiority over Ridge, no "
    "quantum speedup, and no full IEEE-scale gate-level solver is claimed."
)

# Feasibility thresholds (exact per phase spec).
RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
DIRECTION_ERROR_FEASIBLE_MAX = 0.1
GATE_VALIDATION_DEGREE_MAX = 101
GATE_VALIDATION_CONDITION_MAX = 1.0e8
DIAGNOSTIC_ONLY_PROTOCOL = "best_scalar_diagnostic"

# Scaling-only target designs requested by the search. Mapped to Phase 2 design names.
TARGET_DESIGN_ALIASES = {
    "current_global": "global_safe",
    "support_scaled": "support_scaled",
    "margin_1p05": "margin_1.05",
    "margin_1p10": "margin_1.1",
    "degree_adaptive": "degree_adaptive",
}
DEFAULT_TARGET_DESIGNS = (
    "current_global",
    "support_scaled",
    "margin_1p05",
    "margin_1p10",
    "degree_adaptive",
)
DEFAULT_SCALE_PROTOCOLS = (
    "known_C",
    "success_amplitude_proxy",
    "amplitude_estimation_proxy",
    "best_scalar_diagnostic",
)
# Deployable / proxy protocols whose direction agreement governs feasibility.
DEPLOYABLE_PROTOCOLS = (
    "known_C",
    "success_amplitude_proxy",
    "amplitude_estimation_proxy",
)
MATRIX_FREE_PROTOCOL = "matrix_free_polynomial_residual"
MARGIN_FACTORS = (1.05, 1.10)

RESULT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "alpha",
    "degree",
    "target_design",
    "scale_protocol",
    "condition_number",
    "residual",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "norm_error_vs_ridge",
    "scale_recovery_cost_proxy",
    "qsvt_query_count",
    "estimated_total_query_cost",
    "residual_feasible",
    "gate_validation_recommended",
    "rejection_reason",
]


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Purely residual/direction-based feasibility decision (no gate solver)."""

    residual_feasible: bool
    rejection_reason: str


def decide_residual_feasibility(
    *,
    scale_protocol: str,
    residual_ratio_vs_no_update: float,
    direction_error_vs_ridge: float,
) -> FeasibilityVerdict:
    """Feasibility depends ONLY on residual ratio, direction error, and protocol class.

    It never consults gate-level QSVT performance. best_scalar_diagnostic is a
    classical Ridge-referenced diagnostic and can never be residual-feasible.
    """
    if scale_protocol == DIAGNOSTIC_ONLY_PROTOCOL:
        return FeasibilityVerdict(False, "diagnostic_only_protocol_not_deployable")
    if not math.isfinite(residual_ratio_vs_no_update):
        return FeasibilityVerdict(False, "non_finite_residual_ratio")
    if not math.isfinite(direction_error_vs_ridge):
        return FeasibilityVerdict(False, "non_finite_direction_error")
    if residual_ratio_vs_no_update > RESIDUAL_RATIO_FEASIBLE_MAX:
        return FeasibilityVerdict(
            False,
            f"residual_ratio_vs_no_update_{residual_ratio_vs_no_update:.3g}_exceeds_"
            f"{RESIDUAL_RATIO_FEASIBLE_MAX:g}",
        )
    if direction_error_vs_ridge > DIRECTION_ERROR_FEASIBLE_MAX:
        return FeasibilityVerdict(
            False,
            f"direction_error_vs_ridge_{direction_error_vs_ridge:.3g}_exceeds_"
            f"{DIRECTION_ERROR_FEASIBLE_MAX:g}",
        )
    return FeasibilityVerdict(True, "")


def run_qsvt_residual_feasible_config_search(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "target_designs": list(DEFAULT_TARGET_DESIGNS),
        "scale_protocols": list(DEFAULT_SCALE_PROTOCOLS),
        "condition_threshold": 1.0e8,
        "policy_alpha": 1.0e-4,
        "max_selected": 3,
        "grid_size": 4096,
        "gain_cap_factor": 0.5,
        "eps_rel": 1.0e-2,
        "amplitude_max_queries": 1000,
        "seed": 123,
        "include_matrix_free_diagnostics": True,
        "diagnostic_cases": ["ieee30", "ieee57"],
        "output_dir": "outputs/qsvt_residual_feasible_config_search",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    selected = select_policy_subproblems(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        submatrix_size=int(resolved["submatrix_size"]),
        condition_threshold=float(resolved["condition_threshold"]),
        policy_alpha=float(resolved["policy_alpha"]),
        max_selected=int(resolved["max_selected"]),
        seed=int(resolved["seed"]),
    )
    alphas = [float(value) for value in resolved["alphas"]]
    degrees = [int(value) for value in resolved["degrees"]]
    target_designs = [str(value) for value in resolved["target_designs"]]
    scale_protocols = [str(value) for value in resolved["scale_protocols"]]

    rows = evaluate_config_search(
        selected=selected,
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        alphas=alphas,
        degrees=degrees,
        target_designs=target_designs,
        scale_protocols=scale_protocols,
        grid_size=int(resolved["grid_size"]),
        gain_cap_factor=float(resolved["gain_cap_factor"]),
        eps_rel=float(resolved["eps_rel"]),
        amplitude_max_queries=int(resolved["amplitude_max_queries"]),
        seed=int(resolved["seed"]),
    )

    diagnostic_rows: list[dict[str, Any]] = []
    if bool(resolved["include_matrix_free_diagnostics"]):
        diagnostic_rows = evaluate_matrix_free_diagnostics(
            cases=[str(value) for value in resolved["diagnostic_cases"]],
            model=str(resolved["model"]),
            case_source=str(resolved["case_source"]),
            submatrix_size=int(resolved["submatrix_size"]),
            alphas=alphas,
            degrees=degrees,
            grid_size=int(resolved["grid_size"]),
            seed=int(resolved["seed"]),
        )

    rows = _mark_gate_validation_recommended(rows)
    all_rows = rows + diagnostic_rows
    artifacts = write_config_search_outputs(output_dir, resolved, all_rows)
    return {
        "output_dir": output_dir,
        "rows": all_rows,
        "selected_subproblem_count": len(selected),
        "artifacts": artifacts,
    }


def select_policy_subproblems(
    *,
    case: str,
    model: str,
    case_source: str,
    submatrix_size: int,
    condition_threshold: float,
    policy_alpha: float,
    max_selected: int,
    seed: int,
) -> list[SelectedSubproblem]:
    """Select subproblems using ONLY the criteria-based policy (no QSVT residuals)."""
    system, matrix_source = _build_system(
        case=case, model=model, case_source=case_source, seed=int(seed)
    )
    candidates = generate_candidate_subproblems(
        system=system,
        matrix_source=matrix_source,
        case=case,
        model=model,
        submatrix_size=int(submatrix_size),
        candidate_modes=[
            "best_conditioned",
            "high_leverage",
            "metadata_mapped",
            "residual_supported",
            "random_seeded_pool",
            "worst_conditioned_control",
        ],
        seed=int(seed),
    )
    scored = score_candidate_subproblems(
        system=system,
        matrix_source=matrix_source,
        case=case,
        model=model,
        case_source=case_source,
        candidates=candidates,
        condition_threshold=float(condition_threshold),
        alpha=float(policy_alpha),
        max_selected=int(max_selected),
    )
    selected_rows = [row for row in scored if bool(row.get("selected"))]
    return [
        build_selected_subproblem_from_policy_row(
            system=system, matrix_source=matrix_source, row=row
        )
        for row in selected_rows
    ]


def evaluate_config_search(
    *,
    selected: list[SelectedSubproblem],
    case: str,
    model: str,
    alphas: list[float],
    degrees: list[int],
    target_designs: list[str],
    scale_protocols: list[str],
    grid_size: int = 4096,
    gain_cap_factor: float = 0.5,
    eps_rel: float = 1.0e-2,
    amplitude_max_queries: int = 1000,
    seed: int = 123,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subproblem in selected:
        for alpha in alphas:
            for degree in degrees:
                rows.extend(
                    evaluate_config_point(
                        subproblem=subproblem,
                        case=case,
                        model=model,
                        alpha=float(alpha),
                        degree=int(degree),
                        target_designs=target_designs,
                        scale_protocols=scale_protocols,
                        grid_size=int(grid_size),
                        gain_cap_factor=float(gain_cap_factor),
                        eps_rel=float(eps_rel),
                        amplitude_max_queries=int(amplitude_max_queries),
                        seed=int(seed),
                    )
                )
    return rows


def evaluate_config_point(
    *,
    subproblem: SelectedSubproblem,
    case: str,
    model: str,
    alpha: float,
    degree: int,
    target_designs: list[str],
    scale_protocols: list[str],
    grid_size: int = 4096,
    gain_cap_factor: float = 0.5,
    eps_rel: float = 1.0e-2,
    amplitude_max_queries: int = 1000,
    seed: int = 123,
) -> list[dict[str, Any]]:
    subproblem_id = str(subproblem.metadata.get("candidate_id", "selected_subproblem"))
    selection_mode = str(
        subproblem.metadata.get("selection_mode", "selected_subproblems_from_policy")
    )
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    condition_number = _condition_number(H)
    no_update_residual = float(np.linalg.norm(r))
    ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
    ridge_residual = _residual(H, ridge_update, r)

    rows: list[dict[str, Any]] = []
    try:
        problem = _build_normalized_problem(subproblem, alpha=float(alpha))
        design_map = _build_design_map(
            problem=problem,
            degree=int(degree),
            grid_size=int(grid_size),
            gain_cap_factor=float(gain_cap_factor),
        )
    except Exception as exc:
        for design_name in target_designs:
            for protocol in scale_protocols:
                rows.append(
                    _failed_row(
                        case=case,
                        model=model,
                        subproblem_id=subproblem_id,
                        selection_mode=selection_mode,
                        alpha=float(alpha),
                        degree=int(degree),
                        target_design=design_name,
                        scale_protocol=protocol,
                        condition_number=condition_number,
                        rejection_reason=f"normalization_failed:{type(exc).__name__}",
                    )
                )
        return rows

    for design_name in target_designs:
        design = design_map.get(TARGET_DESIGN_ALIASES.get(design_name, design_name))
        if design is None:
            for protocol in scale_protocols:
                rows.append(
                    _failed_row(
                        case=case,
                        model=model,
                        subproblem_id=subproblem_id,
                        selection_mode=selection_mode,
                        alpha=float(alpha),
                        degree=int(degree),
                        target_design=design_name,
                        scale_protocol=protocol,
                        condition_number=condition_number,
                        rejection_reason="unknown_target_design",
                    )
                )
            continue
        action = _design_action(problem=problem, design=design)
        for protocol in scale_protocols:
            rows.append(
                _evaluate_protocol_row(
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    selection_mode=selection_mode,
                    alpha=float(alpha),
                    degree=int(degree),
                    target_design=design_name,
                    scale_protocol=protocol,
                    condition_number=condition_number,
                    action=action,
                    H=problem.H,
                    r=problem.r,
                    ridge_update=ridge_update,
                    ridge_residual=ridge_residual,
                    no_update_residual=no_update_residual,
                    eps_rel=float(eps_rel),
                    amplitude_max_queries=int(amplitude_max_queries),
                    seed=int(seed),
                )
            )
    return rows


@dataclass(frozen=True, slots=True)
class _DesignAction:
    """Bounded QSVT direction for a specific target design plus shared references."""

    unit_direction: np.ndarray
    direction_norm: float
    beta: float
    C_design: float
    physical_known_c_update: np.ndarray
    success_probability_proxy: float


def _build_design_map(
    *,
    problem: NormalizedProblem,
    degree: int,
    grid_size: int,
    gain_cap_factor: float,
) -> dict[str, TargetDesign]:
    base_fit = _fit_raw_polynomial(
        problem,
        degree=int(degree),
        grid_size=int(grid_size),
        domain_min=problem.domain_min,
        instance_aware=False,
    )
    designs = _build_target_designs(
        problem=problem,
        base_fit=base_fit,
        degree=int(degree),
        grid_size=int(grid_size),
        margin_factors=list(MARGIN_FACTORS),
        gain_cap_factor=float(gain_cap_factor),
    )
    return {design.name: design for design in designs}


def _design_action(*, problem: NormalizedProblem, design: TargetDesign) -> _DesignAction:
    sv = problem.singular_values_A
    p_raw = design.fit.polynomial
    C_design = float(design.C_design)
    bounded_filter = p_raw(sv) / C_design
    bounded_operator = problem.U @ (bounded_filter[:, None] * problem.Vh)
    bounded_output = bounded_operator @ problem.r
    bounded_norm = float(np.linalg.norm(bounded_output))
    unit = bounded_output / bounded_norm if bounded_norm > 0.0 else np.zeros_like(bounded_output)
    physical_known_c = rescale_bounded_target_to_original(
        bounded_output, beta=problem.beta, C=C_design
    )
    success_probability_proxy = float(min(1.0, bounded_norm**2))
    return _DesignAction(
        unit_direction=unit,
        direction_norm=bounded_norm,
        beta=problem.beta,
        C_design=C_design,
        physical_known_c_update=physical_known_c,
        success_probability_proxy=success_probability_proxy,
    )


def _protocol_update(
    *,
    protocol: str,
    action: _DesignAction,
    H: np.ndarray,
    r: np.ndarray,
    eps_rel: float,
    amplitude_max_queries: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return (physical update, scale_recovery_cost_proxy) for a scale protocol.

    cost_proxy is the amplitude-amplification query multiplier (1/sqrt(p)) where the
    protocol amplifies a postselected amplitude; it is NaN for protocols that do not.
    """
    beta = action.beta
    C = action.C_design
    unit = action.unit_direction
    p_success = action.success_probability_proxy

    if protocol == "known_C":
        return action.physical_known_c_update, float("nan")
    if protocol == "success_amplitude_proxy":
        magnitude = (C / beta) * math.sqrt(max(p_success, 0.0))
        cost = 1.0 / math.sqrt(max(p_success, 1.0e-12))
        return magnitude * unit, float(cost)
    if protocol == "amplitude_estimation_proxy":
        estimate = estimate_success_probability_iterative_proxy(
            p_success, max(int(amplitude_max_queries), 1), int(seed)
        )
        magnitude = (C / beta) * math.sqrt(max(float(estimate.estimate), 0.0))
        cost = math.ceil(1.0 / (float(eps_rel) * math.sqrt(max(p_success, 1.0e-12))))
        return magnitude * unit, float(cost)
    if protocol == DIAGNOSTIC_ONLY_PROTOCOL:
        scalar = best_scalar_for_residual(H, unit, r)
        return scalar * unit, float("nan")
    raise ValueError(f"unsupported scale protocol: {protocol}")


def _evaluate_protocol_row(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    alpha: float,
    degree: int,
    target_design: str,
    scale_protocol: str,
    condition_number: float,
    action: _DesignAction,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
    ridge_residual: float,
    no_update_residual: float,
    eps_rel: float,
    amplitude_max_queries: int,
    seed: int,
) -> dict[str, Any]:
    update, cost_proxy = _protocol_update(
        protocol=scale_protocol,
        action=action,
        H=H,
        r=r,
        eps_rel=float(eps_rel),
        amplitude_max_queries=int(amplitude_max_queries),
        seed=int(seed),
    )
    if np.all(np.isfinite(update)) and update.size:
        residual = _residual(H, update, r)
    else:
        residual = float("nan")
    residual_ratio_no_update = (
        residual / no_update_residual if no_update_residual > 0.0 else float("nan")
    )
    residual_ratio_ridge = residual / ridge_residual if ridge_residual > 0.0 else float("nan")
    cos_angle = _cos_angle(update, ridge_update)
    direction_error = _direction_error_from_cos(cos_angle)
    norm_error = _norm_error(update, ridge_update)

    query_count = 2 * int(degree) + 1
    if math.isfinite(cost_proxy):
        estimated_total = float(query_count) * float(cost_proxy)
    else:
        estimated_total = float(query_count)

    verdict = decide_residual_feasibility(
        scale_protocol=scale_protocol,
        residual_ratio_vs_no_update=float(residual_ratio_no_update),
        direction_error_vs_ridge=float(direction_error),
    )
    return {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "selection_mode": selection_mode,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_design": target_design,
        "scale_protocol": scale_protocol,
        "condition_number": float(condition_number),
        "residual": float(residual),
        "residual_ratio_vs_no_update": float(residual_ratio_no_update),
        "residual_ratio_vs_ridge": float(residual_ratio_ridge),
        "direction_error_vs_ridge": float(direction_error),
        "cos_angle_vs_ridge": float(cos_angle),
        "norm_error_vs_ridge": float(norm_error),
        "scale_recovery_cost_proxy": float(cost_proxy),
        "qsvt_query_count": int(query_count),
        "estimated_total_query_cost": float(estimated_total),
        "residual_feasible": bool(verdict.residual_feasible),
        "gate_validation_recommended": False,
        "rejection_reason": verdict.rejection_reason,
    }


def evaluate_matrix_free_diagnostics(
    *,
    cases: list[str],
    model: str,
    case_source: str,
    submatrix_size: int,
    alphas: list[float],
    degrees: list[int],
    grid_size: int = 4096,
    seed: int = 123,
) -> list[dict[str, Any]]:
    """Condition-number + polynomial-residual ONLY diagnostics; no gate-level circuits."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            selected = select_policy_subproblems(
                case=case,
                model=model,
                case_source=case_source,
                submatrix_size=int(submatrix_size),
                condition_threshold=1.0e8,
                policy_alpha=1.0e-4,
                max_selected=1,
                seed=int(seed),
            )
        except Exception as exc:
            rows.append(
                _diagnostic_failed_row(
                    case=case,
                    model=model,
                    alpha=float(alphas[0]) if alphas else float("nan"),
                    degree=int(degrees[0]) if degrees else 0,
                    rejection_reason=f"diagnostic_selection_failed:{type(exc).__name__}",
                )
            )
            continue
        if not selected:
            rows.append(
                _diagnostic_failed_row(
                    case=case,
                    model=model,
                    alpha=float(alphas[0]) if alphas else float("nan"),
                    degree=int(degrees[0]) if degrees else 0,
                    rejection_reason="no_policy_selected_subproblem",
                )
            )
            continue
        subproblem = selected[0]
        subproblem_id = str(subproblem.metadata.get("candidate_id", "diagnostic_subproblem"))
        H = np.asarray(subproblem.H_tilde, dtype=np.float64)
        r = np.asarray(subproblem.r_tilde, dtype=np.float64)
        condition_number = _condition_number(H)
        no_update_residual = float(np.linalg.norm(r))
        for alpha in alphas:
            ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
            ridge_residual = _residual(H, ridge_update, r)
            try:
                problem = _build_normalized_problem(subproblem, alpha=float(alpha))
            except Exception as exc:
                rows.append(
                    _diagnostic_failed_row(
                        case=case,
                        model=model,
                        alpha=float(alpha),
                        degree=int(degrees[0]) if degrees else 0,
                        rejection_reason=f"normalization_failed:{type(exc).__name__}",
                        subproblem_id=subproblem_id,
                        condition_number=condition_number,
                    )
                )
                continue
            for degree in degrees:
                base_fit = _fit_raw_polynomial(
                    problem,
                    degree=int(degree),
                    grid_size=int(grid_size),
                    domain_min=problem.domain_min,
                    instance_aware=False,
                )
                sv = problem.singular_values_A
                physical = (1.0 / problem.beta) * (
                    problem.U @ (base_fit.polynomial(sv)[:, None] * problem.Vh) @ problem.r
                )
                residual = _residual(problem.H, physical, problem.r)
                residual_ratio_no_update = (
                    residual / no_update_residual if no_update_residual > 0.0 else float("nan")
                )
                residual_ratio_ridge = (
                    residual / ridge_residual if ridge_residual > 0.0 else float("nan")
                )
                cos_angle = _cos_angle(physical, ridge_update)
                direction_error = _direction_error_from_cos(cos_angle)
                query_count = 2 * int(degree) + 1
                row = {column: float("nan") for column in RESULT_COLUMNS}
                row.update(
                    {
                        "case": case,
                        "model": model,
                        "subproblem_id": subproblem_id,
                        "selection_mode": "matrix_free_diagnostic_from_policy",
                        "alpha": float(alpha),
                        "degree": int(degree),
                        "target_design": "current_global",
                        "scale_protocol": "matrix_free_polynomial_residual",
                        "condition_number": float(condition_number),
                        "residual": float(residual),
                        "residual_ratio_vs_no_update": float(residual_ratio_no_update),
                        "residual_ratio_vs_ridge": float(residual_ratio_ridge),
                        "direction_error_vs_ridge": float(direction_error),
                        "cos_angle_vs_ridge": float(cos_angle),
                        "norm_error_vs_ridge": float(_norm_error(physical, ridge_update)),
                        "scale_recovery_cost_proxy": float("nan"),
                        "qsvt_query_count": int(query_count),
                        "estimated_total_query_cost": float(query_count),
                        "residual_feasible": False,
                        "gate_validation_recommended": False,
                        "rejection_reason": "matrix_free_diagnostic_not_a_deployable_protocol",
                    }
                )
                rows.append(row)
    return rows


def _mark_gate_validation_recommended(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recommend gate validation for the top-3 feasible configs with manageable cost.

    Selection here is among already-residual-feasible rows only and uses residual
    quality, NOT gate-level QSVT output (which has not been computed in this phase).
    """
    feasible = [
        row
        for row in rows
        if bool(row.get("residual_feasible"))
        and int(row.get("degree", 0)) <= GATE_VALIDATION_DEGREE_MAX
        and _is_stable_condition(row.get("condition_number"))
    ]
    feasible.sort(
        key=lambda row: (
            float(row.get("residual_ratio_vs_no_update", float("inf"))),
            float(row.get("direction_error_vs_ridge", float("inf"))),
            int(row.get("degree", 0)),
            str(row.get("subproblem_id", "")),
            str(row.get("target_design", "")),
            str(row.get("scale_protocol", "")),
        )
    )
    for row in feasible[:3]:
        row["gate_validation_recommended"] = True
    return rows


def write_config_search_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, RESULT_COLUMNS)
    feasible_frame = frame[frame["residual_feasible"] == True].copy()  # noqa: E712
    rejected_frame = frame[
        (frame["residual_feasible"] != True)  # noqa: E712
        | (frame["rejection_reason"].astype(str).str.len() > 0)
    ].copy()

    all_path = output_dir / "all_config_results.csv"
    feasible_path = output_dir / "residual_feasible_configs.csv"
    rejected_path = output_dir / "failed_or_rejected_configs.csv"
    interpretation_path = output_dir / "config_search_interpretation.md"

    frame.to_csv(all_path, index=False)
    feasible_frame.to_csv(feasible_path, index=False)
    rejected_frame.to_csv(rejected_path, index=False)
    interpretation_path.write_text(config_search_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "all_config_results": str(all_path),
            "residual_feasible_configs": str(feasible_path),
            "failed_or_rejected_configs": str(rejected_path),
            "config_search_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "all_config_results": all_path,
        "residual_feasible_configs": feasible_path,
        "failed_or_rejected_configs": rejected_path,
        "config_search_interpretation": interpretation_path,
    }


def config_search_interpretation(frame: pd.DataFrame) -> str:
    total = len(frame)
    feasible_mask = frame["residual_feasible"] == True  # noqa: E712
    feasible = int(feasible_mask.sum())
    recommended = int((frame["gate_validation_recommended"] == True).sum())  # noqa: E712
    rejected = int(total - feasible)

    reasons = frame[~feasible_mask]["rejection_reason"].astype(str).replace("", np.nan).dropna()
    reason_counts = reasons.value_counts().to_dict()
    top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]

    direction_limited = _is_direction_limited(frame)
    feasible_rows = frame[feasible_mask]
    if feasible:
        best = feasible_rows.loc[
            feasible_rows["residual_ratio_vs_no_update"].astype(float).idxmin()
        ]
        best_lines = [
            f"- Best feasible residual ratio vs no-update: "
            f"{float(best['residual_ratio_vs_no_update']):.6g}",
            f"- Best feasible config: subproblem={best['subproblem_id']}, "
            f"alpha={float(best['alpha']):.3g}, degree={int(best['degree'])}, "
            f"target_design={best['target_design']}, scale_protocol={best['scale_protocol']}",
            f"- Direction error vs Ridge at best feasible: "
            f"{float(best['direction_error_vs_ridge']):.6g}",
        ]
    else:
        best_lines = [
            "- No residual-feasible configuration was found. This is an honest negative "
            "result: under the criteria-based subproblem selection and deployable/proxy "
            "scale recovery, no configuration met residual_ratio_vs_no_update <= 0.1 with "
            "direction_error_vs_ridge <= 0.1."
        ]

    deployable_errors = (
        frame[frame["scale_protocol"].isin(DEPLOYABLE_PROTOCOLS)]["direction_error_vs_ridge"]
        .astype(float)
        .dropna()
    )
    best_direction_error = (
        float(deployable_errors.min()) if not deployable_errors.empty else float("nan")
    )

    return "\n".join(
        [
            "# Residual-Feasible QSVT Configuration Search",
            "",
            CLAIM,
            "",
            "## Selection Discipline",
            "- Subproblems are selected ONLY by the criteria-based selection policy "
            "(selection_mode=selected_subproblems_from_policy).",
            "- Gate-level QSVT performance is NOT used to select subproblems or to decide "
            "residual feasibility.",
            "",
            "## Counts",
            f"- Configurations tested: {total}",
            f"- Residual-feasible configurations: {feasible}",
            f"- Recommended for gate validation: {recommended}",
            f"- Failed or rejected configurations: {rejected}",
            "",
            "## Main Rejection Reasons",
            *(
                [f"- {reason}: {count}" for reason, count in top_reasons]
                if top_reasons
                else ["- None recorded."]
            ),
            "",
            "## Best Feasible Result",
            *best_lines,
            "",
            "## Direction-Limited Assessment",
            f"- Best (smallest) direction_error_vs_ridge across deployable/proxy protocols: "
            f"{best_direction_error:.6g}.",
            f"- Search is direction-limited: {direction_limited}.",
            "- Direction-limited means the bounded/polynomial QSVT output points away from the "
            "Ridge update, so no scale-recovery magnitude can recover a useful update.",
            "- Magnitude-limited means the direction agrees with Ridge but no deployable scale-"
            "recovery protocol (known_C, success/amplitude proxies) attains a small enough "
            "residual; only the classical best-scalar diagnostic (excluded as non-deployable) "
            "can close the residual.",
            "",
        ]
    )


def _is_direction_limited(frame: pd.DataFrame) -> str:
    deployable = frame[frame["scale_protocol"].isin(DEPLOYABLE_PROTOCOLS)]
    errors = deployable["direction_error_vs_ridge"].astype(float).dropna()
    if errors.empty:
        return "inconclusive"
    best_direction_error = float(errors.min())
    feasible = int((frame["residual_feasible"] == True).sum())  # noqa: E712
    if best_direction_error <= DIRECTION_ERROR_FEASIBLE_MAX and feasible == 0:
        # Direction can match, but no deployable scale recovery achieves a small residual:
        # the binding constraint is magnitude recovery, not direction.
        return (
            f"no, magnitude-limited (deployable direction error reaches "
            f"{best_direction_error:.4g} <= {DIRECTION_ERROR_FEASIBLE_MAX:g}, yet no deployable "
            "scale-recovery protocol attains residual_ratio_vs_no_update <= "
            f"{RESIDUAL_RATIO_FEASIBLE_MAX:g}; the gap is in recovered magnitude/scale)"
        )
    if feasible == 0 and best_direction_error > DIRECTION_ERROR_FEASIBLE_MAX:
        return (
            f"yes (no feasible config and best deployable direction error "
            f"{best_direction_error:.4g} exceeds {DIRECTION_ERROR_FEASIBLE_MAX:g})"
        )
    if best_direction_error > DIRECTION_ERROR_FEASIBLE_MAX:
        return (
            f"partially (some feasible rows exist but best deployable direction error "
            f"{best_direction_error:.4g} still exceeds {DIRECTION_ERROR_FEASIBLE_MAX:g})"
        )
    return f"no (deployable direction error reaches {best_direction_error:.4g})"


def _failed_row(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    alpha: float,
    degree: int,
    target_design: str,
    scale_protocol: str,
    condition_number: float,
    rejection_reason: str,
) -> dict[str, Any]:
    row = {column: float("nan") for column in RESULT_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "selection_mode": selection_mode,
            "alpha": float(alpha),
            "degree": int(degree),
            "target_design": target_design,
            "scale_protocol": scale_protocol,
            "condition_number": float(condition_number),
            "qsvt_query_count": 2 * int(degree) + 1,
            "estimated_total_query_cost": float(2 * int(degree) + 1),
            "residual_feasible": False,
            "gate_validation_recommended": False,
            "rejection_reason": rejection_reason,
        }
    )
    return row


def _diagnostic_failed_row(
    *,
    case: str,
    model: str,
    alpha: float,
    degree: int,
    rejection_reason: str,
    subproblem_id: str = "diagnostic_subproblem",
    condition_number: float = float("nan"),
) -> dict[str, Any]:
    row = {column: float("nan") for column in RESULT_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "selection_mode": "matrix_free_diagnostic_from_policy",
            "alpha": float(alpha),
            "degree": int(degree),
            "target_design": "current_global",
            "scale_protocol": "matrix_free_polynomial_residual",
            "condition_number": float(condition_number),
            "qsvt_query_count": 2 * int(degree) + 1,
            "estimated_total_query_cost": float(2 * int(degree) + 1),
            "residual_feasible": False,
            "gate_validation_recommended": False,
            "rejection_reason": rejection_reason,
        }
    )
    return row


def _is_stable_condition(value: Any) -> bool:
    try:
        condition = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(condition) and condition <= GATE_VALIDATION_CONDITION_MAX


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
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    return sigma_max / sigma_min if sigma_min > 1.0e-14 else float("inf")


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _cos_angle(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def _direction_error_from_cos(cos_angle: float) -> float:
    if not math.isfinite(cos_angle):
        return float("nan")
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - cos_angle))))


def _norm_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    return float(abs(candidate_norm - reference_norm) / max(reference_norm, 1.0e-15))
