from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.bounded_target_designs import _build_normalized_problem
from robust_qsvt_se.qsvt.codesigned_bounded_targets import (
    QSVT_SAFE_TOLERANCE,
    SUCCESS_PROBABILITY_FLOOR,
    build_codesigned_solution,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.weighted_singular_support import SingularSupport, compute_singular_support
from robust_qsvt_se.utils.io import ensure_directory

OVERSHOOT_MECHANISM_CLAIM = (
    "Overshoot-boundary diagnostic for the co-designed QSVT-safe bounded targets on a selected "
    "IEEE-derived subproblem. It refines the degree boundary near the overshoot onset and "
    "explains, per direction, whether the failure at high degree is a boundedness violation, a "
    "dominant off-support peak, or a high-weight singular-support fit error that breaks the "
    "Ridge direction. Because the bounding constant C cancels in the known-C update, the "
    "deployable update direction is set by the polynomial values at the support singular "
    "values, so a high-degree fit that corrupts those values breaks the direction even while "
    "the bounded polynomial stays within the QSVT bound. Ridge/Tikhonov remains the reference "
    "filter; QSVT is an implementation pathway for the same regularized spectral filter. No "
    "QSVT superiority over Ridge/Tikhonov, quantum speedup, quantum advantage, full IEEE-scale "
    "gate-level solving, or hardware execution is claimed."
)

DEFAULT_DEGREES = (41, 43, 45, 47, 49, 51)
DEFAULT_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_TARGET_FAMILIES = ("weighted_support_ls", "residual_aware")

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
FEASIBLE_RATIO_MAX = 0.1
FEASIBLE_DIRECTION_MAX = 0.1
OFF_SUPPORT_MARGIN_FACTOR = 2.0
OFF_SUPPORT_DISTANCE_MIN = 0.02
HIGH_WEIGHT_ERROR_MAX = 0.1

OVERSHOOT_FAILURE_MODES = (
    "no_overshoot_residual_feasible",
    "off_support_peak_but_still_feasible",
    "off_support_peak_breaks_direction",
    "singular_support_error_breaks_direction",
    "boundedness_violation",
    "phase_synthesis_limited",
    "inconclusive",
)

# Modes that count as a genuine overshoot failure (direction/boundedness collapse), as
# opposed to a feasible target or a merely marginal (inconclusive) low-success row.
FEASIBLE_MODES = ("no_overshoot_residual_feasible", "off_support_peak_but_still_feasible")
OVERSHOOT_MODES = (
    "off_support_peak_breaks_direction",
    "singular_support_error_breaks_direction",
    "boundedness_violation",
)

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "qsvt_safe",
    "max_abs_polynomial_on_grid",
    "max_abs_location",
    "nearest_actual_singular_value",
    "off_support_peak_detected",
    "off_support_peak_distance_to_singular_support",
    "high_weight_singular_error",
    "direction_error_vs_ridge",
    "residual_ratio_vs_no_update",
    "overshoot_failure_mode",
    "phase_synthesis_status_if_checked",
    "dominant_limitation",
]

OFF_SUPPORT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "max_abs_polynomial_on_grid",
    "max_abs_location",
    "nearest_actual_singular_value",
    "off_support_peak_detected",
    "off_support_peak_distance_to_singular_support",
    "overshoot_failure_mode",
]

SINGULAR_DIRECTION_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "high_weight_singular_error",
    "direction_error_vs_ridge",
    "residual_ratio_vs_no_update",
    "overshoot_failure_mode",
]


def run_qsvt_overshoot_mechanism_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": list(DEFAULT_ALPHAS),
        "degrees": list(DEFAULT_DEGREES),
        "target_families": list(DEFAULT_TARGET_FAMILIES),
        "grid_size": 4096,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "check_phase_synthesis": False,
        "phase_timeout_seconds": 25,
        "output_dir": "outputs/qsvt_overshoot_mechanism_diagnostic",
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
    rows = evaluate_overshoot_mechanism(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        target_families=[str(value) for value in resolved["target_families"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        grid_size=int(resolved["grid_size"]),
        check_phase_synthesis=bool(resolved["check_phase_synthesis"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
    )
    artifacts = write_overshoot_mechanism_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_overshoot_mechanism(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    target_families: list[str],
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
    check_phase_synthesis: bool = False,
    phase_timeout_seconds: int = 25,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    no_update = float(np.linalg.norm(r))
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        try:
            problem = _build_normalized_problem(subproblem, alpha=float(alpha))
            support = compute_singular_support(H, r, alpha=float(alpha))
        except Exception:
            for degree in degrees:
                for family in target_families:
                    rows.append(
                        _runtime_limited_row(
                            case, model, subproblem_id, float(alpha), int(degree), str(family)
                        )
                    )
            continue
        support_values_A = problem.singular_values_A[problem.singular_values_A > 1.0e-14]
        for degree in degrees:
            for family in target_families:
                rows.append(
                    _evaluate_overshoot_config(
                        subproblem=subproblem,
                        H=H,
                        r=r,
                        no_update=no_update,
                        support_values_A=support_values_A,
                        support=support,
                        alpha=float(alpha),
                        degree=int(degree),
                        target_family=str(family),
                        case=case,
                        model=model,
                        subproblem_id=subproblem_id,
                        grid_size=int(grid_size),
                        check_phase_synthesis=check_phase_synthesis,
                        phase_timeout_seconds=int(phase_timeout_seconds),
                    )
                )
    return rows


def _evaluate_overshoot_config(
    *,
    subproblem: SelectedSubproblem,
    H: np.ndarray,
    r: np.ndarray,
    no_update: float,
    support_values_A: np.ndarray,
    support: SingularSupport,
    alpha: float,
    degree: int,
    target_family: str,
    case: str,
    model: str,
    subproblem_id: str,
    grid_size: int,
    check_phase_synthesis: bool,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    base = {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_family": target_family,
    }
    solution = build_codesigned_solution(
        subproblem,
        alpha=float(alpha),
        degree=int(degree),
        target_family=target_family,
        grid_size=int(grid_size),
    )
    if solution is None:
        return _runtime_limited_row(case, model, subproblem_id, alpha, degree, target_family)

    p_raw = solution.raw_polynomial
    beta = max(float(solution.beta), np.finfo(float).eps)
    c_scale = max(float(solution.c_scale), np.finfo(float).eps)

    peak = detect_off_support_peak(p_raw, support_values_A)
    max_abs_on_grid = float(peak["max_abs_raw"] / c_scale)
    # The finer diagnostic grid can catch a boundedness breach at extreme degree that the
    # solution's coarser bounding grid under-sampled; keep qsvt_safe consistent with it.
    qsvt_safe = bool(solution.qsvt_safe) and (max_abs_on_grid <= 1.0 + QSVT_SAFE_TOLERANCE)

    update = np.asarray(solution.known_c_update, dtype=np.float64)
    residual = (
        float(np.linalg.norm(H @ update - r)) if np.all(np.isfinite(update)) else float("nan")
    )
    residual_ratio = residual / no_update if no_update > 0.0 else float("nan")
    direction_error = float(solution.direction_error_vs_ridge)
    success = float(solution.success_probability_proxy)
    deployable = solution.deployability_class in DEPLOYABLE_CLASSES
    high_weight_error = _high_weight_singular_error(p_raw, beta, support)

    phase_status = "not_checked"
    phase_failed = False
    if check_phase_synthesis:
        phase_status = _phase_synthesis_status(
            H, r, alpha, solution, phase_timeout_seconds=int(phase_timeout_seconds)
        )
        phase_failed = phase_status not in {"synthesized", "not_checked"}

    failure_mode = classify_overshoot_failure_mode(
        qsvt_safe=qsvt_safe,
        deployable=deployable,
        success_probability=success,
        direction_error=direction_error,
        residual_ratio_no_update=residual_ratio,
        off_support_peak_detected=bool(peak["detected"]),
        high_weight_singular_error=high_weight_error,
        phase_synthesis_failed=phase_failed,
    )

    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(base)
    row.update(
        {
            "qsvt_safe": qsvt_safe,
            "max_abs_polynomial_on_grid": max_abs_on_grid,
            "max_abs_location": float(peak["location"]),
            "nearest_actual_singular_value": float(peak["nearest_singular_value"]),
            "off_support_peak_detected": bool(peak["detected"]),
            "off_support_peak_distance_to_singular_support": float(peak["distance"]),
            "high_weight_singular_error": float(high_weight_error),
            "direction_error_vs_ridge": float(direction_error),
            "residual_ratio_vs_no_update": float(residual_ratio),
            "overshoot_failure_mode": failure_mode,
            "phase_synthesis_status_if_checked": phase_status,
            "dominant_limitation": _dominant_limitation(failure_mode, success),
        }
    )
    return row


def detect_off_support_peak(
    p_raw: Any, support_values_A: np.ndarray, *, grid_points: int = 16385
) -> dict[str, Any]:
    """Locate the dominant peak of |p_raw| on [-1, 1] and decide whether it is off support.

    The peak is "off support" when it sits away from every actual (normalized) singular value
    and dominates the on-support response by more than ``OFF_SUPPORT_MARGIN_FACTOR``. The
    largest normalized singular value is at the x=+/-1 boundary, so a boundary peak is treated
    as on-support rather than as a genuine off-support overshoot.
    """

    support = np.asarray(support_values_A, dtype=np.float64)
    support = support[support > 1.0e-14]
    grid = np.linspace(-1.0, 1.0, int(grid_points), dtype=np.float64)
    magnitude = np.abs(np.asarray(p_raw(grid), dtype=np.float64))
    peak_index = int(np.argmax(magnitude))
    location = float(grid[peak_index])
    max_abs_raw = float(magnitude[peak_index])
    support_max_abs = (
        float(np.max(np.abs(np.asarray(p_raw(support), dtype=np.float64)))) if support.size else 0.0
    )
    margin = max_abs_raw / max(support_max_abs, np.finfo(float).eps)
    if support.size:
        distance = float(np.min(np.abs(abs(location) - support)))
        nearest = float(support[int(np.argmin(np.abs(abs(location) - support)))])
    else:
        distance = float("nan")
        nearest = float("nan")
    detected = bool(
        support.size and distance > OFF_SUPPORT_DISTANCE_MIN and margin > OFF_SUPPORT_MARGIN_FACTOR
    )
    return {
        "detected": detected,
        "distance": distance,
        "location": location,
        "nearest_singular_value": nearest,
        "margin": float(margin),
        "max_abs_raw": max_abs_raw,
        "support_max_abs": support_max_abs,
    }


def classify_overshoot_failure_mode(
    *,
    qsvt_safe: bool,
    deployable: bool,
    success_probability: float,
    direction_error: float,
    residual_ratio_no_update: float,
    off_support_peak_detected: bool,
    high_weight_singular_error: float,
    phase_synthesis_failed: bool = False,
) -> str:
    if phase_synthesis_failed:
        return "phase_synthesis_limited"
    direction_ok = math.isfinite(direction_error) and direction_error <= FEASIBLE_DIRECTION_MAX
    residual_ok = (
        math.isfinite(residual_ratio_no_update) and residual_ratio_no_update <= FEASIBLE_RATIO_MAX
    )
    success_ok = (
        math.isfinite(success_probability) and success_probability >= SUCCESS_PROBABILITY_FLOOR
    )
    feasible = qsvt_safe and deployable and direction_ok and residual_ok and success_ok
    if feasible:
        return (
            "off_support_peak_but_still_feasible"
            if off_support_peak_detected
            else "no_overshoot_residual_feasible"
        )
    if not qsvt_safe:
        return "boundedness_violation"
    if not direction_ok:
        if off_support_peak_detected:
            return "off_support_peak_breaks_direction"
        # The known-C update direction is set by the polynomial values at the support singular
        # values (C cancels), so a broken direction without a dominant off-support peak is a
        # high-weight singular-support fit error.
        return "singular_support_error_breaks_direction"
    return "inconclusive"


def _high_weight_singular_error(p_raw: Any, beta: float, support: SingularSupport) -> float:
    """Relative filter error at the highest-combined-weight support singular direction.

    The physical filter of the bounded QSVT polynomial at sigma is (1/beta) p_raw(sigma/beta);
    this compares it with the Ridge filter sigma/(sigma^2 + alpha) at the most impactful
    direction.
    """

    sigma = np.asarray(support.sigma, dtype=np.float64)
    weights = np.asarray(support.combined_weight, dtype=np.float64)
    ridge_filter = np.asarray(support.ridge_filter_value, dtype=np.float64)
    if sigma.size == 0 or weights.size == 0:
        return float("nan")
    index = int(np.argmax(weights[: sigma.size]))
    physical = float(p_raw(sigma[index] / beta) / beta)
    reference = float(ridge_filter[index])
    if abs(reference) <= 1.0e-30:
        return float("nan")
    return abs(physical - reference) / abs(reference)


def _phase_synthesis_status(
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    solution: Any,
    *,
    phase_timeout_seconds: int,
) -> str:
    from robust_qsvt_se.qsvt.codesigned_gate_validation import _build_codesigned_gate_circuit

    try:
        gate = _build_codesigned_gate_circuit(
            H=H,
            r=r,
            alpha=float(alpha),
            solution=solution,
            phase_timeout_seconds=int(phase_timeout_seconds),
        )
    except Exception as exc:  # pragma: no cover - backend/runtime dependent
        return f"error:{type(exc).__name__}"
    return str(gate.get("status", "runtime_limited"))


def _dominant_limitation(failure_mode: str, success_probability: float) -> str:
    if failure_mode in {"no_overshoot_residual_feasible", "off_support_peak_but_still_feasible"}:
        return "none"
    if failure_mode == "boundedness_violation":
        return "boundedness"
    if failure_mode == "off_support_peak_breaks_direction":
        return "off_support_overshoot"
    if failure_mode == "singular_support_error_breaks_direction":
        return "singular_support_fit_error"
    if failure_mode == "phase_synthesis_limited":
        return "phase_synthesis_runtime_limited"
    if math.isfinite(success_probability) and success_probability < SUCCESS_PROBABILITY_FLOOR:
        return "low_success_amplitude"
    return "residual_magnitude_or_conditioning"


def write_overshoot_mechanism_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    off_support_frame = summary_frame[OFF_SUPPORT_COLUMNS].copy()
    singular_frame = summary_frame[SINGULAR_DIRECTION_COLUMNS].copy()

    summary_path = output_dir / "overshoot_mechanism_summary.csv"
    off_support_path = output_dir / "off_support_peak_locations.csv"
    singular_path = output_dir / "singular_direction_error_near_boundary.csv"
    interpretation_path = output_dir / "overshoot_mechanism_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    off_support_frame.to_csv(off_support_path, index=False)
    singular_frame.to_csv(singular_path, index=False)
    interpretation_path.write_text(
        overshoot_mechanism_interpretation(summary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "overshoot_mechanism_summary": str(summary_path),
            "off_support_peak_locations": str(off_support_path),
            "singular_direction_error_near_boundary": str(singular_path),
            "overshoot_mechanism_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=OVERSHOOT_MECHANISM_CLAIM,
    )
    return {
        "manifest": manifest,
        "overshoot_mechanism_summary": summary_path,
        "off_support_peak_locations": off_support_path,
        "singular_direction_error_near_boundary": singular_path,
        "overshoot_mechanism_interpretation": interpretation_path,
    }


def overshoot_mechanism_interpretation(summary_frame: pd.DataFrame) -> str:
    if summary_frame.empty:
        return "\n".join(["# Overshoot-Mechanism Diagnostic", "", OVERSHOOT_MECHANISM_CLAIM, ""])
    numeric = summary_frame.copy()
    for column in (
        "degree",
        "direction_error_vs_ridge",
        "high_weight_singular_error",
        "residual_ratio_vs_no_update",
        "max_abs_polynomial_on_grid",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    feasible = numeric[numeric["overshoot_failure_mode"].isin(FEASIBLE_MODES)]
    overshoot_failures = numeric[numeric["overshoot_failure_mode"].isin(OVERSHOOT_MODES)]
    overshoot_degrees = {int(value) for value in overshoot_failures["degree"].dropna()}
    onset_degree = min(overshoot_degrees) if overshoot_degrees else None
    # Safest degree: the largest tested degree at which no genuine overshoot failure occurs.
    clean_degrees = {int(value) for value in feasible["degree"].dropna()} - overshoot_degrees
    safest_max_degree = max(clean_degrees) if clean_degrees else None
    mode_counts = (
        numeric["overshoot_failure_mode"].value_counts().to_dict() if not numeric.empty else {}
    )
    boundedness_failures = int((numeric["qsvt_safe"].astype(str).str.lower() == "false").sum())
    off_support_breaks = int(
        (numeric["overshoot_failure_mode"] == "off_support_peak_breaks_direction").sum()
    )
    support_breaks = int(
        (numeric["overshoot_failure_mode"] == "singular_support_error_breaks_direction").sum()
    )

    onset_rows = numeric[numeric["degree"] == onset_degree] if onset_degree is not None else numeric
    onset_dir = (
        float(onset_rows["direction_error_vs_ridge"].max())
        if not onset_rows.empty
        else float("nan")
    )
    onset_hw = (
        float(onset_rows["high_weight_singular_error"].max())
        if not onset_rows.empty
        else float("nan")
    )
    onset_off_support = bool(
        (onset_rows["off_support_peak_detected"].astype(str).str.lower() == "true").any()
    )
    dominant_mode = (
        "singular_support_error_breaks_direction"
        if support_breaks >= off_support_breaks
        else "off_support_peak_breaks_direction"
    )
    onset_fitted = "no" if onset_hw > HIGH_WEIGHT_ERROR_MAX else "yes"
    off_support_phrase = (
        "a dominant off-support peak is present"
        if onset_off_support
        else "no dominant off-support peak at onset; the failure is on the high-weight "
        "singular support"
    )

    return "\n".join(
        [
            "# Overshoot-Mechanism Diagnostic",
            "",
            OVERSHOOT_MECHANISM_CLAIM,
            "",
            "## Counts",
            f"- Configurations tested: {len(summary_frame)}",
            f"- Failure-mode counts: {mode_counts}",
            "",
            "## Required Answers",
            f"1. Why does the boundary degree (overshoot onset {onset_degree}) fail? "
            f"dominant failure mode = {dominant_mode}; at onset the worst direction error is "
            f"{onset_dir:.4g} and the worst high-weight singular error is {onset_hw:.4g}.",
            f"2. Boundedness vs direction: {boundedness_failures} configurations violate "
            "boundedness (max|p|>1); the bounded polynomial otherwise stays within the QSVT "
            "bound and the failure is loss of Ridge-direction alignment, not boundedness.",
            "3. Are the actual singular values still fitted at the onset degree? "
            f"{onset_fitted} (high-weight singular error "
            f"{onset_hw:.4g} vs threshold {HIGH_WEIGHT_ERROR_MAX}).",
            "4. Is the overshoot off-support or on high-weight singular support? "
            f"{off_support_phrase} "
            f"(off_support_breaks={off_support_breaks}, support_fit_breaks={support_breaks}).",
            f"5. Safest maximum degree to recommend: "
            f"{safest_max_degree if safest_max_degree is not None else 'none'}.",
            "",
            "## Mechanism",
            "- The bounding constant C cancels in the known-C update, so the deployable update "
            "direction depends only on the polynomial values at the support singular values.",
            "- At the onset degree the higher-degree least-squares fit develops oscillations that "
            "corrupt those on-support values (the high-weight singular error jumps), which breaks "
            "the Ridge direction while the bounded polynomial is still within the QSVT bound.",
            "- A dominant off-support peak (and any boundedness stress) appears only at the "
            "highest degrees, after the direction has already degraded.",
            "",
        ]
    )


def _runtime_limited_row(
    case: str, model: str, subproblem_id: str, alpha: float, degree: int, target_family: str
) -> dict[str, Any]:
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "alpha": float(alpha),
            "degree": int(degree),
            "target_family": target_family,
            "qsvt_safe": False,
            "off_support_peak_detected": False,
            "overshoot_failure_mode": "phase_synthesis_limited",
            "phase_synthesis_status_if_checked": "target_construction_failed",
            "dominant_limitation": "target_construction_failed",
        }
    )
    return row


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
