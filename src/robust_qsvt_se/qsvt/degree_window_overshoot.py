from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_bounded_targets import (
    QSVT_SAFE_TOLERANCE,
    SUCCESS_PROBABILITY_FLOOR,
    build_codesigned_solution,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.utils.io import ensure_directory

DEGREE_WINDOW_CLAIM = (
    "Degree-vs-overshoot window analysis for co-designed QSVT-safe bounded targets on a "
    "selected IEEE14-derived subproblem. It maps the low-to-moderate odd-degree window in "
    "which a singular-support-weighted bounded target stays QSVT-safe, Ridge-aligned, and "
    "residual-feasible under deployable amplitude/known-C scale recovery, and locates the "
    "degree at which off-support overshoot degrades the Ridge direction. Ridge/Tikhonov "
    "remains the reference filter; QSVT is an implementation pathway for the same "
    "regularized spectral filter. No QSVT superiority over Ridge/Tikhonov, quantum speedup, "
    "quantum advantage, full IEEE-scale gate-level solving, or hardware execution is claimed."
)

DEFAULT_DEGREES = (15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49)
DEFAULT_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_TARGET_FAMILIES = (
    "weighted_support_ls",
    "residual_aware",
    "success_amplitude_aware",
    "degree_scheduled",
)

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
FEASIBLE_RATIO_MAX = 0.1
FEASIBLE_DIRECTION_MAX = 0.1
OVERSHOOT_SUPPORT_FACTOR = 2.0
DIRECTION_BLOWUP_MAX = 0.25

DEGREE_WINDOW_CLASSES = (
    "residual_feasible",
    "direction_aligned_but_not_residual_feasible",
    "overshoot_risk",
    "not_qsvt_safe",
    "low_success_probability",
    "runtime_limited",
)

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "qsvt_safe",
    "max_abs_polynomial_on_grid",
    "overshoot_detected",
    "overshoot_margin",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "success_probability_proxy",
    "amplitude_estimated_success_if_available",
    "qsvt_query_count",
    "gate_validation_recommended",
    "degree_window_class",
    "dominant_limitation",
]

OVERSHOOT_BOUNDARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "target_family",
    "min_feasible_degree",
    "max_feasible_degree",
    "overshoot_onset_degree",
    "best_residual_ratio_vs_no_update",
    "best_residual_degree",
    "feasible_degree_count",
]


def run_qsvt_degree_window_overshoot(config: dict[str, Any]) -> dict[str, Any]:
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
        "output_dir": "outputs/qsvt_degree_window_overshoot",
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
    rows = evaluate_degree_window(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        target_families=[str(value) for value in resolved["target_families"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        grid_size=int(resolved["grid_size"]),
    )
    artifacts = write_degree_window_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_degree_window(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    target_families: list[str],
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    no_update = float(np.linalg.norm(r))
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
        ridge_residual = float(np.linalg.norm(H @ ridge_update - r))
        for degree in degrees:
            for family in target_families:
                rows.append(
                    _evaluate_degree_window_config(
                        subproblem=subproblem,
                        H=H,
                        r=r,
                        ridge_update=ridge_update,
                        ridge_residual=ridge_residual,
                        no_update=no_update,
                        alpha=float(alpha),
                        degree=int(degree),
                        target_family=str(family),
                        case=case,
                        model=model,
                        subproblem_id=subproblem_id,
                        grid_size=int(grid_size),
                    )
                )
    return _mark_gate_validation_recommended(rows)


def _evaluate_degree_window_config(
    *,
    subproblem: SelectedSubproblem,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
    ridge_residual: float,
    no_update: float,
    alpha: float,
    degree: int,
    target_family: str,
    case: str,
    model: str,
    subproblem_id: str,
    grid_size: int,
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
        return _runtime_limited_row(base)

    # The physically-faithful success amplitude makes the deployable amplitude-recovered
    # update identical to the known-C-rescaled update, so the residual is measured on it.
    update = np.asarray(solution.known_c_update, dtype=np.float64)
    residual = (
        float(np.linalg.norm(H @ update - r)) if np.all(np.isfinite(update)) else float("nan")
    )
    residual_ratio_no_update = residual / no_update if no_update > 0.0 else float("nan")
    residual_ratio_ridge = residual / ridge_residual if ridge_residual > 0.0 else float("nan")
    cos_angle = _cos_angle(update, ridge_update)
    direction_error = float(solution.direction_error_vs_ridge)

    raw_max_abs, support_max_abs = _polynomial_extents(solution, H)
    max_abs_on_grid = raw_max_abs / solution.c_scale if solution.c_scale > 0.0 else float("nan")
    overshoot_margin = raw_max_abs / max(support_max_abs, np.finfo(float).eps)
    overshoot_detected = detect_overshoot(
        max_abs_polynomial_on_grid=max_abs_on_grid,
        direction_error_vs_ridge=direction_error,
        overshoot_margin=overshoot_margin,
    )

    success = float(solution.success_probability_proxy)
    deployable = solution.deployability_class in DEPLOYABLE_CLASSES
    degree_window_class = _classify_degree_window(
        qsvt_safe=bool(solution.qsvt_safe),
        deployable=deployable,
        success_probability=success,
        direction_error=direction_error,
        residual_ratio_no_update=residual_ratio_no_update,
        overshoot_detected=overshoot_detected,
    )
    dominant_limitation = _dominant_limitation(
        qsvt_safe=bool(solution.qsvt_safe),
        deployable=deployable,
        success_probability=success,
        direction_error=direction_error,
        residual_ratio_no_update=residual_ratio_no_update,
        overshoot_detected=overshoot_detected,
        degree_window_class=degree_window_class,
    )

    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(base)
    row.update(
        {
            "weighting_scheme": solution.weighting_scheme,
            "qsvt_safe": bool(solution.qsvt_safe),
            "max_abs_polynomial_on_grid": float(max_abs_on_grid),
            "overshoot_detected": bool(overshoot_detected),
            "overshoot_margin": float(overshoot_margin),
            "direction_error_vs_ridge": float(direction_error),
            "cos_angle_vs_ridge": float(cos_angle),
            "residual_ratio_vs_no_update": float(residual_ratio_no_update),
            "residual_ratio_vs_ridge": float(residual_ratio_ridge),
            "success_probability_proxy": float(success),
            "amplitude_estimated_success_if_available": float("nan"),
            "qsvt_query_count": int(degree),
            "gate_validation_recommended": False,
            "degree_window_class": degree_window_class,
            "dominant_limitation": dominant_limitation,
        }
    )
    return row


def detect_overshoot(
    *,
    max_abs_polynomial_on_grid: float,
    direction_error_vs_ridge: float,
    overshoot_margin: float = 1.0,
) -> bool:
    """Flag overshoot when the bounded polynomial breaks the QSVT bound, the off-support
    peak dominates the on-support response, or the Ridge direction collapses."""

    over_bound = (
        math.isfinite(max_abs_polynomial_on_grid)
        and max_abs_polynomial_on_grid > 1.0 + QSVT_SAFE_TOLERANCE
    )
    over_direction = (
        not math.isfinite(direction_error_vs_ridge)
        or direction_error_vs_ridge > DIRECTION_BLOWUP_MAX
    )
    over_support = math.isfinite(overshoot_margin) and overshoot_margin > OVERSHOOT_SUPPORT_FACTOR
    return bool(over_bound or over_direction or over_support)


def _classify_degree_window(
    *,
    qsvt_safe: bool,
    deployable: bool,
    success_probability: float,
    direction_error: float,
    residual_ratio_no_update: float,
    overshoot_detected: bool,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    direction_ok = math.isfinite(direction_error) and direction_error <= FEASIBLE_DIRECTION_MAX
    residual_ok = (
        math.isfinite(residual_ratio_no_update) and residual_ratio_no_update <= FEASIBLE_RATIO_MAX
    )
    success_ok = (
        math.isfinite(success_probability) and success_probability >= SUCCESS_PROBABILITY_FLOOR
    )
    if deployable and direction_ok and residual_ok and success_ok:
        return "residual_feasible"
    if direction_ok and residual_ok and not success_ok:
        return "low_success_probability"
    if overshoot_detected or not direction_ok:
        return "overshoot_risk"
    return "direction_aligned_but_not_residual_feasible"


def _dominant_limitation(
    *,
    qsvt_safe: bool,
    deployable: bool,
    success_probability: float,
    direction_error: float,
    residual_ratio_no_update: float,
    overshoot_detected: bool,
    degree_window_class: str,
) -> str:
    if degree_window_class == "residual_feasible":
        return "none"
    if not qsvt_safe:
        return "not_qsvt_safe"
    if not deployable:
        return "non_deployable_class"
    if degree_window_class == "low_success_probability":
        return "low_success_amplitude"
    if degree_window_class == "overshoot_risk":
        return "off_support_overshoot" if overshoot_detected else "direction_error"
    return "residual_magnitude_or_conditioning"


def _polynomial_extents(solution: Any, H: np.ndarray) -> tuple[float, float]:
    """Return (raw_max_abs over [-1,1], raw_max_abs over the support singular values)."""

    p_raw = solution.raw_polynomial
    grid = np.linspace(-1.0, 1.0, 8193, dtype=np.float64)
    raw_max_abs = float(np.max(np.abs(p_raw(grid))))
    beta = max(float(solution.beta), np.finfo(float).eps)
    singular_values = np.linalg.svd(np.asarray(H, dtype=np.float64).T / beta, compute_uv=False)
    support = singular_values[singular_values > 1.0e-14]
    support_max_abs = float(np.max(np.abs(p_raw(support)))) if support.size else 0.0
    return raw_max_abs, support_max_abs


def _mark_gate_validation_recommended(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["gate_validation_recommended"] = row.get("degree_window_class") == "residual_feasible"
    return rows


def write_degree_window_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    feasible_frame = summary_frame[
        summary_frame["degree_window_class"] == "residual_feasible"
    ].copy()
    boundary_frame = _overshoot_boundary_frame(summary_frame)

    summary_path = output_dir / "degree_window_summary.csv"
    boundary_path = output_dir / "overshoot_boundary.csv"
    feasible_path = output_dir / "feasible_degree_window.csv"
    interpretation_path = output_dir / "degree_window_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    boundary_frame.to_csv(boundary_path, index=False)
    feasible_frame.to_csv(feasible_path, index=False)
    interpretation_path.write_text(
        degree_window_interpretation(summary_frame, boundary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "degree_window_summary": str(summary_path),
            "overshoot_boundary": str(boundary_path),
            "feasible_degree_window": str(feasible_path),
            "degree_window_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=DEGREE_WINDOW_CLAIM,
    )
    return {
        "manifest": manifest,
        "degree_window_summary": summary_path,
        "overshoot_boundary": boundary_path,
        "feasible_degree_window": feasible_path,
        "degree_window_interpretation": interpretation_path,
    }


def _overshoot_boundary_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame(columns=OVERSHOOT_BOUNDARY_COLUMNS)
    numeric = summary_frame.copy()
    numeric["degree"] = pd.to_numeric(numeric["degree"], errors="coerce")
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    group_keys = ["case", "model", "subproblem_id", "alpha", "target_family"]
    for keys, group in numeric.groupby(group_keys, dropna=False):
        feasible = group[group["degree_window_class"] == "residual_feasible"]
        overshoot = group[group["degree_window_class"] == "overshoot_risk"]
        best_degree = float("nan")
        best_ratio = float("nan")
        if not feasible.empty and feasible["residual_ratio_vs_no_update"].notna().any():
            best_row = feasible.loc[feasible["residual_ratio_vs_no_update"].idxmin()]
            best_degree = float(best_row["degree"])
            best_ratio = float(best_row["residual_ratio_vs_no_update"])
        rows.append(
            {
                "case": keys[0],
                "model": keys[1],
                "subproblem_id": keys[2],
                "alpha": float(keys[3]),
                "target_family": keys[4],
                "min_feasible_degree": (
                    float(feasible["degree"].min()) if not feasible.empty else float("nan")
                ),
                "max_feasible_degree": (
                    float(feasible["degree"].max()) if not feasible.empty else float("nan")
                ),
                "overshoot_onset_degree": (
                    float(overshoot["degree"].min()) if not overshoot.empty else float("nan")
                ),
                "best_residual_ratio_vs_no_update": best_ratio,
                "best_residual_degree": best_degree,
                "feasible_degree_count": len(feasible),
            }
        )
    return pd.DataFrame(rows, columns=OVERSHOOT_BOUNDARY_COLUMNS)


def degree_window_interpretation(summary_frame: pd.DataFrame, boundary_frame: pd.DataFrame) -> str:
    if summary_frame.empty:
        return "\n".join(["# Degree-vs-Overshoot Window", "", DEGREE_WINDOW_CLAIM, ""])

    numeric = summary_frame.copy()
    numeric["degree"] = pd.to_numeric(numeric["degree"], errors="coerce")
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    numeric["direction_error_vs_ridge"] = pd.to_numeric(
        numeric["direction_error_vs_ridge"], errors="coerce"
    )
    numeric["success_probability_proxy"] = pd.to_numeric(
        numeric["success_probability_proxy"], errors="coerce"
    )

    feasible = numeric[numeric["degree_window_class"] == "residual_feasible"]
    feasible_min = float(feasible["degree"].min()) if not feasible.empty else float("nan")
    feasible_max = float(feasible["degree"].max()) if not feasible.empty else float("nan")
    overshoot = numeric[numeric["degree_window_class"] == "overshoot_risk"]
    overshoot_onset = float(overshoot["degree"].min()) if not overshoot.empty else float("nan")

    best_line = "none"
    best_ratio = float("nan")
    best_direction = float("nan")
    if not feasible.empty and feasible["residual_ratio_vs_no_update"].notna().any():
        best = feasible.loc[feasible["residual_ratio_vs_no_update"].idxmin()]
        best_ratio = float(best["residual_ratio_vs_no_update"])
        best_direction = float(best["direction_error_vs_ridge"])
        best_line = (
            f"target_family={best['target_family']}, alpha={float(best['alpha']):.3g}, "
            f"degree={int(best['degree'])}, ratio={best_ratio:.6g}, "
            f"direction_error={best_direction:.6g}"
        )

    # Most stable family: largest feasible degree span, tie-broken by feasible count.
    most_stable = "none"
    best_span = -1.0
    for family, group in feasible.groupby("target_family"):
        span = float(group["degree"].max() - group["degree"].min())
        score = span * 100.0 + len(group)
        if score > best_span:
            best_span = score
            most_stable = str(family)

    success_min = (
        float(feasible["success_probability_proxy"].min()) if not feasible.empty else float("nan")
    )
    success_max = (
        float(feasible["success_probability_proxy"].max()) if not feasible.empty else float("nan")
    )

    # Does higher degree help or harm direction alignment?
    direction_trend = "harms"
    completed = numeric[numeric["direction_error_vs_ridge"].notna()]
    if not completed.empty:
        low_band = completed[completed["degree"] <= 25]["direction_error_vs_ridge"].median()
        high_band = completed[completed["degree"] >= 41]["direction_error_vs_ridge"].median()
        if math.isfinite(low_band) and math.isfinite(high_band):
            direction_trend = "improves" if high_band < low_band else "harms"

    recommended = int((summary_frame["gate_validation_recommended"] == True).sum())  # noqa: E712

    return "\n".join(
        [
            "# Degree-vs-Overshoot Window",
            "",
            DEGREE_WINDOW_CLAIM,
            "",
            "## Counts",
            f"- Configurations tested: {len(summary_frame)}",
            f"- Residual-feasible configurations: {len(feasible)}",
            f"- Overshoot-risk configurations: {len(overshoot)}",
            f"- Gate-validation-recommended configurations: {recommended}",
            "",
            "## Required Answers",
            f"1. Exact feasible degree window: [{feasible_min:.0f}, {feasible_max:.0f}] "
            f"(odd degrees).",
            f"2. Most stable target family across degrees: {most_stable}.",
            f"3. Overshoot onset degree (first overshoot_risk): {overshoot_onset:.0f}.",
            f"4. Does higher degree help direction alignment? Higher degree {direction_trend} "
            "direction alignment.",
            f"5. Configurations to gate-validate next: the {recommended} residual-feasible rows "
            "(Phase 2 selects at most five).",
            "",
            "## Best Residual-Feasible Configuration",
            f"- {best_line}",
            f"- Success-probability proxy across the feasible window: "
            f"[{success_min:.4g}, {success_max:.4g}]",
            "",
            "## Per-(alpha, family) Overshoot Boundary",
            f"- See overshoot_boundary.csv ({len(boundary_frame)} group rows): feasible degree "
            "span and overshoot onset per alpha and target family.",
            "",
        ]
    )


def _runtime_limited_row(base: dict[str, Any]) -> dict[str, Any]:
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(base)
    row.update(
        {
            "weighting_scheme": "n/a",
            "qsvt_safe": False,
            "overshoot_detected": False,
            "qsvt_query_count": int(base.get("degree", 0)),
            "gate_validation_recommended": False,
            "degree_window_class": "runtime_limited",
            "dominant_limitation": "target_construction_failed",
        }
    )
    return row


def _cos_angle(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
