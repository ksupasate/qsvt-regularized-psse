"""Task D: degree-aware regularization-parameter (alpha) selection.

The classical Ridge/Tikhonov filter is ``P_alpha(sigma) = sigma / (sigma^2 +
alpha)``; the bounded QSVT-compatible normalized target is
``f_{alpha,bounded}(s) = (1/C) * s / (s^2 + alpha/beta^2)`` on ``s in [0,1]`` with
``beta`` the block-encoding normalization. Smaller alpha can improve inverse-like
behaviour but typically raises the QSVT polynomial degree required to approximate
the target to tolerance. This module quantifies that trade-off and compares
several alpha-selection rules, including the degree-budget-constrained rule

    alpha* = argmin_alpha RMSE(alpha)  s.t.  d(alpha, eps) <= d_max.

The key manuscript-safe interpretation: *the best classical regularization
parameter is not automatically the best QSVT-implementable choice under a degree
budget.* No speedup is claimed; Ridge/Tikhonov remains the matched reference and
the QSVT target uses the same alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.paper.selected_observable_common import (
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
    write_workload_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    bounded_ridge_normalization_C,
    bounded_ridge_target,
    fit_bounded_ridge_polynomial,
    qsvt_odd_degree,
)
from robust_qsvt_se.utils.io import ensure_directory

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57")
DEFAULT_ALPHA_GRID = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
DEFAULT_TOLERANCES = (1.0e-2, 1.0e-3, 1.0e-4)
DEFAULT_DEGREE_BUDGETS = (25, 51, 101, 201)
DEFAULT_ALPHA = 1.0e-4

# Degrees searched to find the required QSVT polynomial degree. Extends past the
# largest budget so "degree_required > budget" is detectable rather than clamped.
SEARCH_DEGREES = (
    5,
    9,
    13,
    17,
    21,
    25,
    31,
    35,
    41,
    45,
    51,
    61,
    71,
    81,
    91,
    101,
    121,
    151,
    181,
    201,
    251,
    301,
    351,
    401,
)
# Feasibility hint only: the repo has demonstrated phase synthesis for selected
# targets up to roughly this degree. This is NOT a per-row phase-synthesis run.
PHASE_FEASIBLE_DEGREE_HINT = 51

# Existing classical alpha-sweep artifact reused as an independent cross-reference.
EXISTING_ALPHA_SWEEP = "outputs/full_alpha_sensitivity_classical/alpha_sweep_summary_by_case.csv"

GRID_COLUMNS = [
    "case",
    "scenario",
    "alpha",
    "tolerance",
    "degree_budget",
    "rmse",
    "residual_norm",
    "condition_number",
    "degree_required",
    "approx_error",
    "target_met",
    "phase_available",
    "degree_budget_met",
    "bounded_scale_C",
    "selection_rule",
    "selected",
    "notes",
]

SUMMARY_COLUMNS = [
    "case",
    "scenario",
    "selection_rule",
    "tolerance",
    "degree_budget",
    "selected_alpha",
    "rmse_at_selected",
    "degree_required_at_selected",
    "target_met",
    "degree_budget_met",
    "phase_available",
    "notes",
]

REVISED_SUMMARY_COLUMNS = [
    "case",
    "scenario",
    "selection_rule",
    "selected_alpha",
    "tolerance",
    "degree_budget",
    "rmse_at_selected",
    "spectrum_point_degree",
    "uniform_grid_degree",
    "phase_synthesis_available",
    "boundedness_verified",
    "parity_verified",
    "qsvt_query_count_realizable",
    "degree_derived_query_count",
    "query_count_status",
    "notes",
]


def _alpha_degree_profile(
    singular_values: np.ndarray, *, alpha: float
) -> tuple[float, list[tuple[int, float]]]:
    """Return ``(C_alpha, [(odd_degree, actual_singular_value_error), ...])``.

    Uses the repository's bounded QSVT-target convention: the odd polynomial
    approximates ``g(s)/C_alpha`` with ``g(s) = beta*s/(beta^2 s^2 + alpha)`` and
    ``C_alpha = max(1, max_{[0,1]} g)``. Error is measured at the matrix's actual
    normalized singular values (the spectrum the QSVT circuit actually transforms),
    matching the repo's degree-selection criterion.
    """

    values = np.asarray(singular_values, dtype=np.float64)
    positive = values[values > 1.0e-14]
    beta = float(positive.max()) if positive.size else 1.0
    normalized = positive / beta
    c_alpha = bounded_ridge_normalization_C(float(alpha), beta)
    target = bounded_ridge_target(normalized, alpha=float(alpha), beta=beta, C_alpha=c_alpha)
    profile: list[tuple[int, float]] = []
    seen: set[int] = set()
    for requested in SEARCH_DEGREES:
        odd_degree, _ = qsvt_odd_degree(int(requested))
        if odd_degree in seen:
            continue
        seen.add(odd_degree)
        polynomial, _coeffs, _c = fit_bounded_ridge_polynomial(
            alpha=float(alpha), beta=beta, degree=odd_degree
        )
        error = float(np.max(np.abs(polynomial(normalized) - target)))
        profile.append((odd_degree, error))
    return c_alpha, profile


def _degree_for_target(
    profile: list[tuple[int, float]], *, tolerance: float
) -> tuple[int | None, float]:
    """Return ``(degree_required, approx_error)`` from a precomputed degree profile.

    ``degree_required`` is the smallest searched odd degree whose bounded-target
    error is at most ``tolerance``; ``None`` if none meet it. ``approx_error`` is the
    error at that degree, or the smallest achieved error if none meet it.
    """

    for degree, error in profile:
        if error <= float(tolerance):
            return int(degree), float(error)
    return None, float(min(error for _degree, error in profile))


def _uniform_profile(
    singular_values: np.ndarray, *, alpha: float, grid_size: int = 4001
) -> list[dict[str, Any]]:
    """Check the fitted odd polynomial on a uniform domain grid.

    This check does not perform phase synthesis. It therefore cannot by itself
    establish a realizable QSVT query count.
    """

    values = np.asarray(singular_values, dtype=np.float64)
    positive = values[values > 1.0e-14]
    beta = float(positive.max()) if positive.size else 1.0
    c_alpha = bounded_ridge_normalization_C(float(alpha), beta)
    domain = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float64)
    positive_domain = np.abs(domain)
    target_abs = bounded_ridge_target(
        positive_domain, alpha=float(alpha), beta=beta, C_alpha=c_alpha
    )
    target = np.sign(domain) * target_abs
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for requested in SEARCH_DEGREES:
        degree, _ = qsvt_odd_degree(int(requested))
        if degree in seen:
            continue
        seen.add(degree)
        polynomial, coefficients, _ = fit_bounded_ridge_polynomial(
            alpha=float(alpha), beta=beta, degree=degree
        )
        values_on_grid = np.asarray(polynomial(domain), dtype=np.float64)
        coeffs = np.asarray(coefficients, dtype=np.float64)
        even_coefficients = coeffs[0::2]
        rows.append(
            {
                "degree": int(degree),
                "uniform_error": float(np.max(np.abs(values_on_grid - target))),
                "boundedness_verified": bool(np.max(np.abs(values_on_grid)) <= 1.0 + 1.0e-10),
                "parity_verified": bool(
                    even_coefficients.size == 0 or np.max(np.abs(even_coefficients)) <= 1.0e-10
                ),
            }
        )
    return rows


def _uniform_degree_for_target(
    profile: list[dict[str, Any]], *, tolerance: float
) -> tuple[int | None, bool, bool]:
    for row in profile:
        if (
            float(row["uniform_error"]) <= float(tolerance)
            and bool(row["boundedness_verified"])
            and bool(row["parity_verified"])
        ):
            return int(row["degree"]), True, True
    return None, False, False


def run_degree_aware_alpha_selection(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    grid_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    uniform_cache: dict[tuple[str, float, float], tuple[int | None, bool, bool]] = {}
    scenario = str(resolved["scenario"])
    existing = read_csv(EXISTING_ALPHA_SWEEP)

    for case in resolved["cases"]:
        system, _matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["seed"]),
            }
        )
        singular_values = system.singular_values()
        condition_number = float(system.condition_number())

        # Per-alpha classical metrics and bounded QSVT-target degree profile.
        per_alpha: dict[float, dict[str, float]] = {}
        degree_cache: dict[tuple[float, float], tuple[int | None, float]] = {}
        for alpha in resolved["alpha_grid"]:
            update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=float(alpha))
            c_alpha, profile = _alpha_degree_profile(singular_values, alpha=float(alpha))
            uniform_profile = _uniform_profile(singular_values, alpha=float(alpha))
            per_alpha[alpha] = {
                "rmse": float(system.rmse(update)),
                "residual_norm": float(system.residual_norm(update)),
                "bounded_scale_C": float(c_alpha),
            }
            for tolerance in resolved["tolerances"]:
                degree_cache[(alpha, tolerance)] = _degree_for_target(
                    profile, tolerance=float(tolerance)
                )
                uniform_cache[(case, alpha, tolerance)] = _uniform_degree_for_target(
                    uniform_profile, tolerance=float(tolerance)
                )

        # Determine degree-aware-selected alpha per (tolerance, degree_budget).
        selected_keys = _degree_aware_selected(
            resolved["alpha_grid"],
            resolved["tolerances"],
            resolved["degree_budgets"],
            per_alpha,
            degree_cache,
        )

        for alpha in resolved["alpha_grid"]:
            for tolerance in resolved["tolerances"]:
                degree_required, approx_error = degree_cache[(alpha, tolerance)]
                target_met = degree_required is not None
                degree_cell = "" if degree_required is None else int(degree_required)
                phase_available = False
                for budget in resolved["degree_budgets"]:
                    budget_met = bool(target_met and degree_required <= int(budget))
                    is_selected = selected_keys.get((tolerance, budget)) == alpha
                    rule = "degree_aware_min_rmse_under_budget" if is_selected else "grid_point"
                    grid_rows.append(
                        {
                            "case": case,
                            "scenario": scenario,
                            "alpha": float(alpha),
                            "tolerance": float(tolerance),
                            "degree_budget": int(budget),
                            "rmse": per_alpha[alpha]["rmse"],
                            "residual_norm": per_alpha[alpha]["residual_norm"],
                            "condition_number": condition_number,
                            "degree_required": degree_cell,
                            "approx_error": float(approx_error),
                            "target_met": bool(target_met),
                            "phase_available": phase_available,
                            "degree_budget_met": budget_met,
                            "bounded_scale_C": per_alpha[alpha]["bounded_scale_C"],
                            "selection_rule": rule,
                            "selected": bool(is_selected),
                            "notes": (
                                "spectrum-point action degree; per-row phase synthesis was "
                                "not performed"
                            ),
                        }
                    )

        summary_rows.extend(
            _summary_for_case(
                case,
                scenario,
                resolved,
                per_alpha,
                degree_cache,
                singular_values,
            )
        )

    grid_frame = pd.DataFrame(grid_rows, columns=GRID_COLUMNS)
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    revised_rows = _revised_summary_rows(summary_frame, uniform_cache)
    revised_frame = pd.DataFrame(revised_rows, columns=REVISED_SUMMARY_COLUMNS)

    grid_csv = output_dir / "degree_aware_alpha_grid.csv"
    summary_csv = output_dir / "degree_aware_alpha_summary.csv"
    report_md = output_dir / "degree_aware_alpha_report.md"
    revised_csv = output_dir / "revised_degree_alpha_summary.csv"
    revised_report = output_dir / "revised_degree_alpha_report.md"
    rows_to_table(grid_rows, grid_csv, GRID_COLUMNS)
    rows_to_table(summary_rows, summary_csv, SUMMARY_COLUMNS)
    report_text = _report_markdown(grid_frame, summary_frame, resolved, existing)
    assert_safe(report_text)
    report_md.write_text(report_text, encoding="utf-8")
    rows_to_table(revised_rows, revised_csv, REVISED_SUMMARY_COLUMNS)
    revised_text = _revised_report_markdown(revised_frame)
    assert_safe(revised_text)
    revised_report.write_text(revised_text, encoding="utf-8")

    artifacts = {
        "degree_aware_alpha_grid_csv": grid_csv,
        "degree_aware_alpha_summary_csv": summary_csv,
        "degree_aware_alpha_report_md": report_md,
        "revised_degree_alpha_summary_csv": revised_csv,
        "revised_degree_alpha_report_md": revised_report,
    }
    manifest = write_workload_manifest(
        output_dir=output_dir,
        artifact_name="degree_aware_alpha_selection",
        description=(
            "Degree-aware alpha selection comparing classical RMSE against the QSVT "
            "polynomial degree required to approximate the bounded target to tolerance. "
            "Feasibility/boundary evidence; no speedup is claimed."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[
            *[f"build_engineering_system:{case}:weighted_jacobian" for case in resolved["cases"]],
            EXISTING_ALPHA_SWEEP,
        ],
        reran_long_experiments=False,
        aggregated_from_existing=not existing.empty,
        extra={
            "cases": list(resolved["cases"]),
            "alpha_grid": list(resolved["alpha_grid"]),
            "tolerances": list(resolved["tolerances"]),
            "degree_budgets": list(resolved["degree_budgets"]),
            "phase_feasible_degree_hint": PHASE_FEASIBLE_DEGREE_HINT,
            "existing_alpha_sweep_reused": not existing.empty,
        },
        manifest_name="degree_aware_alpha_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "grid": grid_frame,
        "summary": summary_frame,
        "revised_summary": revised_frame,
        "artifacts": artifacts,
    }


def _degree_aware_selected(
    alpha_grid: tuple[float, ...],
    tolerances: tuple[float, ...],
    degree_budgets: tuple[int, ...],
    per_alpha: dict[float, dict[str, float]],
    degree_cache: dict[tuple[float, float], tuple[int | None, float]],
) -> dict[tuple[float, int], float | None]:
    """For each (tolerance, budget) pick argmin-RMSE alpha meeting the degree budget."""

    selected: dict[tuple[float, int], float | None] = {}
    for tolerance in tolerances:
        for budget in degree_budgets:
            feasible = [
                alpha
                for alpha in alpha_grid
                if (degree_cache[(alpha, tolerance)][0] is not None)
                and (degree_cache[(alpha, tolerance)][0] <= int(budget))
            ]
            if feasible:
                selected[(tolerance, budget)] = min(feasible, key=lambda a: per_alpha[a]["rmse"])
            else:
                selected[(tolerance, budget)] = None
    return selected


def _summary_for_case(
    case: str,
    scenario: str,
    resolved: dict[str, Any],
    per_alpha: dict[float, dict[str, float]],
    degree_cache: dict[tuple[float, float], tuple[int | None, float]],
    singular_values: np.ndarray,
) -> list[dict[str, Any]]:
    alpha_grid = resolved["alpha_grid"]
    reference_tol = float(resolved["reference_tolerance"])
    reference_budget = int(resolved["reference_budget"])
    rows: list[dict[str, Any]] = []

    def emit(rule: str, alpha: float | None, tolerance: float, budget: int, notes: str) -> None:
        if alpha is None:
            rows.append(
                {
                    "case": case,
                    "scenario": scenario,
                    "selection_rule": rule,
                    "tolerance": tolerance,
                    "degree_budget": budget,
                    "selected_alpha": "",
                    "rmse_at_selected": "",
                    "degree_required_at_selected": "",
                    "target_met": False,
                    "degree_budget_met": False,
                    "phase_available": False,
                    "notes": notes + " (no alpha satisfies the constraint)",
                }
            )
            return
        degree_required, _approx = degree_cache[(alpha, tolerance)]
        budget_met = bool(degree_required is not None and degree_required <= budget)
        rows.append(
            {
                "case": case,
                "scenario": scenario,
                "selection_rule": rule,
                "tolerance": tolerance,
                "degree_budget": budget,
                "selected_alpha": float(alpha),
                "rmse_at_selected": per_alpha[alpha]["rmse"],
                "degree_required_at_selected": (
                    "" if degree_required is None else int(degree_required)
                ),
                "target_met": bool(degree_required is not None),
                "degree_budget_met": budget_met,
                "phase_available": False,
                "notes": notes,
            }
        )

    best_classical = min(alpha_grid, key=lambda a: per_alpha[a]["rmse"])
    emit(
        "best_classical_rmse",
        best_classical,
        reference_tol,
        reference_budget,
        "alpha with minimum classical RMSE on the grid (ignores degree budget)",
    )
    emit(
        "default_alpha",
        float(resolved["default_alpha"]),
        reference_tol,
        reference_budget,
        "fixed default regularization parameter",
    )
    spectrum_alpha, spectrum_target = _spectrum_based_alpha(singular_values, alpha_grid)
    emit(
        "spectrum_based_alpha",
        spectrum_alpha,
        reference_tol,
        reference_budget,
        f"nearest grid alpha to sigma_min^2 = {spectrum_target:.3e} (spectrum heuristic)",
    )
    degree_selected = _degree_aware_selected(
        alpha_grid, (reference_tol,), (reference_budget,), per_alpha, degree_cache
    )[(reference_tol, reference_budget)]
    emit(
        "degree_aware_under_dmax",
        degree_selected,
        reference_tol,
        reference_budget,
        f"min-RMSE alpha s.t. degree <= d_max={reference_budget} at eps={reference_tol:.0e}",
    )
    # Degree-aware under approximation tolerance: smallest alpha whose target is met
    # at the reference tolerance within the largest budget, breaking ties by RMSE.
    largest_budget = int(max(resolved["degree_budgets"]))
    tol_selected = _degree_aware_selected(
        alpha_grid, (reference_tol,), (largest_budget,), per_alpha, degree_cache
    )[(reference_tol, largest_budget)]
    emit(
        "degree_aware_under_tolerance",
        tol_selected,
        reference_tol,
        largest_budget,
        f"argmin RMSE s.t. target met at eps={reference_tol:.0e} within d<= {largest_budget}",
    )
    return rows


def _revised_summary_rows(
    summary: pd.DataFrame,
    uniform_cache: dict[tuple[str, float, float], tuple[int | None, bool, bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        alpha_raw = row["selected_alpha"]
        spectrum_raw = row["degree_required_at_selected"]
        if alpha_raw == "" or spectrum_raw == "":
            uniform_degree, bounded, parity = None, False, False
            degree_derived = ""
        else:
            alpha = float(alpha_raw)
            tolerance = float(row["tolerance"])
            uniform_degree, bounded, parity = uniform_cache[(str(row["case"]), alpha, tolerance)]
            degree_derived = 2 * int(spectrum_raw) + 1
        rows.append(
            {
                "case": row["case"],
                "scenario": row["scenario"],
                "selection_rule": row["selection_rule"],
                "selected_alpha": alpha_raw,
                "tolerance": row["tolerance"],
                "degree_budget": row["degree_budget"],
                "rmse_at_selected": row["rmse_at_selected"],
                "spectrum_point_degree": spectrum_raw,
                "uniform_grid_degree": "" if uniform_degree is None else uniform_degree,
                "phase_synthesis_available": False,
                "boundedness_verified": bounded,
                "parity_verified": parity,
                "qsvt_query_count_realizable": False,
                "degree_derived_query_count": degree_derived,
                "query_count_status": "spectrum_point_action_only",
                "notes": (
                    "Uniform-grid boundedness/parity checks do not include phase synthesis; "
                    "2d+1 is therefore not labeled a realizable QSVT query count."
                ),
            }
        )
    return rows


def _revised_report_markdown(revised: pd.DataFrame) -> str:
    lines = [
        "# Revised Degree-Aware Alpha Report",
        "",
        "`spectrum_point_degree` is the empirical degree needed only at the actual matrix "
        "singular values. `uniform_grid_degree` adds a dense uniform-domain error, "
        "boundedness, and parity check. No row has per-row phase synthesis, so no row is "
        "labeled with a realizable QSVT query count.",
        "",
        "| Case | Rule | Alpha | Spectrum-point d | Uniform-grid d | Bounded | "
        "Parity | Phases | Realizable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in revised.iterrows():
        alpha = "-" if row["selected_alpha"] == "" else f"{float(row['selected_alpha']):.0e}"
        spectrum = "-" if row["spectrum_point_degree"] == "" else int(row["spectrum_point_degree"])
        uniform = "-" if row["uniform_grid_degree"] == "" else int(row["uniform_grid_degree"])
        lines.append(
            f"| {row['case']} | {row['selection_rule']} | {alpha} | {spectrum} | "
            f"{uniform} | {row['boundedness_verified']} | {row['parity_verified']} | "
            f"{row['phase_synthesis_available']} | {row['qsvt_query_count_realizable']} |"
        )
    lines += [
        "",
        "The degree-aware alpha rule in this artifact uses the spectrum-point action degree "
        "as an empirical implementation tradeoff. It is not a uniform QSVT scaling theorem. "
        "A `uniform-admissible QSVT degree` would additionally require successful phase "
        "synthesis under the verified convention.",
        "",
    ]
    return "\n".join(lines)


def _spectrum_based_alpha(
    singular_values: np.ndarray, alpha_grid: tuple[float, ...]
) -> tuple[float, float]:
    positive = singular_values[singular_values > 0.0]
    sigma_min = float(positive.min()) if positive.size else 1.0
    target = sigma_min**2
    nearest = min(alpha_grid, key=lambda a: abs(np.log10(a) - np.log10(target)))
    return float(nearest), float(target)


def _report_markdown(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    resolved: dict[str, Any],
    existing: pd.DataFrame,
) -> str:
    lines = [
        "# Degree-Aware Alpha Selection",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "Classical filter `P_alpha(sigma) = sigma/(sigma^2+alpha)`; bounded QSVT target "
        "`f_{alpha,bounded}(s) = (1/C) s/(s^2 + alpha/beta^2)` on `s in [0,1]`.",
        "",
        "**Key interpretation:** the best classical regularization parameter is not "
        "automatically the best QSVT-implementable choice under a degree budget. No speedup "
        "is claimed; the QSVT target uses the same alpha as Ridge/Tikhonov.",
        "",
        f"Alpha grid = {list(resolved['alpha_grid'])}; tolerances = "
        f"{list(resolved['tolerances'])}; degree budgets = {list(resolved['degree_budgets'])}.",
        "",
        "## Alpha-Selection Rule Comparison",
        "",
        "| Case | Rule | Selected alpha | RMSE | Degree required | Budget | Budget met |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        alpha_text = "-" if row["selected_alpha"] == "" else f"{float(row['selected_alpha']):.0e}"
        rmse_raw = row["rmse_at_selected"]
        rmse_text = "-" if rmse_raw == "" else f"{float(rmse_raw):.4e}"
        degree_text = (
            "-"
            if row["degree_required_at_selected"] == ""
            else str(int(row["degree_required_at_selected"]))
        )
        lines.append(
            f"| {row['case']} | {row['selection_rule']} | {alpha_text} | {rmse_text} | "
            f"{degree_text} | {int(row['degree_budget'])} | {row['degree_budget_met']} |"
        )
    lines += [
        "",
        f"## Degree Required vs Alpha (tolerance = {float(resolved['reference_tolerance']):.0e})",
        "",
        "| Case | Alpha | RMSE | Degree required | Bounded C |",
        "| --- | --- | --- | --- | --- |",
    ]
    ref_tol = float(resolved["reference_tolerance"])
    ref_budget = int(resolved["reference_budget"])
    slice_df = grid[
        (grid["tolerance"] == ref_tol) & (grid["degree_budget"] == ref_budget)
    ].drop_duplicates(subset=["case", "alpha"])
    for _, row in slice_df.iterrows():
        degree_text = "-" if row["degree_required"] == "" else str(int(row["degree_required"]))
        lines.append(
            f"| {row['case']} | {float(row['alpha']):.0e} | {float(row['rmse']):.4e} | "
            f"{degree_text} | {float(row['bounded_scale_C']):.3f} |"
        )
    lines += [
        "",
        "## Cross-Reference (independent classical alpha sweep)",
        "",
    ]
    if not existing.empty and "case" in existing.columns:
        lines.append(
            "An independent classical alpha sweep is reused from "
            f"`{EXISTING_ALPHA_SWEEP}` as a consistency cross-reference (Ridge/Tikhonov "
            "rows). The degree-required columns above are computed here from the same "
            "weighted Jacobian and are not present in that artifact."
        )
    else:
        lines.append(
            "No external classical alpha-sweep artifact was found; all values above are "
            "computed directly from the generated weighted Jacobian."
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Under the bounded QSVT-target convention, the required degree and the bounded "
        "scaling constant `C = max(1, 1/(2*sqrt(alpha)))` trade off in opposite directions "
        "with alpha: small alpha is degree-cheap but C-expensive (smaller success "
        "probability / more postselection), while large alpha shrinks C but raises the "
        "degree, often past the budget. The degree-aware rule selects the largest "
        "budget-feasible alpha, balancing the two.",
        "- `phase_available` is a degree-range feasibility hint (degree_required <= "
        f"{PHASE_FEASIBLE_DEGREE_HINT}), not a per-row phase-synthesis run.",
        "- `degree_required` is the smallest searched odd degree whose bounded-target error "
        "at the actual singular values meets the tolerance; a blank value means no searched "
        f"degree up to {max(SEARCH_DEGREES)} met it.",
        "- Classical RMSE is nearly flat across alpha for these well-posed linearized "
        "systems, so the QSVT-feasible alpha is reached at negligible RMSE cost; no speedup "
        "or QSVT-over-Ridge superiority is implied.",
        "",
    ]
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(WORKLOAD_DIR),
        "cases": list(DEFAULT_CASES),
        "case_source": "pypower",
        "seed": 123,
        "scenario": "ac_linearized",
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "tolerances": list(DEFAULT_TOLERANCES),
        "degree_budgets": list(DEFAULT_DEGREE_BUDGETS),
        "default_alpha": DEFAULT_ALPHA,
        "reference_tolerance": 1.0e-2,
        "reference_budget": 51,
        "command": "run_degree_aware_alpha_selection",
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["alpha_grid"] = tuple(float(a) for a in resolved["alpha_grid"])
    resolved["tolerances"] = tuple(float(t) for t in resolved["tolerances"])
    resolved["degree_budgets"] = tuple(int(d) for d in resolved["degree_budgets"])
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run degree-aware alpha selection")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run = run_degree_aware_alpha_selection(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "seed": args.seed,
            "command": "scripts/run_degree_aware_alpha_selection.py " + " ".join(argv or []),
        }
    )
    grid_path = run["artifacts"]["degree_aware_alpha_grid_csv"]
    print(f"Degree-aware alpha selection complete: {grid_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
