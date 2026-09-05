from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory

CLAIM = (
    "Alternative bounded-target design comparison for the QSVT-compatible "
    "regularized spectral filter on a selected IEEE-derived subproblem. It "
    "isolates whether the residual gap to Ridge/Tikhonov is driven by bounded-"
    "target normalization (C), the polynomial fitting interval, or polynomial "
    "construction. Scaling-only target designs leave the known-C-rescaled "
    "physical update unchanged; only target-shape changes move the residual, at "
    "the cost of generality. Ridge/Tikhonov remains the reference target; no "
    "QSVT superiority, quantum speedup, or full IEEE-scale gate-level claim is "
    "made."
)

QSVT_SAFE_TOLERANCE = 1.0e-6

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "design_family",
    "C_design",
    "C_global",
    "C_support",
    "boundedness_margin",
    "raw_max_abs_on_unit_interval",
    "domain_min",
    "fit_domain_min",
    "fit_domain_max",
    "pointwise_error_uniform_grid",
    "pointwise_error_singular_values",
    "weighted_filter_error",
    "polynomial_action_residual",
    "C_rescaled_polynomial_residual",
    "best_scalar_polynomial_residual",
    "ridge_residual",
    "direction_error_vs_ridge",
    "norm_error_vs_ridge",
    "success_probability_proxy",
    "qsvt_compatibility_status",
    "generalizability_status",
    "design_classification",
    "run_status",
    "failure_reason_if_any",
]

FILTER_VALUE_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_design",
    "singular_value_normalized",
    "raw_target_value",
    "raw_polynomial_value",
    "bounded_polynomial_value",
    "absolute_error",
]

RESIDUAL_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_design",
    "design_family",
    "C_design",
    "boundedness_margin",
    "polynomial_action_residual",
    "C_rescaled_polynomial_residual",
    "best_scalar_polynomial_residual",
    "ridge_residual",
    "direction_error_vs_ridge",
    "norm_error_vs_ridge",
    "qsvt_compatibility_status",
    "design_classification",
    "run_status",
]


@dataclass(frozen=True, slots=True)
class NormalizedProblem:
    """Block-encoding-normalized view of a selected subproblem."""

    H: np.ndarray
    r: np.ndarray
    beta: float
    A: np.ndarray
    U: np.ndarray
    singular_values_A: np.ndarray
    Vh: np.ndarray
    alpha_norm: float
    domain_min: float
    ridge_update: np.ndarray
    ridge_residual: float


@dataclass(frozen=True, slots=True)
class RawPolynomialFit:
    """Raw odd polynomial approximating x / (x^2 + alpha_norm) on the support."""

    polynomial: Polynomial
    fit_domain_min: float
    fit_domain_max: float
    raw_max_abs_on_unit_interval: float
    instance_aware: bool


@dataclass(frozen=True, slots=True)
class TargetDesign:
    """A bounded-target design: scaling constant plus optional shape change."""

    name: str
    family: str  # "scaling_only" | "shape_changing"
    generalizability: str  # "general" | "instance_aware" | "diagnostic"
    C_design: float
    C_global: float
    C_support: float
    fit: RawPolynomialFit


def run_qsvt_bounded_target_design_study(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "margin_factors": [1.01, 1.05, 1.10, 1.25, 1.50],
        "seed": 123,
        "grid_size": 4096,
        "gain_cap_factor": 0.5,
        "subproblem_id": "ieee14_ac_bounded_target_4x4",
        "output_dir": "outputs/qsvt_bounded_target_design_study",
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
    summary_rows, filter_rows = evaluate_target_design_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        margin_factors=[float(value) for value in resolved["margin_factors"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        grid_size=int(resolved["grid_size"]),
        gain_cap_factor=float(resolved["gain_cap_factor"]),
    )
    artifacts = write_bounded_target_design_outputs(output_dir, resolved, summary_rows, filter_rows)
    return {
        "output_dir": output_dir,
        "rows": summary_rows,
        "filter_rows": filter_rows,
        "artifacts": artifacts,
    }


def evaluate_target_design_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    margin_factors: list[float],
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
    gain_cap_factor: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for degree in degrees:
            rows, samples = evaluate_target_designs_for_config(
                subproblem=subproblem,
                alpha=float(alpha),
                degree=int(degree),
                margin_factors=margin_factors,
                case=case,
                model=model,
                subproblem_id=subproblem_id,
                grid_size=int(grid_size),
                gain_cap_factor=float(gain_cap_factor),
            )
            summary_rows.extend(rows)
            filter_rows.extend(samples)
    return summary_rows, filter_rows


def evaluate_target_designs_for_config(
    *,
    subproblem: SelectedSubproblem,
    alpha: float,
    degree: int,
    margin_factors: list[float],
    case: str = "custom",
    model: str = "weighted_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
    gain_cap_factor: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
    }
    try:
        problem = _build_normalized_problem(subproblem, alpha=float(alpha))
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
            margin_factors=margin_factors,
            gain_cap_factor=float(gain_cap_factor),
        )
        summary_rows: list[dict[str, Any]] = []
        filter_rows: list[dict[str, Any]] = []
        for design in designs:
            row, samples = _evaluate_single_design(
                problem=problem,
                design=design,
                base=base,
                grid_size=int(grid_size),
            )
            summary_rows.append(row)
            filter_rows.extend(samples)
        return summary_rows, filter_rows
    except Exception as exc:
        row = {column: np.nan for column in SUMMARY_COLUMNS}
        row.update(base)
        row.update(
            {
                "target_design": "all",
                "run_status": "failed",
                "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
            }
        )
        return [row], []


def write_bounded_target_design_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(summary_rows, SUMMARY_COLUMNS)
    residual_frame = summary_frame[RESIDUAL_COLUMNS].copy()
    filter_frame = _frame_with_columns(filter_rows, FILTER_VALUE_COLUMNS)

    summary_path = output_dir / "target_design_summary.csv"
    filter_path = output_dir / "target_design_filter_values.csv"
    residual_path = output_dir / "target_design_residuals.csv"
    tradeoff_path = output_dir / "target_design_tradeoff.md"

    summary_frame.to_csv(summary_path, index=False)
    filter_frame.to_csv(filter_path, index=False)
    residual_frame.to_csv(residual_path, index=False)
    tradeoff_path.write_text(target_design_tradeoff(summary_frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "target_design_summary": str(summary_path),
            "target_design_filter_values": str(filter_path),
            "target_design_residuals": str(residual_path),
            "target_design_tradeoff": str(tradeoff_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "target_design_summary": summary_path,
        "target_design_filter_values": filter_path,
        "target_design_residuals": residual_path,
        "target_design_tradeoff": tradeoff_path,
    }


def target_design_tradeoff(frame: pd.DataFrame) -> str:
    completed = frame[frame["run_status"] == "completed"].copy()
    if completed.empty:
        return "\n".join(
            ["# Bounded-Target Design Tradeoff", "", CLAIM, "", "- No completed rows."]
        )

    completed["C_rescaled_polynomial_residual"] = completed[
        "C_rescaled_polynomial_residual"
    ].astype(float)
    scaling_only = completed[completed["design_family"] == "scaling_only"]
    shape_changing = completed[completed["design_family"] == "shape_changing"]

    over_conservative = _too_conservatively_scaled(completed)
    support_helps = _support_scaling_helps(completed)
    gap_from_design = _gap_caused_by_design(scaling_only)
    better_residual_line = _better_residual_without_violation(completed)

    best = completed.loc[completed["best_scalar_polynomial_residual"].astype(float).idxmin()]
    safe = completed[completed["qsvt_compatibility_status"] == "qsvt_safe"]
    safe_general = completed[completed["design_classification"] == "qsvt_safe_general"]

    return "\n".join(
        [
            "# Bounded-Target Design Tradeoff",
            "",
            CLAIM,
            "",
            "## Required Answers",
            f"1. Current target too conservatively scaled? {over_conservative}",
            f"2. Does support-aware scaling help? {support_helps}",
            f"3. Is the residual gap caused by target design rather than phase synthesis? "
            f"{gap_from_design}",
            f"4. Does any target design give better residual without violating boundedness? "
            f"{better_residual_line}",
            "",
            "## Design Counts",
            f"- Total completed (design, alpha, degree) rows: {len(completed)}",
            f"- QSVT-safe rows (boundedness_margin <= 1 + {QSVT_SAFE_TOLERANCE:g}): {len(safe)}",
            f"- QSVT-safe general-purpose rows: {len(safe_general)}",
            f"- Scaling-only rows: {len(scaling_only)}",
            f"- Shape-changing rows: {len(shape_changing)}",
            "",
            "## Best Row (by best-scalar residual)",
            f"- Target design: {best['target_design']}",
            f"- Alpha: {float(best['alpha']):.17g}",
            f"- Degree: {int(best['degree'])}",
            f"- Boundedness margin: {float(best['boundedness_margin']):.17g}",
            f"- C-rescaled polynomial residual: "
            f"{float(best['C_rescaled_polynomial_residual']):.17g}",
            f"- Best-scalar polynomial residual: "
            f"{float(best['best_scalar_polynomial_residual']):.17g}",
            f"- Ridge residual: {float(best['ridge_residual']):.17g}",
            f"- QSVT compatibility: {best['qsvt_compatibility_status']}",
            f"- Classification: {best['design_classification']}",
            "",
            "## Interpretation",
            "- For scaling-only designs (global_safe, support_scaled, margin_*), the "
            "physical known-C-rescaled update equals (1/beta) p_raw(A) r and is independent "
            "of C_design. C only sets the boundedness margin and success-probability proxy.",
            "- Support-aware and margin scaling improve the boundedness margin and "
            "success-probability proxy but leave the known-C residual unchanged.",
            "- Only shape-changing variants (clipped_inverse_like, piecewise_support_aware) "
            "can move the residual, and they do so as regularized / instance-aware designs, "
            "not as general-purpose replacements for the Ridge reference.",
            "",
        ]
    )


def _build_normalized_problem(subproblem: SelectedSubproblem, *, alpha: float) -> NormalizedProblem:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    if H.ndim != 2 or r.ndim != 1 or H.shape[0] != r.size:
        raise ValueError("H_tilde must be m x n and r_tilde must have length m")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    U, singular_values_A, Vh = np.linalg.svd(A, full_matrices=False)
    positive = singular_values_A[singular_values_A > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected subproblem has no positive singular values")
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
    alpha_norm = float(alpha) / beta**2
    ridge = ridge_tikhonov_update(H, r, alpha=float(alpha))
    return NormalizedProblem(
        H=H,
        r=r,
        beta=beta,
        A=A,
        U=U,
        singular_values_A=singular_values_A,
        Vh=Vh,
        alpha_norm=alpha_norm,
        domain_min=domain_min,
        ridge_update=ridge,
        ridge_residual=_residual(H, ridge, r),
    )


def _fit_raw_polynomial(
    problem: NormalizedProblem,
    *,
    degree: int,
    grid_size: int,
    domain_min: float,
    instance_aware: bool,
    gain_cap: float | None = None,
) -> RawPolynomialFit:
    """Fit the raw odd polynomial ~ x/(x^2+alpha_norm); recover by multiplying by C0.

    The QSVT polynomial routine returns a scaled, bounded fit p_scaled = p_raw / C0.
    Multiplying the returned power coefficients by ``scale_factor`` recovers the raw
    polynomial that directly approximates the unbounded target on the support. A
    finite ``gain_cap`` clips the fitting target to ``min(target, gain_cap)``,
    deliberately changing the target shape to limit small-singular-value gain.
    """
    fit_domain_max = 1.0
    fit_domain_min = float(min(max(1.0e-6, domain_min), 0.95))
    grid = max(int(grid_size), int(degree) + 2)
    approximation = fit_odd_regularized_polynomial(
        alpha=problem.alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=fit_domain_min,
        domain_max=fit_domain_max,
        grid_size=grid,
    )
    C0 = float(approximation.scale_factor)
    raw_coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64) * C0
    if gain_cap is not None and np.isfinite(gain_cap):
        polynomial = _fit_clipped_raw_polynomial(
            problem,
            degree=int(degree),
            grid_size=grid,
            fit_domain_min=fit_domain_min,
            fit_domain_max=fit_domain_max,
            gain_cap=float(gain_cap),
        )
    else:
        polynomial = Polynomial(raw_coefficients)
    qsvt_grid = np.linspace(-1.0, 1.0, max(8193, 256 * int(degree) + 1), dtype=np.float64)
    raw_max_abs = float(np.max(np.abs(polynomial(qsvt_grid))))
    return RawPolynomialFit(
        polynomial=polynomial,
        fit_domain_min=fit_domain_min,
        fit_domain_max=fit_domain_max,
        raw_max_abs_on_unit_interval=raw_max_abs,
        instance_aware=bool(instance_aware),
    )


def _fit_clipped_raw_polynomial(
    problem: NormalizedProblem,
    *,
    degree: int,
    grid_size: int,
    fit_domain_min: float,
    fit_domain_max: float,
    gain_cap: float,
) -> Polynomial:
    """Fit an odd polynomial to the clipped target min(x/(x^2+alpha_norm), gain_cap).

    Enforces odd parity via p(x) = x q(x^2), mirroring fit_odd_regularized_polynomial,
    but against the gain-limited (shape-changed) target. This regularized design is
    NOT identical to Ridge/Tikhonov; it caps the achievable gain on small singular
    values to keep the polynomial bounded.
    """
    reduced_degree = (int(degree) - 1) // 2
    y_min = fit_domain_min**2
    y_max = fit_domain_max**2
    y_grid = np.linspace(y_min, y_max, int(grid_size), dtype=np.float64)
    x_grid = np.sqrt(y_grid)
    raw_target = x_grid / (y_grid + problem.alpha_norm)
    capped_target = np.minimum(raw_target, float(gain_cap))
    q_target = capped_target / x_grid
    q_polynomial = Chebyshev.fit(
        y_grid, q_target, deg=reduced_degree, domain=[y_min, y_max]
    ).convert(kind=Polynomial)
    power_coefficients = np.zeros(int(degree) + 1, dtype=np.float64)
    for index, coefficient in enumerate(q_polynomial.coef):
        target_index = 2 * index + 1
        if target_index <= int(degree):
            power_coefficients[target_index] = coefficient
    return Polynomial(power_coefficients)


def _build_target_designs(
    *,
    problem: NormalizedProblem,
    base_fit: RawPolynomialFit,
    degree: int,
    grid_size: int,
    margin_factors: list[float],
    gain_cap_factor: float,
) -> list[TargetDesign]:
    alpha_norm = problem.alpha_norm
    positive = problem.singular_values_A[problem.singular_values_A > 1.0e-14]
    sigma_min_norm = float(np.min(positive))

    dense_grid = np.linspace(sigma_min_norm, 1.0, max(int(grid_size) * 4, 20000), dtype=np.float64)
    C_global = float(np.max(dense_grid / (dense_grid**2 + alpha_norm)))
    C_support = float(np.max(positive / (positive**2 + alpha_norm)))

    def make(name: str, family: str, generalizability: str, C_design: float, fit: RawPolynomialFit):
        return TargetDesign(
            name=name,
            family=family,
            generalizability=generalizability,
            C_design=float(C_design),
            C_global=C_global,
            C_support=C_support,
            fit=fit,
        )

    designs: list[TargetDesign] = [
        make("global_safe", "scaling_only", "general", C_global, base_fit),
        make("support_scaled", "scaling_only", "instance_aware", C_support, base_fit),
    ]
    for factor in margin_factors:
        designs.append(
            make(
                f"margin_{factor:g}", "scaling_only", "instance_aware", factor * C_support, base_fit
            )
        )

    # degree_adaptive: smallest safe C over [-1,1] => C = max|p_raw| * (1 + tol).
    C_safe = base_fit.raw_max_abs_on_unit_interval * (1.0 + QSVT_SAFE_TOLERANCE)
    designs.append(make("degree_adaptive", "scaling_only", "general", C_safe, base_fit))

    # clipped_inverse_like: shape change (gain cap) scaled by C_support.
    clipped_fit = _fit_raw_polynomial(
        problem,
        degree=int(degree),
        grid_size=int(grid_size),
        domain_min=problem.domain_min,
        instance_aware=False,
        gain_cap=float(gain_cap_factor) * C_support,
    )
    designs.append(
        make("clipped_inverse_like", "shape_changing", "instance_aware", C_support, clipped_fit)
    )

    # piecewise_support_aware: fit only on observed singular interval (instance-aware).
    piecewise_fit = _fit_raw_polynomial(
        problem,
        degree=int(degree),
        grid_size=int(grid_size),
        domain_min=max(1.0e-6, 0.95 * sigma_min_norm),
        instance_aware=True,
    )
    designs.append(
        make(
            "piecewise_support_aware", "shape_changing", "instance_aware", C_support, piecewise_fit
        )
    )
    return designs


def _evaluate_single_design(
    *,
    problem: NormalizedProblem,
    design: TargetDesign,
    base: dict[str, Any],
    grid_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit = design.fit
    C_design = float(design.C_design)
    p_raw = fit.polynomial
    alpha_norm = problem.alpha_norm

    sv = problem.singular_values_A
    raw_target_singular = sv / (sv**2 + alpha_norm)
    raw_poly_singular = p_raw(sv)
    bounded_poly_singular = raw_poly_singular / C_design
    bounded_target_singular = raw_target_singular / C_design

    grid = np.linspace(fit.fit_domain_min, 1.0, max(int(grid_size), 512), dtype=np.float64)
    raw_target_grid = grid / (grid**2 + alpha_norm)
    bounded_target_grid = raw_target_grid / C_design
    bounded_poly_grid = p_raw(grid) / C_design
    pointwise_error_grid = float(np.max(np.abs(bounded_poly_grid - bounded_target_grid)))
    pointwise_error_singular = float(
        np.max(np.abs(bounded_poly_singular - bounded_target_singular))
    )

    # Weighted filter error: weight by the ridge spectral kernel 1/(sv^2+alpha_norm),
    # emphasizing the small-singular-value regime that dominates the regularized solve.
    weights = 1.0 / (sv**2 + alpha_norm)
    weighted_filter_error = float(
        np.sqrt(np.sum(weights * (raw_poly_singular - raw_target_singular) ** 2) / np.sum(weights))
    )

    boundedness_margin = float(fit.raw_max_abs_on_unit_interval / C_design)
    qsvt_safe = boundedness_margin <= 1.0 + QSVT_SAFE_TOLERANCE

    # Bounded QSVT output on the normalized residual, then known-C physical rescale.
    bounded_operator = problem.U @ ((p_raw(sv) / C_design)[:, None] * problem.Vh)
    bounded_output = bounded_operator @ problem.r
    physical_update = (C_design / problem.beta) * bounded_output  # == (1/beta) p_raw(A) r
    polynomial_action_residual = _residual(problem.H, physical_update, problem.r)
    # C_rescaled residual is the same physical residual: documents C-independence.
    c_rescaled_residual = polynomial_action_residual
    best_scalar = best_scalar_for_residual(problem.H, physical_update, problem.r)
    best_scalar_residual = _residual(problem.H, best_scalar * physical_update, problem.r)

    success_probability_proxy = float(min(1.0, float(np.linalg.norm(bounded_output)) ** 2))

    classification = _classify_design(
        family=design.family,
        generalizability=design.generalizability,
        qsvt_safe=qsvt_safe,
        best_scalar_residual=best_scalar_residual,
        ridge_residual=problem.ridge_residual,
    )

    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(base)
    row.update(
        {
            "target_design": design.name,
            "design_family": design.family,
            "C_design": C_design,
            "C_global": design.C_global,
            "C_support": design.C_support,
            "boundedness_margin": boundedness_margin,
            "raw_max_abs_on_unit_interval": fit.raw_max_abs_on_unit_interval,
            "domain_min": problem.domain_min,
            "fit_domain_min": fit.fit_domain_min,
            "fit_domain_max": fit.fit_domain_max,
            "pointwise_error_uniform_grid": pointwise_error_grid,
            "pointwise_error_singular_values": pointwise_error_singular,
            "weighted_filter_error": weighted_filter_error,
            "polynomial_action_residual": polynomial_action_residual,
            "C_rescaled_polynomial_residual": c_rescaled_residual,
            "best_scalar_polynomial_residual": best_scalar_residual,
            "ridge_residual": problem.ridge_residual,
            "direction_error_vs_ridge": _direction_error(physical_update, problem.ridge_update),
            "norm_error_vs_ridge": _norm_error(physical_update, problem.ridge_update),
            "success_probability_proxy": success_probability_proxy,
            "qsvt_compatibility_status": "qsvt_safe" if qsvt_safe else "not_qsvt_safe",
            "generalizability_status": design.generalizability,
            "design_classification": classification,
            "run_status": "completed",
            "failure_reason_if_any": "",
        }
    )
    samples = [
        {
            "case": base["case"],
            "alpha": base["alpha"],
            "degree": base["degree"],
            "target_design": design.name,
            "singular_value_normalized": float(value),
            "raw_target_value": float(raw_target_singular[index]),
            "raw_polynomial_value": float(raw_poly_singular[index]),
            "bounded_polynomial_value": float(bounded_poly_singular[index]),
            "absolute_error": float(
                abs(bounded_poly_singular[index] - bounded_target_singular[index])
            ),
        }
        for index, value in enumerate(sv)
    ]
    return row, samples


def _classify_design(
    *,
    family: str,
    generalizability: str,
    qsvt_safe: bool,
    best_scalar_residual: float,
    ridge_residual: float,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    if generalizability == "diagnostic":
        return "diagnostic_only"
    poor = best_scalar_residual > max(10.0 * ridge_residual, 1.0e-6)
    if family == "scaling_only":
        if generalizability == "general":
            return "qsvt_safe_but_poor_residual" if poor else "qsvt_safe_general"
        return "qsvt_safe_instance_aware"
    # shape_changing
    return "qsvt_safe_instance_aware"


def _too_conservatively_scaled(frame: pd.DataFrame) -> str:
    global_rows = frame[frame["target_design"] == "global_safe"]
    support_rows = frame[frame["target_design"] == "support_scaled"]
    if global_rows.empty or support_rows.empty:
        return "inconclusive"
    g_margin = float(global_rows["boundedness_margin"].astype(float).median())
    s_margin = float(support_rows["boundedness_margin"].astype(float).median())
    # Conservative scaling => margin well below 1 (C larger than needed on the support).
    if g_margin < 0.9 and s_margin < 0.9:
        return (
            f"yes (global margin median {g_margin:.4g}, support margin median {s_margin:.4g} "
            "are below 1, so C exceeds the support requirement; this only loses "
            "success probability, not residual)"
        )
    return (
        f"no (margins are near or above 1: global {g_margin:.4g}, support {s_margin:.4g}; "
        "high-degree polynomials overshoot off the fitting interval)"
    )


def _support_scaling_helps(frame: pd.DataFrame) -> str:
    global_rows = frame[frame["target_design"] == "global_safe"]
    support_rows = frame[frame["target_design"] == "support_scaled"]
    if global_rows.empty or support_rows.empty:
        return "inconclusive"
    g_resid = float(global_rows["C_rescaled_polynomial_residual"].astype(float).median())
    s_resid = float(support_rows["C_rescaled_polynomial_residual"].astype(float).median())
    g_prob = float(global_rows["success_probability_proxy"].astype(float).median())
    s_prob = float(support_rows["success_probability_proxy"].astype(float).median())
    residual_same = abs(g_resid - s_resid) <= 1.0e-9 * max(abs(g_resid), 1.0)
    return (
        "for success probability / boundedness only, not the known-C residual "
        f"(residuals match: {residual_same}; success-proxy global {g_prob:.4g} vs "
        f"support {s_prob:.4g})"
    )


def _gap_caused_by_design(scaling_only: pd.DataFrame) -> str:
    if scaling_only.empty:
        return "inconclusive"
    residuals = scaling_only["C_rescaled_polynomial_residual"].astype(float)
    spread = float(residuals.max() - residuals.min())
    # Scaling-only designs share the same physical residual at fixed (alpha, degree).
    by_config = scaling_only.groupby(["alpha", "degree"])["C_rescaled_polynomial_residual"].agg(
        lambda values: float(values.astype(float).max() - values.astype(float).min())
    )
    max_within = float(by_config.max()) if len(by_config) else float("nan")
    if max_within <= 1.0e-9 * max(float(residuals.max()), 1.0):
        return (
            "no for scaling/normalization (all scaling-only target designs give the same "
            f"known-C residual at fixed alpha/degree; within-config spread {max_within:.3e}). "
            "The residual gap tracks degree/conditioning and target shape, not C; overall "
            f"scaling-only residual spread {spread:.3e} is driven by alpha/degree, not design"
        )
    return f"partially (scaling-only residual spread {spread:.3e})"


def _better_residual_without_violation(frame: pd.DataFrame) -> str:
    safe = frame[frame["qsvt_compatibility_status"] == "qsvt_safe"].copy()
    if safe.empty:
        return "no qsvt-safe rows"
    safe["best_scalar_polynomial_residual"] = safe["best_scalar_polynomial_residual"].astype(float)
    safe["ridge_residual"] = safe["ridge_residual"].astype(float)
    scaling = safe[safe["design_family"] == "scaling_only"]
    shape = safe[safe["design_family"] == "shape_changing"]
    best_scaling = (
        float(scaling["best_scalar_polynomial_residual"].min())
        if not scaling.empty
        else float("inf")
    )
    if shape.empty:
        return (
            "only via degree/instance choices among scaling-only safe designs "
            f"(best safe scaling residual {best_scaling:.4g})"
        )
    best_shape_row = shape.loc[shape["best_scalar_polynomial_residual"].idxmin()]
    best_shape = float(best_shape_row["best_scalar_polynomial_residual"])
    if best_shape < best_scaling:
        return (
            "yes, only the shape-changing instance-aware variant "
            f"({best_shape_row['target_design']}, residual {best_shape:.4g}) beats the best "
            f"safe scaling-only residual {best_scaling:.4g}, at the cost of generality"
        )
    return (
        "no: shape-changing safe designs do not beat the best safe scaling-only residual "
        f"({best_shape:.4g} vs {best_scaling:.4g})"
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    candidate_norm = float(np.linalg.norm(candidate_values))
    reference_norm = float(np.linalg.norm(reference_values))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate_values, reference_values) / (candidate_norm * reference_norm))
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - np.clip(cosine, -1.0, 1.0)))))


def _norm_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    return float(abs(candidate_norm - reference_norm) / max(reference_norm, 1.0e-15))
