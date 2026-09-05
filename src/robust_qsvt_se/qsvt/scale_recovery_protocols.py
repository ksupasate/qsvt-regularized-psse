from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.norm_success import estimate_success_probability_iterative_proxy
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory

SCALE_RECOVERY_CLAIM = (
    "This study compares scale-recovery protocols that map the bounded QSVT output "
    "direction to a physical state-estimation update magnitude. Ridge/Tikhonov is the "
    "reference filter and the only ground truth. Best-scalar and ridge-norm rows are "
    "classical diagnostics, not deployable protocols. Norm and amplitude proxies are "
    "simulator-metadata proxies, not implemented quantum readout routines. No QSVT "
    "superiority over Ridge and no quantum speedup is claimed."
)

CLAIM_ALLOWED = "scale-recovery protocol / norm-recovery proxy"
CLAIM_DISALLOWED = "QSVT beats Ridge; quantum speedup"

DEPLOYABILITY_LABELS = {
    "classical_diagnostic_only",
    "simulator_assisted_proxy",
    "quantum_protocol_proxy",
    "potentially_deployable_under_oracle_assumptions",
    "not_deployable",
}

SCALE_RECOVERY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "alpha",
    "degree",
    "condition_number",
    "scale_protocol",
    "scale_factor",
    "residual",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "relative_update_error_vs_ridge",
    "direction_error_vs_ridge",
    "requires_ridge_reference",
    "requires_simulator_metadata",
    "requires_amplitude_estimation",
    "query_cost_proxy",
    "deployability_class",
    "dominant_limitation",
    "claim_allowed",
    "claim_disallowed",
]

PROXY_PROTOCOLS = ("success_amplitude_proxy", "amplitude_estimation_proxy")


@dataclass(frozen=True, slots=True)
class BoundedQsvtAction:
    """Bounded QSVT direction and the reference quantities shared by every protocol."""

    direction: np.ndarray
    direction_norm: float
    beta: float
    scale_factor_C: float
    ridge_update: np.ndarray
    ridge_residual: float
    residual_no_update: float
    success_probability_proxy: float
    known_c_residual: float
    best_scalar_residual: float
    direction_error_vs_ridge: float


def run_scale_recovery_protocols(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "seed": 123,
        "grid_size": 4096,
        "eps_rel": 1.0e-2,
        "amplitude_max_queries": 1000,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "selection_mode": "high_leverage",
        "output_dir": "outputs/qsvt_scale_recovery_protocols",
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
    rows = evaluate_scale_recovery_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        selection_mode=str(resolved["selection_mode"]),
        grid_size=int(resolved["grid_size"]),
        eps_rel=float(resolved["eps_rel"]),
        amplitude_max_queries=int(resolved["amplitude_max_queries"]),
        seed=int(resolved["seed"]),
    )
    artifacts = write_scale_recovery_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_scale_recovery_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    grid_size: int = 4096,
    eps_rel: float = 1.0e-2,
    amplitude_max_queries: int = 1000,
    seed: int = 123,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for degree in degrees:
            rows.extend(
                evaluate_scale_recovery_point(
                    subproblem=subproblem,
                    alpha=float(alpha),
                    degree=int(degree),
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    selection_mode=selection_mode,
                    grid_size=int(grid_size),
                    eps_rel=float(eps_rel),
                    amplitude_max_queries=int(amplitude_max_queries),
                    seed=int(seed),
                )
            )
    return rows


def evaluate_scale_recovery_point(
    *,
    subproblem: SelectedSubproblem,
    alpha: float,
    degree: int,
    case: str = "custom",
    model: str = "weighted_linearized",
    subproblem_id: str = "subproblem",
    selection_mode: str = "unit_test",
    grid_size: int = 4096,
    eps_rel: float = 1.0e-2,
    amplitude_max_queries: int = 1000,
    seed: int = 123,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    condition_number = _condition_number(H)
    action = compute_bounded_qsvt_action(
        H, r, alpha=float(alpha), degree=int(degree), grid_size=grid_size
    )
    protocols = _protocol_updates(
        action=action,
        H=H,
        r=r,
        alpha=float(alpha),
        eps_rel=float(eps_rel),
        amplitude_max_queries=int(amplitude_max_queries),
        seed=int(seed),
    )
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        rows.append(
            _protocol_row(
                protocol=protocol,
                action=action,
                H=H,
                r=r,
                case=case,
                model=model,
                subproblem_id=subproblem_id,
                selection_mode=selection_mode,
                alpha=float(alpha),
                degree=int(degree),
                condition_number=condition_number,
            )
        )
    return rows


def compute_bounded_qsvt_action(
    H: np.ndarray,
    r: np.ndarray,
    *,
    alpha: float,
    degree: int,
    grid_size: int = 4096,
) -> BoundedQsvtAction:
    """Build the bounded QSVT output direction u = poly_operator @ r per the standard recipe."""

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    positive = singular_values_A[singular_values_A > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected subproblem has no positive singular values")
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
    alpha_norm = float(alpha) / beta**2
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=domain_min,
        domain_max=1.0,
        grid_size=max(int(grid_size), int(degree) + 2),
    )
    scale_factor_C = float(approximation.scale_factor)
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients, dtype=np.float64) / scale_factor_C
    )
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    polynomial_operator = U @ (polynomial(singular_values)[:, None] * Vh)
    direction = polynomial_operator @ r
    direction_norm = float(np.linalg.norm(direction))

    ridge = ridge_tikhonov_update(H, r, alpha=float(alpha))
    ridge_residual = _residual(H, ridge, r)
    residual_no_update = float(np.linalg.norm(r))
    success_probability_proxy = float(min(1.0, direction_norm**2))

    known_c_update = rescale_bounded_target_to_original(direction, beta=beta, C=scale_factor_C)
    known_c_residual = _residual(H, known_c_update, r)
    best_scalar = best_scalar_for_residual(H, direction, r)
    best_scalar_residual = _residual(H, best_scalar * direction, r)
    direction_error = _direction_error(direction, ridge)

    return BoundedQsvtAction(
        direction=direction,
        direction_norm=direction_norm,
        beta=beta,
        scale_factor_C=scale_factor_C,
        ridge_update=ridge,
        ridge_residual=ridge_residual,
        residual_no_update=residual_no_update,
        success_probability_proxy=success_probability_proxy,
        known_c_residual=known_c_residual,
        best_scalar_residual=best_scalar_residual,
        direction_error_vs_ridge=direction_error,
    )


@dataclass(frozen=True, slots=True)
class _ProtocolResult:
    scale_protocol: str
    scale_factor: float
    update: np.ndarray
    requires_ridge_reference: bool
    requires_simulator_metadata: bool
    requires_amplitude_estimation: bool
    query_cost_proxy: float
    deployability_class: str


def _protocol_updates(
    *,
    action: BoundedQsvtAction,
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    eps_rel: float,
    amplitude_max_queries: int,
    seed: int,
) -> list[_ProtocolResult]:
    u = action.direction
    u_norm = action.direction_norm
    beta = action.beta
    C = action.scale_factor_C
    ridge = action.ridge_update
    unit = u / u_norm if u_norm > 0.0 else np.zeros_like(u)
    ridge_norm = float(np.linalg.norm(ridge))
    p_success = action.success_probability_proxy

    results: list[_ProtocolResult] = []

    # 1) known_C: analytic constant rescaling (C/beta) u. Oracle-assumption deployable.
    known_c_update = rescale_bounded_target_to_original(u, beta=beta, C=C)
    results.append(
        _ProtocolResult(
            scale_protocol="known_C",
            scale_factor=float(C / beta),
            update=known_c_update,
            requires_ridge_reference=False,
            requires_simulator_metadata=False,
            requires_amplitude_estimation=False,
            query_cost_proxy=float("nan"),
            deployability_class="potentially_deployable_under_oracle_assumptions",
        )
    )

    # 2) ridge_norm diagnostic: rescale unit direction to the ridge update norm.
    results.append(
        _ProtocolResult(
            scale_protocol="ridge_norm",
            scale_factor=(ridge_norm / u_norm) if u_norm > 0.0 else float("nan"),
            update=ridge_norm * unit,
            requires_ridge_reference=True,
            requires_simulator_metadata=False,
            requires_amplitude_estimation=False,
            query_cost_proxy=float("nan"),
            deployability_class="classical_diagnostic_only",
        )
    )

    # 3) best_scalar diagnostic: least-squares-optimal scalar on the QSVT direction.
    best_scalar = best_scalar_for_residual(H, u, r)
    results.append(
        _ProtocolResult(
            scale_protocol="best_scalar_diagnostic",
            scale_factor=float(best_scalar),
            update=best_scalar * u,
            requires_ridge_reference=True,
            requires_simulator_metadata=False,
            requires_amplitude_estimation=False,
            query_cost_proxy=float("nan"),
            deployability_class="classical_diagnostic_only",
        )
    )

    # 4) success_amplitude_proxy: magnitude from success-probability metadata only.
    success_magnitude = (C / beta) * math.sqrt(p_success)
    results.append(
        _ProtocolResult(
            scale_protocol="success_amplitude_proxy",
            scale_factor=(success_magnitude / u_norm) if u_norm > 0.0 else float("nan"),
            update=success_magnitude * unit,
            requires_ridge_reference=False,
            requires_simulator_metadata=True,
            requires_amplitude_estimation=False,
            query_cost_proxy=float("nan"),
            deployability_class="quantum_protocol_proxy",
        )
    )

    # 5) amplitude_estimation_proxy: amplitude-estimation-style 1/eps query scaling.
    estimate = estimate_success_probability_iterative_proxy(
        p_success, max(int(amplitude_max_queries), 1), int(seed)
    )
    estimated_p = float(estimate.estimate)
    ae_magnitude = (C / beta) * math.sqrt(max(estimated_p, 0.0))
    query_cost_proxy = float(math.ceil(1.0 / (float(eps_rel) * math.sqrt(max(p_success, 1.0e-12)))))
    results.append(
        _ProtocolResult(
            scale_protocol="amplitude_estimation_proxy",
            scale_factor=(ae_magnitude / u_norm) if u_norm > 0.0 else float("nan"),
            update=ae_magnitude * unit,
            requires_ridge_reference=False,
            requires_simulator_metadata=True,
            requires_amplitude_estimation=True,
            query_cost_proxy=query_cost_proxy,
            deployability_class="quantum_protocol_proxy",
        )
    )

    # 6) observable_calibrated: simulator-reference observable on largest-|ridge| component.
    observable_index = int(np.argmax(np.abs(ridge))) if ridge.size else 0
    observable = np.zeros_like(u)
    if observable.size:
        observable[observable_index] = 1.0
    o_dot_u = float(np.dot(observable, u))
    observable_scalar = (
        float(np.dot(observable, ridge) / o_dot_u) if abs(o_dot_u) > 1.0e-12 else float("nan")
    )
    observable_update = (
        observable_scalar * u if math.isfinite(observable_scalar) else np.full_like(u, np.nan)
    )
    results.append(
        _ProtocolResult(
            scale_protocol="observable_calibrated",
            scale_factor=observable_scalar,
            update=observable_update,
            requires_ridge_reference=False,
            requires_simulator_metadata=True,
            requires_amplitude_estimation=False,
            query_cost_proxy=float("nan"),
            deployability_class="simulator_assisted_proxy",
        )
    )
    return results


def _protocol_row(
    *,
    protocol: _ProtocolResult,
    action: BoundedQsvtAction,
    H: np.ndarray,
    r: np.ndarray,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    alpha: float,
    degree: int,
    condition_number: float,
) -> dict[str, Any]:
    update = protocol.update
    if np.all(np.isfinite(update)):
        residual = _residual(H, update, r)
        relative_update_error = _relative_error(update, action.ridge_update)
    else:
        residual = float("nan")
        relative_update_error = float("nan")
    residual_ratio_no_update = (
        residual / action.residual_no_update if action.residual_no_update > 0.0 else float("nan")
    )
    residual_ratio_ridge = (
        residual / action.ridge_residual if action.ridge_residual > 0.0 else float("nan")
    )
    return {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "selection_mode": selection_mode,
        "alpha": float(alpha),
        "degree": int(degree),
        "condition_number": float(condition_number),
        "scale_protocol": protocol.scale_protocol,
        "scale_factor": float(protocol.scale_factor),
        "residual": float(residual),
        "residual_ratio_vs_no_update": float(residual_ratio_no_update),
        "residual_ratio_vs_ridge": float(residual_ratio_ridge),
        "relative_update_error_vs_ridge": float(relative_update_error),
        "direction_error_vs_ridge": float(action.direction_error_vs_ridge),
        "requires_ridge_reference": bool(protocol.requires_ridge_reference),
        "requires_simulator_metadata": bool(protocol.requires_simulator_metadata),
        "requires_amplitude_estimation": bool(protocol.requires_amplitude_estimation),
        "query_cost_proxy": float(protocol.query_cost_proxy),
        "deployability_class": protocol.deployability_class,
        "dominant_limitation": _dominant_limitation(action),
        "claim_allowed": CLAIM_ALLOWED,
        "claim_disallowed": CLAIM_DISALLOWED,
    }


def write_scale_recovery_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = pd.DataFrame(rows)
    for column in SCALE_RECOVERY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[SCALE_RECOVERY_COLUMNS]

    summary_path = output_dir / "scale_recovery_summary.csv"
    residual_path = output_dir / "residual_by_scale_protocol.csv"
    cost_path = output_dir / "norm_recovery_cost_proxy.csv"
    interpretation_path = output_dir / "scale_recovery_interpretation.md"

    frame.to_csv(summary_path, index=False)
    frame.to_csv(residual_path, index=False)
    cost_columns = [
        "case",
        "model",
        "subproblem_id",
        "alpha",
        "degree",
        "scale_protocol",
        "scale_factor",
        "residual",
        "residual_ratio_vs_ridge",
        "requires_simulator_metadata",
        "requires_amplitude_estimation",
        "query_cost_proxy",
        "deployability_class",
    ]
    proxy_frame = frame[frame["scale_protocol"].isin(PROXY_PROTOCOLS)][cost_columns]
    proxy_frame.to_csv(cost_path, index=False)
    interpretation_path.write_text(scale_recovery_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "scale_recovery_summary": str(summary_path),
            "residual_by_scale_protocol": str(residual_path),
            "norm_recovery_cost_proxy": str(cost_path),
            "scale_recovery_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=SCALE_RECOVERY_CLAIM,
    )
    return {
        "manifest": manifest,
        "scale_recovery_summary": summary_path,
        "residual_by_scale_protocol": residual_path,
        "norm_recovery_cost_proxy": cost_path,
        "scale_recovery_interpretation": interpretation_path,
    }


def scale_recovery_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# QSVT Scale-Recovery Protocols", "", SCALE_RECOVERY_CLAIM, ""])

    best_per_protocol = (
        frame.groupby("scale_protocol", sort=True)["residual_ratio_vs_ridge"].min().to_dict()
    )
    known_c_best = best_per_protocol.get("known_C", float("nan"))
    best_scalar_best = best_per_protocol.get("best_scalar_diagnostic", float("nan"))
    success_proxy_best = best_per_protocol.get("success_amplitude_proxy", float("nan"))
    ae_proxy_best = best_per_protocol.get("amplitude_estimation_proxy", float("nan"))
    observable_best = best_per_protocol.get("observable_calibrated", float("nan"))

    known_c_improves = "yes" if known_c_best < 1.0 else "no"
    best_implementable_proxy = min(
        (value for value in (success_proxy_best, ae_proxy_best) if math.isfinite(value)),
        default=float("nan"),
    )
    proxy_reaches_best_scalar = (
        "yes"
        if math.isfinite(best_implementable_proxy)
        and math.isfinite(best_scalar_best)
        and best_implementable_proxy <= 1.5 * best_scalar_best
        else "no"
    )
    best_scalar_much_better = (
        "yes"
        if math.isfinite(best_scalar_best)
        and math.isfinite(known_c_best)
        and best_scalar_best < 0.5 * known_c_best
        else "no"
    )
    limitation_counts = frame["dominant_limitation"].value_counts().to_dict()
    dominant = max(limitation_counts, key=limitation_counts.get) if limitation_counts else "n/a"
    most_promising = _most_promising_protocol(best_per_protocol)
    known_c_no_update_ratio = float(
        frame[frame["scale_protocol"] == "known_C"]["residual_ratio_vs_no_update"].min()
    )

    return "\n".join(
        [
            "# QSVT Scale-Recovery Protocols",
            "",
            SCALE_RECOVERY_CLAIM,
            "",
            "## Best residual ratio vs Ridge per protocol (min over alpha, degree)",
            f"- known_C: {known_c_best:.6g}",
            f"- ridge_norm: {best_per_protocol.get('ridge_norm', float('nan')):.6g}",
            f"- best_scalar_diagnostic: {best_scalar_best:.6g}",
            f"- success_amplitude_proxy: {success_proxy_best:.6g}",
            f"- amplitude_estimation_proxy: {ae_proxy_best:.6g}",
            f"- observable_calibrated: {observable_best:.6g}",
            "",
            "## Required Answers",
            "1. Does known-C rescaling improve residual versus no update? "
            f"{known_c_improves} "
            f"(known_C residual ratio vs no-update min {known_c_no_update_ratio:.6g}).",
            "2. Does any norm proxy approach the best-scalar diagnostic? "
            f"{proxy_reaches_best_scalar} "
            f"(best implementable proxy ratio {best_implementable_proxy:.6g} vs "
            f"best-scalar {best_scalar_best:.6g}).",
            "3. Is the best-scalar diagnostic much better than implementable proxies? "
            f"{best_scalar_much_better}.",
            "4. Is the residual gap magnitude-limited or direction-limited? "
            f"Dominant limitation across rows: {dominant}.",
            "5. Most promising scale-recovery strategy for future QSVT implementation: "
            f"{most_promising}. The best-scalar and ridge-norm rows are classical "
            "diagnostics (require the Ridge reference) and are not deployable; known_C "
            "is the only oracle-assumption-deployable protocol, and the amplitude/"
            "success proxies are the implementable quantum-side candidates.",
            "",
        ]
    )


def _most_promising_protocol(best_per_protocol: dict[str, float]) -> str:
    deployable = {
        "known_C": best_per_protocol.get("known_C", float("inf")),
        "success_amplitude_proxy": best_per_protocol.get("success_amplitude_proxy", float("inf")),
        "amplitude_estimation_proxy": best_per_protocol.get(
            "amplitude_estimation_proxy", float("inf")
        ),
    }
    finite = {key: value for key, value in deployable.items() if math.isfinite(value)}
    if not finite:
        return "inconclusive"
    return min(finite, key=finite.get)


def _dominant_limitation(action: BoundedQsvtAction) -> str:
    if action.direction_error_vs_ridge > 0.1:
        return "direction_limited"
    if action.best_scalar_residual < 0.5 * action.known_c_residual:
        return "magnitude_limited"
    return "polynomial_or_conditioning"


def _condition_number(H: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(H, dtype=np.float64), compute_uv=False)
    if singular_values.size == 0:
        return float("nan")
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    return sigma_max / sigma_min if sigma_min > 1.0e-14 else float("inf")


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if candidate_norm <= 1.0e-30 or reference_norm <= 1.0e-30:
        return float("nan")
    return float(np.linalg.norm(candidate / candidate_norm - reference / reference_norm))


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_values - reference_values)
        / max(float(np.linalg.norm(reference_values)), 1.0e-15)
    )
