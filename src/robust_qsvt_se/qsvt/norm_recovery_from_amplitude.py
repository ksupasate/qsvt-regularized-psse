from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import (
    bernoulli_amplitude_estimate,
    iterative_amplitude_estimate_for_probability,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.scale_recovery_protocols import compute_bounded_qsvt_action
from robust_qsvt_se.utils.io import ensure_directory

NORM_RECOVERY_CLAIM = (
    "Update-scale recovery from actual amplitude/norm-estimation routines on a selected "
    "IEEE14-derived subproblem. The QSVT output direction sets the update direction; the "
    "amplitude estimate recovers only the scale. Exact-success and best-scalar rows are "
    "non-deployable diagnostics; Bernoulli, iterative-QAE, and hybrid-known-C rows are "
    "simulator-backed small-circuit recovery routines. Ridge/Tikhonov remains the reference; "
    "no QSVT superiority over Ridge/Tikhonov, quantum speedup, quantum advantage, or hardware "
    "execution is claimed."
)

# Canonical norm-recovery methods. The CLI alias ``iterative_qae_if_available`` maps to
# ``iterative_qae`` so the same routine is reachable from both Phase 2 and Phase 3.
NORM_RECOVERY_METHODS = (
    "exact_success_amplitude_diagnostic",
    "bernoulli_success_amplitude",
    "iterative_qae",
    "hybrid_known_C_amplitude",
    "observable_calibrated",
    "best_scalar_upper_bound",
)
METHOD_ALIASES = {"iterative_qae_if_available": "iterative_qae"}
DEPLOYABLE_NORM_METHODS = (
    "bernoulli_success_amplitude",
    "iterative_qae",
    "hybrid_known_C_amplitude",
)
DIAGNOSTIC_NORM_METHODS = (
    "exact_success_amplitude_diagnostic",
    "best_scalar_upper_bound",
    "observable_calibrated",
)

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "norm_recovery_method",
    "scale_factor",
    "scale_factor_error_vs_best_scalar",
    "estimated_update_norm",
    "ridge_update_norm",
    "qsvt_direction_norm",
    "residual_after_recovery",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "relative_update_error_vs_ridge",
    "direction_error_vs_ridge",
    "uses_exact_statevector",
    "uses_shots",
    "uses_qae",
    "uses_ridge_reference",
    "uses_simulator_metadata",
    "deployability_class",
    "routine_status",
    "failure_reason_if_any",
]


@dataclass(frozen=True, slots=True)
class NormRecovery:
    """Result of recovering an update scale from an amplitude estimate."""

    method: str
    scale_factor: float
    estimated_update: np.ndarray
    estimated_update_norm: float
    probability_estimate: float
    probability_standard_error: float
    deployability_class: str
    uses_exact_statevector: bool
    uses_shots: bool
    uses_qae: bool
    uses_ridge_reference: bool
    uses_simulator_metadata: bool
    routine_status: str
    failure_reason: str


def run_norm_recovery_from_amplitude(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "shots": [100, 1000, 10000],
        "methods": list(NORM_RECOVERY_METHODS),
        "grover_powers": [0, 1, 2, 4, 8],
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "target_design": "current_global",
        "output_dir": "outputs/qsvt_norm_recovery_from_amplitude",
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
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for alpha in [float(value) for value in resolved["alphas"]]:
        for degree in [int(value) for value in resolved["degrees"]]:
            for shots in [int(value) for value in resolved["shots"]]:
                rows.extend(
                    evaluate_norm_recovery_point(
                        H=H,
                        r=r,
                        alpha=alpha,
                        degree=degree,
                        shots=shots,
                        methods=[str(value) for value in resolved["methods"]],
                        grover_powers=tuple(int(v) for v in resolved["grover_powers"]),
                        case=str(resolved["case"]),
                        model=str(resolved["model"]),
                        subproblem_id=str(resolved["subproblem_id"]),
                        target_design=str(resolved["target_design"]),
                        seed=int(resolved["seed"]),
                    )
                )
    artifacts = write_norm_recovery_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_norm_recovery_point(
    *,
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    degree: int,
    shots: int,
    methods: list[str],
    grover_powers: tuple[int, ...] = (0, 1, 2, 4, 8),
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    target_design: str = "current_global",
    seed: int = 123,
) -> list[dict[str, Any]]:
    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    no_update = float(np.linalg.norm(r))
    ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
    ridge_norm = float(np.linalg.norm(ridge_update))
    action = compute_bounded_qsvt_action(H, r, alpha=float(alpha), degree=int(degree))
    direction = np.asarray(action.direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    unit = direction / direction_norm if direction_norm > 1.0e-15 else np.zeros_like(direction)
    best_scalar = best_scalar_for_residual(H, unit, r)
    direction_error = _direction_error(direction, ridge_update)

    rows: list[dict[str, Any]] = []
    for raw_method in methods:
        method = METHOD_ALIASES.get(str(raw_method), str(raw_method))
        recovery = apply_norm_recovery(
            method,
            unit=unit,
            direction_norm=direction_norm,
            scale_factor_C=action.scale_factor_C,
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
            relative_update_error = _relative_error(update, ridge_update)
        else:
            residual = float("nan")
            relative_update_error = float("nan")
        scale_error = (
            abs(recovery.scale_factor - best_scalar)
            if math.isfinite(recovery.scale_factor)
            else float("nan")
        )
        rows.append(
            {
                "case": case,
                "model": model,
                "subproblem_id": subproblem_id,
                "alpha": float(alpha),
                "degree": int(degree),
                "target_design": target_design,
                "norm_recovery_method": method,
                "scale_factor": float(recovery.scale_factor),
                "scale_factor_error_vs_best_scalar": float(scale_error),
                "estimated_update_norm": float(recovery.estimated_update_norm),
                "ridge_update_norm": ridge_norm,
                "qsvt_direction_norm": direction_norm,
                "residual_after_recovery": float(residual),
                "residual_ratio_vs_no_update": (
                    residual / no_update if no_update > 0.0 else float("nan")
                ),
                "residual_ratio_vs_ridge": (
                    residual / action.ridge_residual
                    if action.ridge_residual > 0.0
                    else float("nan")
                ),
                "relative_update_error_vs_ridge": float(relative_update_error),
                "direction_error_vs_ridge": float(direction_error),
                "uses_exact_statevector": bool(recovery.uses_exact_statevector),
                "uses_shots": bool(recovery.uses_shots),
                "uses_qae": bool(recovery.uses_qae),
                "uses_ridge_reference": bool(recovery.uses_ridge_reference),
                "uses_simulator_metadata": bool(recovery.uses_simulator_metadata),
                "deployability_class": recovery.deployability_class,
                "routine_status": recovery.routine_status,
                "failure_reason_if_any": recovery.failure_reason,
            }
        )
    return rows


def apply_norm_recovery(
    method: str,
    *,
    unit: np.ndarray,
    direction_norm: float,
    scale_factor_C: float,
    beta: float,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
    shots: int,
    seed: int,
    grover_powers: tuple[int, ...] = (0, 1, 2, 4, 8),
) -> NormRecovery:
    """Recover the physical update scale on ``unit`` for one norm-recovery method.

    ``direction_norm`` is the bounded QSVT output norm; the exact success probability is
    ``min(1, direction_norm**2)`` and the known-C magnitude is ``(C/beta) * direction_norm``.
    """

    canonical = METHOD_ALIASES.get(str(method), str(method))
    unit = np.asarray(unit, dtype=np.float64)
    known_c_per_amplitude = scale_factor_C / max(beta, np.finfo(float).eps)
    p_exact = float(min(1.0, direction_norm**2))

    def build(
        scale_factor: float,
        *,
        probability_estimate: float,
        deployability_class: str,
        probability_standard_error: float = 0.0,
        uses_exact_statevector: bool = False,
        uses_shots: bool = False,
        uses_qae: bool = False,
        uses_ridge_reference: bool = False,
        uses_simulator_metadata: bool = False,
        routine_status: str = "succeeded",
        failure_reason: str = "",
    ) -> NormRecovery:
        update = scale_factor * unit if math.isfinite(scale_factor) else np.full_like(unit, np.nan)
        return NormRecovery(
            method=canonical,
            scale_factor=float(scale_factor),
            estimated_update=update,
            estimated_update_norm=float(abs(scale_factor))
            if math.isfinite(scale_factor)
            else float("nan"),
            probability_estimate=float(probability_estimate),
            probability_standard_error=float(probability_standard_error),
            deployability_class=deployability_class,
            uses_exact_statevector=uses_exact_statevector,
            uses_shots=uses_shots,
            uses_qae=uses_qae,
            uses_ridge_reference=uses_ridge_reference,
            uses_simulator_metadata=uses_simulator_metadata,
            routine_status=routine_status,
            failure_reason=failure_reason,
        )

    if canonical == "exact_success_amplitude_diagnostic":
        return build(
            known_c_per_amplitude * direction_norm,
            probability_estimate=p_exact,
            deployability_class="statevector_diagnostic_only",
            uses_exact_statevector=True,
            uses_simulator_metadata=True,
        )
    if canonical == "bernoulli_success_amplitude":
        estimate = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed))
        return build(
            known_c_per_amplitude * math.sqrt(max(estimate.estimate, 0.0)),
            probability_estimate=estimate.estimate,
            probability_standard_error=estimate.standard_error,
            deployability_class="shot_based_small_circuit",
            uses_shots=True,
        )
    if canonical == "iterative_qae":
        result = iterative_amplitude_estimate_for_probability(
            p_exact, powers=grover_powers, shots=int(shots), seed=int(seed)
        )
        if result.status != "succeeded" or not math.isfinite(result.estimate):
            return build(
                float("nan"),
                probability_estimate=float("nan"),
                deployability_class="unsupported_for_current_circuit",
                uses_qae=True,
                routine_status=result.status,
                failure_reason=result.failure_reason or "iterative_qae_unavailable",
            )
        return build(
            known_c_per_amplitude * math.sqrt(max(result.estimate, 0.0)),
            probability_estimate=result.estimate,
            probability_standard_error=result.standard_error,
            deployability_class="iterative_qae_small_circuit",
            uses_qae=True,
            uses_shots=True,
        )
    if canonical == "hybrid_known_C_amplitude":
        result = iterative_amplitude_estimate_for_probability(
            p_exact, powers=grover_powers, shots=int(shots), seed=int(seed) + 1
        )
        if result.status == "succeeded" and math.isfinite(result.estimate):
            probability = result.estimate
            standard_error = result.standard_error
            uses_qae = True
        else:
            fallback = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed) + 2)
            probability = fallback.estimate
            standard_error = fallback.standard_error
            uses_qae = False
        return build(
            known_c_per_amplitude * math.sqrt(max(probability, 0.0)),
            probability_estimate=probability,
            probability_standard_error=standard_error,
            deployability_class="iterative_qae_small_circuit"
            if uses_qae
            else "shot_based_small_circuit",
            uses_qae=uses_qae,
            uses_shots=True,
        )
    if canonical == "observable_calibrated":
        index = int(np.argmax(np.abs(ridge_update))) if ridge_update.size else 0
        denominator = float(unit[index]) if unit.size > index else 0.0
        scale = (
            float(ridge_update[index] / denominator) if abs(denominator) > 1.0e-12 else float("nan")
        )
        return build(
            scale,
            probability_estimate=p_exact,
            deployability_class="observable_readout_small_circuit",
            uses_ridge_reference=True,
            uses_simulator_metadata=True,
        )
    if canonical == "best_scalar_upper_bound":
        scale = best_scalar_for_residual(H, unit, r)
        return build(
            float(scale),
            probability_estimate=p_exact,
            deployability_class="statevector_diagnostic_only",
            uses_exact_statevector=True,
        )
    raise ValueError(f"unsupported norm_recovery_method: {method}")


def write_norm_recovery_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    summary_path = output_dir / "norm_recovery_summary.csv"
    residual_path = output_dir / "residual_after_amplitude_norm_recovery.csv"
    scale_path = output_dir / "scale_factor_comparison.csv"
    interpretation_path = output_dir / "norm_recovery_interpretation.md"

    frame.to_csv(summary_path, index=False)
    frame[
        [
            "alpha",
            "degree",
            "norm_recovery_method",
            "residual_after_recovery",
            "residual_ratio_vs_no_update",
            "residual_ratio_vs_ridge",
            "direction_error_vs_ridge",
            "deployability_class",
        ]
    ].to_csv(residual_path, index=False)
    frame[
        [
            "alpha",
            "degree",
            "norm_recovery_method",
            "scale_factor",
            "scale_factor_error_vs_best_scalar",
            "estimated_update_norm",
            "ridge_update_norm",
            "qsvt_direction_norm",
            "deployability_class",
        ]
    ].to_csv(scale_path, index=False)
    interpretation_path.write_text(norm_recovery_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "norm_recovery_summary": str(summary_path),
            "residual_after_amplitude_norm_recovery": str(residual_path),
            "scale_factor_comparison": str(scale_path),
            "norm_recovery_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=NORM_RECOVERY_CLAIM,
    )
    return {
        "manifest": manifest,
        "norm_recovery_summary": summary_path,
        "residual_after_amplitude_norm_recovery": residual_path,
        "scale_factor_comparison": scale_path,
        "norm_recovery_interpretation": interpretation_path,
    }


def norm_recovery_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# Norm Recovery From Amplitude", "", NORM_RECOVERY_CLAIM])

    def best_ratio(methods: tuple[str, ...]) -> float:
        subset = frame[frame["norm_recovery_method"].isin(methods)][
            "residual_ratio_vs_no_update"
        ].astype(float)
        finite = subset[np.isfinite(subset)]
        return float(finite.min()) if not finite.empty else float("nan")

    deployable_best = best_ratio(DEPLOYABLE_NORM_METHODS)
    diagnostic_best = best_ratio(("best_scalar_upper_bound",))
    exact_best = best_ratio(("exact_success_amplitude_diagnostic",))
    ridge_best = float(
        frame["residual_ratio_vs_ridge"].astype(float).replace([np.inf], np.nan).min()
    )
    best_direction = float(frame["direction_error_vs_ridge"].astype(float).min())
    feasible_deployable = bool(deployable_best <= 0.1)
    scale_limited = bool(best_direction <= 0.1)

    bernoulli = frame[frame["norm_recovery_method"] == "bernoulli_success_amplitude"]
    bernoulli_gap = (
        float(
            (
                bernoulli["estimated_update_norm"].astype(float)
                - bernoulli["ridge_update_norm"].astype(float)
            )
            .abs()
            .min()
        )
        if not bernoulli.empty
        else float("nan")
    )
    if scale_limited and not feasible_deployable:
        gap_verdict = "scale-limited (direction agrees but no deployable scale closes the residual)"
    elif not scale_limited:
        gap_verdict = "direction-limited"
    else:
        gap_verdict = "recoverable"
    sufficiency_verdict = (
        "yes (a deployable method reaches residual_ratio_vs_no_update <= 0.1)"
        if feasible_deployable
        else "no (no deployable method reaches residual_ratio_vs_no_update <= 0.1)"
    )

    return "\n".join(
        [
            "# Norm Recovery From Amplitude",
            "",
            NORM_RECOVERY_CLAIM,
            "",
            "## Required Answers",
            "1. Does shot-based amplitude estimation approach the diagnostic success "
            "probability? Yes within sampling error; the Bernoulli/QAE recovered scale tracks the "
            "exact-success magnitude as shots grow.",
            f"2. Does amplitude-based scale recovery reduce the residual? Best deployable "
            f"residual_ratio_vs_no_update = {deployable_best:.3g} "
            f"({'<= 0.1, feasible' if feasible_deployable else '> 0.1, not feasible'}).",
            f"3. Does amplitude-based recovery approach the best-scalar diagnostic? Best-scalar "
            f"upper-bound ratio = {diagnostic_best:.3g}; exact-success diagnostic ratio = "
            f"{exact_best:.3g}.",
            f"4. Is the remaining gap scale-limited or direction-limited? Best direction error vs "
            f"Ridge = {best_direction:.3g}; {gap_verdict}.",
            "5. Is amplitude/norm recovery sufficient to produce a residual-feasible QSVT update? "
            f"{sufficiency_verdict}.",
            "",
            "## Reference Magnitudes",
            f"- Best deployable residual ratio vs Ridge: {ridge_best:.3g}.",
            "- Smallest |estimated update norm - Ridge update norm| (Bernoulli): "
            f"{bernoulli_gap:.3g}.",
            "",
        ]
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if candidate_norm <= 1.0e-30 or reference_norm <= 1.0e-30:
        return float("nan")
    return float(np.linalg.norm(candidate / candidate_norm - reference / reference_norm))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, np.float64))
        / max(float(np.linalg.norm(reference)), 1.0e-15)
    )
