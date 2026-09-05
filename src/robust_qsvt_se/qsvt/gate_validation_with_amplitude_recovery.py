from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import (
    bernoulli_amplitude_estimate,
    build_qsvt_amplitude_problem,
    iterative_amplitude_estimate,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.scale_recovery_protocols import compute_bounded_qsvt_action
from robust_qsvt_se.utils.io import ensure_directory

GATE_AMPLITUDE_CLAIM = (
    "Dense gate-level QSVT validation with actual amplitude-based norm recovery on selected "
    "IEEE14-derived subproblems. Configurations are read from the amplitude-recovery search "
    "outputs, never selected by gate-level performance. The success amplitude is estimated with "
    "the implemented small-circuit routine and used to recover the update scale. This is "
    "selected-subproblem dense simulator evidence, not full IEEE-scale hardware execution, and "
    "claims no QSVT superiority over Ridge/Tikhonov, quantum speedup, or quantum advantage."
)

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

GATE_AMPLITUDE_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "norm_recovery_method",
    "shots",
    "gate_run_status",
    "amplitude_estimation_status",
    "success_probability_exact",
    "success_probability_estimated",
    "scale_factor_estimated",
    "gate_residual_raw",
    "gate_residual_amplitude_recovered",
    "ridge_residual",
    "residual_ratio_vs_no_update",
    "state_error_gate_vs_ridge",
    "state_error_gate_vs_polynomial",
    "direction_error_gate_vs_ridge",
    "circuit_depth",
    "two_qubit_gates",
    "residual_feasible_after_gate",
    "dominant_gate_or_readout_limitation",
]


def run_gate_validation_with_amplitude_recovery(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": (
            "outputs/qsvt_residual_feasible_with_amplitude_recovery/residual_feasible_configs.csv"
        ),
        "fallback_input": (
            "outputs/qsvt_residual_feasible_with_amplitude_recovery/diagnostic_feasible_configs.csv"
        ),
        "closest_input": (
            "outputs/qsvt_residual_feasible_with_amplitude_recovery/"
            "all_amplitude_recovery_configs.csv"
        ),
        "max_configs": 3,
        "shots": 1000,
        "seed": 123,
        "case_source": "pypower",
        "submatrix_size": 4,
        "grover_powers": [0, 1, 2, 4, 8],
        "max_manageable_degree": 101,
        "output_dir": "outputs/qsvt_gate_validation_with_amplitude_recovery",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    selected, source = select_amplitude_gate_configs(
        _read_csv(resolved["input"]),
        _read_csv(resolved["fallback_input"]),
        _read_csv(resolved["closest_input"]),
        max_configs=int(resolved["max_configs"]),
        max_manageable_degree=int(resolved["max_manageable_degree"]),
    )
    rows = [
        validate_single_amplitude_gate_config(
            row=row,
            shots=int(resolved["shots"]),
            seed=int(resolved["seed"]),
            case_source=str(resolved["case_source"]),
            default_submatrix_size=int(resolved["submatrix_size"]),
            grover_powers=tuple(int(value) for value in resolved["grover_powers"]),
            source=source,
        )
        for row in selected.to_dict("records")
    ]
    artifacts = write_gate_amplitude_outputs(output_dir, resolved, selected, rows, source=source)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "source": source,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_amplitude_gate_configs(
    feasible_frame: pd.DataFrame,
    diagnostic_frame: pd.DataFrame,
    closest_frame: pd.DataFrame,
    *,
    max_configs: int,
    max_manageable_degree: int = 101,
) -> tuple[pd.DataFrame, str]:
    """Select configs from the amplitude-recovery search outputs only (never gate performance)."""

    feasible = _eligible(feasible_frame, max_manageable_degree)
    if not feasible.empty:
        return _rank(feasible).head(int(max_configs)).reset_index(drop=True), "residual_feasible"

    diagnostic = _eligible(diagnostic_frame, max_manageable_degree)
    if not diagnostic.empty:
        return (
            _rank(diagnostic).head(int(max_configs)).reset_index(drop=True),
            "diagnostic_feasible",
        )

    closest = _eligible(closest_frame, max_manageable_degree)
    if "norm_recovery_method" in closest.columns:
        closest = closest[closest["norm_recovery_method"].isin(DEPLOYABLE_NORM_METHODS)]
    if not closest.empty:
        return _rank(closest).head(int(max_configs)).reset_index(drop=True), "closest_rejected"
    return pd.DataFrame(columns=list(closest_frame.columns)), "none"


def validate_single_amplitude_gate_config(
    *,
    row: dict[str, Any],
    shots: int,
    seed: int,
    case_source: str = "pypower",
    default_submatrix_size: int = 4,
    grover_powers: tuple[int, ...] = (0, 1, 2, 4, 8),
    source: str = "residual_feasible",
) -> dict[str, Any]:
    alpha = float(row["alpha"])
    degree = int(row.get("degree", row.get("requested_degree")))
    case = str(row.get("case", "ieee14"))
    model = str(row.get("model", "ac_linearized"))
    subproblem_id = str(row.get("subproblem_id", f"{case}_{model}_subproblem"))
    target_design = str(row.get("target_design", "current_global"))
    method = str(row.get("norm_recovery_method", "bernoulli_success_amplitude"))
    submatrix_size = _submatrix_size_from_row(row, default_submatrix_size)
    result = _base_row(
        case=case,
        model=model,
        subproblem_id=subproblem_id,
        alpha=alpha,
        degree=degree,
        target_design=target_design,
        method=method,
        shots=int(shots),
    )
    try:
        subproblem = extract_state_estimation_subproblem(
            case=case,
            model=model,
            submatrix_size=submatrix_size,
            seed=int(seed),
            case_source=str(case_source),
        )
        H = np.asarray(subproblem.H_tilde, dtype=np.float64)
        r = np.asarray(subproblem.r_tilde, dtype=np.float64)
        no_update = float(np.linalg.norm(r))
        action = compute_bounded_qsvt_action(H, r, alpha=alpha, degree=degree)
        poly_update = rescale_bounded_target_to_original(
            action.direction, beta=action.beta, C=action.scale_factor_C
        )
        computation = solve_gate_level_state_estimation_problem(
            H_tilde=H,
            r_tilde=r,
            alpha=alpha,
            degree=degree,
            shots=int(shots),
            seed=int(seed),
            metadata=subproblem.metadata,
            transpile_qubit_limit=0,
            export_qasm=False,
        )
        gate_update = np.asarray(computation.qsvt_update, dtype=np.float64)
        ridge_update = np.asarray(computation.ridge_update, dtype=np.float64)
        success_probability_exact = float(computation.summary["success_probability"])
        result["gate_run_status"] = "completed"

        p_estimate, amplitude_status = _estimate_success_amplitude(
            H=H,
            r=r,
            alpha=alpha,
            degree=degree,
            method=method,
            shots=int(shots),
            seed=int(seed),
            grover_powers=grover_powers,
            success_probability_exact=success_probability_exact,
        )
        recovered_update, scale_factor = _apply_amplitude_recovery(
            method=method,
            gate_update=gate_update,
            success_probability_exact=success_probability_exact,
            success_probability_estimate=p_estimate,
            H=H,
            r=r,
            ridge_update=ridge_update,
        )

        gate_residual_raw = _residual(H, gate_update, r)
        gate_residual_recovered = _residual(H, recovered_update, r)
        ridge_residual = _residual(H, ridge_update, r)
        ratio = gate_residual_recovered / no_update if no_update > 0.0 else float("nan")
        direction_error = _direction_error(gate_update, ridge_update)
        feasible_after_gate = bool(
            math.isfinite(ratio) and ratio <= 0.1 and method in DEPLOYABLE_NORM_METHODS
        )
        limitation = _dominant_limitation(
            ratio=ratio,
            direction_error=direction_error,
            gate_update=gate_update,
            H=H,
            r=r,
            no_update=no_update,
            method=method,
        )
        result.update(
            {
                "amplitude_estimation_status": amplitude_status,
                "success_probability_exact": success_probability_exact,
                "success_probability_estimated": float(p_estimate),
                "scale_factor_estimated": float(scale_factor),
                "gate_residual_raw": gate_residual_raw,
                "gate_residual_amplitude_recovered": gate_residual_recovered,
                "ridge_residual": ridge_residual,
                "residual_ratio_vs_no_update": float(ratio),
                "state_error_gate_vs_ridge": _relative_error(gate_update, ridge_update),
                "state_error_gate_vs_polynomial": _relative_error(gate_update, poly_update),
                "direction_error_gate_vs_ridge": float(direction_error),
                "circuit_depth": int(computation.summary["circuit_depth"]),
                "two_qubit_gates": int(computation.summary.get("two_qubit_gate_count", 0)),
                "residual_feasible_after_gate": feasible_after_gate,
                "dominant_gate_or_readout_limitation": limitation,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive; recorded as a failed gate run
        result.update(
            {
                "gate_run_status": "failed",
                "amplitude_estimation_status": "not_run",
                "dominant_gate_or_readout_limitation": _failure_limitation(exc),
            }
        )
    return result


def _estimate_success_amplitude(
    *,
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    degree: int,
    method: str,
    shots: int,
    seed: int,
    grover_powers: tuple[int, ...],
    success_probability_exact: float,
) -> tuple[float, str]:
    if method in {"exact_success_amplitude_diagnostic", "best_scalar_upper_bound"}:
        return success_probability_exact, "exact_diagnostic"
    if method == "iterative_qae":
        try:
            problem = build_qsvt_amplitude_problem(H, r, alpha=float(alpha), degree=int(degree))
            result = iterative_amplitude_estimate(
                problem.unitary_A,
                problem.encoded_dimension,
                powers=grover_powers,
                shots=int(shots),
                seed=int(seed),
                exact_probability=problem.exact_success_probability,
            )
            if result.status == "succeeded" and math.isfinite(result.estimate):
                return float(result.estimate), "iterative_qae_succeeded"
            return (
                bernoulli_amplitude_estimate(
                    success_probability_exact, int(shots), int(seed)
                ).estimate,
                "iterative_qae_unsupported_fell_back_to_bernoulli",
            )
        except Exception as exc:  # pragma: no cover - backend dependent
            return (
                bernoulli_amplitude_estimate(
                    success_probability_exact, int(shots), int(seed)
                ).estimate,
                f"iterative_qae_failed:{type(exc).__name__}",
            )
    estimate = bernoulli_amplitude_estimate(success_probability_exact, int(shots), int(seed))
    return float(estimate.estimate), "bernoulli_succeeded"


def _apply_amplitude_recovery(
    *,
    method: str,
    gate_update: np.ndarray,
    success_probability_exact: float,
    success_probability_estimate: float,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
) -> tuple[np.ndarray, float]:
    gate = np.asarray(gate_update, dtype=np.float64)
    if method == "exact_success_amplitude_diagnostic":
        return gate, 1.0
    if method == "best_scalar_upper_bound":
        scalar = best_scalar_for_residual(H, gate, r)
        return scalar * gate, float(scalar)
    if method == "observable_calibrated":
        index = int(np.argmax(np.abs(ridge_update))) if ridge_update.size else 0
        denominator = float(gate[index]) if gate.size > index else 0.0
        scalar = (
            float(ridge_update[index] / denominator) if abs(denominator) > 1.0e-12 else float("nan")
        )
        update = scalar * gate if math.isfinite(scalar) else np.full_like(gate, np.nan)
        return update, scalar
    # Deployable amplitude methods: scale the known gate output by the estimated/exact
    # success-amplitude ratio sqrt(p_hat / p_exact).
    if success_probability_exact <= 1.0e-15:
        return gate, 1.0
    factor = math.sqrt(max(success_probability_estimate, 0.0) / success_probability_exact)
    return factor * gate, float(factor)


def write_gate_amplitude_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    selected: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Path]:
    selected_path = output_dir / "selected_gate_configs.csv"
    results_path = output_dir / "gate_amplitude_recovery_results.csv"
    interpretation_path = output_dir / "gate_amplitude_recovery_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results = pd.DataFrame(rows)
    for column in GATE_AMPLITUDE_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results = results[GATE_AMPLITUDE_COLUMNS] if not results.empty else results
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(
        gate_amplitude_interpretation(results, source=source), encoding="utf-8"
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "selected_gate_configs": str(selected_path),
            "gate_amplitude_recovery_results": str(results_path),
            "gate_amplitude_recovery_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=GATE_AMPLITUDE_CLAIM,
    )
    return {
        "manifest": manifest,
        "selected_gate_configs": selected_path,
        "gate_amplitude_recovery_results": results_path,
        "gate_amplitude_recovery_interpretation": interpretation_path,
    }


def gate_amplitude_interpretation(frame: pd.DataFrame, *, source: str) -> str:
    lines = [
        "# Gate-Level Validation With Amplitude Recovery",
        "",
        GATE_AMPLITUDE_CLAIM,
        "",
    ]
    if frame.empty:
        lines += [
            "No configurations were available to validate.",
            "No deployable amplitude-recovered configuration was residual-feasible; gate "
            "validation was not run.",
            "",
        ]
        return "\n".join(lines)

    completed = frame[frame["gate_run_status"] == "completed"]
    feasible = frame[frame["residual_feasible_after_gate"] == True]  # noqa: E712
    best = (
        float(completed["gate_residual_amplitude_recovered"].astype(float).min())
        if not completed.empty
        else float("nan")
    )
    if source == "residual_feasible" and not feasible.empty:
        headline = (
            "The small dense QSVT pipeline supports a residual-feasible update under the tested "
            "amplitude-based norm recovery."
        )
    elif source in {"diagnostic_feasible", "closest_rejected"}:
        headline = (
            "No deployable amplitude-recovered configuration was residual-feasible before gate "
            "validation; gate validation was run only as closest-case diagnostic evidence."
        )
    else:
        headline = (
            "The bottleneck shifts from polynomial/norm-recovery diagnostics to gate-level "
            "extraction, readout noise, or circuit-level scaling."
        )
    lines += [
        f"- Source of configurations: {source}",
        f"- Configurations validated: {len(frame)}",
        f"- Successful gate runs: {int((frame['gate_run_status'] == 'completed').sum())}",
        f"- Residual-feasible after gate validation: {len(feasible)}",
        f"- Best amplitude-recovered gate residual: {best:.6g}",
        "",
        headline,
        "",
    ]
    return "\n".join(lines)


def _eligible(frame: pd.DataFrame, max_degree: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    if "degree" in data.columns:
        degrees = pd.to_numeric(data["degree"], errors="coerce")
        data = data[degrees.notna() & (degrees <= int(max_degree))]
    return data


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["residual_ratio_vs_no_update", "direction_error_vs_ridge", "degree"]
        if column in frame.columns
    ]
    if not sort_columns:
        return frame.drop_duplicates()
    data = frame.copy()
    for column in sort_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    ranked = data.sort_values(sort_columns, kind="mergesort")
    subset = [
        column
        for column in ["subproblem_id", "alpha", "degree", "target_design", "norm_recovery_method"]
        if column in ranked.columns
    ]
    return ranked.drop_duplicates(subset=subset) if subset else ranked


def _dominant_limitation(
    *,
    ratio: float,
    direction_error: float,
    gate_update: np.ndarray,
    H: np.ndarray,
    r: np.ndarray,
    no_update: float,
    method: str,
) -> str:
    if math.isfinite(ratio) and ratio <= 0.1:
        # A low gate residual reached only by a non-deployable diagnostic is not deployable.
        return "diagnostic_method_not_deployable" if method in DIAGNOSTIC_NORM_METHODS else "none"
    if math.isfinite(direction_error) and direction_error > 0.25:
        return "gate_extraction_direction"
    best_scalar = best_scalar_for_residual(H, gate_update, r)
    best_scalar_ratio = (
        _residual(H, best_scalar * gate_update, r) / no_update if no_update > 0.0 else float("nan")
    )
    if math.isfinite(best_scalar_ratio) and best_scalar_ratio <= 0.1:
        return "norm_scale_recovery"
    if method in DIAGNOSTIC_NORM_METHODS:
        return "diagnostic_method_not_deployable"
    return "polynomial_target_or_readout"


def _failure_limitation(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "phase" in text or "angle" in text or "synth" in text:
        return "phase_synthesis"
    return "gate_extraction"


def _base_row(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    alpha: float,
    degree: int,
    target_design: str,
    method: str,
    shots: int,
) -> dict[str, Any]:
    row = {column: np.nan for column in GATE_AMPLITUDE_COLUMNS}
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
            "gate_run_status": "not_run",
            "amplitude_estimation_status": "not_run",
            "residual_feasible_after_gate": False,
            "dominant_gate_or_readout_limitation": "none",
        }
    )
    return row


def _submatrix_size_from_row(row: dict[str, Any], default: int) -> int:
    matrix_shape = str(row.get("matrix_shape", ""))
    if "x" in matrix_shape:
        try:
            return int(matrix_shape.split("x", maxsplit=1)[0])
        except ValueError:
            pass
    return int(default)


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if candidate_norm <= 1.0e-30 or reference_norm <= 1.0e-30:
        return float("nan")
    return float(np.linalg.norm(candidate / candidate_norm - reference / reference_norm))


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, np.float64))
        / max(float(np.linalg.norm(reference)), 1.0e-15)
    )
