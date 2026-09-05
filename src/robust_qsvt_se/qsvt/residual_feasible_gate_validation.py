from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.scale_recovery_protocols import compute_bounded_qsvt_action
from robust_qsvt_se.utils.io import ensure_directory

RESIDUAL_FEASIBLE_GATE_CLAIM = (
    "Dense gate-level QSVT validation is run only for configurations read from the "
    "criteria-based residual-feasible configuration search, never selected by gate-level "
    "QSVT performance. It is selected-subproblem dense simulator evidence, not full "
    "IEEE-scale hardware execution, and claims no QSVT superiority over Ridge/Tikhonov."
)

# Deployable / proxy magnitude-recovery protocols that estimate the physical QSVT output
# magnitude (on the simulator the estimate equals the physically rescaled gate output).
PHYSICAL_OUTPUT_PROTOCOLS = (
    "known_C",
    "success_amplitude_proxy",
    "amplitude_estimation_proxy",
)
# Protocols not eligible to drive a deployable update (kept only as diagnostics in Phase 3).
DIAGNOSTIC_PROTOCOLS = ("best_scalar_diagnostic", "ridge_norm")
NON_GATE_PROTOCOLS = ("matrix_free_polynomial_residual",)

GATE_VALIDATION_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "scale_protocol",
    "poly_residual",
    "gate_residual_raw",
    "gate_residual_scaled",
    "ridge_residual",
    "residual_ratio_vs_no_update",
    "state_error_gate_vs_poly",
    "state_error_gate_vs_ridge",
    "success_probability",
    "circuit_depth",
    "two_qubit_gates",
    "validation_status",
    "dominant_gate_limitation",
]


def run_residual_feasible_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_residual_feasible_config_search/residual_feasible_configs.csv",
        "fallback_input": "outputs/qsvt_residual_feasible_config_search/all_config_results.csv",
        "fallback": True,
        "max_configs": 3,
        "shots": 1000,
        "seed": 123,
        "case_source": "pypower",
        "submatrix_size": 4,
        "max_manageable_degree": 101,
        "condition_max": 1.0e8,
        "output_dir": "outputs/qsvt_residual_feasible_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    feasible_frame = _read_csv(resolved["input"])
    fallback_frame = (
        _read_csv(resolved["fallback_input"]) if bool(resolved["fallback"]) else pd.DataFrame()
    )
    selected, used_fallback = select_gate_validation_configs(
        feasible_frame,
        fallback_frame,
        max_configs=int(resolved["max_configs"]),
        max_manageable_degree=int(resolved["max_manageable_degree"]),
        condition_max=float(resolved["condition_max"]),
        fallback=bool(resolved["fallback"]),
    )
    rows = [
        validate_single_gate_config(
            row=row,
            shots=int(resolved["shots"]),
            seed=int(resolved["seed"]),
            case_source=str(resolved["case_source"]),
            default_submatrix_size=int(resolved["submatrix_size"]),
            used_fallback=used_fallback,
        )
        for row in selected.to_dict("records")
    ]
    artifacts = write_gate_validation_outputs(
        output_dir, resolved, selected, rows, used_fallback=used_fallback
    )
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "used_fallback": used_fallback,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_gate_validation_configs(
    feasible_frame: pd.DataFrame,
    fallback_frame: pd.DataFrame,
    *,
    max_configs: int,
    max_manageable_degree: int = 101,
    condition_max: float = 1.0e8,
    fallback: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Select up to ``max_configs`` configs FROM the Phase 3 output files only.

    Feasibility is read from disk; this function never re-derives feasibility from
    gate-level QSVT performance. When no residual-feasible config exists it optionally
    falls back to the closest deployable-protocol configurations from the full results.
    """

    feasible = _eligible(feasible_frame, max_manageable_degree, condition_max)
    if "residual_feasible" in feasible.columns:
        feasible = feasible[feasible["residual_feasible"].astype(str).str.lower().eq("true")]
    if not feasible.empty:
        ranked = _rank(feasible)
        deduped = ranked.drop_duplicates(subset=["subproblem_id", "alpha", "degree"])
        return deduped.head(int(max_configs)).reset_index(drop=True), False

    if not fallback or fallback_frame.empty:
        return pd.DataFrame(columns=list(fallback_frame.columns)), False

    pool = _eligible(fallback_frame, max_manageable_degree, condition_max)
    if "scale_protocol" in pool.columns:
        excluded = set(DIAGNOSTIC_PROTOCOLS) | set(NON_GATE_PROTOCOLS)
        pool = pool[~pool["scale_protocol"].isin(excluded)]
    if pool.empty:
        return pd.DataFrame(columns=list(fallback_frame.columns)), True
    ranked = _rank(pool)
    deduped = ranked.drop_duplicates(subset=["subproblem_id", "alpha", "degree"])
    return deduped.head(int(max_configs)).reset_index(drop=True), True


def validate_single_gate_config(
    *,
    row: dict[str, Any],
    shots: int,
    seed: int,
    case_source: str = "pypower",
    default_submatrix_size: int = 4,
    used_fallback: bool = False,
) -> dict[str, Any]:
    alpha = float(row["alpha"])
    degree = int(row.get("degree", row.get("requested_degree")))
    case = str(row.get("case", "ieee14"))
    model = str(row.get("model", "ac_linearized"))
    subproblem_id = str(row.get("subproblem_id", f"{case}_{model}_subproblem"))
    target_design = str(row.get("target_design", "current_global"))
    scale_protocol = str(row.get("scale_protocol", "known_C"))
    submatrix_size = _submatrix_size_from_row(row, default_submatrix_size)
    result = _base_row(
        case=case,
        model=model,
        subproblem_id=subproblem_id,
        alpha=alpha,
        degree=degree,
        target_design=target_design,
        scale_protocol=scale_protocol,
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
        success_probability = float(computation.summary["success_probability"])

        scaled_update, _ = apply_scale_protocol(
            scale_protocol,
            gate_update=gate_update,
            H=H,
            r=r,
            ridge_update=ridge_update,
            success_probability=success_probability,
        )
        no_update = float(np.linalg.norm(r))
        gate_residual_raw = _residual(H, gate_update, r)
        gate_residual_scaled = _residual(H, scaled_update, r)
        ridge_residual = _residual(H, ridge_update, r)
        best_scalar = best_scalar_for_residual(H, gate_update, r)
        best_scalar_residual = _residual(H, best_scalar * gate_update, r)
        direction_error = _direction_error(gate_update, ridge_update)
        ratio_scaled = gate_residual_scaled / no_update if no_update > 0.0 else float("nan")

        status = _validation_status(ratio_scaled=ratio_scaled, used_fallback=used_fallback)
        limitation = _dominant_gate_limitation(
            direction_error=direction_error,
            ratio_scaled=ratio_scaled,
            best_scalar_ratio=best_scalar_residual / no_update if no_update > 0.0 else float("nan"),
            condition_number=float(row.get("condition_number", _condition_number(H))),
            poly_ratio=action.known_c_residual / no_update if no_update > 0.0 else float("nan"),
            degree=degree,
        )
        result.update(
            {
                "poly_residual": float(action.known_c_residual),
                "gate_residual_raw": gate_residual_raw,
                "gate_residual_scaled": gate_residual_scaled,
                "ridge_residual": ridge_residual,
                "residual_ratio_vs_no_update": float(ratio_scaled),
                "state_error_gate_vs_poly": _relative_error(gate_update, poly_update),
                "state_error_gate_vs_ridge": _relative_error(gate_update, ridge_update),
                "success_probability": success_probability,
                "circuit_depth": int(computation.summary["circuit_depth"]),
                "two_qubit_gates": int(computation.summary.get("two_qubit_gate_count", 0)),
                "validation_status": status,
                "dominant_gate_limitation": limitation,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive; recorded as a failed validation
        result.update(
            {
                "validation_status": "failed",
                "dominant_gate_limitation": _failure_limitation(exc),
            }
        )
    return result


def apply_scale_protocol(
    protocol: str,
    *,
    gate_update: np.ndarray,
    H: np.ndarray,
    r: np.ndarray,
    ridge_update: np.ndarray,
    success_probability: float,
) -> tuple[np.ndarray, float]:
    """Apply a Phase-1 scale-recovery protocol as a scalar on the gate output direction.

    The dense gate solver already returns the physically rescaled (known-C-style) update,
    so the deployable/proxy protocols that estimate that same physical magnitude reduce to
    the raw output on the simulator; the diagnostics (best-scalar, ridge-norm) recover the
    magnitude-optimal or ridge-matched scaling and are reported for comparison only.
    """

    gate = np.asarray(gate_update, dtype=np.float64)
    gate_norm = float(np.linalg.norm(gate))
    if protocol in PHYSICAL_OUTPUT_PROTOCOLS:
        return gate, 1.0
    if protocol == "best_scalar_diagnostic":
        scalar = best_scalar_for_residual(H, gate, r)
        return scalar * gate, float(scalar)
    if protocol == "ridge_norm":
        ridge_norm = float(np.linalg.norm(ridge_update))
        scalar = ridge_norm / gate_norm if gate_norm > 0.0 else float("nan")
        return scalar * gate, float(scalar)
    if protocol == "observable_calibrated":
        index = int(np.argmax(np.abs(ridge_update))) if ridge_update.size else 0
        denom = float(gate[index]) if gate.size > index else 0.0
        scalar = float(ridge_update[index] / denom) if abs(denom) > 1.0e-12 else float("nan")
        update = scalar * gate if math.isfinite(scalar) else np.full_like(gate, np.nan)
        return update, scalar
    return gate, 1.0


def write_gate_validation_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    selected: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    used_fallback: bool,
) -> dict[str, Path]:
    selected_path = output_dir / "selected_gate_configs.csv"
    results_path = output_dir / "gate_validation_results.csv"
    interpretation_path = output_dir / "gate_validation_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results = pd.DataFrame(rows)
    for column in GATE_VALIDATION_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results = results[GATE_VALIDATION_COLUMNS] if not results.empty else results
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(
        gate_validation_interpretation(results, used_fallback=used_fallback),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "selected_gate_configs": str(selected_path),
            "gate_validation_results": str(results_path),
            "gate_validation_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=RESIDUAL_FEASIBLE_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "selected_gate_configs": selected_path,
        "gate_validation_results": results_path,
        "gate_validation_interpretation": interpretation_path,
    }


def gate_validation_interpretation(frame: pd.DataFrame, *, used_fallback: bool) -> str:
    lines = ["# Residual-Feasible Gate-Level Validation", "", RESIDUAL_FEASIBLE_GATE_CLAIM, ""]
    if frame.empty:
        lines += [
            "No configurations were available to validate.",
            "The current bounded target and scale-recovery designs do not produce "
            "Ridge-feasible updates on the selected IEEE-derived subproblems.",
            "",
        ]
        return "\n".join(lines)

    completed = frame[frame["validation_status"] != "failed"].copy()
    feasible_preserved = frame[frame["validation_status"] == "residual_feasible_preserved"]
    best_scaled = (
        float(completed["gate_residual_scaled"].astype(float).min())
        if not completed.empty
        else float("nan")
    )
    if used_fallback:
        headline = (
            "No residual-feasible configuration existed in the Phase 3 search. "
            "The current bounded target and scale-recovery designs do not produce "
            "Ridge-feasible updates on the selected IEEE-derived subproblems. "
            "The closest deployable-protocol configurations were exercised through the "
            "dense gate-level pipeline for completeness (validation_status="
            "exercised_not_feasible)."
        )
    elif not feasible_preserved.empty:
        headline = (
            "The selected dense gate-level QSVT pipeline supports a residual-feasible "
            "update under the tested scale protocol."
        )
    else:
        headline = (
            "Polynomial configurations entered gate validation but the gate output did "
            "not preserve residual feasibility: the bottleneck shifts to gate-level "
            "implementation or extraction."
        )
    source = "fallback (closest)" if used_fallback else "Phase 3 residual_feasible_configs.csv"
    lines += [
        f"- Source of configurations: {source}",
        f"- Configurations validated: {len(frame)}",
        f"- Successful gate runs: {int((frame['validation_status'] != 'failed').sum())}",
        f"- Residual-feasibility preserved at gate level: {len(feasible_preserved)}",
        f"- Best scaled gate residual: {best_scaled:.6g}",
        "",
        headline,
        "",
    ]
    return "\n".join(lines)


def _eligible(frame: pd.DataFrame, max_degree: int, condition_max: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    if "degree" in data.columns:
        degrees = pd.to_numeric(data["degree"], errors="coerce")
        data = data[degrees.notna() & (degrees <= int(max_degree))]
    if "condition_number" in data.columns:
        condition = pd.to_numeric(data["condition_number"], errors="coerce")
        data = data[
            condition.notna() & np.isfinite(condition) & (condition <= float(condition_max))
        ]
    return data


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["residual_ratio_vs_no_update", "direction_error_vs_ridge", "degree"]
        if column in frame.columns
    ]
    if not sort_columns:
        return frame
    data = frame.copy()
    for column in sort_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(sort_columns, kind="mergesort")


def _validation_status(*, ratio_scaled: float, used_fallback: bool) -> str:
    if math.isfinite(ratio_scaled) and ratio_scaled <= 0.1:
        return "residual_feasible_preserved"
    return "exercised_not_feasible" if used_fallback else "feasible_lost_at_gate"


def _dominant_gate_limitation(
    *,
    direction_error: float,
    ratio_scaled: float,
    best_scalar_ratio: float,
    condition_number: float,
    poly_ratio: float,
    degree: int,
) -> str:
    if math.isfinite(ratio_scaled) and ratio_scaled <= 0.1:
        return "none"
    if math.isfinite(direction_error) and direction_error > 0.25:
        return "gate_extraction"
    if (
        math.isfinite(best_scalar_ratio)
        and best_scalar_ratio <= 0.1
        and (not math.isfinite(ratio_scaled) or ratio_scaled > 0.1)
    ):
        return "norm_sign_recovery"
    if math.isfinite(condition_number) and condition_number > 1.0e8:
        return "conditioning"
    if math.isfinite(poly_ratio) and poly_ratio > 0.1 and int(degree) <= 51:
        return "degree_limited"
    return "gate_extraction"


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
    scale_protocol: str,
) -> dict[str, Any]:
    row = {column: np.nan for column in GATE_VALIDATION_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "alpha": float(alpha),
            "degree": int(degree),
            "target_design": target_design,
            "scale_protocol": scale_protocol,
            "validation_status": "not_run",
            "dominant_gate_limitation": "none",
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


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_values - reference_values)
        / max(float(np.linalg.norm(reference_values)), 1.0e-15)
    )
