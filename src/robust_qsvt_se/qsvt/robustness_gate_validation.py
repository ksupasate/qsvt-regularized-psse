from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import bernoulli_amplitude_estimate
from robust_qsvt_se.qsvt.codesigned_bounded_targets import build_codesigned_solution
from robust_qsvt_se.qsvt.codesigned_gate_validation import _build_codesigned_gate_circuit
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    build_selected_subproblem_from_policy_row,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

ROBUSTNESS_GATE_CLAIM = (
    "Dense gate-level validation of co-designed QSVT-safe targets on additional "
    "criteria-selected IEEE14-derived subproblems beyond the original high-leverage block. "
    "Each candidate subproblem is reconstructed from its selection indices, the co-designed "
    "phases are synthesized from the bounded coefficients, the structured QSVT operator "
    "circuit is simulated exactly, and the success amplitude is estimated with the "
    "implemented small-circuit routine to recover the update scale. This is "
    "selected-subproblem dense simulator evidence, not full IEEE-scale hardware execution, "
    "and claims no QSVT superiority over Ridge/Tikhonov, quantum speedup, or quantum "
    "advantage."
)

RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
DIRECTION_FEASIBLE_MAX = 0.1
CONTROL_MODE = "worst_conditioned_control"

GATE_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "alpha",
    "degree",
    "target_family",
    "gate_status",
    "phase_synthesis_status",
    "gate_residual_scaled",
    "residual_ratio_vs_no_update",
    "state_error_gate_vs_polynomial",
    "state_error_gate_vs_ridge",
    "direction_error_gate_vs_ridge",
    "success_probability_exact",
    "success_probability_estimated",
    "circuit_depth",
    "two_qubit_gates",
    "residual_feasible_after_gate",
    "dominant_limitation",
]

SELECTED_COLUMNS = [
    "selection_reason",
    "subproblem_id",
    "selection_mode",
    "row_indices",
    "col_indices",
    "alpha",
    "degree",
    "target_family",
    "condition_number",
    "residual_ratio_vs_no_update",
]


def run_qsvt_robustness_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_codesigned_subproblem_robustness/"
        "robustness_gate_validation_candidates.csv",
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "max_configs": 3,
        "shots": 1000,
        "seed": 123,
        "grid_size": 4096,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_robustness_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    candidate_frame = _read_csv(Path(resolved["input"]))
    selected = select_robustness_gate_candidates(
        candidate_frame, max_configs=int(resolved["max_configs"])
    )

    rows: list[dict[str, Any]] = []
    if not selected.empty:
        system, matrix_source = _build_system(
            case=str(resolved["case"]),
            model=str(resolved["model"]),
            case_source=str(resolved["case_source"]),
            seed=int(resolved["seed"]),
        )
        for record in selected.to_dict("records"):
            rows.append(
                validate_robustness_gate_config(
                    row=record,
                    system=system,
                    matrix_source=matrix_source,
                    shots=int(resolved["shots"]),
                    seed=int(resolved["seed"]),
                    grid_size=int(resolved["grid_size"]),
                    phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
                )
            )

    artifacts = write_robustness_gate_outputs(output_dir, resolved, selected, rows)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_robustness_gate_candidates(
    candidate_frame: pd.DataFrame, *, max_configs: int
) -> pd.DataFrame:
    """Pick at most ``max_configs`` diverse candidates: one best-conditioned, one
    metadata-mapped, and one non-high-leverage, excluding the worst-conditioned control."""

    if candidate_frame is None or candidate_frame.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS)
    frame = candidate_frame.copy()
    for column in ("condition_number", "residual_ratio_vs_no_update", "degree", "alpha"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "selection_mode" in frame.columns:
        frame = frame[frame["selection_mode"].astype(str) != CONTROL_MODE]
    if frame.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS)

    chosen: dict[Any, list[str]] = {}

    def add(reason: str, index: Any) -> None:
        if index is not None and index in frame.index:
            chosen.setdefault(index, []).append(reason)

    if "condition_number" in frame.columns and frame["condition_number"].notna().any():
        add("best_conditioned", frame["condition_number"].idxmin())
    metadata = frame[frame["selection_mode"].astype(str) == "metadata_mapped"]
    if not metadata.empty:
        add("metadata_mapped", metadata.index[0])
    non_high = frame[frame["selection_mode"].astype(str) != "high_leverage"]
    if not non_high.empty:
        add("non_high_leverage", non_high.index[0])
    # Backfill with remaining candidates so we still report up to max_configs distinct rows.
    for index in frame.index:
        add("additional_candidate", index)

    selected_rows: list[dict[str, Any]] = []
    for index, reasons in chosen.items():
        if len(selected_rows) >= int(max_configs):
            break
        record = frame.loc[index].to_dict()
        record["selection_reason"] = "+".join(dict.fromkeys(reasons))
        selected_rows.append(record)

    out = pd.DataFrame(selected_rows)
    for column in SELECTED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[SELECTED_COLUMNS].reset_index(drop=True)


def validate_robustness_gate_config(
    *,
    row: dict[str, Any],
    system: Any,
    matrix_source: str,
    shots: int,
    seed: int,
    grid_size: int = 4096,
    phase_timeout_seconds: int = 40,
) -> dict[str, Any]:
    base = {column: np.nan for column in GATE_COLUMNS}
    base.update(
        {
            "case": row.get("case", "ieee14"),
            "model": row.get("model", "ac_linearized"),
            "subproblem_id": row.get("subproblem_id", "selected_subproblem"),
            "selection_mode": row.get("selection_mode", "unknown"),
            "alpha": _as_float(row.get("alpha")),
            "degree": _as_int(row.get("degree")),
            "target_family": row.get("target_family", "residual_aware"),
            "gate_status": "not_run",
            "phase_synthesis_status": "not_run",
            "residual_feasible_after_gate": False,
            "dominant_limitation": "not_run",
        }
    )
    try:
        subproblem = _reconstruct_subproblem(row, system=system, matrix_source=matrix_source)
        H = np.asarray(subproblem.H_tilde, dtype=np.float64)
        r = np.asarray(subproblem.r_tilde, dtype=np.float64)
        no_update = float(np.linalg.norm(r))
        alpha = float(row["alpha"])
        degree = int(float(row["degree"]))
        target_family = str(row.get("target_family", "residual_aware"))

        ridge_update = ridge_tikhonov_update(H, r, alpha=alpha)
        ridge_residual = float(np.linalg.norm(H @ ridge_update - r))

        solution = build_codesigned_solution(
            subproblem,
            alpha=alpha,
            degree=degree,
            target_family=target_family,
            grid_size=int(grid_size),
        )
        if solution is None:
            base.update(
                {
                    "gate_status": "failed",
                    "phase_synthesis_status": "polynomial_construction_failed",
                    "dominant_limitation": "polynomial_construction",
                }
            )
            return base

        gate = _build_codesigned_gate_circuit(
            H=H,
            r=r,
            alpha=alpha,
            solution=solution,
            phase_timeout_seconds=int(phase_timeout_seconds),
        )
        if gate.get("status") != "synthesized":
            base.update(
                {
                    "gate_status": "not_run",
                    "phase_synthesis_status": gate.get("status", "runtime_limited"),
                    "success_probability_exact": float(solution.success_probability_proxy),
                    "dominant_limitation": "phase_synthesis_runtime_limited",
                }
            )
            return base

        gate_update = np.asarray(gate["gate_update"], dtype=np.float64)
        p_exact = float(gate["success_probability_exact"])
        estimate = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed))
        scale_factor = (
            math.sqrt(max(float(estimate.estimate), 0.0) / p_exact) if p_exact > 1.0e-15 else 1.0
        )
        scaled_update = scale_factor * gate_update
        gate_residual_scaled = float(np.linalg.norm(H @ scaled_update - r))
        ratio = gate_residual_scaled / no_update if no_update > 0.0 else float("nan")
        direction_error = _direction_error(scaled_update, ridge_update)
        feasible_after_gate = math.isfinite(ratio) and ratio <= RESIDUAL_RATIO_FEASIBLE_MAX

        base.update(
            {
                "gate_status": "completed",
                "phase_synthesis_status": "synthesized",
                "gate_residual_scaled": gate_residual_scaled,
                "residual_ratio_vs_no_update": float(ratio),
                "state_error_gate_vs_polynomial": _relative_error(
                    gate_update, solution.known_c_update
                ),
                "state_error_gate_vs_ridge": _relative_error(scaled_update, ridge_update),
                "direction_error_gate_vs_ridge": float(direction_error),
                "success_probability_exact": float(p_exact),
                "success_probability_estimated": float(estimate.estimate),
                "circuit_depth": int(gate["circuit_depth"]),
                "two_qubit_gates": int(gate["two_qubit_gates"]),
                "residual_feasible_after_gate": bool(feasible_after_gate),
                "dominant_limitation": _dominant_limitation(ratio, direction_error),
                "ridge_residual": float(ridge_residual),
            }
        )
        return base
    except Exception as exc:
        base.update(
            {
                "gate_status": "failed",
                "phase_synthesis_status": "error",
                "dominant_limitation": f"error:{type(exc).__name__}",
            }
        )
        return base


def _reconstruct_subproblem(
    row: dict[str, Any], *, system: Any, matrix_source: str
) -> SelectedSubproblem:
    policy_row = {
        "case": row.get("case", "ieee14"),
        "model": row.get("model", "ac_linearized"),
        "candidate_id": row.get("subproblem_id", "selected_subproblem"),
        "selection_source": row.get("selection_mode", "selection_policy"),
        "row_indices": row.get("row_indices", ""),
        "col_indices": row.get("col_indices", ""),
    }
    return build_selected_subproblem_from_policy_row(
        system=system, matrix_source=matrix_source, row=policy_row
    )


def _dominant_limitation(ratio: float, direction_error: float) -> str:
    if math.isfinite(ratio) and ratio <= RESIDUAL_RATIO_FEASIBLE_MAX:
        return "none"
    if not math.isfinite(direction_error) or direction_error > DIRECTION_FEASIBLE_MAX:
        return "gate_extraction_direction"
    return "polynomial_target_or_readout"


def write_robustness_gate_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    selected: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    results = pd.DataFrame(rows)
    for column in GATE_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results = results[GATE_COLUMNS]

    selected_path = output_dir / "robustness_gate_selected_configs.csv"
    results_path = output_dir / "robustness_gate_results.csv"
    interpretation_path = output_dir / "robustness_gate_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(robustness_gate_interpretation(results), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "robustness_gate_selected_configs": str(selected_path),
            "robustness_gate_results": str(results_path),
            "robustness_gate_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=ROBUSTNESS_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "robustness_gate_selected_configs": selected_path,
        "robustness_gate_results": results_path,
        "robustness_gate_interpretation": interpretation_path,
    }


def robustness_gate_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# Robustness Gate-Level Validation",
                "",
                ROBUSTNESS_GATE_CLAIM,
                "",
                "- No robustness candidates were available for gate validation.",
                "",
            ]
        )
    completed = frame[frame["gate_status"] == "completed"]
    feasible = completed[completed["residual_feasible_after_gate"] == True]  # noqa: E712
    non_high = feasible[feasible["selection_mode"].astype(str) != "high_leverage"]
    transferred = not non_high.empty

    if transferred:
        headline = (
            "The solver prototype transfers to a small criteria-selected family of "
            "IEEE-derived subproblems."
        )
    else:
        headline = (
            "The solver prototype remains validated only on the original high-leverage block."
        )

    best_ratio = (
        float(pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )
    feasible_modes = sorted(set(feasible["selection_mode"].astype(str)))

    return "\n".join(
        [
            "# Robustness Gate-Level Validation",
            "",
            ROBUSTNESS_GATE_CLAIM,
            "",
            "## Counts",
            f"- Configurations validated: {len(frame)}",
            f"- Completed gate runs: {len(completed)}",
            f"- Residual-feasible after gate: {len(feasible)}",
            f"- Feasible selection modes after gate: {', '.join(feasible_modes) or 'none'}",
            "",
            "## Best Gate Result",
            f"- Best residual_ratio_vs_no_update: {best_ratio:.6g}",
            "",
            "## Headline",
            f"- {headline}",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate - reference) / max(float(np.linalg.norm(reference)), 1.0e-15)
    )


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - np.clip(cosine, -1.0, 1.0)))))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
