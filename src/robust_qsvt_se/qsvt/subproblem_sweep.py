from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.dc_linear import build_dc_weighted_system
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    _subproblem_metadata,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.utils.io import ensure_directory
from robust_qsvt_se.utils.seed import make_rng

SUBPROBLEM_SWEEP_CLAIM = (
    "This is a deterministic subproblem robustness sweep for selected "
    "IEEE-derived gate-level QSVT update evidence. It does not establish full "
    "IEEE-scale gate-level solving or QSVT superiority over Ridge/Tikhonov."
)


def run_gate_level_qsvt_subproblem_sweep(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": [
            "top_left",
            "random_seeded",
            "best_conditioned",
            "worst_conditioned",
            "high_leverage",
        ],
        "alpha": 1.0e-4,
        "degree": 51,
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/gate_level_qsvt_subproblem_sweep",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    system, matrix_source = _build_system(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        seed=int(resolved["seed"]),
    )
    summary_rows: list[dict[str, Any]] = []
    for mode in [str(value) for value in resolved["selection_modes"]]:
        try:
            subproblem = select_subproblem(
                system=system,
                matrix_source=matrix_source,
                case=str(resolved["case"]),
                model=str(resolved["model"]),
                submatrix_size=int(resolved["submatrix_size"]),
                selection_mode=mode,
                seed=int(resolved["seed"]),
            )
            computation = solve_gate_level_state_estimation_problem(
                H_tilde=subproblem.H_tilde,
                r_tilde=subproblem.r_tilde,
                alpha=float(resolved["alpha"]),
                degree=int(resolved["degree"]),
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                metadata=subproblem.metadata,
                transpile_qubit_limit=0,
                export_qasm=False,
            )
            row = _summary_row(mode, subproblem, computation)
        except Exception as exc:  # pragma: no cover - version/resource dependent
            row = _failure_row(mode, int(resolved["submatrix_size"]), exc)
        summary_rows.append(row)

    artifacts = _write_outputs(output_dir, resolved, summary_rows)
    return {"output_dir": output_dir, "summary_rows": summary_rows, "artifacts": artifacts}


def select_subproblem(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case: str,
    model: str,
    submatrix_size: int,
    selection_mode: str,
    seed: int,
) -> SelectedSubproblem:
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    size = int(submatrix_size)
    if size <= 0 or size > min(H_full.shape):
        raise ValueError("submatrix_size must fit the weighted system")
    mode = str(selection_mode)
    if mode == "top_left":
        rows = np.arange(size)
        columns = np.arange(size)
    elif mode == "random_seeded":
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(H_full.shape[0], size=size, replace=False))
        columns = np.sort(rng.choice(H_full.shape[1], size=size, replace=False))
    elif mode == "high_leverage":
        columns = _top_indices(np.linalg.norm(H_full, axis=0), size)
        rows = _top_indices(np.linalg.norm(H_full[:, columns], axis=1), size)
    elif mode == "physically_mapped_if_available":
        columns = _physically_mapped_columns(system, size)
        rows = _top_indices(np.linalg.norm(H_full[:, columns], axis=1), size)
    elif mode in {"best_conditioned", "worst_conditioned"}:
        rows, columns = _conditioned_selection(
            H_full,
            size=size,
            choose_best=mode == "best_conditioned",
        )
    else:
        raise ValueError(f"unsupported selection mode: {selection_mode}")
    H_sub = H_full[np.ix_(rows, columns)]
    metadata = _subproblem_metadata(
        system=system,
        matrix_source=matrix_source,
        case=case,
        model=model,
        selected_rows=np.asarray(rows, dtype=np.int64),
        selected_columns=np.asarray(columns, dtype=np.int64),
    )
    metadata["selection_mode"] = mode
    return SelectedSubproblem(H_tilde=H_sub, r_tilde=r_full[rows], metadata=metadata)


def classify_subproblem(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "unstable_or_condition_sensitive"
    ratio = float(row["residual_reduction_ratio"])
    condition = float(row["condition_number"])
    state_error = float(row["state_error_vs_ridge"])
    if not np.isfinite(ratio) or not np.isfinite(condition):
        return "unstable_or_condition_sensitive"
    if condition > 1.0e8 or state_error > 1.5:
        return "unstable_or_condition_sensitive"
    if ratio < 0.25:
        return "strong_residual_reduction"
    if ratio < 0.9:
        return "moderate_residual_reduction"
    return "no_residual_reduction"


def _build_system(
    *,
    case: str,
    model: str,
    case_source: str,
    seed: int,
) -> tuple[WeightedSystem, str]:
    if model == "ac_linearized":
        return build_engineering_system(
            {
                "case_name": case,
                "case_source": case_source,
                "matrix_source": f"{case}_ac_weighted_jacobian",
                "seed": int(seed),
            }
        )
    if model == "dc_linearized":
        system = build_dc_weighted_system(
            case_name=case,
            case_source=case_source,
            angle_scale=0.05,
            measurement_config={
                "include_branch_flows": True,
                "include_bus_injections": True,
                "angle_buses": (),
                "flow_std": 0.02,
                "injection_std": 0.03,
            },
            rng=make_rng(seed),
            metadata={"qsvt_subproblem_sweep_source": True},
        )
        return system, f"{case}_{case_source}_dc_weighted_jacobian"
    raise ValueError("model must be ac_linearized or dc_linearized")


def _conditioned_selection(
    H_full: np.ndarray,
    *,
    size: int,
    choose_best: bool,
) -> tuple[np.ndarray, np.ndarray]:
    column_pool_size = min(max(size + 6, 2 * size), H_full.shape[1])
    column_pool = _top_indices(np.linalg.norm(H_full, axis=0), column_pool_size)
    best_rows: np.ndarray | None = None
    best_columns: np.ndarray | None = None
    best_score: float | None = None
    for column_tuple in combinations(column_pool.tolist(), size):
        columns = np.asarray(column_tuple, dtype=np.int64)
        rows = _top_indices(np.linalg.norm(H_full[:, columns], axis=1), size)
        singular_values = np.linalg.svd(H_full[np.ix_(rows, columns)], compute_uv=False)
        condition = _condition_number(singular_values)
        score = condition if np.isfinite(condition) else 1.0e300
        if best_score is None:
            take = True
        elif choose_best:
            take = score < best_score
        else:
            take = score > best_score
        if take:
            best_score = score
            best_rows = rows
            best_columns = columns
    if best_rows is None or best_columns is None:
        raise RuntimeError("conditioned selection did not produce a candidate")
    return np.sort(best_rows), np.sort(best_columns)


def _physically_mapped_columns(system: WeightedSystem, size: int) -> np.ndarray:
    angle_indices = [int(value) for value in system.metadata.get("angle_state_indices", [])]
    voltage_indices = [
        int(value) for value in system.metadata.get("voltage_magnitude_state_indices", [])
    ]
    selected = (angle_indices[: size // 2] + voltage_indices[: size - size // 2])[:size]
    if len(selected) < size:
        selected = list(range(size))
    return np.asarray(sorted(selected), dtype=np.int64)


def _summary_row(
    mode: str,
    subproblem: SelectedSubproblem,
    computation: Any,
) -> dict[str, Any]:
    H = subproblem.H_tilde
    r = subproblem.r_tilde
    singular_values = np.linalg.svd(H, compute_uv=False)
    best_scalar = best_scalar_for_residual(H, computation.qsvt_update, r)
    best_scaled_residual = float(np.linalg.norm(H @ (best_scalar * computation.qsvt_update) - r))
    row = {
        "selection_mode": mode,
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "condition_number": _condition_number(singular_values),
        "sigma_min": float(np.min(singular_values)),
        "sigma_max": float(np.max(singular_values)),
        "ridge_residual": computation.summary["residual_after_ridge_update"],
        "qsvt_residual": computation.summary["residual_after_qsvt_update"],
        "no_update_residual": computation.summary["residual_before_update"],
        "residual_reduction_ratio": computation.summary["residual_reduction_ratio_vs_no_update"],
        "state_error_vs_ridge": computation.summary["phase_or_sign_aligned_state_error"],
        "best_scalar": best_scalar,
        "best_scalar_rescaled_residual": best_scaled_residual,
        "success_probability": computation.summary["success_probability"],
        "observable_error": computation.summary["max_exact_observable_error"],
        "selected_rows": " ".join(
            str(value) for value in subproblem.metadata["selected_measurement_indices"]
        ),
        "selected_columns": " ".join(
            str(value) for value in subproblem.metadata["selected_state_indices"]
        ),
        "run_status": "completed",
        "failure_reason_if_any": "",
    }
    row["classification"] = classify_subproblem(row)
    return row


def _failure_row(mode: str, size: int, exc: Exception) -> dict[str, Any]:
    row = {
        "selection_mode": mode,
        "matrix_shape": f"{size}x{size}",
        "condition_number": np.nan,
        "sigma_min": np.nan,
        "sigma_max": np.nan,
        "ridge_residual": np.nan,
        "qsvt_residual": np.nan,
        "no_update_residual": np.nan,
        "residual_reduction_ratio": np.nan,
        "state_error_vs_ridge": np.nan,
        "best_scalar": np.nan,
        "best_scalar_rescaled_residual": np.nan,
        "success_probability": np.nan,
        "observable_error": np.nan,
        "selected_rows": "",
        "selected_columns": "",
        "run_status": "failed",
        "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
    }
    row["classification"] = classify_subproblem(row)
    return row


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_path = output_dir / "subproblem_summary.csv"
    residual_path = output_dir / "residual_vs_conditioning.csv"
    state_path = output_dir / "state_error_vs_conditioning.csv"
    best_worst_path = output_dir / "best_and_worst_cases.md"
    interpretation_path = output_dir / "subproblem_sweep_interpretation.md"
    frame = pd.DataFrame(rows)
    frame.to_csv(summary_path, index=False)
    frame[
        [
            "selection_mode",
            "condition_number",
            "no_update_residual",
            "qsvt_residual",
            "ridge_residual",
            "residual_reduction_ratio",
            "classification",
        ]
    ].to_csv(residual_path, index=False)
    frame[
        [
            "selection_mode",
            "condition_number",
            "state_error_vs_ridge",
            "best_scalar_rescaled_residual",
            "classification",
        ]
    ].to_csv(state_path, index=False)
    best_worst_path.write_text(_best_worst_markdown(rows), encoding="utf-8")
    interpretation_path.write_text(_interpretation_markdown(rows), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "subproblem_summary": str(summary_path),
            "residual_vs_conditioning": str(residual_path),
            "state_error_vs_conditioning": str(state_path),
            "best_and_worst_cases": str(best_worst_path),
            "subproblem_sweep_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=SUBPROBLEM_SWEEP_CLAIM,
    )
    return {
        "manifest": manifest,
        "subproblem_summary": summary_path,
        "residual_vs_conditioning": residual_path,
        "state_error_vs_conditioning": state_path,
        "best_and_worst_cases": best_worst_path,
        "subproblem_sweep_interpretation": interpretation_path,
    }


def _best_worst_markdown(rows: list[dict[str, Any]]) -> str:
    completed = [row for row in rows if row["run_status"] == "completed"]
    if not completed:
        return "# Best and Worst Subproblem Cases\n\nNo completed cases.\n"
    best = min(completed, key=lambda row: float(row["residual_reduction_ratio"]))
    worst = max(completed, key=lambda row: float(row["residual_reduction_ratio"]))
    return "\n".join(
        [
            "# Best and Worst Subproblem Cases",
            "",
            "- Best residual ratio: "
            f"{best['selection_mode']} = {best['residual_reduction_ratio']:.17g}",
            "- Worst residual ratio: "
            f"{worst['selection_mode']} = {worst['residual_reduction_ratio']:.17g}",
            "",
            SUBPROBLEM_SWEEP_CLAIM,
            "",
        ]
    )


def _interpretation_markdown(rows: list[dict[str, Any]]) -> str:
    counts = pd.Series([row["classification"] for row in rows]).value_counts().to_dict()
    strong = int(counts.get("strong_residual_reduction", 0))
    moderate = int(counts.get("moderate_residual_reduction", 0))
    robust = strong + moderate >= 3
    statement = (
        "Multiple deterministic selection modes show residual reduction."
        if robust
        else "Residual reduction is not consistent across deterministic selection modes."
    )
    return "\n".join(
        [
            "# Gate-Level QSVT Subproblem Sweep Interpretation",
            "",
            SUBPROBLEM_SWEEP_CLAIM,
            "",
            f"- Class counts: {counts}",
            f"- Robustness statement: {statement}",
            "- Do not claim robustness unless multiple modes show consistent reduction.",
            "",
        ]
    )


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[: int(count)])


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())
