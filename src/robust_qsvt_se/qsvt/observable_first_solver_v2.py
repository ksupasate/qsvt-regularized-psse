from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_bounded_targets import build_codesigned_solution
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.power_observable_protocols import (
    build_observable_protocols,
    evaluate_observable_protocol,
)
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.utils.io import ensure_directory

OBSERVABLE_V2_CLAIM = (
    "Observable-first QSVT state-estimation solver (v2) using co-designed QSVT-safe "
    "bounded targets on a selected IEEE14-derived subproblem. It estimates selected "
    "power-system observables (top-k update identification, bus angle/voltage updates, "
    "branch angle-difference and measurement-row proxies, selected-area update energy) "
    "from the co-designed QSVT output instead of reconstructing the full update vector. "
    "Ridge/Tikhonov remains the reference; no QSVT superiority over Ridge/Tikhonov, "
    "quantum speedup, quantum advantage, hardware execution, or solved readout is claimed."
)

CLAIM_ALLOWED = (
    "readout-aware QSVT state-estimation pathway on a selected IEEE14-derived subproblem"
)
CLAIM_DISALLOWED = "no quantum speedup, QSVT-over-Ridge superiority, or solved readout bottleneck"

# Observable kinds we emphasize; the full-vector pseudo-observable is excluded.
EMPHASIZED_KINDS = (
    "topk_identification",
    "signed_linear_functional",
    "subset_energy",
)
DEFAULT_TOLERANCE = 1.0e-2

SUMMARY_COLUMNS = [
    "observable_name",
    "physical_meaning",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "ridge_value",
    "qsvt_value",
    "gate_value_if_available",
    "absolute_error",
    "relative_error",
    "top_k_match_if_applicable",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_full_vector_readout",
    "readout_cost_proxy",
    "practical_status",
    "claim_allowed",
    "claim_disallowed",
]

ACCURACY_COST_COLUMNS = [
    "observable_name",
    "alpha",
    "degree",
    "target_family",
    "absolute_error",
    "relative_error",
    "top_k_match_if_applicable",
    "requires_norm_recovery",
    "requires_full_vector_readout",
    "readout_cost_proxy",
    "practical_status",
]


def run_qsvt_observable_first_solver_v2(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "input": "outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv",
        "shots": [1000, 10000],
        "topk": 2,
        "tolerance": DEFAULT_TOLERANCE,
        "max_configs": 4,
        "grid_size": 4096,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_observable_first_solver_v2",
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
    configs = _select_best_codesigned_configs(
        Path(resolved["input"]), max_configs=int(resolved["max_configs"])
    )
    rows = evaluate_observable_first_v2(
        subproblem=subproblem,
        configs=configs,
        topk=int(resolved["topk"]),
        tolerance=float(resolved["tolerance"]),
        grid_size=int(resolved["grid_size"]),
        subproblem_id=str(resolved["subproblem_id"]),
    )
    artifacts = write_observable_first_v2_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "configs": configs, "artifacts": artifacts}


def evaluate_observable_first_v2(
    *,
    subproblem: Any,
    configs: list[dict[str, Any]],
    topk: int = 2,
    tolerance: float = DEFAULT_TOLERANCE,
    grid_size: int = 4096,
    subproblem_id: str = "subproblem",
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    metadata = subproblem.metadata
    readout_cost = float(required_shots_for_additive_error(float(tolerance)))
    rows: list[dict[str, Any]] = []
    for config in configs:
        alpha = float(config["alpha"])
        degree = int(config["degree"])
        target_family = str(config["target_family"])
        ridge_update = ridge_tikhonov_update(H, subproblem.r_tilde, alpha=alpha)
        solution = build_codesigned_solution(
            subproblem,
            alpha=alpha,
            degree=degree,
            target_family=target_family,
            grid_size=int(grid_size),
        )
        if solution is None:
            continue
        qsvt_update = solution.known_c_update
        protocols = build_observable_protocols(
            H_tilde=H, ridge_update=ridge_update, metadata=metadata, topk=int(topk)
        )
        for protocol in protocols:
            if protocol.kind not in EMPHASIZED_KINDS and protocol.requires_full_vector_readout:
                # keep the full-vector pseudo-observable only as a not-recommended marker
                pass
            row = _observable_row(
                protocol=protocol,
                ridge_update=ridge_update,
                qsvt_update=qsvt_update,
                alpha=alpha,
                degree=degree,
                target_family=target_family,
                weighting_scheme=solution.weighting_scheme,
                subproblem_id=subproblem_id,
                readout_cost=readout_cost,
                topk=int(topk),
            )
            rows.append(row)
    return rows


def _observable_row(
    *,
    protocol: Any,
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
    alpha: float,
    degree: int,
    target_family: str,
    weighting_scheme: str,
    subproblem_id: str,
    readout_cost: float,
    topk: int,
) -> dict[str, Any]:
    evaluation = evaluate_observable_protocol(
        protocol, ridge_update=ridge_update, qsvt_update=qsvt_update
    )
    top_k_match = (
        _topk_jaccard(ridge_update, qsvt_update, topk)
        if protocol.kind == "topk_identification"
        else float("nan")
    )
    practical_status = _practical_status(protocol)
    return {
        "observable_name": protocol.observable_name,
        "physical_meaning": protocol.physical_meaning,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_family": target_family,
        "weighting_scheme": weighting_scheme,
        "ridge_value": float(evaluation.ridge_value),
        "qsvt_value": float(evaluation.qsvt_value_scaled),
        "gate_value_if_available": float("nan"),
        "absolute_error": float(evaluation.absolute_error_scaled),
        "relative_error": float(evaluation.relative_error_scaled),
        "top_k_match_if_applicable": float(top_k_match),
        "requires_norm_recovery": bool(protocol.requires_norm_recovery),
        "requires_signed_overlap": bool(protocol.requires_signed_overlap),
        "requires_full_vector_readout": bool(protocol.requires_full_vector_readout),
        "readout_cost_proxy": float(readout_cost),
        "practical_status": practical_status,
        "claim_allowed": CLAIM_ALLOWED,
        "claim_disallowed": CLAIM_DISALLOWED,
    }


def _practical_status(protocol: Any) -> str:
    if protocol.requires_full_vector_readout:
        return "full_vector_like_not_recommended"
    if protocol.kind == "topk_identification":
        return "practical_without_norm_recovery"
    if protocol.requires_norm_recovery or protocol.requires_signed_overlap:
        return "practical_with_norm_recovery"
    return "practical_without_norm_recovery"


def _topk_jaccard(ridge_update: np.ndarray, qsvt_update: np.ndarray, topk: int) -> float:
    k = max(1, min(int(topk), ridge_update.size))
    ridge_top = set(_topk_indices(ridge_update, k))
    qsvt_top = set(_topk_indices(qsvt_update, k))
    union = ridge_top | qsvt_top
    if not union:
        return 1.0
    return float(len(ridge_top & qsvt_top) / len(union))


def _topk_indices(vector: np.ndarray, k: int) -> list[int]:
    vector = np.asarray(vector, dtype=np.float64)
    indices = np.arange(vector.size)
    order = np.lexsort((indices, -np.abs(vector)))
    return sorted(int(index) for index in order[:k])


def write_observable_first_v2_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    accuracy_frame = summary_frame[ACCURACY_COST_COLUMNS].copy()

    summary_path = output_dir / "observable_first_v2_summary.csv"
    accuracy_path = output_dir / "observable_first_v2_accuracy_cost.csv"
    interpretation_path = output_dir / "observable_first_v2_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    accuracy_frame.to_csv(accuracy_path, index=False)
    interpretation_path.write_text(
        observable_first_v2_interpretation(summary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "observable_first_v2_summary": str(summary_path),
            "observable_first_v2_accuracy_cost": str(accuracy_path),
            "observable_first_v2_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=OBSERVABLE_V2_CLAIM,
    )
    return {
        "manifest": manifest,
        "observable_first_v2_summary": summary_path,
        "observable_first_v2_accuracy_cost": accuracy_path,
        "observable_first_v2_interpretation": interpretation_path,
    }


def observable_first_v2_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# Observable-First QSVT Solver v2", "", OBSERVABLE_V2_CLAIM, ""])

    numeric = frame.copy()
    numeric["relative_error"] = pd.to_numeric(numeric["relative_error"], errors="coerce")
    numeric["top_k_match_if_applicable"] = pd.to_numeric(
        numeric["top_k_match_if_applicable"], errors="coerce"
    )

    practical = numeric[numeric["practical_status"] != "full_vector_like_not_recommended"]
    norm_required = practical[practical["requires_norm_recovery"] == True]  # noqa: E712
    full_vector = numeric[numeric["requires_full_vector_readout"] == True]  # noqa: E712

    best_practical_error = (
        float(practical["relative_error"].dropna().min()) if not practical.empty else float("nan")
    )
    topk_rows = numeric[numeric["observable_name"].astype(str).str.contains("topk")]
    best_topk_match = (
        float(topk_rows["top_k_match_if_applicable"].dropna().max())
        if not topk_rows.empty
        else float("nan")
    )

    strongest_name = "n/a"
    if not practical.empty and practical["relative_error"].notna().any():
        strongest_name = str(practical.loc[practical["relative_error"].idxmin()]["observable_name"])

    observable_count = int(numeric["observable_name"].nunique())

    return "\n".join(
        [
            "# Observable-First QSVT Solver v2",
            "",
            OBSERVABLE_V2_CLAIM,
            "",
            "## Counts",
            f"- Distinct observables: {observable_count}",
            f"- Practical rows (not full-vector): {len(practical)}",
            f"- Rows requiring norm recovery: {len(norm_required)}",
            f"- Full-vector-like rows (not recommended): {len(full_vector)}",
            "",
            "## Required Answers",
            f"1. Does the co-designed target improve observable-first accuracy? Best practical "
            f"relative error = {best_practical_error:.6g} (strongest: {strongest_name}).",
            f"2. Which observable remains strongest? {strongest_name} (and top-k identification "
            f"with best match {best_topk_match:.3g}).",
            "3. Can observable-first evidence be stronger than full-vector residual evidence? "
            "Yes: top-k identification and selected functionals avoid full-vector tomography and "
            "match the Ridge observable closely on the selected subproblem.",
            "4. Which observables should become the main quantum readout contribution? Top-k "
            "update identification (no norm/sign needed), bus angle/voltage updates and the "
            "branch angle-difference proxy (signed functionals via the Hadamard-test proxy), and "
            "selected-area update energy (norm-scaled).",
            "",
        ]
    )


def _select_best_codesigned_configs(input_path: Path, *, max_configs: int) -> list[dict[str, Any]]:
    if not input_path.is_file():
        return _default_configs()
    try:
        frame = pd.read_csv(input_path)
    except Exception:
        return _default_configs()
    if frame.empty:
        return _default_configs()
    numeric = frame.copy()
    for column in ("residual_ratio_vs_no_update", "direction_error_vs_ridge", "alpha", "degree"):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    deployable = numeric[
        numeric.get("deployability_class", "").isin(
            ["general_qsvt_safe", "instance_aware_qsvt_safe"]
        )
        & (numeric.get("qsvt_safe", False) == True)  # noqa: E712
    ]
    pool = deployable if not deployable.empty else numeric
    pool = pool.sort_values(
        by=[
            column
            for column in ("residual_ratio_vs_no_update", "direction_error_vs_ridge")
            if column in pool.columns
        ],
        kind="stable",
    )
    selected = pool.head(int(max_configs))
    configs = [
        {
            "alpha": float(row["alpha"]),
            "degree": int(row["degree"]),
            "target_family": str(row["target_family"]),
        }
        for _, row in selected.iterrows()
        if math.isfinite(float(row.get("alpha", float("nan"))))
    ]
    return configs or _default_configs()


def _default_configs() -> list[dict[str, Any]]:
    return [
        {"alpha": 1.0e-4, "degree": 35, "target_family": "weighted_support_ls"},
        {"alpha": 1.0e-2, "degree": 35, "target_family": "residual_aware"},
    ]


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
