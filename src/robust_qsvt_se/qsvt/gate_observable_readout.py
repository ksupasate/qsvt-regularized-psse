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
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.power_observable_protocols import (
    build_observable_protocols,
    evaluate_observable_protocol,
)
from robust_qsvt_se.utils.io import ensure_directory

GATE_OBSERVABLE_CLAIM = (
    "Gate-level observable-first readout for co-designed QSVT-safe targets on a selected "
    "IEEE14-derived subproblem. The postselected dense gate-level output direction is read "
    "through selected power-system observables (top-k update identification, bus "
    "angle/voltage updates, branch angle-difference and measurement-row proxies, "
    "selected-area update energy) and compared with the Ridge reference, the polynomial "
    "action, and a shot-estimated readout. Full-vector reconstruction is not emphasized. "
    "Ridge/Tikhonov remains the reference; no QSVT superiority over Ridge/Tikhonov, quantum "
    "speedup, quantum advantage, hardware execution, or solved readout bottleneck is claimed."
)

CLAIM_ALLOWED = "observable-first QSVT readout on a selected IEEE14-derived subproblem"
CLAIM_DISALLOWED = "no quantum speedup, QSVT-over-Ridge superiority, or solved readout bottleneck"

DEFAULT_OBSERVABLES = (
    "top_k_update_identification",
    "bus_angle_update",
    "bus_voltage_update",
    "branch_angle_difference_proxy",
    "measurement_row_correction_proxy",
    "selected_area_update_energy",
)
DEFAULT_SHOTS = (1000, 10000)

SUMMARY_COLUMNS = [
    "observable_name",
    "physical_meaning",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "ridge_value",
    "polynomial_value",
    "gate_value",
    "shot_estimated_value_if_available",
    "absolute_error_gate_vs_ridge",
    "relative_error_gate_vs_ridge",
    "top_k_match_if_applicable",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_full_vector_readout",
    "readout_protocol",
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
    "absolute_error_gate_vs_ridge",
    "relative_error_gate_vs_ridge",
    "top_k_match_if_applicable",
    "requires_norm_recovery",
    "requires_full_vector_readout",
    "readout_cost_proxy",
    "practical_status",
]


def run_qsvt_gate_observable_readout(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "input": "outputs/qsvt_degree_window_gate_validation/degree_window_gate_results.csv",
        "observables": list(DEFAULT_OBSERVABLES),
        "shots": list(DEFAULT_SHOTS),
        "topk": 2,
        "max_configs": 3,
        "grid_size": 4096,
        "seed": 123,
        "phase_timeout_seconds": 40,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_gate_observable_readout",
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
    configs = _select_gate_validated_configs(
        Path(resolved["input"]), max_configs=int(resolved["max_configs"])
    )
    rows = evaluate_gate_observable_readout(
        subproblem=subproblem,
        configs=configs,
        observables=[str(value) for value in resolved["observables"]],
        shots=[int(value) for value in resolved["shots"]],
        topk=int(resolved["topk"]),
        grid_size=int(resolved["grid_size"]),
        seed=int(resolved["seed"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
        subproblem_id=str(resolved["subproblem_id"]),
    )
    artifacts = write_gate_observable_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "configs": configs, "artifacts": artifacts}


def evaluate_gate_observable_readout(
    *,
    subproblem: Any,
    configs: list[dict[str, Any]],
    observables: list[str],
    shots: list[int],
    topk: int = 2,
    grid_size: int = 4096,
    seed: int = 123,
    phase_timeout_seconds: int = 40,
    subproblem_id: str = "subproblem",
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    requested = [name for name in observables if name != "full_state_vector_reconstruction"]
    max_shots = max(shots) if shots else 1000
    rows: list[dict[str, Any]] = []
    for config in configs:
        alpha = float(config["alpha"])
        degree = int(config["degree"])
        target_family = str(config["target_family"])
        ridge_update = ridge_tikhonov_update(H, r, alpha=alpha)
        solution = build_codesigned_solution(
            subproblem,
            alpha=alpha,
            degree=degree,
            target_family=target_family,
            grid_size=int(grid_size),
        )
        if solution is None:
            continue
        gate = _build_codesigned_gate_circuit(
            H=H,
            r=r,
            alpha=alpha,
            solution=solution,
            phase_timeout_seconds=int(phase_timeout_seconds),
        )
        if gate.get("status") != "synthesized":
            continue
        gate_update = np.asarray(gate["gate_update"], dtype=np.float64)
        p_exact = float(gate["success_probability_exact"])
        shot_update = _shot_estimated_update(gate_update, p_exact, max_shots, int(seed))

        protocols = build_observable_protocols(
            H_tilde=H, ridge_update=ridge_update, metadata=subproblem.metadata, topk=int(topk)
        )
        protocol_by_request = _map_protocols(protocols)
        for name in requested:
            protocol = protocol_by_request.get(name)
            if protocol is None:
                continue
            rows.append(
                _observable_row(
                    observable_name=name,
                    protocol=protocol,
                    ridge_update=ridge_update,
                    polynomial_update=np.asarray(solution.known_c_update, dtype=np.float64),
                    gate_update=gate_update,
                    shot_update=shot_update,
                    alpha=alpha,
                    degree=degree,
                    target_family=target_family,
                    subproblem_id=subproblem_id,
                    topk=int(topk),
                    readout_cost=float(max_shots),
                )
            )
    return rows


def _observable_row(
    *,
    observable_name: str,
    protocol: Any,
    ridge_update: np.ndarray,
    polynomial_update: np.ndarray,
    gate_update: np.ndarray,
    shot_update: np.ndarray,
    alpha: float,
    degree: int,
    target_family: str,
    subproblem_id: str,
    topk: int,
    readout_cost: float,
) -> dict[str, Any]:
    ridge_eval = evaluate_observable_protocol(
        protocol, ridge_update=ridge_update, qsvt_update=ridge_update
    )
    ridge_value = float(ridge_eval.ridge_value)
    polynomial_value = float(
        evaluate_observable_protocol(
            protocol, ridge_update=ridge_update, qsvt_update=polynomial_update
        ).qsvt_value_scaled
    )
    gate_value = float(
        evaluate_observable_protocol(
            protocol, ridge_update=ridge_update, qsvt_update=gate_update
        ).qsvt_value_scaled
    )
    shot_value = float(
        evaluate_observable_protocol(
            protocol, ridge_update=ridge_update, qsvt_update=shot_update
        ).qsvt_value_scaled
    )
    absolute = abs(gate_value - ridge_value)
    relative = absolute / abs(ridge_value) if abs(ridge_value) > 1.0e-15 else float("nan")
    top_k_match = (
        _topk_jaccard(ridge_update, gate_update, int(topk))
        if protocol.kind == "topk_identification"
        else float("nan")
    )
    return {
        "observable_name": observable_name,
        "physical_meaning": protocol.physical_meaning,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_family": target_family,
        "ridge_value": ridge_value,
        "polynomial_value": polynomial_value,
        "gate_value": gate_value,
        "shot_estimated_value_if_available": shot_value,
        "absolute_error_gate_vs_ridge": float(absolute),
        "relative_error_gate_vs_ridge": float(relative),
        "top_k_match_if_applicable": float(top_k_match),
        "requires_norm_recovery": bool(protocol.requires_norm_recovery),
        "requires_signed_overlap": bool(protocol.requires_signed_overlap),
        "requires_full_vector_readout": bool(protocol.requires_full_vector_readout),
        "readout_protocol": protocol.protocol_type,
        "readout_cost_proxy": float(readout_cost),
        "practical_status": _practical_status(protocol),
        "claim_allowed": CLAIM_ALLOWED,
        "claim_disallowed": CLAIM_DISALLOWED,
    }


def _map_protocols(protocols: list[Any]) -> dict[str, Any]:
    """Map the built observable protocols onto the requested observable vocabulary."""

    mapping: dict[str, Any] = {}
    for protocol in protocols:
        name = str(protocol.observable_name)
        if protocol.kind == "topk_identification":
            mapping.setdefault("top_k_update_identification", protocol)
        elif protocol.kind == "subset_energy":
            mapping.setdefault("selected_area_update_energy", protocol)
        elif protocol.requires_full_vector_readout:
            continue
        elif "branch_angle_difference" in name:
            mapping.setdefault("branch_angle_difference_proxy", protocol)
        elif "measurement_row" in name:
            mapping.setdefault("measurement_row_correction_proxy", protocol)
        elif "voltage_update" in name:
            mapping.setdefault("bus_voltage_update", protocol)
        elif "angle_update" in name:
            mapping.setdefault("bus_angle_update", protocol)
    return mapping


def _practical_status(protocol: Any) -> str:
    if protocol.requires_full_vector_readout:
        return "full_vector_like_not_recommended"
    if protocol.kind == "topk_identification":
        return "practical_without_norm_recovery"
    if protocol.requires_norm_recovery or protocol.requires_signed_overlap:
        return "practical_with_norm_recovery"
    return "practical_without_norm_recovery"


def _shot_estimated_update(
    gate_update: np.ndarray, p_exact: float, shots: int, seed: int
) -> np.ndarray:
    if p_exact <= 1.0e-15:
        return np.asarray(gate_update, dtype=np.float64)
    estimate = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed))
    scale = math.sqrt(max(float(estimate.estimate), 0.0) / p_exact)
    return float(scale) * np.asarray(gate_update, dtype=np.float64)


def _topk_jaccard(ridge_update: np.ndarray, gate_update: np.ndarray, topk: int) -> float:
    k = max(1, min(int(topk), ridge_update.size))
    ridge_top = set(_topk_indices(ridge_update, k))
    gate_top = set(_topk_indices(gate_update, k))
    union = ridge_top | gate_top
    if not union:
        return 1.0
    return float(len(ridge_top & gate_top) / len(union))


def _topk_indices(vector: np.ndarray, k: int) -> list[int]:
    vector = np.asarray(vector, dtype=np.float64)
    indices = np.arange(vector.size)
    order = np.lexsort((indices, -np.abs(vector)))
    return sorted(int(index) for index in order[:k])


def _select_gate_validated_configs(input_path: Path, *, max_configs: int) -> list[dict[str, Any]]:
    if not input_path.is_file():
        return _default_configs()
    try:
        frame = pd.read_csv(input_path)
    except Exception:
        return _default_configs()
    if frame.empty:
        return _default_configs()
    numeric = frame.copy()
    for column in ("residual_ratio_vs_no_update", "degree", "alpha"):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    feasible = numeric
    if "residual_feasible_after_gate" in numeric.columns:
        mask = numeric["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        if mask.any():
            feasible = numeric[mask]
    feasible = feasible.sort_values(
        by=[c for c in ("residual_ratio_vs_no_update", "degree") if c in feasible.columns],
        kind="stable",
    )
    selected = feasible.head(int(max_configs))
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
    return [{"alpha": 1.0e-2, "degree": 29, "target_family": "weighted_support_ls"}]


def write_gate_observable_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    accuracy_frame = summary_frame[ACCURACY_COST_COLUMNS].copy()

    summary_path = output_dir / "gate_observable_values.csv"
    accuracy_path = output_dir / "gate_observable_accuracy_cost.csv"
    interpretation_path = output_dir / "gate_observable_readout_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    accuracy_frame.to_csv(accuracy_path, index=False)
    interpretation_path.write_text(gate_observable_interpretation(summary_frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "gate_observable_values": str(summary_path),
            "gate_observable_accuracy_cost": str(accuracy_path),
            "gate_observable_readout_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=GATE_OBSERVABLE_CLAIM,
    )
    return {
        "manifest": manifest,
        "gate_observable_values": summary_path,
        "gate_observable_accuracy_cost": accuracy_path,
        "gate_observable_readout_interpretation": interpretation_path,
    }


def gate_observable_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# Gate-Level Observable Readout",
                "",
                GATE_OBSERVABLE_CLAIM,
                "",
                "- No gate-validated configuration produced an observable readout.",
                "",
            ]
        )
    numeric = frame.copy()
    numeric["relative_error_gate_vs_ridge"] = pd.to_numeric(
        numeric["relative_error_gate_vs_ridge"], errors="coerce"
    )
    numeric["top_k_match_if_applicable"] = pd.to_numeric(
        numeric["top_k_match_if_applicable"], errors="coerce"
    )

    confirmed = sorted(set(numeric["observable_name"].astype(str)))
    topk_rows = numeric[numeric["observable_name"] == "top_k_update_identification"]
    topk_match = (
        float(topk_rows["top_k_match_if_applicable"].dropna().min())
        if not topk_rows.empty
        else float("nan")
    )
    signed = numeric[
        (numeric["requires_signed_overlap"] == True)  # noqa: E712
        & (numeric["observable_name"] != "top_k_update_identification")
    ]
    best_signed_error = (
        float(signed["relative_error_gate_vs_ridge"].dropna().max())
        if not signed.empty
        else float("nan")
    )
    norm_required = sorted(
        set(numeric[numeric["requires_norm_recovery"] == True]["observable_name"].astype(str))  # noqa: E712
    )

    return "\n".join(
        [
            "# Gate-Level Observable Readout",
            "",
            GATE_OBSERVABLE_CLAIM,
            "",
            "## Counts",
            f"- Observable rows: {len(frame)}",
            f"- Distinct observables gate-confirmed: {len(confirmed)}",
            "",
            "## Required Answers",
            f"1. Which observables are gate-confirmed? {', '.join(confirmed)}.",
            f"2. Does top-k identification remain exact after gate validation? "
            f"min top-k match = {topk_match:.3g} (1.0 means exact).",
            f"3. Do signed power-system observables remain accurate after gate validation? worst "
            f"signed relative error vs Ridge = {best_signed_error:.3g}.",
            "4. Which observables to emphasize: top-k update identification (no norm/sign), bus "
            "angle/voltage updates and the branch angle-difference proxy (signed functionals via "
            "the Hadamard-test proxy), and selected-area update energy (norm-scaled).",
            f"5. Which observables still require norm/full-vector recovery: "
            f"{', '.join(norm_required)} (norm recovery only); no emphasized observable requires "
            "full-vector readout.",
            "",
        ]
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
