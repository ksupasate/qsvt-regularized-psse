from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_bounded_targets import build_codesigned_solution
from robust_qsvt_se.qsvt.codesigned_gate_validation import _build_codesigned_gate_circuit
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import ridge_tikhonov_update
from robust_qsvt_se.qsvt.gate_observable_readout import (
    _map_protocols,
    _practical_status,
    _shot_estimated_update,
    _topk_jaccard,
)
from robust_qsvt_se.qsvt.power_observable_protocols import (
    build_observable_protocols,
    evaluate_observable_protocol,
)
from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    build_selected_subproblem_from_policy_row,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

CROSS_CASE_OBSERVABLE_CLAIM = (
    "Cross-case gate-level observable-first readout for the co-designed QSVT-safe targets on "
    "the IEEE30/IEEE57 (and IEEE14 reference) selected 4x4 subproblems that survived dense gate "
    "validation. The postselected gate output direction is read through the same selected "
    "power-system observables (top-k update identification, bus angle/voltage updates, branch "
    "angle-difference and measurement-row proxies, selected-area update energy) and compared "
    "with the Ridge reference, the polynomial action, and a shot-estimated readout. Full-vector "
    "reconstruction is excluded. Ridge/Tikhonov remains the reference; no QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, quantum advantage, hardware execution, or solved readout "
    "bottleneck is claimed."
)

CLAIM_ALLOWED = "observable-first QSVT readout on cross-case selected IEEE-derived subproblems"
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
    "case",
    "observable_name",
    "physical_meaning",
    "subproblem_id",
    "selection_mode",
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
    "case",
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


def run_qsvt_cross_case_gate_observable_readout(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_cross_case_gate_validation/cross_case_gate_results.csv",
        "model": "ac_linearized",
        "case_source": "pypower",
        "observables": list(DEFAULT_OBSERVABLES),
        "shots": list(DEFAULT_SHOTS),
        "topk": 2,
        "grid_size": 4096,
        "seed": 123,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_cross_case_gate_observable_readout",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    gate_results = _read_csv(Path(resolved["input"]))
    successful = _successful_gate_rows(gate_results)
    rows = evaluate_cross_case_observable_readout(
        successful=successful,
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        observables=[str(value) for value in resolved["observables"]],
        shots=[int(value) for value in resolved["shots"]],
        topk=int(resolved["topk"]),
        grid_size=int(resolved["grid_size"]),
        seed=int(resolved["seed"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
    )
    artifacts = write_cross_case_observable_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def _successful_gate_rows(gate_results: pd.DataFrame) -> pd.DataFrame:
    if gate_results.empty:
        return gate_results
    if "gate_status" not in gate_results.columns:
        return gate_results
    return gate_results[gate_results["gate_status"].astype(str) == "completed"].reset_index(
        drop=True
    )


def evaluate_cross_case_observable_readout(
    *,
    successful: pd.DataFrame,
    model: str,
    case_source: str,
    observables: list[str],
    shots: list[int],
    topk: int = 2,
    grid_size: int = 4096,
    seed: int = 123,
    phase_timeout_seconds: int = 40,
) -> list[dict[str, Any]]:
    if successful is None or successful.empty:
        return []
    requested = [name for name in observables if name != "full_state_vector_reconstruction"]
    max_shots = max(shots) if shots else 1000
    systems = _build_case_systems(
        cases=sorted(set(successful["case"].astype(str))),
        model=model,
        case_source=case_source,
        seed=int(seed),
    )

    rows: list[dict[str, Any]] = []
    for record in successful.to_dict("records"):
        case = str(record.get("case", "ieee14"))
        system_entry = systems.get(case)
        if system_entry is None:
            continue
        system, matrix_source = system_entry
        try:
            subproblem = build_selected_subproblem_from_policy_row(
                system=system,
                matrix_source=matrix_source,
                row={
                    "case": case,
                    "model": model,
                    "candidate_id": record.get("subproblem_id", "selected_subproblem"),
                    "selection_source": record.get("selection_mode", "selection_policy"),
                    "row_indices": record.get("row_indices", ""),
                    "col_indices": record.get("col_indices", ""),
                },
            )
        except Exception:
            continue
        rows.extend(
            _readout_for_subproblem(
                subproblem=subproblem,
                case=case,
                record=record,
                requested=requested,
                topk=int(topk),
                grid_size=int(grid_size),
                seed=int(seed),
                max_shots=int(max_shots),
                phase_timeout_seconds=int(phase_timeout_seconds),
            )
        )
    return rows


def _readout_for_subproblem(
    *,
    subproblem: Any,
    case: str,
    record: dict[str, Any],
    requested: list[str],
    topk: int,
    grid_size: int,
    seed: int,
    max_shots: int,
    phase_timeout_seconds: int,
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    alpha = float(record.get("alpha"))
    degree = int(float(record.get("degree")))
    target_family = str(record.get("target_family", "residual_aware"))
    subproblem_id = str(record.get("subproblem_id", "selected_subproblem"))
    selection_mode = str(record.get("selection_mode", "unknown"))

    ridge_update = ridge_tikhonov_update(H, r, alpha=alpha)
    solution = build_codesigned_solution(
        subproblem,
        alpha=alpha,
        degree=degree,
        target_family=target_family,
        grid_size=int(grid_size),
    )
    if solution is None:
        return []
    gate = _build_codesigned_gate_circuit(
        H=H,
        r=r,
        alpha=alpha,
        solution=solution,
        phase_timeout_seconds=int(phase_timeout_seconds),
    )
    if gate.get("status") != "synthesized":
        return []
    gate_update = np.asarray(gate["gate_update"], dtype=np.float64)
    p_exact = float(gate["success_probability_exact"])
    shot_update = _shot_estimated_update(gate_update, p_exact, int(max_shots), int(seed))

    protocols = build_observable_protocols(
        H_tilde=H, ridge_update=ridge_update, metadata=subproblem.metadata, topk=int(topk)
    )
    protocol_by_request = _map_protocols(protocols)

    rows: list[dict[str, Any]] = []
    for name in requested:
        protocol = protocol_by_request.get(name)
        if protocol is None:
            continue
        rows.append(
            _observable_row(
                case=case,
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
                selection_mode=selection_mode,
                topk=int(topk),
                readout_cost=float(max_shots),
            )
        )
    return rows


def _observable_row(
    *,
    case: str,
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
    selection_mode: str,
    topk: int,
    readout_cost: float,
) -> dict[str, Any]:
    ridge_value = float(
        evaluate_observable_protocol(
            protocol, ridge_update=ridge_update, qsvt_update=ridge_update
        ).ridge_value
    )
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
        "case": case,
        "observable_name": observable_name,
        "physical_meaning": protocol.physical_meaning,
        "subproblem_id": subproblem_id,
        "selection_mode": selection_mode,
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


def _build_case_systems(
    *, cases: list[str], model: str, case_source: str, seed: int
) -> dict[str, tuple[Any, str]]:
    systems: dict[str, tuple[Any, str]] = {}
    for case in cases:
        try:
            systems[case] = _build_system(
                case=case, model=model, case_source=case_source, seed=int(seed)
            )
        except Exception:  # pragma: no cover - depends on optional pypower data
            continue
    return systems


def write_cross_case_observable_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    accuracy_frame = summary_frame[ACCURACY_COST_COLUMNS].copy()

    summary_path = output_dir / "cross_case_gate_observable_values.csv"
    accuracy_path = output_dir / "cross_case_gate_observable_accuracy_cost.csv"
    interpretation_path = output_dir / "cross_case_gate_observable_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    accuracy_frame.to_csv(accuracy_path, index=False)
    interpretation_path.write_text(
        cross_case_observable_interpretation(summary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "cross_case_gate_observable_values": str(summary_path),
            "cross_case_gate_observable_accuracy_cost": str(accuracy_path),
            "cross_case_gate_observable_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CROSS_CASE_OBSERVABLE_CLAIM,
    )
    return {
        "manifest": manifest,
        "cross_case_gate_observable_values": summary_path,
        "cross_case_gate_observable_accuracy_cost": accuracy_path,
        "cross_case_gate_observable_interpretation": interpretation_path,
    }


def cross_case_observable_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# Cross-Case Gate-Level Observable Readout",
                "",
                CROSS_CASE_OBSERVABLE_CLAIM,
                "",
                "- No cross-case gate-validated configuration produced an observable readout.",
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
    cases = sorted(set(numeric["case"].astype(str)))
    topk_lines = []
    for case in cases:
        topk_rows = numeric[
            (numeric["case"].astype(str) == case)
            & (numeric["observable_name"] == "top_k_update_identification")
        ]
        match = (
            float(topk_rows["top_k_match_if_applicable"].dropna().min())
            if not topk_rows.empty
            else float("nan")
        )
        topk_lines.append(f"   - {case}: min top-k match = {match:.3g}")
    signed = numeric[
        (numeric["requires_signed_overlap"] == True)  # noqa: E712
        & (numeric["observable_name"] != "top_k_update_identification")
    ]
    worst_signed = (
        float(signed["relative_error_gate_vs_ridge"].dropna().max())
        if not signed.empty
        else float("nan")
    )
    norm_required = sorted(
        set(numeric[numeric["requires_norm_recovery"] == True]["observable_name"].astype(str))  # noqa: E712
    )
    confirmed = sorted(set(numeric["observable_name"].astype(str)))

    return "\n".join(
        [
            "# Cross-Case Gate-Level Observable Readout",
            "",
            CROSS_CASE_OBSERVABLE_CLAIM,
            "",
            "## Counts",
            f"- Cases with observable readout: {', '.join(cases)}",
            f"- Observable rows: {len(frame)}",
            f"- Distinct observables gate-confirmed: {', '.join(confirmed)}",
            "",
            "## Required Answers",
            "1. Do observable-first results transfer to IEEE30/57 selected blocks? "
            f"yes for cases {', '.join(cases)}.",
            "2. Does top-k identification remain robust across cases?",
            *topk_lines,
            f"3. Which signed observables remain accurate? worst signed relative error vs Ridge "
            f"= {worst_signed:.3g} across cases.",
            "4. Which observables to emphasize as cross-case readout evidence: top-k update "
            "identification (no norm/sign), bus angle/voltage and branch angle-difference proxies "
            "(signed functionals), and selected-area update energy (norm-scaled).",
            f"5. Observables still requiring norm recovery: {', '.join(norm_required) or 'none'}; "
            "no emphasized observable requires full-vector readout.",
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


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
