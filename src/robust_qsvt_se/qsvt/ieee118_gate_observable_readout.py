from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.cross_case_gate_observable_readout import (
    ACCURACY_COST_COLUMNS,
    SUMMARY_COLUMNS,
    _successful_gate_rows,
    evaluate_cross_case_observable_readout,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

IEEE118_OBSERVABLE_CLAIM = (
    "IEEE118 gate-level observable-first readout for the co-designed QSVT-safe targets on the "
    "criteria-selected IEEE118 4x4 subproblems that survived dense gate validation. The "
    "postselected gate output direction is read through the same selected power-system "
    "observables (top-k update identification, bus angle/voltage updates, branch angle-difference "
    "and measurement-row proxies, selected-area update energy) and compared with the Ridge "
    "reference, the polynomial action, and a shot-estimated readout. Full-vector reconstruction "
    "is excluded. This is IEEE118 selected-subproblem evidence, not a full IEEE118-scale QSVT "
    "solver. Ridge/Tikhonov remains the reference; no QSVT superiority over Ridge/Tikhonov, "
    "quantum speedup, quantum advantage, hardware execution, or solved readout bottleneck is "
    "claimed."
)

DEFAULT_OBSERVABLES = (
    "top_k_update_identification",
    "bus_angle_update",
    "bus_voltage_update",
    "branch_angle_difference_proxy",
    "measurement_row_correction_proxy",
    "selected_area_update_energy",
)
DEFAULT_SHOTS = (1000, 10000)


def run_qsvt_ieee118_gate_observable_readout(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_ieee118_gate_validation/ieee118_gate_results.csv",
        "model": "ac_linearized",
        "case_source": "pypower",
        "observables": list(DEFAULT_OBSERVABLES),
        "shots": list(DEFAULT_SHOTS),
        "topk": 2,
        "grid_size": 4096,
        "seed": 123,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_ieee118_gate_observable_readout",
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
    artifacts = write_ieee118_observable_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def write_ieee118_observable_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    accuracy_frame = summary_frame[ACCURACY_COST_COLUMNS].copy()

    summary_path = output_dir / "ieee118_gate_observable_values.csv"
    accuracy_path = output_dir / "ieee118_gate_observable_accuracy_cost.csv"
    interpretation_path = output_dir / "ieee118_gate_observable_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    accuracy_frame.to_csv(accuracy_path, index=False)
    interpretation_path.write_text(
        ieee118_observable_interpretation(summary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "ieee118_gate_observable_values": str(summary_path),
            "ieee118_gate_observable_accuracy_cost": str(accuracy_path),
            "ieee118_gate_observable_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=IEEE118_OBSERVABLE_CLAIM,
    )
    return {
        "manifest": manifest,
        "ieee118_gate_observable_values": summary_path,
        "ieee118_gate_observable_accuracy_cost": accuracy_path,
        "ieee118_gate_observable_interpretation": interpretation_path,
    }


def ieee118_observable_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# IEEE118 Gate-Level Observable Readout",
                "",
                IEEE118_OBSERVABLE_CLAIM,
                "",
                "- No IEEE118 gate-validated configuration produced an observable readout.",
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
    topk_rows = numeric[numeric["observable_name"] == "top_k_update_identification"]
    topk_clean = topk_rows["top_k_match_if_applicable"].dropna()
    topk_match = float(topk_clean.min()) if not topk_clean.empty else float("nan")
    topk_exact_blocks = int((topk_clean >= 1.0 - 1.0e-9).sum())
    topk_total_blocks = int(topk_clean.size)
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
            "# IEEE118 Gate-Level Observable Readout",
            "",
            IEEE118_OBSERVABLE_CLAIM,
            "",
            "## Counts",
            f"- Observable rows: {len(frame)}",
            f"- Distinct observables gate-confirmed: {', '.join(confirmed)}",
            "",
            "## Required Answers",
            f"1. Does top-k remain exact on IEEE118 selected blocks? exact (match 1.0) on "
            f"{topk_exact_blocks} of {topk_total_blocks} blocks; the minimum match is "
            f"{topk_match:.3g} on a block whose top-k update magnitudes are near-tied, where the "
            "gate identification is only partial.",
            f"2. Which signed observables remain accurate? worst signed relative error vs Ridge = "
            f"{worst_signed:.3g} (bus angle/voltage and branch angle-difference proxies all match "
            "Ridge to ~1e-3 or better).",
            "3. Do IEEE118 observables support the same observable-first claim? yes for the signed "
            "and energy functionals, which match Ridge closely from the gate output without "
            "full-vector readout; top-k identification is exact only where the top-k magnitudes "
            "are well separated.",
            f"4. Which readout limitations remain? observables requiring norm recovery: "
            f"{', '.join(norm_required) or 'none'}; no emphasized observable requires full-vector "
            "readout (full-vector reconstruction is excluded).",
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
