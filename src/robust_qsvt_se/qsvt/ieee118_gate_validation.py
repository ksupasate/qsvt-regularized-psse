from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.robustness_gate_validation import validate_robustness_gate_config
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

IEEE118_GATE_CLAIM = (
    "Dense gate-level validation of the co-designed QSVT-safe targets on criteria-selected "
    "IEEE118 4x4 subproblems that were residual-feasible at the matrix level. Each candidate is "
    "reconstructed from its selection indices for the IEEE118 system, the co-designed phases are "
    "synthesized from the bounded coefficients, the structured QSVT operator circuit is simulated "
    "exactly, and the success amplitude is estimated with the implemented small-circuit routine "
    "to recover the update scale. This is IEEE118 selected-subproblem dense-simulator evidence, "
    "not a full IEEE118-scale QSVT solver or hardware execution, and claims no QSVT superiority "
    "over Ridge/Tikhonov, quantum speedup, or quantum advantage."
)

CONTROL_MODE = "worst_conditioned_control"
PREFERRED_MODES = ("high_leverage", "metadata_mapped", "residual_supported", "best_conditioned")

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
    "ridge_residual",
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
    "row_indices",
    "col_indices",
]

SELECTED_COLUMNS = [
    "selection_reason",
    "case",
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


def run_qsvt_ieee118_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_ieee118_selected_robustness/ieee118_gate_validation_candidates.csv",
        "case": "ieee118",
        "model": "ac_linearized",
        "case_source": "pypower",
        "max_configs": 3,
        "shots": 1000,
        "seed": 123,
        "grid_size": 4096,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_ieee118_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    candidate_frame = _read_csv(Path(resolved["input"]))
    selected = select_ieee118_gate_candidates(
        candidate_frame, max_configs=int(resolved["max_configs"])
    )

    rows: list[dict[str, Any]] = []
    if not selected.empty:
        try:
            system, matrix_source = _build_system(
                case=str(resolved["case"]),
                model=str(resolved["model"]),
                case_source=str(resolved["case_source"]),
                seed=int(resolved["seed"]),
            )
        except Exception:  # pragma: no cover - depends on optional pypower data
            system, matrix_source = None, ""
        for record in selected.to_dict("records"):
            rows.append(
                _validate_ieee118_config(
                    row=record,
                    system=system,
                    matrix_source=matrix_source,
                    shots=int(resolved["shots"]),
                    seed=int(resolved["seed"]),
                    grid_size=int(resolved["grid_size"]),
                    phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
                )
            )

    artifacts = write_ieee118_gate_outputs(output_dir, resolved, selected, rows)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_ieee118_gate_candidates(
    candidate_frame: pd.DataFrame, *, max_configs: int
) -> pd.DataFrame:
    """Pick at most ``max_configs`` IEEE118 candidates by preference order.

    Reads only from the candidate output: prefers high_leverage, then metadata_mapped, then
    residual_supported, then best_conditioned; always excludes the worst-conditioned control from
    positive evidence; fills any remaining slots by best residual ratio.
    """

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
    frame = frame.reset_index(drop=True)
    frame["_rank_ratio"] = pd.to_numeric(
        frame.get("residual_ratio_vs_no_update"), errors="coerce"
    ).fillna(float("inf"))

    chosen: list[int] = []
    reasons: dict[int, str] = {}

    def best_for_mode(mode: str) -> int | None:
        sub = frame[
            (frame["selection_mode"].astype(str) == mode) & (~frame.index.isin(chosen))
        ].sort_values(by=["_rank_ratio"])
        return None if sub.empty else int(sub.index[0])

    for mode in PREFERRED_MODES:
        if len(chosen) >= int(max_configs):
            break
        index = best_for_mode(mode)
        if index is not None:
            chosen.append(index)
            reasons[index] = f"preferred:{mode}"

    for index in frame.sort_values(by=["_rank_ratio"]).index:
        if len(chosen) >= int(max_configs):
            break
        if index not in chosen:
            chosen.append(int(index))
            reasons.setdefault(int(index), "additional_candidate")

    selected_rows: list[dict[str, Any]] = []
    for index in chosen:
        record = frame.loc[index].to_dict()
        record.setdefault("case", "ieee118")
        record["selection_reason"] = reasons.get(index, "additional_candidate")
        selected_rows.append(record)
    out = pd.DataFrame(selected_rows)
    for column in SELECTED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[SELECTED_COLUMNS].reset_index(drop=True)


def _validate_ieee118_config(
    *,
    row: dict[str, Any],
    system: Any,
    matrix_source: str,
    shots: int,
    seed: int,
    grid_size: int,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    base = {column: np.nan for column in GATE_COLUMNS}
    base.update(
        {
            "case": row.get("case", "ieee118"),
            "model": row.get("model", "ac_linearized"),
            "subproblem_id": row.get("subproblem_id", "selected_subproblem"),
            "selection_mode": row.get("selection_mode", "unknown"),
            "row_indices": row.get("row_indices", ""),
            "col_indices": row.get("col_indices", ""),
            "gate_status": "not_run",
            "phase_synthesis_status": "not_run",
            "residual_feasible_after_gate": False,
            "dominant_limitation": "not_run",
        }
    )
    if system is None:
        base.update(
            {
                "gate_status": "failed",
                "phase_synthesis_status": "case_system_unavailable",
                "dominant_limitation": "case_system_unavailable",
            }
        )
        return base

    result = validate_robustness_gate_config(
        row={**row, "case": "ieee118"},
        system=system,
        matrix_source=matrix_source,
        shots=int(shots),
        seed=int(seed),
        grid_size=int(grid_size),
        phase_timeout_seconds=int(phase_timeout_seconds),
    )
    for column in GATE_COLUMNS:
        if column in result:
            base[column] = result[column]
    base["row_indices"] = row.get("row_indices", "")
    base["col_indices"] = row.get("col_indices", "")
    return base


def surviving_subproblems(results: pd.DataFrame) -> list[str]:
    if results.empty or "residual_feasible_after_gate" not in results.columns:
        return []
    mask = results["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
    return sorted(set(results[mask].get("subproblem_id", pd.Series(dtype=str)).astype(str)))


def write_ieee118_gate_outputs(
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

    selected_path = output_dir / "ieee118_gate_selected_configs.csv"
    results_path = output_dir / "ieee118_gate_results.csv"
    interpretation_path = output_dir / "ieee118_gate_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(ieee118_gate_interpretation(results), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "ieee118_gate_selected_configs": str(selected_path),
            "ieee118_gate_results": str(results_path),
            "ieee118_gate_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=IEEE118_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "ieee118_gate_selected_configs": selected_path,
        "ieee118_gate_results": results_path,
        "ieee118_gate_interpretation": interpretation_path,
    }


def ieee118_gate_interpretation(results: pd.DataFrame) -> str:
    if results.empty:
        return "\n".join(
            [
                "# IEEE118 Gate-Level Validation",
                "",
                IEEE118_GATE_CLAIM,
                "",
                "- IEEE118 remains boundary/negative evidence under the current "
                "selected-subproblem criteria.",
                "",
            ]
        )
    completed = results[results["gate_status"] == "completed"]
    survivors = surviving_subproblems(results)
    candidates_existed = len(results) > 0

    if survivors:
        headline = (
            "The cross-case selected-subproblem solver prototype extends to IEEE118 selected "
            "4x4 blocks."
        )
    elif candidates_existed:
        headline = "IEEE118 polynomial-level feasibility does not survive gate validation."
    else:
        headline = "IEEE118 remains boundary/negative evidence under the current selected-subproblem criteria."  # noqa: E501

    best_ratio = (
        float(pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )
    worst_state_error = (
        float(pd.to_numeric(completed["state_error_gate_vs_polynomial"], errors="coerce").max())
        if not completed.empty
        else float("nan")
    )
    feasible_after = (
        int((completed["residual_feasible_after_gate"].astype(str).str.lower() == "true").sum())
        if not completed.empty
        else 0
    )

    return "\n".join(
        [
            "# IEEE118 Gate-Level Validation",
            "",
            IEEE118_GATE_CLAIM,
            "",
            "## Counts",
            f"- Configurations validated: {len(results)}",
            f"- Completed gate runs: {len(completed)}",
            f"- Residual-feasible after gate: {feasible_after}",
            f"- Surviving IEEE118 subproblems: {', '.join(survivors) or 'none'}",
            "",
            "## Required Answers",
            f"- Best residual_ratio_vs_no_update (completed): {best_ratio:.6g}.",
            f"- Worst state error gate vs polynomial (completed): {worst_state_error:.6g}.",
            f"- Residual feasibility survives gate validation: {'yes' if survivors else 'no'}.",
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
