from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.robustness_gate_validation import validate_robustness_gate_config
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

CROSS_CASE_GATE_CLAIM = (
    "Dense gate-level validation of the co-designed QSVT-safe targets on cross-case "
    "criteria-selected IEEE30/IEEE57 (and IEEE14 reference) 4x4 subproblems. Each selected "
    "candidate subproblem is reconstructed from its selection indices for its own case, the "
    "co-designed phases are synthesized from the bounded coefficients, the structured QSVT "
    "operator circuit is simulated exactly, and the success amplitude is estimated with the "
    "implemented small-circuit routine to recover the update scale. This is "
    "selected-subproblem dense-simulator evidence, not full IEEE-scale hardware execution, and "
    "claims no QSVT superiority over Ridge/Tikhonov, quantum speedup, or quantum advantage."
)

CONTROL_MODE = "worst_conditioned_control"
CROSS_CASES = ("ieee30", "ieee57")

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
    "cross_case_validation_status",
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


def run_qsvt_cross_case_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_cross_case_codesigned_robustness/"
        "cross_case_gate_validation_candidates.csv",
        "model": "ac_linearized",
        "case_source": "pypower",
        "max_configs": 6,
        "shots": 1000,
        "seed": 123,
        "grid_size": 4096,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_cross_case_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    candidate_frame = _read_csv(Path(resolved["input"]))
    selected = select_cross_case_gate_candidates(
        candidate_frame, max_configs=int(resolved["max_configs"])
    )

    rows: list[dict[str, Any]] = []
    if not selected.empty:
        systems = _build_case_systems(
            cases=sorted(set(selected["case"].astype(str))),
            model=str(resolved["model"]),
            case_source=str(resolved["case_source"]),
            seed=int(resolved["seed"]),
        )
        for record in selected.to_dict("records"):
            rows.append(
                _validate_cross_case_config(
                    row=record,
                    systems=systems,
                    shots=int(resolved["shots"]),
                    seed=int(resolved["seed"]),
                    grid_size=int(resolved["grid_size"]),
                    phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
                )
            )

    artifacts = write_cross_case_gate_outputs(output_dir, resolved, selected, rows)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_cross_case_gate_candidates(
    candidate_frame: pd.DataFrame, *, max_configs: int
) -> pd.DataFrame:
    """Pick at most ``max_configs`` candidates with cross-case coverage.

    Guarantees at least one IEEE30 and one IEEE57 candidate when available, prefers
    non-control criteria-selected blocks ranked by residual ratio, and fills remaining slots
    with the best remaining cross-case candidates before falling back to IEEE14.
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

    def best_for_case(case: str) -> int | None:
        sub = frame[(frame["case"].astype(str) == case) & (~frame.index.isin(chosen))].sort_values(
            by=["_rank_ratio"]
        )
        return None if sub.empty else int(sub.index[0])

    for case in CROSS_CASES:
        index = best_for_case(case)
        if index is not None and len(chosen) < int(max_configs):
            chosen.append(index)
            reasons[index] = f"{case}_representative"

    # Fill remaining slots: cross-case candidates first, then IEEE14 reference, by residual ratio.
    frame["_case_priority"] = (
        frame["case"].astype(str).map(lambda value: 0 if value in CROSS_CASES else 1)
    )
    fill_order = frame.sort_values(by=["_case_priority", "_rank_ratio"])
    for index in fill_order.index:
        if len(chosen) >= int(max_configs):
            break
        if index not in chosen:
            chosen.append(int(index))
            reasons.setdefault(int(index), "additional_cross_case_candidate")

    selected_rows: list[dict[str, Any]] = []
    for index in chosen:
        record = frame.loc[index].to_dict()
        record["selection_reason"] = reasons.get(index, "additional_cross_case_candidate")
        selected_rows.append(record)
    out = pd.DataFrame(selected_rows)
    for column in SELECTED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[SELECTED_COLUMNS].reset_index(drop=True)


def _validate_cross_case_config(
    *,
    row: dict[str, Any],
    systems: dict[str, tuple[Any, str]],
    shots: int,
    seed: int,
    grid_size: int,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    case = str(row.get("case", "ieee14"))
    base = {column: np.nan for column in GATE_COLUMNS}
    base.update(
        {
            "case": case,
            "model": row.get("model", "ac_linearized"),
            "subproblem_id": row.get("subproblem_id", "selected_subproblem"),
            "selection_mode": row.get("selection_mode", "unknown"),
            "row_indices": row.get("row_indices", ""),
            "col_indices": row.get("col_indices", ""),
            "gate_status": "not_run",
            "phase_synthesis_status": "not_run",
            "residual_feasible_after_gate": False,
            "cross_case_validation_status": "not_run",
            "dominant_limitation": "not_run",
        }
    )
    system_entry = systems.get(case)
    if system_entry is None:
        base.update(
            {
                "gate_status": "failed",
                "phase_synthesis_status": "case_system_unavailable",
                "cross_case_validation_status": "case_unavailable",
                "dominant_limitation": "case_system_unavailable",
            }
        )
        return base

    system, matrix_source = system_entry
    result = validate_robustness_gate_config(
        row=row,
        system=system,
        matrix_source=matrix_source,
        shots=int(shots),
        seed=int(seed),
        grid_size=int(grid_size),
        phase_timeout_seconds=int(phase_timeout_seconds),
    )
    for column in GATE_COLUMNS:
        if column in result and column not in {"cross_case_validation_status"}:
            base[column] = result[column]
    base["row_indices"] = row.get("row_indices", "")
    base["col_indices"] = row.get("col_indices", "")
    base["cross_case_validation_status"] = _cross_case_status(result)
    return base


def _cross_case_status(result: dict[str, Any]) -> str:
    status = str(result.get("gate_status", "not_run"))
    if status == "completed":
        return (
            "survived_gate"
            if bool(result.get("residual_feasible_after_gate"))
            else "completed_not_feasible"
        )
    if status == "not_run":
        return "phase_synthesis_limited"
    return "gate_failed"


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


def write_cross_case_gate_outputs(
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

    selected_path = output_dir / "cross_case_gate_selected_configs.csv"
    results_path = output_dir / "cross_case_gate_results.csv"
    interpretation_path = output_dir / "cross_case_gate_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(cross_case_gate_interpretation(results), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "cross_case_gate_selected_configs": str(selected_path),
            "cross_case_gate_results": str(results_path),
            "cross_case_gate_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CROSS_CASE_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "cross_case_gate_selected_configs": selected_path,
        "cross_case_gate_results": results_path,
        "cross_case_gate_interpretation": interpretation_path,
    }


def surviving_cases(results: pd.DataFrame) -> list[str]:
    if results.empty or "residual_feasible_after_gate" not in results.columns:
        return []
    mask = results["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
    return sorted(set(results[mask].get("case", pd.Series(dtype=str)).astype(str)))


def cross_case_gate_interpretation(results: pd.DataFrame) -> str:
    if results.empty:
        return "\n".join(
            [
                "# Cross-Case Gate-Level Validation",
                "",
                CROSS_CASE_GATE_CLAIM,
                "",
                "- No cross-case candidates were available for gate validation.",
                "",
            ]
        )
    completed = results[results["gate_status"] == "completed"]
    survivors = surviving_cases(results)
    ieee30 = "ieee30" in survivors
    ieee57 = "ieee57" in survivors
    cross_case_survivors = [case for case in survivors if case in CROSS_CASES]

    if ieee30 and ieee57:
        headline = (
            "The selected-subproblem solver prototype transfers across IEEE14/30/57 selected "
            "4x4 blocks."
        )
    elif cross_case_survivors:
        headline = "The cross-case evidence is partial and should be framed conservatively."
    else:
        headline = "The solver prototype remains IEEE14-selected-family evidence only."

    best_ratio = (
        float(pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )
    best_state_error = (
        float(pd.to_numeric(completed["state_error_gate_vs_polynomial"], errors="coerce").max())
        if not completed.empty
        else float("nan")
    )

    return "\n".join(
        [
            "# Cross-Case Gate-Level Validation",
            "",
            CROSS_CASE_GATE_CLAIM,
            "",
            "## Counts",
            f"- Configurations validated: {len(results)}",
            f"- Completed gate runs: {len(completed)}",
            f"- Cases with a residual-feasible gate result: {', '.join(survivors) or 'none'}",
            "",
            "## Required Answers",
            f"- IEEE30 survived gate validation: {'yes' if ieee30 else 'no'}.",
            f"- IEEE57 survived gate validation: {'yes' if ieee57 else 'no'}.",
            f"- Best residual_ratio_vs_no_update (completed): {best_ratio:.6g}.",
            f"- Worst state error gate vs polynomial (completed): {best_state_error:.6g}.",
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
