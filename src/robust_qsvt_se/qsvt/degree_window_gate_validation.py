from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_gate_validation import validate_single_codesigned_gate_config
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

DEGREE_WINDOW_GATE_CLAIM = (
    "Dense gate-level validation across the co-designed degree window on a selected "
    "IEEE14-derived subproblem. Representative configurations from the feasible degree "
    "window (best residual, best success amplitude, lowest and highest feasible degree, "
    "most stable family) are synthesized from their co-designed bounded coefficients, the "
    "structured QSVT operator circuit is simulated exactly, and the success amplitude is "
    "estimated with the implemented small-circuit routine to recover the update scale. This "
    "is selected-subproblem dense simulator evidence, not full IEEE-scale hardware "
    "execution, and claims no QSVT superiority over Ridge/Tikhonov, quantum speedup, or "
    "quantum advantage."
)

RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
DIRECTION_FEASIBLE_MAX = 0.1

GATE_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "qsvt_safe",
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
    "degree_window_class_after_gate",
    "dominant_gate_limitation",
]

SELECTED_COLUMNS = [
    "selection_reason",
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "degree_window_class",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "success_probability_proxy",
]


def run_qsvt_degree_window_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_degree_window_overshoot/feasible_degree_window.csv",
        "max_configs": 5,
        "shots": 1000,
        "seed": 123,
        "case_source": "pypower",
        "submatrix_size": 4,
        "grid_size": 4096,
        "phase_timeout_seconds": 40,
        "output_dir": "outputs/qsvt_degree_window_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    feasible_frame = _read_csv(Path(resolved["input"]))
    selected = select_degree_window_gate_configs(
        feasible_frame, max_configs=int(resolved["max_configs"])
    )

    rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        rows.append(
            _validate_degree_window_config(
                row=record,
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                case_source=str(resolved["case_source"]),
                default_submatrix_size=int(resolved["submatrix_size"]),
                grid_size=int(resolved["grid_size"]),
                phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
            )
        )

    artifacts = write_degree_window_gate_outputs(output_dir, resolved, selected, rows)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_degree_window_gate_configs(
    feasible_frame: pd.DataFrame, *, max_configs: int
) -> pd.DataFrame:
    """Pick up to ``max_configs`` representative configurations across the degree window.

    The five selection intents are: best residual, best success amplitude, lowest feasible
    degree, highest feasible degree (before overshoot), and a most-stable-family
    representative. Duplicates collapse to a single row whose ``selection_reason`` lists every
    intent it satisfies.
    """

    if feasible_frame is None or feasible_frame.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS)
    frame = feasible_frame.copy()
    for column in (
        "residual_ratio_vs_no_update",
        "direction_error_vs_ridge",
        "degree",
        "success_probability_proxy",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["degree"])
    if frame.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS)

    intents: list[tuple[str, int]] = []

    def add(reason: str, index: Any) -> None:
        if index is not None and index in frame.index:
            intents.append((reason, int(frame.index.get_loc(index))))

    if frame["residual_ratio_vs_no_update"].notna().any():
        add("best_residual", frame["residual_ratio_vs_no_update"].idxmin())
    if frame["success_probability_proxy"].notna().any():
        add("best_success_probability", frame["success_probability_proxy"].idxmax())
    add("lowest_feasible_degree", frame["degree"].idxmin())
    add("highest_feasible_degree", frame["degree"].idxmax())
    stable_index = _most_stable_family_index(frame)
    add("most_stable_family", stable_index)

    ordered: dict[int, list[str]] = {}
    positions = list(frame.index)
    for reason, position in intents:
        absolute_index = positions[position]
        ordered.setdefault(absolute_index, []).append(reason)

    selected_rows: list[dict[str, Any]] = []
    for absolute_index, reasons in ordered.items():
        if len(selected_rows) >= int(max_configs):
            break
        record = frame.loc[absolute_index].to_dict()
        record["selection_reason"] = "+".join(dict.fromkeys(reasons))
        selected_rows.append(record)

    out = pd.DataFrame(selected_rows)
    for column in SELECTED_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[SELECTED_COLUMNS].reset_index(drop=True)


def _most_stable_family_index(frame: pd.DataFrame) -> Any:
    if "target_family" not in frame.columns:
        return None
    best_family = None
    best_score = -1.0
    for family, group in frame.groupby("target_family"):
        span = float(group["degree"].max() - group["degree"].min())
        score = span * 100.0 + len(group)
        if score > best_score:
            best_score = score
            best_family = family
    if best_family is None:
        return None
    family_rows = frame[frame["target_family"] == best_family]
    median_degree = float(family_rows["degree"].median())
    family_rows = family_rows.assign(_distance=(family_rows["degree"] - median_degree).abs())
    return family_rows.sort_values(["_distance", "degree"]).index[0]


def _validate_degree_window_config(
    *,
    row: dict[str, Any],
    shots: int,
    seed: int,
    case_source: str,
    default_submatrix_size: int,
    grid_size: int,
    phase_timeout_seconds: int,
) -> dict[str, Any]:
    gate = validate_single_codesigned_gate_config(
        row=row,
        shots=int(shots),
        seed=int(seed),
        case_source=str(case_source),
        default_submatrix_size=int(default_submatrix_size),
        grid_size=int(grid_size),
        phase_timeout_seconds=int(phase_timeout_seconds),
        source="deployable_residual_feasible",
    )
    out = {column: np.nan for column in GATE_COLUMNS}
    for column in GATE_COLUMNS:
        if column in gate:
            out[column] = gate[column]
    feasible_after_gate = bool(gate.get("residual_feasible_after_gate", False))
    out["residual_feasible_after_gate"] = feasible_after_gate
    out["degree_window_class_after_gate"] = _class_after_gate(gate)
    out["dominant_gate_limitation"] = gate.get("dominant_limitation", "not_run")
    return out


def _class_after_gate(gate: dict[str, Any]) -> str:
    if bool(gate.get("residual_feasible_after_gate", False)):
        return "residual_feasible"
    if str(gate.get("gate_status")) != "completed":
        return "runtime_limited"
    direction = gate.get("direction_error_gate_vs_ridge", float("nan"))
    try:
        direction_value = float(direction)
    except (TypeError, ValueError):
        direction_value = float("nan")
    if not math.isfinite(direction_value) or direction_value > DIRECTION_FEASIBLE_MAX:
        return "overshoot_risk"
    return "direction_aligned_but_not_residual_feasible"


def write_degree_window_gate_outputs(
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

    selected_path = output_dir / "selected_degree_window_gate_configs.csv"
    results_path = output_dir / "degree_window_gate_results.csv"
    interpretation_path = output_dir / "degree_window_gate_interpretation.md"

    selected.to_csv(selected_path, index=False)
    results.to_csv(results_path, index=False)
    interpretation_path.write_text(degree_window_gate_interpretation(results), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "selected_degree_window_gate_configs": str(selected_path),
            "degree_window_gate_results": str(results_path),
            "degree_window_gate_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=DEGREE_WINDOW_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "selected_degree_window_gate_configs": selected_path,
        "degree_window_gate_results": results_path,
        "degree_window_gate_interpretation": interpretation_path,
    }


def degree_window_gate_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(
            [
                "# Degree-Window Gate-Level Validation",
                "",
                DEGREE_WINDOW_GATE_CLAIM,
                "",
                "- No feasible degree-window configurations were available for gate validation.",
                "",
            ]
        )
    completed = frame[frame["gate_status"] == "completed"]
    feasible = completed[completed["residual_feasible_after_gate"] == True]  # noqa: E712
    survived_degrees = sorted(
        {int(value) for value in pd.to_numeric(feasible["degree"], errors="coerce").dropna()}
    )
    best_ratio = (
        float(pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )
    best_state_poly = (
        float(pd.to_numeric(completed["state_error_gate_vs_polynomial"], errors="coerce").max())
        if not completed.empty
        else float("nan")
    )
    best_direction = (
        float(pd.to_numeric(completed["direction_error_gate_vs_ridge"], errors="coerce").min())
        if not completed.empty
        else float("nan")
    )

    if len(feasible) >= 2:
        headline = (
            "The selected-subproblem solver prototype is gate-validated over a concrete "
            "low-degree window."
        )
    elif len(feasible) == 1:
        headline = (
            "The solver prototype is gate-validated only at a narrow degree point; report this "
            "as a limitation."
        )
    else:
        headline = (
            "The polynomial-level feasibility does not survive gate validation; do not claim "
            "solver prototype readiness."
        )

    return "\n".join(
        [
            "# Degree-Window Gate-Level Validation",
            "",
            DEGREE_WINDOW_GATE_CLAIM,
            "",
            "## Counts",
            f"- Configurations validated: {len(frame)}",
            f"- Completed gate runs: {len(completed)}",
            f"- Residual-feasible after gate: {len(feasible)}",
            f"- Degrees that survived gate validation: {survived_degrees or 'none'}",
            "",
            "## Best Gate Result",
            f"- Best residual_ratio_vs_no_update: {best_ratio:.6g}",
            f"- Worst state_error_gate_vs_polynomial (gate vs polynomial action): "
            f"{best_state_poly:.3g}",
            f"- Best direction_error_gate_vs_ridge: {best_direction:.6g}",
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
