from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

PROTOTYPE_PACKAGE_CLAIM = (
    "Final selected-subproblem QSVT solver-prototype evidence package. It consolidates the "
    "co-designed QSVT-safe target degree window, the dense gate-level validation, the "
    "gate-level observable-first readout, and the robustness study across criteria-selected "
    "IEEE14-derived subproblems into a single conservative evidence summary. Ridge/Tikhonov "
    "remains the reference filter; QSVT is an implementation pathway for the same "
    "regularized spectral filter. No full IEEE-scale QSVT solver, quantum speedup, quantum "
    "advantage, QSVT superiority over Ridge/Tikhonov, full nonlinear AC solver, hardware "
    "execution, real PMU/SCADA validation, or solved readout bottleneck is claimed."
)

CLAIM_CLASSES = (
    "implemented_selected_solver_prototype",
    "controlled_selected_subproblem_evidence",
    "observable_first_evidence",
    "resource_model_evidence",
    "assumption_only",
    "unsupported",
)

PROTOTYPE_CLASSIFICATIONS = (
    "draft_ready_with_major_limitations",
    "solver_prototype_ready_single_selected_subproblem",
    "solver_prototype_ready_selected_subproblem_family",
)

SUMMARY_COLUMNS = ["question", "metric", "value", "source"]
CLAIM_COLUMNS = ["claim", "claim_class", "evidence_path", "artifact_present", "note"]
DEGREE_WINDOW_COLUMNS = [
    "target_family",
    "min_feasible_degree",
    "max_feasible_degree",
    "overshoot_onset_degree",
    "best_residual_ratio_vs_no_update",
    "best_degree",
]
ROBUSTNESS_COLUMNS = [
    "subproblem_id",
    "selection_mode",
    "condition_number",
    "any_residual_feasible",
    "best_residual_ratio_vs_no_update",
    "best_degree",
    "is_control",
]


def build_qsvt_selected_solver_prototype_package(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_selected_solver_prototype",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    classification = selected_solver_prototype_classification(input_root)
    summary_rows = solver_prototype_summary_rows(input_root, classification)
    claim_rows = solver_prototype_claim_rows(input_root)
    degree_window = _degree_window_summary(input_root)
    gate_results = _gate_validated_solver_results(input_root)
    gate_observables = _gate_observable_results(input_root)
    robustness = _robustness_summary(input_root)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        summary_rows=summary_rows,
        claim_rows=claim_rows,
        degree_window=degree_window,
        gate_results=gate_results,
        gate_observables=gate_observables,
        robustness=robustness,
        classification=classification,
        input_root=input_root,
    )
    return {
        "output_dir": output_dir,
        "classification": classification,
        "summary_rows": summary_rows,
        "claim_rows": claim_rows,
        "artifacts": artifacts,
    }


def selected_solver_prototype_classification(input_root: Path) -> str:
    """Conservative final classification.

    - At least two non-identical criteria-selected subproblems survive gate validation ->
      ``solver_prototype_ready_selected_subproblem_family``.
    - Only the original high-leverage block survives gate validation ->
      ``solver_prototype_ready_single_selected_subproblem``.
    - Otherwise ``draft_ready_with_major_limitations``.
    """

    survivors = _gate_surviving_subproblems(input_root)
    if len(survivors) >= 2:
        return "solver_prototype_ready_selected_subproblem_family"
    if survivors:
        return "solver_prototype_ready_single_selected_subproblem"
    return "draft_ready_with_major_limitations"


def _gate_surviving_subproblems(input_root: Path) -> set[str]:
    survivors: set[str] = set()
    degree_gate = _read_csv(
        input_root / "qsvt_degree_window_gate_validation" / "degree_window_gate_results.csv"
    )
    if _has_feasible_after_gate(degree_gate):
        # Degree-window validation runs on the original high-leverage block.
        survivors.add("high_leverage")
    robustness_gate = _read_csv(
        input_root / "qsvt_robustness_gate_validation" / "robustness_gate_results.csv"
    )
    if not robustness_gate.empty and "residual_feasible_after_gate" in robustness_gate.columns:
        mask = robustness_gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        feasible = robustness_gate[mask]
        key_column = "selection_mode" if "selection_mode" in feasible.columns else "subproblem_id"
        for value in feasible.get(key_column, pd.Series(dtype=str)).astype(str):
            survivors.add(value)
    return survivors


def solver_prototype_summary_rows(input_root: Path, classification: str) -> list[dict[str, Any]]:
    degree_window = _read_csv(
        input_root / "qsvt_degree_window_overshoot" / "feasible_degree_window.csv"
    )
    feasible_degrees = (
        sorted({int(v) for v in pd.to_numeric(degree_window["degree"], errors="coerce").dropna()})
        if not degree_window.empty
        else []
    )
    best_ratio = (
        float(pd.to_numeric(degree_window["residual_ratio_vs_no_update"], errors="coerce").min())
        if not degree_window.empty
        else float("nan")
    )
    degree_gate = _read_csv(
        input_root / "qsvt_degree_window_gate_validation" / "degree_window_gate_results.csv"
    )
    gate_feasible_degrees = _feasible_degrees_after_gate(degree_gate)
    observables = _read_csv(
        input_root / "qsvt_gate_observable_readout" / "gate_observable_values.csv"
    )
    confirmed_observables = (
        sorted(set(observables["observable_name"].astype(str))) if not observables.empty else []
    )
    robustness = _read_csv(
        input_root / "qsvt_codesigned_subproblem_robustness" / "robustness_by_subproblem.csv"
    )
    robust_classification = _robustness_classification_from_frame(robustness)
    survivors = sorted(_gate_surviving_subproblems(input_root))

    rows = [
        _summary_row(
            "What exactly is implemented",
            "implemented",
            "co-designed QSVT-safe bounded-target degree schedule with dense gate-level "
            "validation, gate-level observable-first readout, and criteria-selected "
            "subproblem robustness",
            "selected_solver_prototype_package",
        ),
        _summary_row(
            "Which IEEE-derived subproblems are solved",
            "subproblems",
            ", ".join(survivors) or "none",
            "qsvt_robustness_gate_validation",
        ),
        _summary_row(
            "What degree window is safe",
            "feasible_degree_window",
            f"{feasible_degrees[0]}..{feasible_degrees[-1]} (odd)" if feasible_degrees else "none",
            "qsvt_degree_window_overshoot",
        ),
        _summary_row(
            "What residual ratio is achieved",
            "best_residual_ratio_vs_no_update",
            f"{best_ratio:.6g}",
            "qsvt_degree_window_overshoot",
        ),
        _summary_row(
            "Does gate validation preserve feasibility",
            "gate_feasible_degrees",
            f"{gate_feasible_degrees}" if gate_feasible_degrees else "no",
            "qsvt_degree_window_gate_validation",
        ),
        _summary_row(
            "Which observables are gate-confirmed",
            "gate_confirmed_observables",
            ", ".join(confirmed_observables) or "none",
            "qsvt_gate_observable_readout",
        ),
        _summary_row(
            "How robust is the method across subproblems",
            "robustness_classification",
            robust_classification,
            "qsvt_codesigned_subproblem_robustness",
        ),
        _summary_row(
            "What remains assumption-only",
            "assumption_only",
            "full IEEE-scale QSVT sparse-oracle pathway and readout of the full output direction",
            "sparse_oracle_assumption_ledger",
        ),
        _summary_row(
            "What is explicitly unsupported",
            "unsupported",
            "QSVT superiority over Ridge/Tikhonov; quantum speedup or advantage; full "
            "IEEE-scale hardware execution; real PMU/SCADA validation",
            "claim_boundary",
        ),
        _summary_row(
            "Final prototype classification",
            "classification",
            classification,
            "selected_solver_prototype_package",
        ),
    ]
    return rows


def solver_prototype_claim_rows(input_root: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "Co-designed QSVT-safe bounded targets implement the Ridge filter over a concrete "
            "low-degree window on a selected IEEE14-derived subproblem",
            "implemented_selected_solver_prototype",
            input_root / "qsvt_degree_window_overshoot",
            "Singular-support-weighted and residual-aware targets; degrees 15-45 stay "
            "Ridge-aligned and residual-feasible, overshoot begins at 47.",
        ),
        (
            "The co-designed residual-feasible update survives dense gate-level validation across "
            "the degree window",
            "controlled_selected_subproblem_evidence",
            input_root / "qsvt_degree_window_gate_validation",
            "Phases synthesized from co-designed bounded coefficients; the gate output matches "
            "the polynomial action and the amplitude-recovered residual stays feasible.",
        ),
        (
            "The solver prototype transfers to additional criteria-selected IEEE14-derived "
            "subproblems",
            "controlled_selected_subproblem_evidence",
            input_root / "qsvt_robustness_gate_validation",
            "High-leverage, metadata-mapped, and residual-supported well-conditioned blocks are "
            "gate-validated; rank-deficient and worst-conditioned blocks are not.",
        ),
        (
            "Selected power-system observables are gate-confirmed without full-vector readout",
            "observable_first_evidence",
            input_root / "qsvt_gate_observable_readout",
            "Top-k identification stays exact; bus angle/voltage, branch angle-difference, "
            "measurement-row, and selected-area energy match Ridge to ~1e-3 from the gate output.",
        ),
        (
            "Actual small-circuit amplitude-estimation recovers the update scale",
            "controlled_selected_subproblem_evidence",
            input_root / "qsvt_amplitude_estimation_routines",
            "Bernoulli and iterative MLAE routines estimate the postselection success amplitude.",
        ),
        (
            "Hardware-aware sparse-oracle cost model covers IEEE14/30/57/118/300",
            "resource_model_evidence",
            input_root / "hardware_aware_oracle_cost_model",
            "Oracle-model resource estimate, not hardware-native oracle synthesis.",
        ),
        (
            "Full IEEE-scale QSVT sparse-oracle pathway",
            "assumption_only",
            input_root / "sparse_oracle_assumption_ledger",
            "Oracle-model assumptions only; not implemented circuits.",
        ),
        (
            "Readout of the full QSVT output direction",
            "assumption_only",
            input_root / "qsvt_gate_observable_readout",
            "Only selected observables are read out; full-vector reconstruction stays "
            "not-recommended.",
        ),
        (
            "QSVT numerical superiority over Ridge/Tikhonov",
            "unsupported",
            input_root,
            "Ridge/Tikhonov remains the reference; QSVT implements the same filter.",
        ),
        (
            "Quantum speedup or advantage",
            "unsupported",
            input_root,
            "No complexity or hardware-execution evidence supports this.",
        ),
        (
            "Full IEEE-scale hardware execution or real PMU/SCADA validation",
            "unsupported",
            input_root,
            "Dense simulator evidence on selected subproblems only.",
        ),
    ]
    rows = []
    for claim, claim_class, path, note in specs:
        present = path.exists()
        resolved_class = claim_class if (claim_class == "unsupported" or present) else "unsupported"
        rows.append(
            {
                "claim": claim,
                "claim_class": resolved_class,
                "evidence_path": str(path),
                "artifact_present": present,
                "note": note,
            }
        )
    return rows


def _degree_window_summary(input_root: Path) -> pd.DataFrame:
    boundary = _read_csv(input_root / "qsvt_degree_window_overshoot" / "overshoot_boundary.csv")
    if boundary.empty:
        return pd.DataFrame(columns=DEGREE_WINDOW_COLUMNS)
    numeric = boundary.copy()
    for column in (
        "min_feasible_degree",
        "max_feasible_degree",
        "overshoot_onset_degree",
        "best_residual_ratio_vs_no_update",
        "best_residual_degree",
    ):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for family, group in numeric.groupby("target_family"):
        ratios = group["best_residual_ratio_vs_no_update"].dropna()
        if not ratios.empty:
            best_row = group.loc[ratios.idxmin()]
            best_ratio = float(best_row["best_residual_ratio_vs_no_update"])
            best_degree = float(best_row.get("best_residual_degree", float("nan")))
        else:
            best_ratio = float("nan")
            best_degree = float("nan")
        rows.append(
            {
                "target_family": str(family),
                "min_feasible_degree": float(group["min_feasible_degree"].min()),
                "max_feasible_degree": float(group["max_feasible_degree"].max()),
                "overshoot_onset_degree": float(group["overshoot_onset_degree"].min()),
                "best_residual_ratio_vs_no_update": best_ratio,
                "best_degree": best_degree,
            }
        )
    return pd.DataFrame(rows, columns=DEGREE_WINDOW_COLUMNS)


def _gate_validated_solver_results(input_root: Path) -> pd.DataFrame:
    frames = []
    degree_gate = _read_csv(
        input_root / "qsvt_degree_window_gate_validation" / "degree_window_gate_results.csv"
    )
    if not degree_gate.empty:
        degree_gate = degree_gate.copy()
        degree_gate.insert(0, "source", "degree_window")
        frames.append(degree_gate)
    robustness_gate = _read_csv(
        input_root / "qsvt_robustness_gate_validation" / "robustness_gate_results.csv"
    )
    if not robustness_gate.empty:
        robustness_gate = robustness_gate.copy()
        robustness_gate.insert(0, "source", "robustness")
        frames.append(robustness_gate)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _gate_observable_results(input_root: Path) -> pd.DataFrame:
    return _read_csv(input_root / "qsvt_gate_observable_readout" / "gate_observable_values.csv")


def _robustness_summary(input_root: Path) -> pd.DataFrame:
    frame = _read_csv(
        input_root / "qsvt_codesigned_subproblem_robustness" / "robustness_by_subproblem.csv"
    )
    if frame.empty:
        return pd.DataFrame(columns=ROBUSTNESS_COLUMNS)
    for column in ROBUSTNESS_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[ROBUSTNESS_COLUMNS].reset_index(drop=True)


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    summary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    degree_window: pd.DataFrame,
    gate_results: pd.DataFrame,
    gate_observables: pd.DataFrame,
    robustness: pd.DataFrame,
    classification: str,
    input_root: Path,
) -> dict[str, Path]:
    summary_path = output_dir / "solver_prototype_summary.csv"
    degree_window_path = output_dir / "degree_window_summary.csv"
    gate_results_path = output_dir / "gate_validated_solver_results.csv"
    gate_observable_path = output_dir / "gate_observable_results.csv"
    robustness_path = output_dir / "robustness_summary.csv"
    claims_path = output_dir / "solver_prototype_claims.csv"
    limitations_path = output_dir / "solver_prototype_limitations.md"
    interpretation_path = output_dir / "solver_prototype_interpretation.md"

    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    degree_window.to_csv(degree_window_path, index=False)
    gate_results.to_csv(gate_results_path, index=False)
    gate_observables.to_csv(gate_observable_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    pd.DataFrame(claim_rows, columns=CLAIM_COLUMNS).to_csv(claims_path, index=False)
    limitations_path.write_text(_limitations_markdown(classification), encoding="utf-8")
    interpretation_path.write_text(
        _interpretation_markdown(summary_rows, claim_rows, classification), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "solver_prototype_summary": str(summary_path),
            "degree_window_summary": str(degree_window_path),
            "gate_validated_solver_results": str(gate_results_path),
            "gate_observable_results": str(gate_observable_path),
            "robustness_summary": str(robustness_path),
            "solver_prototype_claims": str(claims_path),
            "solver_prototype_limitations": str(limitations_path),
            "solver_prototype_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=PROTOTYPE_PACKAGE_CLAIM,
    )
    return {
        "manifest": manifest,
        "solver_prototype_summary": summary_path,
        "degree_window_summary": degree_window_path,
        "gate_validated_solver_results": gate_results_path,
        "gate_observable_results": gate_observable_path,
        "robustness_summary": robustness_path,
        "solver_prototype_claims": claims_path,
        "solver_prototype_limitations": limitations_path,
        "solver_prototype_interpretation": interpretation_path,
    }


def _interpretation_markdown(
    summary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    classification: str,
) -> str:
    def claims(claim_class: str) -> list[str]:
        return [
            str(row["claim"])
            for row in claim_rows
            if row["claim_class"] == claim_class and bool(row["artifact_present"])
        ]

    lines = [
        "# Selected Solver-Prototype Evidence Package",
        "",
        PROTOTYPE_PACKAGE_CLAIM,
        "",
        f"## Final Prototype Classification: {classification}",
        "",
        "## Summary Answers",
    ]
    for row in summary_rows:
        lines.append(f"- **{row['question']}**: {row['value']} (source: {row['source']}).")
    lines.extend(
        [
            "",
            "## Implemented Selected Solver Prototype",
            *[f"- {item}" for item in claims("implemented_selected_solver_prototype")],
            "",
            "## Controlled Selected-Subproblem Evidence",
            *[f"- {item}" for item in claims("controlled_selected_subproblem_evidence")],
            "",
            "## Observable-First Evidence",
            *[f"- {item}" for item in claims("observable_first_evidence")],
            "",
            "## Assumption-Only",
            *[f"- {item}" for item in claims("assumption_only")],
            "",
            "## Unsupported (Do Not Claim)",
            *[f"- {row['claim']}" for row in claim_rows if row["claim_class"] == "unsupported"],
            "",
        ]
    )
    return "\n".join(lines)


def _limitations_markdown(classification: str) -> str:
    return "\n".join(
        [
            "# Selected Solver-Prototype Limitations",
            "",
            PROTOTYPE_PACKAGE_CLAIM,
            "",
            f"Final classification: {classification}.",
            "",
            "## Limitations",
            "- Feasibility and gate validation are established only on criteria-selected "
            "IEEE14-derived 4x4 subproblems; this is not an IEEE-scale claim.",
            "- The co-designed targets stay Ridge-aligned and residual-feasible only over a "
            "low-to-moderate odd-degree window (15-45); degree >= 47 overshoots off-support and "
            "loses Ridge-direction alignment.",
            "- Readout of the full output direction is assumed; only the selected observables "
            "(top-k, bus angle/voltage, branch angle-difference, measurement-row, selected-area "
            "energy) are read out, and full-vector reconstruction stays not-recommended.",
            "- Rank-deficient and worst-conditioned blocks are not residual-feasible; robustness "
            "is a property of well-conditioned, residual-supported, criteria-selected blocks.",
            "- The full IEEE-scale QSVT pathway remains an oracle-model assumption; no hardware "
            "execution, quantum speedup, QSVT-over-Ridge superiority, full nonlinear AC solver, "
            "real PMU/SCADA validation, or solved readout bottleneck is demonstrated.",
            "",
        ]
    )


def _has_feasible_after_gate(frame: pd.DataFrame) -> bool:
    if frame.empty or "residual_feasible_after_gate" not in frame.columns:
        return False
    return bool(frame["residual_feasible_after_gate"].astype(str).str.lower().eq("true").any())


def _feasible_degrees_after_gate(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "residual_feasible_after_gate" not in frame.columns:
        return []
    mask = frame["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
    feasible = frame[mask]
    return sorted({int(v) for v in pd.to_numeric(feasible["degree"], errors="coerce").dropna()})


def _robustness_classification_from_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "inconclusive"
    feasible = frame[
        (frame["any_residual_feasible"].astype(str).str.lower() == "true")
        & (frame["is_control"].astype(str).str.lower() != "true")
    ]
    modes = sorted(set(feasible.get("selection_mode", pd.Series(dtype=str)).astype(str)))
    n = len(modes)
    if n == 0:
        return "inconclusive"
    if modes == ["high_leverage"]:
        return "single_block_only"
    if n <= 2:
        return "narrow_selected_family"
    if n == 3:
        return "moderate_selected_family"
    return "broad_selected_family"


def _summary_row(question: str, metric: str, value: str, source: str) -> dict[str, Any]:
    return {"question": question, "metric": metric, "value": value, "source": source}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
