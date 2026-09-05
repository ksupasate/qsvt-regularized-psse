from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.overshoot_mechanism_diagnostic import FEASIBLE_MODES, OVERSHOOT_MODES
from robust_qsvt_se.utils.io import ensure_directory

CROSS_CASE_PACKAGE_CLAIM = (
    "Cross-case selected-subproblem QSVT solver-prototype evidence package. It consolidates the "
    "overshoot-mechanism diagnostic, the cross-case IEEE14/IEEE30/IEEE57 selected-subproblem "
    "robustness, the cross-case dense gate-level validation, and the cross-case gate-level "
    "observable readout into a single conservative evidence summary. Ridge/Tikhonov remains the "
    "reference filter; QSVT is an implementation pathway for the same regularized spectral "
    "filter. This remains a selected-subproblem study. No full IEEE-scale QSVT solver, quantum "
    "speedup, quantum advantage, QSVT superiority over Ridge/Tikhonov, full nonlinear AC solver, "
    "hardware execution, real PMU/SCADA validation, or solved readout bottleneck is claimed."
)

CROSS_CASE_PROTOTYPE_CLASSIFICATIONS = (
    "solver_prototype_ready_ieee14_selected_family",
    "solver_prototype_ready_cross_case_partial",
    "solver_prototype_ready_cross_case_selected_family",
)

CROSS_CASES = ("ieee30", "ieee57")
SUMMARY_COLUMNS = ["question", "metric", "value", "source"]


def build_qsvt_cross_case_solver_prototype_package(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_cross_case_solver_prototype",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    classification = cross_case_solver_prototype_classification(input_root)
    summary_rows = cross_case_solver_summary_rows(input_root, classification)
    overshoot = _read_csv(
        input_root / "qsvt_overshoot_mechanism_diagnostic" / "overshoot_mechanism_summary.csv"
    )
    gate_results = _read_csv(
        input_root / "qsvt_cross_case_gate_validation" / "cross_case_gate_results.csv"
    )
    observables = _read_csv(
        input_root
        / "qsvt_cross_case_gate_observable_readout"
        / "cross_case_gate_observable_values.csv"
    )

    artifacts = _write_outputs(
        output_dir,
        resolved,
        summary_rows=summary_rows,
        overshoot=overshoot,
        gate_results=gate_results,
        observables=observables,
        classification=classification,
    )
    return {
        "output_dir": output_dir,
        "classification": classification,
        "summary_rows": summary_rows,
        "artifacts": artifacts,
    }


def cross_case_solver_prototype_classification(input_root: Path) -> str:
    """Conservative cross-case classification from gate-validation survival.

    - Both IEEE30 and IEEE57 survive gate validation ->
      ``solver_prototype_ready_cross_case_selected_family``.
    - Exactly one of IEEE30/IEEE57 survives -> ``solver_prototype_ready_cross_case_partial``.
    - Neither survives -> ``solver_prototype_ready_ieee14_selected_family``.
    """

    survivors = _gate_surviving_cases(input_root)
    ieee30 = "ieee30" in survivors
    ieee57 = "ieee57" in survivors
    if ieee30 and ieee57:
        return "solver_prototype_ready_cross_case_selected_family"
    if ieee30 or ieee57:
        return "solver_prototype_ready_cross_case_partial"
    return "solver_prototype_ready_ieee14_selected_family"


def _gate_surviving_cases(input_root: Path) -> set[str]:
    gate = _read_csv(input_root / "qsvt_cross_case_gate_validation" / "cross_case_gate_results.csv")
    if gate.empty or "residual_feasible_after_gate" not in gate.columns:
        return set()
    mask = gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
    return set(gate[mask].get("case", pd.Series(dtype=str)).astype(str))


def cross_case_solver_summary_rows(input_root: Path, classification: str) -> list[dict[str, Any]]:
    by_case = _read_csv(
        input_root / "qsvt_cross_case_codesigned_robustness" / "cross_case_by_case_summary.csv"
    )
    cases_tested = sorted(set(by_case["case"].astype(str))) if not by_case.empty else []
    feasible_subproblems = _feasible_subproblems_by_case(input_root)
    survivors = sorted(_gate_surviving_cases(input_root))
    safest_degree, onset_degree, dominant_mode = _overshoot_summary(input_root)
    observable_cases, confirmed_observables = _observable_summary(input_root)

    transfers_beyond_ieee14 = bool(set(survivors) & set(CROSS_CASES))

    return [
        _row(
            "Which IEEE cases were tested",
            "cases_tested",
            ", ".join(cases_tested) or "none",
            "qsvt_cross_case_codesigned_robustness",
        ),
        _row(
            "Which selected subproblems were residual-feasible",
            "residual_feasible_subproblems",
            _format_case_modes(feasible_subproblems),
            "qsvt_cross_case_codesigned_robustness",
        ),
        _row(
            "Which selected subproblems survived gate validation",
            "gate_surviving_cases",
            ", ".join(survivors) or "none",
            "qsvt_cross_case_gate_validation",
        ),
        _row(
            "Does the solver prototype transfer beyond IEEE14",
            "transfers_beyond_ieee14",
            "yes" if transfers_beyond_ieee14 else "no",
            "qsvt_cross_case_gate_validation",
        ),
        _row(
            "What is the safe degree boundary",
            "safe_degree_boundary",
            f"safe up to degree {safest_degree}; overshoot onset {onset_degree}"
            if safest_degree is not None
            else "none",
            "qsvt_overshoot_mechanism_diagnostic",
        ),
        _row(
            "What is the overshoot mechanism",
            "overshoot_mechanism",
            dominant_mode,
            "qsvt_overshoot_mechanism_diagnostic",
        ),
        _row(
            "Which observables transfer across cases",
            "cross_case_observables",
            f"{', '.join(confirmed_observables) or 'none'} (cases: "
            f"{', '.join(observable_cases) or 'none'})",
            "qsvt_cross_case_gate_observable_readout",
        ),
        _row(
            "What remains selected-subproblem-only",
            "selected_subproblem_only",
            "feasibility and gate validation are established only on criteria-selected "
            "IEEE14/30/57 4x4 blocks; not an IEEE-scale claim",
            "claim_boundary",
        ),
        _row(
            "What remains assumption-only",
            "assumption_only",
            "full IEEE-scale QSVT sparse-oracle pathway and readout of the full output direction",
            "sparse_oracle_assumption_ledger",
        ),
        _row(
            "What is explicitly unsupported",
            "unsupported",
            "QSVT superiority over Ridge/Tikhonov; quantum speedup or advantage; full "
            "IEEE-scale hardware execution; real PMU/SCADA validation",
            "claim_boundary",
        ),
        _row(
            "Final cross-case prototype classification",
            "classification",
            classification,
            "cross_case_solver_prototype_package",
        ),
    ]


def _feasible_subproblems_by_case(input_root: Path) -> dict[str, list[str]]:
    by_subproblem = _read_csv(
        input_root
        / "qsvt_cross_case_codesigned_robustness"
        / "cross_case_by_subproblem_summary.csv"
    )
    if by_subproblem.empty:
        return {}
    feasible = by_subproblem[
        (by_subproblem["any_residual_feasible"].astype(str).str.lower() == "true")
        & (by_subproblem["is_control"].astype(str).str.lower() != "true")
    ]
    result: dict[str, list[str]] = {}
    for case, group in feasible.groupby("case"):
        result[str(case)] = sorted(set(group["selection_mode"].astype(str)))
    return result


def _overshoot_summary(input_root: Path) -> tuple[int | None, int | None, str]:
    frame = _read_csv(
        input_root / "qsvt_overshoot_mechanism_diagnostic" / "overshoot_mechanism_summary.csv"
    )
    if frame.empty or "overshoot_failure_mode" not in frame.columns:
        return None, None, "not_mapped"
    frame = frame.copy()
    frame["degree"] = pd.to_numeric(frame["degree"], errors="coerce")
    feasible = frame[frame["overshoot_failure_mode"].isin(FEASIBLE_MODES)]
    overshoot = frame[frame["overshoot_failure_mode"].isin(OVERSHOOT_MODES)]
    overshoot_degrees = {int(v) for v in overshoot["degree"].dropna()}
    onset = min(overshoot_degrees) if overshoot_degrees else None
    clean = {int(v) for v in feasible["degree"].dropna()} - overshoot_degrees
    safest = max(clean) if clean else None
    dominant_mode = (
        str(overshoot["overshoot_failure_mode"].value_counts().idxmax())
        if not overshoot.empty
        else "no_overshoot_observed"
    )
    return safest, onset, dominant_mode


def _observable_summary(input_root: Path) -> tuple[list[str], list[str]]:
    frame = _read_csv(
        input_root
        / "qsvt_cross_case_gate_observable_readout"
        / "cross_case_gate_observable_values.csv"
    )
    if frame.empty:
        return [], []
    cases = sorted(set(frame["case"].astype(str)))
    observables = sorted(set(frame["observable_name"].astype(str)))
    return cases, observables


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    summary_rows: list[dict[str, Any]],
    overshoot: pd.DataFrame,
    gate_results: pd.DataFrame,
    observables: pd.DataFrame,
    classification: str,
) -> dict[str, Path]:
    summary_path = output_dir / "cross_case_solver_summary.csv"
    overshoot_path = output_dir / "overshoot_mechanism_summary.csv"
    gate_path = output_dir / "cross_case_gate_validated_results.csv"
    observable_path = output_dir / "cross_case_observable_results.csv"
    limitations_path = output_dir / "cross_case_limitations.md"
    interpretation_path = output_dir / "cross_case_solver_interpretation.md"

    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    overshoot.to_csv(overshoot_path, index=False)
    gate_results.to_csv(gate_path, index=False)
    observables.to_csv(observable_path, index=False)
    limitations_path.write_text(_limitations_markdown(classification), encoding="utf-8")
    interpretation_path.write_text(
        _interpretation_markdown(summary_rows, classification), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "cross_case_solver_summary": str(summary_path),
            "overshoot_mechanism_summary": str(overshoot_path),
            "cross_case_gate_validated_results": str(gate_path),
            "cross_case_observable_results": str(observable_path),
            "cross_case_limitations": str(limitations_path),
            "cross_case_solver_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CROSS_CASE_PACKAGE_CLAIM,
    )
    return {
        "manifest": manifest,
        "cross_case_solver_summary": summary_path,
        "overshoot_mechanism_summary": overshoot_path,
        "cross_case_gate_validated_results": gate_path,
        "cross_case_observable_results": observable_path,
        "cross_case_limitations": limitations_path,
        "cross_case_solver_interpretation": interpretation_path,
    }


def _interpretation_markdown(summary_rows: list[dict[str, Any]], classification: str) -> str:
    lines = [
        "# Cross-Case Selected Solver-Prototype Evidence Package",
        "",
        CROSS_CASE_PACKAGE_CLAIM,
        "",
        f"## Final Cross-Case Prototype Classification: {classification}",
        "",
        "## Summary Answers",
    ]
    for row in summary_rows:
        lines.append(f"- **{row['question']}**: {row['value']} (source: {row['source']}).")
    lines.append("")
    return "\n".join(lines)


def _limitations_markdown(classification: str) -> str:
    return "\n".join(
        [
            "# Cross-Case Solver-Prototype Limitations",
            "",
            CROSS_CASE_PACKAGE_CLAIM,
            "",
            f"Final classification: {classification}.",
            "",
            "## Limitations",
            "- Feasibility and gate validation are established only on criteria-selected "
            "IEEE14/IEEE30/IEEE57 4x4 subproblems; this is not an IEEE-scale claim.",
            "- The co-designed targets stay Ridge-aligned and residual-feasible only over a "
            "low-to-moderate odd-degree window; degree above the safe boundary loses Ridge "
            "alignment through a high-weight singular-support fit error before any dominant "
            "off-support peak or boundedness breach appears.",
            "- Readout of the full output direction is assumed; only the selected observables "
            "are read out, and full-vector reconstruction stays excluded.",
            "- Rank-deficient and worst-conditioned blocks (including the worst-conditioned "
            "control and the random pool) are not residual-feasible; robustness is a property of "
            "well-conditioned, residual-supported, criteria-selected blocks.",
            "- The full IEEE-scale QSVT pathway remains an oracle-model assumption; no hardware "
            "execution, quantum speedup, QSVT-over-Ridge superiority, full nonlinear AC solver, "
            "real PMU/SCADA validation, or solved readout bottleneck is demonstrated.",
            "",
        ]
    )


def _format_case_modes(case_modes: dict[str, list[str]]) -> str:
    if not case_modes:
        return "none"
    return "; ".join(f"{case}: {', '.join(modes)}" for case, modes in case_modes.items())


def _row(question: str, metric: str, value: str, source: str) -> dict[str, Any]:
    return {"question": question, "metric": metric, "value": value, "source": source}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
