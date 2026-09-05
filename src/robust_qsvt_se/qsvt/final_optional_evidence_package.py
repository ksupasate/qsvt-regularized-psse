from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_solver_prototype_package import (
    cross_case_solver_prototype_classification,
)
from robust_qsvt_se.qsvt.direction_resolved_overshoot_decomposition import NO_FAILURE_MECHANISMS
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

FINAL_OPTIONAL_CLAIM = (
    "Final optional evidence package consolidating the IEEE118 selected (4x4) subproblem "
    "extension and the direction-resolved overshoot decomposition on top of the cross-case "
    "IEEE14/30/57 selected-subproblem QSVT solver prototype. It separates positive evidence "
    "(IEEE118 selected blocks that survive gate validation), boundary evidence (the singular "
    "directions and degree that bound the safe window), assumption-only claims, and unsupported "
    "claims. This remains a selected-subproblem study, not a full IEEE-scale QSVT solver. "
    "Ridge/Tikhonov remains the reference filter; QSVT is an implementation pathway for the same "
    "regularized spectral filter. No full IEEE-scale QSVT solver, quantum speedup, quantum "
    "advantage, QSVT superiority over Ridge/Tikhonov, full nonlinear AC solver, hardware "
    "execution, or real PMU/SCADA validation is claimed."
)

FINAL_OPTIONAL_CLASSIFICATIONS = (
    "solver_prototype_ready_cross_case_selected_family",
    "solver_prototype_ready_cross_case_plus_ieee118_selected",
    "solver_prototype_ready_cross_case_with_ieee118_boundary",
)

CONTROL_MODE = "worst_conditioned_control"
SUMMARY_COLUMNS = ["question", "metric", "value", "source"]
CLAIM_COLUMNS = ["claim", "evidence_class", "value", "source"]


def build_qsvt_final_optional_evidence_package(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_final_optional_evidence_package",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    classification = final_optional_evidence_classification(input_root)
    ieee118 = _ieee118_extension_summary(input_root)
    overshoot = _direction_resolved_summary(input_root)
    summary_rows = final_optional_summary_rows(input_root, classification, ieee118, overshoot)
    claim_rows = final_optional_claim_rows(classification, ieee118)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        classification=classification,
        ieee118=ieee118,
        overshoot=overshoot,
        summary_rows=summary_rows,
        claim_rows=claim_rows,
    )
    return {
        "output_dir": output_dir,
        "classification": classification,
        "ieee118": ieee118,
        "overshoot": overshoot,
        "summary_rows": summary_rows,
        "claim_rows": claim_rows,
        "artifacts": artifacts,
    }


def final_optional_evidence_classification(input_root: Path) -> str:
    """Conservative classification combining the cross-case base with the IEEE118 result.

    - At least one IEEE118 selected subproblem survives gate validation ->
      ``solver_prototype_ready_cross_case_plus_ieee118_selected``.
    - IEEE118 robustness produced candidates but none survive gate validation ->
      ``solver_prototype_ready_cross_case_with_ieee118_boundary``.
    - IEEE118 was not run successfully -> keep the cross-case base classification.
    """

    survivors = _ieee118_gate_survivors(input_root)
    if survivors:
        return "solver_prototype_ready_cross_case_plus_ieee118_selected"
    if _ieee118_had_candidates(input_root) or _ieee118_robustness_ran(input_root):
        return "solver_prototype_ready_cross_case_with_ieee118_boundary"
    # IEEE118 was not run successfully: keep the cross-case base classification unchanged.
    return cross_case_solver_prototype_classification(input_root)


def _ieee118_gate_survivors(input_root: Path) -> list[str]:
    gate = _read_csv(input_root / "qsvt_ieee118_gate_validation" / "ieee118_gate_results.csv")
    if gate.empty or "residual_feasible_after_gate" not in gate.columns:
        return []
    mask = gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
    return sorted(set(gate[mask].get("subproblem_id", pd.Series(dtype=str)).astype(str)))


def _ieee118_had_candidates(input_root: Path) -> bool:
    candidates = _read_csv(
        input_root / "qsvt_ieee118_selected_robustness" / "ieee118_gate_validation_candidates.csv"
    )
    return not candidates.empty


def _ieee118_robustness_ran(input_root: Path) -> bool:
    robustness = _read_csv(
        input_root / "qsvt_ieee118_selected_robustness" / "ieee118_all_configs.csv"
    )
    return not robustness.empty


def _ieee118_extension_summary(input_root: Path) -> dict[str, Any]:
    by_subproblem = _read_csv(
        input_root / "qsvt_ieee118_selected_robustness" / "ieee118_by_subproblem_summary.csv"
    )
    feasible_modes: list[str] = []
    classification = "not_run"
    if not by_subproblem.empty:
        non_control = by_subproblem[by_subproblem["is_control"].astype(str).str.lower() != "true"]
        feasible = non_control[
            non_control["any_residual_feasible"].astype(str).str.lower() == "true"
        ]
        feasible_modes = sorted(set(feasible["selection_mode"].astype(str)))
        n = len(feasible)
        classification = (
            "selected_subproblem_family"
            if n >= 2
            else ("single_selected_block" if n == 1 else "no_feasible_selected_blocks")
        )

    gate = _read_csv(input_root / "qsvt_ieee118_gate_validation" / "ieee118_gate_results.csv")
    gate_survivors = _ieee118_gate_survivors(input_root)
    best_ratio = float("nan")
    worst_state_error = float("nan")
    if not gate.empty:
        completed = gate[gate.get("gate_status", "").astype(str) == "completed"]
        if not completed.empty:
            best_ratio = float(
                pd.to_numeric(completed["residual_ratio_vs_no_update"], errors="coerce").min()
            )
            worst_state_error = float(
                pd.to_numeric(completed["state_error_gate_vs_polynomial"], errors="coerce").max()
            )

    observables = _read_csv(
        input_root / "qsvt_ieee118_gate_observable_readout" / "ieee118_gate_observable_values.csv"
    )
    confirmed_observables = (
        sorted(set(observables["observable_name"].astype(str))) if not observables.empty else []
    )
    return {
        "robustness_classification": classification,
        "feasible_modes": feasible_modes,
        "gate_survivors": gate_survivors,
        "extends": bool(gate_survivors),
        "best_residual_ratio": best_ratio,
        "worst_state_error_gate_vs_polynomial": worst_state_error,
        "confirmed_observables": confirmed_observables,
    }


def _direction_resolved_summary(input_root: Path) -> dict[str, Any]:
    summary = _read_csv(
        input_root
        / "qsvt_direction_resolved_overshoot_decomposition"
        / "direction_resolved_error_summary.csv"
    )
    if summary.empty or "failure_mechanism" not in summary.columns:
        return {
            "dominant_degree_47_mechanism": "not_mapped",
            "dominant_indices": [],
            "reference_safe_degree": None,
            "conservative_safe_degree": None,
            "cases_tested": [],
        }
    frame = summary.copy()
    frame["degree"] = pd.to_numeric(frame["degree"], errors="coerce")
    failing = frame[~frame["failure_mechanism"].isin(NO_FAILURE_MECHANISMS)]
    degree_47 = frame[frame["degree"] == 47]
    failing_47 = degree_47[~degree_47["failure_mechanism"].isin(NO_FAILURE_MECHANISMS)]
    dominant_mechanism = (
        str(failing_47["failure_mechanism"].value_counts().idxmax())
        if not failing_47.empty
        else "no_failure"
    )
    dominant_indices = (
        sorted({int(v) for v in failing_47["dominant_singular_index"].dropna()})
        if not failing_47.empty
        else []
    )
    failing_degrees = {int(v) for v in failing["degree"].dropna()}
    ieee14 = frame[frame["case"].astype(str) == "ieee14"]
    ieee14_failing = {
        int(v)
        for v in ieee14[~ieee14["failure_mechanism"].isin(NO_FAILURE_MECHANISMS)]["degree"].dropna()
    }
    reference_safe = (
        max({int(v) for v in ieee14["degree"].dropna()} - ieee14_failing)
        if not ieee14.empty and ({int(v) for v in ieee14["degree"].dropna()} - ieee14_failing)
        else None
    )
    conservative_safe = (
        max({int(v) for v in frame["degree"].dropna()} - failing_degrees)
        if ({int(v) for v in frame["degree"].dropna()} - failing_degrees)
        else None
    )
    return {
        "dominant_degree_47_mechanism": dominant_mechanism,
        "dominant_indices": dominant_indices,
        "reference_safe_degree": reference_safe,
        "conservative_safe_degree": conservative_safe,
        "cases_tested": sorted(set(frame["case"].astype(str))),
    }


def final_optional_summary_rows(
    input_root: Path,
    classification: str,
    ieee118: dict[str, Any],
    overshoot: dict[str, Any],
) -> list[dict[str, Any]]:
    extends = ieee118["extends"]
    boundary = (
        "IEEE118 selected blocks survive dense gate validation"
        if extends
        else (
            "IEEE118 polynomial-level feasibility did not survive gate validation"
            if ieee118["robustness_classification"] != "not_run"
            else "IEEE118 was not run"
        )
    )
    return [
        _row(
            "Did the selected-subproblem solver prototype extend to IEEE118",
            "ieee118_extends",
            "yes" if extends else "no",
            "qsvt_ieee118_gate_validation",
        ),
        _row(
            "If not, what boundary did IEEE118 reveal",
            "ieee118_boundary",
            boundary,
            "qsvt_ieee118_gate_validation",
        ),
        _row(
            "Which singular directions explain the degree-47 overshoot",
            "degree_47_mechanism",
            f"{overshoot['dominant_degree_47_mechanism']} at singular_index "
            f"{overshoot['dominant_indices'] or 'none'} (leading near-boundary high-weight "
            "directions)",
            "qsvt_direction_resolved_overshoot_decomposition",
        ),
        _row(
            "What is the manuscript-ready degree cutoff",
            "degree_cutoff",
            f"degree {overshoot['reference_safe_degree']} (IEEE14 reference; onset 47); most "
            f"conservative cross-case cutoff {overshoot['conservative_safe_degree']}",
            "qsvt_direction_resolved_overshoot_decomposition",
        ),
        _row(
            "Does this optional evidence change the paper claim",
            "paper_claim_impact",
            (
                "strengthens: the selected-subproblem family now extends to IEEE118 under gate "
                "validation"
                if extends
                else "clarifies limitations only: IEEE118 is boundary evidence"
            ),
            "final_optional_evidence_package",
        ),
        _row(
            "Does this optional evidence change any unsupported claims",
            "unsupported_claims_impact",
            "no: QSVT-over-Ridge superiority, quantum speedup/advantage, full IEEE-scale hardware "
            "execution, and real PMU/SCADA validation remain unsupported",
            "claim_boundary",
        ),
        _row(
            "Final optional evidence classification",
            "classification",
            classification,
            "final_optional_evidence_package",
        ),
    ]


def final_optional_claim_rows(classification: str, ieee118: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ieee118["extends"]:
        rows.append(
            _claim(
                "IEEE118 selected (4x4) blocks survive dense gate validation",
                "positive_evidence",
                f"surviving subproblems: {', '.join(ieee118['gate_survivors'])}",
                "qsvt_ieee118_gate_validation",
            )
        )
    else:
        rows.append(
            _claim(
                "IEEE118 selected-subproblem feasibility under gate validation",
                "boundary_evidence",
                ieee118["robustness_classification"],
                "qsvt_ieee118_gate_validation",
            )
        )
    rows.append(
        _claim(
            "Direction-resolved overshoot decomposition (degree-47 boundary)",
            "boundary_evidence",
            "leading-direction amplitude distortion at near-boundary high-weight directions",
            "qsvt_direction_resolved_overshoot_decomposition",
        )
    )
    rows.append(
        _claim(
            "Full IEEE-scale QSVT sparse-oracle pathway and full output-direction readout",
            "assumption_only",
            "oracle-model assumption; only selected observables are read out",
            "sparse_oracle_assumption_ledger",
        )
    )
    rows.append(
        _claim(
            "QSVT superiority over Ridge/Tikhonov, quantum speedup/advantage, full IEEE-scale "
            "hardware execution, real PMU/SCADA validation",
            "unsupported",
            "explicitly not demonstrated",
            "claim_boundary",
        )
    )
    return rows


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    classification: str,
    ieee118: dict[str, Any],
    overshoot: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    ieee118_path = output_dir / "ieee118_extension_summary.csv"
    overshoot_path = output_dir / "direction_resolved_overshoot_summary.csv"
    claims_path = output_dir / "optional_evidence_claims.csv"
    limitations_path = output_dir / "optional_evidence_limitations.md"
    interpretation_path = output_dir / "optional_evidence_interpretation.md"

    pd.DataFrame([_flatten(ieee118)]).to_csv(ieee118_path, index=False)
    pd.DataFrame([_flatten(overshoot)]).to_csv(overshoot_path, index=False)
    pd.DataFrame(claim_rows, columns=CLAIM_COLUMNS).to_csv(claims_path, index=False)
    limitations_path.write_text(_limitations_markdown(classification), encoding="utf-8")
    interpretation_path.write_text(
        _interpretation_markdown(summary_rows, claim_rows, classification), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "ieee118_extension_summary": str(ieee118_path),
            "direction_resolved_overshoot_summary": str(overshoot_path),
            "optional_evidence_claims": str(claims_path),
            "optional_evidence_limitations": str(limitations_path),
            "optional_evidence_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=FINAL_OPTIONAL_CLAIM,
    )
    return {
        "manifest": manifest,
        "ieee118_extension_summary": ieee118_path,
        "direction_resolved_overshoot_summary": overshoot_path,
        "optional_evidence_claims": claims_path,
        "optional_evidence_limitations": limitations_path,
        "optional_evidence_interpretation": interpretation_path,
    }


def _interpretation_markdown(
    summary_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    classification: str,
) -> str:
    lines = [
        "# Final Optional Evidence Package",
        "",
        FINAL_OPTIONAL_CLAIM,
        "",
        f"## Final Optional Classification: {classification}",
        "",
        "## Summary Answers",
    ]
    for row in summary_rows:
        lines.append(f"- **{row['question']}**: {row['value']} (source: {row['source']}).")
    lines.append("")
    lines.append("## Evidence Separation")
    for cls in ("positive_evidence", "boundary_evidence", "assumption_only", "unsupported"):
        members = [r["claim"] for r in claim_rows if r["evidence_class"] == cls]
        lines.append(f"- **{cls}**:")
        lines.extend(f"  - {item}" for item in members)
    lines.append("")
    return "\n".join(lines)


def _limitations_markdown(classification: str) -> str:
    return "\n".join(
        [
            "# Final Optional Evidence Limitations",
            "",
            FINAL_OPTIONAL_CLAIM,
            "",
            f"Final classification: {classification}.",
            "",
            "## Limitations",
            "- IEEE118 evidence is a criteria-selected 4x4 subproblem test, not a full "
            "IEEE118-scale QSVT solver; rank-deficient (random pool, ill-conditioned) blocks are "
            "not residual-feasible.",
            "- The direction-resolved decomposition explains the degree-47 boundary as a "
            "leading-direction amplitude distortion; it does not extend the safe window, and the "
            "most conservative cross-case cutoff is lower for the hardest IEEE57 blocks.",
            "- Readout of the full output direction is still assumed; only selected observables "
            "are read out, top-k identification is exact only when the top-k magnitudes are well "
            "separated, and full-vector reconstruction stays excluded.",
            "- No full IEEE-scale QSVT solver, hardware execution, quantum speedup, "
            "QSVT-over-Ridge superiority, full nonlinear AC solver, or real PMU/SCADA validation "
            "is demonstrated.",
            "",
        ]
    )


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat[key] = ", ".join(str(v) for v in value) if isinstance(value, list) else value
    return flat


def _row(question: str, metric: str, value: str, source: str) -> dict[str, Any]:
    return {"question": question, "metric": metric, "value": value, "source": source}


def _claim(claim: str, evidence_class: str, value: str, source: str) -> dict[str, Any]:
    return {"claim": claim, "evidence_class": evidence_class, "value": value, "source": source}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
