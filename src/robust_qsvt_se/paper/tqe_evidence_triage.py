from __future__ import annotations

# ruff: noqa: E501
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_ROOT = Path("outputs/tqe_qsvt_additional_experiments")

EVIDENCE_COLUMNS = [
    "evidence_id",
    "claim_supported",
    "artifact_path",
    "artifact_type",
    "recommended_location",
    "strength",
    "main_reason",
    "limitation",
    "required_claim_qualifier",
    "recommended_manuscript_section",
    "reviewer_question_answered",
]

REVIEWER_COLUMNS = [
    "question_id",
    "reviewer_question",
    "short_answer",
    "supporting_artifact_paths",
    "main_manuscript_location",
    "supplement_location",
    "limitation_wording",
    "unsafe_wording_to_avoid",
]


def build_evidence_triage(config: dict[str, Any] | None = None) -> dict[str, Any]:
    output_root = Path((config or {}).get("output_root", OUTPUT_ROOT))
    tables_dir = ensure_directory(output_root / "tables")
    reports_dir = ensure_directory(output_root / "reports")
    rows = curated_evidence_rows(output_root)
    evidence = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
    main = evidence[evidence["recommended_location"] == "main"].copy()
    supplement = evidence[evidence["recommended_location"].isin(["supplement", "appendix"])].copy()
    main_csv = tables_dir / "table_main_paper_evidence_map.csv"
    supplement_csv = tables_dir / "table_supplement_evidence_map.csv"
    report_path = reports_dir / "tqe_evidence_triage_report.md"
    checklist = pd.DataFrame(reviewer_question_rows(output_root), columns=REVIEWER_COLUMNS)
    checklist_csv = tables_dir / "table_reviewer_question_checklist.csv"
    checklist_report = reports_dir / "reviewer_question_checklist.md"

    main.to_csv(main_csv, index=False)
    supplement.to_csv(supplement_csv, index=False)
    checklist.to_csv(checklist_csv, index=False)
    report_path.write_text(
        triage_report(evidence=evidence, main_csv=main_csv, supplement_csv=supplement_csv),
        encoding="utf-8",
    )
    checklist_report.write_text(
        reviewer_checklist_markdown(checklist=checklist, checklist_csv=checklist_csv),
        encoding="utf-8",
    )
    return {
        "evidence": evidence,
        "main": main,
        "supplement": supplement,
        "reviewer_checklist": checklist,
        "artifacts": {
            "main_csv": main_csv,
            "supplement_csv": supplement_csv,
            "report": report_path,
            "reviewer_checklist_csv": checklist_csv,
            "reviewer_checklist_report": checklist_report,
        },
    }


def curated_evidence_rows(root: Path) -> list[dict[str, str]]:
    return [
        _row(
            "E1",
            "Degree-alpha-precision tradeoff for bounded Ridge/Tikhonov QSVT targets.",
            root / "tables/table_degree_alpha_precision_summary.csv",
            "main",
            "strong",
            "Directly quantifies degree cost versus alpha and epsilon.",
            "Polynomial degree is a cost diagnostic, not a speedup claim.",
            "QSVT-compatible bounded target; matched Ridge/Tikhonov map.",
            "QSVT implementation pathway: degree tradeoff",
            "Are phases/degrees feasible as alpha changes?",
        ),
        _row(
            "E2",
            "Dense selected-subproblem block encodings are explicitly verified.",
            root / "tables/table_block_encoding_resource_summary.csv",
            "main",
            "strong",
            "Reports gamma, qubits, block error, and unitarity error.",
            "Dense construction is not scalable for full PSSE.",
            "Explicit dense selected-subproblem proof of concept.",
            "QSVT implementation pathway: block encoding",
            "Is block encoding actually implemented?",
        ),
        _row(
            "E3",
            "Circuit-level dense block action reproduces normalized weighted-Jacobian blocks.",
            root / "tables/table_circuit_level_block_encoding_summary.csv",
            "main",
            "strong",
            "Operator and statevector verification support circuit-object evidence.",
            "Dense circuit decomposition is not a scalable sparse-oracle implementation.",
            "Selected-subproblem circuit-level verification.",
            "QSVT implementation pathway: block encoding",
            "Do circuit objects reproduce the block encoding?",
        ),
        _row(
            "E4",
            "QSVT-compatible polynomial update reproduces matched Ridge/Tikhonov update.",
            root / "tables/table_end_to_end_qsvt_vs_ridge_summary.csv",
            "main",
            "strong",
            "Matrix-level update and residual metrics compare directly to matched Ridge.",
            "No QSVT-over-Ridge superiority claim follows from matched-map agreement.",
            "Implementation-pathway consistency with matched spectral map.",
            "Regularized spectral filtering and validation",
            "Is the QSVT target just Ridge?",
        ),
        _row(
            "E5",
            "Integrated gate-level QSVT works on selected 4x4 and rescued 8x8 blocks.",
            root / "full_gate_level_qsvt_coverage/phase_synthesis_8x8_rescue/"
            "phase_synthesis_8x8_rescue_summary.csv",
            "main",
            "strong",
            "Rescued 8x8 rows demonstrate target-admissibility handling.",
            "Selected-subproblem simulator evidence only; transpilation may be budget-skipped.",
            "Admissibility-aware target contraction with matched physical rescaling.",
            "QSVT implementation pathway: gate-level selected subproblems",
            "What happened to the 8x8 failures?",
        ),
        _row(
            "E6",
            "Observable-first readout estimates selected energy-style observables.",
            root / "tables/table_observable_first_readout_summary.csv",
            "main",
            "moderate",
            "Shot-simulation diagnostics separate accessible energy observables from signed values.",
            "Full-vector readout remains outside scope.",
            "Selected-observable readout, not tomography or full-vector recovery.",
            "Readout and observables",
            "How is the quantum output read?",
        ),
        _row(
            "E7",
            "Sparse index/value oracle emulator reconstructs generated weighted Jacobians.",
            root / "tables/table_sparse_oracle_block_encoding_summary.csv",
            "main",
            "moderate",
            "Full-matrix sparsity and oracle-level resources support scalable access assumptions.",
            "No full reversible sparse value-oracle circuit is implemented.",
            "Sparse-access model and resource estimate.",
            "Sparse-oracle scalability model",
            "Is the block encoding scalable?",
        ),
        _row(
            "E8",
            "Nonlinear AC iterations expose per-iteration QSVT degree and conditioning needs.",
            root / "tables/table_nonlinear_ac_qsvt_feasibility_summary.csv",
            "main",
            "diagnostic",
            "Classical nonlinear loop records QSVT-compatible target requirements.",
            "QSVT is not executed inside the nonlinear AC loop.",
            "Per-iteration feasibility diagnostic only.",
            "Nonlinear AC feasibility",
            "Does this work in nonlinear AC PSSE?",
        ),
        _row(
            "E9",
            "Alpha selection shows RMSE/degree tradeoffs.",
            root / "tables/table_alpha_selection_diagnostic_summary.csv",
            "main",
            "diagnostic",
            "Connects alpha choice to regularization stability and QSVT degree cost.",
            "Not a field-calibrated alpha tuning rule.",
            "Diagnostic alpha tradeoff, not operational tuning.",
            "Regularization and alpha selection",
            "How is alpha selected?",
        ),
        _row(
            "S1",
            "Full gate-level coverage, forensic rows, remediation, and rescue details.",
            root / "full_gate_level_qsvt_coverage/full_gate_level_qsvt_coverage_results.csv",
            "supplement",
            "boundary",
            "Raw rows preserve successes, failures, and budget boundaries.",
            "Skipped or failed rows cannot be cited as success.",
            "Use as audit trail and feasibility-boundary evidence.",
            "Supplement: gate-level audit details",
            "Which selected circuits were attempted?",
        ),
        _row(
            "S2",
            "Noise sensitivity of dense proof-of-concept circuits.",
            root / "tables/table_noise_sensitivity_integrated_qsvt.csv",
            "supplement",
            "diagnostic",
            "Documents simulator noise sensitivity as a limitation.",
            "Does not establish hardware robustness.",
            "Noise diagnostic only.",
            "Supplement: robustness audits",
            "What happens under noise?",
        ),
        _row(
            "S3",
            "Reactive P/Q measurement-row composition affects conditioning.",
            root / "tables/table_reactive_pq_row_composition_summary.csv",
            "supplement",
            "diagnostic",
            "Power-system measurement composition supports engineering interpretation.",
            "Not a QSVT-over-Ridge claim.",
            "Conditioning ablation.",
            "Supplement: PSSE engineering audits",
            "Does measurement-row composition matter?",
        ),
        _row(
            "S4",
            "Tiny reversible sparse index-oracle lookup prototype.",
            root / "tables/table_tiny_reversible_sparse_oracle_lookup.csv",
            "supplement",
            "boundary",
            "Shows a toy lookup circuit only.",
            "Not a scalable sparse value oracle.",
            "Tiny prototype only.",
            "Supplement: sparse oracle details",
            "Is the sparse oracle a real circuit?",
        ),
        _row(
            "N1",
            "Skipped-by-budget rows and failed phase synthesis attempts.",
            root / "full_gate_level_qsvt_coverage/full_gate_level_qsvt_degree_remediation.csv",
            "limitation_only",
            "boundary",
            "Documents feasibility boundaries transparently.",
            "Do not use as positive validation evidence.",
            "Report as boundary rows.",
            "Limitations",
            "What are the remaining limitations?",
        ),
    ]


def reviewer_question_rows(root: Path) -> list[dict[str, str]]:
    questions = [
        (
            "RQ1",
            "What does QSVT add beyond Ridge/Tikhonov?",
            "It provides a QSVT-compatible implementation pathway for the matched regularized spectral map.",
            "tables/table_end_to_end_qsvt_vs_ridge_summary.csv; tables/table_degree_alpha_precision_summary.csv",
            "Introduction; QSVT implementation pathway",
            "Degree and end-to-end details",
            "No numerical superiority over Ridge/Tikhonov is claimed.",
            "QSVT outperforms Ridge.",
        ),
        (
            "RQ2",
            "Is the QSVT target just Ridge?",
            "Yes, deliberately: the target is the matched bounded Ridge/Tikhonov spectral map.",
            "reports/target_contraction_physical_rescaling_explanation.md",
            "Regularized spectral filtering",
            "End-to-end solver appendix",
            "The comparison validates implementation consistency, not a better estimator.",
            "QSVT is better than Tikhonov.",
        ),
        (
            "RQ3",
            "Is block encoding actually implemented?",
            "Dense selected-subproblem block encodings and circuit objects are implemented and verified.",
            "tables/table_block_encoding_resource_summary.csv; tables/table_circuit_level_block_encoding_summary.csv",
            "QSVT implementation pathway",
            "Block-encoding audit details",
            "Dense selected-subproblem construction only.",
            "Scalable block encoding is solved.",
        ),
        (
            "RQ4",
            "Is the block encoding scalable?",
            "A sparse-access oracle model is audited; a full scalable reversible circuit is not implemented.",
            "tables/table_sparse_oracle_block_encoding_summary.csv",
            "Sparse-oracle scalability model",
            "Sparse oracle raw audit",
            "Oracle-level resource estimate only.",
            "Scalable sparse oracle implemented.",
        ),
        (
            "RQ5",
            "Are QSVT phases actually synthesized?",
            "Yes for selected low-degree and rescued 8x8 settings; hard-case failures are also reported.",
            "reports/phase_synthesis_8x8_rescue_report.md; tables/table_phase_synthesis_hard_case_audit.csv",
            "Gate-level selected subproblems",
            "Phase-synthesis audit",
            "Only attempted settings are claimed.",
            "All hard cases synthesize.",
        ),
        (
            "RQ6",
            "Do gate-level circuits reproduce the target?",
            "Selected circuits reproduce the synthesized polynomial transform within numerical tolerance.",
            "full_gate_level_qsvt_coverage/phase_synthesis_8x8_rescue/phase_synthesis_8x8_rescue_summary.csv",
            "Gate-level validation",
            "Full coverage audit",
            "Dense simulator circuits only.",
            "Hardware execution is demonstrated.",
        ),
        (
            "RQ7",
            "What happened to the original 8x8 failures?",
            "They were rescued by admissibility-aware target contraction and matched physical rescaling.",
            "reports/phase_synthesis_8x8_rescue_report.md",
            "Gate-level validation",
            "Forensic and remediation reports",
            "The rescue does not imply full-scale execution.",
            "All 8x8 QSVT problems are solved.",
        ),
        (
            "RQ8",
            "How is the quantum output read?",
            "Selected energy-style observables are estimated through computational-basis sampling.",
            "tables/table_observable_first_readout_summary.csv",
            "Readout and observables",
            "Readout raw counts",
            "This is not full-vector recovery.",
            "Full readout is solved.",
        ),
        (
            "RQ9",
            "Can signed quantities be read?",
            "Signed quantities require phase/sign-aware access and are separated from basis-sampling observables.",
            "tables/table_signed_phase_aware_readout_summary.csv",
            "Readout and observables",
            "Signed readout diagnostic",
            "Statevector signed diagnostics are not ordinary basis-shot access.",
            "Basis sampling recovers signed components.",
        ),
        (
            "RQ10",
            "Does this solve full-vector readout?",
            "No. The evidence is selected-observable readout only.",
            "reports/observable_first_readout_report.md",
            "Limitations",
            "Readout supplement",
            "Full-vector recovery remains outside scope.",
            "Full-vector readout solved.",
        ),
        (
            "RQ11",
            "Does this work in nonlinear AC PSSE?",
            "Only per-iteration feasibility diagnostics are provided; QSVT is not inside the loop.",
            "tables/table_nonlinear_ac_qsvt_feasibility_summary.csv",
            "Nonlinear AC feasibility",
            "Nonlinear raw diagnostics",
            "No nonlinear QSVT-in-the-loop estimator is implemented.",
            "QSVT solves nonlinear PSSE.",
        ),
        (
            "RQ12",
            "Does the method outperform Ridge?",
            "No such claim is made; the QSVT-compatible target is matched to Ridge/Tikhonov.",
            "reports/end_to_end_qsvt_vs_ridge_report.md",
            "Limitations",
            "End-to-end details",
            "Agreement with matched Ridge is the validation target.",
            "QSVT outperforms Ridge.",
        ),
        (
            "RQ13",
            "Is there any speedup claim?",
            "No. Resource and implementation-pathway diagnostics are reported without speedup claims.",
            "reports/claim_boundary_audit_report.md",
            "Limitations",
            "Claim-boundary audit",
            "No quantum speedup is demonstrated.",
            "Quantum speedup is demonstrated.",
        ),
        (
            "RQ14",
            "How is alpha selected?",
            "Alpha is treated through diagnostic tradeoffs, not an operational tuning rule.",
            "tables/table_alpha_selection_diagnostic_summary.csv",
            "Regularization and alpha selection",
            "Alpha diagnostic details",
            "Not field-calibrated.",
            "Alpha is optimally tuned for deployment.",
        ),
        (
            "RQ15",
            "What happens under noise?",
            "Noise sensitivity is reported as a limitation for dense proof-of-concept circuits.",
            "tables/table_noise_sensitivity_integrated_qsvt.csv",
            "Limitations",
            "Noise audit",
            "No hardware robustness claim.",
            "The circuit is hardware ready.",
        ),
        (
            "RQ16",
            "Does measurement-row composition matter?",
            "Yes; P/Q and branch/injection row composition affects conditioning.",
            "tables/table_reactive_pq_row_composition_summary.csv",
            "Classical PSSE benchmark",
            "P/Q ablation",
            "Engineering conditioning evidence, not QSVT advantage.",
            "QSVT fixes measurement design.",
        ),
        (
            "RQ17",
            "Is the sparse oracle a real quantum circuit?",
            "Only a tiny index-lookup prototype is built; full reversible value oracle remains future work.",
            "tables/table_tiny_reversible_sparse_oracle_lookup.csv",
            "Sparse-oracle scalability model",
            "Tiny oracle prototype",
            "Do not call it a scalable oracle.",
            "Full sparse oracle is implemented.",
        ),
        (
            "RQ18",
            "Are there hardware results?",
            "No. All circuit evidence is simulator-level.",
            "reports/full_gate_level_qsvt_coverage_report.md",
            "Limitations",
            "Gate-level audit",
            "No hardware execution is claimed.",
            "Hardware execution demonstrated.",
        ),
        (
            "RQ19",
            "What are the remaining limitations?",
            "Scalable oracle circuits, full-vector readout, hardware, speedup, and nonlinear QSVT loop remain future work.",
            "reports/tqe_manuscript_readiness_report.md",
            "Limitations and future work",
            "Readiness package",
            "State limitations explicitly.",
            "Full IEEE-scale solver is complete.",
        ),
        (
            "RQ20",
            "What results belong in supplement?",
            "Raw CSVs, robustness audits, noise, P/Q ablation, oracle prototype, forensic/remediation/rescue details.",
            "tables/table_supplement_evidence_map.csv",
            "Supplement overview",
            "Supplement evidence map",
            "Main paper should use consolidated summaries.",
            "Use every raw audit as a main claim.",
        ),
    ]
    return [
        {
            "question_id": item[0],
            "reviewer_question": item[1],
            "short_answer": item[2],
            "supporting_artifact_paths": str(root / item[3]),
            "main_manuscript_location": item[4],
            "supplement_location": item[5],
            "limitation_wording": item[6],
            "unsafe_wording_to_avoid": item[7],
        }
        for item in questions
    ]


def triage_report(*, evidence: pd.DataFrame, main_csv: Path, supplement_csv: Path) -> str:
    counts = evidence["recommended_location"].value_counts().to_dict()
    strengths = evidence["strength"].value_counts().to_dict()
    return "\n".join(
        [
            "# TQE Evidence Triage Report",
            "",
            f"- Evidence items: {len(evidence)}",
            f"- Location counts: `{counts}`",
            f"- Strength counts: `{strengths}`",
            f"- Main evidence map: `{main_csv}`",
            f"- Supplement evidence map: `{supplement_csv}`",
            "",
            "Main-paper evidence should use consolidated rows and explicit qualifiers. "
            "Raw failures, skipped-by-budget rows, and noise sensitivity details belong "
            "in the supplement or limitations unless directly cited as boundaries.",
            "",
        ]
    )


def reviewer_checklist_markdown(*, checklist: pd.DataFrame, checklist_csv: Path) -> str:
    lines = [
        "# Reviewer-Question Checklist",
        "",
        f"- Checklist CSV: `{checklist_csv}`",
        "",
    ]
    for row in checklist.itertuples(index=False):
        lines.extend(
            [
                f"## {row.question_id}. {row.reviewer_question}",
                "",
                f"- Short answer: {row.short_answer}",
                f"- Supporting artifacts: `{row.supporting_artifact_paths}`",
                f"- Limitation wording: {row.limitation_wording}",
                f"- Unsafe wording to avoid: `{row.unsafe_wording_to_avoid}`",
                "",
            ]
        )
    return "\n".join(lines)


def _row(
    evidence_id: str,
    claim_supported: str,
    artifact_path: Path,
    recommended_location: str,
    strength: str,
    main_reason: str,
    limitation: str,
    qualifier: str,
    section: str,
    reviewer_question: str,
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "claim_supported": claim_supported,
        "artifact_path": str(artifact_path),
        "artifact_type": artifact_path.suffix.lower().lstrip(".") or "directory",
        "recommended_location": recommended_location,
        "strength": strength,
        "main_reason": main_reason,
        "limitation": limitation,
        "required_claim_qualifier": qualifier,
        "recommended_manuscript_section": section,
        "reviewer_question_answered": reviewer_question,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build TQE evidence triage maps")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = build_evidence_triage({"output_root": args.output_root})
    print(f"Wrote evidence triage report to {run['artifacts']['report']}")


if __name__ == "__main__":
    main()
