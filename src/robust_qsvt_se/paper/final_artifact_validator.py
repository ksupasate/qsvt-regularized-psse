"""Phase 5 hardening: final artifact schema and provenance validator.

Checks that the manuscript package is internally consistent, traceable, and not missing
critical source artifacts: indexed figures/tables exist, every figure has source data or
is schematic, referenced claim IDs are real, CSVs have headers, header-only files have a
documented reason, the unsupported / assumption-only claims are preserved, and the
required reproducibility artifacts are present. It reports pass / warning / fail / n/a and
fabricates nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/validate_final_manuscript_artifacts.py"

REPORT_COLUMNS = [
    "check_id",
    "check_name",
    "status",
    "severity",
    "file_path",
    "details",
    "recommended_action",
]

_UNSUPPORTED_CLAIMS = ("C11", "C12", "C13")
_ASSUMPTION_ONLY_CLAIM = "C10"
_CRITICAL_CSVS = (
    "claim_support_matrix_final.csv",
    "manuscript_artifact_index.csv",
    "final_figures/final_figure_index.csv",
    "final_tables/main_table_index.csv",
)
_SELF_DIR = "final_artifact_validation"


def build_final_artifact_validator(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs/final_manuscript_package"))
    output_dir = Path(config.get("output_dir", input_root / _SELF_DIR))
    ensure_directory(output_dir)

    rows: list[dict[str, Any]] = []
    rows.extend(_check_rendered_figures(input_root))
    rows.extend(_check_figure_source_data(input_root))
    rows.extend(_check_main_tables(input_root))
    rows.extend(_check_appendix_tables(input_root))
    rows.extend(_check_claim_ids(input_root))
    rows.extend(_check_csv_headers(input_root, output_dir))
    rows.extend(_check_critical_non_empty(input_root))
    rows.extend(_check_header_only_documented(input_root, output_dir))
    rows.extend(_check_unsupported_claims(input_root))
    rows.extend(_check_assumption_only_claim(input_root))
    rows.extend(_check_qsvt_ridge_equivalence(input_root))
    rows.extend(_check_smoke_not_validation(input_root))
    rows.extend(_check_required_artifacts(input_root))
    rows.extend(_check_readout_hardening_artifacts(input_root))
    rows.extend(_check_readout_codesign_two_view(input_root))
    rows.extend(_check_phase_synthesis_refinement(input_root))
    rows.extend(_check_implementation_verification_layer(input_root))
    rows.extend(_check_pre_manuscript_usability_audit(input_root))
    rows.extend(_check_full_repo_evidence_audit(input_root))
    rows.extend(_check_pre_manuscript_final_gate(input_root))

    return _write_outputs(
        output_dir=output_dir,
        rows=rows,
        input_config={"input_root": str(input_root), "output_dir": str(output_dir)},
    )


def _resolve(input_root: Path, stored: str) -> Path:
    candidate = Path(str(stored))
    if candidate.exists():
        return candidate
    matches = list(input_root.rglob(candidate.name))
    return matches[0] if matches else candidate


def _row(
    check_id: str, name: str, status: str, file_path: str, details: str, action: str = ""
) -> dict[str, Any]:
    severity = {"failed": "high", "warning": "medium"}.get(status, "info")
    return {
        "check_id": check_id,
        "check_name": name,
        "status": status,
        "severity": severity,
        "file_path": file_path,
        "details": details,
        "recommended_action": action or ("none" if status in {"passed", "not_applicable"} else ""),
    }


def _check_rendered_figures(input_root: Path) -> list[dict[str, Any]]:
    index = read_csv(input_root / "final_figures" / "final_figure_index.csv")
    if index.empty:
        return [
            _row("V01", "rendered_figures_exist", "not_applicable", "", "no figure index found")
        ]
    missing: list[str] = []
    for _, record in index.iterrows():
        for column in ("file_pdf", "file_png"):
            path = str(record.get(column, ""))
            if path and not _resolve(input_root, path).exists():
                missing.append(f"{record.get('figure_id', '')}:{column}")
    if missing:
        return [
            _row(
                "V01",
                "rendered_figures_exist",
                "failed",
                "final_figures/final_figure_index.csv",
                f"missing rendered figure files: {', '.join(missing[:10])}",
                "re-run scripts/render_final_manuscript_figures.py",
            )
        ]
    return [
        _row(
            "V01",
            "rendered_figures_exist",
            "passed",
            "final_figures/final_figure_index.csv",
            f"{len(index)} figures present as both PDF and PNG",
        )
    ]


def _check_figure_source_data(input_root: Path) -> list[dict[str, Any]]:
    index = read_csv(input_root / "final_figures" / "final_figure_index.csv")
    if index.empty:
        return [_row("V02", "figure_has_source_or_schematic", "not_applicable", "", "no figures")]
    undocumented = [
        str(record.get("figure_id", ""))
        for _, record in index.iterrows()
        if not str(record.get("source_data", "")).strip()
        and "schematic" not in str(record.get("source_data", "")).lower()
    ]
    if undocumented:
        return [
            _row(
                "V02",
                "figure_has_source_or_schematic",
                "failed",
                "final_figures/final_figure_index.csv",
                f"figures without source data or schematic label: {', '.join(undocumented)}",
                "add source_data or mark the figure schematic",
            )
        ]
    return [
        _row(
            "V02",
            "figure_has_source_or_schematic",
            "passed",
            "final_figures/final_figure_index.csv",
            "every figure has source data or is labelled schematic",
        )
    ]


def _check_main_tables(input_root: Path) -> list[dict[str, Any]]:
    index = read_csv(input_root / "final_tables" / "main_table_index.csv")
    if index.empty:
        return [_row("V03", "main_tables_exist", "not_applicable", "", "no main table index")]
    missing = [
        str(record.get("table_id", ""))
        for _, record in index.iterrows()
        if not _resolve(input_root, str(record.get("selected_output_path", ""))).exists()
    ]
    if missing:
        return [
            _row(
                "V03",
                "main_tables_exist",
                "failed",
                "final_tables/main_table_index.csv",
                f"missing main tables: {', '.join(missing)}",
                "re-run scripts/select_final_manuscript_tables.py",
            )
        ]
    return [
        _row(
            "V03",
            "main_tables_exist",
            "passed",
            "final_tables/main_table_index.csv",
            f"{len(index)} selected main tables present",
        )
    ]


def _check_appendix_tables(input_root: Path) -> list[dict[str, Any]]:
    index = read_csv(input_root / "final_tables" / "appendix_table_index.csv")
    if index.empty:
        return [
            _row("V04", "appendix_tables_exist_or_missing", "not_applicable", "", "no appendix")
        ]
    recorded_missing = read_csv(input_root / "final_tables" / "missing_table_sources.csv")
    missing_ids = (
        set(recorded_missing.get("table_id", pd.Series(dtype=str)).astype(str))
        if not recorded_missing.empty
        else set()
    )
    undocumented = [
        str(record.get("table_id", ""))
        for _, record in index.iterrows()
        if not _resolve(input_root, str(record.get("selected_output_path", ""))).exists()
        and str(record.get("table_id", "")) not in missing_ids
    ]
    if undocumented:
        return [
            _row(
                "V04",
                "appendix_tables_exist_or_missing",
                "failed",
                "final_tables/appendix_table_index.csv",
                f"appendix tables missing and not recorded: {', '.join(undocumented)}",
                "regenerate the table or record it in missing_table_sources.csv",
            )
        ]
    return [
        _row(
            "V04",
            "appendix_tables_exist_or_missing",
            "passed",
            "final_tables/appendix_table_index.csv",
            f"{len(index)} appendix tables present or recorded missing",
        )
    ]


def _claim_ids(input_root: Path) -> set[str]:
    matrix = read_csv(input_root / "claim_support_matrix_final.csv")
    if matrix.empty or "claim_id" not in matrix.columns:
        return set()
    return {str(value).strip() for value in matrix["claim_id"]}


def _check_claim_ids(input_root: Path) -> list[dict[str, Any]]:
    known = _claim_ids(input_root)
    if not known:
        return [_row("V05", "claim_ids_valid", "not_applicable", "", "no claim matrix")]
    unknown: set[str] = set()
    for rel, column in (
        ("final_figures/final_figure_index.csv", "claim_supported"),
        ("final_tables/main_table_index.csv", "claim_supported"),
        ("final_tables/appendix_table_index.csv", "claim_supported"),
    ):
        frame = read_csv(input_root / rel)
        if frame.empty or column not in frame.columns:
            continue
        for cell in frame[column].astype(str):
            for token in cell.replace(",", ";").split(";"):
                claim = token.strip()
                if claim.upper().startswith("C") and claim[1:].isdigit() and claim not in known:
                    unknown.add(claim)
    if unknown:
        return [
            _row(
                "V05",
                "claim_ids_valid",
                "failed",
                "claim_support_matrix_final.csv",
                f"referenced claim IDs not in matrix: {', '.join(sorted(unknown))}",
                "fix the claim reference or add the claim to the matrix",
            )
        ]
    return [
        _row(
            "V05",
            "claim_ids_valid",
            "passed",
            "claim_support_matrix_final.csv",
            "all referenced claim IDs exist in the claim matrix",
        )
    ]


def _iter_csvs(input_root: Path, output_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_root.rglob("*.csv"))
        if output_dir.resolve() not in path.resolve().parents and path.parent.name != _SELF_DIR
    ]


def _check_csv_headers(input_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    headerless: list[str] = []
    for path in _iter_csvs(input_root, output_dir):
        try:
            first = path.read_text(encoding="utf-8").splitlines()[:1]
        except (OSError, UnicodeDecodeError, IndexError):
            first = []
        if not first or not first[0].strip():
            headerless.append(path.name)
    if headerless:
        return [
            _row(
                "V06",
                "csv_files_have_headers",
                "failed",
                str(input_root),
                f"CSV files without a header line: {', '.join(headerless[:10])}",
                "ensure every generated CSV writes a header row",
            )
        ]
    return [
        _row("V06", "csv_files_have_headers", "passed", str(input_root), "all CSVs have headers")
    ]


def _check_critical_non_empty(input_root: Path) -> list[dict[str, Any]]:
    empty = [rel for rel in _CRITICAL_CSVS if read_csv(input_root / rel).empty]
    if empty:
        return [
            _row(
                "V07",
                "critical_csvs_non_empty",
                "failed",
                str(input_root),
                f"critical CSVs empty: {', '.join(empty)}",
                "rebuild the package",
            )
        ]
    return [
        _row(
            "V07",
            "critical_csvs_non_empty",
            "passed",
            str(input_root),
            "all critical CSVs are non-empty",
        )
    ]


def _check_header_only_documented(input_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    undocumented: list[str] = []
    for path in _iter_csvs(input_root, output_dir):
        if "missing" in path.name.lower():
            continue
        if not read_csv(path).empty:
            continue
        siblings = list(path.parent.glob("missing_*.csv")) + list(path.parent.glob("*missing*.csv"))
        if not siblings:
            undocumented.append(path.name)
    if undocumented:
        return [
            _row(
                "V08",
                "header_only_files_documented",
                "warning",
                str(input_root),
                f"header-only CSVs without a sibling missing/unavailable record: "
                f"{', '.join(undocumented[:10])}",
                "document the missing/unavailable reason next to the header-only file",
            )
        ]
    return [
        _row(
            "V08",
            "header_only_files_documented",
            "passed",
            str(input_root),
            "header-only files have a documented missing/unavailable reason",
        )
    ]


def _claim_status(input_root: Path, claim_id: str) -> str | None:
    matrix = read_csv(input_root / "claim_support_matrix_final.csv")
    if matrix.empty or "claim_id" not in matrix.columns:
        return None
    hit = matrix[matrix["claim_id"].astype(str) == claim_id]
    if hit.empty:
        return None
    return str(hit.iloc[0].get("support_status", ""))


def _check_unsupported_claims(input_root: Path) -> list[dict[str, Any]]:
    promoted = [
        claim
        for claim in _UNSUPPORTED_CLAIMS
        if (_claim_status(input_root, claim) or "") not in {"unsupported_do_not_claim", ""}
    ]
    missing_matrix = _claim_status(input_root, _UNSUPPORTED_CLAIMS[0]) is None
    if missing_matrix:
        return [
            _row("V09", "unsupported_claims_preserved", "not_applicable", "", "no claim matrix")
        ]
    if promoted:
        return [
            _row(
                "V09",
                "unsupported_claims_preserved",
                "failed",
                "claim_support_matrix_final.csv",
                f"unsupported claims promoted: {', '.join(promoted)}",
                "restore unsupported_do_not_claim for C11/C12/C13",
            )
        ]
    return [
        _row(
            "V09",
            "unsupported_claims_preserved",
            "passed",
            "claim_support_matrix_final.csv",
            "C11/C12/C13 remain unsupported_do_not_claim",
        )
    ]


def _check_assumption_only_claim(input_root: Path) -> list[dict[str, Any]]:
    status = _claim_status(input_root, _ASSUMPTION_ONLY_CLAIM)
    if status is None:
        return [_row("V10", "c10_assumption_only", "not_applicable", "", "C10 not in matrix")]
    if status != "assumption_only":
        return [
            _row(
                "V10",
                "c10_assumption_only",
                "failed",
                "claim_support_matrix_final.csv",
                f"C10 full-vector readout is '{status}', expected assumption_only",
                "restore assumption_only for C10",
            )
        ]
    return [
        _row(
            "V10",
            "c10_assumption_only",
            "passed",
            "claim_support_matrix_final.csv",
            "C10 full-vector readout remains assumption_only",
        )
    ]


def _check_qsvt_ridge_equivalence(input_root: Path) -> list[dict[str, Any]]:
    needles = ("equal to ridge", "equals ridge", "equivalent to ridge", "identical to ridge")
    for path in sorted(input_root.rglob("*.md")) + sorted(input_root.rglob("*.csv")):
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        if any(needle in text for needle in needles):
            return [
                _row(
                    "V11",
                    "qsvt_ridge_equivalence_documented",
                    "passed",
                    str(path),
                    "QSVT-target / Ridge equivalence is documented",
                )
            ]
    return [
        _row(
            "V11",
            "qsvt_ridge_equivalence_documented",
            "warning",
            str(input_root),
            "QSVT-target / Ridge equivalence wording not found",
            "state the QSVT-target == Ridge equivalence in a summary or table",
        )
    ]


def _check_smoke_not_validation(input_root: Path) -> list[dict[str, Any]]:
    for rel in (
        "test_quality_appendix/test_quality_appendix_table.csv",
        "evidence_freeze/frozen_test_summary.csv",
    ):
        frame = read_csv(input_root / rel)
        if frame.empty:
            continue
        column = "counts_as_scientific_validation"
        name_col = "metric" if "metric" in frame.columns else "test_category"
        if column not in frame.columns or name_col not in frame.columns:
            continue
        smoke = frame[frame[name_col].astype(str) == "smoke_only"]
        if not smoke.empty and str(smoke.iloc[0][column]).lower() in {"no", "false"}:
            return [
                _row(
                    "V12",
                    "smoke_not_scientific_validation",
                    "passed",
                    rel,
                    "smoke tests are not counted as scientific validation",
                )
            ]
    return [
        _row(
            "V12",
            "smoke_not_scientific_validation",
            "warning",
            str(input_root),
            "could not confirm smoke tests are excluded from scientific validation",
            "ensure the test-quality appendix marks smoke tests as non-validation",
        )
    ]


def _check_required_artifacts(input_root: Path) -> list[dict[str, Any]]:
    required = (
        ("V13", "evidence_freeze_present", "evidence_freeze/EVIDENCE_FREEZE.md", "failed"),
        ("V14", "final_rebuild_commands_present", "final_rebuild_commands.md", "failed"),
        ("V15", "do_not_claim_present", "claim_boundaries/DO_NOT_CLAIM.md", "failed"),
        ("V16", "claim_lint_present", "claim_lint/claim_lint_report.csv", "warning"),
        (
            "V17",
            "canonical_numbers_present",
            "canonical_numbers/canonical_paper_numbers.json",
            "warning",
        ),
        (
            "V18",
            "pre_submission_check_present",
            "pre_submission_check/pre_submission_check_summary.md",
            "warning",
        ),
    )
    rows: list[dict[str, Any]] = []
    for check_id, name, rel, fail_status in required:
        if (input_root / rel).exists():
            rows.append(_row(check_id, name, "passed", rel, "present"))
        else:
            rows.append(
                _row(
                    check_id,
                    name,
                    fail_status,
                    rel,
                    "required artifact not found",
                    "generate the artifact (see final_rebuild_commands.md)",
                )
            )
    return rows


_READOUT_HARDENING_ARTIFACTS = (
    "full_vector_readout/gate_level_full_vector_summary.csv",
    "full_vector_readout/global_sign_audit.csv",
    "full_vector_readout/shot_based_norm_recovery.csv",
    "full_vector_readout/readout_confidence_intervals.csv",
    "full_vector_readout/sign_projector_validation.csv",
    "full_vector_readout/full_vector_readout_total_cost.csv",
    "full_vector_readout/readout_error_decomposition.csv",
    "full_vector_readout/sampling_seed_manifest.csv",
)


def _check_readout_hardening_artifacts(input_root: Path) -> list[dict[str, Any]]:
    """V19: recognize the readout-hardening artifacts and confirm no Ridge sign leakage.

    Not-applicable when the readout phase was not run; a warning (never a hard failure) when
    the directory exists but some hardening artifacts are missing or a global-sign audit row
    uses Ridge for reconstruction.
    """

    readout_dir = input_root / "full_vector_readout"
    if not readout_dir.is_dir():
        return [_row("V19", "readout_hardening_artifacts", "not_applicable", "", "readout not run")]
    missing = [rel for rel in _READOUT_HARDENING_ARTIFACTS if not (input_root / rel).is_file()]
    if missing:
        return [
            _row(
                "V19",
                "readout_hardening_artifacts",
                "warning",
                str(readout_dir),
                "readout-hardening artifacts missing: "
                + ", ".join(p.split("/")[-1] for p in missing),
                "run scripts/run_full_vector_readout_demo.py and rebuild the package",
            )
        ]
    audit = read_csv(input_root / "full_vector_readout" / "global_sign_audit.csv")
    if (
        not audit.empty
        and "uses_ridge_for_reconstruction" in audit.columns
        and not (audit["uses_ridge_for_reconstruction"].astype(str).str.lower() == "no").all()
    ):
        return [
            _row(
                "V19",
                "readout_hardening_artifacts",
                "warning",
                "full_vector_readout/global_sign_audit.csv",
                "a global-sign audit row uses Ridge for reconstruction (expected no)",
                "fix the reconstruction to use a Ridge-independent global-sign convention",
            )
        ]
    return [
        _row(
            "V19",
            "readout_hardening_artifacts",
            "passed",
            str(readout_dir),
            "readout-hardening artifacts present; no Ridge sign leakage in reconstruction",
        )
    ]


_READOUT_CODESIGN_ARTIFACTS = (
    "full_vector_readout/readout_matched_alpha_view.csv",
    "full_vector_readout/readout_per_subproblem_validated_view.csv",
    "full_vector_readout/readout_two_view_summary.csv",
)


def _check_readout_codesign_two_view(input_root: Path) -> list[dict[str, Any]]:
    """V20: two-view co-design artifacts exist, high matched-alpha errors are preserved, and
    no full-scale readout is claimed solved.

    Not-applicable when the co-design pass was not run; a warning (never a hard failure) when a
    view is missing, a matched-alpha high-error row is dropped from the diagnostic view, or the
    two-view summary marks the full-scale readout as solved.
    """

    matched = input_root / "full_vector_readout" / "readout_matched_alpha_view.csv"
    if not matched.is_file():
        return [_row("V20", "readout_codesign_two_view", "not_applicable", "", "co-design not run")]
    missing = [rel for rel in _READOUT_CODESIGN_ARTIFACTS if not (input_root / rel).is_file()]
    if missing:
        return [
            _row(
                "V20",
                "readout_codesign_two_view",
                "warning",
                str(input_root / "full_vector_readout"),
                "two-view co-design artifacts missing: "
                + ", ".join(p.split("/")[-1] for p in missing),
                "run scripts/run_readout_alpha_degree_sweep.py and "
                "scripts/build_readout_two_view_summary.py",
            )
        ]
    matched_view = read_csv(matched)
    if "gate_error_vs_ridge" in matched_view.columns:
        errors = pd.to_numeric(matched_view["gate_error_vs_ridge"], errors="coerce").dropna()
        if errors.empty or float(errors.max()) <= 0.05:
            return [
                _row(
                    "V20",
                    "readout_codesign_two_view",
                    "warning",
                    str(matched),
                    "matched-alpha view has no high-error diagnostic row (expected preserved "
                    "finite-degree cases)",
                    "regenerate the matched-alpha view without filtering high-error rows",
                )
            ]
    summary = read_csv(input_root / "full_vector_readout" / "readout_two_view_summary.csv")
    interpretations = " ".join(
        summary.get("manuscript_interpretation", pd.Series(dtype=str)).astype(str).tolist()
    ).lower()
    if (
        "full ieee-scale full-vector readout" in interpretations
        and "outside" not in interpretations
    ):
        return [
            _row(
                "V20",
                "readout_codesign_two_view",
                "warning",
                str(input_root / "full_vector_readout" / "readout_two_view_summary.csv"),
                "two-view summary appears to promote full-scale readout",
                "restore the conservative scope wording",
            )
        ]
    return [
        _row(
            "V20",
            "readout_codesign_two_view",
            "passed",
            str(input_root / "full_vector_readout"),
            "two-view co-design present; matched-alpha high-error cases preserved; "
            "no full-scale readout claimed",
        )
    ]


_PHASE_REFINEMENT_ARTIFACTS = (
    "phase_synthesis_refinement/baseline_current_readout_codesign.csv",
    "phase_synthesis_refinement/degree_47_49_diagnostic_sweep.csv",
    "phase_synthesis_refinement/phase_solver_variant_comparison.csv",
    "phase_synthesis_refinement/polynomial_fit_mode_comparison.csv",
    "phase_synthesis_refinement/three_view_readout_summary.csv",
    "phase_synthesis_refinement/selection_leakage_audit.csv",
    "phase_synthesis_refinement/phase_failure_taxonomy.csv",
    "phase_synthesis_refinement/phase_synthesis_refinement_scope_note.md",
)


def _check_phase_synthesis_refinement(input_root: Path) -> list[dict[str, Any]]:
    """V21: phase-synthesis refinement artifacts exist, selection has no true-state leakage, and
    the scope note keeps the conservative claim boundary (no full-scale readout / no speedup).

    Not-applicable when refinement was not run; a warning (never a hard failure) when an artifact
    is missing, the selection-leakage audit shows true-state/RMSE use, or the scope note promotes
    full-scale readout or QSVT superiority over Ridge.
    """

    baseline = input_root / "phase_synthesis_refinement" / "baseline_current_readout_codesign.csv"
    if not baseline.is_file():
        return [
            _row("V21", "phase_synthesis_refinement", "not_applicable", "", "refinement not run")
        ]
    missing = [rel for rel in _PHASE_REFINEMENT_ARTIFACTS if not (input_root / rel).is_file()]
    if missing:
        return [
            _row(
                "V21",
                "phase_synthesis_refinement",
                "warning",
                str(input_root / "phase_synthesis_refinement"),
                "phase-synthesis refinement artifacts missing: "
                + ", ".join(p.split("/")[-1] for p in missing),
                "run scripts/run_phase_synthesis_refinement.py",
            )
        ]
    leakage = read_csv(input_root / "phase_synthesis_refinement" / "selection_leakage_audit.csv")
    for column in ("uses_true_state", "uses_state_rmse"):
        if column in leakage.columns:
            flags = leakage[column].astype(str).str.lower().isin({"true", "1"})
            if bool(flags.any()):
                return [
                    _row(
                        "V21",
                        "phase_synthesis_refinement",
                        "warning",
                        str(
                            input_root
                            / "phase_synthesis_refinement"
                            / "selection_leakage_audit.csv"
                        ),
                        f"selection-leakage audit flags {column}; selection must not use the "
                        "true state or true-state RMSE",
                        "remove true-state/oracle metrics from config selection",
                    )
                ]
    scope = input_root / "phase_synthesis_refinement" / "phase_synthesis_refinement_scope_note.md"
    text = scope.read_text(encoding="utf-8").lower() if scope.is_file() else ""
    promotes_full_scale = (
        "efficient full ieee-scale full-vector readout" in text and "does not" not in text
    )
    # match only the affirmative overclaim, not the negated boundary statement
    affirms_superiority = (
        "qsvt outperforms ridge" in text and "not a claim that qsvt outperforms ridge" not in text
    )
    claims_superiority = affirms_superiority or "quantum speedup is demonstrated" in text
    if promotes_full_scale or claims_superiority:
        return [
            _row(
                "V21",
                "phase_synthesis_refinement",
                "warning",
                str(scope),
                "phase-synthesis refinement scope note appears to overclaim (full-scale readout "
                "or QSVT superiority/speedup)",
                "restore the conservative scope wording",
            )
        ]
    return [
        _row(
            "V21",
            "phase_synthesis_refinement",
            "passed",
            str(input_root / "phase_synthesis_refinement"),
            "phase-synthesis refinement present; no true-state selection leakage; "
            "conservative scope preserved",
        )
    ]


_IMPLEMENTATION_VERIFICATION_REQUIRED = {
    "jacobian_validation": (
        "ac_jacobian_finite_difference_validation.csv",
        "ac_jacobian_validation_summary.csv",
        "ac_jacobian_validation_summary.md",
        "dc_jacobian_finite_difference_validation.csv",
        "dc_jacobian_validation_summary.csv",
        "dc_jacobian_validation_summary.md",
        "weighted_jacobian_consistency_audit.csv",
        "weighted_jacobian_consistency_summary.md",
    ),
    "measurement_row_metadata_audit": (
        "row_metadata_audit.csv",
        "subset_row_composition_summary.csv",
        "row_mask_consistency_checks.csv",
        "measurement_row_metadata_summary.md",
    ),
    "statistical_summary": (
        "estimator_seed_variability_summary.csv",
        "nonlinear_convergence_summary.csv",
        "measurement_ablation_statistical_summary.csv",
        "reactive_conditioning_statistical_summary.csv",
        "readout_sampling_statistical_summary.csv",
        "phase_refinement_statistical_summary.csv",
        "statistical_aggregation_manifest.csv",
        "statistical_aggregation_summary.md",
    ),
    "main_tables": (
        "T1_measurement_inventory.csv",
        "T2_classical_alpha_summary.csv",
        "T3_reactive_conditioning_summary.csv",
        "T4_structured_stress_baseline_summary.csv",
        "T5_nonlinear_ac_summary.csv",
        "T6_qsvt_gate_readout_summary.csv",
        "T7_phase_refinement_summary.csv",
        "T8_claim_boundary_summary.csv",
        "main_tables_manifest.csv",
        "main_tables_summary.md",
    ),
}


def _check_implementation_verification_layer(input_root: Path) -> list[dict[str, Any]]:
    present_dirs = [
        name for name in _IMPLEMENTATION_VERIFICATION_REQUIRED if (input_root / name).is_dir()
    ]
    if not present_dirs:
        return [
            _row(
                "V22",
                "implementation_verification_layer",
                "not_applicable",
                "",
                "implementation-verification layer not present",
            )
        ]

    missing = [
        f"{directory}/{filename}"
        for directory, filenames in _IMPLEMENTATION_VERIFICATION_REQUIRED.items()
        for filename in filenames
        if not (input_root / directory / filename).is_file()
    ]
    if missing:
        return [
            _row(
                "V22",
                "implementation_verification_layer",
                "failed",
                str(input_root),
                "implementation-verification files missing: " + ", ".join(missing[:12]),
                "run the Jacobian, row-metadata, statistical-summary, and main-table builders",
            )
        ]

    failures = _implementation_verification_failures(input_root)
    if failures:
        return [
            _row(
                "V22",
                "implementation_verification_layer",
                "failed",
                str(input_root),
                "; ".join(failures[:10]),
                "inspect and regenerate the failing verification artifacts",
            )
        ]
    return [
        _row(
            "V22",
            "implementation_verification_layer",
            "passed",
            str(input_root),
            "Jacobian validation, row metadata/masks, statistical summaries, and main tables "
            "are present with no failure statuses or boundary violations",
        )
    ]


def _implementation_verification_failures(input_root: Path) -> list[str]:
    failures: list[str] = []
    for rel in (
        "jacobian_validation/ac_jacobian_validation_summary.csv",
        "jacobian_validation/dc_jacobian_validation_summary.csv",
    ):
        frame = read_csv(input_root / rel)
        if "status" in frame.columns:
            bad = frame["status"].astype(str).str.contains("fail|build_failed", case=False)
            if bool(bad.any()):
                failures.append(f"{rel} contains failing status rows")

    weighted = read_csv(input_root / "jacobian_validation/weighted_jacobian_consistency_audit.csv")
    if "status" in weighted.columns:
        bad = weighted["status"].astype(str).isin(["invalid_variance", "build_failed"])
        if bool(bad.any()):
            failures.append("weighted Jacobian audit contains invalid variance/build failures")

    masks = read_csv(
        input_root / "measurement_row_metadata_audit" / "row_mask_consistency_checks.csv"
    )
    if "status" in masks.columns:
        bad = masks["status"].astype(str).str.startswith("fail")
        if bool(bad.any()):
            failures.append("row mask audit contains failure statuses")

    stats = read_csv(input_root / "statistical_summary" / "estimator_seed_variability_summary.csv")
    if "qsvt_outperforms_ridge" in stats.columns:
        trueish = stats["qsvt_outperforms_ridge"].astype(str).str.lower().isin({"true", "1", "yes"})
        if bool(trueish.any()):
            failures.append("statistical summary reports QSVT target as outperforming Ridge")

    t6 = read_csv(input_root / "main_tables" / "T6_qsvt_gate_readout_summary.csv")
    if not t6.empty:
        text = " ".join(t6.astype(str).to_numpy().ravel()).lower()
        if "full ieee-scale" in text:
            failures.append("T6 appears to claim full IEEE-scale readout")

    t8 = read_csv(input_root / "main_tables" / "T8_claim_boundary_summary.csv")
    if not t8.empty and {"claim_id", "status"}.issubset(t8.columns):
        status = dict(zip(t8["claim_id"].astype(str), t8["status"].astype(str), strict=False))
        for claim_id in ("C11", "C12", "C13"):
            if status.get(claim_id) != "unsupported_do_not_claim":
                failures.append(f"{claim_id} is not preserved as unsupported_do_not_claim in T8")
        if status.get("C10") != "assumption_only":
            failures.append("C10 full-readout boundary is not preserved as assumption_only in T8")
    return failures


_PRE_MANUSCRIPT_REQUIRED = (
    "package_snapshot.csv",
    "package_snapshot_summary.md",
    "stale_artifact_audit.csv",
    "stale_artifact_audit_summary.md",
    "table_usability_audit.csv",
    "table_usability_summary.md",
    "warning_classification.csv",
    "warning_classification_summary.md",
    "status_value_inventory.csv",
    "status_value_summary.md",
    "numeric_sanity_audit.csv",
    "numeric_sanity_summary.md",
    "claim_to_artifact_map.csv",
    "claim_to_artifact_summary.md",
    "rebuild_reproducibility_audit.csv",
    "rebuild_reproducibility_summary.md",
    "paper_use_decision_manifest.csv",
    "paper_use_decision_summary.md",
    "pre_manuscript_usability_scorecard.csv",
    "pre_manuscript_usability_summary.md",
)


def _check_pre_manuscript_usability_audit(input_root: Path) -> list[dict[str, Any]]:
    base = input_root / "pre_manuscript_usability_audit"
    if not base.is_dir():
        return [
            _row(
                "V23",
                "pre_manuscript_usability_audit",
                "not_applicable",
                "",
                "pre-manuscript usability audit not present",
            )
        ]
    missing = [name for name in _PRE_MANUSCRIPT_REQUIRED if not (base / name).is_file()]
    if missing:
        return [
            _row(
                "V23",
                "pre_manuscript_usability_audit",
                "failed",
                str(base),
                "pre-manuscript audit files missing: " + ", ".join(missing[:12]),
                "run scripts/run_pre_manuscript_usability_audit.py",
            )
        ]
    failures = _pre_manuscript_failures(base)
    if failures:
        return [
            _row(
                "V23",
                "pre_manuscript_usability_audit",
                "failed",
                str(base),
                "; ".join(failures[:10]),
                "fix audit blockers before using the package for manuscript writing",
            )
        ]
    return [
        _row(
            "V23",
            "pre_manuscript_usability_audit",
            "passed",
            str(base),
            "pre-manuscript usability audit is present and reports no blockers",
        )
    ]


_FULL_REPO_AUDIT_REQUIRED = (
    "repo_environment_snapshot.csv",
    "planned_experiment_inventory.csv",
    "source_to_output_provenance.csv",
    "reproducibility_rerun_audit.csv",
    "suspicious_result_audit.csv",
    "test_quality_full_repo_audit.csv",
    "smoke_only_tests.csv",
    "real_validation_tests.csv",
    "scientific_validation_suite.csv",
    "claim_boundary_full_repo_audit.csv",
    "unsupported_claim_audit.csv",
    "experiment_completion_matrix.csv",
    "artifact_use_readiness_audit.csv",
    "full_repo_evidence_scorecard.csv",
    "full_repo_evidence_audit_summary.md",
)


_PRE_MANUSCRIPT_FINAL_GATE_REQUIRED = (
    "needs_review_rows_extracted.csv",
    "needs_review_extraction_summary.md",
    "needs_review_triage.csv",
    "needs_review_triage_summary.md",
    "main_claim_impact_audit.csv",
    "main_claim_impact_summary.md",
    "needs_review_artifact_use_crosscheck.csv",
    "needs_review_artifact_use_summary.md",
    "needs_review_numeric_status_check.csv",
    "needs_review_numeric_status_summary.md",
    "main_claim_test_support_audit.csv",
    "main_claim_test_support_summary.md",
    "final_evidence_freeze_manifest.csv",
    "final_evidence_freeze_summary.md",
    "manuscript_readiness_decision.csv",
    "manuscript_readiness_decision.md",
)


def _check_full_repo_evidence_audit(input_root: Path) -> list[dict[str, Any]]:
    """V24: full-repo audit exists and reports no blockers.

    Missing full-repo audit files are warnings so older package fixtures stay valid. Once the
    audit exists, real blockers in its scorecard are validation failures; documented future work
    and limitations are not failures.
    """

    base = input_root / "full_repo_evidence_audit"
    if not base.is_dir():
        return [
            _row(
                "V24",
                "full_repo_evidence_audit",
                "warning",
                "full_repo_evidence_audit",
                "full-repository evidence audit not present",
                "run scripts/run_full_repo_evidence_audit.py and rebuild the package",
            )
        ]
    missing = [name for name in _FULL_REPO_AUDIT_REQUIRED if not (base / name).is_file()]
    if missing:
        return [
            _row(
                "V24",
                "full_repo_evidence_audit",
                "warning",
                str(base),
                "full-repo audit files missing: " + ", ".join(missing[:10]),
                "rerun scripts/run_full_repo_evidence_audit.py",
            )
        ]
    scorecard = read_csv(base / "full_repo_evidence_scorecard.csv")
    if scorecard.empty or "category" not in scorecard.columns:
        return [
            _row(
                "V24",
                "full_repo_evidence_audit",
                "warning",
                str(base / "full_repo_evidence_scorecard.csv"),
                "scorecard missing or malformed",
                "rerun scripts/run_full_repo_evidence_audit.py",
            )
        ]
    overall = scorecard[scorecard["category"].astype(str) == "overall"]
    if overall.empty:
        return [
            _row(
                "V24",
                "full_repo_evidence_audit",
                "warning",
                str(base / "full_repo_evidence_scorecard.csv"),
                "scorecard has no overall row",
                "rerun scripts/run_full_repo_evidence_audit.py",
            )
        ]
    row = overall.iloc[0]
    try:
        blockers = int(row.get("blocker_count", 0))
    except (TypeError, ValueError):
        blockers = 0
    status = str(row.get("status", ""))
    if blockers > 0 or status == "blocked":
        return [
            _row(
                "V24",
                "full_repo_evidence_audit",
                "failed",
                str(base / "full_repo_evidence_scorecard.csv"),
                f"full-repo audit reports status={status}, blockers={blockers}",
                "fix audit blockers before manuscript submission",
            )
        ]
    return [
        _row(
            "V24",
            "full_repo_evidence_audit",
            "passed",
            str(base),
            f"full-repo audit present with status={status}, blockers={blockers}; "
            "future work and documented limitations are allowed",
        )
    ]


def _check_pre_manuscript_final_gate(input_root: Path) -> list[dict[str, Any]]:
    """V25: final gate exists and reports no blockers.

    Missing final-gate files are warnings for older fixtures. Once the gate exists, blockers
    are validation failures; limitation notes, diagnostic-only rows, and future-work rows are
    allowed.
    """

    base = input_root / "pre_manuscript_final_gate"
    if not base.is_dir():
        return [
            _row(
                "V25",
                "pre_manuscript_final_gate",
                "warning",
                "pre_manuscript_final_gate",
                "pre-manuscript final gate not present",
                "run scripts/run_pre_manuscript_final_gate.py and rebuild the package",
            )
        ]
    missing = [name for name in _PRE_MANUSCRIPT_FINAL_GATE_REQUIRED if not (base / name).is_file()]
    if missing:
        return [
            _row(
                "V25",
                "pre_manuscript_final_gate",
                "warning",
                str(base),
                "pre-manuscript final gate files missing: " + ", ".join(missing[:10]),
                "rerun scripts/run_pre_manuscript_final_gate.py",
            )
        ]
    decision = read_csv(base / "manuscript_readiness_decision.csv")
    if decision.empty or "decision_item" not in decision.columns:
        return [
            _row(
                "V25",
                "pre_manuscript_final_gate",
                "warning",
                str(base / "manuscript_readiness_decision.csv"),
                "decision file missing or malformed",
                "rerun scripts/run_pre_manuscript_final_gate.py",
            )
        ]
    row = decision[decision["decision_item"].astype(str) == "manuscript_readiness"]
    if row.empty:
        return [
            _row(
                "V25",
                "pre_manuscript_final_gate",
                "warning",
                str(base / "manuscript_readiness_decision.csv"),
                "decision file has no manuscript_readiness row",
                "rerun scripts/run_pre_manuscript_final_gate.py",
            )
        ]
    status = str(row.iloc[0].get("value", ""))
    try:
        blockers = int(row.iloc[0].get("blocker_count", 0))
    except (TypeError, ValueError):
        blockers = 0
    if blockers > 0 or status == "blocked":
        return [
            _row(
                "V25",
                "pre_manuscript_final_gate",
                "failed",
                str(base / "manuscript_readiness_decision.csv"),
                f"pre-manuscript final gate reports status={status}, blockers={blockers}",
                "fix final-gate blockers before manuscript writing",
            )
        ]
    return [
        _row(
            "V25",
            "pre_manuscript_final_gate",
            "passed",
            str(base),
            f"pre-manuscript final gate present with status={status}, blockers={blockers}; "
            "limitation notes, diagnostic-only rows, and future-work notes are allowed",
        )
    ]


def _pre_manuscript_failures(base: Path) -> list[str]:
    failures: list[str] = []
    scorecard = read_csv(base / "pre_manuscript_usability_scorecard.csv")
    if not scorecard.empty and {"category", "blocker_count", "status"}.issubset(scorecard.columns):
        overall = scorecard[scorecard["category"].astype(str) == "overall"]
        if not overall.empty:
            row = overall.iloc[0]
            try:
                blockers = int(row.get("blocker_count", 0))
            except (TypeError, ValueError):
                blockers = 0
            status = str(row.get("status", ""))
            if blockers > 0 or status in {"blocked", "needs_fix_before_manuscript"}:
                failures.append(
                    f"pre-manuscript scorecard reports status={status}, blockers={blockers}"
                )
    warnings = read_csv(base / "warning_classification.csv")
    if "is_blocker" in warnings.columns:
        blocker = warnings["is_blocker"].astype(str).str.lower().isin({"true", "1", "yes"})
        if bool(blocker.any()):
            failures.append("warning classification contains blocker rows")
    numeric = read_csv(base / "numeric_sanity_audit.csv")
    if "is_blocker" in numeric.columns:
        blocker = numeric["is_blocker"].astype(str).str.lower().isin({"true", "1", "yes"})
        if bool(blocker.any()):
            failures.append("numeric sanity audit contains blocker rows")
    tables = read_csv(base / "table_usability_audit.csv")
    if "usability_status" in tables.columns:
        bad = tables["usability_status"].astype(str).isin({"needs_fix", "do_not_use"})
        if bool(bad.any()):
            failures.append("main table usability audit contains needs_fix/do_not_use rows")
    return failures


def _summary_markdown(rows: list[dict[str, Any]], counts: dict[str, int]) -> str:
    failures = [r for r in rows if r["status"] == "failed"]
    warnings = [r for r in rows if r["status"] == "warning"]
    lines = [
        "# Final Artifact Validation",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "## Result",
        f"- passed: {counts.get('passed', 0)}",
        f"- warning: {counts.get('warning', 0)}",
        f"- failed: {counts.get('failed', 0)}",
        f"- not_applicable: {counts.get('not_applicable', 0)}",
        f"- overall: {'PASS' if not failures else 'FAIL'}",
        "",
        "## Failures",
        *(
            [f"- [{r['check_id']}] {r['check_name']}: {r['details']}" for r in failures]
            or ["- none"]
        ),
        "",
        "## Warnings",
        *(
            [f"- [{r['check_id']}] {r['check_name']}: {r['details']}" for r in warnings]
            or ["- none"]
        ),
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *, output_dir: Path, rows: list[dict[str, Any]], input_config: dict[str, Any]
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report_path = rows_to_table(rows, output_dir / "artifact_validation_report.csv", REPORT_COLUMNS)
    summary_path = output_dir / "artifact_validation_summary.md"
    summary_path.write_text(_summary_markdown(rows, counts), encoding="utf-8")

    artifacts = {
        "artifact_validation_report": str(report_path),
        "artifact_validation_summary": str(summary_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "rows": rows,
        "counts": counts,
        "passed": counts.get("passed", 0),
        "warnings": counts.get("warning", 0),
        "failures": counts.get("failed", 0),
        "ok": counts.get("failed", 0) == 0,
        "artifacts": artifacts,
    }
