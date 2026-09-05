"""Full-repository evidence-integrity and reproducibility audit.

This module audits the manuscript evidence package without promoting any claim.
It records missing, diagnostic-only, runtime-limited, and future-work artifacts as
such. The downstream artifact validator is expected to fail only when this audit
reports real blockers, not when it reports documented limitations.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.paper.test_quality_audit import SCIENTIFIC_CATEGORIES, _inventory_rows
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_full_repo_evidence_audit.py"

ENV_COLUMNS = ["item", "value", "status", "notes"]

INVENTORY_COLUMNS = [
    "experiment_family",
    "expected_source_files",
    "expected_scripts",
    "expected_output_dirs",
    "actual_source_files_found",
    "actual_scripts_found",
    "actual_output_dirs_found",
    "primary_result_files",
    "row_counts",
    "status",
    "evidence_strength",
    "missing_or_future_reason",
    "notes",
]

PROVENANCE_COLUMNS = [
    "output_artifact",
    "artifact_group",
    "source_script",
    "source_module",
    "source_input_artifacts",
    "generated_or_manual",
    "row_count",
    "has_source_artifact_column",
    "has_manifest_entry",
    "mirrored_from",
    "is_latest",
    "is_superseded",
    "status",
    "notes",
]

RERUN_COLUMNS = [
    "command_name",
    "command",
    "ran",
    "exit_code",
    "runtime_seconds",
    "expected_outputs",
    "outputs_exist",
    "row_counts_before",
    "row_counts_after",
    "changed_after_rerun",
    "allowed_change",
    "status",
    "notes",
]

SUSPICIOUS_COLUMNS = [
    "artifact_path",
    "row_index",
    "suspicion_type",
    "severity",
    "is_blocker",
    "evidence",
    "recommended_action",
    "notes",
]

TEST_COLUMNS = [
    "test_file",
    "test_name",
    "classification",
    "evidence_family",
    "checks_numeric_behavior",
    "checks_claim_boundary",
    "checks_artifact_schema_only",
    "is_smoke_only",
    "supports_scientific_claim",
    "notes",
]

CLAIM_COLUMNS = [
    "claim_id",
    "claim_text",
    "current_status",
    "expected_status",
    "status_matches_expected",
    "supporting_artifacts",
    "contradicting_artifacts",
    "risk_level",
    "recommended_action",
    "notes",
]

COMPLETION_COLUMNS = [
    "experiment_family",
    "planned",
    "implemented",
    "outputs_exist",
    "tests_exist",
    "tests_are_real_validation",
    "rerunnable",
    "used_in_main_tables",
    "used_in_appendix",
    "diagnostic_only",
    "future_work",
    "limitations",
    "completion_status",
    "notes",
]

READINESS_COLUMNS = [
    "artifact_path",
    "artifact_group",
    "recommended_use",
    "paper_section_hint",
    "status",
    "has_blocker",
    "readiness_status",
    "source_artifact",
    "safe_claim_text",
    "required_limitation_text",
    "do_not_claim_text",
    "notes",
]

SCORECARD_COLUMNS = [
    "category",
    "status",
    "blocker_count",
    "needs_review_count",
    "warning_count",
    "complete_count",
    "partial_count",
    "future_work_count",
    "recommended_action",
    "notes",
]

_TEXT_SUFFIXES = {".csv", ".md", ".json", ".txt", ".tex"}
_KEY_PROVENANCE_DIRS = (
    "outputs/final_manuscript_package",
    "outputs/jacobian_validation",
    "outputs/measurement_row_metadata_audit",
    "outputs/full_vector_readout",
    "outputs/phase_synthesis_refinement",
    "outputs/pre_manuscript_usability_audit",
)


@dataclass(frozen=True)
class ExperimentFamily:
    name: str
    sources: tuple[str, ...]
    scripts: tuple[str, ...]
    output_dirs: tuple[str, ...]
    primary_files: tuple[str, ...]
    evidence_strength: str
    notes: str = ""
    future_reason: str = ""
    status_if_present: str = "present_and_reproducible"


@dataclass(frozen=True)
class RerunCommand:
    name: str
    command: str
    expected_outputs: tuple[str, ...]
    run_by_default: bool = True
    not_run_reason: str = ""


EXPERIMENT_FAMILIES: tuple[ExperimentFamily, ...] = (
    ExperimentFamily(
        "measurement_inventory",
        ("src/robust_qsvt_se/measurement/*.py", "src/robust_qsvt_se/data/*.py"),
        ("scripts/build_paper_measurement_inventory.py", "scripts/export_measurement_inventory.py"),
        (
            "outputs/measurement_inventory",
            "outputs/final_manuscript_package/phase1_measurement_inventory",
        ),
        (
            "outputs/final_manuscript_package/phase1_measurement_inventory/"
            "paper_table_measurement_inventory.csv",
        ),
        "integration_regression",
        "IEEE/PYPOWER measurement rows are generated from model code, not field data.",
    ),
    ExperimentFamily(
        "jacobian_validation",
        ("src/robust_qsvt_se/paper/jacobian_validation.py",),
        ("scripts/run_jacobian_validation.py",),
        ("outputs/jacobian_validation", "outputs/final_manuscript_package/jacobian_validation"),
        ("outputs/jacobian_validation/ac_jacobian_validation_summary.csv",),
        "real_validation",
        "Finite-difference validation of AC/DC Jacobian implementations.",
    ),
    ExperimentFamily(
        "measurement_row_metadata_audit",
        ("src/robust_qsvt_se/paper/measurement_row_metadata_audit.py",),
        ("scripts/run_measurement_row_metadata_audit.py",),
        (
            "outputs/measurement_row_metadata_audit",
            "outputs/final_manuscript_package/measurement_row_metadata_audit",
        ),
        ("outputs/measurement_row_metadata_audit/row_mask_consistency_checks.csv",),
        "real_validation",
        "Audits row metadata and implemented row-mask views.",
    ),
    ExperimentFamily(
        "weighted_jacobian_consistency",
        ("src/robust_qsvt_se/paper/jacobian_validation.py",),
        ("scripts/run_jacobian_validation.py",),
        ("outputs/jacobian_validation", "outputs/final_manuscript_package/jacobian_validation"),
        ("outputs/jacobian_validation/weighted_jacobian_consistency_audit.csv",),
        "real_validation",
        "Checks weighted rows and variance consistency.",
    ),
    ExperimentFamily(
        "classical_spectral_filtering",
        (
            "src/robust_qsvt_se/estimators/*.py",
            "src/robust_qsvt_se/qsvt/filters.py",
            "src/robust_qsvt_se/paper/classical_spectral_filtering_audit.py",
        ),
        (
            "scripts/build_classical_spectral_filtering_audit.py",
            "scripts/recheck_classical_spectral_filtering_audit.py",
        ),
        (
            "outputs/final_manuscript_package/phase2_classical_spectral_filtering",
            "outputs/final_manuscript_package/phase2_classical_recheck",
        ),
        (
            "outputs/final_manuscript_package/phase2_classical_spectral_filtering/"
            "paper_table_classical_main_results.csv",
        ),
        "real_validation",
        "Classical baselines and matched-alpha QSVT target.",
    ),
    ExperimentFamily(
        "full_alpha_sweep_classical",
        ("src/robust_qsvt_se/paper/full_alpha_sweep_classical.py",),
        ("scripts/run_full_alpha_sweep_classical.py",),
        (
            "outputs/full_alpha_sensitivity_classical",
            "outputs/final_manuscript_package/phase3_full_classical_alpha_sweep",
        ),
        (
            "outputs/final_manuscript_package/phase3_full_classical_alpha_sweep/"
            "alpha_sweep_summary_by_case.csv",
        ),
        "real_validation",
        "Full per-case classical alpha sweep; best alpha remains diagnostic-only.",
    ),
    ExperimentFamily(
        "reactive_power_conditioning",
        ("src/robust_qsvt_se/paper/reactive_power_conditioning_table.py",),
        ("scripts/build_reactive_power_conditioning_table.py",),
        (
            "outputs/final_manuscript_package/reactive_power_conditioning",
            "outputs/measurement_redundancy",
        ),
        (
            "outputs/final_manuscript_package/reactive_power_conditioning/"
            "reactive_power_conditioning_table.csv",
        ),
        "real_validation",
        "Reactive P/Q conditioning presentation from controlled benchmark artifacts.",
    ),
    ExperimentFamily(
        "measurement_type_ablation",
        ("src/robust_qsvt_se/paper/measurement_type_ablation.py",),
        ("scripts/run_measurement_type_ablation.py",),
        (
            "outputs/measurement_type_ablation",
            "outputs/final_manuscript_package/phase5_measurement_type_ablation",
        ),
        (
            "outputs/final_manuscript_package/phase5_measurement_type_ablation/"
            "condition_by_measurement_subset.csv",
        ),
        "real_validation",
        "Per-measurement-type drop ablation; field-calibrated missingness is future work.",
    ),
    ExperimentFamily(
        "structured_stress",
        ("src/robust_qsvt_se/paper/structured_stress_ablation_consolidation.py",),
        ("scripts/build_structured_stress_ablation_consolidation.py",),
        ("outputs/final_manuscript_package/phase5_structured_stress_ablation",),
        (
            "outputs/final_manuscript_package/phase5_structured_stress_ablation/"
            "paper_table_structured_stress_ablation.csv",
        ),
        "integration_regression",
        "Structured stress is controlled benchmark stress, not field-calibrated statistics.",
    ),
    ExperimentFamily(
        "compound_structured_stress",
        ("src/robust_qsvt_se/paper/compound_structured_stress.py",),
        ("scripts/run_compound_structured_stress.py",),
        (
            "outputs/structured_compound_stress",
            "outputs/final_manuscript_package/phase5_compound_structured_stress",
        ),
        (
            "outputs/final_manuscript_package/phase5_compound_structured_stress/"
            "compound_stress_summary.csv",
        ),
        "real_validation",
        "Compound/weak-area/spatial stress remains controlled benchmark evidence.",
    ),
    ExperimentFamily(
        "nonlinear_ac_consolidation",
        ("src/robust_qsvt_se/paper/nonlinear_ac_consolidation.py",),
        ("scripts/build_nonlinear_ac_consolidation.py",),
        ("outputs/final_manuscript_package/phase4_nonlinear_ac",),
        (
            "outputs/final_manuscript_package/phase4_nonlinear_ac/"
            "paper_table_nonlinear_ac_convergence.csv",
        ),
        "integration_regression",
        "Classical nonlinear AC consistency only; not QSVT in the nonlinear loop.",
    ),
    ExperimentFamily(
        "nonlinear_trajectory_extraction",
        ("src/robust_qsvt_se/paper/nonlinear_trajectory_extraction.py",),
        ("scripts/extract_nonlinear_ac_trajectories.py",),
        ("outputs/final_manuscript_package/nonlinear_trajectories",),
        (
            "outputs/final_manuscript_package/nonlinear_trajectories/"
            "figure_data_nonlinear_residual_vs_iteration.csv",
        ),
        "diagnostic",
        "Trajectory extraction is supporting/diagnostic for nonlinear behavior.",
    ),
    ExperimentFamily(
        "initializer_ablation",
        ("src/robust_qsvt_se/paper/initializer_ablation.py",),
        ("scripts/run_initializer_ablation.py",),
        (
            "outputs/nonlinear_initializer_ablation",
            "outputs/final_manuscript_package/initializer_ablation",
        ),
        ("outputs/final_manuscript_package/initializer_ablation/initializer_ablation_summary.csv",),
        "diagnostic",
        "Initializer ablation is supporting evidence and may be partial.",
        status_if_present="diagnostic_only",
    ),
    ExperimentFamily(
        "ieee300_runtime_extension",
        ("src/robust_qsvt_se/paper/ieee300_runtime_extension.py",),
        ("scripts/run_ieee300_runtime_extension.py",),
        (
            "outputs/ieee300_runtime_extension",
            "outputs/final_manuscript_package/ieee300_runtime_extension",
        ),
        (
            "outputs/final_manuscript_package/ieee300_runtime_extension/ieee300_runtime_limited_rows.csv",
        ),
        "diagnostic",
        "Large-case IEEE300 results are runtime-limited reduced diagnostics.",
        status_if_present="runtime_limited",
    ),
    ExperimentFamily(
        "qsvt_target_phase_validation",
        ("src/robust_qsvt_se/qsvt/*.py", "src/robust_qsvt_se/paper/qsvt_resource_phase_summary.py"),
        (
            "scripts/run_qsvt_selected_alpha_phase_validation.py",
            "scripts/build_qsvt_resource_phase_summary.py",
        ),
        (
            "outputs/qsvt_phase_validation_paper",
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase",
        ),
        (
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase/"
            "paper_table_qsvt_phase_error.csv",
        ),
        "real_validation",
        "Selected target/phase validation, not full hardware execution.",
    ),
    ExperimentFamily(
        "qsvt_degree_window",
        ("src/robust_qsvt_se/qsvt/*.py",),
        (
            "scripts/run_qsvt_degree_window_overshoot.py",
            "scripts/run_qsvt_degree_window_gate_validation.py",
        ),
        (
            "outputs/qsvt_degree_window_overshoot",
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase",
        ),
        (
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase/"
            "paper_table_qsvt_target_and_degree_window.csv",
        ),
        "real_validation",
        "Degree-window/overshoot diagnostics for selected subproblems.",
    ),
    ExperimentFamily(
        "qsvt_gate_validation",
        ("src/robust_qsvt_se/qsvt/*.py",),
        (
            "scripts/run_qsvt_degree_window_gate_validation.py",
            "scripts/run_qsvt_gate_validation_with_amplitude_recovery.py",
        ),
        (
            "outputs/qsvt_degree_window_gate_validation",
            "outputs/qsvt_gate_validation_with_amplitude_recovery",
        ),
        (
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase/paper_table_qsvt_gate_validation.csv",
        ),
        "real_validation",
        "Gate validation on selected small subproblems only.",
    ),
    ExperimentFamily(
        "qsvt_cross_case_gate_validation",
        ("src/robust_qsvt_se/qsvt/*.py",),
        ("scripts/run_qsvt_cross_case_gate_validation.py",),
        ("outputs/qsvt_cross_case_gate_validation",),
        (
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase/paper_table_qsvt_gate_validation.csv",
        ),
        "real_validation",
        "Cross-case selected-subproblem gate validation.",
    ),
    ExperimentFamily(
        "qsvt_ieee118_extension",
        ("src/robust_qsvt_se/qsvt/*.py",),
        (
            "scripts/run_qsvt_ieee118_extension_audit.py",
            "scripts/run_qsvt_ieee118_gate_validation.py",
        ),
        ("outputs/qsvt_ieee118_extension_audit", "outputs/qsvt_ieee118_gate_validation"),
        (
            "outputs/final_manuscript_package/phase6_qsvt_resource_phase/paper_table_qsvt_gate_validation.csv",
        ),
        "real_validation",
        "IEEE118 selected-block extension; not full IEEE118 QSVT execution.",
    ),
    ExperimentFamily(
        "full_vector_readout",
        ("src/robust_qsvt_se/paper/full_vector_readout.py",),
        ("scripts/run_full_vector_readout_demo.py",),
        ("outputs/full_vector_readout", "outputs/final_manuscript_package/full_vector_readout"),
        (
            "outputs/final_manuscript_package/full_vector_readout/full_vector_readout_overall_summary.csv",
        ),
        "real_validation",
        (
            "Selected-subproblem full-vector readout only; full IEEE-scale readout remains "
            "future work."
        ),
    ),
    ExperimentFamily(
        "readout_alpha_degree_codesign",
        ("src/robust_qsvt_se/paper/readout_alpha_degree_codesign.py",),
        ("scripts/run_readout_alpha_degree_sweep.py",),
        ("outputs/full_vector_readout", "outputs/final_manuscript_package/full_vector_readout"),
        ("outputs/final_manuscript_package/full_vector_readout/readout_two_view_summary.csv",),
        "diagnostic",
        "Co-design is diagnostic/limited and does not solve full-scale readout.",
        status_if_present="diagnostic_only",
    ),
    ExperimentFamily(
        "phase_synthesis_refinement",
        ("src/robust_qsvt_se/paper/phase_synthesis_refinement.py",),
        ("scripts/run_phase_synthesis_refinement.py",),
        (
            "outputs/phase_synthesis_refinement",
            "outputs/final_manuscript_package/phase_synthesis_refinement",
        ),
        (
            "outputs/final_manuscript_package/phase_synthesis_refinement/three_view_readout_summary.csv",
        ),
        "diagnostic",
        "Finite-degree/phase-synthesis refinement is diagnostic and selected-subproblem scoped.",
        status_if_present="diagnostic_only",
    ),
    ExperimentFamily(
        "statistical_aggregation",
        ("src/robust_qsvt_se/paper/final_statistical_aggregation.py",),
        ("scripts/build_final_statistical_aggregation.py",),
        ("outputs/final_manuscript_package/statistical_summary",),
        (
            "outputs/final_manuscript_package/statistical_summary/statistical_aggregation_manifest.csv",
        ),
        "artifact_schema",
        "Aggregates existing artifacts; does not create new experimental evidence.",
    ),
    ExperimentFamily(
        "main_paper_tables",
        ("src/robust_qsvt_se/paper/main_paper_tables.py",),
        ("scripts/build_main_paper_tables.py",),
        ("outputs/final_manuscript_package/main_tables",),
        ("outputs/final_manuscript_package/main_tables/main_tables_manifest.csv",),
        "artifact_schema",
        "Canonical table builder from source artifacts.",
    ),
    ExperimentFamily(
        "claim_boundaries",
        (
            "src/robust_qsvt_se/paper/claim_boundary_writer.py",
            "src/robust_qsvt_se/paper/claim_lint.py",
        ),
        ("scripts/build_claim_boundary_docs.py", "scripts/lint_claim_boundaries.py"),
        (
            "outputs/final_manuscript_package/claim_boundaries",
            "outputs/final_manuscript_package/claim_lint",
        ),
        ("outputs/final_manuscript_package/claim_boundaries/DO_NOT_CLAIM.md",),
        "claim_boundary_only",
        "Claim-boundary documentation; no new numerical validation.",
    ),
    ExperimentFamily(
        "pre_manuscript_usability_audit",
        ("src/robust_qsvt_se/paper/pre_manuscript_usability_audit.py",),
        ("scripts/run_pre_manuscript_usability_audit.py",),
        (
            "outputs/pre_manuscript_usability_audit",
            "outputs/final_manuscript_package/pre_manuscript_usability_audit",
        ),
        (
            "outputs/final_manuscript_package/pre_manuscript_usability_audit/"
            "paper_use_decision_manifest.csv",
        ),
        "artifact_schema",
        "Paper-use/readiness classification for artifacts.",
    ),
    ExperimentFamily(
        "final_manuscript_package",
        ("src/robust_qsvt_se/paper/final_manuscript_package.py",),
        ("scripts/build_final_manuscript_package.py",),
        ("outputs/final_manuscript_package",),
        ("outputs/final_manuscript_package/manuscript_artifact_index.csv",),
        "artifact_schema",
        "Final package assembly/indexing.",
    ),
    ExperimentFamily(
        "claim_lint",
        ("src/robust_qsvt_se/paper/claim_lint.py",),
        ("scripts/lint_claim_boundaries.py",),
        ("outputs/final_manuscript_package/claim_lint",),
        ("outputs/final_manuscript_package/claim_lint/claim_lint_report.csv",),
        "claim_boundary_only",
        "Claim-boundary scan; high-risk positive overclaims are blockers.",
    ),
    ExperimentFamily(
        "artifact_validator",
        ("src/robust_qsvt_se/paper/final_artifact_validator.py",),
        ("scripts/validate_final_manuscript_artifacts.py",),
        ("outputs/final_manuscript_package/final_artifact_validation",),
        (
            "outputs/final_manuscript_package/final_artifact_validation/artifact_validation_report.csv",
        ),
        "artifact_schema",
        "Package integrity validator.",
    ),
    ExperimentFamily(
        "pre_submission_check",
        (),
        ("scripts/pre_submission_check.py",),
        ("outputs/final_manuscript_package/pre_submission_check",),
        ("outputs/final_manuscript_package/pre_submission_check/pre_submission_check_report.json",),
        "integration_regression",
        "One-command readiness check; heavy optional jobs are off by default.",
    ),
)


RERUN_COMMANDS: tuple[RerunCommand, ...] = (
    RerunCommand(
        "jacobian_validation",
        ".venv/bin/python scripts/run_jacobian_validation.py --input-root outputs "
        "--output-dir outputs/jacobian_validation --cases ieee14 ieee30 ieee57 ieee118 "
        "--epsilon 1e-6",
        (
            "outputs/jacobian_validation/ac_jacobian_validation_summary.csv",
            "outputs/jacobian_validation/dc_jacobian_validation_summary.csv",
            "outputs/jacobian_validation/weighted_jacobian_consistency_audit.csv",
        ),
    ),
    RerunCommand(
        "measurement_row_metadata_audit",
        ".venv/bin/python scripts/run_measurement_row_metadata_audit.py --input-root outputs "
        "--output-dir outputs/measurement_row_metadata_audit --cases ieee14 ieee30 ieee57 ieee118",
        (
            "outputs/measurement_row_metadata_audit/row_metadata_audit.csv",
            "outputs/measurement_row_metadata_audit/row_mask_consistency_checks.csv",
        ),
    ),
    RerunCommand(
        "final_statistical_aggregation",
        ".venv/bin/python scripts/build_final_statistical_aggregation.py --input-root outputs "
        "--package-root outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package/statistical_summary",
        (
            "outputs/final_manuscript_package/statistical_summary/statistical_aggregation_manifest.csv",
        ),
    ),
    RerunCommand(
        "main_paper_tables",
        ".venv/bin/python scripts/build_main_paper_tables.py --input-root outputs "
        "--package-root outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package/main_tables",
        ("outputs/final_manuscript_package/main_tables/main_tables_manifest.csv",),
    ),
    RerunCommand(
        "pre_manuscript_usability_audit",
        ".venv/bin/python scripts/run_pre_manuscript_usability_audit.py --input-root outputs "
        "--package-root outputs/final_manuscript_package --output-dir "
        "outputs/pre_manuscript_usability_audit",
        ("outputs/pre_manuscript_usability_audit/pre_manuscript_usability_scorecard.csv",),
    ),
    RerunCommand(
        "final_manuscript_package",
        ".venv/bin/python scripts/build_final_manuscript_package.py --input-root outputs "
        "--phase-root outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package",
        ("outputs/final_manuscript_package/manuscript_artifact_index.csv",),
    ),
    RerunCommand(
        "claim_lint",
        ".venv/bin/python scripts/lint_claim_boundaries.py --input-root "
        "outputs/final_manuscript_package --output-dir outputs/final_manuscript_package/claim_lint",
        ("outputs/final_manuscript_package/claim_lint/claim_lint_report.csv",),
    ),
    RerunCommand(
        "artifact_validator",
        ".venv/bin/python scripts/validate_final_manuscript_artifacts.py --input-root "
        "outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package/final_artifact_validation",
        (
            "outputs/final_manuscript_package/final_artifact_validation/artifact_validation_report.csv",
        ),
    ),
    RerunCommand(
        "canonical_paper_numbers",
        ".venv/bin/python scripts/check_paper_numbers.py --input-root "
        "outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package/canonical_numbers",
        ("outputs/final_manuscript_package/canonical_numbers/canonical_paper_numbers.json",),
    ),
    RerunCommand(
        "pre_submission_check",
        ".venv/bin/python scripts/pre_submission_check.py --raw-output-root outputs "
        "--package-root outputs/final_manuscript_package --output-dir "
        "outputs/final_manuscript_package/pre_submission_check",
        ("outputs/final_manuscript_package/pre_submission_check/pre_submission_check_report.json",),
        run_by_default=False,
        not_run_reason=(
            "runtime-limited inside the audit to avoid recursively running the full "
            "pre-submission check; run explicitly after package rebuild"
        ),
    ),
)


def build_full_repo_evidence_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Build the full-repository evidence audit outputs."""

    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = ensure_directory(config.get("output_dir", input_root / "full_repo_evidence_audit"))
    repo_root = Path(config.get("repo_root", ".")).resolve()
    run_reruns = bool(config.get("run_reruns", True))
    rerun_timeout = int(config.get("rerun_timeout_seconds", 900))

    env_rows = _environment_snapshot(repo_root)
    inventory_rows = _planned_experiment_inventory(repo_root)
    provenance_rows = _source_to_output_provenance(repo_root, output_dir)
    rerun_rows = _reproducibility_reruns(
        repo_root=repo_root,
        output_dir=output_dir,
        run_reruns=run_reruns,
        timeout_seconds=rerun_timeout,
    )
    suspicious_rows = _suspicious_result_audit(repo_root, output_dir)
    test_rows = _test_quality_rows(repo_root)
    claim_rows, unsupported_rows = _claim_boundary_rows(package_root)
    completion_rows = _completion_rows(
        inventory_rows=inventory_rows,
        rerun_rows=rerun_rows,
        test_rows=test_rows,
        package_root=package_root,
    )
    readiness_rows = _artifact_use_readiness_rows(package_root)
    scorecard_rows = _scorecard_rows(
        env_rows=env_rows,
        inventory_rows=inventory_rows,
        provenance_rows=provenance_rows,
        rerun_rows=rerun_rows,
        suspicious_rows=suspicious_rows,
        test_rows=test_rows,
        claim_rows=claim_rows,
        completion_rows=completion_rows,
        readiness_rows=readiness_rows,
    )

    artifacts = _write_audit_outputs(
        output_dir=output_dir,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
            "run_reruns": run_reruns,
            "rerun_timeout_seconds": rerun_timeout,
        },
        env_rows=env_rows,
        inventory_rows=inventory_rows,
        provenance_rows=provenance_rows,
        rerun_rows=rerun_rows,
        suspicious_rows=suspicious_rows,
        test_rows=test_rows,
        claim_rows=claim_rows,
        unsupported_rows=unsupported_rows,
        completion_rows=completion_rows,
        readiness_rows=readiness_rows,
        scorecard_rows=scorecard_rows,
    )
    _mirror_audit(output_dir, package_root / "full_repo_evidence_audit")

    overall = _overall_score(scorecard_rows)
    return {
        "output_dir": output_dir,
        "package_mirror_dir": package_root / "full_repo_evidence_audit",
        "artifacts": artifacts,
        "overall_status": overall.get("status", "unknown"),
        "blocker_count": int(overall.get("blocker_count", 0)),
        "needs_review_count": int(overall.get("needs_review_count", 0)),
        "warning_count": int(overall.get("warning_count", 0)),
        "rows": {
            "environment": env_rows,
            "inventory": inventory_rows,
            "provenance": provenance_rows,
            "reruns": rerun_rows,
            "suspicious": suspicious_rows,
            "tests": test_rows,
            "claims": claim_rows,
            "completion": completion_rows,
            "readiness": readiness_rows,
            "scorecard": scorecard_rows,
        },
    }


def _environment_snapshot(repo_root: Path) -> list[dict[str, Any]]:
    commands = (
        ("pwd", "pwd"),
        ("python_version", "python --version || true"),
        ("venv_python_version", ".venv/bin/python --version || true"),
        ("git_status_short", "git status --short || true"),
        ("git_rev_parse_head", "git rev-parse HEAD || true"),
        (
            "artifact_count",
            "find src scripts tests outputs/final_manuscript_package -maxdepth 2 -type f "
            "| sort | wc -l",
        ),
    )
    rows: list[dict[str, Any]] = []
    for item, command in commands:
        result = _run_shell(command, repo_root, timeout_seconds=60)
        value = _clean_output(result.stdout or result.stderr)
        status = "ok" if result.returncode == 0 else "failed"
        notes = f"command: {command}"
        if item.startswith("git_") and "not a git repository" in value.lower():
            value = "git_hash_unavailable" if item == "git_rev_parse_head" else value
            status = "unavailable"
            notes = "not_a_git_repository_from_tool_view"
        rows.append({"item": item, "value": value, "status": status, "notes": notes})
    return rows


def _planned_experiment_inventory(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in EXPERIMENT_FAMILIES:
        actual_sources = _existing_globs(repo_root, family.sources)
        actual_scripts = _existing_paths(repo_root, family.scripts)
        actual_dirs = _existing_paths(repo_root, family.output_dirs)
        primary_files = _existing_paths(repo_root, family.primary_files)
        row_counts = _row_count_summary(repo_root, family.primary_files)
        status = _inventory_status(family, actual_scripts, actual_dirs, primary_files)
        reason = "" if actual_dirs else family.future_reason or "no output directory found"
        rows.append(
            {
                "experiment_family": family.name,
                "expected_source_files": ";".join(family.sources),
                "expected_scripts": ";".join(family.scripts),
                "expected_output_dirs": ";".join(family.output_dirs),
                "actual_source_files_found": ";".join(actual_sources),
                "actual_scripts_found": ";".join(actual_scripts),
                "actual_output_dirs_found": ";".join(actual_dirs),
                "primary_result_files": ";".join(primary_files),
                "row_counts": row_counts,
                "status": status,
                "evidence_strength": family.evidence_strength,
                "missing_or_future_reason": reason,
                "notes": family.notes,
            }
        )
    return rows


def _inventory_status(
    family: ExperimentFamily,
    scripts: list[str],
    dirs: list[str],
    primary_files: list[str],
) -> str:
    if not dirs:
        return "future_work" if family.future_reason else "missing"
    if family.status_if_present in {"runtime_limited", "diagnostic_only"} and primary_files:
        return family.status_if_present
    if primary_files:
        return family.status_if_present
    if scripts:
        return "present_but_schema_only"
    return "present_but_not_rerun"


def _source_to_output_provenance(repo_root: Path, audit_output_dir: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for rel in _KEY_PROVENANCE_DIRS:
        base = repo_root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if audit_output_dir.resolve() in path.resolve().parents:
                continue
            if "full_repo_evidence_audit" in path.parts:
                continue
            paths.append(path)

    rows: list[dict[str, Any]] = []
    for path in paths:
        group = _artifact_group(repo_root, path)
        source_script = _source_script_for_group(group)
        source_module = _source_module_for_group(group)
        row_count = _artifact_row_count(path)
        source_inputs = _source_inputs_for_artifact(path)
        has_source_col = _has_source_column(path)
        manifest_entry = _has_manifest_entry(path)
        mirrored_from = _mirrored_from(repo_root, path, group)
        generated = _generated_or_manual(path)
        status = _provenance_status(
            path=path,
            source_script=source_script,
            manifest_entry=manifest_entry,
            mirrored_from=mirrored_from,
            generated=generated,
        )
        rows.append(
            {
                "output_artifact": str(path),
                "artifact_group": group,
                "source_script": source_script,
                "source_module": source_module,
                "source_input_artifacts": source_inputs,
                "generated_or_manual": generated,
                "row_count": row_count,
                "has_source_artifact_column": _yes_no(has_source_col),
                "has_manifest_entry": _yes_no(manifest_entry),
                "mirrored_from": mirrored_from,
                "is_latest": _yes_no(_is_latest(path, mirrored_from)),
                "is_superseded": _yes_no("historical" in path.parts or "superseded" in path.name),
                "status": status,
                "notes": _provenance_notes(path, status),
            }
        )
    return rows


def _reproducibility_reruns(
    *, repo_root: Path, output_dir: Path, run_reruns: bool, timeout_seconds: int
) -> list[dict[str, Any]]:
    previous = output_dir / "reproducibility_rerun_audit.csv"
    if not run_reruns and previous.is_file():
        frame = read_csv(previous)
        if not frame.empty:
            return frame.to_dict(orient="records")

    rows: list[dict[str, Any]] = []
    for spec in RERUN_COMMANDS:
        expected = [repo_root / rel for rel in spec.expected_outputs]
        before_counts = _row_count_mapping(expected)
        before_hashes = _hash_mapping(expected)
        scripts = _scripts_in_command(spec.command)
        missing_scripts = [script for script in scripts if not (repo_root / script).is_file()]
        should_run = run_reruns and spec.run_by_default and not missing_scripts
        if not should_run:
            status = "not_run_missing_script" if missing_scripts else "not_run_runtime_limited"
            reason = (
                f"missing scripts: {', '.join(missing_scripts)}"
                if missing_scripts
                else spec.not_run_reason or "reruns disabled by config"
            )
            after_counts = _row_count_mapping(expected)
            rows.append(
                {
                    "command_name": spec.name,
                    "command": spec.command,
                    "ran": "no",
                    "exit_code": "",
                    "runtime_seconds": "0.000",
                    "expected_outputs": ";".join(spec.expected_outputs),
                    "outputs_exist": _yes_no(all(path.exists() for path in expected)),
                    "row_counts_before": json.dumps(before_counts, sort_keys=True),
                    "row_counts_after": json.dumps(after_counts, sort_keys=True),
                    "changed_after_rerun": "no",
                    "allowed_change": "yes",
                    "status": status,
                    "notes": reason,
                }
            )
            continue

        start = time.monotonic()
        result = _run_shell(spec.command, repo_root, timeout_seconds=timeout_seconds)
        runtime = time.monotonic() - start
        after_counts = _row_count_mapping(expected)
        after_hashes = _hash_mapping(expected)
        changed = before_hashes != after_hashes
        outputs_exist = all(path.exists() for path in expected)
        if result.timed_out:
            status = "not_run_runtime_limited"
            note = f"timed out after {timeout_seconds}s"
        elif result.returncode == 0 and outputs_exist:
            status = "pass_changed_allowed" if changed else "pass_reproducible"
            note = _trim(result.stdout or result.stderr, 300)
        elif result.returncode == 0:
            status = "needs_review"
            note = "command exited 0 but expected outputs are missing"
        else:
            status = "failed"
            note = _trim(result.stderr or result.stdout, 500)
        rows.append(
            {
                "command_name": spec.name,
                "command": spec.command,
                "ran": "yes",
                "exit_code": result.returncode if result.returncode is not None else "",
                "runtime_seconds": f"{runtime:.3f}",
                "expected_outputs": ";".join(spec.expected_outputs),
                "outputs_exist": _yes_no(outputs_exist),
                "row_counts_before": json.dumps(before_counts, sort_keys=True),
                "row_counts_after": json.dumps(after_counts, sort_keys=True),
                "changed_after_rerun": _yes_no(changed),
                "allowed_change": "yes",
                "status": status,
                "notes": note,
            }
        )
    return rows


def _suspicious_result_audit(repo_root: Path, audit_output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scan_roots = [repo_root / "outputs" / "final_manuscript_package"]
    for rel in (
        "outputs/jacobian_validation",
        "outputs/measurement_row_metadata_audit",
        "outputs/full_vector_readout",
        "outputs/phase_synthesis_refinement",
        "outputs/pre_manuscript_usability_audit",
    ):
        path = repo_root / rel
        if path.is_dir():
            scan_roots.append(path)

    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".md", ".txt", ".tex"}:
                continue
            if audit_output_dir.resolve() in path.resolve().parents:
                continue
            if "full_repo_evidence_audit" in path.parts:
                continue
            rows.extend(_suspicious_for_path(path))
    if not rows:
        rows.append(
            {
                "artifact_path": "",
                "row_index": "",
                "suspicion_type": "none_detected",
                "severity": "info",
                "is_blocker": "no",
                "evidence": "heuristic scan found no suspicious blockers",
                "recommended_action": "none",
                "notes": "absence of findings is not proof of absence",
            }
        )
    return rows


def _suspicious_for_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return _text_suspicious(path)
    frame = read_csv(path)
    rows: list[dict[str, Any]] = []
    if frame.empty and not _is_documented_empty(path):
        rows.append(
            _suspicious_row(
                path,
                "",
                "row_count_zero_without_missing_record",
                "warning",
                "CSV has a header but no data rows and no missing/unavailable marker nearby",
                "document why the table is header-only or regenerate it",
            )
        )
    if _is_scientific_result_path(path) and not _has_source_column(path):
        rows.append(
            _suspicious_row(
                path,
                "",
                "missing_source_artifact_column",
                "warning",
                "scientific/result-looking CSV has no source_artifact/source_script column",
                "ensure provenance is supplied by a manifest or add source columns",
            )
        )
    rows.extend(_constant_numeric_suspicions(path, frame))
    rows.extend(_pass_with_bad_metric_suspicions(path, frame))
    rows.extend(_duplicate_case_metric_suspicions(path, frame))
    rows.extend(_qsvt_over_ridge_suspicions(path, frame))
    return rows


def _suspicious_row(
    path: Path,
    row_index: Any,
    suspicion_type: str,
    severity: str,
    evidence: str,
    action: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "artifact_path": str(path),
        "row_index": row_index,
        "suspicion_type": suspicion_type,
        "severity": severity,
        "is_blocker": _yes_no(severity == "blocker"),
        "evidence": evidence,
        "recommended_action": action,
        "notes": notes,
    }


def _constant_numeric_suspicions(path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(frame) < 4:
        return rows
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 4:
            continue
        unique = set(float(v) for v in values.unique())
        if unique in ({0.0}, {1.0}):
            rows.append(
                _suspicious_row(
                    path,
                    "",
                    "all_zero_or_all_one_metric_column",
                    "warning",
                    (
                        f"column {column} has constant value {next(iter(unique))} "
                        f"across {len(values)} rows"
                    ),
                    "confirm the column is a flag/status encoding or recompute metrics",
                )
            )
    return rows


def _pass_with_bad_metric_suspicions(path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "status" not in frame.columns:
        return []
    numeric_columns = [
        col
        for col in frame.columns
        if any(token in col.lower() for token in ("rmse", "error", "residual", "metric"))
    ]
    rows: list[dict[str, Any]] = []
    pass_mask = frame["status"].astype(str).str.lower().isin({"pass", "passed"})
    for index, record in frame[pass_mask].iterrows():
        for col in numeric_columns:
            value = pd.to_numeric(pd.Series([record.get(col)]), errors="coerce").iloc[0]
            raw = str(record.get(col, "")).lower()
            if raw in {"nan", "inf", "-inf"} or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                rows.append(
                    _suspicious_row(
                        path,
                        index,
                        "pass_status_with_nan_or_inf_metric",
                        "needs_review",
                        f"status=pass with {col}={record.get(col)}",
                        "inspect the row and separate diagnostic failures from passing metrics",
                    )
                )
    return rows


def _duplicate_case_metric_suspicions(path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or "case" not in frame.columns or len(frame) < 4:
        return []
    metric_cols = [
        col
        for col in frame.columns
        if any(token in col.lower() for token in ("rmse", "error", "residual", "condition"))
    ]
    if not metric_cols:
        return []
    collapsed = frame[["case", *metric_cols]].drop_duplicates()
    if collapsed["case"].nunique() > 1 and len(collapsed[metric_cols].drop_duplicates()) == 1:
        return [
            _suspicious_row(
                path,
                "",
                "constant_metric_values_across_cases",
                "needs_review",
                "all cases share identical metric tuple values",
                "verify this is expected aggregation and not a copied result row",
            )
        ]
    return []


def _qsvt_over_ridge_suspicions(path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in frame.iterrows():
        text = " ".join(str(value).lower() for value in record.to_dict().values())
        if "qsvt_outperforms_ridge" in text and any(
            token in text for token in ("true", "yes", "1")
        ):
            rows.append(
                _suspicious_row(
                    path,
                    index,
                    "qsvt_target_reported_better_than_ridge",
                    "blocker",
                    "row appears to report QSVT-target outperforming Ridge",
                    "restore matched-alpha equivalence boundary or explain a different estimator",
                )
            )
    return rows


def _text_suspicious(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8").lower()
    except (OSError, UnicodeDecodeError):
        return []
    risky = (
        "qsvt beats ridge",
        "qsvt outperforms ridge",
        "quantum speedup is demonstrated",
        "real pmu/scada validation",
        "field-calibrated statistics are supported",
        "full ieee-scale full-vector readout is solved",
    )
    rows: list[dict[str, Any]] = []
    for phrase in risky:
        if phrase not in text or _safe_risk_phrase_context(text, phrase):
            continue
        rows.append(
            _suspicious_row(
                path,
                "",
                "unsupported_claim_appears_supported",
                "blocker",
                f"found phrase: {phrase}",
                "revise to a negated limitation or remove the claim",
            )
        )
    return rows


def _safe_risk_phrase_context(text: str, phrase: str) -> bool:
    index = text.find(phrase)
    if index < 0:
        return True
    context = text[max(0, index - 180) : index + len(phrase) + 180]
    safe_cues = (
        "not ",
        "no ",
        "do not",
        "does not",
        "cannot",
        "never",
        "unsupported",
        "do-not-claim",
        "do_not_claim",
        "disallowed",
        "limitation",
        "future work",
        "out of scope",
        "not demonstrated",
        "not claimed",
        "rather than",
        "allowed/disallowed",
    )
    return any(cue in context for cue in safe_cues)


def _test_quality_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _inventory_rows(repo_root / "tests"):
        category = str(record["test_category"])
        scientific = category in SCIENTIFIC_CATEGORIES
        rows.append(
            {
                "test_file": record["test_file"],
                "test_name": record["test_name"],
                "classification": category,
                "evidence_family": record["paper_claims_supported"],
                "checks_numeric_behavior": record["uses_numeric_assertions"],
                "checks_claim_boundary": _yes_no(category == "claim_boundary_guard"),
                "checks_artifact_schema_only": _yes_no(category == "artifact_schema"),
                "is_smoke_only": _yes_no(category == "smoke_only"),
                "supports_scientific_claim": _yes_no(scientific and category != "smoke_only"),
                "notes": record["reason_for_classification"],
            }
        )
    return rows


def _claim_boundary_rows(package_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = read_csv(package_root / "claim_support_matrix_final.csv")
    family = read_csv(package_root / "claim_boundaries" / "readout_claim_family_table.csv")
    lint = read_csv(package_root / "claim_lint" / "claim_lint_report.csv")
    high_lint = (
        lint[lint["risk_level"].astype(str) == "high"]
        if not lint.empty and "risk_level" in lint.columns
        else pd.DataFrame()
    )

    expected = (
        (
            "C10",
            "Legacy full update-vector readout family remains parent/assumption-only.",
            "assumption_only",
        ),
        (
            "C10a",
            "Selected-subproblem readout is supported with explicit limitations.",
            "supported_with_limitations",
        ),
        (
            "C10b",
            "Full IEEE-scale full-vector readout remains assumption/future work.",
            "assumption_only",
        ),
        (
            "C11",
            "QSVT-over-Ridge superiority remains unsupported/do-not-claim.",
            "unsupported_do_not_claim",
        ),
        ("C12", "Quantum speedup remains unsupported/do-not-claim.", "unsupported_do_not_claim"),
        (
            "C13",
            "Real PMU/SCADA validation remains unsupported/do-not-claim.",
            "unsupported_do_not_claim",
        ),
        ("FIELD_STATS", "Field-calibrated statistics remain future work.", "future_work"),
        ("FULL_NONLINEAR_QSVT", "Full nonlinear QSVT loop remains future work.", "future_work"),
        ("HARDWARE_EXECUTION", "Hardware execution remains future work.", "future_work"),
    )

    rows: list[dict[str, Any]] = []
    for claim_id, text, expected_status in expected:
        current, artifacts = _current_claim_status(claim_id, matrix, family, package_root)
        matches = current == expected_status
        risk = "none" if matches else ("high" if claim_id.startswith("C") else "medium")
        rows.append(
            {
                "claim_id": claim_id,
                "claim_text": text,
                "current_status": current,
                "expected_status": expected_status,
                "status_matches_expected": _yes_no(matches),
                "supporting_artifacts": artifacts,
                "contradicting_artifacts": _lint_artifacts_for_claim(high_lint, claim_id),
                "risk_level": risk,
                "recommended_action": "none" if matches else "restore conservative claim boundary",
                "notes": _claim_note(claim_id),
            }
        )

    unsupported_rows: list[dict[str, Any]] = [
        row for row in rows if row["risk_level"] in {"medium", "high", "blocker"}
    ]
    for _, record in high_lint.iterrows():
        unsupported_rows.append(
            {
                "claim_id": "CLAIM_LINT_HIGH_RISK",
                "claim_text": str(record.get("line_excerpt", "")),
                "current_status": "positive_or_ambiguous_claim",
                "expected_status": "no high-risk positive overclaims",
                "status_matches_expected": "no",
                "supporting_artifacts": "",
                "contradicting_artifacts": str(record.get("file_path", "")),
                "risk_level": "high",
                "recommended_action": str(record.get("recommended_action", "")),
                "notes": str(record.get("matched_phrase", "")),
            }
        )
    return rows, unsupported_rows


def _current_claim_status(
    claim_id: str, matrix: pd.DataFrame, family: pd.DataFrame, package_root: Path
) -> tuple[str, str]:
    if claim_id in {"C10", "C10a", "C10b"} and not family.empty:
        hit = family[family["claim_id"].astype(str) == claim_id]
        if not hit.empty:
            return str(
                hit.iloc[0].get("status", "")
            ), "claim_boundaries/readout_claim_family_table.csv"
    if claim_id.startswith("C") and not matrix.empty:
        hit = matrix[matrix["claim_id"].astype(str) == claim_id]
        if not hit.empty:
            return str(hit.iloc[0].get("support_status", "")), "claim_support_matrix_final.csv"
    future_markers = {
        "FIELD_STATS": "field-calibrated",
        "FULL_NONLINEAR_QSVT": "full nonlinear",
        "HARDWARE_EXECUTION": "hardware execution",
    }
    marker = future_markers.get(claim_id, "")
    if marker:
        text = _combined_text(package_root, suffixes={".md", ".csv"})
        if marker in text and any(
            cue in text for cue in ("future work", "not demonstrated", "unsupported")
        ):
            return "future_work", "limitations and claim-boundary artifacts"
    return "missing", ""


def _completion_rows(
    *,
    inventory_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    package_root: Path,
) -> list[dict[str, Any]]:
    rerun_by_name = {row["command_name"]: row for row in rerun_rows}
    test_text = " ".join(
        f"{row['test_file']} {row['evidence_family']}" for row in test_rows
    ).lower()
    main_manifest = read_csv(package_root / "main_tables" / "main_tables_manifest.csv")
    appendix = read_csv(package_root / "final_tables" / "appendix_table_index.csv")
    rows: list[dict[str, Any]] = []
    for record in inventory_rows:
        family = str(record["experiment_family"])
        status = str(record["status"])
        outputs_exist = bool(str(record["actual_output_dirs_found"]).strip())
        tests_exist = _family_has_tests(family, test_text)
        tests_real = tests_exist and _family_has_real_tests(family, test_rows)
        rerunnable = _family_rerunnable(family, rerun_by_name)
        used_main = _family_used_in_frame(family, main_manifest)
        used_appendix = _family_used_in_frame(family, appendix)
        diagnostic = status == "diagnostic_only"
        future = status == "future_work" or "future work" in str(record["notes"]).lower()
        completion = _completion_status(status, outputs_exist, tests_real, diagnostic, future)
        rows.append(
            {
                "experiment_family": family,
                "planned": "yes",
                "implemented": _yes_no(
                    bool(record["actual_scripts_found"] or record["actual_source_files_found"])
                ),
                "outputs_exist": _yes_no(outputs_exist),
                "tests_exist": _yes_no(tests_exist),
                "tests_are_real_validation": _yes_no(tests_real),
                "rerunnable": _yes_no(rerunnable),
                "used_in_main_tables": _yes_no(used_main),
                "used_in_appendix": _yes_no(used_appendix),
                "diagnostic_only": _yes_no(diagnostic),
                "future_work": _yes_no(future),
                "limitations": record["notes"],
                "completion_status": completion,
                "notes": record["missing_or_future_reason"],
            }
        )
    return rows


def _artifact_use_readiness_rows(package_root: Path) -> list[dict[str, Any]]:
    manifest = read_csv(
        package_root / "pre_manuscript_usability_audit" / "paper_use_decision_manifest.csv"
    )
    if manifest.empty:
        return [
            {
                "artifact_path": "",
                "artifact_group": "",
                "recommended_use": "",
                "paper_section_hint": "",
                "status": "missing",
                "has_blocker": "yes",
                "readiness_status": "missing_manifest",
                "source_artifact": "",
                "safe_claim_text": "",
                "required_limitation_text": "",
                "do_not_claim_text": "",
                "notes": "paper_use_decision_manifest.csv is missing",
            }
        ]
    rows: list[dict[str, Any]] = []
    for _, record in manifest.iterrows():
        recommended = str(record.get("recommended_use", ""))
        status = str(record.get("status", ""))
        blocker = status in {"needs_fix", "blocked"} or (
            recommended in {"main_text", "main_text_table", "main"}
            and status not in {"usable_main", "main_text_ready"}
        )
        readiness = _readiness_status(recommended, status)
        rows.append(
            {
                "artifact_path": record.get("artifact_path", ""),
                "artifact_group": record.get("artifact_group", ""),
                "recommended_use": recommended,
                "paper_section_hint": record.get("paper_section_hint", ""),
                "status": status,
                "has_blocker": _yes_no(blocker),
                "readiness_status": readiness,
                "source_artifact": record.get("source_artifact", ""),
                "safe_claim_text": record.get("safe_claim_text", ""),
                "required_limitation_text": record.get("required_limitation_text", ""),
                "do_not_claim_text": record.get("do_not_claim_text", ""),
                "notes": record.get("notes", ""),
            }
        )
    return rows


def _scorecard_rows(
    *,
    env_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    env_warnings = sum(1 for row in env_rows if row["status"] != "ok")
    rows.append(
        _score_row(
            "repository_environment",
            "ready_with_minor_notes",
            0,
            0,
            env_warnings,
            1,
            0,
            0,
            "record unavailable Git state if applicable",
            "environment captured",
        )
    )

    missing = _count_status(inventory_rows, "status", "missing")
    future = _count_status(inventory_rows, "status", "future_work")
    limited = _count_status(inventory_rows, "status", "runtime_limited")
    rows.append(
        _score_row(
            "planned_experiment_inventory",
            "ready_with_minor_notes" if missing == 0 else "needs_fix_before_manuscript",
            0,
            missing,
            limited,
            _count_status(inventory_rows, "status", "present_and_reproducible"),
            0,
            future,
            "review missing/future classifications",
            "planned families audited",
        )
    )

    untraceable = sum(
        1
        for row in provenance_rows
        if row["status"] in {"untraceable_needs_review", "source_missing"}
        and _is_scientific_result_string(str(row["output_artifact"]))
    )
    source_unclear = _count_status(provenance_rows, "status", "source_unclear")
    rows.append(
        _score_row(
            "source_to_output_provenance",
            "blocked" if untraceable else "ready_with_minor_notes",
            untraceable,
            source_unclear,
            0,
            _count_status(provenance_rows, "status", "traceable"),
            0,
            0,
            "inspect source-unclear artifacts",
            "mirrored/manual/generated outputs classified",
        )
    )

    failed_reruns = _count_status(rerun_rows, "status", "failed")
    not_run = sum(1 for row in rerun_rows if str(row["status"]).startswith("not_run"))
    rows.append(
        _score_row(
            "reproducibility_rerun",
            "blocked" if failed_reruns else "ready_with_minor_notes",
            failed_reruns,
            not_run,
            0,
            _count_status(rerun_rows, "status", "pass_reproducible")
            + _count_status(rerun_rows, "status", "pass_changed_allowed"),
            0,
            0,
            "rerun deferred runtime-limited commands explicitly",
            "key scripts rerun or marked not run",
        )
    )

    suspicious_blockers = _count_status(suspicious_rows, "is_blocker", "yes")
    suspicious_review = _count_status(suspicious_rows, "severity", "needs_review")
    suspicious_warning = _count_status(suspicious_rows, "severity", "warning")
    rows.append(
        _score_row(
            "suspicious_result_detection",
            "blocked" if suspicious_blockers else "ready_with_minor_notes",
            suspicious_blockers,
            suspicious_review,
            suspicious_warning,
            0,
            0,
            0,
            "review warnings; fix blockers if any",
            "heuristic, not proof of absence",
        )
    )

    sci_smoke = sum(
        1
        for row in test_rows
        if row["supports_scientific_claim"] == "yes" and row["is_smoke_only"] == "yes"
    )
    smoke = _count_status(test_rows, "classification", "smoke_only")
    rows.append(
        _score_row(
            "test_quality",
            "blocked" if sci_smoke else "ready_with_minor_notes",
            sci_smoke,
            0,
            smoke,
            _count_status(test_rows, "classification", "real_validation"),
            0,
            0,
            "keep smoke-only tests outside scientific suite",
            "scientific suite smoke-only count is explicit",
        )
    )

    claim_high = sum(1 for row in claim_rows if row["risk_level"] in {"high", "blocker"})
    rows.append(
        _score_row(
            "claim_boundary",
            "blocked" if claim_high else "ready_with_minor_notes",
            claim_high,
            0,
            0,
            len(claim_rows) - claim_high,
            0,
            0,
            "restore conservative wording for any high-risk claim",
            "C10/C10a/C10b and C11-C13 checked",
        )
    )

    completion_missing = _count_status(completion_rows, "completion_status", "missing_needs_fix")
    completion_future = _count_status(completion_rows, "completion_status", "future_work")
    completion_limited = _count_status(
        completion_rows, "completion_status", "complete_with_limitations"
    )
    rows.append(
        _score_row(
            "experiment_completion",
            "needs_fix_before_manuscript" if completion_missing else "ready_with_minor_notes",
            0,
            completion_missing,
            0,
            _count_status(completion_rows, "completion_status", "complete"),
            completion_limited,
            completion_future,
            "do not promote future/diagnostic rows",
            "completion matrix generated",
        )
    )

    readiness_blockers = _count_status(readiness_rows, "has_blocker", "yes")
    readiness_diag = _count_status(readiness_rows, "readiness_status", "diagnostic_only")
    rows.append(
        _score_row(
            "artifact_use_readiness",
            "blocked" if readiness_blockers else "ready_with_minor_notes",
            readiness_blockers,
            0,
            0,
            _count_status(readiness_rows, "readiness_status", "main_text_ready"),
            readiness_diag,
            _count_status(readiness_rows, "readiness_status", "future_work"),
            "use diagnostic-only artifacts only as diagnostics",
            "paper-use manifest audited",
        )
    )

    non_overall = rows[:]
    blockers = sum(int(row["blocker_count"]) for row in non_overall)
    needs = sum(int(row["needs_review_count"]) for row in non_overall)
    warnings = sum(int(row["warning_count"]) for row in non_overall)
    complete = sum(int(row["complete_count"]) for row in non_overall)
    partial = sum(int(row["partial_count"]) for row in non_overall)
    future_work = sum(int(row["future_work_count"]) for row in non_overall)
    if blockers:
        overall_status = "blocked"
    elif needs or warnings or future_work or partial:
        overall_status = "ready_with_minor_notes"
    else:
        overall_status = "ready_for_manuscript"
    rows.append(
        {
            "category": "overall",
            "status": overall_status,
            "blocker_count": blockers,
            "needs_review_count": needs,
            "warning_count": warnings,
            "complete_count": complete,
            "partial_count": partial,
            "future_work_count": future_work,
            "recommended_action": (
                "fix blockers before manuscript"
                if blockers
                else "proceed with documented limitations"
            ),
            "notes": "overall status is blocker-sensitive; future work alone is not a failure",
        }
    )
    return rows


def _score_row(
    category: str,
    status: str,
    blockers: int,
    needs: int,
    warnings: int,
    complete: int,
    partial: int,
    future: int,
    action: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "status": status,
        "blocker_count": blockers,
        "needs_review_count": needs,
        "warning_count": warnings,
        "complete_count": complete,
        "partial_count": partial,
        "future_work_count": future,
        "recommended_action": action,
        "notes": notes,
    }


def _write_audit_outputs(
    *,
    output_dir: Path,
    input_config: dict[str, Any],
    env_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    unsupported_rows: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    scorecard_rows: list[dict[str, Any]],
) -> dict[str, str]:
    paths: dict[str, Path] = {}
    paths["repo_environment_snapshot_csv"] = rows_to_table(
        env_rows, output_dir / "repo_environment_snapshot.csv", ENV_COLUMNS
    )
    paths["repo_environment_snapshot_md"] = _write_md(
        output_dir / "repo_environment_snapshot.md",
        "Repository and Environment Snapshot",
        env_rows,
        "Records exact commands requested by the audit prompt.",
    )
    paths["planned_experiment_inventory_csv"] = rows_to_table(
        inventory_rows, output_dir / "planned_experiment_inventory.csv", INVENTORY_COLUMNS
    )
    paths["planned_experiment_inventory_md"] = _write_md(
        output_dir / "planned_experiment_inventory.md",
        "Planned Experiment Inventory",
        inventory_rows,
        "Canonical inventory of planned families and their current evidence status.",
    )
    paths["source_to_output_provenance_csv"] = rows_to_table(
        provenance_rows, output_dir / "source_to_output_provenance.csv", PROVENANCE_COLUMNS
    )
    paths["source_to_output_provenance_md"] = _write_md(
        output_dir / "source_to_output_provenance.md",
        "Source-to-Output Provenance",
        provenance_rows,
        "Traceability of key package and phase artifacts.",
    )
    paths["reproducibility_rerun_audit_csv"] = rows_to_table(
        rerun_rows, output_dir / "reproducibility_rerun_audit.csv", RERUN_COLUMNS
    )
    paths["reproducibility_rerun_summary_md"] = _write_rerun_summary(
        output_dir / "reproducibility_rerun_summary.md", rerun_rows
    )
    paths["suspicious_result_audit_csv"] = rows_to_table(
        suspicious_rows, output_dir / "suspicious_result_audit.csv", SUSPICIOUS_COLUMNS
    )
    paths["suspicious_result_summary_md"] = _write_suspicious_summary(
        output_dir / "suspicious_result_summary.md", suspicious_rows
    )
    paths["test_quality_full_repo_audit_csv"] = rows_to_table(
        test_rows, output_dir / "test_quality_full_repo_audit.csv", TEST_COLUMNS
    )
    paths["smoke_only_tests_csv"] = rows_to_table(
        [row for row in test_rows if row["is_smoke_only"] == "yes"],
        output_dir / "smoke_only_tests.csv",
        TEST_COLUMNS,
    )
    paths["real_validation_tests_csv"] = rows_to_table(
        [row for row in test_rows if row["classification"] == "real_validation"],
        output_dir / "real_validation_tests.csv",
        TEST_COLUMNS,
    )
    paths["scientific_validation_suite_csv"] = rows_to_table(
        [row for row in test_rows if row["supports_scientific_claim"] == "yes"],
        output_dir / "scientific_validation_suite.csv",
        TEST_COLUMNS,
    )
    paths["test_quality_summary_md"] = _write_test_summary(
        output_dir / "test_quality_summary.md", test_rows
    )
    paths["claim_boundary_full_repo_audit_csv"] = rows_to_table(
        claim_rows, output_dir / "claim_boundary_full_repo_audit.csv", CLAIM_COLUMNS
    )
    paths["unsupported_claim_audit_csv"] = rows_to_table(
        unsupported_rows, output_dir / "unsupported_claim_audit.csv", CLAIM_COLUMNS
    )
    paths["claim_boundary_summary_md"] = _write_claim_summary(
        output_dir / "claim_boundary_summary.md", claim_rows, unsupported_rows
    )
    paths["experiment_completion_matrix_csv"] = rows_to_table(
        completion_rows, output_dir / "experiment_completion_matrix.csv", COMPLETION_COLUMNS
    )
    paths["experiment_completion_summary_md"] = _write_completion_summary(
        output_dir / "experiment_completion_summary.md", completion_rows
    )
    paths["artifact_use_readiness_audit_csv"] = rows_to_table(
        readiness_rows, output_dir / "artifact_use_readiness_audit.csv", READINESS_COLUMNS
    )
    paths["artifact_use_readiness_summary_md"] = _write_readiness_summary(
        output_dir / "artifact_use_readiness_summary.md", readiness_rows
    )
    paths["full_repo_evidence_scorecard_csv"] = rows_to_table(
        scorecard_rows, output_dir / "full_repo_evidence_scorecard.csv", SCORECARD_COLUMNS
    )
    paths["full_repo_evidence_audit_summary_md"] = _write_scorecard_summary(
        output_dir / "full_repo_evidence_audit_summary.md", scorecard_rows
    )

    manifest = write_manifest(
        output_dir,
        artifacts={key: str(path) for key, path in paths.items()},
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    paths["manifest"] = manifest
    return {key: str(path) for key, path in paths.items()}


def _write_md(path: Path, title: str, rows: list[dict[str, Any]], intro: str) -> Path:
    counts = _counts_by(rows, "status")
    lines = [
        f"# {title}",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        intro,
        "",
        "## Summary",
        *[f"- {key}: {value}" for key, value in sorted(counts.items())],
        f"- rows: {len(rows)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_rerun_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    counts = _counts_by(rows, "status")
    not_run_lines = [
        f"- {row['command_name']}: {row['notes']}"
        for row in rows
        if str(row["status"]).startswith("not_run")
    ]
    lines = [
        "# Reproducibility Rerun Summary",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Key scripts were rerun when practical. Runtime-limited or recursive checks are recorded "
        "rather than silently skipped.",
        "",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "## Not Run",
        *(not_run_lines or ["- none"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_suspicious_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    counts = _counts_by(rows, "severity")
    blockers = _count_status(rows, "is_blocker", "yes")
    lines = [
        "# Suspicious / Fake-Result Heuristic Audit",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "This is a heuristic scan, not a proof. Warnings and needs-review rows are not treated "
        "as fabrication by themselves.",
        "",
        f"- blockers: {blockers}",
        *[f"- {severity}: {count}" for severity, count in sorted(counts.items())],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_test_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    counts = _counts_by(rows, "classification")
    sci = [row for row in rows if row["supports_scientific_claim"] == "yes"]
    sci_smoke = [row for row in sci if row["is_smoke_only"] == "yes"]
    lines = [
        "# Test Quality and Smoke-Test Audit",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        f"- total_tests_inventory_count: {len(rows)}",
        f"- real_validation_count: {counts.get('real_validation', 0)}",
        f"- integration_regression_count: {counts.get('integration_regression', 0)}",
        f"- artifact_schema_count: {counts.get('artifact_schema', 0)}",
        f"- claim_boundary_guard_count: {counts.get('claim_boundary_guard', 0)}",
        f"- unit_behavior_count: {counts.get('unit_behavior', 0)}",
        f"- smoke_only_count: {counts.get('smoke_only', 0)}",
        f"- unknown_count: {counts.get('unknown', 0)}",
        f"- scientific_validation_suite_count: {len(sci)}",
        f"- scientific_validation_suite_smoke_only_count: {len(sci_smoke)}",
        "",
        "The scientific validation suite has zero smoke-only tests by construction when "
        "`scientific_validation_suite_smoke_only_count` is 0.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_claim_summary(
    path: Path, rows: list[dict[str, Any]], unsupported: list[dict[str, Any]]
) -> Path:
    high = sum(1 for row in rows if row["risk_level"] in {"high", "blocker"})
    lines = [
        "# Claim Boundary Full-Repo Audit",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        (
            "- C10/C10a/C10b rows checked: "
            f"{sum(1 for row in rows if str(row['claim_id']).startswith('C10'))}"
        ),
        f"- high-risk claims: {high}",
        f"- unsupported-claim findings: {len(unsupported)}",
        "- C11 QSVT-over-Ridge, C12 speedup, and C13 real PMU/SCADA remain unsupported when "
        "their statuses match `unsupported_do_not_claim`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_completion_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    counts = _counts_by(rows, "completion_status")
    lines = [
        "# Experiment Completion Matrix Summary",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_readiness_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    counts = _counts_by(rows, "readiness_status")
    blockers = _count_status(rows, "has_blocker", "yes")
    lines = [
        "# Artifact Use and Manuscript Readiness Audit",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        f"- blockers: {blockers}",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_scorecard_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    overall = _overall_score(rows)
    lines = [
        "# Full-Repo Evidence Audit Summary",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        f"- overall_status: {overall.get('status', 'unknown')}",
        f"- blocker_count: {overall.get('blocker_count', 0)}",
        f"- needs_review_count: {overall.get('needs_review_count', 0)}",
        f"- warning_count: {overall.get('warning_count', 0)}",
        "",
        "This audit verifies evidence integrity and reproducibility of existing artifacts only. "
        "It does not demonstrate efficient full IEEE-scale full-vector readout, quantum speedup, "
        "QSVT numerical superiority over Ridge/Tikhonov, full nonlinear QSVT solving, real "
        "PMU/SCADA validation, field-calibrated stress statistics, hardware execution, or "
        "deployment readiness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _mirror_audit(source: Path, destination: Path) -> None:
    ensure_directory(destination)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json"}:
            shutil.copy2(path, destination / path.name)


def _existing_globs(repo_root: Path, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path) for path in sorted(repo_root.glob(pattern)) if path.exists())
    return found


def _existing_paths(repo_root: Path, paths: tuple[str, ...]) -> list[str]:
    return [str(repo_root / rel) for rel in paths if (repo_root / rel).exists()]


def _row_count_summary(repo_root: Path, paths: tuple[str, ...]) -> str:
    counts = {
        rel: _artifact_row_count(repo_root / rel)
        for rel in paths
        if (repo_root / rel).exists() and (repo_root / rel).suffix.lower() == ".csv"
    }
    return json.dumps(counts, sort_keys=True)


def _artifact_row_count(path: Path) -> Any:
    if not path.is_file():
        return ""
    if path.suffix.lower() == ".csv":
        frame = read_csv(path)
        return len(frame)
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return ""


def _artifact_group(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "outputs" and parts[1] == "final_manuscript_package":
        if path.parent == repo_root / "outputs" / "final_manuscript_package":
            return "final_manuscript_package"
        return parts[2] if len(parts) > 2 else "final_manuscript_package"
    if len(parts) >= 2 and parts[0] == "outputs":
        return parts[1]
    return parts[0]


def _source_script_for_group(group: str) -> str:
    mapping = {family.name: ";".join(family.scripts) for family in EXPERIMENT_FAMILIES}
    mapping.update(
        {
            "phase0_evidence_audit": "scripts/build_manuscript_evidence_audit.py",
            "evidence_freeze": "scripts/freeze_manuscript_evidence.py",
            "final_figures": "scripts/render_final_manuscript_figures.py",
            "final_tables": "scripts/select_final_manuscript_tables.py",
            "gap_closing_audit": "scripts/run_gap_closing_audit.py",
            "phase1_measurement_inventory": "scripts/build_paper_measurement_inventory.py",
            "phase2_classical_spectral_filtering": (
                "scripts/build_classical_spectral_filtering_audit.py"
            ),
            "phase2_classical_recheck": "scripts/recheck_classical_spectral_filtering_audit.py",
            "phase3_alpha_sensitivity": "scripts/build_alpha_sensitivity_consolidation.py",
            "phase3_full_classical_alpha_sweep": "scripts/run_full_alpha_sweep_classical.py",
            "phase2_baseline_coverage_extension": "scripts/run_baseline_coverage_extension.py",
            "phase4_nonlinear_ac": "scripts/build_nonlinear_ac_consolidation.py",
            "phase4_nonlinear_alpha_stress": "scripts/run_nonlinear_ac_alpha_stress.py",
            "phase5_structured_stress_ablation": (
                "scripts/build_structured_stress_ablation_consolidation.py"
            ),
            "phase5_measurement_type_ablation": "scripts/run_measurement_type_ablation.py",
            "phase5_compound_structured_stress": "scripts/run_compound_structured_stress.py",
            "phase6_qsvt_resource_phase": "scripts/build_qsvt_resource_phase_summary.py",
            "phase6_readout_limitation_formalization": (
                "scripts/build_readout_limitation_formalization.py"
            ),
            "metrics": "scripts/build_metric_definitions.py",
            "claim_boundaries": "scripts/build_claim_boundary_docs.py",
            "nonlinear_trajectories": "scripts/extract_nonlinear_ac_trajectories.py",
            "initializer_ablation": "scripts/run_initializer_ablation.py",
            "ieee300_runtime_extension": "scripts/run_ieee300_runtime_extension.py",
            "full_vector_readout": "scripts/run_full_vector_readout_demo.py",
            "phase_synthesis_refinement": "scripts/run_phase_synthesis_refinement.py",
            "jacobian_validation": "scripts/run_jacobian_validation.py",
            "measurement_row_metadata_audit": "scripts/run_measurement_row_metadata_audit.py",
            "pre_manuscript_usability_audit": "scripts/run_pre_manuscript_usability_audit.py",
            "test_quality_audit": "scripts/audit_test_quality.py",
            "test_quality_appendix": "scripts/build_test_quality_appendix.py",
            "statistical_summary": "scripts/build_final_statistical_aggregation.py",
            "main_tables": "scripts/build_main_paper_tables.py",
            "claim_lint": "scripts/lint_claim_boundaries.py",
            "manuscript_claim_lint": "scripts/lint_manuscript_claims.py",
            "final_artifact_validation": "scripts/validate_final_manuscript_artifacts.py",
            "latex_assets": "scripts/export_latex_assets.py",
            "canonical_numbers": "scripts/check_paper_numbers.py",
            "pre_submission_check": "scripts/pre_submission_check.py",
            "full_repo_evidence_audit": "scripts/run_full_repo_evidence_audit.py",
        }
    )
    return mapping.get(group, "")


def _source_module_for_group(group: str) -> str:
    normalized = group.removeprefix("phase0_").removeprefix("phase1_")
    normalized = normalized.removeprefix("phase2_").removeprefix("phase3_")
    normalized = normalized.removeprefix("phase4_").removeprefix("phase5_")
    normalized = normalized.removeprefix("phase6_")
    candidates = {
        "full_classical_alpha_sweep": "full_alpha_sweep_classical",
        "classical_spectral_filtering": "classical_spectral_filtering_audit",
        "structured_stress_ablation": "structured_stress_ablation_consolidation",
        "qsvt_resource_phase": "qsvt_resource_phase_summary",
    }
    module = candidates.get(normalized, normalized)
    return f"robust_qsvt_se.paper.{module}" if _source_script_for_group(group) else ""


def _source_inputs_for_artifact(path: Path) -> str:
    if path.suffix.lower() != ".csv":
        return ""
    frame = read_csv(path)
    for column in ("source_artifact", "source_artifacts", "source_input_artifacts"):
        if column in frame.columns:
            values = [str(value) for value in frame[column].dropna().astype(str).unique()]
            return ";".join(values[:20])
    return ""


def _has_source_column(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    frame = read_csv(path)
    return any(
        column in frame.columns
        for column in (
            "source_artifact",
            "source_artifacts",
            "source_script",
            "source_input_artifacts",
        )
    )


def _has_manifest_entry(path: Path) -> bool:
    manifest = path.parent / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    text = json.dumps(data)
    return path.name in text or str(path) in text


def _mirrored_from(repo_root: Path, path: Path, group: str) -> str:
    try:
        rel = path.relative_to(repo_root / "outputs" / "final_manuscript_package" / group)
    except ValueError:
        return ""
    raw = repo_root / "outputs" / group / rel
    if raw.exists() and raw.resolve() != path.resolve() and _same_file(raw, path):
        return str(raw)
    return ""


def _same_file(a: Path, b: Path) -> bool:
    try:
        return filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def _generated_or_manual(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return "manual_or_generated_doc"
    return "generated"


def _provenance_status(
    *,
    path: Path,
    source_script: str,
    manifest_entry: bool,
    mirrored_from: str,
    generated: str,
) -> str:
    if mirrored_from:
        return "traceable_mirrored"
    if generated == "manual_or_generated_doc" and source_script:
        return "traceable_manual_doc"
    if source_script and (
        manifest_entry or path.suffix.lower() != ".csv" or _has_source_column(path)
    ):
        return "traceable"
    if source_script:
        return "traceable"
    if _is_scientific_result_path(path):
        return "source_missing"
    return "source_unclear"


def _provenance_notes(path: Path, status: str) -> str:
    if status == "source_missing":
        return "scientific-looking artifact lacks an inferred source script"
    if status == "source_unclear":
        return "source inferred weakly; review if used as a scientific result"
    if path.suffix.lower() == ".md":
        return "documentation artifact; verify claims against linked tables"
    return ""


def _is_latest(path: Path, mirrored_from: str) -> bool:
    if not mirrored_from:
        return True
    raw = Path(mirrored_from)
    try:
        return path.stat().st_mtime >= raw.stat().st_mtime
    except OSError:
        return False


def _hash_mapping(paths: list[Path]) -> dict[str, str]:
    return {str(path): _file_hash(path) for path in paths if path.exists()}


def _row_count_mapping(paths: list[Path]) -> dict[str, Any]:
    return {str(path): _artifact_row_count(path) for path in paths if path.exists()}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _scripts_in_command(command: str) -> list[str]:
    return [
        part for part in command.split() if part.startswith("scripts/") and part.endswith(".py")
    ]


def _is_documented_empty(path: Path) -> bool:
    lowered = path.name.lower()
    if any(token in lowered for token in ("missing", "unavailable", "manifest")):
        return True
    siblings = list(path.parent.glob("missing_*.csv")) + list(path.parent.glob("*missing*.csv"))
    return bool(siblings)


def _is_scientific_result_path(path: Path) -> bool:
    return _is_scientific_result_string(str(path))


def _is_scientific_result_string(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in (
            "paper_table",
            "main_table",
            "summary",
            "figure_data",
            "validation",
            "readout",
            "alpha",
            "stress",
            "jacobian",
            "metric",
        )
    )


def _current_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _combined_text(root: Path, *, suffixes: set[str]) -> str:
    chunks: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            chunks.append(_current_text(path).lower())
    return "\n".join(chunks)


def _lint_artifacts_for_claim(high_lint: pd.DataFrame, claim_id: str) -> str:
    if high_lint.empty:
        return ""
    return ";".join(str(path) for path in high_lint.get("file_path", pd.Series(dtype=str)).unique())


def _claim_note(claim_id: str) -> str:
    notes = {
        "C10": "Legacy parent should not be counted as independent selected-readout evidence.",
        "C10a": "Selected small subproblems only, with explicit shot/readout limitations.",
        "C10b": "Efficient full-scale readout is not demonstrated.",
        "C11": "QSVT-target equals Ridge/Tikhonov for matched alpha in the classical simulator.",
        "C12": "No speedup or advantage follows from classical simulation/resource diagnostics.",
        "C13": "IEEE/PYPOWER benchmarks are not real PMU/SCADA field data.",
    }
    return notes.get(claim_id, "Boundary item must remain limitation/future work.")


def _family_has_tests(family: str, test_text: str) -> bool:
    tokens = family.lower().split("_")
    return any(token and token in test_text for token in tokens)


def _family_has_real_tests(family: str, test_rows: list[dict[str, Any]]) -> bool:
    tokens = [token for token in family.lower().split("_") if token]
    for row in test_rows:
        haystack = f"{row['test_file']} {row['evidence_family']}".lower()
        if any(token in haystack for token in tokens) and row["classification"] in {
            "real_validation",
            "claim_boundary_guard",
            "integration_regression",
            "artifact_schema",
        }:
            return True
    return False


def _family_rerunnable(family: str, rerun_by_name: dict[str, dict[str, Any]]) -> bool:
    return any(family in key or key in family for key in rerun_by_name)


def _family_used_in_frame(family: str, frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    text = " ".join(frame.astype(str).to_numpy().ravel()).lower()
    tokens = [token for token in family.lower().split("_") if token]
    return any(token in text for token in tokens)


def _completion_status(
    status: str, outputs_exist: bool, tests_real: bool, diagnostic: bool, future: bool
) -> str:
    if future and status == "future_work":
        return "future_work"
    if status == "missing":
        return "missing_needs_fix"
    if diagnostic:
        return "diagnostic_only"
    if status == "runtime_limited":
        return "complete_with_limitations"
    if outputs_exist and tests_real:
        return "complete"
    if outputs_exist:
        return "complete_with_limitations"
    return "partial_not_blocking"


def _readiness_status(recommended: str, status: str) -> str:
    if recommended in {"main_text", "main_text_table", "main"} or status == "usable_main":
        return "main_text_ready"
    if recommended == "appendix" or status == "usable_appendix":
        return "appendix_ready"
    if recommended == "diagnostic_only" or status == "diagnostic_only":
        return "diagnostic_only"
    if recommended == "future_work" or status == "future_work":
        return "future_work"
    if recommended == "do_not_use" or status == "do_not_use":
        return "do_not_use"
    return "appendix_ready" if status.startswith("usable") else status or "unknown"


def _count_status(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")) == value)


def _counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _overall_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("category") == "overall":
            return row
    return {}


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _clean_output(text: str) -> str:
    cleaned = text.strip()
    return cleaned if cleaned else "empty"


def _trim(text: str, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned[:limit]


@dataclass(frozen=True)
class ShellResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _run_shell(command: str, cwd: Path, *, timeout_seconds: int) -> ShellResult:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return ShellResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return ShellResult(
            None,
            str(exc.stdout or ""),
            str(exc.stderr or ""),
            timed_out=True,
        )


def run_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full-repository evidence audit.")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--package-root", default="outputs/final_manuscript_package")
    parser.add_argument("--output-dir", default="outputs/full_repo_evidence_audit")
    parser.add_argument("--skip-reruns", action="store_true")
    parser.add_argument("--rerun-timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    run = build_full_repo_evidence_audit(
        {
            "input_root": args.input_root,
            "package_root": args.package_root,
            "output_dir": args.output_dir,
            "run_reruns": not args.skip_reruns,
            "rerun_timeout_seconds": args.rerun_timeout_seconds,
        }
    )
    print(f"Wrote full-repo evidence audit to {run['output_dir']}")
    print(
        "overall="
        f"{run['overall_status']} blockers={run['blocker_count']} "
        f"needs_review={run['needs_review_count']} warnings={run['warning_count']}"
    )
    print(f"mirrored audit to {run['package_mirror_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
