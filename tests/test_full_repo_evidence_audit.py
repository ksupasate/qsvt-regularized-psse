"""Full-repository evidence audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.canonical_paper_numbers import build_canonical_paper_numbers
from robust_qsvt_se.paper.final_artifact_validator import build_final_artifact_validator
from robust_qsvt_se.paper.full_repo_evidence_audit import build_full_repo_evidence_audit


def _write_minimal_package(root: Path) -> None:
    (root / "src" / "robust_qsvt_se" / "paper").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "outputs" / "final_manuscript_package" / "claim_boundaries").mkdir(parents=True)
    (root / "outputs" / "final_manuscript_package" / "claim_lint").mkdir()
    (root / "outputs" / "final_manuscript_package" / "pre_manuscript_usability_audit").mkdir()
    (root / "outputs" / "final_manuscript_package" / "main_tables").mkdir()
    (root / "outputs" / "final_manuscript_package" / "final_tables").mkdir()
    (root / "outputs" / "final_manuscript_package" / "final_artifact_validation").mkdir()
    (root / "outputs" / "jacobian_validation").mkdir(parents=True)

    (root / "tests" / "test_numeric.py").write_text(
        "def test_numeric_value():\n    assert abs(1.0 - 1.0) <= 1e-12\n",
        encoding="utf-8",
    )
    (root / "scripts" / "run_jacobian_validation.py").write_text("# script\n", encoding="utf-8")
    (root / "src" / "robust_qsvt_se" / "paper" / "jacobian_validation.py").write_text(
        "# module\n", encoding="utf-8"
    )

    pd.DataFrame(
        {"case": ["ieee14"], "status": ["pass"], "frobenius_relative_error": [1e-9]}
    ).to_csv(
        root / "outputs" / "jacobian_validation" / "ac_jacobian_validation_summary.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "claim_id": ["C10", "C11", "C12", "C13"],
            "support_status": [
                "assumption_only",
                "unsupported_do_not_claim",
                "unsupported_do_not_claim",
                "unsupported_do_not_claim",
            ],
        }
    ).to_csv(
        root / "outputs" / "final_manuscript_package" / "claim_support_matrix_final.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "claim_id": ["C10", "C10a", "C10b"],
            "status": ["assumption_only", "supported_with_limitations", "assumption_only"],
        }
    ).to_csv(
        root
        / "outputs"
        / "final_manuscript_package"
        / "claim_boundaries"
        / "readout_claim_family_table.csv",
        index=False,
    )
    pd.DataFrame(
        columns=[
            "file_path",
            "line_number",
            "matched_phrase",
            "risk_level",
            "classification",
            "recommended_action",
            "line_excerpt",
        ]
    ).to_csv(
        root / "outputs" / "final_manuscript_package" / "claim_lint" / "claim_lint_report.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "artifact_path": ["outputs/final_manuscript_package/table.csv"],
            "artifact_group": ["main_tables"],
            "recommended_use": ["main_text"],
            "paper_section_hint": ["Results"],
            "safe_claim_text": ["controlled benchmark"],
            "required_limitation_text": ["no field data"],
            "do_not_claim_text": ["quantum speedup"],
            "source_artifact": ["source.csv"],
            "status": ["usable_main"],
            "notes": [""],
        }
    ).to_csv(
        root
        / "outputs"
        / "final_manuscript_package"
        / "pre_manuscript_usability_audit"
        / "paper_use_decision_manifest.csv",
        index=False,
    )
    pd.DataFrame(
        {"table": ["T1.csv"], "source_artifacts": ["jacobian_validation"], "status": ["pass"]}
    ).to_csv(
        root / "outputs" / "final_manuscript_package" / "main_tables" / "main_tables_manifest.csv",
        index=False,
    )
    pd.DataFrame({"table_id": ["T1"], "source_artifacts": ["jacobian_validation"]}).to_csv(
        root / "outputs" / "final_manuscript_package" / "final_tables" / "appendix_table_index.csv",
        index=False,
    )


def test_full_repo_audit_writes_required_outputs_and_mirror(tmp_path: Path) -> None:
    _write_minimal_package(tmp_path)
    run = build_full_repo_evidence_audit(
        {
            "repo_root": str(tmp_path),
            "input_root": str(tmp_path / "outputs"),
            "package_root": str(tmp_path / "outputs" / "final_manuscript_package"),
            "output_dir": str(tmp_path / "outputs" / "full_repo_evidence_audit"),
            "run_reruns": False,
        }
    )

    audit_dir = Path(run["output_dir"])
    package_dir = Path(run["package_mirror_dir"])
    required = [
        "repo_environment_snapshot.csv",
        "planned_experiment_inventory.csv",
        "source_to_output_provenance.csv",
        "reproducibility_rerun_audit.csv",
        "suspicious_result_audit.csv",
        "test_quality_full_repo_audit.csv",
        "scientific_validation_suite.csv",
        "claim_boundary_full_repo_audit.csv",
        "experiment_completion_matrix.csv",
        "artifact_use_readiness_audit.csv",
        "full_repo_evidence_scorecard.csv",
        "full_repo_evidence_audit_summary.md",
    ]
    for name in required:
        assert (audit_dir / name).is_file()
        assert (package_dir / name).is_file()

    tests = pd.read_csv(audit_dir / "scientific_validation_suite.csv")
    assert not tests.empty
    assert not (tests["is_smoke_only"].astype(str) == "yes").any()


def test_full_repo_audit_blocker_drives_validator_failure(tmp_path: Path) -> None:
    _write_minimal_package(tmp_path)
    run = build_full_repo_evidence_audit(
        {
            "repo_root": str(tmp_path),
            "input_root": str(tmp_path / "outputs"),
            "package_root": str(tmp_path / "outputs" / "final_manuscript_package"),
            "output_dir": str(tmp_path / "outputs" / "full_repo_evidence_audit"),
            "run_reruns": False,
        }
    )
    scorecard = pd.read_csv(Path(run["package_mirror_dir"]) / "full_repo_evidence_scorecard.csv")
    scorecard.loc[scorecard["category"] == "overall", "status"] = "blocked"
    scorecard.loc[scorecard["category"] == "overall", "blocker_count"] = 1
    scorecard.to_csv(
        Path(run["package_mirror_dir"]) / "full_repo_evidence_scorecard.csv", index=False
    )

    validation = build_final_artifact_validator(
        {
            "input_root": str(tmp_path / "outputs" / "final_manuscript_package"),
            "output_dir": str(
                tmp_path / "outputs" / "final_manuscript_package" / "final_artifact_validation"
            ),
        }
    )
    report = pd.read_csv(validation["artifacts"]["artifact_validation_report"])
    v24 = report[report["check_id"] == "V24"].iloc[0]
    assert v24["status"] == "failed"


def test_canonical_numbers_include_full_repo_audit(tmp_path: Path) -> None:
    _write_minimal_package(tmp_path)
    build_full_repo_evidence_audit(
        {
            "repo_root": str(tmp_path),
            "input_root": str(tmp_path / "outputs"),
            "package_root": str(tmp_path / "outputs" / "final_manuscript_package"),
            "output_dir": str(tmp_path / "outputs" / "full_repo_evidence_audit"),
            "run_reruns": False,
        }
    )
    run = build_canonical_paper_numbers(
        {
            "input_root": str(tmp_path / "outputs" / "final_manuscript_package"),
            "output_dir": str(
                tmp_path / "outputs" / "final_manuscript_package" / "canonical_numbers"
            ),
        }
    )
    numbers = run["numbers"]
    assert "full_repo_audit_overall_status" in numbers
    assert "full_repo_audit_scientific_validation_smoke_only_count" in numbers

    data = json.loads(Path(run["artifacts"]["canonical_paper_numbers_json"]).read_text("utf-8"))
    assert data["numbers"]["full_repo_audit_scientific_validation_smoke_only_count"] == 0
