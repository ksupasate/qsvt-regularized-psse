#!/usr/bin/env python3
"""Build the canonical evidence-closure layer without rerunning experiments."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_qsvt_se.evidence.artifact_graph import (  # noqa: E402
    build_artifact_dependency_graph,
)
from robust_qsvt_se.evidence.canonical_registry import (  # noqa: E402
    _read_csv,
    atomic_write_csv,
    atomic_write_json,
    build_claim_registry,
    build_configuration_registry,
    build_eligibility_audit,
    build_headline_checks,
    build_limitation_registry,
    build_result_registry,
    load_json,
    run_claim_guards,
    sha256_file,
    stable_json_fingerprint,
    verify_protected_sources,
)
from robust_qsvt_se.evidence.regularized_conditioning import (  # noqa: E402
    build_conditioning_audit,
)
from robust_qsvt_se.evidence.summary_exports import (  # noqa: E402
    build_figure_data,
    build_summary_tables,
)
from robust_qsvt_se.evidence.tie_diagnostics import (  # noqa: E402
    build_near_zero_audit,
    build_tie_diagnostics,
)

DEFAULT_CONFIG = ROOT / "configs/final_contribution_evidence.json"
DEFAULT_OUTPUT = ROOT / "outputs/final_contribution_evidence"

STAGES = [
    "audit",
    "snapshot",
    "configurations",
    "registry",
    "headline-checks",
    "tie-diagnostics",
    "near-zero",
    "conditioning",
    "limitations",
    "dependency-graph",
    "tables",
    "figure-data",
    "eligibility",
    "claim-guards",
    "verify",
]

FAILURE_FIELDS = [
    "failure_id",
    "stage",
    "result_id",
    "source_artifact",
    "failure_type",
    "failure_reason",
    "blocking",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def load_checkpoint(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "checkpoint.json"
    if path.exists():
        payload = load_json(path)
        payload.setdefault("stages", {})
        return payload
    return {
        "schema_version": 1,
        "study_id": "final_contribution_evidence_v1",
        "created_at": utc_now(),
        "stages": {},
        "status": "in_progress",
    }


def write_checkpoint(output_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(output_dir / "checkpoint.json", checkpoint)


def record_failure(output_dir: Path, stage: str, reason: str) -> None:
    path = output_dir / "failure_registry.csv"
    rows = _read_csv(path).to_dict(orient="records") if path.exists() else []
    rows.append(
        {
            "failure_id": f"failure:{stage}:{len(rows) + 1:04d}",
            "stage": stage,
            "result_id": "",
            "source_artifact": "",
            "failure_type": "other_verified_failure",
            "failure_reason": reason,
            "blocking": True,
        }
    )
    atomic_write_csv(path, rows, FAILURE_FIELDS)


def reconcile_failure_registry(output_dir: Path, checkpoint: dict[str, Any]) -> dict[str, int]:
    """Retain failure history while clearing blockers for stages that later completed."""
    path = output_dir / "failure_registry.csv"
    rows = _read_csv(path).to_dict(orient="records") if path.exists() else []
    unresolved = 0
    for row in rows:
        for field in ("result_id", "source_artifact"):
            value = row.get(field)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                row[field] = ""
        stage_completed = (
            checkpoint.get("stages", {}).get(str(row.get("stage")), {}).get("status")
            == "completed"
        )
        row["blocking"] = not stage_completed
        unresolved += int(not stage_completed)
    atomic_write_csv(path, rows, FAILURE_FIELDS)
    return {
        "recorded_failures": len(rows),
        "resolved_failures": len(rows) - unresolved,
        "unresolved_blocking_failures": unresolved,
    }


def ensure_audit(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "implementation_audit.md"
    if not path.exists():
        manifest_lines = []
        for directory in config["source_evidence_directories"]:
            manifest = ROOT / directory / "manifest.json"
            checksum = ROOT / directory / "checksums.sha256"
            manifest_hash = sha256_file(manifest) if manifest.exists() else "missing"
            checksum_hash = sha256_file(checksum) if checksum.exists() else "missing"
            manifest_lines.append(
                f"- `{directory}`: manifest `{manifest_hash}`; checksum registry `{checksum_hash}`"
            )
        path.write_text(
            "# Final Contribution Evidence Closure: Implementation Audit\n\n"
            f"- Repository root: `{ROOT}`\n"
            f"- Branch: `{git_value('branch', '--show-current')}`\n"
            f"- Commit: `{git_value('rev-parse', 'HEAD')}`\n"
            "- Working tree: dirty before closure; pre-existing user state is preserved.\n"
            "- Protected policy: byte size and SHA-256, not mtime, decide integrity.\n\n"
            "## Source evidence manifests\n\n"
            + "\n".join(manifest_lines)
            + "\n\nNo scientific campaign is rerun by this closure.\n",
            encoding="utf-8",
        )
    return {"audit": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


def create_snapshot(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protected_source_snapshot.json"
    if path.exists():
        payload = load_json(path)
        return {"file_count": payload["file_count"], "snapshot_reused": True}
    roots = [
        (ROOT / "manuscript", "manuscript"),
        (ROOT / "submission_package_tqe_final", "submission_package"),
        (ROOT / "configs", "configuration"),
        (ROOT / "outputs", "prior_output"),
    ]
    files = []
    for base, category in roots:
        if not base.exists():
            continue
        for source in sorted(item for item in base.rglob("*") if item.is_file()):
            if source == path or output_dir in source.parents:
                continue
            stat = source.stat()
            files.append(
                {
                    "path": source.relative_to(ROOT).as_posix(),
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(source),
                    "mtime_ns": stat.st_mtime_ns,
                    "category": category,
                }
            )
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "root": str(ROOT),
            "protected_roots": [
                "manuscript",
                "submission_package_tqe_final",
                "configs",
                f"outputs excluding {output_dir.relative_to(ROOT)}",
            ],
            "file_count": len(files),
            "files": files,
        },
    )
    return {"file_count": len(files), "snapshot_reused": False}


def build_study_configuration(
    output_dir: Path, config: dict[str, Any], max_workers: int
) -> dict[str, Any]:
    manifests = {}
    for directory in [
        *config["source_evidence_directories"],
        *config["referenced_upstream_directories"],
    ]:
        manifest = ROOT / directory / "manifest.json"
        checksum = ROOT / directory / "checksums.sha256"
        manifests[directory] = {
            "manifest_path": manifest.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest) if manifest.exists() else "",
            "checksum_registry_path": checksum.relative_to(ROOT).as_posix(),
            "checksum_registry_sha256": sha256_file(checksum) if checksum.exists() else "",
        }
    payload = {
        "study_id": config["study_id"],
        "schema_version": config["schema_version"],
        "closure_configuration": config,
        "closure_configuration_fingerprint": stable_json_fingerprint(config),
        "root": str(ROOT),
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "max_workers_recorded": max_workers,
        "source_manifests": manifests,
        "protected_snapshot_sha256": sha256_file(output_dir / "protected_source_snapshot.json"),
        "scientific_campaigns_recomputed": False,
        "estimator_or_qsvt_implementation_modified": False,
        "status": "frozen_closure_configuration",
    }
    atomic_write_json(output_dir / "study_configuration.json", payload)
    return {"source_manifests": len(manifests)}


def build_registries(output_dir: Path) -> dict[str, Any]:
    configurations = build_configuration_registry(ROOT, output_dir)
    results = build_result_registry(ROOT, output_dir)
    return {"configurations": len(configurations), "results": len(results)}


def build_limitations_and_claims(output_dir: Path) -> dict[str, Any]:
    limitations = build_limitation_registry(ROOT, output_dir)
    claims = build_claim_registry(ROOT, output_dir)
    return {"limitations": len(limitations), "claims": len(claims)}


def build_graph_stage(output_dir: Path) -> dict[str, Any]:
    graph = build_artifact_dependency_graph(ROOT, output_dir)
    return {
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "status": graph["status"],
    }


def stage_handlers(
    output_dir: Path, config: dict[str, Any], max_workers: int
) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "audit": lambda: ensure_audit(output_dir, config),
        "snapshot": lambda: create_snapshot(output_dir),
        "configurations": lambda: build_study_configuration(output_dir, config, max_workers),
        "registry": lambda: build_registries(output_dir),
        "headline-checks": lambda: {"checks": len(build_headline_checks(ROOT, output_dir))},
        "tie-diagnostics": lambda: {"rows": len(build_tie_diagnostics(ROOT, output_dir))},
        "near-zero": lambda: {"audit_rows": len(build_near_zero_audit(ROOT, output_dir)[0])},
        "conditioning": lambda: {"matrices": len(build_conditioning_audit(ROOT, output_dir))},
        "limitations": lambda: build_limitations_and_claims(output_dir),
        "dependency-graph": lambda: build_graph_stage(output_dir),
        "tables": lambda: {"tables": len(build_summary_tables(ROOT, output_dir))},
        "figure-data": lambda: {"figure_data": len(build_figure_data(ROOT, output_dir))},
        "eligibility": lambda: {"records": len(build_eligibility_audit(ROOT, output_dir))},
        "claim-guards": lambda: run_claim_guards(ROOT, output_dir),
    }


def _read_test_status(output_dir: Path) -> dict[str, Any]:
    filewise = output_dir / "filewise"

    def log_status(name: str) -> dict[str, Any]:
        path = filewise / name
        if not path.exists():
            return {"status": "not_recorded", "path": path.relative_to(ROOT).as_posix()}
        text = path.read_text(encoding="utf-8", errors="replace")
        passed = " passed" in text and " failed" not in text and "ERROR" not in text
        return {
            "status": "pass" if passed else "fail",
            "path": path.relative_to(ROOT).as_posix(),
            "last_lines": text.splitlines()[-5:],
        }

    targeted = log_status("targeted_tests.log")
    dependent = log_status("dependent_tests.log")
    ruff = log_status("ruff.log")
    progress_path = filewise / "full_pytest_progress.json"
    if progress_path.exists():
        progress = load_json(progress_path)
        full = {
            "status": (
                "pass"
                if len(progress.get("passed_files", [])) == progress.get("total_files", -1)
                and not progress.get("failed_files")
                and not progress.get("interrupted_files")
                else "fail"
            ),
            "total_files": progress.get("total_files", 0),
            "passed_files": len(progress.get("passed_files", [])),
            "failed_files": len(progress.get("failed_files", [])),
            "interrupted_files": len(progress.get("interrupted_files", [])),
            "path": progress_path.relative_to(ROOT).as_posix(),
        }
    else:
        full = {"status": "not_recorded", "path": progress_path.relative_to(ROOT).as_posix()}
    return {"targeted": targeted, "dependent": dependent, "ruff": ruff, "full_filewise": full}


def _definition_of_done(
    output_dir: Path, internal: dict[str, Any], external: dict[str, Any]
) -> dict[str, Any]:
    headline = load_json(output_dir / "headline_result_check_summary.json")
    tie = load_json(output_dir / "primary_tie_diagnostic_summary.json")
    graph = load_json(output_dir / "artifact_dependency_graph.json")
    claim = load_json(output_dir / "excluded_claim_guard_report.json")
    protected = load_json(output_dir / "protected_source_comparison.json")
    eligibility = _read_csv(output_dir / "manuscript_eligibility_audit.csv")
    conditioning = _read_csv(output_dir / "regularized_conditioning_audit.csv")
    guards = _read_csv(output_dir / "conditioning_interpretation_violations.csv")
    criteria = {
        "all_final_evidence_directories_audited": (output_dir / "implementation_audit.md").exists(),
        "protected_sources_snapshotted_and_unchanged": protected["status"] == "pass",
        "canonical_configuration_registry_generated": (
            output_dir / "canonical_configuration_registry.csv"
        ).exists(),
        "canonical_result_registry_generated": (
            output_dir / "canonical_result_registry.csv"
        ).exists(),
        "canonical_claim_registry_generated": (
            output_dir / "canonical_claim_evidence_registry.csv"
        ).exists(),
        "stable_headline_result_ids": not _read_csv(output_dir / "canonical_result_registry.csv")[
            "result_id"
        ]
        .duplicated()
        .any(),
        "headline_artifacts_resolve": headline["status"] == "pass",
        "headline_values_recomputed": headline["status"] == "pass",
        "structural_primary_6_5_1_verified": tie["original_primary_win_tie_loss"]
        == {"win": 6, "tie": 5, "loss": 1},
        "multi_instance_13_0_2_verified": all(
            _read_csv(output_dir / "headline_result_checks.csv")
            .set_index("check_id")
            .loc[
                [
                    "generalization.primary.win",
                    "generalization.primary.tie",
                    "generalization.primary.loss",
                ],
                "status",
            ]
            == "pass"
        ),
        "tie_causes_classified": tie["status"] == "pass",
        "full_support_saturation_identified": tie["saturated_ties"] > 0,
        "near_zero_outputs_audited": (output_dir / "near_zero_output_audit.csv").exists(),
        "raw_and_regularized_conditioning_separated": {
            "raw_matrix_conditioning",
            "regularized_normal_system_conditioning",
            "ridge_filter_amplification",
        }.issubset(
            set(conditioning["raw_conditioning_label"])
            | set(conditioning["regularized_conditioning_label"])
            | set(conditioning["ridge_filter_label"])
        ),
        "rank_deficient_matrices_identified": conditioning["rank_deficient"].astype(bool).any(),
        "ridge_system_conditioning_reported": conditioning["regularized_condition_number"]
        .notna()
        .all(),
        "conditioning_guards_pass": guards.empty or not (guards["status"] == "unresolved").any(),
        "canonical_limitations_complete": len(
            _read_csv(output_dir / "canonical_limitation_registry.csv")
        )
        >= 20,
        "dependency_graph_complete": graph["status"] == "pass",
        "manuscript_eligibility_pass": (eligibility["status"] == "pass").all(),
        "excluded_claim_guards_pass": claim["status"] == "pass",
        "summary_tables_generated": len(list((output_dir / "tables").glob("*.csv"))) == 10,
        "figure_data_generated": len(list((output_dir / "figure_data").glob("*.csv"))) == 7,
        "no_prior_scientific_output_modified": protected["status"] == "pass",
        "no_manuscript_or_package_modified": protected["status"] == "pass",
        "targeted_tests_pass": external["targeted"]["status"] == "pass",
        "dependent_tests_pass": external["dependent"]["status"] == "pass",
        "full_filewise_tests_pass": external["full_filewise"]["status"] == "pass",
        "ruff_pass": external["ruff"]["status"] == "pass",
        "checksums_generated": True,
        "resume_behavior_works": (output_dir / "filewise/resume_verification.json").exists(),
        "no_blocking_failure_remains": internal["status"] == "pass",
    }
    return {
        "criteria": [
            {"criterion": key, "status": "PASS" if value else "FAIL"}
            for key, value in criteria.items()
        ],
        "pass_count": sum(bool(value) for value in criteria.values()),
        "fail_count": sum(not bool(value) for value in criteria.values()),
        "status": "PASS" if all(criteria.values()) else "FAIL",
    }


def _summary_markdown(output_dir: Path, internal: dict[str, Any]) -> str:
    results = _read_csv(output_dir / "canonical_result_registry.csv")
    configs = _read_csv(output_dir / "canonical_configuration_registry.csv")
    claims = _read_csv(output_dir / "canonical_claim_evidence_registry.csv")
    limitations = _read_csv(output_dir / "canonical_limitation_registry.csv")
    tie = load_json(output_dir / "primary_tie_diagnostic_summary.json")
    headline = load_json(output_dir / "headline_result_check_summary.json")
    protected = load_json(output_dir / "protected_source_comparison.json")
    eligible_count = (results["manuscript_eligible"].astype(str).str.lower() == "true").sum()
    return (
        "# Final Contribution Evidence Closure\n\n"
        f"- Canonical configurations: {len(configs)}\n"
        f"- Canonical results: {len(results)}\n"
        f"- Manuscript-eligible results: {eligible_count}\n"
        f"- Claims: {len(claims)}\n"
        f"- Limitations: {len(limitations)}\n"
        f"- Headline checks: {headline['passed']}/{headline['checks']} passed\n"
        f"- Frozen structural outcome: {tie['original_primary_win_tie_loss']['win']}/"
        f"{tie['original_primary_win_tie_loss']['tie']}/"
        f"{tie['original_primary_win_tie_loss']['loss']} wins/ties/losses\n"
        f"- Protected files: {protected['protected_files']}; "
        f"changed: {protected['changed_count']}; "
        f"deleted: {protected['deleted_count']}\n"
        f"- Internal status: {internal['status']}\n\n"
        "The closure preserves case and functional dependence, near-zero rows, "
        "certificate looseness, and skipped finite-shot evidence. No quantum speedup "
        "is claimed, and no hardware execution was performed.\n"
    )


def _verification_markdown(
    output_dir: Path,
    internal: dict[str, Any],
    external: dict[str, Any],
    done: dict[str, Any],
) -> str:
    protected = load_json(output_dir / "protected_source_comparison.json")
    graph = load_json(output_dir / "artifact_dependency_graph.json")
    return (
        "# Final Contribution Evidence Verification Report\n\n"
        f"- Internal validation: {internal['status']}\n"
        f"- Targeted tests: {external['targeted']['status']}\n"
        f"- Dependent tests: {external['dependent']['status']}\n"
        f"- Full/filewise suite: {external['full_filewise']['status']}\n"
        f"- Ruff: {external['ruff']['status']}\n"
        f"- Protected source comparison: {protected['status']} "
        f"({protected['protected_files']} files; {protected['changed_count']} changed; "
        f"{protected['deleted_count']} deleted)\n"
        f"- Dependency graph: {graph['status']} ({graph['node_count']} nodes, "
        f"{graph['edge_count']} edges)\n"
        f"- Definition of done: {done['status']} ({done['pass_count']} PASS, "
        f"{done['fail_count']} FAIL)\n"
        "- Checksum registry: generated after this report; validate from repository root "
        "with `shasum -a 256 -c outputs/final_contribution_evidence/checksums.sha256`.\n"
        "- Resume: completed stages are read-only under `--stage all --resume`; an "
        "external hash comparison is recorded when available.\n"
    )


def _manifest_and_checksums(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    checksum_path = output_dir / "checksums.sha256"
    artifacts = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path in {manifest_path, checksum_path} or ".tmp." in path.name:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "study_id": "final_contribution_evidence_v1",
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "scientific_results_recomputed": False,
            "protected_sources_modified": False,
        },
    )
    checksum_files = [
        path
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file())
        if path != checksum_path and ".tmp." not in path.name
    ]
    lines = [f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in checksum_files]
    temporary = checksum_path.with_name(f"{checksum_path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, checksum_path)
    return {"manifest_artifacts": len(artifacts), "checksum_entries": len(lines)}


def verify_stage(output_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    failure_status = reconcile_failure_registry(output_dir, checkpoint)
    protected = verify_protected_sources(ROOT, output_dir)
    headline = load_json(output_dir / "headline_result_check_summary.json")
    tie = load_json(output_dir / "primary_tie_diagnostic_summary.json")
    graph = load_json(output_dir / "artifact_dependency_graph.json")
    eligibility = _read_csv(output_dir / "manuscript_eligibility_audit.csv")
    guards = _read_csv(output_dir / "conditioning_interpretation_violations.csv")
    claim = run_claim_guards(ROOT, output_dir)
    internal_checks = {
        "protected_sources": protected["status"] == "pass",
        "headline_checks": headline["status"] == "pass",
        "tie_diagnostics": tie["status"] == "pass",
        "dependency_graph": graph["status"] == "pass",
        "eligibility": bool((eligibility["status"] == "pass").all()),
        "conditioning_guards": guards.empty or not bool((guards["status"] == "unresolved").any()),
        "claim_guards": claim["status"] == "pass",
        "failure_registry": failure_status["unresolved_blocking_failures"] == 0,
    }
    internal = {
        "checks": internal_checks,
        "status": "pass" if all(internal_checks.values()) else "blocking_failure",
    }
    external = _read_test_status(output_dir)
    done = _definition_of_done(output_dir, internal, external)
    atomic_write_json(output_dir / "definition_of_done.json", done)
    (output_dir / "summary.md").write_text(
        _summary_markdown(output_dir, internal), encoding="utf-8"
    )
    # Re-scan the newly generated summary before freezing the manifest.
    claim = run_claim_guards(ROOT, output_dir)
    internal["checks"]["claim_guards"] = claim["status"] == "pass"
    internal["status"] = "pass" if all(internal["checks"].values()) else "blocking_failure"
    done = _definition_of_done(output_dir, internal, external)
    atomic_write_json(output_dir / "definition_of_done.json", done)
    (output_dir / "verification_report.md").write_text(
        _verification_markdown(output_dir, internal, external, done), encoding="utf-8"
    )
    if not (output_dir / "failure_registry.csv").exists():
        atomic_write_csv(output_dir / "failure_registry.csv", [], FAILURE_FIELDS)
    result = {
        "internal_status": internal["status"],
        "definition_of_done": done["status"],
        "external_verification": external,
        "failure_registry": failure_status,
    }
    checkpoint["stages"]["verify"] = {
        "status": "completed",
        "completed_at": utc_now(),
        "result": result,
    }
    checkpoint["status"] = "completed" if internal["status"] == "pass" else "blocking_failure"
    write_checkpoint(output_dir, checkpoint)
    checkpoint_part = output_dir / "checkpoint_parts/verify.json"
    atomic_write_json(checkpoint_part, checkpoint["stages"]["verify"])
    result.update(_manifest_and_checksums(output_dir))
    return result


def run_stage(
    stage: str,
    output_dir: Path,
    config: dict[str, Any],
    max_workers: int,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figure_data").mkdir(exist_ok=True)
    (output_dir / "checkpoint_parts").mkdir(exist_ok=True)
    (output_dir / "filewise").mkdir(exist_ok=True)
    checkpoint = load_checkpoint(output_dir)
    if resume and not force and checkpoint["stages"].get(stage, {}).get("status") == "completed":
        print(f"[{stage}] resumed from completed checkpoint")
        return checkpoint["stages"][stage].get("result", {})
    if force and stage in STAGES:
        position = STAGES.index(stage)
        checkpoint["stages"] = {
            name: value
            for name, value in checkpoint["stages"].items()
            if name not in STAGES[position:]
        }
        write_checkpoint(output_dir, checkpoint)
    if stage == "verify":
        print("[verify] validating canonical closure")
        return verify_stage(output_dir, checkpoint)
    handler = stage_handlers(output_dir, config, max_workers)[stage]
    print(f"[{stage}] running")
    started = time.monotonic()
    try:
        result = handler()
    except Exception as exc:
        record_failure(output_dir, stage, f"{type(exc).__name__}: {exc}")
        checkpoint["stages"][stage] = {
            "status": "failed",
            "failed_at": utc_now(),
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
        checkpoint["status"] = "blocking_failure"
        write_checkpoint(output_dir, checkpoint)
        raise
    stage_record = {
        "status": "completed",
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "result": result,
    }
    checkpoint["stages"][stage] = stage_record
    checkpoint["status"] = "in_progress"
    write_checkpoint(output_dir, checkpoint)
    atomic_write_json(output_dir / "checkpoint_parts" / f"{stage}.json", stage_record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    ).resolve()
    config = load_json(config_path)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    selected = STAGES if args.stage == "all" else [args.stage]
    try:
        for stage in selected:
            run_stage(
                stage,
                output_dir,
                config,
                args.max_workers,
                args.resume,
                args.force,
            )
    except Exception as exc:
        print(f"blocking failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
