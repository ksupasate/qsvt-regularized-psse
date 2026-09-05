"""Phase 10 WP F: consolidated implementation-completion index and verification.

Links every Phase 10 output package, records verification, scans all generated
text for forbidden claim wording, aggregates checksums, and states honestly what
remains unresolved after this task.  It reads the packages produced by WP A-E;
missing packages are recorded, never silently skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper.phase10_common import (
    PHASE10_CLAIM_BOUNDARY,
    forbidden_in,
    json_ready,
    sha256_file,
    write_phase10_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_DIR = Path("outputs/phase10_implementation_completion")

PHASE10_PACKAGES: tuple[dict[str, str], ...] = (
    {
        "work_package": "A",
        "name": "Complete 8x8 sparse block-encoding wrapper",
        "directory": "outputs/phase10_sparse_wrapper_8x8_complete",
        "experiment_id": "phase10_sparse_wrapper_8x8_complete",
        "status_summary": (
            "Phase 9 edge-coloring blocker fixed (deterministic slot assignment); complete 8x8 "
            "wrapper compiled and statevector-validated; QSVT integration matches dense dilation, "
            "exact SVT, and matched Ridge"
        ),
    },
    {
        "work_package": "B",
        "name": "Full rectangular PSSE selected-output QSVT (IEEE 14 and IEEE 30)",
        "directory": "outputs/phase10_full_rectangular_selected_output_qsvt",
        "experiment_id": "phase10_full_rectangular_selected_output_qsvt",
        "status_summary": (
            "Full rectangular A=H^T/beta executed on the statevector simulator for IEEE 14 "
            "(82x27) and IEEE 30 (172x59); degree-aware alpha tiers match full-system Ridge; "
            "canonical/sigma-matched tiers recorded degree-limited; IEEE 57/118/300 modeled"
        ),
    },
    {
        "work_package": "C",
        "name": "Residual loading and repeat-cost accounting",
        "directory": "outputs/phase10_residual_loading_accounting",
        "experiment_id": "phase10_residual_loading_accounting",
        "status_summary": (
            "Three explicit loading modes (dense Initialize, binary-tree Möttönen loader, QROM "
            "cost model) with per-attempt and nonlinear-loop repetition accounting"
        ),
    },
    {
        "work_package": "D",
        "name": "Nonlinear AC QSVT-in-the-loop (IEEE 14)",
        "directory": "outputs/phase10_nonlinear_qsvt_in_loop",
        "experiment_id": "phase10_nonlinear_qsvt_in_loop",
        "status_summary": (
            "Gauss-Newton AC loop with matrix-level and full-rectangular statevector QSVT "
            "updates; QSVT target matches Ridge at matched alpha per iteration; beta_k/lambda_k "
            "recomputed and residual/Jacobian rebuilt each iteration"
        ),
    },
    {
        "work_package": "E",
        "name": "End-to-end resource and classical comparator ledger",
        "directory": "outputs/phase10_end_to_end_resource_ledger",
        "experiment_id": "phase10_end_to_end_resource_ledger",
        "status_summary": (
            "Consolidated ledgers with explicit execution tiers; classical wall-clock and "
            "quantum query/T-count units kept separate; no competitiveness claim"
        ),
    },
)

UNRESOLVED_AFTER_PHASE10: tuple[str, ...] = (
    "No quantum-hardware run: every Phase 10 execution is a classical statevector or sampled-"
    "counts simulation.",
    "No field PMU/SCADA validation: IEEE/PYPOWER cases provide benchmark network models with "
    "generated measurement rows, not field data.",
    "No quantum speedup: the QSVT path implements the same Ridge/Tikhonov filter at matched "
    "alpha; no asymptotic or wall-clock advantage is demonstrated.",
    "No QSVT-over-Ridge numerical superiority at matched alpha: matched-alpha agreement is the "
    "pass criterion, not a superiority result.",
    "No practical competitiveness: classical selected-output and full Ridge solves are "
    "sub-millisecond for these cases; the quantum path is not competitive at these sizes.",
    "Larger IEEE-scale compiled circuits: IEEE 57/118/300 remain resource-estimated with "
    "polynomial certification only; their quantum side is not executed or compiled.",
    "Scalable residual loading: the binary-tree and QROM loaders are explicit but not "
    "asymptotically cheaper than dense loading; no scalable structured loader is compiled at "
    "IEEE scale.",
    "Full-vector recovery: selected-output readout is executed; full-vector recovery is "
    "accounted as ~ n * selected but not run coordinate-by-coordinate at scale.",
    "Canonical alpha = 1e-4 QSVT execution: the bounded-polynomial synthesis ceiling "
    "(degree <= 45, monomial basis) makes the smallest-alpha tiers degree-limited; only "
    "degree-aware alpha tiers execute and pass.",
    "IEEE-scale sparse block encoding: only the 8x8 sparse wrapper is compiled; a scalable "
    "sparse-oracle synthesis remains modeled.",
)


def _scan_forbidden(directory: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not directory.is_dir():
        return findings
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = forbidden_in(text)
        if hits:
            findings.append({"file": str(path), "forbidden_phrases": hits})
    return findings


def _package_record(package: dict[str, str]) -> dict[str, Any]:
    directory = Path(package["directory"])
    manifest_path = directory / "manifest.json"
    readme_path = directory / "README.md"
    checksums_path = directory / "checksums.sha256"
    present = directory.is_dir()
    artifacts = sorted(p.name for p in directory.iterdir() if p.is_file()) if present else []
    return {
        "work_package": package["work_package"],
        "name": package["name"],
        "directory": package["directory"],
        "experiment_id": package["experiment_id"],
        "status_summary": package["status_summary"],
        "present": present,
        "has_manifest": manifest_path.is_file(),
        "has_readme": readme_path.is_file(),
        "has_checksums": checksums_path.is_file(),
        "artifact_files": artifacts,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }


def run_phase10_completion_index(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "command": "scripts/run_phase10_implementation_completion.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    package_records = [_package_record(package) for package in PHASE10_PACKAGES]
    forbidden_findings: dict[str, list[dict[str, Any]]] = {}
    all_checksum_lines: list[str] = []
    for package in PHASE10_PACKAGES:
        directory = Path(package["directory"])
        findings = _scan_forbidden(directory)
        if findings:
            forbidden_findings[package["experiment_id"]] = findings
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.name != "phase_cache":
                    try:
                        rel = path.relative_to(output_dir.parent)
                    except ValueError:
                        rel = path
                    all_checksum_lines.append(f"{sha256_file(path)}  {rel}\n")

    all_present = all(record["present"] for record in package_records)
    all_have_manifest = all(record["has_manifest"] for record in package_records)
    claim_safe = not forbidden_findings

    index_json = output_dir / "phase10_index.json"
    summary_md = output_dir / "phase10_summary.md"
    unresolved_md = output_dir / "phase10_unresolved_after_completion.md"
    claim_report_txt = output_dir / "phase10_claim_safety_report.txt"
    all_checksums = output_dir / "phase10_all_checksums.sha256"

    index = {
        "phase": "phase10_implementation_completion",
        "all_packages_present": all_present,
        "all_packages_have_manifest": all_have_manifest,
        "claim_safe": claim_safe,
        "packages": package_records,
        "claim_boundary": PHASE10_CLAIM_BOUNDARY,
        "unresolved_after_phase10": list(UNRESOLVED_AFTER_PHASE10),
    }
    write_json(index_json, json_ready(index))
    summary_md.write_text(_summary_markdown(package_records, claim_safe), encoding="utf-8")
    unresolved_md.write_text(_unresolved_markdown(), encoding="utf-8")
    claim_report_txt.write_text(
        _claim_report(forbidden_findings, package_records), encoding="utf-8"
    )
    all_checksums.write_text("".join(sorted(all_checksum_lines)), encoding="utf-8")

    artifacts = {
        "phase10_index_json": index_json,
        "phase10_summary_md": summary_md,
        "phase10_unresolved_after_completion_md": unresolved_md,
        "phase10_claim_safety_report_txt": claim_report_txt,
        "phase10_all_checksums_sha256": all_checksums,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_implementation_completion",
        script_name="scripts/run_phase10_implementation_completion.py",
        command=str(resolved["command"]),
        description=(
            "Consolidated Phase 10 implementation-completion index: links every Phase 10 "
            "package, records verification and claim-safety, aggregates checksums, and lists "
            "the limitations that remain unresolved after this task."
        ),
        artifacts=artifacts,
        extra={
            "all_packages_present": all_present,
            "all_packages_have_manifest": all_have_manifest,
            "claim_safe": claim_safe,
            "package_count": len(package_records),
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "index": index,
        "package_records": package_records,
        "forbidden_findings": forbidden_findings,
        "claim_safe": claim_safe,
        "artifacts": artifacts,
    }


def _summary_markdown(package_records: list[dict[str, Any]], claim_safe: bool) -> str:
    lines = [
        "# Phase 10 Implementation Completion Summary",
        "",
        PHASE10_CLAIM_BOUNDARY,
        "",
        f"- All packages present: **{all(r['present'] for r in package_records)}**",
        f"- All packages have manifests: **{all(r['has_manifest'] for r in package_records)}**",
        f"- Claim-safe (no forbidden wording in generated text): **{claim_safe}**",
        "",
        "## Packages",
        "",
        "| WP | package | present | manifest | checksums | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in package_records:
        lines.append(
            f"| {record['work_package']} | {record['experiment_id']} | {record['present']} | "
            f"{record['has_manifest']} | {record['has_checksums']} | {record['status_summary']} |"
        )
    lines += [
        "",
        "See `phase10_unresolved_after_completion.md` for the limitations that remain after "
        "this task and `phase10_claim_safety_report.txt` for the forbidden-wording scan.",
        "",
    ]
    return "\n".join(lines)


def _unresolved_markdown() -> str:
    lines = [
        "# Phase 10: Unresolved After Completion",
        "",
        "Even after Phase 10 implementation, the following remain unresolved or explicitly "
        "not claimed. They are stated here so that no downstream reader over-reads the "
        "executed evidence.",
        "",
    ]
    for item in UNRESOLVED_AFTER_PHASE10:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _claim_report(
    forbidden_findings: dict[str, list[dict[str, Any]]], package_records: list[dict[str, Any]]
) -> str:
    lines = [
        "PHASE 10 CLAIM-SAFETY REPORT",
        "=" * 32,
        "",
        "Scan: forbidden-phrase check over all .md/.txt/.csv/.json in each Phase 10 package.",
        "",
    ]
    if not forbidden_findings:
        lines.append("RESULT: PASS - no forbidden claim wording found in any Phase 10 package.")
    else:
        lines.append("RESULT: FAIL - forbidden wording found:")
        for experiment_id, findings in forbidden_findings.items():
            lines.append(f"  {experiment_id}:")
            for finding in findings:
                lines.append(f"    {finding['file']}: {finding['forbidden_phrases']}")
    lines += [
        "",
        "Packages scanned:",
    ]
    for record in package_records:
        lines.append(
            f"  WP {record['work_package']} {record['experiment_id']}: "
            f"present={record['present']}, files={len(record['artifact_files'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 WP F: consolidated implementation-completion index"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    run = run_phase10_completion_index(
        {
            "output_dir": args.output_dir,
            "command": "scripts/run_phase10_implementation_completion.py " + " ".join(argv or []),
        }
    )
    print(f"Claim-safe: {run['claim_safe']}")
    for record in run["package_records"]:
        print(
            f"  WP {record['work_package']} {record['experiment_id']}: "
            f"present={record['present']} manifest={record['has_manifest']}"
        )
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
