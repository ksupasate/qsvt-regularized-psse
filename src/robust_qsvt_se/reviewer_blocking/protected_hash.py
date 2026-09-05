"""Compact hash snapshots of protected evidence roots (before/after guard).

Each protected root is summarised by file count, total bytes, and a single combined SHA-256 over
the sorted ``(relative_path, file_sha256)`` list.  A curated set of critical existing source /
config files is hashed individually so any accidental edit to a protected solver / QSVT / phase /
reg-core / state-metadata file is caught byte-for-byte.  Identical format to the pre-implementation
snapshot so before/after comparison is exact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from robust_qsvt_se.reviewer_blocking.common import REPO_ROOT, sha256_file

PROTECTED_ROOTS = (
    "manuscript",
    "submission_package_tqe_final",
    "outputs/final_contribution_evidence",
    "outputs/final_falsification_and_submission",
    "outputs/sparse_integrated_chain",
    "outputs/sparse_error_precision_study",
    "outputs/output_aware_sparse_selection",
    "outputs/output_aware_generalization",
    "outputs/output_aware_structural_generalization",
    "outputs/tqe_blocking_revision",
    "outputs/tqe_implementation_revision",
)

PROTECTED_FILES = (
    "src/robust_qsvt_se/qsvt/output_aware_sparse_selection.py",
    "src/robust_qsvt_se/qsvt/output_aware_generalization.py",
    "src/robust_qsvt_se/qsvt/output_aware_structural_generalization.py",
    "src/robust_qsvt_se/qsvt/sparse_integrated_chain.py",
    "src/robust_qsvt_se/qsvt/sparse_error_precision_study.py",
    "src/robust_qsvt_se/qsvt/ridge_output_certificate.py",
    "src/robust_qsvt_se/qsvt/rectangular_convention.py",
    "src/robust_qsvt_se/qsvt/state_metadata.py",
    "src/robust_qsvt_se/qsvt/phase_synthesis.py",
    "src/robust_qsvt_se/estimators/ridge.py",
    "src/robust_qsvt_se/paper/tqe_revision_core.py",
    "src/robust_qsvt_se/paper/selected_observable_qsvt_common.py",
    "src/robust_qsvt_se/experiments/tqe_revision_evidence.py",
    "src/robust_qsvt_se/measurement/ac_linear.py",
    "src/robust_qsvt_se/qsvt/engineering_utils.py",
    "configs/output_aware_sparse_selection.json",
    "configs/output_aware_generalization.json",
    "configs/output_aware_structural_generalization.json",
    "configs/sparse_error_precision_study.json",
)


def snapshot(root: Path = REPO_ROOT) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for relative_root in PROTECTED_ROOTS:
        absolute = root / relative_root
        if not absolute.exists():
            roots[relative_root] = {"exists": False}
            continue
        files = sorted(p for p in absolute.rglob("*") if p.is_file())
        combined = hashlib.sha256()
        total_bytes = 0
        for path in files:
            rel = str(path.relative_to(root))
            combined.update(rel.encode("utf-8"))
            combined.update(b"\0")
            combined.update(sha256_file(path).encode("utf-8"))
            combined.update(b"\n")
            total_bytes += path.stat().st_size
        roots[relative_root] = {
            "exists": True,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "combined_sha256": combined.hexdigest(),
        }
    individual = {
        relative_file: (
            sha256_file(root / relative_file)
            if (root / relative_file).is_file()
            else "MISSING"
        )
        for relative_file in PROTECTED_FILES
    }
    return {"roots": roots, "critical_source_files": individual}


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed_roots: list[str] = []
    for name, record in after["roots"].items():
        prior = before["roots"].get(name, {})
        if prior.get("combined_sha256") != record.get("combined_sha256"):
            changed_roots.append(name)
    changed_files = [
        name
        for name, digest in after["critical_source_files"].items()
        if before["critical_source_files"].get(name) != digest
    ]
    return {
        "changed_roots": sorted(changed_roots),
        "changed_critical_files": sorted(changed_files),
        "unchanged": not changed_roots and not changed_files,
    }
