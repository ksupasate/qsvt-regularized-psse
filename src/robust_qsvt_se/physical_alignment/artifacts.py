"""Atomic evidence IO, provenance, manifests, and protected-hash verification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, na_rep="")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_provenance(root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "scipy",
        "pandas",
        "pypower",
        "PyYAML",
        "pytest",
        "qiskit",
        "qiskit-aer",
        "pennylane",
        "pyqsp",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repository_root": str(root.resolve()),
        "git_branch": git_output(root, "branch", "--show-current"),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_output(root, "status", "--porcelain")),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def protected_snapshot(root: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for relative_root in baseline["roots"]:
        absolute = root / relative_root
        if not absolute.exists():
            roots[relative_root] = {"exists": False, "file_count": 0, "total_bytes": 0}
            continue
        files = sorted(path for path in absolute.rglob("*") if path.is_file())
        combined = hashlib.sha256()
        total_bytes = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            combined.update(relative.encode("utf-8"))
            combined.update(b"\0")
            combined.update(sha256_file(path).encode("ascii"))
            combined.update(b"\n")
            total_bytes += path.stat().st_size
        roots[relative_root] = {
            "exists": True,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "combined_sha256": combined.hexdigest(),
        }
    critical = {
        relative: sha256_file(root / relative) if (root / relative).is_file() else "MISSING"
        for relative in baseline["critical_source_files"]
    }
    return {"roots": roots, "critical_source_files": critical}


def compare_protected_hashes(root: Path, baseline_path: str | Path) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    after = protected_snapshot(root, baseline)
    root_rows = []
    changed_roots = []
    for relative, before in baseline["roots"].items():
        current = after["roots"].get(relative, {})
        unchanged = before == current
        if not unchanged:
            changed_roots.append(relative)
        root_rows.append(
            {
                "path": relative,
                "unchanged": unchanged,
                "before": before,
                "after": current,
            }
        )
    critical_rows = []
    changed_critical = []
    for relative, before_digest in baseline["critical_source_files"].items():
        after_digest = after["critical_source_files"].get(relative)
        unchanged = before_digest == after_digest
        if not unchanged:
            changed_critical.append(relative)
        critical_rows.append(
            {
                "path": relative,
                "unchanged": unchanged,
                "before_sha256": before_digest,
                "after_sha256": after_digest,
            }
        )
    return {
        "status": "pass" if not changed_roots and not changed_critical else "fail",
        "all_protected_unchanged": not changed_roots,
        "all_critical_source_files_unchanged": not changed_critical,
        "changed_roots": changed_roots,
        "changed_critical_source_files": changed_critical,
        "root_comparisons": root_rows,
        "critical_source_comparisons": critical_rows,
        "baseline_path": str(baseline_path),
    }


def write_artifact_manifest(
    output_root: str | Path, *, configuration_id: str, configuration_hash: str
) -> dict[str, Any]:
    root = Path(output_root)
    manifest_path = root / "artifact_manifest.json"
    checksum_path = root / "checksums.sha256"
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    known = {path.relative_to(root).as_posix(): path for path in paths}
    known.setdefault("artifact_manifest.json", manifest_path)
    known.setdefault("checksums.sha256", checksum_path)
    rows = []
    for relative, path in sorted(known.items()):
        self_referential = relative in {"artifact_manifest.json", "checksums.sha256"}
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": None if self_referential else sha256_file(path),
                "status": (
                    "self_referential_checksum_exclusion" if self_referential else "generated"
                ),
            }
        )
    manifest = {
        "configuration_id": configuration_id,
        "configuration_hash": configuration_hash,
        "root": str(root),
        "file_count_including_manifest_and_checksums": len(rows),
        "files": rows,
        "checksum_policy": (
            "checksums.sha256 hashes every generated file except itself; artifact_manifest.json "
            "lists itself and checksums.sha256 with explicit self-referential exclusions"
        ),
    }
    atomic_write_json(manifest_path, manifest)
    checksum_targets = sorted(
        path for path in root.rglob("*") if path.is_file() and path != checksum_path
    )
    text = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_targets
    )
    atomic_write_text(checksum_path, text)
    manifest["checksum_entry_count"] = len(checksum_targets)
    return manifest


def validate_checksums(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    checksum_path = root / "checksums.sha256"
    failures = []
    entries = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        entries += 1
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "status": "pass" if not failures else "fail",
        "entries": entries,
        "failures": failures,
    }
