"""Shared scaffolding for the reviewer-blocking experiments.

Atomic IO, provenance capture, deterministic array fingerprints (identical to
``sparse_integrated_chain.stable_array_fingerprint``), and the frozen 8x8 block ->
physical-state binding used by every workstream.  Kept import-light: the heavy
qiskit/pennylane stack is only imported by the workstreams that need it, never here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

CLAIM_BOUNDARY = (
    "Controlled 8x8 IEEE-14-derived numerical and classical-simulator evidence for "
    "QSVT-compatible implementation pathways of the same Ridge/Tikhonov regularized "
    "spectral filter. No quantum speedup, quantum advantage, QSVT-over-Ridge numerical "
    "superiority, field PMU/SCADA validation, IEEE-scale hardware execution, or "
    "practical-competitiveness claim is made."
)


# --------------------------------------------------------------------------- IO


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_fingerprint(values: np.ndarray) -> str:
    """SHA-256 of contiguous float64 bytes (matches ``stable_array_fingerprint``)."""

    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration JSON must contain an object")
    return payload


# --------------------------------------------------------------------- provenance


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode()
        return bool(output.strip())
    except Exception:
        return True


def environment_record() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "qiskit", "qiskit_aer", "pennylane"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not_installed"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_versions": versions,
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "single_thread_env": {
            variable: os.environ.get(variable)
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "MPLBACKEND",
            )
        },
    }


def provenance_block(config_path: str | Path, resolved_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "config_path": str(config_path),
        "resolved_config": json_ready(resolved_config),
        "environment": environment_record(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# --------------------------------------------------------- frozen block binding


@dataclass(frozen=True, slots=True)
class BlockColumnRecord:
    """One block column bound to a physical IEEE-14 state."""

    local_index: int
    global_state_index: int
    state_type: str  # "angle" or "voltage"
    bus_id: int


@dataclass(frozen=True, slots=True)
class PhysicalBlockBinding:
    seed: int
    row_count: int
    col_count: int
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    state_dimension: int
    columns: tuple[BlockColumnRecord, ...]
    branches: tuple[tuple[int, int], ...]
    representable_angle_branches: tuple[tuple[int, int], ...]
    measurement_labels: tuple[str, ...]
    measurement_types: tuple[str, ...]
    measurement_buses: tuple[Any, ...]
    weak_area_buses: tuple[int, ...]
    slack_bus: int

    def angle_columns(self) -> tuple[BlockColumnRecord, ...]:
        return tuple(record for record in self.columns if record.state_type == "angle")

    def voltage_columns(self) -> tuple[BlockColumnRecord, ...]:
        return tuple(record for record in self.columns if record.state_type == "voltage")

    def column_for(self, *, state_type: str, bus_id: int) -> BlockColumnRecord | None:
        for record in self.columns:
            if record.state_type == state_type and record.bus_id == bus_id:
                return record
        return None

    def bus_set(self, state_type: str) -> tuple[int, ...]:
        return tuple(
            record.bus_id for record in self.columns if record.state_type == state_type
        )


def bus_set_is_connected(
    buses: set[int], branches: tuple[tuple[int, int], ...]
) -> bool:
    """Return True iff ``buses`` induces a connected subgraph via real branches."""

    if len(buses) <= 1:
        return True
    adjacency: dict[int, set[int]] = {bus: set() for bus in buses}
    for from_bus, to_bus in branches:
        if from_bus in buses and to_bus in buses:
            adjacency[from_bus].add(to_bus)
            adjacency[to_bus].add(from_bus)
    start = next(iter(buses))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == buses


def build_physical_block_binding(
    seed: int = 123, *, row_count: int = 8, col_count: int = 8
) -> PhysicalBlockBinding:
    """Rebuild the deterministic block and bind each column to a physical state.

    Import-light: only touches the classical AC-Jacobian pathway (no qiskit).  The
    selected columns are the largest-column-norm state indices, an outcome-independent
    property of the weighted Jacobian, identical to the frozen campaign's selection.
    """

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
    from robust_qsvt_se.measurement.ac_linear import load_ac_case
    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
    from robust_qsvt_se.qsvt.state_metadata import (
        build_state_metadata_from_system_metadata,
    )

    system, _source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    residual = np.asarray(system.r_tilde, dtype=np.float64)
    _block, _rblock, rows, cols = select_deterministic_block(
        matrix, residual, row_count=row_count, col_count=col_count
    )
    meta = build_state_metadata_from_system_metadata(system.metadata)
    columns: list[BlockColumnRecord] = []
    for local_index, global_index in enumerate(int(value) for value in cols):
        record = meta.record_for_index(global_index)
        columns.append(
            BlockColumnRecord(
                local_index=local_index,
                global_state_index=global_index,
                state_type=str(record.state_type),
                bus_id=int(record.bus_id) if record.bus_id is not None else -1,
            )
        )
    case = load_ac_case("ieee14", case_source="pypower")
    branches = tuple(
        (int(branch.from_bus), int(branch.to_bus)) for branch in case.branches
    )
    angle_buses_in_block = {rec.bus_id for rec in columns if rec.state_type == "angle"}
    representable = tuple(
        (from_bus, to_bus)
        for from_bus, to_bus in branches
        if from_bus in angle_buses_in_block and to_bus in angle_buses_in_block
    )
    md = system.metadata
    labels = tuple(str(md.get("measurement_labels", [])[int(r)]) for r in rows)
    types = tuple(str(md.get("measurement_types", [])[int(r)]) for r in rows)
    buses = tuple(md.get("measurement_buses", [None] * matrix.shape[0])[int(r)] for r in rows)
    return PhysicalBlockBinding(
        seed=int(seed),
        row_count=int(row_count),
        col_count=int(col_count),
        selected_rows=tuple(int(value) for value in rows),
        selected_columns=tuple(int(value) for value in cols),
        state_dimension=int(meta.dimension),
        columns=tuple(columns),
        branches=branches,
        representable_angle_branches=representable,
        measurement_labels=labels,
        measurement_types=types,
        measurement_buses=buses,
        weak_area_buses=tuple(int(b) for b in (md.get("weak_area_buses") or [])),
        slack_bus=int(md.get("slack_bus", -1)),
    )


# --------------------------------------------------------- manifest / checksums


def write_manifest_and_checksums(
    root: str | Path,
    *,
    study_id: str,
    exclude_names: tuple[str, ...] = ("manifest.json", "checksums.sha256"),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` + ``checksums.sha256`` over a single output subtree."""

    base = Path(root)
    artifacts = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.name not in exclude_names
    )
    manifest = {
        "study_id": study_id,
        "generated_at": now_iso(),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "artifact_count": len(artifacts),
        "claim_boundary": CLAIM_BOUNDARY,
        "artifacts": [
            {
                "path": str(path.relative_to(base)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    if extra:
        manifest.update(json_ready(extra))
    atomic_write_json(base / "manifest.json", manifest)
    checksum_targets = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    text = (
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(base)}" for path in checksum_targets
        )
        + "\n"
    )
    checksum_path = base / "checksums.sha256"
    temporary = checksum_path.with_name(f"{checksum_path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, checksum_path)
    return manifest
