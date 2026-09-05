"""Identity-verified in-memory rebuilds of the frozen generic sparse-QSVT workloads.

Reused by the fault-tolerant logical resource estimate and the noise-model boundary study.
Nothing is written into the frozen ``outputs/generic_sparse_qsvt_compiler/`` directory: the
frozen CSVs are read for identity verification only, and every rebuilt workload must match its
frozen ``workload_digest`` (and, where recorded, its frozen final-circuit hash) exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompiledSparseQSVT,
    compile_from_bundle,
)
from robust_qsvt_se.qsvt.generic_sparse_scaling import (
    _balanced_magnitude_support,
    _bundle_for_matrix,
)
from robust_qsvt_se.qsvt.generic_sparse_workloads import (
    build_canonical_compiler_inputs,
    build_second_ieee30_compiler_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_EVIDENCE = REPO_ROOT / "outputs" / "generic_sparse_qsvt_compiler"

CANONICAL_ID = "ieee14_sparse_quantized_8x8_d31_selected_v1"
SECOND_ID = "ieee30_sparse_quantized_8x8_d31_selected_v1"
ANCHOR_4X4_ID = "ieee14_sparse_quantized_4x4_d31_scaling_anchor_v1"
SCALING_16X16_ID = "ieee14_sparse_quantized_16x16_d31_scaling_v1"


def _read_rows(name: str) -> list[dict[str, str]]:
    with (FROZEN_EVIDENCE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def frozen_ledger_row(workload_id: str) -> dict[str, str]:
    """The frozen resource row (identity targets + displayed ledger values)."""

    sources = {
        CANONICAL_ID: "canonical_resource_ledger_generic.csv",
        SECOND_ID: "second_workload_resource_ledger.csv",
        ANCHOR_4X4_ID: "dimension_scaling.csv",
        SCALING_16X16_ID: "dimension_scaling.csv",
    }
    rows = _read_rows(sources[workload_id])
    matches = [row for row in rows if row.get("workload_id") == workload_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen row for {workload_id}; found {len(matches)}")
    return matches[0]


def verify_workload_identity(compiled: CompiledSparseQSVT, frozen: dict[str, str]) -> None:
    """Fail loudly if the rebuilt workload differs from its frozen registration."""

    if compiled.workload_digest != frozen["workload_digest"]:
        raise RuntimeError(
            f"{compiled.workload_id}: rebuilt workload digest "
            f"{compiled.workload_digest} != frozen {frozen['workload_digest']}"
        )
    if frozen.get("matrix_hash"):
        actual = compiled.component_hashes["matrix_original"]
        if actual != frozen["matrix_hash"]:
            raise RuntimeError(f"{compiled.workload_id}: matrix hash drift ({actual})")
    frozen_circuit_hash = frozen.get("source_circuit_hash", "")
    if frozen_circuit_hash:
        primary = compiled.functional_spec.primary_functional_id
        actual = compiled.component_hashes[f"source_final_circuit:{primary}"]
        if actual != frozen_circuit_hash:
            raise RuntimeError(
                f"{compiled.workload_id}: final circuit hash {actual} != frozen "
                f"{frozen_circuit_hash}"
            )


def rebuild_canonical() -> CompiledSparseQSVT:
    compiled = compile_from_bundle(build_canonical_compiler_inputs())
    verify_workload_identity(compiled, frozen_ledger_row(CANONICAL_ID))
    return compiled


def rebuild_second() -> CompiledSparseQSVT:
    compiled = compile_from_bundle(build_second_ieee30_compiler_inputs())
    verify_workload_identity(compiled, frozen_ledger_row(SECOND_ID))
    return compiled


def rebuild_dimension_anchor_4x4(canonical: CompiledSparseQSVT) -> CompiledSparseQSVT:
    """The smallest executed circuit: frozen IEEE-14 4x4 anchor, canonical phases, d=31."""

    matrix = np.load(
        REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_4x4.npy"
    )
    residual = np.load(
        REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_residual_ieee14_4x4.npy"
    )
    support = _balanced_magnitude_support(matrix, budget=12, slots=3)
    bundle = _bundle_for_matrix(
        canonical,
        matrix=matrix,
        residual=residual,
        coordinates=support,
        matrix_id="ieee14_primary_4x4_anchor",
        workload_id=ANCHOR_4X4_ID,
        source="outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_4x4.npy",
    )
    compiled = compile_from_bundle(bundle)
    verify_workload_identity(compiled, frozen_ledger_row(ANCHOR_4X4_ID))
    return compiled


def rebuild_dimension_16x16(canonical: CompiledSparseQSVT) -> CompiledSparseQSVT:
    """The transpile-only 16x16 dimension row, rebuilt exactly as in the frozen scaling study."""

    import json

    matrix = np.load(
        REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_16x16.npy"
    )
    residual = np.load(
        REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_residual_ieee14_16x16.npy"
    )
    payload = json.loads(
        (
            REPO_ROOT
            / "outputs/cross_case_larger_block_validation/larger_block_16x16/support_paths.json"
        ).read_text(encoding="utf-8")
    )
    support_id = "ieee14_block_16x16_seed123_sensitivity_refined_mean_k16_s3"
    support = tuple(tuple(int(v) for v in pair) for pair in payload[support_id])
    bundle = _bundle_for_matrix(
        canonical,
        matrix=matrix,
        residual=residual,
        coordinates=support,
        matrix_id="ieee14_frozen_16x16_block",
        workload_id=SCALING_16X16_ID,
        source="outputs/cross_case_larger_block_validation/larger_block_16x16/block_inventory.json",
    )
    compiled = compile_from_bundle(bundle)
    verify_workload_identity(compiled, frozen_ledger_row(SCALING_16X16_ID))
    return compiled


__all__ = [
    "ANCHOR_4X4_ID",
    "CANONICAL_ID",
    "SCALING_16X16_ID",
    "SECOND_ID",
    "frozen_ledger_row",
    "rebuild_canonical",
    "rebuild_dimension_16x16",
    "rebuild_dimension_anchor_4x4",
    "rebuild_second",
    "verify_workload_identity",
]
