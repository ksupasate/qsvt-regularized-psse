"""Case- and block-size-parameterized scaffolding for the transfer study.

Generalizes the three IEEE-14 ``8x8`` couplings in the frozen reviewer-blocking modules:

* :func:`build_case_block_binding` — physical block binding for any ``(case, size, seed)``
  (generalizes ``reviewer_blocking.common.build_physical_block_binding``);
* :func:`generate_case_residual` / :func:`build_case_tasks` — case-specific controlled residuals
  (generalize ``output_aware_sparse_selection.generate_controlled_residual``, which hardcodes
  IEEE-14);
* :func:`build_case_design` / :func:`build_case_full_system` — deterministic block design and full
  system for any case (generalize ``exact_loss_baselines.build_small_design`` and
  ``joint_feasibility.build_full_system``).

When ``case_name="ieee14"`` and the ``8x8`` block, every function reproduces the frozen behavior
byte-for-byte (asserted by regression tests).  Physical functionals, selectors, exact-loss
baselines, QSVT feasibility, and the cost model are reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from robust_qsvt_se.reviewer_blocking.common import (
    BlockColumnRecord,
    PhysicalBlockBinding,
)
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import SmallDesign
from robust_qsvt_se.reviewer_blocking.joint_feasibility import FullSystem
from robust_qsvt_se.reviewer_blocking.physical_functionals import (
    FunctionalRecord,
    UnavailableFunctional,
    build_all_physical_functionals,
)

SINGULAR_TOLERANCE = 1.0e-10


def json_safe_floats(value: Any) -> Any:
    """Recursively replace non-finite floats (inf/nan) with strings for strict JSON writing.

    ``atomic_write_json`` uses ``allow_nan=False``; rank-deficient blocks legitimately produce
    ``inf`` raw condition numbers, which must round-trip as ``"inf"`` rather than crash.
    """

    if isinstance(value, dict):
        return {key: json_safe_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_floats(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return value


# --------------------------------------------------------------- system cache


@lru_cache(maxsize=64)
def _cached_system(case_name: str, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one weighted AC-Jacobian system; cached by ``(case, seed)`` to avoid rebuilds."""

    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system

    system, _source = build_engineering_system(
        {
            "case_name": case_name,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    residual = np.asarray(system.r_tilde, dtype=np.float64)
    x_true = None if system.x_true is None else np.asarray(system.x_true, dtype=np.float64)
    metadata = dict(system.metadata)
    metadata["_x_true"] = x_true
    return matrix, residual, metadata


# --------------------------------------------------------------- block binding


def build_case_block_binding(
    case_name: str, seed: int = 123, *, row_count: int = 8, col_count: int = 8
) -> PhysicalBlockBinding:
    """Deterministic block + physical-state binding for any IEEE case and block size.

    Case-parameterized generalization of ``reviewer_blocking.common.build_physical_block_binding``
    (which hardcodes ``ieee14``).  The block is the identical outcome-independent
    ``largest_row_col_norms`` selection; each column is bound to its physical state, and real
    branches with both angle endpoints in-block are recorded for functional representability.
    """

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
    from robust_qsvt_se.measurement.ac_linear import load_ac_case
    from robust_qsvt_se.qsvt.state_metadata import (
        build_state_metadata_from_system_metadata,
    )

    matrix, residual, metadata = _cached_system(case_name, int(seed))
    _block, _rblock, rows, cols = select_deterministic_block(
        matrix, residual, row_count=row_count, col_count=col_count
    )
    meta = build_state_metadata_from_system_metadata(metadata)
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
    case = load_ac_case(case_name, case_source="pypower")
    branches = tuple((int(b.from_bus), int(b.to_bus)) for b in case.branches)
    angle_buses_in_block = {rec.bus_id for rec in columns if rec.state_type == "angle"}
    representable = tuple(
        (from_bus, to_bus)
        for from_bus, to_bus in branches
        if from_bus in angle_buses_in_block and to_bus in angle_buses_in_block
    )
    labels = tuple(str(metadata.get("measurement_labels", [])[int(r)]) for r in rows)
    types = tuple(str(metadata.get("measurement_types", [])[int(r)]) for r in rows)
    buses = tuple(
        metadata.get("measurement_buses", [None] * matrix.shape[0])[int(r)] for r in rows
    )
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
        weak_area_buses=tuple(int(b) for b in (metadata.get("weak_area_buses") or [])),
        slack_bus=int(metadata.get("slack_bus", -1)),
    )


# --------------------------------------------------------------- conditioning


def block_conditioning(matrix: np.ndarray, *, alpha_probe: float = 0.069) -> dict[str, Any]:
    """Raw and regularized normal-system conditioning diagnostics of a block."""

    block = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(block, compute_uv=False)
    positive = singular[singular > SINGULAR_TOLERANCE]
    rank = int(np.linalg.matrix_rank(block))
    raw_kappa = (
        float(singular[0] / singular[-1]) if singular[-1] > SINGULAR_TOLERANCE else float("inf")
    )
    gram = block.T @ block + float(alpha_probe) * np.eye(block.shape[1])
    gram_sv = np.linalg.svd(gram, compute_uv=False)
    reg_kappa = float(gram_sv[0] / gram_sv[-1])
    nnz = int(np.count_nonzero(np.abs(block) > 1.0e-12))
    return {
        "rows": int(block.shape[0]),
        "cols": int(block.shape[1]),
        "rank": rank,
        "full_rank": bool(rank == min(block.shape)),
        "raw_condition_number": raw_kappa,
        "spectral_norm": float(singular[0]) if singular.size else 0.0,
        "min_singular_value": float(singular[-1]) if singular.size else 0.0,
        "min_positive_singular_value": float(positive.min()) if positive.size else 0.0,
        "regularized_normal_system_condition_number": reg_kappa,
        "regularization_probe_alpha": float(alpha_probe),
        "nonzeros": nnz,
        "density": float(nnz / block.size) if block.size else 0.0,
    }


# --------------------------------------------------------------- design


@dataclass(slots=True)
class CaseDesign:
    """A deterministic block design plus its physical-functional binding and inventory."""

    small: SmallDesign
    binding: PhysicalBlockBinding
    functional_records: list[FunctionalRecord]
    unavailable: list[UnavailableFunctional]
    physical_functional_ids: tuple[str, ...]
    legacy_functional_ids: tuple[str, ...]
    conditioning: dict[str, Any]


def build_case_design(
    case_name: str, seed: int = 123, *, dimension: int = 8
) -> CaseDesign:
    """Deterministic ``dimension x dimension`` block for ``case_name`` with physical functionals.

    ``alpha = 4 * sigma_min_pos^2`` (identical to ``exact_loss_baselines.build_small_design``).
    ``small.functionals`` carries *all* functionals (physical families + the 3 legacy generic
    ones); the physical subset drives output-aware scoring, the legacy subset anchors the
    identical-semantics regression test.
    """

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block

    matrix_full, residual_full, _metadata = _cached_system(case_name, int(seed))
    block, _rblock, rows, cols = select_deterministic_block(
        matrix_full, residual_full, row_count=dimension, col_count=dimension
    )
    block = np.asarray(block, dtype=np.float64)
    conditioning = block_conditioning(block)
    if conditioning["min_positive_singular_value"] <= 0.0:
        raise ValueError("block has no positive singular values")
    alpha = 4.0 * conditioning["min_positive_singular_value"] ** 2

    binding = build_case_block_binding(
        case_name, int(seed), row_count=dimension, col_count=dimension
    )
    records, unavailable = build_all_physical_functionals(binding)
    functionals = {
        rec.functional_id: np.asarray(rec.vector, dtype=np.float64) for rec in records
    }
    physical_ids = tuple(
        rec.functional_id for rec in records if rec.family != "legacy_predetermined"
    )
    legacy_ids = tuple(
        rec.functional_id for rec in records if rec.family == "legacy_predetermined"
    )
    small = SmallDesign(
        label=f"{case_name}_block_{dimension}x{dimension}_seed{seed}",
        dimension=int(dimension),
        matrix=block,
        alpha=float(alpha),
        selected_rows=tuple(int(v) for v in rows),
        selected_columns=tuple(int(v) for v in cols),
        functionals=functionals,
    )
    return CaseDesign(
        small=small,
        binding=binding,
        functional_records=records,
        unavailable=unavailable,
        physical_functional_ids=physical_ids,
        legacy_functional_ids=legacy_ids,
        conditioning=conditioning,
    )


# --------------------------------------------------------------- residual / tasks


def generate_case_residual(
    case_name: str, seed_id: int, selected_rows: tuple[int, ...]
) -> np.ndarray:
    """Case-specific controlled residual ``r_tilde[selected_rows]`` for one seed.

    Generalizes ``output_aware_sparse_selection.generate_controlled_residual`` (hardcoded to
    IEEE-14).  Identical code path for ``ieee14``; the seed drives the AC linearization
    perturbation of the case-specific system.
    """

    _matrix, residual, _metadata = _cached_system(case_name, int(seed_id))
    rows = np.asarray(selected_rows, dtype=np.int64)
    sliced = residual[rows]
    if sliced.shape != (len(rows),) or not np.all(np.isfinite(sliced)):
        raise RuntimeError("controlled residual generation returned invalid values")
    if np.linalg.norm(sliced) <= 1.0e-15:
        raise RuntimeError("controlled residual generation returned a zero residual")
    return np.asarray(sliced, dtype=np.float64)


def build_case_tasks(
    case_name: str,
    design: SmallDesign,
    seeds: list[int],
    split: str,
    functional_ids: tuple[str, ...],
):
    """Case-specific analogue of ``exact_loss_baselines.build_tasks`` (per-seed residual reuse)."""

    from robust_qsvt_se.qsvt.output_aware_sparse_selection import RidgeTask

    tasks: list[RidgeTask] = []
    for seed in seeds:
        residual = generate_case_residual(case_name, int(seed), design.selected_rows)
        for functional_id in functional_ids:
            tasks.append(
                RidgeTask(
                    task_id=f"{split}_seed{seed}_{functional_id}",
                    seed_id=int(seed),
                    split=split,
                    residual=np.asarray(residual, dtype=np.float64),
                    functional_id=functional_id,
                    functional=np.asarray(design.functionals[functional_id], dtype=np.float64),
                )
            )
    return tasks


# --------------------------------------------------------------- full system


def build_case_full_system(case_name: str, seed: int = 123) -> FullSystem:
    """Full weighted AC subsystem for any case.

    Generalizes ``joint_feasibility.build_full_system``.
    """

    matrix, residual, metadata = _cached_system(case_name, int(seed))
    x_true = metadata.get("_x_true")
    if x_true is None:
        raise ValueError("full system is missing ground-truth state")
    return FullSystem(
        matrix=np.asarray(matrix, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        x_true=np.asarray(x_true, dtype=np.float64),
        angle_indices=tuple(int(i) for i in metadata.get("angle_state_indices", [])),
        voltage_indices=tuple(
            int(i) for i in metadata.get("voltage_magnitude_state_indices", [])
        ),
    )
