"""Read-only binding of frozen structural groups to physical truth/functionals."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.data.cases import load_ac_case
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint
from robust_qsvt_se.reviewer_blocking.common import BlockColumnRecord, PhysicalBlockBinding
from robust_qsvt_se.reviewer_blocking.physical_functionals import (
    build_all_physical_functionals,
)

UNIT_NORM_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StructuralInstance:
    instance_id: str
    structural_group_id: str
    ieee_case: str
    realization_order: int
    matrix_seed: int
    matrix: np.ndarray
    matrix_fingerprint: str
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    state_metadata: tuple[dict[str, Any], ...]
    measurement_metadata: tuple[dict[str, Any], ...]
    alpha: float
    candidate_id: str
    candidate_policy: str


@dataclass(frozen=True, slots=True)
class FunctionalSpec:
    functional_id: str
    family: str
    classification: str
    vector: np.ndarray | None
    status: str
    unavailable_reason: str
    state_type: str
    local_indices: tuple[int, ...]
    global_state_indices: tuple[int, ...]
    bus_ids: tuple[int, ...]
    area_members: tuple[int, ...]
    branch_id: str
    label: str
    selection_policy: str
    selection_inputs: str
    source_builder: str


@dataclass(frozen=True, slots=True)
class ControlledReference:
    full_matrix: np.ndarray
    full_residual: np.ndarray
    delta_true: np.ndarray
    independently_reconstructed_delta: np.ndarray
    reconstruction_max_abs_error: float
    metadata: dict[str, Any]


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise ValueError(f"cannot parse JSON list from {type(value).__name__}")


def _descriptor_key(row: Mapping[str, Any]) -> str:
    payload = {
        "case": row["ieee_case"],
        "rows": _parse_json_list(row["selected_rows"]),
        "columns": _parse_json_list(row["selected_columns"]),
        "support_pattern_fingerprint": row["support_pattern_fingerprint"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_registries(
    source_root: str | Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(source_root)
    settings = config["structure_design"]
    groups = pd.read_csv(root / settings["registry"])
    instances = pd.read_csv(root / settings["instance_registry"])
    if len(groups) != 12 or groups["structural_group_id"].nunique() != 12:
        raise RuntimeError("frozen structural registry must contain exactly 12 unique groups")
    expected_counts = {case: int(settings["groups_per_case"]) for case in config["cases"]}
    observed_counts = groups.groupby("ieee_case").size().to_dict()
    if observed_counts != expected_counts:
        raise RuntimeError(f"incorrect group counts: {observed_counts} != {expected_counts}")
    if bool(groups["selector_outcomes_used_for_selection"].astype(bool).any()):
        raise RuntimeError("outcome leakage is declared in structural group selection")
    if not bool(groups["status"].eq("included").all()):
        raise RuntimeError("primary structural registry contains a non-included group")
    groups = groups.copy()
    groups["row_column_support_descriptor"] = [
        _descriptor_key(row) for row in groups.to_dict(orient="records")
    ]
    if groups["row_column_support_descriptor"].nunique() != len(groups):
        raise RuntimeError("structural group row/column/support descriptors are not unique")

    if len(instances) != 24 or instances["instance_id"].nunique() != 24:
        raise RuntimeError("frozen instance registry must contain exactly 24 unique realizations")
    counts = instances.groupby("structural_group_id").size()
    if not bool((counts == int(settings["realizations_per_group"])).all()):
        raise RuntimeError("every structure must have exactly two numerical realizations")
    if bool(instances["selector_outcomes_used_for_inclusion"].astype(bool).any()):
        raise RuntimeError("outcome leakage is declared in realization inclusion")
    if not bool(instances["inclusion_status"].eq("included").all()):
        raise RuntimeError("instance registry contains a non-included primary realization")

    split_rows: list[dict[str, Any]] = []
    for row in instances.itertuples(index=False):
        split_path = root / settings["residual_splits_directory"] / f"{row.instance_id}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        training = [int(seed) for seed in payload["training_seed_ids"]]
        held_out = [int(seed) for seed in payload["held_out_seed_ids"]]
        if len(training) != int(settings["training_seed_count_per_instance"]):
            raise RuntimeError(f"wrong training seed count for {row.instance_id}")
        if len(held_out) != int(settings["held_out_seed_count_per_instance"]):
            raise RuntimeError(f"wrong held-out seed count for {row.instance_id}")
        overlap = sorted(set(training) & set(held_out))
        if overlap:
            raise RuntimeError(f"training/held-out seed overlap for {row.instance_id}: {overlap}")
        split_rows.append(
            {
                "instance_id": row.instance_id,
                "structural_group_id": row.structural_group_id,
                "ieee_case": row.ieee_case,
                "training_seed_ids": training,
                "held_out_seed_ids": held_out,
                "training_seed_count": len(training),
                "held_out_seed_count": len(held_out),
                "seed_overlap_count": 0,
                "declared_before_selector_evaluation": bool(
                    payload["declared_before_selector_evaluation"]
                ),
                "residual_fingerprints": payload["residual_fingerprints"],
                "split_path": str(split_path),
            }
        )
    return groups, instances, pd.DataFrame(split_rows)


def load_instance(source_root: str | Path, instance_id: str) -> StructuralInstance:
    path = Path(source_root) / "instances" / f"{instance_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    observed = stable_array_fingerprint(matrix)
    if observed != payload["matrix_fingerprint"]:
        raise RuntimeError(f"matrix fingerprint mismatch for {instance_id}")
    if payload["inclusion_status"] != "included":
        raise RuntimeError(f"attempted to load excluded instance {instance_id}")
    return StructuralInstance(
        instance_id=str(payload["instance_id"]),
        structural_group_id=str(payload["structural_group_id"]),
        ieee_case=str(payload["ieee_case"]),
        realization_order=int(payload["realization_order"]),
        matrix_seed=int(payload["matrix_selection_seed"]),
        matrix=matrix,
        matrix_fingerprint=observed,
        selected_rows=tuple(int(value) for value in payload["selected_rows"]),
        selected_columns=tuple(int(value) for value in payload["selected_columns"]),
        state_metadata=tuple(dict(row) for row in payload["state_metadata"]),
        measurement_metadata=tuple(dict(row) for row in payload["measurement_metadata"]),
        alpha=float(payload["regularization_alpha"]),
        candidate_id=str(payload["candidate_id"]),
        candidate_policy=str(payload["candidate_policy"]),
    )


@lru_cache(maxsize=2048)
def controlled_reference(case_name: str, seed: int) -> ControlledReference:
    system, _source = build_engineering_system(
        {
            "case_name": str(case_name),
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    metadata = dict(system.metadata)
    absolute_true = np.asarray(metadata.get("true_state"), dtype=np.float64)
    x0 = np.asarray(metadata.get("linearization_state"), dtype=np.float64)
    independently_reconstructed = absolute_true - x0
    stored = np.asarray(system.x_true, dtype=np.float64)
    if stored.shape != independently_reconstructed.shape:
        raise RuntimeError("stored and independently reconstructed truth shapes differ")
    max_error = float(np.max(np.abs(stored - independently_reconstructed)))
    if max_error > 1e-13:
        raise RuntimeError(f"stored truth is not x_true-x0: max error={max_error}")
    return ControlledReference(
        full_matrix=np.asarray(system.H_tilde, dtype=np.float64),
        full_residual=np.asarray(system.r_tilde, dtype=np.float64),
        delta_true=stored,
        independently_reconstructed_delta=independently_reconstructed,
        reconstruction_max_abs_error=max_error,
        metadata=metadata,
    )


def instance_residual_and_truth(
    instance: StructuralInstance, seed: int
) -> tuple[np.ndarray, np.ndarray, ControlledReference]:
    reference = controlled_reference(instance.ieee_case, int(seed))
    rows = np.asarray(instance.selected_rows, dtype=np.int64)
    columns = np.asarray(instance.selected_columns, dtype=np.int64)
    return (
        reference.full_residual[rows].copy(),
        reference.delta_true[columns].copy(),
        reference,
    )


def _physical_binding(instance: StructuralInstance) -> PhysicalBlockBinding:
    reference = controlled_reference(instance.ieee_case, instance.matrix_seed)
    case = load_ac_case(instance.ieee_case, case_source="pypower")
    branches = tuple((int(branch.from_bus), int(branch.to_bus)) for branch in case.branches)
    columns = tuple(
        BlockColumnRecord(
            local_index=int(row["local_index"]),
            global_state_index=int(row["full_state_index"]),
            state_type=str(row["state_type"]),
            bus_id=int(row["bus_id"]),
        )
        for row in instance.state_metadata
    )
    angle_buses = {row.bus_id for row in columns if row.state_type == "angle"}
    representable = tuple(
        (from_bus, to_bus)
        for from_bus, to_bus in branches
        if from_bus in angle_buses and to_bus in angle_buses
    )
    measurement_buses = reference.metadata.get("measurement_buses", [])
    return PhysicalBlockBinding(
        seed=instance.matrix_seed,
        row_count=len(instance.selected_rows),
        col_count=len(instance.selected_columns),
        selected_rows=instance.selected_rows,
        selected_columns=instance.selected_columns,
        state_dimension=len(reference.delta_true),
        columns=columns,
        branches=branches,
        representable_angle_branches=representable,
        measurement_labels=tuple(
            str(row["measurement_label"]) for row in instance.measurement_metadata
        ),
        measurement_types=tuple(
            str(row["measurement_type"]) for row in instance.measurement_metadata
        ),
        measurement_buses=tuple(measurement_buses[index] for index in instance.selected_rows),
        weak_area_buses=tuple(int(bus) for bus in reference.metadata.get("weak_area_buses", [])),
        slack_bus=int(reference.metadata.get("slack_bus", -1)),
    )


def _connected_components(
    buses: set[int], branches: tuple[tuple[int, int], ...]
) -> list[tuple[int, ...]]:
    adjacency = {bus: set() for bus in buses}
    for first, second in branches:
        if first in buses and second in buses:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(buses)
    components = []
    while remaining:
        start = min(remaining)
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        remaining -= seen
        components.append(tuple(sorted(seen)))
    return sorted(components, key=lambda item: (-len(item), item))


def _component_functionals(
    binding: PhysicalBlockBinding,
) -> tuple[list[FunctionalSpec], list[FunctionalSpec]]:
    available: list[FunctionalSpec] = []
    unavailable: list[FunctionalSpec] = []
    for state_type in ("angle", "voltage"):
        by_bus = {
            record.bus_id: record for record in binding.columns if record.state_type == state_type
        }
        components = [
            component
            for component in _connected_components(set(by_bus), binding.branches)
            if len(component) >= 2
        ]
        if not components:
            unavailable.append(
                FunctionalSpec(
                    functional_id=f"connected_{state_type}_component_aggregate",
                    family="connected_area_aggregate",
                    classification="unavailable_physical",
                    vector=None,
                    status="unavailable",
                    unavailable_reason=(
                        f"no connected component of at least two in-block {state_type} buses"
                    ),
                    state_type=state_type,
                    local_indices=(),
                    global_state_indices=(),
                    bus_ids=(),
                    area_members=(),
                    branch_id="",
                    label="UNAVAILABLE",
                    selection_policy="enumerate induced connected components with size >= 2",
                    selection_inputs="topology_and_state_index_metadata_only",
                    source_builder="physical_alignment_connected_component_extension",
                )
            )
            continue
        for component in components:
            vector = np.zeros(binding.col_count, dtype=np.float64)
            scale = 1.0 / math.sqrt(len(component))
            records = [by_bus[bus] for bus in component]
            for record in records:
                vector[record.local_index] = scale
            available.append(
                FunctionalSpec(
                    functional_id=(
                        f"connected_{state_type}_component_"
                        + "_".join(str(bus) for bus in component)
                        + "_aggregate"
                    ),
                    family="connected_area_aggregate",
                    classification="physical",
                    vector=vector,
                    status="available",
                    unavailable_reason="",
                    state_type=state_type,
                    local_indices=tuple(record.local_index for record in records),
                    global_state_indices=tuple(record.global_state_index for record in records),
                    bus_ids=component,
                    area_members=component,
                    branch_id="",
                    label=f"unit-norm aggregate over connected {state_type} buses {component}",
                    selection_policy="enumerate induced connected components with size >= 2",
                    selection_inputs="topology_and_state_index_metadata_only",
                    source_builder="physical_alignment_connected_component_extension",
                )
            )
    return available, unavailable


def build_instance_functionals(instance: StructuralInstance) -> list[FunctionalSpec]:
    binding = _physical_binding(instance)
    records, unavailable = build_all_physical_functionals(binding)
    specs: list[FunctionalSpec] = []
    existing_ids = set()
    existing_available_ids = set()
    for record in records:
        classification = (
            "legacy_diagnostic" if record.family == "legacy_predetermined" else "physical"
        )
        vector = np.asarray(record.vector, dtype=np.float64)
        norm_error = abs(float(np.linalg.norm(vector)) - 1.0)
        if norm_error > UNIT_NORM_TOLERANCE:
            raise RuntimeError(
                f"functional {record.functional_id} is not unit norm: error={norm_error}"
            )
        existing_ids.add(record.functional_id)
        existing_available_ids.add(record.functional_id)
        specs.append(
            FunctionalSpec(
                functional_id=record.functional_id,
                family=record.family,
                classification=classification,
                vector=vector,
                status="available",
                unavailable_reason="",
                state_type=record.state_type,
                local_indices=tuple(record.local_block_indices),
                global_state_indices=tuple(record.global_state_indices),
                bus_ids=tuple(record.global_bus_ids),
                area_members=tuple(record.area_members),
                branch_id=record.branch_id,
                label=record.label,
                selection_policy=record.deterministic_selection_policy,
                selection_inputs=record.selection_inputs,
                source_builder="reviewer_blocking.build_all_physical_functionals",
            )
        )
    for record in unavailable:
        existing_ids.add(record.requested_functional_id)
        specs.append(
            FunctionalSpec(
                functional_id=record.requested_functional_id,
                family=record.family,
                classification="unavailable_physical",
                vector=None,
                status="unavailable",
                unavailable_reason=record.reason_unavailable,
                state_type="",
                local_indices=(),
                global_state_indices=(),
                bus_ids=(),
                area_members=(),
                branch_id="",
                label="UNAVAILABLE",
                selection_policy=record.deterministic_selection_policy,
                selection_inputs="topology_and_state_index_metadata_only",
                source_builder="reviewer_blocking.build_all_physical_functionals",
            )
        )
    component_available, component_unavailable = _component_functionals(binding)
    for record in [*component_available, *component_unavailable]:
        full_area_id = f"connected_block_{record.state_type}_area_aggregate"
        if full_area_id in existing_available_ids:
            continue
        if record.functional_id not in existing_ids:
            specs.append(record)
    specs.sort(key=lambda row: (row.classification, row.family, row.functional_id))
    return specs


def functional_registry_rows(
    instance: StructuralInstance, functionals: list[FunctionalSpec]
) -> list[dict[str, Any]]:
    rows = []
    for functional in functionals:
        vector = functional.vector
        rows.append(
            {
                "instance_id": instance.instance_id,
                "structural_group_id": instance.structural_group_id,
                "ieee_case": instance.ieee_case,
                "realization_order": instance.realization_order,
                "functional_id": functional.functional_id,
                "functional_family": functional.family,
                "classification": functional.classification,
                "status": functional.status,
                "unavailable_reason": functional.unavailable_reason,
                "functional_vector": (
                    json.dumps(vector.tolist(), separators=(",", ":"))
                    if vector is not None
                    else None
                ),
                "functional_norm": float(np.linalg.norm(vector)) if vector is not None else None,
                "unit_norm_error": (
                    abs(float(np.linalg.norm(vector)) - 1.0) if vector is not None else None
                ),
                "state_type": functional.state_type,
                "local_indices": json.dumps(functional.local_indices),
                "global_state_indices": json.dumps(functional.global_state_indices),
                "bus_ids": json.dumps(functional.bus_ids),
                "area_members": json.dumps(functional.area_members),
                "branch_id": functional.branch_id,
                "label": functional.label,
                "deterministic_selection_policy": functional.selection_policy,
                "selection_inputs": functional.selection_inputs,
                "source_builder": functional.source_builder,
                "outcome_values_used_for_selection": False,
                "legacy_substitution_used": False,
            }
        )
    return rows
