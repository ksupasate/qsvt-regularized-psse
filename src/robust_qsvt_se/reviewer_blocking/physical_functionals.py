"""Workstream 1 - physically meaningful selected-output functionals.

Deterministic, outcome-independent functionals derived from IEEE-14 network and state
metadata, defined on the frozen 8x8 block-column space (the eight largest-column-norm
state coordinates).  The three existing generic functionals are recorded unchanged for
backward compatibility; the physical families supplement them.

Every functional is selected using topology and state-index metadata only.  No solved
Ridge/QSVT output, residual, or evaluation outcome is consulted during selection, so a
favourable-output bias is structurally impossible.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    PhysicalBlockBinding,
    array_fingerprint,
    atomic_write_csv,
    atomic_write_json,
    build_physical_block_binding,
    bus_set_is_connected,
    load_config,
    now_iso,
    provenance_block,
    write_manifest_and_checksums,
)

STUDY_ID = "reviewer_blocking_physical_functionals_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_reviewer_blocking_experiments/physical_functionals")
DEFAULT_CONFIG_PATH = Path("configs/tqe_reviewer_blocking/physical_functionals.json")

UNIT_NORM_TOLERANCE = 1e-12

LEGACY_FUNCTIONAL_INTERPRETATION = {
    "coordinate_e0": "angle",
    "signed_difference_e0_minus_e1": "angle",
    "aggregate_e0_to_e3": "angle",
}


@dataclass(slots=True)
class FunctionalRecord:
    functional_id: str
    family: str
    label: str
    reason_physically_meaningful: str
    dimension: int
    vector: tuple[float, ...]
    local_block_indices: tuple[int, ...]
    global_state_indices: tuple[int, ...]
    global_bus_ids: tuple[int, ...]
    state_type: str
    branch_id: str
    area_members: tuple[int, ...]
    normalization: str
    normalization_constant: float
    unit_norm: bool
    unit_norm_error: float
    deterministic_selection_policy: str
    is_primary_for_policy: bool
    selection_inputs: str = "topology_and_state_index_metadata_only"
    output_values_available_during_selection: bool = False
    evidence_status: str = "classical_deterministic_definition"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["vector"] = json.dumps([float(v) for v in self.vector], separators=(",", ":"))
        row["local_block_indices"] = json.dumps(list(self.local_block_indices))
        row["global_state_indices"] = json.dumps(list(self.global_state_indices))
        row["global_bus_ids"] = json.dumps(list(self.global_bus_ids))
        row["area_members"] = json.dumps(list(self.area_members))
        return row


@dataclass(slots=True)
class UnavailableFunctional:
    requested_functional_id: str
    family: str
    reason_unavailable: str
    deterministic_selection_policy: str
    requested_target: str


def _unit_norm_error(vector: np.ndarray) -> float:
    return float(abs(np.linalg.norm(vector) - 1.0))


def _finalize_vector(vector: np.ndarray) -> tuple[tuple[float, ...], bool, float]:
    error = _unit_norm_error(vector)
    return tuple(float(v) for v in vector), bool(error <= UNIT_NORM_TOLERANCE), error


def build_coordinate_functionals(binding: PhysicalBlockBinding) -> list[FunctionalRecord]:
    """One unit coordinate functional per in-block physical state (Delta theta / Delta V)."""

    dimension = binding.col_count
    records: list[FunctionalRecord] = []
    for state_type in ("angle", "voltage"):
        typed = [rec for rec in binding.columns if rec.state_type == state_type]
        # Deterministic, outcome-independent primary: smallest global state index.
        primary = min((rec.global_state_index for rec in typed), default=None)
        symbol = "theta" if state_type == "angle" else "V"
        for rec in sorted(typed, key=lambda r: r.global_state_index):
            vector = np.zeros(dimension, dtype=np.float64)
            vector[rec.local_index] = 1.0
            values, unit, error = _finalize_vector(vector)
            records.append(
                FunctionalRecord(
                    functional_id=f"coordinate_{state_type}_bus{rec.bus_id}",
                    family="coordinate",
                    label=f"Delta {symbol} at bus {rec.bus_id}",
                    reason_physically_meaningful=(
                        f"Delta {symbol} update at bus {rec.bus_id} is a primary PSSE state "
                        f"variable ({'phase angle' if state_type == 'angle' else 'voltage magnitude'})."  # noqa: E501
                    ),
                    dimension=dimension,
                    vector=values,
                    local_block_indices=(rec.local_index,),
                    global_state_indices=(rec.global_state_index,),
                    global_bus_ids=(rec.bus_id,),
                    state_type=state_type,
                    branch_id="",
                    area_members=(),
                    normalization="unit_coordinate",
                    normalization_constant=1.0,
                    unit_norm=unit,
                    unit_norm_error=error,
                    deterministic_selection_policy=(
                        "enumerate every in-block state of this type; primary = smallest "
                        "global state index"
                    ),
                    is_primary_for_policy=bool(rec.global_state_index == primary),
                )
            )
    return records


def build_branch_angle_difference_functionals(
    binding: PhysicalBlockBinding,
) -> list[FunctionalRecord]:
    """(e_i - e_j)/sqrt(2) for every real branch with both angle endpoints in-block."""

    dimension = binding.col_count
    angle_by_bus = {
        rec.bus_id: rec for rec in binding.columns if rec.state_type == "angle"
    }
    records: list[FunctionalRecord] = []
    branch_set = set(binding.branches)
    representable = sorted(binding.representable_angle_branches)
    primary = representable[0] if representable else None
    for from_bus, to_bus in representable:
        if (from_bus, to_bus) not in branch_set:
            raise ValueError(f"branch ({from_bus},{to_bus}) is not an active network branch")
        rec_from = angle_by_bus[from_bus]
        rec_to = angle_by_bus[to_bus]
        vector = np.zeros(dimension, dtype=np.float64)
        vector[rec_from.local_index] = 1.0 / math.sqrt(2.0)
        vector[rec_to.local_index] = -1.0 / math.sqrt(2.0)
        values, unit, error = _finalize_vector(vector)
        records.append(
            FunctionalRecord(
                functional_id=f"branch_angle_diff_{from_bus}_{to_bus}",
                family="branch_angle_difference",
                label=f"Delta(theta_{from_bus} - theta_{to_bus}) across branch {from_bus}-{to_bus}",
                reason_physically_meaningful=(
                    f"The angle difference across real branch ({from_bus},{to_bus}) governs "
                    f"active-power flow P ~ (theta_{from_bus}-theta_{to_bus})/x; "
                    "it is a directly monitored PSSE quantity."
                ),
                dimension=dimension,
                vector=values,
                local_block_indices=(rec_from.local_index, rec_to.local_index),
                global_state_indices=(rec_from.global_state_index, rec_to.global_state_index),
                global_bus_ids=(from_bus, to_bus),
                state_type="angle",
                branch_id=f"{from_bus}-{to_bus}",
                area_members=(),
                normalization="one_over_sqrt_2",
                normalization_constant=1.0 / math.sqrt(2.0),
                unit_norm=unit,
                unit_norm_error=error,
                deterministic_selection_policy=(
                    "enumerate real branches with both angle endpoints in-block; primary = "
                    "smallest (from_bus, to_bus)"
                ),
                is_primary_for_policy=bool((from_bus, to_bus) == primary),
            )
        )
    return records


def build_area_aggregate_functionals(
    binding: PhysicalBlockBinding,
) -> tuple[list[FunctionalRecord], list[UnavailableFunctional]]:
    """Connected-area aggregates plus the requested predeclared weak-area aggregate."""

    dimension = binding.col_count
    records: list[FunctionalRecord] = []
    unavailable: list[UnavailableFunctional] = []

    # (a) Requested predeclared weak-area aggregate (case metadata weak_area_buses).
    weak_area = set(binding.weak_area_buses)
    volt_by_bus = {rec.bus_id: rec for rec in binding.columns if rec.state_type == "voltage"}
    weak_in_block = sorted(weak_area & set(volt_by_bus))
    weak_policy = (
        "case metadata weak_area_buses intersected with in-block voltage states; requires >= 2"
    )
    if len(weak_in_block) >= 2:
        records.append(
            _area_record(
                dimension,
                functional_id="weak_area_voltage_aggregate",
                family="area_aggregate",
                buses=weak_in_block,
                columns=[volt_by_bus[b] for b in weak_in_block],
                symbol="V",
                policy=weak_policy,
                reason=(
                    "mean voltage-magnitude update over the case-declared weak area; a coherent "
                    "low-observability voltage sub-region."
                ),
                is_primary=True,
            )
        )
    else:
        unavailable.append(
            UnavailableFunctional(
                requested_functional_id="weak_area_voltage_aggregate",
                family="area_aggregate",
                reason_unavailable=(
                    f"case weak_area_buses={sorted(weak_area)} yields "
                    f"{len(weak_in_block)} in-block voltage states (< 2); not representable"
                ),
                deterministic_selection_policy=weak_policy,
                requested_target="predeclared_weak_area_voltage_profile",
            )
        )

    # (b) Connected in-block voltage-area aggregate (physically coherent sub-network).
    volt_buses = set(binding.bus_set("voltage"))
    if len(volt_buses) >= 2 and bus_set_is_connected(volt_buses, binding.branches):
        members = sorted(volt_buses)
        records.append(
            _area_record(
                dimension,
                functional_id="connected_block_voltage_area_aggregate",
                family="area_aggregate",
                buses=members,
                columns=[volt_by_bus[b] for b in members],
                symbol="V",
                policy=(
                    "all in-block voltage states, verified to form a connected subnetwork "
                    "through real branches"
                ),
                reason=(
                    "mean voltage-magnitude update over a connected local area; a local voltage "
                    "profile of an electrically coherent sub-network."
                ),
                is_primary=True,
            )
        )
    else:
        unavailable.append(
            UnavailableFunctional(
                requested_functional_id="connected_block_voltage_area_aggregate",
                family="area_aggregate",
                reason_unavailable=(
                    f"in-block voltage buses {sorted(volt_buses)} are not a connected >=2 set"
                ),
                deterministic_selection_policy="connected in-block voltage states",
                requested_target="connected_voltage_area_profile",
            )
        )

    # (c) Connected in-block angle-area aggregate.
    angle_by_bus = {rec.bus_id: rec for rec in binding.columns if rec.state_type == "angle"}
    angle_buses = set(binding.bus_set("angle"))
    if len(angle_buses) >= 2 and bus_set_is_connected(angle_buses, binding.branches):
        members = sorted(angle_buses)
        records.append(
            _area_record(
                dimension,
                functional_id="connected_block_angle_area_aggregate",
                family="area_aggregate",
                buses=members,
                columns=[angle_by_bus[b] for b in members],
                symbol="theta",
                policy=(
                    "all in-block angle states, verified to form a connected subnetwork "
                    "through real branches"
                ),
                reason=(
                    "mean phase-angle update over a connected local area; an aggregate angular "
                    "displacement of a coherent sub-network."
                ),
                is_primary=False,
            )
        )
    else:
        unavailable.append(
            UnavailableFunctional(
                requested_functional_id="connected_block_angle_area_aggregate",
                family="area_aggregate",
                reason_unavailable=(
                    f"in-block angle buses {sorted(angle_buses)} are not a connected >=2 set"
                ),
                deterministic_selection_policy="connected in-block angle states",
                requested_target="connected_angle_area_profile",
            )
        )
    return records, unavailable


def _area_record(
    dimension: int,
    *,
    functional_id: str,
    family: str,
    buses: list[int],
    columns: list[Any],
    symbol: str,
    policy: str,
    reason: str,
    is_primary: bool,
) -> FunctionalRecord:
    scale = 1.0 / math.sqrt(len(columns))
    vector = np.zeros(dimension, dtype=np.float64)
    locals_ = tuple(rec.local_index for rec in columns)
    for rec in columns:
        vector[rec.local_index] = scale
    values, unit, error = _finalize_vector(vector)
    return FunctionalRecord(
        functional_id=functional_id,
        family=family,
        label=f"mean Delta {symbol} over area {buses}",
        reason_physically_meaningful=reason,
        dimension=dimension,
        vector=values,
        local_block_indices=locals_,
        global_state_indices=tuple(rec.global_state_index for rec in columns),
        global_bus_ids=tuple(int(b) for b in buses),
        state_type=("angle" if symbol == "theta" else "voltage"),
        branch_id="",
        area_members=tuple(int(b) for b in buses),
        normalization="one_over_sqrt_area_size",
        normalization_constant=scale,
        unit_norm=unit,
        unit_norm_error=error,
        deterministic_selection_policy=policy,
        is_primary_for_policy=is_primary,
    )


def build_legacy_functional_records(
    binding: PhysicalBlockBinding,
) -> list[FunctionalRecord]:
    """The three frozen generic functionals, recorded unchanged with physical interpretation."""

    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        predetermined_selected_functionals,
    )

    dimension = binding.col_count
    legacy = predetermined_selected_functionals(dimension)
    angle_cols = {rec.local_index: rec for rec in binding.columns}
    records: list[FunctionalRecord] = []
    for functional_id, raw in legacy.items():
        vector = np.asarray(raw, dtype=np.float64)
        values, unit, error = _finalize_vector(vector)
        locals_ = tuple(int(i) for i in np.flatnonzero(vector))
        buses = tuple(angle_cols[i].bus_id for i in locals_)
        globals_ = tuple(angle_cols[i].global_state_index for i in locals_)
        records.append(
            FunctionalRecord(
                functional_id=functional_id,
                family="legacy_predetermined",
                label=f"frozen generic functional {functional_id}",
                reason_physically_meaningful=(
                    "unchanged frozen functional retained for backward compatibility; physically "
                    f"acts on angle states of buses {list(buses)}"
                ),
                dimension=dimension,
                vector=values,
                local_block_indices=locals_,
                global_state_indices=globals_,
                global_bus_ids=buses,
                state_type=LEGACY_FUNCTIONAL_INTERPRETATION.get(functional_id, "mixed"),
                branch_id="",
                area_members=(),
                normalization="frozen_predetermined",
                normalization_constant=float(np.max(np.abs(vector))),
                unit_norm=unit,
                unit_norm_error=error,
                deterministic_selection_policy="frozen predetermined (unchanged)",
                is_primary_for_policy=False,
                evidence_status="classical_frozen_backward_compatible",
            )
        )
    return records


def build_all_physical_functionals(
    binding: PhysicalBlockBinding,
) -> tuple[list[FunctionalRecord], list[UnavailableFunctional]]:
    records: list[FunctionalRecord] = []
    records.extend(build_legacy_functional_records(binding))
    records.extend(build_coordinate_functionals(binding))
    records.extend(build_branch_angle_difference_functionals(binding))
    area_records, unavailable = build_area_aggregate_functionals(binding)
    records.extend(area_records)
    # Deterministic ordering by (family, functional_id).
    records.sort(key=lambda r: (r.family, r.functional_id))
    unavailable.sort(key=lambda u: (u.family, u.requested_functional_id))
    return records, unavailable


def instance_functionals(binding: PhysicalBlockBinding) -> dict[str, np.ndarray]:
    """Physical (non-legacy) functionals as {id: unit vector} for downstream workstreams."""

    records, _ = build_all_physical_functionals(binding)
    return {
        record.functional_id: np.asarray(record.vector, dtype=np.float64)
        for record in records
        if record.family != "legacy_predetermined"
    }


def run_physical_functionals(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected study_id: {config.get('study_id')}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("matrix_seed", 123))
    binding = build_physical_block_binding(seed)
    records, unavailable = build_all_physical_functionals(binding)

    for record in records:
        if not record.unit_norm:
            raise RuntimeError(
                f"functional {record.functional_id} is not unit-norm ({record.unit_norm_error})"
            )

    inventory_frame = pd.DataFrame([record.to_row() for record in records])
    atomic_write_csv(destination / "functional_inventory.csv", inventory_frame)

    binding_payload = {
        "seed": binding.seed,
        "selected_rows": list(binding.selected_rows),
        "selected_columns": list(binding.selected_columns),
        "state_dimension": binding.state_dimension,
        "slack_bus": binding.slack_bus,
        "weak_area_buses": list(binding.weak_area_buses),
        "columns": [
            {
                "local_index": rec.local_index,
                "global_state_index": rec.global_state_index,
                "state_type": rec.state_type,
                "bus_id": rec.bus_id,
            }
            for rec in binding.columns
        ],
        "measurement_rows": [
            {"row": int(r), "type": t, "label": lbl}
            for r, t, lbl in zip(
                binding.selected_rows,
                binding.measurement_types,
                binding.measurement_labels,
                strict=True,
            )
        ],
        "representable_angle_branches": [list(b) for b in binding.representable_angle_branches],
        "n_branches": len(binding.branches),
    }
    inventory_json = {
        "study_id": STUDY_ID,
        "generated_at": now_iso(),
        "claim_boundary": CLAIM_BOUNDARY,
        "block_binding": binding_payload,
        "functionals": [
            record.to_row() | {"vector_list": list(record.vector)} for record in records
        ],
        "functional_fingerprints": {
            record.functional_id: array_fingerprint(np.asarray(record.vector))
            for record in records
        },
    }
    atomic_write_json(destination / "functional_inventory.json", inventory_json)

    unavailable_frame = pd.DataFrame([asdict(item) for item in unavailable])
    if unavailable_frame.empty:
        unavailable_frame = pd.DataFrame(
            columns=[
                "requested_functional_id",
                "family",
                "reason_unavailable",
                "deterministic_selection_policy",
                "requested_target",
            ]
        )
    atomic_write_csv(destination / "unavailable_functionals.csv", unavailable_frame)

    _write_protocol(destination, binding, records, unavailable)
    atomic_write_json(
        destination / "provenance.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID},
    )
    write_manifest_and_checksums(
        destination,
        study_id=STUDY_ID,
        extra={
            "functional_count": len(records),
            "unavailable_count": len(unavailable),
            "families": sorted({record.family for record in records}),
        },
    )
    family_counts = (
        inventory_frame.groupby("family")["functional_id"].count().to_dict()
        if not inventory_frame.empty
        else {}
    )
    return {
        "functionals": len(records),
        "unavailable": len(unavailable),
        "family_counts": family_counts,
        "all_unit_norm": bool(all(record.unit_norm for record in records)),
    }


def _write_protocol(
    destination: Path,
    binding: PhysicalBlockBinding,
    records: list[FunctionalRecord],
    unavailable: list[UnavailableFunctional],
) -> None:
    lines = [
        "# Physical Functional Selection Protocol",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Determinism and outcome-independence",
        "",
        "Every functional below is defined from IEEE-14 topology and state-index metadata only "
        "(bus lists, state types, real-branch adjacency, and the outcome-independent "
        "largest-column-norm block selection). No solved Ridge or QSVT output, residual value, or "
        "evaluation outcome is consulted during functional selection; a favourable-output bias is "
        "therefore structurally impossible.",
        "",
        "## Frozen block binding (seed 123)",
        "",
        f"- Selected measurement rows: `{list(binding.selected_rows)}`",
        f"- Selected global state columns: `{list(binding.selected_columns)}`",
        "",
        "| local | global | type | bus |",
        "|---:|---:|---|---:|",
    ]
    for rec in binding.columns:
        lines.append(
            f"| {rec.local_index} | {rec.global_state_index} | {rec.state_type} | {rec.bus_id} |"
        )
    lines += [
        "",
        "## Functional families",
        "",
        "1. **coordinate** - unit coordinate on one physical state (Delta theta / Delta V).",
        "2. **branch_angle_difference** - (e_i - e_j)/sqrt(2) across a real branch whose both "
        "angle endpoints are in-block; governs active-power flow.",
        "3. **area_aggregate** - (1/sqrt|A|) sum over a connected local area (voltage or angle "
        "profile of an electrically coherent sub-network).",
        "4. **legacy_predetermined** - the three frozen generic functionals, unchanged.",
        "",
        "## Selected functionals",
        "",
        "| id | family | label | primary | unit-norm err |",
        "|---|---|---|:--:|---:|",
    ]
    for rec in records:
        lines.append(
            f"| `{rec.functional_id}` | {rec.family} | {rec.label} | "
            f"{'yes' if rec.is_primary_for_policy else ''} | {rec.unit_norm_error:.2e} |"
        )
    lines += ["", "## Unavailable functionals (recorded, never silently substituted)", ""]
    if unavailable:
        lines += ["| requested | family | reason |", "|---|---|---|"]
        for item in unavailable:
            lines.append(
                f"| `{item.requested_functional_id}` | {item.family} | {item.reason_unavailable} |"
            )
    else:
        lines.append("None.")
    lines.append("")
    (destination / "functional_selection_protocol.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
