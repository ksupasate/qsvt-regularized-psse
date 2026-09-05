from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class StateIndexRecord:
    index: int
    state_type: str
    bus_id: int | None
    bus_position: int | None
    description: str

    def to_row(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "state_type": self.state_type,
            "bus_id": self.bus_id,
            "bus_position": self.bus_position,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class StateMetadata:
    records: tuple[StateIndexRecord, ...]
    source: str
    ordering_note: str
    assumptions: tuple[str, ...] = ()

    @property
    def dimension(self) -> int:
        return len(self.records)

    @property
    def angle_indices(self) -> tuple[int, ...]:
        return tuple(record.index for record in self.records if record.state_type == "angle")

    @property
    def voltage_indices(self) -> tuple[int, ...]:
        return tuple(record.index for record in self.records if record.state_type == "voltage")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.to_row() for record in self.records])

    def record_for_index(self, index: int) -> StateIndexRecord:
        selected = int(index)
        for record in self.records:
            if record.index == selected:
                return record
        raise IndexError(f"state index {selected} is not present in metadata")

    def index_for_angle_bus(self, bus_id: int) -> int:
        return self._index_for(bus_id=int(bus_id), state_type="angle")

    def index_for_voltage_bus(self, bus_id: int) -> int:
        return self._index_for(bus_id=int(bus_id), state_type="voltage")

    def indices_for_buses(self, bus_ids: Sequence[int]) -> list[int]:
        requested = {int(bus_id) for bus_id in bus_ids}
        return [
            int(record.index)
            for record in self.records
            if record.bus_id in requested and record.state_type in {"angle", "voltage"}
        ]

    def _index_for(self, *, bus_id: int, state_type: str) -> int:
        for record in self.records:
            if record.bus_id == bus_id and record.state_type == state_type:
                return int(record.index)
        raise KeyError(f"no {state_type} state index is available for bus {bus_id}")


def build_state_metadata_from_system_metadata(metadata: Mapping[str, Any]) -> StateMetadata:
    """Build auditable AC update-vector metadata.

    The AC linearized model in :mod:`robust_qsvt_se.measurement.ac_linear`
    constructs ``Delta x = [Delta theta, Delta V]`` and stores the bus lists and
    index lists in ``WeightedSystem.metadata``. This function consumes that
    explicit metadata when present. If only a dimension is available, it returns
    generic records and records the fallback assumption instead of guessing bus
    semantics.
    """

    angle_buses = _optional_int_list(metadata.get("angle_state_buses"))
    voltage_buses = _optional_int_list(metadata.get("voltage_state_buses"))
    angle_indices = _optional_int_list(metadata.get("angle_state_indices"))
    voltage_indices = _optional_int_list(metadata.get("voltage_magnitude_state_indices"))
    assumptions: list[str] = []

    if angle_buses and voltage_buses:
        if angle_indices is None:
            angle_indices = list(range(len(angle_buses)))
            assumptions.append("angle indices inferred from [Delta theta, Delta V] ordering")
        if voltage_indices is None:
            start = len(angle_indices)
            voltage_indices = list(range(start, start + len(voltage_buses)))
            assumptions.append("voltage indices inferred after angle-state block")
        if len(angle_indices) != len(angle_buses):
            raise ValueError("angle_state_indices and angle_state_buses lengths differ")
        if len(voltage_indices) != len(voltage_buses):
            raise ValueError(
                "voltage_magnitude_state_indices and voltage_state_buses lengths differ"
            )

        records: list[StateIndexRecord] = []
        for position, (index, bus_id) in enumerate(zip(angle_indices, angle_buses, strict=True)):
            records.append(
                StateIndexRecord(
                    index=int(index),
                    state_type="angle",
                    bus_id=int(bus_id),
                    bus_position=int(position),
                    description=f"Delta theta update for bus {int(bus_id)}",
                )
            )
        for position, (index, bus_id) in enumerate(
            zip(voltage_indices, voltage_buses, strict=True)
        ):
            records.append(
                StateIndexRecord(
                    index=int(index),
                    state_type="voltage",
                    bus_id=int(bus_id),
                    bus_position=int(position),
                    description=f"Delta voltage magnitude update for bus {int(bus_id)}",
                )
            )
        records = sorted(records, key=lambda record: record.index)
        return StateMetadata(
            records=tuple(records),
            source="weighted_system_metadata",
            ordering_note="Delta x = [Delta theta, Delta V]",
            assumptions=tuple(assumptions),
        )

    dimension = int(metadata.get("n_states", metadata.get("state_dimension", 0)) or 0)
    if dimension <= 0:
        raise ValueError("state metadata requires AC bus/index metadata or n_states")
    records = tuple(
        StateIndexRecord(
            index=index,
            state_type="unknown",
            bus_id=None,
            bus_position=None,
            description=f"Generic state update component {index}",
        )
        for index in range(dimension)
    )
    return StateMetadata(
        records=records,
        source="generic_dimension_fallback",
        ordering_note="state ordering not exposed by source metadata",
        assumptions=("bus/state semantics are unavailable; observables are index-level only",),
    )


def write_state_metadata_csv(metadata: StateMetadata, path: str | Path) -> Path:
    output_path = Path(path)
    metadata.to_frame().to_csv(output_path, index=False)
    return output_path


def _optional_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    values = [int(item) for item in value]
    return values if values else None
