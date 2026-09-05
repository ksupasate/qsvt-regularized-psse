"""Selected signed-observable builders for physically meaningful PSSE quantities.

These observables are *selected linear functionals* of the matched-alpha
Ridge/Tikhonov update

    dx_alpha = (H~^T H~ + alpha I)^{-1} H~^T r~,

i.e. ``y_l = l^T dx_alpha``. Each builder targets one physically meaningful
quantity of the AC-linearized state update ``dx = [d_theta, d_V]``:

* voltage-magnitude correction at a bus,    e_i^T d_V
* voltage-angle correction at a bus,        e_i^T d_theta
* branch angle-difference correction,       (e_i - e_j)^T d_theta
* area / selected-bus aggregate functional, l^T dx
* energy-style observable for comparison,   ||Pi_A dx||^2

The signed functionals require a *sign-aware* readout (a Hadamard-test-style
inner-product proxy); the energy-style observable is *basis-sampling
accessible*. When bus/state metadata is unavailable, the builders fall back to
coordinate-/block-level observables and label them as such. Each selected
readout estimates **one** functional, never the full signed update vector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.state_metadata import StateMetadata

# Readout-model labels and their classification.
SIGN_AWARE_MODEL = "sign_aware_inner_product_proxy"
BASIS_SAMPLING_MODEL = "basis_sampling_squared_amplitude"

SIGN_AWARE_STATUS = "sign_aware_readout_required"
BASIS_SAMPLING_STATUS = "basis_sampling_accessible"


@dataclass(frozen=True, slots=True)
class SelectedObservable:
    """One selected observable of the update vector with its readout classification.

    For signed linear functionals, ``vector`` holds the functional ``l`` and
    ``support`` lists its nonzero coordinates. For the energy-style observable,
    ``support`` is the projector support ``A`` and ``vector`` is the indicator on
    ``A`` (used only for support bookkeeping).
    """

    observable_id: str
    observable_type: str
    physical_meaning: str
    vector: np.ndarray = field(repr=False)
    support: tuple[int, ...]
    readout_model: str
    basis_sampling_accessible: bool
    sign_aware_required: bool
    full_vector_required: bool
    status: str
    notes: str
    selection_policy: str = "predetermined_from_case_metadata"
    selected_before_solving: bool = True
    depends_on_solution: bool = False
    units_or_normalization: str = "per-unit or radians according to state type"

    @property
    def support_size(self) -> int:
        return len(self.support)

    def exact_value(self, update: np.ndarray) -> float:
        """Exact classical matched value of this observable on ``update``."""

        dx = np.asarray(update, dtype=np.float64)
        if self.readout_model == BASIS_SAMPLING_MODEL:
            return float(np.sum(dx[list(self.support)] ** 2))
        return float(self.vector @ dx)

    def to_row(self, *, case: str, matrix_source: str) -> dict[str, Any]:
        return {
            "case": case,
            "matrix_source": matrix_source,
            "observable_id": self.observable_id,
            "observable_type": self.observable_type,
            "physical_meaning": self.physical_meaning,
            "selection_policy": self.selection_policy,
            "selected_before_solving": self.selected_before_solving,
            "depends_on_solution": self.depends_on_solution,
            "support_size": self.support_size,
            "units_or_normalization": self.units_or_normalization,
            "readout_model": self.readout_model,
            "basis_sampling_accessible": self.basis_sampling_accessible,
            "sign_aware_required": self.sign_aware_required,
            "full_vector_required": self.full_vector_required,
            "status": self.status,
            "notes": self.notes,
        }


def _signed_functional(
    *,
    observable_id: str,
    observable_type: str,
    physical_meaning: str,
    vector: np.ndarray,
    notes: str,
    selection_policy: str = "predetermined_from_case_metadata",
    units_or_normalization: str = "per-unit or radians according to state type",
) -> SelectedObservable:
    support = tuple(int(i) for i in np.nonzero(vector)[0])
    return SelectedObservable(
        observable_id=observable_id,
        observable_type=observable_type,
        physical_meaning=physical_meaning,
        vector=np.asarray(vector, dtype=np.float64),
        support=support,
        readout_model=SIGN_AWARE_MODEL,
        basis_sampling_accessible=False,
        sign_aware_required=True,
        full_vector_required=False,
        status=SIGN_AWARE_STATUS,
        notes=notes,
        selection_policy=selection_policy,
        units_or_normalization=units_or_normalization,
    )


def voltage_magnitude_observable(metadata: StateMetadata, bus_id: int) -> SelectedObservable:
    index = metadata.index_for_voltage_bus(int(bus_id))
    vector = np.zeros(metadata.dimension, dtype=np.float64)
    vector[index] = 1.0
    return _signed_functional(
        observable_id=f"dV_bus{int(bus_id)}",
        observable_type="voltage_magnitude_correction",
        physical_meaning=f"voltage-magnitude correction e_i^T dV at bus {int(bus_id)}",
        vector=vector,
        notes="signed voltage-magnitude update component",
        selection_policy="bus_voltage_selected_bus",
        units_or_normalization="per unit voltage magnitude correction",
    )


def voltage_angle_observable(metadata: StateMetadata, bus_id: int) -> SelectedObservable:
    index = metadata.index_for_angle_bus(int(bus_id))
    vector = np.zeros(metadata.dimension, dtype=np.float64)
    vector[index] = 1.0
    return _signed_functional(
        observable_id=f"dtheta_bus{int(bus_id)}",
        observable_type="voltage_angle_correction",
        physical_meaning=f"voltage-angle correction e_i^T dtheta at bus {int(bus_id)}",
        vector=vector,
        notes="signed voltage-angle update component",
        selection_policy="bus_angle_selected_bus",
        units_or_normalization="radians angle correction",
    )


def branch_angle_difference_observable(
    metadata: StateMetadata, from_bus: int, to_bus: int
) -> SelectedObservable:
    i = metadata.index_for_angle_bus(int(from_bus))
    j = metadata.index_for_angle_bus(int(to_bus))
    vector = np.zeros(metadata.dimension, dtype=np.float64)
    vector[i] += 1.0
    vector[j] -= 1.0
    return _signed_functional(
        observable_id=f"dtheta_diff_{int(from_bus)}_{int(to_bus)}",
        observable_type="branch_angle_difference_correction",
        physical_meaning=(
            f"branch angle-difference correction (e_i - e_j)^T dtheta for branch "
            f"{int(from_bus)}-{int(to_bus)}"
        ),
        vector=vector,
        notes="signed branch angle-difference update (drives real-power flow)",
        selection_policy="branch_angle_diff_first_branch",
        units_or_normalization="radians angle-difference correction",
    )


def area_aggregate_observable(
    metadata: StateMetadata,
    indices: Sequence[int],
    *,
    signs: Sequence[float] | None = None,
    observable_id: str = "area_aggregate",
    physical_meaning: str | None = None,
    selection_policy: str = "predefined_area_set",
    units_or_normalization: str = "sum over selected state corrections",
) -> SelectedObservable:
    idx = [int(i) for i in indices]
    vector = np.zeros(metadata.dimension, dtype=np.float64)
    weights = [1.0] * len(idx) if signs is None else [float(s) for s in signs]
    for position, index in enumerate(idx):
        vector[index] += weights[position]
    meaning = physical_meaning or f"signed aggregate l^T dx over {len(idx)} selected states"
    return _signed_functional(
        observable_id=observable_id,
        observable_type="area_aggregate_functional",
        physical_meaning=meaning,
        vector=vector,
        notes="signed aggregate functional over a selected bus/area set",
        selection_policy=selection_policy,
        units_or_normalization=units_or_normalization,
    )


def energy_observable(
    dimension: int,
    support: Sequence[int],
    *,
    observable_id: str = "energy_selected_block",
    physical_meaning: str | None = None,
) -> SelectedObservable:
    idx = tuple(int(i) for i in support)
    vector = np.zeros(int(dimension), dtype=np.float64)
    for index in idx:
        vector[index] = 1.0
    meaning = physical_meaning or f"energy-style ||Pi_A dx||^2 on {len(idx)} selected states"
    return SelectedObservable(
        observable_id=observable_id,
        observable_type="energy_block",
        physical_meaning=meaning,
        vector=vector,
        support=idx,
        readout_model=BASIS_SAMPLING_MODEL,
        basis_sampling_accessible=True,
        sign_aware_required=False,
        full_vector_required=False,
        status=BASIS_SAMPLING_STATUS,
        notes="energy-style squared-amplitude observable; sign is not recovered",
        selection_policy="predetermined_voltage_state_block",
        units_or_normalization="squared physical update norm on selected support",
    )


def _coordinate_signed(
    dimension: int, index: int, *, label: str, meaning: str
) -> SelectedObservable:
    vector = np.zeros(int(dimension), dtype=np.float64)
    vector[int(index)] = 1.0
    return _signed_functional(
        observable_id=label,
        observable_type="coordinate_level_correction",
        physical_meaning=meaning,
        vector=vector,
        notes="coordinate-level diagnostic (bus/state metadata unavailable)",
    )


def build_selected_observables(
    metadata: StateMetadata,
    update: np.ndarray | None = None,
    *,
    branches: Sequence[tuple[int, int]] | None = None,
    area_size: int = 4,
) -> list[SelectedObservable]:
    """Build a predetermined representative selected-observable set.

    Selection uses only case metadata, fixed bus/branch order, and predefined
    supports. ``update`` is accepted for API compatibility but is deliberately
    ignored: no observable in the main workload may be chosen after solving.
    """

    if metadata.source == "weighted_system_metadata":
        observables = _physical_observables(metadata, branches, area_size)
    else:
        observables = _fallback_observables(metadata.dimension, area_size)

    # De-duplicate by observable_id (tiny systems can collapse selections).
    unique: dict[str, SelectedObservable] = {}
    for observable in observables:
        unique.setdefault(observable.observable_id, observable)
    return list(unique.values())


def _physical_observables(
    metadata: StateMetadata,
    branches: Sequence[tuple[int, int]] | None,
    area_size: int,
) -> list[SelectedObservable]:
    angle_indices = list(metadata.angle_indices)
    voltage_indices = list(metadata.voltage_indices)
    observables: list[SelectedObservable] = []

    if voltage_indices:
        bus = metadata.record_for_index(int(sorted(voltage_indices)[0])).bus_id
        if bus is not None:
            observables.append(voltage_magnitude_observable(metadata, bus))
    if angle_indices:
        bus = metadata.record_for_index(int(sorted(angle_indices)[0])).bus_id
        if bus is not None:
            observables.append(voltage_angle_observable(metadata, bus))

    branch = _pick_branch(metadata, branches)
    if branch is not None:
        observables.append(branch_angle_difference_observable(metadata, branch[0], branch[1]))

    angle_area = [int(i) for i in sorted(angle_indices)[: max(1, int(area_size))]]
    if angle_area:
        observables.append(
            area_aggregate_observable(
                metadata,
                angle_area,
                observable_id="area_first4_angle",
                physical_meaning="predetermined aggregate of the first four angle states",
                selection_policy="area_first4_angle",
                units_or_normalization="sum of radians angle corrections",
            )
        )
    voltage_area = [int(i) for i in sorted(voltage_indices)[: max(1, int(area_size))]]
    if voltage_area:
        observables.append(
            area_aggregate_observable(
                metadata,
                voltage_area,
                observable_id="area_first4_voltage",
                physical_meaning=(
                    "predetermined aggregate of the first four voltage-magnitude states"
                ),
                selection_policy="area_first4_voltage",
                units_or_normalization="sum of per-unit voltage corrections",
            )
        )

    # Energy-style observable on the voltage block (physically grouped support).
    energy_support = voltage_indices or angle_area
    observables.append(
        energy_observable(
            metadata.dimension,
            energy_support,
            observable_id="energy_voltage_block",
            physical_meaning="energy-style ||Pi_A dx||^2 on the voltage-magnitude state block",
        )
    )
    return observables


def _fallback_observables(dimension: int, area_size: int) -> list[SelectedObservable]:
    area_idx = list(range(min(max(1, int(area_size)), int(dimension))))
    observables = [
        _coordinate_signed(
            dimension, 0, label="coord_0", meaning="predetermined first coordinate component"
        ),
    ]
    observables.append(
        SelectedObservable(
            observable_id="block_first4_signed",
            observable_type="block_level_functional",
            physical_meaning="predetermined aggregate over the first four coordinates",
            vector=_indicator(dimension, area_idx, [1.0] * len(area_idx)),
            support=tuple(area_idx),
            readout_model=SIGN_AWARE_MODEL,
            basis_sampling_accessible=False,
            sign_aware_required=True,
            full_vector_required=False,
            status=SIGN_AWARE_STATUS,
            notes="block-level diagnostic (bus/state metadata unavailable)",
            selection_policy="first_four_coordinates",
            units_or_normalization="sum in unspecified coordinate units",
        )
    )
    observables.append(
        energy_observable(
            dimension,
            area_idx,
            observable_id="energy_first4_block",
            physical_meaning="energy-style ||Pi_A dx||^2 on the first coordinate block",
        )
    )
    return observables


def _indicator(dimension: int, indices: Sequence[int], signs: Sequence[float]) -> np.ndarray:
    vector = np.zeros(int(dimension), dtype=np.float64)
    for index, sign in zip(indices, signs, strict=True):
        vector[int(index)] += float(sign)
    return vector


def _pick_branch(
    metadata: StateMetadata, branches: Sequence[tuple[int, int]] | None
) -> tuple[int, int] | None:
    angle_buses = {
        metadata.record_for_index(i).bus_id
        for i in metadata.angle_indices
        if metadata.record_for_index(i).bus_id is not None
    }
    if branches:
        for from_bus, to_bus in branches:
            if int(from_bus) in angle_buses and int(to_bus) in angle_buses:
                return int(from_bus), int(to_bus)
    # Fall back to the first two distinct angle buses (both are valid state indices).
    ordered = [
        metadata.record_for_index(i).bus_id
        for i in sorted(metadata.angle_indices)
        if metadata.record_for_index(i).bus_id is not None
    ]
    if len(ordered) >= 2:
        return int(ordered[0]), int(ordered[1])
    return None
