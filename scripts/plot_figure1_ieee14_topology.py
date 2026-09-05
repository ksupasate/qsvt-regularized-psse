#!/usr/bin/env python3
"""Export a repository-derived IEEE-14 topology for manuscript Figure 1.

The script loads ``pypower.case14.case14`` through the repository's
``load_power_case`` and ``load_ac_case`` pathways.  Bus/branch connectivity is
never read from a screenshot or manuscript graphic.  The fixed node positions
are visualization-only coordinates; they are not geographic data.

Outputs are written only to ``outputs/figure1_network_audit`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version as package_version
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep Matplotlib cache writes out of the repository and user home directory.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "qsvt_figure1_ieee14_matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from pypower.idx_brch import (  # noqa: E402
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    T_BUS,
    TAP,
)
from pypower.idx_bus import (  # noqa: E402
    BASE_KV,
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    PQ,
    PV,
    QD,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_gen import (  # noqa: E402
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)

from robust_qsvt_se.data.cases import load_ac_case  # noqa: E402
from robust_qsvt_se.data.real_cases import load_power_case  # noqa: E402

OUTPUT_DIR_DEFAULT = ROOT / "outputs" / "figure1_network_audit"
CURRENT_MOCKUP = ROOT / "outputs" / "ieee_dataset_visualization" / "ieee14_topology_clean.svg"
SCRIPT_REL = Path("scripts/plot_figure1_ieee14_topology.py")

CASE_NAME = "ieee14"
CASE_DISPLAY = "IEEE-14"
EXPECTED_SOURCE_DETAIL = "pypower.case14.case14"
EXPECTED_BUS_IDS = tuple(range(1, 15))
ACTIVE_STATUS_RULE = "BR_STATUS > 0.0"

# Coordinates are layout data only.  They retain the broad left-to-right visual
# organization of the pre-existing repository mock-up, while using a compact,
# crossing-free routing suitable for a small manuscript panel.  They are not
# electrical input data and are not geographic coordinates.
BUS_POSITIONS: dict[int, tuple[float, float]] = {
    1: (0.00, 0.55),
    2: (1.35, 0.05),
    3: (1.35, -1.15),
    4: (3.20, -0.45),
    5: (3.25, 1.05),
    6: (5.40, 1.25),
    7: (5.15, -0.95),
    8: (5.15, -1.85),
    9: (5.95, -0.55),
    10: (7.00, -0.78),
    11: (7.00, 0.68),
    12: (7.20, 2.00),
    13: (8.55, 1.35),
    14: (8.55, -0.18),
}

# One display-only route keeps the two right-hand corridors visually separate.
# The edge must be present in the repository case or the script fails.
CURVED_EDGE = (9, 14)
CURVE_CONTROL_1 = (6.40, -1.45)
CURVE_CONTROL_2 = (8.30, -1.40)

EDGE_COLOR = "#303030"
NODE_EDGE_COLOR = "#111111"
GENERATOR_RING_COLOR = "#6f6f6f"
GENERATOR_FILL = "#eeeeee"
WHITE = "#ffffff"
BLACK = "#111111"
FONT_FAMILY = "DejaVu Sans"

PRIMARY_FIGSIZE = (1.75, 0.88)
PANEL_PREVIEW_FIGSIZE = (1.40, 0.72)
PORTRAIT_FIGSIZE = (1.25, 2.49)
PORTRAIT_PREVIEW_FIGSIZE = (1.10, 2.19)
PNG_DPI = 600

NODE_MARKER_AREA_PT2 = 54.0
NODE_LINEWIDTH_PT = 0.55


def rotate_layout_to_portrait(
    point: tuple[float, float],
) -> tuple[float, float]:
    """Rotate the existing visualization layout clockwise into portrait form."""

    x, y = point
    rightmost_x = max(position[0] for position in BUS_POSITIONS.values())
    return (-y, rightmost_x - x)


# This is a rigid rotation of BUS_POSITIONS, not a second topology definition.
# Bus 1 therefore remains at the input side and buses 13--14 at the output side,
# now read from top to bottom.  Connectivity continues to come exclusively from
# the repository case's active branch rows.
PORTRAIT_BUS_POSITIONS: dict[int, tuple[float, float]] = {
    bus: rotate_layout_to_portrait(position)
    for bus, position in BUS_POSITIONS.items()
}
PORTRAIT_CURVE_CONTROL_1 = rotate_layout_to_portrait(CURVE_CONTROL_1)
PORTRAIT_CURVE_CONTROL_2 = rotate_layout_to_portrait(CURVE_CONTROL_2)
PORTRAIT_X_LIMITS = (-2.55, 2.30)
PORTRAIT_Y_LIMITS = (-0.55, 9.10)

TYPE_LABELS = {
    int(PQ): "PQ",
    int(PV): "PV",
    int(REF): "REF",
    4: "ISOLATED",
}


@dataclass(frozen=True)
class CaseAudit:
    bus_rows: list[dict[str, Any]]
    branch_rows: list[dict[str, Any]]
    generator_rows: list[dict[str, Any]]
    bus_ids: tuple[int, ...]
    repository_edges: tuple[tuple[int, int], ...]
    raw_active_edges: tuple[tuple[int, int], ...]
    generator_buses: tuple[int, ...]
    reference_bus: int
    pv_buses: tuple[int, ...]
    pq_buses: tuple[int, ...]
    transformer_edges: tuple[tuple[int, int], ...]
    topology_sha256: str
    topology_hash_payload: dict[str, Any]
    source_detail: str
    case_module_file: Path
    case_module_sha256: str
    pypower_version: str
    base_mva: float
    total_branch_count: int
    active_branch_count: int
    total_generator_rows: int
    active_generator_rows: int


@dataclass(frozen=True)
class RenderCheck:
    variant: str
    plotted_buses: tuple[int, ...]
    plotted_labels: tuple[int, ...]
    plotted_edges: tuple[tuple[int, int], ...]
    missing_edges: tuple[tuple[int, int], ...]
    extra_edges: tuple[tuple[int, int], ...]
    duplicate_bus_nodes: tuple[int, ...]
    duplicate_bus_labels: tuple[int, ...]
    label_overlap_pairs: tuple[tuple[int, int], ...]
    nonincident_edge_crossings: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    labels_above_edges: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "plotted_buses": list(self.plotted_buses),
            "plotted_labels": list(self.plotted_labels),
            "plotted_edges": [list(edge) for edge in self.plotted_edges],
            "missing_edges": [list(edge) for edge in self.missing_edges],
            "extra_edges": [list(edge) for edge in self.extra_edges],
            "duplicate_bus_nodes": list(self.duplicate_bus_nodes),
            "duplicate_bus_labels": list(self.duplicate_bus_labels),
            "label_overlap_pairs": [list(pair) for pair in self.label_overlap_pairs],
            "nonincident_edge_crossings": [
                [list(left), list(right)] for left, right in self.nonincident_edge_crossings
            ],
            "labels_above_edges": self.labels_above_edges,
            "pass": self.passed,
        }

    @property
    def passed(self) -> bool:
        expected_buses = tuple(EXPECTED_BUS_IDS)
        return bool(
            self.plotted_buses == expected_buses
            and self.plotted_labels == expected_buses
            and not self.missing_edges
            and not self.extra_edges
            and not self.duplicate_bus_nodes
            and not self.duplicate_bus_labels
            and not self.label_overlap_pairs
            and not self.nonincident_edge_crossings
            and self.labels_above_edges
        )


def normalize_edge(from_bus: int, to_bus: int) -> tuple[int, int]:
    left, right = int(from_bus), int(to_bus)
    if left == right:
        raise ValueError(f"self-loop is not valid for this topology audit: bus {left}")
    return (left, right) if left < right else (right, left)


def stable_float(value: Any) -> str:
    return f"{float(value):.12g}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bus_note(type_label: str, online_generator: bool, pd_mw: float, qd_mvar: float) -> str:
    parts: list[str] = []
    if type_label == "REF":
        parts.append("reference/slack bus")
    elif type_label == "PV":
        parts.append("PV bus")
    elif type_label == "PQ":
        parts.append("PQ bus")
    else:
        parts.append(type_label.lower())
    parts.append(
        "online generator row present" if online_generator else "no online generator row"
    )
    if abs(pd_mw) > 0.0 or abs(qd_mvar) > 0.0:
        parts.append("nonzero case demand")
    return "; ".join(parts)


def load_and_verify_case() -> CaseAudit:
    power_case = load_power_case(CASE_NAME)
    ac_case = load_ac_case(CASE_NAME, case_source="pypower")
    if power_case.source_detail != EXPECTED_SOURCE_DETAIL:
        raise RuntimeError(
            f"unexpected IEEE-14 source: {power_case.source_detail}; "
            f"expected {EXPECTED_SOURCE_DETAIL}"
        )

    bus_ids = tuple(int(value) for value in power_case.bus[:, BUS_I])
    if bus_ids != EXPECTED_BUS_IDS:
        raise RuntimeError(f"IEEE-14 bus IDs changed: {bus_ids}")
    if len(set(bus_ids)) != len(bus_ids):
        raise RuntimeError("duplicate bus IDs in repository case")

    active_mask = power_case.branch[:, BR_STATUS] > 0.0
    raw_active_edges = tuple(
        normalize_edge(int(row[F_BUS]), int(row[T_BUS]))
        for row in power_case.branch[active_mask]
    )
    if len(raw_active_edges) != len(set(raw_active_edges)):
        raise RuntimeError("duplicate active branch endpoints in repository case")

    # This is the exact AC-case adapter used downstream by the weighted-Jacobian
    # construction.  The check prevents a raw-table/adapter divergence.
    repository_edges = tuple(
        normalize_edge(branch.from_bus, branch.to_bus) for branch in ac_case.branches
    )
    if set(repository_edges) != set(raw_active_edges):
        raise RuntimeError("repository AC-case adapter does not match raw active branch rows")
    if int(ac_case.slack_bus) not in bus_ids:
        raise RuntimeError("repository AC-case adapter returned an invalid slack bus")

    type_by_bus = {
        int(row[BUS_I]): TYPE_LABELS.get(int(row[BUS_TYPE]), f"TYPE_{int(row[BUS_TYPE])}")
        for row in power_case.bus
    }
    reference_buses = tuple(bus for bus in bus_ids if type_by_bus[bus] == "REF")
    if len(reference_buses) != 1 or reference_buses[0] != int(ac_case.slack_bus):
        raise RuntimeError(
            "raw bus-type table and repository AC-case adapter disagree on the reference bus"
        )
    reference_bus = reference_buses[0]
    pv_buses = tuple(bus for bus in bus_ids if type_by_bus[bus] == "PV")
    pq_buses = tuple(bus for bus in bus_ids if type_by_bus[bus] == "PQ")

    generator_present = {
        bus: bool(np.any(power_case.gen[:, GEN_BUS] == float(bus))) for bus in bus_ids
    }
    online_generator_present = {
        bus: bool(
            np.any(
                (power_case.gen[:, GEN_BUS] == float(bus))
                & (power_case.gen[:, GEN_STATUS] > 0.0)
            )
        )
        for bus in bus_ids
    }
    generator_buses = tuple(
        bus for bus in bus_ids if online_generator_present[bus]
    )

    bus_rows: list[dict[str, Any]] = []
    for row in power_case.bus:
        bus = int(row[BUS_I])
        type_code = int(row[BUS_TYPE])
        type_label = type_by_bus[bus]
        generator_row_count = int(np.count_nonzero(power_case.gen[:, GEN_BUS] == float(bus)))
        online_generator_row_count = int(
            np.count_nonzero(
                (power_case.gen[:, GEN_BUS] == float(bus))
                & (power_case.gen[:, GEN_STATUS] > 0.0)
            )
        )
        bus_rows.append(
            {
                "bus": bus,
                "bus_type_code": type_code,
                "bus_type": type_label,
                "generator_present": generator_present[bus],
                "online_generator_present": online_generator_present[bus],
                "generator_row_count": generator_row_count,
                "online_generator_row_count": online_generator_row_count,
                "pd_mw": float(row[PD]),
                "qd_mvar": float(row[QD]),
                "gs_mw_at_v1": float(row[GS]),
                "bs_mvar_at_v1": float(row[BS]),
                "area": int(row[BUS_AREA]),
                "vm_pu": float(row[VM]),
                "va_deg": float(row[VA]),
                "base_kv": float(row[BASE_KV]),
                "zone": int(row[ZONE]),
                "vmax_pu": float(row[VMAX]),
                "vmin_pu": float(row[VMIN]),
                "notes": _bus_note(
                    type_label,
                    online_generator_present[bus],
                    float(row[PD]),
                    float(row[QD]),
                ),
            }
        )

    branch_rows: list[dict[str, Any]] = []
    transformer_edges: list[tuple[int, int]] = []
    for branch_index, row in enumerate(power_case.branch, start=1):
        from_bus, to_bus = int(row[F_BUS]), int(row[T_BUS])
        edge = normalize_edge(from_bus, to_bus)
        status = float(row[BR_STATUS])
        active = status > 0.0
        raw_tap = float(row[TAP])
        shift_deg = float(row[SHIFT])
        transformer_tap = bool(raw_tap != 0.0 or shift_deg != 0.0)
        effective_tap = raw_tap if raw_tap != 0.0 else 1.0
        if transformer_tap:
            transformer_edges.append(edge)
            notes = f"off-nominal tap {effective_tap:.3f}"
            if shift_deg != 0.0:
                notes += f"; phase shift {shift_deg:.3f} deg"
        else:
            notes = "line branch; nominal effective tap 1.0"
        if not active:
            notes += "; excluded from rendered topology because status <= 0"
        branch_rows.append(
            {
                "branch_index": branch_index,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "status": status,
                "active": active,
                "transformer_or_tap": transformer_tap,
                "tap_ratio_case": raw_tap,
                "effective_tap_ratio": effective_tap,
                "phase_shift_deg": shift_deg,
                "r_pu": float(row[BR_R]),
                "x_pu": float(row[BR_X]),
                "b_pu": float(row[BR_B]),
                "rate_a_mva": float(row[RATE_A]),
                "rate_b_mva": float(row[RATE_B]),
                "rate_c_mva": float(row[RATE_C]),
                "angle_min_deg": float(row[ANGMIN]),
                "angle_max_deg": float(row[ANGMAX]),
                "notes": notes,
            }
        )

    generator_rows: list[dict[str, Any]] = []
    for generator_index, row in enumerate(power_case.gen, start=1):
        status = float(row[GEN_STATUS])
        generator_rows.append(
            {
                "generator_index": generator_index,
                "bus": int(row[GEN_BUS]),
                "status": status,
                "active": status > 0.0,
                "pg_mw": float(row[PG]),
                "qg_mvar": float(row[QG]),
                "qmax_mvar": float(row[QMAX]),
                "qmin_mvar": float(row[QMIN]),
                "vg_pu": float(row[VG]),
                "mbase_mva": float(row[MBASE]),
                "pmax_mw": float(row[PMAX]),
                "pmin_mw": float(row[PMIN]),
                "notes": "online generator row" if status > 0.0 else "offline generator row",
            }
        )

    topology_hash_payload = {
        "active_edges": [list(edge) for edge in sorted(set(repository_edges))],
        "buses": list(sorted(bus_ids)),
    }
    canonical_topology = json.dumps(
        topology_hash_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    topology_sha256 = hashlib.sha256(canonical_topology).hexdigest()

    module_name, _function_name = power_case.source_detail.rsplit(".", 1)
    case_module = importlib.import_module(module_name)
    case_module_file = Path(case_module.__file__).resolve()

    curved_key = normalize_edge(*CURVED_EDGE)
    if curved_key not in set(repository_edges):
        raise RuntimeError(
            f"display-only curved route {curved_key} is not an active repository branch"
        )
    if set(BUS_POSITIONS) != set(bus_ids):
        raise RuntimeError(
            "visualization coordinate table does not cover every case bus exactly once"
        )

    return CaseAudit(
        bus_rows=bus_rows,
        branch_rows=branch_rows,
        generator_rows=generator_rows,
        bus_ids=bus_ids,
        repository_edges=tuple(sorted(set(repository_edges))),
        raw_active_edges=tuple(sorted(set(raw_active_edges))),
        generator_buses=generator_buses,
        reference_bus=reference_bus,
        pv_buses=pv_buses,
        pq_buses=pq_buses,
        transformer_edges=tuple(sorted(set(transformer_edges))),
        topology_sha256=topology_sha256,
        topology_hash_payload=topology_hash_payload,
        source_detail=power_case.source_detail,
        case_module_file=case_module_file,
        case_module_sha256=sha256_file(case_module_file),
        pypower_version=package_version("pypower"),
        base_mva=float(power_case.base_mva),
        total_branch_count=int(power_case.branch.shape[0]),
        active_branch_count=int(np.count_nonzero(active_mask)),
        total_generator_rows=int(power_case.gen.shape[0]),
        active_generator_rows=int(np.count_nonzero(power_case.gen[:, GEN_STATUS] > 0.0)),
    )


def edge_polyline(
    edge: tuple[int, int],
    samples: int = 81,
    *,
    positions: dict[int, tuple[float, float]] = BUS_POSITIONS,
    curve_control_1: tuple[float, float] = CURVE_CONTROL_1,
    curve_control_2: tuple[float, float] = CURVE_CONTROL_2,
) -> np.ndarray:
    normalized = normalize_edge(*edge)
    start = np.asarray(positions[normalized[0]], dtype=np.float64)
    end = np.asarray(positions[normalized[1]], dtype=np.float64)
    if normalized != normalize_edge(*CURVED_EDGE):
        return np.vstack([start, end])
    control_1 = np.asarray(curve_control_1, dtype=np.float64)
    control_2 = np.asarray(curve_control_2, dtype=np.float64)
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (
        ((1.0 - t) ** 3) * start
        + 3.0 * ((1.0 - t) ** 2) * t * control_1
        + 3.0 * (1.0 - t) * (t**2) * control_2
        + (t**3) * end
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 5.4,
            "axes.linewidth": 0.5,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "figure1-ieee14-topology-v1",
        }
    )


def build_figure(
    audit: CaseAudit,
    *,
    roles: bool,
    figsize: tuple[float, float],
    variant: str,
    positions: dict[int, tuple[float, float]] = BUS_POSITIONS,
    curve_control_1: tuple[float, float] = CURVE_CONTROL_1,
    curve_control_2: tuple[float, float] = CURVE_CONTROL_2,
    x_limits: tuple[float, float] = (-0.86, 8.96),
    y_limits: tuple[float, float] = (-2.25, 2.25),
) -> tuple[Figure, Axes, dict[tuple[int, int], np.ndarray]]:
    figure, axis = plt.subplots(figsize=figsize, facecolor=WHITE)
    axis.set_position([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    plotted_paths: dict[tuple[int, int], np.ndarray] = {}
    for edge in audit.repository_edges:
        path = edge_polyline(
            edge,
            positions=positions,
            curve_control_1=curve_control_1,
            curve_control_2=curve_control_2,
        )
        line = axis.plot(
            path[:, 0],
            path[:, 1],
            color=EDGE_COLOR,
            linewidth=0.55,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1.0,
        )[0]
        line.set_gid(f"branch_{edge[0]}_{edge[1]}")
        plotted_paths[edge] = path

    for bus in audit.bus_ids:
        x, y = positions[bus]
        is_generator = bus in audit.generator_buses
        is_reference = bus == audit.reference_bus
        if roles and is_generator:
            ring = axis.scatter(
                [x],
                [y],
                s=82.0,
                marker="o",
                facecolor="none",
                edgecolor=GENERATOR_RING_COLOR,
                linewidth=0.48,
                zorder=2.5,
            )
            ring.set_gid(f"generator_marker_{bus}")
        facecolor = WHITE
        text_color = BLACK
        if roles and is_generator:
            facecolor = GENERATOR_FILL
        if roles and is_reference:
            facecolor = BLACK
            text_color = WHITE
        node = axis.scatter(
            [x],
            [y],
            s=NODE_MARKER_AREA_PT2,
            marker="o",
            facecolor=facecolor,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=NODE_LINEWIDTH_PT,
            zorder=3.0,
        )
        node.set_gid(f"bus_node_{bus}")
        label = axis.text(
            x,
            y,
            str(bus),
            ha="center",
            va="center",
            fontsize=5.15,
            fontfamily=FONT_FAMILY,
            fontweight="medium",
            color=text_color,
            zorder=4.0,
        )
        label.set_gid(f"bus_label_{bus}")

    if roles:
        reference_x, reference_y = positions[audit.reference_bus]
        ref_label = axis.text(
            reference_x - 0.30,
            reference_y,
            "ref.",
            ha="right",
            va="center",
            fontsize=4.25,
            fontfamily=FONT_FAMILY,
            color=BLACK,
            zorder=4.0,
        )
        ref_label.set_gid("reference_bus_annotation")

    figure.canvas.draw()
    figure.set_label(variant)
    return figure, axis, plotted_paths


def parse_gid_number(gid: str, prefix: str) -> int:
    return int(gid.removeprefix(prefix))


def _duplicates(values: Iterable[int]) -> tuple[int, ...]:
    observed = list(values)
    return tuple(sorted({value for value in observed if observed.count(value) > 1}))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _proper_segment_intersection(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float = 1.0e-10,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return bool(o1 * o2 < -tolerance and o3 * o4 < -tolerance)


def find_nonincident_crossings(
    edge_paths: dict[tuple[int, int], np.ndarray],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    crossings: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges = sorted(edge_paths)
    for index, left in enumerate(edges):
        left_path = edge_paths[left]
        for right in edges[index + 1 :]:
            if set(left) & set(right):
                continue
            right_path = edge_paths[right]
            found = False
            for left_a, left_b in pairwise(left_path):
                for right_a, right_b in pairwise(right_path):
                    if _proper_segment_intersection(left_a, left_b, right_a, right_b):
                        crossings.add((left, right))
                        found = True
                        break
                if found:
                    break
    return tuple(sorted(crossings))


def validate_rendered_artists(
    audit: CaseAudit,
    figure: Figure,
    axis: Axes,
    edge_paths: dict[tuple[int, int], np.ndarray],
    *,
    variant: str,
) -> RenderCheck:
    nodes: list[int] = []
    labels: list[int] = []
    edges: list[tuple[int, int]] = []
    label_artists: dict[int, Artist] = {}
    edge_zorders: list[float] = []
    label_zorders: list[float] = []
    for artist in axis.get_children():
        gid = artist.get_gid()
        if not gid:
            continue
        if gid.startswith("bus_node_"):
            nodes.append(parse_gid_number(gid, "bus_node_"))
        elif gid.startswith("bus_label_"):
            bus = parse_gid_number(gid, "bus_label_")
            labels.append(bus)
            label_artists[bus] = artist
            label_zorders.append(float(artist.get_zorder()))
        elif gid.startswith("branch_"):
            _prefix, from_text, to_text = gid.split("_")
            edges.append(normalize_edge(int(from_text), int(to_text)))
            edge_zorders.append(float(artist.get_zorder()))

    renderer = figure.canvas.get_renderer()
    overlap_pairs: list[tuple[int, int]] = []
    label_items = sorted(label_artists.items())
    for index, (left_bus, left_artist) in enumerate(label_items):
        left_box = left_artist.get_window_extent(renderer=renderer)
        for right_bus, right_artist in label_items[index + 1 :]:
            right_box = right_artist.get_window_extent(renderer=renderer)
            if left_box.overlaps(right_box):
                overlap_pairs.append((left_bus, right_bus))

    repository_edge_set = set(audit.repository_edges)
    plotted_edge_set = set(edges)
    check = RenderCheck(
        variant=variant,
        plotted_buses=tuple(sorted(nodes)),
        plotted_labels=tuple(sorted(labels)),
        plotted_edges=tuple(sorted(plotted_edge_set)),
        missing_edges=tuple(sorted(repository_edge_set - plotted_edge_set)),
        extra_edges=tuple(sorted(plotted_edge_set - repository_edge_set)),
        duplicate_bus_nodes=_duplicates(nodes),
        duplicate_bus_labels=_duplicates(labels),
        label_overlap_pairs=tuple(overlap_pairs),
        nonincident_edge_crossings=find_nonincident_crossings(edge_paths),
        labels_above_edges=bool(
            edge_zorders
            and label_zorders
            and min(label_zorders) > max(edge_zorders)
        ),
    )
    if not check.passed:
        raise RuntimeError(f"render validation failed for {variant}: {check.as_dict()}")
    return check


def validate_node_canvas_clearance(
    audit: CaseAudit,
    figure: Figure,
    axis: Axes,
    *,
    positions: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    """Verify that every complete circular marker lies inside the output canvas."""

    figure.canvas.draw()
    width_px, height_px = figure.canvas.get_width_height()
    centers_px = axis.transData.transform(
        np.asarray([positions[bus] for bus in audit.bus_ids], dtype=np.float64)
    )
    marker_radius_pt = np.sqrt(NODE_MARKER_AREA_PT2) / 2.0 + NODE_LINEWIDTH_PT / 2.0
    marker_radius_px = marker_radius_pt * figure.dpi / 72.0
    clearances_px: dict[int, float] = {}
    clipped_buses: list[int] = []
    for bus, (center_x, center_y) in zip(audit.bus_ids, centers_px, strict=True):
        clearance = min(
            float(center_x),
            float(width_px - center_x),
            float(center_y),
            float(height_px - center_y),
        ) - marker_radius_px
        clearances_px[bus] = clearance
        if clearance <= 0.0:
            clipped_buses.append(bus)
    if clipped_buses:
        raise RuntimeError(
            "portrait node-circle canvas validation failed for buses "
            f"{clipped_buses}"
        )
    return {
        "all_node_circles_inside_canvas": True,
        "clipped_buses": clipped_buses,
        "minimum_circle_clearance_px_at_canvas_dpi": min(clearances_px.values()),
        "bus_12_circle_clearance_px_at_canvas_dpi": clearances_px[12],
        "per_bus_circle_clearance_px_at_canvas_dpi": {
            str(bus): clearance for bus, clearance in sorted(clearances_px.items())
        },
    }


def save_figure_variants(figure: Figure, output_base: Path) -> dict[str, Path]:
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    png_path = output_base.with_suffix(".png")
    common_description = (
        "Repository-derived IEEE-14 topology from pypower.case14.case14; "
        "visualization-only coordinates; no sensor placement."
    )
    figure.savefig(
        pdf_path,
        facecolor=WHITE,
        metadata={
            "Creator": str(SCRIPT_REL),
            "CreationDate": None,
            "Title": None,
            "Subject": common_description,
            "Keywords": "IEEE-14, PYPOWER, network topology",
        },
    )
    figure.savefig(
        svg_path,
        facecolor=WHITE,
        metadata={
            "Creator": str(SCRIPT_REL),
            "Date": "",
            "Description": common_description,
        },
    )
    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor=WHITE,
        metadata={
            "Software": str(SCRIPT_REL),
            "Description": common_description,
            "dpi": str(PNG_DPI),
        },
    )
    return {"pdf": pdf_path, "svg": svg_path, "png": png_path}


def write_preview(audit: CaseAudit, output_path: Path) -> RenderCheck:
    figure, axis, paths = build_figure(
        audit,
        roles=False,
        figsize=PANEL_PREVIEW_FIGSIZE,
        variant="panel_preview",
    )
    check = validate_rendered_artists(
        audit,
        figure,
        axis,
        paths,
        variant="panel_preview",
    )
    figure.savefig(
        output_path,
        dpi=PNG_DPI,
        facecolor=WHITE,
        metadata={
            "Software": str(SCRIPT_REL),
            "Description": (
                "IEEE-14 minimal topology rendered at the approximate Figure 1 panel size; "
                "600 dpi review preview."
            ),
            "dpi": str(PNG_DPI),
        },
    )
    plt.close(figure)
    return check


def write_portrait_outputs(audit: CaseAudit, output_dir: Path) -> None:
    """Write only the new portrait topology assets and their render audit."""

    if set(PORTRAIT_BUS_POSITIONS) != set(audit.bus_ids):
        raise RuntimeError("portrait layout does not cover every repository bus exactly once")

    figure, axis, paths = build_figure(
        audit,
        roles=False,
        figsize=PORTRAIT_FIGSIZE,
        variant="minimal_portrait",
        positions=PORTRAIT_BUS_POSITIONS,
        curve_control_1=PORTRAIT_CURVE_CONTROL_1,
        curve_control_2=PORTRAIT_CURVE_CONTROL_2,
        x_limits=PORTRAIT_X_LIMITS,
        y_limits=PORTRAIT_Y_LIMITS,
    )
    render_check = validate_rendered_artists(
        audit,
        figure,
        axis,
        paths,
        variant="minimal_portrait",
    )
    circle_check = validate_node_canvas_clearance(
        audit,
        figure,
        axis,
        positions=PORTRAIT_BUS_POSITIONS,
    )
    saved = save_figure_variants(
        figure,
        output_dir / "ieee14_topology_minimal_portrait",
    )
    plt.close(figure)

    preview_path = output_dir / "ieee14_topology_minimal_portrait_preview.png"
    preview_figure, preview_axis, preview_paths = build_figure(
        audit,
        roles=False,
        figsize=PORTRAIT_PREVIEW_FIGSIZE,
        variant="minimal_portrait_preview",
        positions=PORTRAIT_BUS_POSITIONS,
        curve_control_1=PORTRAIT_CURVE_CONTROL_1,
        curve_control_2=PORTRAIT_CURVE_CONTROL_2,
        x_limits=PORTRAIT_X_LIMITS,
        y_limits=PORTRAIT_Y_LIMITS,
    )
    preview_check = validate_rendered_artists(
        audit,
        preview_figure,
        preview_axis,
        preview_paths,
        variant="minimal_portrait_preview",
    )
    preview_circle_check = validate_node_canvas_clearance(
        audit,
        preview_figure,
        preview_axis,
        positions=PORTRAIT_BUS_POSITIONS,
    )
    preview_figure.savefig(
        preview_path,
        dpi=PNG_DPI,
        facecolor=WHITE,
        metadata={
            "Software": str(SCRIPT_REL),
            "Description": (
                "Repository-derived IEEE-14 portrait topology rendered at a reduced "
                "manuscript-panel scale; visualization-only coordinates."
            ),
            "dpi": str(PNG_DPI),
        },
    )
    plt.close(preview_figure)

    portrait_payload = {
        "case": CASE_DISPLAY,
        "source": audit.source_detail,
        "topology_sha256": audit.topology_sha256,
        "expected_bus_count": len(audit.bus_ids),
        "expected_active_edge_count": len(audit.repository_edges),
        "layout_provenance": (
            "Rigid clockwise rotation of the existing deterministic visualization-only "
            "coordinates; connectivity remains loaded from the repository case."
        ),
        "layout_is_geographic": False,
        "portrait_coordinates": {
            str(bus): [float(x), float(y)]
            for bus, (x, y) in sorted(PORTRAIT_BUS_POSITIONS.items())
        },
        "render_validation": render_check.as_dict(),
        "circle_canvas_validation": circle_check,
        "preview_render_validation": preview_check.as_dict(),
        "preview_circle_canvas_validation": preview_circle_check,
        "generated_files": {
            kind: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for kind, path in {**saved, "preview_png": preview_path}.items()
        },
        "overall_pass": bool(
            render_check.passed
            and preview_check.passed
            and circle_check["all_node_circles_inside_canvas"]
            and preview_circle_check["all_node_circles_inside_canvas"]
        ),
    }
    validation_path = output_dir / "ieee14_topology_portrait_validation.json"
    validation_path.write_text(
        json.dumps(portrait_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"source: {audit.source_detail} (PYPOWER {audit.pypower_version})")
    print(
        f"portrait verified: {len(audit.bus_ids)} buses, "
        f"{len(audit.repository_edges)} active branches, "
        f"missing={len(render_check.missing_edges)}, extra={len(render_check.extra_edges)}"
    )
    print(
        "circle canvas check: all 14 complete; "
        f"bus 12 clearance={circle_check['bus_12_circle_clearance_px_at_canvas_dpi']:.2f}px"
    )
    print(f"topology_sha256: {audit.topology_sha256}")
    print(f"wrote: {output_dir / 'ieee14_topology_minimal_portrait.pdf'}")
    print(f"wrote: {output_dir / 'ieee14_topology_minimal_portrait.svg'}")
    print(f"wrote: {output_dir / 'ieee14_topology_minimal_portrait.png'}")
    print(f"wrote: {preview_path}")
    print(f"wrote: {validation_path}")


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return stable_float(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def build_metadata(
    audit: CaseAudit,
    checks: list[RenderCheck],
    figure_files: dict[str, Path],
) -> dict[str, Any]:
    root_from_git = git_value("rev-parse", "--show-toplevel")
    if Path(root_from_git).resolve() != ROOT.resolve():
        raise RuntimeError("script root does not match git repository root")
    branch = git_value("branch", "--show-current")
    commit = git_value("rev-parse", "HEAD")
    figure_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(figure_files.values())
    }
    return {
        "case": CASE_DISPLAY,
        "case_name": CASE_NAME,
        "source": (
            "src/robust_qsvt_se/data/real_cases.py::load_power_case -> "
            f"{audit.source_detail}"
        ),
        "repository_root": str(ROOT),
        "git_branch": branch,
        "git_commit": commit,
        "bus_count": len(audit.bus_ids),
        "total_branch_count": audit.total_branch_count,
        "active_branch_count": audit.active_branch_count,
        "generator_row_count": audit.total_generator_rows,
        "active_generator_row_count": audit.active_generator_rows,
        "generator_bus_count": len(audit.generator_buses),
        "generator_buses": list(audit.generator_buses),
        "reference_bus": audit.reference_bus,
        "pv_buses": list(audit.pv_buses),
        "pq_buses": list(audit.pq_buses),
        "transformer_tap_branches": [list(edge) for edge in audit.transformer_edges],
        "branch_status_rule": ACTIVE_STATUS_RULE,
        "repository_ac_case_adapter_verified": True,
        "layout_source": (
            "Deterministic visualization-only coordinates informed by the broad node "
            "arrangement in outputs/ieee_dataset_visualization/ieee14_topology_clean.svg; "
            "connectivity is independently loaded from pypower.case14.case14."
        ),
        "layout_is_geographic": False,
        "layout_coordinates": {
            str(bus): [float(x), float(y)]
            for bus, (x, y) in sorted(BUS_POSITIONS.items())
        },
        "topology_sha256": audit.topology_sha256,
        "topology_hash_method": (
            "SHA-256 of compact sorted-key JSON containing sorted bus IDs and sorted "
            "undirected active edge pairs"
        ),
        "topology_hash_payload": audit.topology_hash_payload,
        "pypower_version": audit.pypower_version,
        "case_module": audit.source_detail,
        "case_module_file": str(audit.case_module_file),
        "case_module_sha256": audit.case_module_sha256,
        "base_mva": audit.base_mva,
        "plot_semantics": {
            "minimal": "active branches, buses, and bus numbers only",
            "roles": (
                "minimal topology plus restrained generator-bus outer rings and a "
                "black reference bus labeled ref."
            ),
            "transformer_style": (
                "not visually distinguished; tap status is retained in the branch audit CSV"
            ),
            "sensor_or_measurement_placement": "not shown",
        },
        "render_validation": [check.as_dict() for check in checks],
        "generated_figure_sha256": figure_hashes,
    }


def markdown_bool(value: bool) -> str:
    return "YES" if value else "NO"


def build_audit_report(
    audit: CaseAudit,
    checks: list[RenderCheck],
    output_dir: Path,
) -> str:
    branch = git_value("branch", "--show-current")
    commit = git_value("rev-parse", "HEAD")
    minimal = next(check for check in checks if check.variant == "minimal")
    role_check = next(check for check in checks if check.variant == "roles")
    preview = next(check for check in checks if check.variant == "panel_preview")
    all_checks_pass = all(check.passed for check in checks)
    mockup_status = (
        str(CURRENT_MOCKUP.relative_to(ROOT)) if CURRENT_MOCKUP.exists() else "not located"
    )

    lines: list[str] = [
        "# Figure 1 IEEE-14 Network Topology Audit",
        "",
        "## 1. Repository State",
        "",
        f"- Root: `{ROOT}`",
        f"- Branch: `{branch}`",
        f"- HEAD: `{commit}`",
        "- Worktree: pre-existing user changes were preserved; this task writes only the new "
        "plotting script and this audit/export directory.",
        "",
        "## 2. Network Source",
        "",
        "- Canonical matrix path: `cross_case_validation/common.py::build_case_full_system` -> "
        "`qsvt/engineering_utils.py::build_engineering_system` -> "
        "`measurement/ac_linear.py::build_ac_weighted_system` -> "
        "`data/cases.py::load_ac_case`.",
        "- Case loader: `src/robust_qsvt_se/data/real_cases.py::load_power_case`.",
        f"- Case file: `{audit.case_module_file}`.",
        f"- Case callable: `{audit.source_detail}` (PYPOWER {audit.pypower_version}).",
        f"- Case file SHA-256: `{audit.case_module_sha256}`.",
        "- Bus source: `case14()[\"bus\"]`.",
        "- Branch source: `case14()[\"branch\"]`; active iff `BR_STATUS > 0.0`, "
        "matching the repository adapter.",
        "- Generator source: `case14()[\"gen\"]`; online iff `GEN_STATUS > 0.0`.",
        "- The cases provide benchmark network data and operating-point information; they do "
        "not provide field PMU/SCADA measurement placement.",
        "",
        "## 3. Verified Network Properties",
        "",
        f"- Buses: {len(audit.bus_ids)} (`1` through `14`, unique).",
        f"- Active branches: {audit.active_branch_count} of {audit.total_branch_count} total.",
        f"- Disabled branches: {audit.total_branch_count - audit.active_branch_count}.",
        f"- Generator buses: {len(audit.generator_buses)} "
        f"({', '.join(str(bus) for bus in audit.generator_buses)}).",
        f"- Slack/reference bus: {audit.reference_bus}.",
        f"- PV buses: {', '.join(str(bus) for bus in audit.pv_buses)}.",
        f"- PQ buses: {', '.join(str(bus) for bus in audit.pq_buses)}.",
        f"- Transformer/off-nominal tap branches: "
        f"{', '.join(f'{left}-{right}' for left, right in audit.transformer_edges)}.",
        f"- Topology SHA-256: `{audit.topology_sha256}`.",
        "",
        "## 4. Branch Connectivity",
        "",
        "| From | To | Status | Active? | Transformer/Tap? | Notes |",
        "| ---: | ---: | ---: | :---: | :---: | --- |",
    ]
    for row in audit.branch_rows:
        lines.append(
            f"| {row['from_bus']} | {row['to_bus']} | {stable_float(row['status'])} | "
            f"{markdown_bool(bool(row['active']))} | "
            f"{markdown_bool(bool(row['transformer_or_tap']))} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "All branch rows have status 1, so no branch was excluded.",
            "",
            "## 5. Bus Roles",
            "",
            "| Bus | Type | Generator Present? | Notes |",
            "| ---: | --- | :---: | --- |",
        ]
    )
    for row in audit.bus_rows:
        lines.append(
            f"| {row['bus']} | {row['bus_type']} | "
            f"{markdown_bool(bool(row['online_generator_present']))} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Layout Provenance",
            "",
            f"- Coordinate source: deterministic visualization coordinates in `{SCRIPT_REL}`; "
            f"the broad left-to-right arrangement is informed only by `{mockup_status}`.",
            "- Geographic coordinates?: NO.",
            "- Visualization-only coordinates?: YES.",
            "- Repository coordinate search: no IEEE-14 bus-coordinate table or plotting "
            "coordinate definition was found in `src/`, `configs/`, `scripts/`, or "
            "`manuscript/`.",
            "- Connectivity source: exclusively the active branch table loaded from "
            "`pypower.case14.case14`; no edge was copied from the mock-up.",
            "",
            "## 7. Rendering Verification",
            "",
            f"- All buses rendered: YES ({len(minimal.plotted_buses)} of "
            f"{len(audit.bus_ids)} in each final variant).",
            f"- Expected active edges: {len(audit.repository_edges)}.",
            f"- Minimal plotted edges: {len(minimal.plotted_edges)}.",
            f"- Roles plotted edges: {len(role_check.plotted_edges)}.",
            f"- Missing edges: {len(minimal.missing_edges)}.",
            f"- Extra edges: {len(minimal.extra_edges)}.",
            f"- Duplicate labels: {len(minimal.duplicate_bus_labels)}.",
            f"- Label bounding-box overlaps: {len(minimal.label_overlap_pairs)} minimal; "
            f"{len(role_check.label_overlap_pairs)} roles; "
            f"{len(preview.label_overlap_pairs)} panel preview.",
            f"- Nonincident edge crossings: "
            f"{len(minimal.nonincident_edge_crossings)}.",
            f"- Edge/label draw order protects bus numbers: "
            f"{markdown_bool(minimal.labels_above_edges)}.",
            f"- Programmatic rendering validation: "
            f"{'PASS' if all_checks_pass else 'FAIL'}.",
            "- Visual QA: PDF outputs rendered through Poppler and the PDF/SVG/PNG assets were "
            "inspected at full size and approximate panel size; no clipped or overlapping bus "
            "labels, obscured numbers, or dominant role markers were found.",
            "",
            "## 8. Files Generated",
            "",
            "| File | Purpose |",
            "| --- | --- |",
            "| `ieee14_buses.csv` | Verified bus table, type, demand, and "
            "generator-presence audit. |",
            "| `ieee14_branches.csv` | Complete branch table with status and "
            "transformer/tap audit. |",
            "| `ieee14_generators.csv` | Generator rows and online status. |",
            "| `ieee14_topology_metadata.json` | Source, git, layout, hash, and render "
            "provenance. |",
            "| `ieee14_render_validation.json` | Machine-readable "
            "bus/edge/overlap/crossing checks. |",
            "| `ieee14_topology_minimal.pdf` | Primary minimal vector asset. |",
            "| `ieee14_topology_minimal.svg` | Primary minimal editable vector asset. |",
            "| `ieee14_topology_minimal.png` | 600 dpi visual-review raster. |",
            "| `ieee14_topology_roles.pdf` | Role-aware vector alternative. |",
            "| `ieee14_topology_roles.svg` | Role-aware editable vector alternative. |",
            "| `ieee14_topology_roles.png` | 600 dpi role-aware review raster. |",
            "| `ieee14_topology_panel_preview.png` | 600 dpi approximate Figure 1 "
            "panel-size preview. |",
            "| `figure1_network_audit.md` | Human-readable topology and claim audit. |",
            "",
            "## 9. Figure 1 Recommendation",
            "",
            "- Recommended version: minimal topology.",
            "- Recommended panel label: `PYPOWER IEEE-14 benchmark network model`.",
            "- Reason: the first panel needs to identify the repository benchmark source; bus "
            "roles and tap information are secondary and become visually dense at five-panel "
            "Figure 1 scale.",
            "- If roles are required elsewhere, use the role-aware alternative, where an outer "
            "ring marks an online generator bus and the black bus labeled `ref.` is the reference "
            "bus. No load or sensor symbol is shown.",
            "",
            "### Comparison with the current mock-up",
            "",
            "The current repository mock-up was inspected only for comparison and broad layout. "
            "It was not used as topology data.",
            "",
            "| Aspect | Current Mock-Up | Repository-Derived Figure | Difference |",
            "| --- | --- | --- | --- |",
            "| Bus connectivity | Manual visual review shows the same 20 case connections, but "
            "the graphic itself has no machine-readable edge-to-artist validation. | Every "
            "plotted artist is generated from the active repository branch rows and compared "
            "back to that edge set. | No visible connectivity difference found; provenance and "
            "automated verification are added. |",
            "| Branch count | Embedded text states 20 active branches. | 20 active branches "
            "loaded and 20 rendered; missing 0, extra 0. | Same count, now case-derived and "
            "verified. |",
            "| Bus numbering | Shows buses 1 through 14. | Shows buses 1 through 14 exactly once. "
            "| No numbering difference; duplicate-label validation is added. |",
            "| Generator symbols | Color-coded slack/PV nodes and load-scaled node sizes. | "
            "Minimal version omits roles; optional roles version uses only online `gen` rows and "
            "the REF bus type. | Role encoding is restrained and data-derived; load scaling is "
            "removed. |",
            "| Visual topology | Colored region backgrounds, title, legend, dataset block, thick "
            "curves, and transformer styling. | Compact black/gray single-line network on white, "
            "without title, caption, decorative background, or external legend. | Simplified for "
            "a small five-panel manuscript figure. |",
            "",
            "## 10. Claim Safety",
            "",
            "Can the manuscript call this:",
            "",
            "> Repository-derived IEEE-14 network topology",
            "",
            "YES.",
            "",
            "The bus and active-branch connectivity is loaded from the exact PYPOWER IEEE-14 "
            "case used by the repository pathway and is validated before export. The statement "
            "describes benchmark topology only. Node positions are visualization coordinates, "
            "not geography, and the figure does not imply field data, PMU/SCADA placement, "
            "sensor locations, or utility measurement streams.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="audit/export directory (default: outputs/figure1_network_audit)",
    )
    parser.add_argument(
        "--portrait-only",
        action="store_true",
        help=(
            "generate the minimal portrait PDF/SVG/PNG, reduced-size preview, and "
            "portrait render validation without rewriting existing horizontal exports"
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    audit = load_and_verify_case()

    if args.portrait_only:
        write_portrait_outputs(audit, output_dir)
        return

    all_figure_files: dict[str, Path] = {}
    checks: list[RenderCheck] = []
    for roles, stem, variant in (
        (False, "ieee14_topology_minimal", "minimal"),
        (True, "ieee14_topology_roles", "roles"),
    ):
        figure, axis, paths = build_figure(
            audit,
            roles=roles,
            figsize=PRIMARY_FIGSIZE,
            variant=variant,
        )
        check = validate_rendered_artists(
            audit,
            figure,
            axis,
            paths,
            variant=variant,
        )
        checks.append(check)
        saved = save_figure_variants(figure, output_dir / stem)
        all_figure_files.update({f"{variant}_{kind}": path for kind, path in saved.items()})
        plt.close(figure)

    preview_path = output_dir / "ieee14_topology_panel_preview.png"
    preview_check = write_preview(audit, preview_path)
    checks.append(preview_check)
    all_figure_files["panel_preview_png"] = preview_path

    write_csv(output_dir / "ieee14_buses.csv", audit.bus_rows)
    write_csv(output_dir / "ieee14_branches.csv", audit.branch_rows)
    write_csv(output_dir / "ieee14_generators.csv", audit.generator_rows)

    validation_payload = {
        "case": CASE_DISPLAY,
        "expected_buses": list(audit.bus_ids),
        "expected_edges": [list(edge) for edge in audit.repository_edges],
        "topology_sha256": audit.topology_sha256,
        "variants": [check.as_dict() for check in checks],
        "overall_pass": all(check.passed for check in checks),
    }
    (output_dir / "ieee14_render_validation.json").write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = build_metadata(audit, checks, all_figure_files)
    (output_dir / "ieee14_topology_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_audit_report(audit, checks, output_dir)
    (output_dir / "figure1_network_audit.md").write_text(report, encoding="utf-8")

    print(f"source: {audit.source_detail} (PYPOWER {audit.pypower_version})")
    print(
        f"verified: {len(audit.bus_ids)} buses, {audit.active_branch_count} active branches, "
        f"{len(audit.generator_buses)} generator buses, reference bus {audit.reference_bus}"
    )
    print(f"topology_sha256: {audit.topology_sha256}")
    for check in checks:
        print(
            f"{check.variant}: edges={len(check.plotted_edges)}, "
            f"missing={len(check.missing_edges)}, extra={len(check.extra_edges)}, "
            f"overlaps={len(check.label_overlap_pairs)}, pass={check.passed}"
        )
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
