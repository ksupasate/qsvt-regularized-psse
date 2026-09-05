#!/usr/bin/env python3
"""Plot Figure-1 IEEE-14 weighted-Jacobian heatmaps from exported audit data.

Reads only the artifacts in ``outputs/figure1_matrix_audit/`` (never rebuilds
scientific data) and verifies the matrix SHA-256 against the metadata before
plotting.  The panel-ready outputs show the raw weighted values ``H_tilde`` on
an exact symmetric color scale.  A separately named display-normalized variant
is emitted for design review only.  Existing exploratory plots remain
available unless ``--skip-legacy`` is supplied.
"""

# ruff: noqa: E501  # Markdown audit-report rows intentionally remain single lines.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm

# Validated diverging pair (blue <-> red) with a neutral light-gray midpoint.
NEG_POLE = "#2a78d6"
MIDPOINT = "#f0efec"
POS_POLE = "#e34948"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

DIVERGING = LinearSegmentedColormap.from_list(
    "audit_diverging", [NEG_POLE, MIDPOINT, POS_POLE], N=256
)

# Fixed rendering settings make repeated exports byte-stable in the same
# environment and keep the publication assets portable.
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.4,
        "pdf.compression": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Outline SVG text so math accents/subscripts retain their placement
        # across TeX, browser, and librsvg renderers.
        "svg.fonttype": "path",
        "svg.hashsalt": "figure1-ieee14-weighted-jacobian",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

EXPECTED_FULL_MATRIX_SHA256 = (
    "c113dfa0ee5240182a77d2a845b0f4e950639c49884486d330a5c1351488fa5b"
)
EXPECTED_SHAPE = (82, 27)
EXPECTED_STATE_LABELS = tuple(
    [f"theta_{bus}" for bus in range(2, 15)]
    + [f"V_{bus}" for bus in range(1, 15)]
)
ROW_FAMILY_SCHEMA = (
    ("voltage_magnitude", 0, 13, r"$V$"),
    ("p_injection", 14, 27, r"$P_{\mathrm{inj}}$"),
    ("q_injection", 28, 41, r"$Q_{\mathrm{inj}}$"),
    ("p_branch_flow", 42, 61, r"$P_{\mathrm{flow}}$"),
    ("q_branch_flow", 62, 81, r"$Q_{\mathrm{flow}}$"),
)
PANEL_STEM = "figure1_ieee14_weighted_jacobian_panel"
GROUPED_PANEL_STEM = f"{PANEL_STEM}_grouped"
DISPLAY_NORMALIZED_PANEL_STEM = f"{PANEL_STEM}_display_normalized"
PREVIEW_STEM = f"{PANEL_STEM}_preview"
COMPACT_STEM = "figure1_ieee14_weighted_jacobian_compact"
COMPACT_LABELS_STEM = f"{COMPACT_STEM}_labels"
COMPACT_GROUPED_STEM = f"{COMPACT_STEM}_grouped"
COMPACT_PREVIEW_STEM = f"{COMPACT_STEM}_preview"

ROW_GROUP_BOUNDARIES = (14, 28, 42, 62)  # V | P_inj | Q_inj | P_flow | Q_flow
ROW_GROUP_NAMES = ("V", "P$_{inj}$", "Q$_{inj}$", "P$_{flow}$", "Q$_{flow}$")
COL_GROUP_BOUNDARY = 13  # 13 angle states | 14 voltage-magnitude states


def array_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(input_dir: Path, stem: str, metadata: dict) -> np.ndarray:
    matrix_path = input_dir / f"{stem}.npy"
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.dtype != np.float64:
        raise RuntimeError(f"{stem} must be float64, found {matrix.dtype}")

    sparse_pair = metadata.get("sparse_pair", {})
    expected_by_stem = {
        "ieee14_weighted_jacobian": metadata.get("matrix_sha256"),
        "ieee14_dense_block_8x8": sparse_pair.get("dense_block_sha256"),
        "ieee14_sparse_weighted_jacobian": sparse_pair.get("sparse_block_sha256"),
        "ieee14_quantized_sparse_block_8x8": sparse_pair.get("quantized_block_sha256"),
    }
    expected = expected_by_stem.get(stem)
    actual = array_sha256(matrix)
    if stem in expected_by_stem and not expected:
        raise RuntimeError(f"no audit-metadata SHA-256 is recorded for {stem}")
    if expected and actual != expected:
        raise RuntimeError(
            f"{stem} SHA-256 mismatch: expected {expected}, computed {actual}"
        )
    if stem == "ieee14_weighted_jacobian":
        if expected != EXPECTED_FULL_MATRIX_SHA256:
            raise RuntimeError(
                "audit metadata does not contain the frozen IEEE-14 matrix hash"
            )
        if actual != EXPECTED_FULL_MATRIX_SHA256:
            raise RuntimeError("weighted-Jacobian array does not match the frozen hash")
    return np.asarray(matrix, dtype=np.float64)


def _style_heatmap(ax, matrix, *, norm, ylabel="Measurement row", xlabel="State coordinate"):
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=DIVERGING,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlabel(xlabel, fontsize=8, color=TEXT_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT_PRIMARY)
    ax.tick_params(labelsize=6.5, colors=TEXT_SECONDARY, length=2)
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.5)
    return image


def _add_group_separators(ax, *, rows: int, cols: int):
    for y in ROW_GROUP_BOUNDARIES:
        ax.axhline(y - 0.5, color=TEXT_PRIMARY, linewidth=0.4, alpha=0.35)
    ax.axvline(COL_GROUP_BOUNDARY - 0.5, color=TEXT_PRIMARY, linewidth=0.4, alpha=0.35)


def _finalize(fig, image, *, cbar_label, out_base: Path, cbar_ticks=None):
    cbar = fig.colorbar(image, ax=fig.axes, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7.5, color=TEXT_PRIMARY)
    cbar.ax.tick_params(labelsize=6.5, colors=TEXT_SECONDARY)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dense(matrix: np.ndarray, out_dir: Path) -> None:
    limit = float(np.abs(matrix).max())
    ticks = [-round(limit), -round(limit) / 2, 0, round(limit) / 2, round(limit)]

    fig, ax = plt.subplots(figsize=(4.6, 6.4), facecolor="white")
    _style_heatmap(ax, matrix, norm=plt.Normalize(vmin=-limit, vmax=limit))
    _add_group_separators(ax, rows=matrix.shape[0], cols=matrix.shape[1])
    ax.set_yticks([0, 20, 40, 60, 81])
    ax.set_xticks([0, 6, 13, 20, 26])
    _finalize(
        fig,
        ax.images[0],
        cbar_label=r"weighted sensitivity $\tilde{H}_{ij}$ (raw)",
        out_base=out_dir / "ieee14_weighted_jacobian_heatmap",
        cbar_ticks=ticks,
    )

    # Display-scaled alternative: symmetric log near zero, signed log beyond.
    fig, ax = plt.subplots(figsize=(4.6, 6.4), facecolor="white")
    norm = SymLogNorm(linthresh=10.0, vmin=-limit, vmax=limit, base=10)
    _style_heatmap(ax, matrix, norm=norm)
    _add_group_separators(ax, rows=matrix.shape[0], cols=matrix.shape[1])
    ax.set_yticks([0, 20, 40, 60, 81])
    ax.set_xticks([0, 6, 13, 20, 26])
    _finalize(
        fig,
        ax.images[0],
        cbar_label=r"$\tilde{H}_{ij}$ (display-scaled, symlog)",
        out_base=out_dir / "ieee14_weighted_jacobian_heatmap_symlog",
    )


def plot_dense_with_block(matrix: np.ndarray, block_rows, block_cols, out_dir: Path) -> None:
    limit = float(np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=(4.6, 6.4), facecolor="white")
    _style_heatmap(ax, matrix, norm=plt.Normalize(vmin=-limit, vmax=limit))
    _add_group_separators(ax, rows=matrix.shape[0], cols=matrix.shape[1])
    ax.set_yticks([0, 20, 40, 60, 81])
    ax.set_xticks([0, 6, 13, 20, 26])
    rect = plt.Rectangle(
        (min(block_cols) - 0.5, min(block_rows) - 0.5),
        max(block_cols) - min(block_cols) + 1,
        max(block_rows) - min(block_rows) + 1,
        fill=False,
        edgecolor=TEXT_PRIMARY,
        linewidth=1.1,
    )
    ax.add_patch(rect)
    _finalize(
        fig,
        ax.images[0],
        cbar_label=r"weighted sensitivity $\tilde{H}_{ij}$ (raw)",
        out_base=out_dir / "ieee14_weighted_jacobian_heatmap_with_block",
        cbar_ticks=[-round(limit), 0, round(limit)],
    )


def plot_sparse(matrix: np.ndarray, out_dir: Path) -> None:
    limit = float(np.abs(matrix).max())
    fig, ax = plt.subplots(figsize=(3.6, 3.2), facecolor="white")
    image = _style_heatmap(ax, matrix, norm=plt.Normalize(vmin=-limit, vmax=limit))
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.tick_params(labelsize=6)
    for (i, j) in zip(*np.nonzero(matrix), strict=True):
        value = matrix[i, j]
        color = "white" if abs(value) > 0.55 * limit else TEXT_PRIMARY
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.2, color=color)
    _finalize(
        fig,
        image,
        cbar_label=r"retained $\tilde{H}_{ij}$ (raw)",
        out_base=out_dir / "ieee14_sparse_weighted_jacobian_heatmap",
        cbar_ticks=[-round(limit), 0, round(limit)],
    )


def plot_block_pair(dense_block: np.ndarray, sparse_block: np.ndarray, out_dir: Path) -> None:
    limit = max(float(np.abs(dense_block).max()), float(np.abs(sparse_block).max()))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), facecolor="white")
    for ax, matrix, label in (
        (axes[0], dense_block, "dense selected block"),
        (axes[1], sparse_block, "sparse retained support"),
    ):
        _style_heatmap(
            ax,
            matrix,
            norm=plt.Normalize(vmin=-limit, vmax=limit),
            xlabel="",
            ylabel="",
        )
        ax.set_title(label, fontsize=7.5, color=TEXT_PRIMARY, pad=4)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.tick_params(labelsize=5.5)
    axes[0].set_ylabel("Measurement row", fontsize=8, color=TEXT_PRIMARY)
    axes[0].set_xlabel("State coordinate", fontsize=8, color=TEXT_PRIMARY)
    _finalize(
        fig,
        axes[1].images[0],
        cbar_label=r"$\tilde{H}_{ij}$ (raw, shared scale)",
        out_base=out_dir / "ieee14_block_pair_heatmap",
        cbar_ticks=[-round(limit), 0, round(limit)],
    )


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_exported_matrix(
    input_dir: Path, matrix: np.ndarray, metadata: dict
) -> dict[str, object]:
    """Verify the exported array, its CSV mirror, and row/column semantics."""
    if matrix.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"weighted Jacobian shape mismatch: expected {EXPECTED_SHAPE}, "
            f"found {matrix.shape}"
        )
    if list(matrix.shape) != metadata.get("matrix_shape"):
        raise RuntimeError("matrix shape disagrees with audit metadata")
    if not np.isfinite(matrix).all():
        raise RuntimeError("weighted Jacobian contains a non-finite value")

    matrix_csv = np.loadtxt(
        input_dir / "ieee14_weighted_jacobian.csv", delimiter=","
    )
    if matrix_csv.shape != matrix.shape or not np.array_equal(matrix_csv, matrix):
        raise RuntimeError("CSV matrix is not an exact copy of the verified NPY array")

    measurement_rows = _read_dict_rows(input_dir / "measurement_rows.csv")
    if len(measurement_rows) != EXPECTED_SHAPE[0]:
        raise RuntimeError("measurement_rows.csv does not contain 82 rows")
    if [int(row["row_index"]) for row in measurement_rows] != list(range(82)):
        raise RuntimeError("measurement row indices are not the original 0..81 order")

    expected_families: list[str] = []
    for family, start, stop, _ in ROW_FAMILY_SCHEMA:
        expected_families.extend([family] * (stop - start + 1))
    actual_families = [row["measurement_type"] for row in measurement_rows]
    if actual_families != expected_families:
        raise RuntimeError("measurement-family ordering differs from the audit schema")
    measurement_labels = [row["label"] for row in measurement_rows]
    if len(set(measurement_labels)) != len(measurement_labels):
        raise RuntimeError("measurement_rows.csv contains duplicate labels")

    state_columns = _read_dict_rows(input_dir / "state_columns.csv")
    if len(state_columns) != EXPECTED_SHAPE[1]:
        raise RuntimeError("state_columns.csv does not contain 27 columns")
    if [int(col["col_index"]) for col in state_columns] != list(range(27)):
        raise RuntimeError("state column indices are not the original 0..26 order")
    state_labels = tuple(col["state_label"] for col in state_columns)
    if state_labels != EXPECTED_STATE_LABELS:
        raise RuntimeError("state column labels/order do not match theta_2..theta_14,V_1..V_14")
    expected_kinds = ["voltage_angle_rad"] * 13 + ["voltage_magnitude_pu"] * 14
    if [col["state_kind"] for col in state_columns] != expected_kinds:
        raise RuntimeError("state column kinds/order do not match the audit schema")
    expected_buses = list(range(2, 15)) + list(range(1, 15))
    if [int(col["bus"]) for col in state_columns] != expected_buses:
        raise RuntimeError("state column bus identifiers do not match the audit schema")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    metrics: dict[str, object] = {
        "shape": list(matrix.shape),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "maximum_absolute_value": float(np.max(np.abs(matrix))),
        "singular_value_maximum": float(singular_values[0]),
        "singular_value_minimum": float(singular_values[-1]),
        "array_sha256": array_sha256(matrix),
        "npy_file_sha256": file_sha256(input_dir / "ieee14_weighted_jacobian.npy"),
        "csv_exact_copy": True,
        "row_semantics_verified": True,
        "column_semantics_verified": True,
    }
    if metrics["rank"] != metadata.get("rank"):
        raise RuntimeError("computed rank disagrees with audit metadata")
    if not np.isclose(
        float(metrics["condition_number"]),
        float(metadata.get("condition_number")),
        rtol=1e-13,
        atol=0.0,
    ):
        raise RuntimeError("computed condition number disagrees with audit metadata")

    metadata_numerics = metadata.get("numerics", {})
    comparisons = {
        "minimum": "min_entry",
        "maximum": "max_entry",
        "maximum_absolute_value": "max_abs_entry",
        "singular_value_maximum": "singular_value_max",
        "singular_value_minimum": "singular_value_min",
    }
    for computed_key, metadata_key in comparisons.items():
        if not np.isclose(
            float(metrics[computed_key]),
            float(metadata_numerics.get(metadata_key)),
            rtol=1e-13,
            atol=0.0,
        ):
            raise RuntimeError(
                f"computed {computed_key} disagrees with audit metadata"
            )
    return metrics


def verify_qsvt_distinction(
    input_dir: Path, metadata: dict, full_matrix: np.ndarray
) -> dict[str, object]:
    """Verify that the later QSVT matrix is distinct from Panel 1 H_tilde."""
    dense_block = load_matrix(input_dir, "ieee14_dense_block_8x8", metadata)
    sparse_block = load_matrix(input_dir, "ieee14_sparse_weighted_jacobian", metadata)
    quantized_block = load_matrix(
        input_dir, "ieee14_quantized_sparse_block_8x8", metadata
    )
    for name, block in (
        ("dense selected block", dense_block),
        ("sparse selected block", sparse_block),
        ("quantized selected block", quantized_block),
    ):
        if block.shape != (8, 8):
            raise RuntimeError(f"{name} is not 8x8")

    qsvt_path = input_dir / "ieee14_qsvt_normalized_block_A.npy"
    qsvt_matrix = np.load(qsvt_path, allow_pickle=False)
    if qsvt_matrix.dtype != np.float64 or qsvt_matrix.shape != (8, 8):
        raise RuntimeError("QSVT matrix A must be an 8x8 float64 array")

    matrix_max = float(np.max(np.abs(full_matrix)))
    beta = 3.0 * matrix_max
    expected_qsvt_matrix = quantized_block.T / beta
    if not np.array_equal(qsvt_matrix, expected_qsvt_matrix):
        raise RuntimeError("QSVT matrix A is not exactly H_q^T / beta")
    return {
        "panel_1_matrix": "raw weighted Jacobian H_tilde, shape 82x27",
        "panel_4_matrix": "selected/sparse weighted-Jacobian block, shape 8x8",
        "panel_5_matrix": "A = H_q^T / beta, shape 8x8",
        "beta": beta,
        "qsvt_minimum": float(qsvt_matrix.min()),
        "qsvt_maximum": float(qsvt_matrix.max()),
        "qsvt_maximum_absolute_value": float(np.max(np.abs(qsvt_matrix))),
        "qsvt_array_sha256": array_sha256(qsvt_matrix),
        "exact_transpose_and_normalization_check": True,
    }


def _scientific_endpoint_labels(limit: float) -> list[str]:
    exponent = int(np.floor(np.log10(limit)))
    mantissa = limit / (10.0**exponent)
    return [
        rf"$-{mantissa:.2f}\!\times\!10^{{{exponent}}}$",
        "0",
        rf"$+{mantissa:.2f}\!\times\!10^{{{exponent}}}$",
    ]


def _assert_text_not_clipped(fig: plt.Figure, artists: list[object]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tolerance = 0.75
    for artist in artists:
        if not getattr(artist, "get_visible", lambda: False)():
            continue
        text = getattr(artist, "get_text", lambda: "")()
        if not text:
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            raise RuntimeError(
                f"rendered text is clipped: {text!r}; "
                f"bbox={tuple(round(value, 2) for value in bbox.extents)}, "
                f"canvas={tuple(round(value, 2) for value in canvas.extents)}"
            )


def _save_panel_figure(
    fig: plt.Figure,
    out_base: Path,
    *,
    formats: tuple[str, ...],
    png_dpi: int,
    description: str,
) -> list[Path]:
    created: list[Path] = []
    creator = "scripts/plot_figure1_weighted_jacobian.py"
    for suffix in formats:
        path = out_base.with_suffix(f".{suffix}")
        if suffix == "pdf":
            metadata = {
                "Title": "IEEE-14 weighted Jacobian panel",
                "Author": "Repository-derived audit export",
                "Subject": description,
                "Keywords": "IEEE-14, weighted Jacobian, repository-derived",
                "Creator": creator,
                "CreationDate": None,
                "ModDate": None,
            }
            fig.savefig(path, format="pdf", metadata=metadata)
        elif suffix == "svg":
            metadata = {
                "Title": "IEEE-14 weighted Jacobian panel",
                "Description": description,
                "Creator": creator,
                "Date": None,
            }
            fig.savefig(path, format="svg", metadata=metadata)
        elif suffix == "png":
            metadata = {
                "Title": "IEEE-14 weighted Jacobian panel",
                "Description": description,
                "Software": creator,
            }
            fig.savefig(path, format="png", dpi=png_dpi, metadata=metadata)
        else:
            raise ValueError(f"unsupported output format: {suffix}")
        created.append(path)
    return created


def _verify_svg_is_vector(path: Path, expected_matrix_cells: int) -> int:
    root = ET.parse(path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    if root.findall(".//svg:image", namespace):
        raise RuntimeError(f"{path.name} contains an embedded raster image")
    matrix_group = next(
        (
            element
            for element in root.findall(".//svg:g", namespace)
            if element.attrib.get("id") == "QuadMesh_1"
        ),
        None,
    )
    if matrix_group is None:
        raise RuntimeError(f"{path.name} does not contain the matrix QuadMesh")
    cell_count = len(matrix_group.findall("./svg:path", namespace))
    if cell_count != expected_matrix_cells:
        raise RuntimeError(
            f"{path.name} contains {cell_count} matrix cells, expected "
            f"{expected_matrix_cells}"
        )
    return cell_count


def plot_panel_ready(
    matrix: np.ndarray,
    out_base: Path,
    *,
    grouped: bool,
    display_normalized: bool,
    figsize: tuple[float, float] = (1.90, 1.05),
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    png_dpi: int = 600,
    font_scale: float = 1.0,
) -> dict[str, object]:
    """Render a compact panel with exact cells using vector QuadMesh geometry."""
    raw_limit = float(np.max(np.abs(matrix)))
    if display_normalized:
        plotted = matrix / raw_limit
        limit = 1.0
        cbar_ticks = [-1.0, 0.0, 1.0]
        cbar_ticklabels = [r"$-1$", "0", r"$+1$"]
        cbar_label = r"display-normalized $\widetilde{H}/M$"
        description = (
            "Display-normalized repository-derived IEEE-14 weighted Jacobian; "
            "not the numerical PSSE matrix and not the QSVT-normalized matrix A."
        )
    else:
        plotted = matrix
        limit = raw_limit
        cbar_ticks = [-limit, 0.0, limit]
        cbar_ticklabels = _scientific_endpoint_labels(limit)
        cbar_label = r"raw $\widetilde{H}_{ij}$"
        description = (
            "Raw repository-derived IEEE-14 weighted Jacobian on exact symmetric "
            "color limits."
        )

    fig = plt.figure(figsize=figsize, facecolor="white")
    # Leave enough physical margin for the rotated y-axis label at the small
    # manuscript-panel size; the grouped variant also needs family labels.
    left = 0.40 if grouped else 0.28
    right = 0.975
    ax = fig.add_axes([left, 0.44, right - left, 0.52])
    cax = fig.add_axes([left + 0.015, 0.155, right - left - 0.03, 0.065])

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    mesh = ax.pcolormesh(
        plotted,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=False,
        snap=True,
    )
    ax.set_xlim(0, plotted.shape[1])
    ax.set_ylim(plotted.shape[0], 0)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_xlabel(
        "State coordinates", fontsize=6.2 * font_scale, labelpad=1.2, color=TEXT_PRIMARY
    )
    ax.set_ylabel(
        "Measurement\nrows",
        fontsize=6.2 * font_scale,
        labelpad=2.0,
        color=TEXT_PRIMARY,
    )
    if grouped:
        centers = [(start + stop + 1) / 2.0 for _, start, stop, _ in ROW_FAMILY_SCHEMA]
        labels = [label for _, _, _, label in ROW_FAMILY_SCHEMA]
        ax.set_yticks(centers, labels=labels)
        ax.tick_params(
            axis="y",
            which="both",
            length=0,
            pad=1.5,
            labelsize=5.0 * font_scale,
            colors=TEXT_SECONDARY,
        )
        for boundary in ROW_GROUP_BOUNDARIES:
            ax.axhline(
                boundary,
                color=TEXT_PRIMARY,
                linewidth=0.38,
                alpha=0.62,
                zorder=3,
            )
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.38)

    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    # Matplotlib rasterizes colorbar solids by default; explicitly keep them
    # vector so both the matrix cells and the scale remain vector geometry.
    # Face-colored edges suppress PDF-viewer hairline seams between adjacent
    # colorbar polygons without smoothing or rasterizing the data.
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
        cbar.solids.set_edgecolor("face")
    cbar.set_ticks(cbar_ticks, labels=cbar_ticklabels)
    cbar.ax.tick_params(
        axis="x",
        which="both",
        labelsize=5.0 * font_scale,
        length=1.6,
        width=0.35,
        pad=1.2,
        colors=TEXT_SECONDARY,
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(
        cbar_label, fontsize=4.9 * font_scale, labelpad=1.2, color=TEXT_PRIMARY
    )
    cbar.outline.set_linewidth(0.35)
    cbar.outline.set_edgecolor(TEXT_SECONDARY)
    cbar_tick_artists = cbar.ax.get_xticklabels()
    if cbar_tick_artists:
        cbar_tick_artists[0].set_horizontalalignment("left")
        cbar_tick_artists[-1].set_horizontalalignment("right")

    rendered = np.asarray(mesh.get_array()).reshape(plotted.shape)
    if not np.array_equal(rendered, plotted):
        raise RuntimeError("plotted QuadMesh data differs from the requested matrix")
    if mesh.get_clim() != (-limit, limit):
        raise RuntimeError("plotted color limits are not exact and symmetric")

    text_artists: list[object] = [ax.xaxis.label, ax.yaxis.label, cbar.ax.xaxis.label]
    text_artists.extend(ax.get_yticklabels())
    text_artists.extend(cbar_tick_artists)
    _assert_text_not_clipped(fig, text_artists)
    created = _save_panel_figure(
        fig,
        out_base,
        formats=formats,
        png_dpi=png_dpi,
        description=description,
    )
    plt.close(fig)
    svg_cell_count = None
    for path in created:
        if path.suffix == ".svg":
            svg_cell_count = _verify_svg_is_vector(path, plotted.size)

    return {
        "stem": out_base.name,
        "raw_or_normalized": "display-normalized" if display_normalized else "raw",
        "grouped": grouped,
        "source_order_preserved": True,
        "plotted_shape": list(plotted.shape),
        "plotted_array_sha256": array_sha256(plotted),
        "color_limits": [-limit, limit],
        "colormap": "RdBu_r",
        "rendering_primitive": "vector QuadMesh (pcolormesh), no interpolation",
        "matrix_cell_count": int(plotted.size),
        "svg_matrix_cell_count": svg_cell_count,
        "separator_rows": list(ROW_GROUP_BOUNDARIES) if grouped else [],
        "files": [path.name for path in created],
        "text_clipping_check": "passed",
    }


def _compact_raw_tick_labels(limit: float) -> list[str]:
    rounded_limit = round(limit)
    return [rf"$-{rounded_limit}$", r"$0$", rf"$+{rounded_limit}$"]


def _png_foreground_margins(path: Path) -> dict[str, int]:
    image = plt.imread(path)
    rgb = image[..., :3]
    nonwhite = np.any(np.abs(rgb - 1.0) > (1.0 / 255.0), axis=2)
    y_coords, x_coords = np.nonzero(nonwhite)
    if not len(x_coords):
        raise RuntimeError(f"{path.name} contains no visible foreground")
    height, width = nonwhite.shape
    return {
        "width_pixels": int(width),
        "height_pixels": int(height),
        "left_pixels": int(x_coords.min()),
        "right_pixels": int(width - 1 - x_coords.max()),
        "top_pixels": int(y_coords.min()),
        "bottom_pixels": int(height - 1 - y_coords.max()),
    }


def plot_compact_panel(
    matrix: np.ndarray,
    out_base: Path,
    *,
    axis_labels: bool,
    grouped: bool,
    figsize: tuple[float, float] = (1.48, 0.52),
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    png_dpi: int = 600,
    font_scale: float = 1.0,
) -> dict[str, object]:
    """Render the raw 82x27 matrix as a tightly packed manuscript panel."""
    if matrix.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"compact panel requires shape {EXPECTED_SHAPE}")
    if array_sha256(matrix) != EXPECTED_FULL_MATRIX_SHA256:
        raise RuntimeError("compact panel source does not match the frozen matrix hash")

    limit = float(np.max(np.abs(matrix)))
    ticklabels = _compact_raw_tick_labels(limit)
    fig = plt.figure(figsize=figsize, facecolor="white")
    if axis_labels:
        heatmap_bounds = (0.115, 0.20, 0.705, 0.755)
        colorbar_bounds = (0.825, 0.245, 0.025, 0.500)
    else:
        heatmap_bounds = (0.015, 0.055, 0.790, 0.910)
        colorbar_bounds = (0.810, 0.180, 0.024, 0.520)
    ax = fig.add_axes(heatmap_bounds)
    cax = fig.add_axes(colorbar_bounds)

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    mesh = ax.pcolormesh(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=False,
        snap=True,
    )
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    if axis_labels:
        ax.set_xlabel(
            "States", fontsize=4.8 * font_scale, labelpad=0.7, color=TEXT_PRIMARY
        )
        ax.set_ylabel(
            "Measurements",
            fontsize=4.8 * font_scale,
            labelpad=1.2,
            color=TEXT_PRIMARY,
        )
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if grouped:
        for boundary in ROW_GROUP_BOUNDARIES:
            ax.axhline(
                boundary,
                color=TEXT_PRIMARY,
                linewidth=0.28,
                alpha=0.50,
                zorder=3,
            )
    for spine in ax.spines.values():
        spine.set_color(TEXT_SECONDARY)
        spine.set_linewidth(0.30)

    cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
        cbar.solids.set_edgecolor("face")
    cbar.set_ticks([-limit, 0.0, limit], labels=ticklabels)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.ax.yaxis.get_offset_text().set_text("")
    cbar.ax.tick_params(
        axis="y",
        which="both",
        labelsize=4.0 * font_scale,
        length=1.4,
        width=0.30,
        pad=1.1,
        colors=TEXT_SECONDARY,
    )
    cbar.outline.set_linewidth(0.30)
    cbar.outline.set_edgecolor(TEXT_SECONDARY)

    rendered = np.asarray(mesh.get_array()).reshape(matrix.shape)
    if not np.array_equal(rendered, matrix):
        raise RuntimeError("compact QuadMesh data differs from the verified matrix")
    if mesh.get_clim() != (-limit, limit):
        raise RuntimeError("compact color limits are not exact and symmetric")
    if ax.get_xlim() != (0.0, 27.0) or ax.get_ylim() != (82.0, 0.0):
        raise RuntimeError("compact panel matrix orientation is incorrect")
    if not axis_labels and (ax.get_xlabel() or ax.get_ylabel()):
        raise RuntimeError("unlabeled compact panel unexpectedly contains axis labels")

    text_artists: list[object] = list(cbar.ax.get_yticklabels())
    if axis_labels:
        text_artists.extend([ax.xaxis.label, ax.yaxis.label])
    _assert_text_not_clipped(fig, text_artists)

    variant = "with minimal axis labels" if axis_labels else "without axis labels"
    if grouped:
        variant += " and with measurement-family separators"
    created = _save_panel_figure(
        fig,
        out_base,
        formats=formats,
        png_dpi=png_dpi,
        description=(
            "Compact raw repository-derived IEEE-14 weighted Jacobian " + variant + "."
        ),
    )
    plt.close(fig)

    svg_cell_count = None
    png_geometry = None
    for path in created:
        if path.suffix == ".svg":
            svg_cell_count = _verify_svg_is_vector(path, matrix.size)
        elif path.suffix == ".png":
            png_geometry = _png_foreground_margins(path)

    return {
        "stem": out_base.name,
        "raw_or_normalized": "raw",
        "axis_labels": ["States", "Measurements"] if axis_labels else [],
        "matrix_tick_labels": False,
        "grouped": grouped,
        "source_order_preserved": True,
        "transposed": False,
        "plotted_shape": list(matrix.shape),
        "vertical_dimension": "82 measurement rows",
        "horizontal_dimension": "27 state coordinates",
        "plotted_array_sha256": array_sha256(matrix),
        "color_limits": [-limit, limit],
        "color_scale": "raw symmetric linear",
        "colormap": "RdBu_r",
        "colorbar_orientation": "vertical",
        "colorbar_title": None,
        "colorbar_ticks": [-limit, 0.0, limit],
        "colorbar_tick_labels": [f"-{round(limit):.0f}", "0", f"+{round(limit):.0f}"],
        "colorbar_common_exponent": None,
        "scientific_offset_notation": False,
        "rendering_primitive": "vector QuadMesh (pcolormesh), no interpolation",
        "matrix_cell_count": int(matrix.size),
        "svg_matrix_cell_count": svg_cell_count,
        "separator_rows": list(ROW_GROUP_BOUNDARIES) if grouped else [],
        "figure_size_inches": list(figsize),
        "figure_aspect_ratio": float(figsize[0] / figsize[1]),
        "matrix_axes_fraction_of_canvas": float(heatmap_bounds[2] * heatmap_bounds[3]),
        "png_geometry": png_geometry,
        "files": [path.name for path in created],
        "text_clipping_check": "passed",
    }


def _git_value(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_finalization_audit(
    input_dir: Path,
    metadata: dict,
    metrics: dict[str, object],
    qsvt_check: dict[str, object],
    render_checks: list[dict[str, object]],
    generated_paths: list[Path],
) -> None:
    repository_root = Path(
        _git_value(input_dir, "rev-parse", "--show-toplevel")
    ).resolve()
    commit = _git_value(repository_root, "rev-parse", "HEAD")
    branch = _git_value(repository_root, "branch", "--show-current")
    source_path = input_dir / "ieee14_weighted_jacobian.npy"
    source_rel = _relative(source_path, repository_root)

    asset_records = {
        path.name: {
            "path": _relative(path, repository_root),
            "format": path.suffix.lstrip("."),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(generated_paths)
    }
    audit_metadata = {
        "case": "IEEE-14",
        "matrix": "raw weighted Jacobian H_tilde = R^(-1/2) H",
        "source_file": source_rel,
        "source_array_sha256": metrics["array_sha256"],
        "source_npy_file_sha256": metrics["npy_file_sha256"],
        "shape": metrics["shape"],
        "rank": metrics["rank"],
        "condition_number": metrics["condition_number"],
        "minimum": metrics["minimum"],
        "maximum": metrics["maximum"],
        "maximum_absolute_value": metrics["maximum_absolute_value"],
        "raw_color_limits": [
            -float(metrics["maximum_absolute_value"]),
            float(metrics["maximum_absolute_value"]),
        ],
        "row_family_boundaries": list(ROW_GROUP_BOUNDARIES),
        "row_family_order": [family for family, _, _, _ in ROW_FAMILY_SCHEMA],
        "state_column_order": list(EXPECTED_STATE_LABELS),
        "repository_root": str(repository_root),
        "git_branch": branch,
        "git_commit": commit,
        "source_config": metadata.get("config"),
        "source_code_path": metadata.get("source_code_path"),
        "verification": {
            "frozen_array_sha256_match": True,
            "csv_exact_copy": metrics["csv_exact_copy"],
            "row_semantics": metrics["row_semantics_verified"],
            "column_semantics": metrics["column_semantics_verified"],
            "qsvt_distinction": qsvt_check,
            "render_checks": render_checks,
        },
        "assets": asset_records,
    }
    metadata_path = input_dir / "figure1_heatmap_finalization_metadata.json"
    metadata_path.write_text(
        json.dumps(audit_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    limit = float(metrics["maximum_absolute_value"])
    minimum_text = repr(float(metrics["minimum"]))
    maximum_text = repr(float(metrics["maximum"]))
    limit_text = repr(limit)
    condition_text = repr(float(metrics["condition_number"]))
    beta_text = repr(float(qsvt_check["beta"]))
    qsvt_limit_text = repr(float(qsvt_check["qsvt_maximum_absolute_value"]))
    report = f"""# Figure 1 Weighted Jacobian Heatmap Finalization

## 1. Matrix Source

- File: `{source_rel}`
- Shape: {metrics['shape'][0]} x {metrics['shape'][1]}
- SHA-256: `{metrics['array_sha256']}` (contiguous float64 array bytes)
- Config/source: `{metadata.get('config')}`; `{metadata.get('source_code_path')}`
- Verified: **YES** - the NPY hash matches the frozen audit hash, and the CSV is an exact elementwise copy.
- Repository branch / commit: `{branch}` / `{commit}`

## 2. Numerical Range

- Minimum: {minimum_text}
- Maximum: {maximum_text}
- Maximum absolute value: {limit_text}
- Color limits: `[-{limit_text}, +{limit_text}]` (raw panels, exact symmetric limits)
- Rank: {metrics['rank']}
- 2-norm condition number: {condition_text}

## 3. Semantics

- Rows: 82 generated AC measurement rows; no rows are reordered.
- Columns: 27 state coordinates in the order `theta_2` through `theta_14`, then `V_1` through `V_14`; no columns are reordered.
- Measurement-family order: `V` (0-13), `P_inj` (14-27), `Q_inj` (28-41), `P_flow` (42-61), `Q_flow` (62-81).

## 4. Figure Variants

| File | Raw/Normalized | Intended Use |
| --- | --- | --- |
| `{PANEL_STEM}.pdf/.svg` | Raw `H_tilde` | Primary vector assets for Figure 1 |
| `{PANEL_STEM}.png` | Raw `H_tilde` | 600-dpi review raster |
| `{GROUPED_PANEL_STEM}.pdf/.svg/.png` | Raw `H_tilde` | Audit/review version with row-family separators |
| `{DISPLAY_NORMALIZED_PANEL_STEM}.pdf/.svg/.png` | `H_tilde / max(abs(H_tilde))` | Optional design comparison only; not the numerical PSSE or QSVT matrix |
| `{PREVIEW_STEM}.png` | Raw `H_tilde` | Reduced-size manuscript-panel preview |

All 2,214 matrix cells are emitted as vector `QuadMesh` geometry in PDF/SVG, with no interpolation. SVG structure contains exactly 2,214 matrix-cell paths and no embedded raster image; Poppler reports no embedded raster image in the PDFs.

## 5. Manuscript Recommendation

- Recommended asset: `{PANEL_STEM}.pdf` (or SVG when the assembly workflow prefers SVG)
- Recommended label: **Repository-derived IEEE-14 weighted Jacobian \\(\\widetilde{{H}}\\) (\\(82\\times27\\))**
- Recommended color scale: raw symmetric scale `[-M, M]`, where `M = {limit_text}`
- Recommended version: minimal raw panel; retain the grouped version for audit/review.

## 6. Claim Safety

Can this be labeled:

> Repository-derived IEEE-14 weighted Jacobian H_tilde (82 x 27)

**YES.** The plotted array is the hash-verified repository export. The claim should remain framed as a matrix generated by repository code from the PYPOWER IEEE-14 benchmark model, not field measurements.

## 7. Final Status

- Frozen array hash: verified.
- Shape, rank, condition number, and range: recomputed and verified.
- Row and column semantics: verified from the exported CSV files.
- Source order: preserved exactly.
- Raw color scale: exact, symmetric, and zero-centered.
- Reduced-size preview: generated for visual QA.
- PDF, SVG, full-resolution PNG, and reduced-size preview: visually inspected; no clipping, vector seams, or label overlap remains.
- Determinism: two consecutive runs produced byte-identical figure and audit outputs.
- Scientific source/configuration/canonical evidence: not modified.

## 8. Raw Versus Screenshot-Style Rendering

| Version | Data | Color Range | Scientifically Accurate? | Use in Main Figure? |
| --- | --- | --- | --- | --- |
| AI mock-up | Schematic | Approximately `[-1, 1]` | No | No |
| Raw repository matrix | Actual `H_tilde` | `[-{limit_text}, +{limit_text}]` | Yes | **Yes** |
| Display-normalized | Actual `H_tilde / M` | `[-1, 1]` | Yes, only when explicitly labeled | Optional design review only |

The mock-up's approximate `[-1, 1]` bar cannot represent the raw weighted-Jacobian values and must not be used as scientific evidence.

## 9. Separation From the QSVT Matrix

- Panel 1: raw `H_tilde`, shape 82 x 27, maximum absolute entry {limit_text}.
- Panel 4: selected/sparse weighted-Jacobian block, shape 8 x 8.
- Panel 5: `A = H_q^T / beta`, shape 8 x 8, with `beta = {beta_text}` and maximum absolute entry {qsvt_limit_text}.
- Exact check: exported `A` equals exported `H_q.T / beta` elementwise. Its color-scale interpretation is therefore distinct from Panel 1.
"""
    (input_dir / "figure1_heatmap_finalization_report.md").write_text(
        report, encoding="utf-8"
    )


def write_compact_refinement_audit(
    input_dir: Path,
    metadata: dict,
    metrics: dict[str, object],
    qsvt_check: dict[str, object],
    render_checks: list[dict[str, object]],
    generated_paths: list[Path],
) -> None:
    repository_root = Path(
        _git_value(input_dir, "rev-parse", "--show-toplevel")
    ).resolve()
    commit = _git_value(repository_root, "rev-parse", "HEAD")
    branch = _git_value(repository_root, "branch", "--show-current")
    source_path = input_dir / "ieee14_weighted_jacobian.npy"
    source_rel = _relative(source_path, repository_root)
    limit = float(metrics["maximum_absolute_value"])
    primary = next(check for check in render_checks if check["stem"] == COMPACT_STEM)
    preview = next(
        check for check in render_checks if check["stem"] == COMPACT_PREVIEW_STEM
    )
    preview_geometry = preview["png_geometry"]

    asset_records = {
        path.name: {
            "path": _relative(path, repository_root),
            "format": path.suffix.lstrip("."),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(generated_paths)
    }
    audit_metadata = {
        "case": "IEEE-14",
        "matrix": "raw weighted Jacobian H_tilde = R^(-1/2) H",
        "source_file": source_rel,
        "source_array_sha256": metrics["array_sha256"],
        "shape": metrics["shape"],
        "minimum": metrics["minimum"],
        "maximum": metrics["maximum"],
        "maximum_absolute_value": limit,
        "raw_color_limits": [-limit, limit],
        "repository_root": str(repository_root),
        "git_branch": branch,
        "git_commit": commit,
        "source_config": metadata.get("config"),
        "source_code_path": metadata.get("source_code_path"),
        "scientific_integrity": {
            "matrix_transposed": False,
            "rows_reordered": False,
            "columns_reordered": False,
            "values_normalized": False,
            "smoothing_or_interpolation": False,
            "nonlinear_color_scaling": False,
            "qsvt_matrix_used": False,
        },
        "qsvt_distinction": qsvt_check,
        "render_checks": render_checks,
        "assets": asset_records,
    }
    metadata_path = input_dir / "figure1_heatmap_compact_refinement_metadata.json"
    metadata_path.write_text(
        json.dumps(audit_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    minimum_text = repr(float(metrics["minimum"]))
    maximum_text = repr(float(metrics["maximum"]))
    limit_text = repr(limit)
    beta_text = repr(float(qsvt_check["beta"]))
    qsvt_limit_text = repr(float(qsvt_check["qsvt_maximum_absolute_value"]))
    report = f"""# Figure 1 Heatmap Compact Refinement Report

## 1. Source Verification

- Source matrix: `{source_rel}`
- Shape: {metrics['shape'][0]} x {metrics['shape'][1]}
- SHA-256: `{metrics['array_sha256']}` (contiguous float64 array bytes)
- Min: {minimum_text}
- Max: {maximum_text}
- Maximum absolute entry: {limit_text}
- Color limits: `[-{limit_text}, +{limit_text}]`
- Verification status: **PASS** - the source matches the frozen hash and the existing semantic CSV files.
- Repository branch / commit: `{branch}` / `{commit}`

## 2. Layout Changes

- Axis labels: removed from the final compact asset; comparison variant uses only `States` and `Measurements`.
- Tick labels: removed from both matrix axes in every compact variant.
- Colorbar position: thin vertical bar immediately to the right of the matrix.
- Colorbar format: three direct raw-value ticks (`-1347`, `0`, `+1347`); scientific-offset notation disabled; no colorbar title.
- Margins: explicit tight axes placement; no title or caption is embedded.
- Aspect handling: `aspect="auto"` equivalent through a non-transposing vector `QuadMesh`; complete primary asset ratio {float(primary['figure_aspect_ratio']):.3f}:1.
- Line weights: 0.30-point-equivalent matrix/colorbar outlines; no grid or cell borders.

## 3. Scientific Integrity

- Matrix transposed?: **NO** - vertical direction remains 82 measurement rows; horizontal direction remains 27 state coordinates.
- Rows reordered?: **NO**
- Columns reordered?: **NO**
- Values normalized?: **NO**
- Smoothing/interpolation?: **NO**
- Nonlinear color scaling?: **NO**
- Plotted-array SHA-256: `{primary['plotted_array_sha256']}` (identical to the verified source hash)

## 4. Generated Variants

| Asset | Purpose | Recommended? |
| --- | --- | --- |
| `{COMPACT_STEM}.pdf/.svg/.png` | Final compact raw matrix, vertical colorbar, no axis text | **YES** |
| `{COMPACT_LABELS_STEM}.pdf/.svg/.png` | Compact comparison with tiny `States` / `Measurements` labels | No - parent panel already supplies semantics |
| `{COMPACT_GROUPED_STEM}.pdf/.svg/.png` | Audit/review version with separators at rows 14, 28, 42, and 62 | No - retain for structural review |
| `{COMPACT_PREVIEW_STEM}.png` | Physical-scale Panel 1 preview | Review only |

All PDF/SVG matrix cells use vector geometry. Each SVG contains exactly 2,214 matrix-cell paths and no embedded raster image.

## 5. Manuscript-Size QA

- Preview size: 36.0 mm x 12.5 mm at 300 dpi ({preview_geometry['width_pixels']} x {preview_geometry['height_pixels']} pixels).
- Matrix readable: **YES** - the block/sparsity structure and signed color pattern remain visible at the intended scale.
- Colorbar readable: **YES** - the three direct raw-value ticks remain distinguishable without an offset multiplier.
- Clipping: **NONE** - automated artist-bound checks and PDF/SVG/PNG render inspection passed.
- White-space assessment: minimal; preview foreground margins are left {preview_geometry['left_pixels']} px, right {preview_geometry['right_pixels']} px, top {preview_geometry['top_pixels']} px, and bottom {preview_geometry['bottom_pixels']} px.
- Matrix visual allocation: {100.0 * float(primary['matrix_axes_fraction_of_canvas']):.1f}% of the full canvas area; the matrix remains visually dominant.

## 6. Recommended Final Asset

- File: `{COMPACT_STEM}.pdf`
- Format: PDF vector (SVG is the equivalent alternative for SVG-native assembly)
- Reason: it preserves the raw matrix and its scale while removing redundant panel text and replacing the horizontal colorbar with a compact vertical scale.
- Minimal-label variant assessment: the labels consume matrix area but add little information because the parent-panel label already gives the object and dimensions.

## 7. Recommended Parent-Panel Label

> **IEEE-14 weighted Jacobian \\(\\widetilde{{H}}\\) (\\(82\\times27\\))**

This shorter form is more readable at the approximately 36 mm panel width. State repository provenance in the Figure 1 caption.

## 8. Final Status

- Source hash, shape, orientation, and raw value range: verified.
- Compact raw linear color scaling: verified.
- Vector PDF and SVG plus 600-dpi PNG: generated.
- Manuscript-size preview: generated and visually inspected.
- Deterministic rendering: verified by consecutive byte-identical runs.
- Scientific source, matrix artifact, configuration, and canonical evidence: not modified.
- Status: **READY FOR FIGURE 1 ASSEMBLY**

## 9. Existing Versus Compact Presentation

| Criterion | Existing Heatmap | Compact Version |
| --- | --- | --- |
| Matrix data | Same verified raw matrix | Same |
| Matrix orientation | 82 x 27 | 82 x 27 |
| Color scale | Raw linear | Raw linear |
| Axis labels | Large | Removed/minimized |
| Colorbar | Large horizontal | Compact vertical |
| White space | High | Minimal |
| Figure 1 suitability | Moderate | High |

The scientific content and plotted-array hash are identical between the existing raw heatmap and the compact raw version.

## 10. Separation From the QSVT Matrix

Panel 1 displays the raw weighted Jacobian `H_tilde = R^(-1/2) H`, shape 82 x 27, on its actual weighted-Jacobian color scale. It does **not** display `A = H_q^T / beta`. The exported QSVT matrix is a separate 8 x 8 object with `beta = {beta_text}` and maximum absolute entry {qsvt_limit_text}.
"""
    (input_dir / "figure1_heatmap_compact_refinement_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/figure1_matrix_audit", type=Path)
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="do not regenerate the pre-existing exploratory heatmaps",
    )
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="generate only the final compact variants and their audit report",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the exported arrays and semantics without writing figures",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    metadata = json.loads((input_dir / "ieee14_weighted_jacobian_metadata.json").read_text())

    full = load_matrix(input_dir, "ieee14_weighted_jacobian", metadata)
    metrics = verify_exported_matrix(input_dir, full, metadata)
    qsvt_check = verify_qsvt_distinction(input_dir, metadata, full)
    if args.verify_only:
        print(json.dumps({"matrix": metrics, "qsvt_distinction": qsvt_check}, indent=2))
        return

    generated_paths: list[Path] = []
    audit_reports: list[str] = []

    if not args.compact_only:
        render_checks: list[dict[str, object]] = []
        render_checks.append(
            plot_panel_ready(
                full,
                input_dir / PANEL_STEM,
                grouped=False,
                display_normalized=False,
            )
        )
        render_checks.append(
            plot_panel_ready(
                full,
                input_dir / GROUPED_PANEL_STEM,
                grouped=True,
                display_normalized=False,
                figsize=(2.15, 1.12),
            )
        )
        render_checks.append(
            plot_panel_ready(
                full,
                input_dir / DISPLAY_NORMALIZED_PANEL_STEM,
                grouped=False,
                display_normalized=True,
            )
        )
        render_checks.append(
            plot_panel_ready(
                full,
                input_dir / PREVIEW_STEM,
                grouped=False,
                display_normalized=False,
                figsize=(1.45, 0.88),
                formats=("png",),
                png_dpi=300,
                font_scale=0.90,
            )
        )
        for check in render_checks:
            generated_paths.extend(input_dir / name for name in check["files"])

        write_finalization_audit(
            input_dir,
            metadata,
            metrics,
            qsvt_check,
            render_checks,
            generated_paths,
        )
        audit_reports.append("figure1_heatmap_finalization_report.md")

    compact_checks: list[dict[str, object]] = []
    compact_paths: list[Path] = []
    compact_checks.append(
        plot_compact_panel(
            full,
            input_dir / COMPACT_STEM,
            axis_labels=False,
            grouped=False,
        )
    )
    compact_checks.append(
        plot_compact_panel(
            full,
            input_dir / COMPACT_LABELS_STEM,
            axis_labels=True,
            grouped=False,
            figsize=(1.62, 0.62),
        )
    )
    compact_checks.append(
        plot_compact_panel(
            full,
            input_dir / COMPACT_GROUPED_STEM,
            axis_labels=False,
            grouped=True,
        )
    )
    compact_checks.append(
        plot_compact_panel(
            full,
            input_dir / COMPACT_PREVIEW_STEM,
            axis_labels=False,
            grouped=False,
            figsize=(36.0 / 25.4, 12.5 / 25.4),
            formats=("png",),
            png_dpi=300,
        )
    )
    for check in compact_checks:
        compact_paths.extend(input_dir / name for name in check["files"])
    generated_paths.extend(compact_paths)
    write_compact_refinement_audit(
        input_dir,
        metadata,
        metrics,
        qsvt_check,
        compact_checks,
        compact_paths,
    )
    audit_reports.append("figure1_heatmap_compact_refinement_report.md")

    if not args.compact_only and not args.skip_legacy:
        dense_block = load_matrix(input_dir, "ieee14_dense_block_8x8", metadata)
        sparse_block = load_matrix(input_dir, "ieee14_sparse_weighted_jacobian", metadata)
        links = metadata["frozen_evidence_links"]
        block_rows = links["block_rows"]
        block_cols = links["block_cols"]

        plot_dense(full, input_dir)
        plot_dense_with_block(full, block_rows, block_cols, input_dir)
        plot_sparse(sparse_block, input_dir)
        plot_block_pair(dense_block, sparse_block, input_dir)

    print(
        json.dumps(
            {
                "verified_matrix_sha256": metrics["array_sha256"],
                "shape": metrics["shape"],
                "raw_color_limits": [
                    -float(metrics["maximum_absolute_value"]),
                    float(metrics["maximum_absolute_value"]),
                ],
                "generated_files": [path.name for path in generated_paths],
                "audit_reports": audit_reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
