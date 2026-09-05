#!/usr/bin/env python3
"""Generate matched Figure-1 heatmaps for the canonical IEEE-14 8x8 matrices.

The source arrays are repository artifacts from the frozen canonical workload.
They are verified against the candidate hashes and the stored reconstruction
before plotting.  Arrays are passed to Matplotlib without transpose, row/column
reordering, smoothing, or interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "qsvt_figure1_matrix_panel_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "outputs" / "figure1_matrix_audit"
CANDIDATE_FREEZE = REPOSITORY_ROOT / "outputs" / "joint_four_condition" / "candidate_freeze.json"
RECONSTRUCTION = (
    REPOSITORY_ROOT
    / "outputs"
    / "phase10_sparse_wrapper_8x8_complete"
    / "sparse_wrapper_8x8_block_reconstruction.json"
)

DENSE_SOURCE = SOURCE_DIR / "ieee14_dense_block_8x8.npy"
SPARSE_SOURCE = SOURCE_DIR / "ieee14_sparse_weighted_jacobian.npy"
DENSE_CSV_SOURCE = SOURCE_DIR / "ieee14_dense_block_8x8.csv"
SPARSE_CSV_SOURCE = SOURCE_DIR / "ieee14_sparse_weighted_jacobian.csv"

FIGSIZE_IN = (2.25, 2.35)
PNG_DPI = 600
PREVIEW_DPI = 300
MATRIX_RECT = (0.060, 0.055, 0.780, 0.7468085106382979)
COLORBAR_RECT = (0.852, MATRIX_RECT[1], 0.024, MATRIX_RECT[3])
TITLE_X = MATRIX_RECT[0] + MATRIX_RECT[2] / 2.0

NEGATIVE_COLOR = "#2166AC"
ZERO_COLOR = "#FFFFFF"
POSITIVE_COLOR = "#B2182B"
GRID_COLOR = "#D7D7D7"
FRAME_COLOR = "#3F3F3F"
TEXT_COLOR = "#111111"

DIVERGING = LinearSegmentedColormap.from_list(
    "figure1_blue_white_red",
    [NEGATIVE_COLOR, ZERO_COLOR, POSITIVE_COLOR],
    N=256,
)

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.linewidth": 0.60,
        "pdf.compression": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "figure1-canonical-matrix-panels",
        "path.simplify": False,
    }
)


def array_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_pair() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dense = np.load(DENSE_SOURCE, allow_pickle=False)
    sparse = np.load(SPARSE_SOURCE, allow_pickle=False)
    if dense.dtype != np.float64 or sparse.dtype != np.float64:
        raise RuntimeError("canonical matrix artifacts must be float64")
    if dense.shape != (8, 8) or sparse.shape != (8, 8):
        raise RuntimeError("canonical matrix artifacts must both be 8x8")
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise RuntimeError("canonical matrix artifacts contain non-finite values")

    dense_csv = np.loadtxt(DENSE_CSV_SOURCE, delimiter=",")
    sparse_csv = np.loadtxt(SPARSE_CSV_SOURCE, delimiter=",")
    if not np.array_equal(dense_csv, dense):
        raise RuntimeError("dense CSV does not exactly match the dense NPY artifact")
    if not np.array_equal(sparse_csv, sparse):
        raise RuntimeError("sparse CSV does not exactly match the sparse NPY artifact")

    freeze = json.loads(CANDIDATE_FREEZE.read_text(encoding="utf-8"))
    expected_dense_hash = str(freeze["component_hashes"]["matrix_original"])
    expected_sparse_hash = str(freeze["component_hashes"]["matrix_supported_exact"])
    dense_hash = array_sha256(dense)
    sparse_hash = array_sha256(sparse)
    if dense_hash != expected_dense_hash:
        raise RuntimeError(
            f"dense block hash mismatch: expected {expected_dense_hash}, found {dense_hash}"
        )
    if sparse_hash != expected_sparse_hash:
        raise RuntimeError(
            f"sparse block hash mismatch: expected {expected_sparse_hash}, found {sparse_hash}"
        )

    reconstruction = json.loads(RECONSTRUCTION.read_text(encoding="utf-8"))
    reconstructed_dense = np.asarray(reconstruction["original_block"], dtype=np.float64)
    reconstructed_sparse = np.asarray(reconstruction["sparsified_block"], dtype=np.float64)
    if not np.array_equal(reconstructed_dense, dense):
        raise RuntimeError("dense source differs from the frozen reconstruction")
    if not np.array_equal(reconstructed_sparse, sparse):
        raise RuntimeError("sparse source differs from the frozen reconstruction")

    support_mask = np.zeros((8, 8), dtype=bool)
    for row, column in freeze["support_coordinates"]:
        support_mask[int(row), int(column)] = True
    actual_support = sparse != 0.0
    if not np.array_equal(actual_support, support_mask):
        raise RuntimeError("sparse nonzero pattern differs from the frozen support coordinates")
    if int(actual_support.sum()) != 16:
        raise RuntimeError("canonical sparse support must contain exactly 16 entries")
    if not np.array_equal(sparse[support_mask], dense[support_mask]):
        raise RuntimeError("retained sparse values do not equal the corresponding dense entries")
    if not np.all(sparse[~support_mask] == 0.0):
        raise RuntimeError("off-support sparse entries are not exactly zero")

    scale = float(np.max(np.abs(dense)))
    if scale <= 0.0 or not np.isfinite(scale):
        raise RuntimeError("invalid shared display scale")
    if not np.isclose(float(np.max(np.abs(sparse))), scale, rtol=0.0, atol=0.0):
        raise RuntimeError("dense and sparse blocks do not share the canonical maximum magnitude")

    verification: dict[str, object] = {
        "workload_id": freeze["workload_id"],
        "source_representation": "raw weighted-Jacobian block values",
        "sparse_representation": "exact retained values before quantization",
        "dense_source": str(DENSE_SOURCE.relative_to(REPOSITORY_ROOT)),
        "sparse_source": str(SPARSE_SOURCE.relative_to(REPOSITORY_ROOT)),
        "reconstruction_source": str(RECONSTRUCTION.relative_to(REPOSITORY_ROOT)),
        "candidate_freeze_source": str(CANDIDATE_FREEZE.relative_to(REPOSITORY_ROOT)),
        "dense_array_sha256": dense_hash,
        "sparse_array_sha256": sparse_hash,
        "dense_shape": list(dense.shape),
        "sparse_shape": list(sparse.shape),
        "dense_nonzero_count": int(np.count_nonzero(dense)),
        "sparse_nonzero_count": int(np.count_nonzero(sparse)),
        "support_coordinates": freeze["support_coordinates"],
        "shared_display_scale_max_abs_raw": scale,
        "colorbar_definition": "raw value divided by shared max-absolute dense-block value",
        "row_order_preserved": True,
        "column_order_preserved": True,
        "transposed": False,
        "interpolation": "none (vector pcolormesh cells)",
    }
    return dense, sparse, verification


def _artist_is_inside_figure(fig: plt.Figure, artist: object) -> bool:
    renderer = fig.canvas.get_renderer()
    figure_bounds = fig.bbox
    artist_bounds = artist.get_window_extent(renderer=renderer)
    tolerance = 0.5
    return bool(
        artist_bounds.x0 >= figure_bounds.x0 - tolerance
        and artist_bounds.y0 >= figure_bounds.y0 - tolerance
        and artist_bounds.x1 <= figure_bounds.x1 + tolerance
        and artist_bounds.y1 <= figure_bounds.y1 + tolerance
    )


def _build_panel(
    matrix: np.ndarray,
    *,
    title: str,
    mathematical_label: str,
    display_scale: float,
) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.figure(figsize=FIGSIZE_IN, facecolor="white")
    ax = fig.add_axes(MATRIX_RECT)
    cax = fig.add_axes(COLORBAR_RECT)

    coordinates = np.arange(9, dtype=float)
    norm = Normalize(vmin=-display_scale, vmax=display_scale, clip=True)
    mesh = ax.pcolormesh(
        coordinates,
        coordinates,
        matrix,
        cmap=DIVERGING,
        norm=norm,
        shading="flat",
        edgecolors=GRID_COLOR,
        linewidth=0.38,
        antialiased=True,
        rasterized=False,
        snap=True,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 8.0)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", length=0.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.62)

    # Draw the colorbar explicitly as vector cells.  Matplotlib's standard PDF
    # colorbar backend may rasterize the gradient even when the matrix itself is
    # vector geometry.
    # The custom map is piecewise linear blue -> white -> red, so these three
    # Gouraud nodes reproduce it exactly without inflating the SVG.
    color_nodes = np.asarray([-display_scale, 0.0, display_scale], dtype=float)
    color_x, color_y = np.meshgrid(np.asarray([0.0, 1.0]), color_nodes)
    color_values = np.repeat(color_nodes[:, np.newaxis], 2, axis=1)
    cax.pcolormesh(
        color_x,
        color_y,
        color_values,
        cmap=DIVERGING,
        norm=norm,
        shading="gouraud",
        edgecolors="none",
        linewidth=0.0,
        antialiased=True,
        rasterized=False,
    )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(-display_scale, display_scale)
    cax.set_xticks([])
    cax.set_yticks([-display_scale, 0.0, display_scale])
    cax.set_yticklabels([r"$-1$", r"$0$", r"$+1$"])
    cax.yaxis.tick_right()
    cax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=2.4,
        width=0.55,
        labelsize=7.2,
        colors=TEXT_COLOR,
        pad=2.0,
    )
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.60)
        spine.set_edgecolor(FRAME_COLOR)

    title_artist = fig.text(
        TITLE_X,
        0.944,
        title,
        ha="center",
        va="center",
        fontsize=9.7,
        color=TEXT_COLOR,
    )
    math_artist = fig.text(
        TITLE_X,
        0.855,
        mathematical_label,
        ha="center",
        va="center",
        fontsize=9.5,
        color=TEXT_COLOR,
    )

    fig.canvas.draw()
    matrix_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    if abs(matrix_bbox.width - matrix_bbox.height) > 1.0:
        raise RuntimeError("matrix axes are not square")
    if not _artist_is_inside_figure(fig, title_artist):
        raise RuntimeError("panel title is clipped")
    if not _artist_is_inside_figure(fig, math_artist):
        raise RuntimeError("panel mathematical label is clipped")
    if any(not _artist_is_inside_figure(fig, label) for label in cax.get_yticklabels()):
        raise RuntimeError("colorbar tick label is clipped")

    plotted = np.asarray(mesh.get_array(), dtype=np.float64)
    if plotted.shape != (8, 8) or not np.array_equal(plotted, matrix):
        raise RuntimeError("Matplotlib did not receive the source matrix in its original order")
    if ax.get_title() != "" or ax.get_xlabel() != "" or ax.get_ylabel() != "":
        raise RuntimeError("unexpected axes title or axis label")
    if len(ax.get_xticklabels()) != 0 or len(ax.get_yticklabels()) != 0:
        raise RuntimeError("axis tick labels are present")

    panel_check: dict[str, object] = {
        "title": title,
        "mathematical_label": mathematical_label,
        "plotted_array_sha256": array_sha256(plotted),
        "plotted_shape": list(plotted.shape),
        "matrix_axes_square": True,
        "axis_tick_labels_present": False,
        "cell_boundaries": True,
        "colorbar_tick_labels_bottom_to_top": ["-1", "0", "+1"],
        "colorbar_tick_labels_top_to_bottom": ["+1", "0", "-1"],
        "colorbar_label": None,
        "colorbar_width_in": COLORBAR_RECT[2] * FIGSIZE_IN[0],
        "matrix_colorbar_gap_in": (
            COLORBAR_RECT[0] - MATRIX_RECT[0] - MATRIX_RECT[2]
        )
        * FIGSIZE_IN[0],
        "display_vmin_raw": -display_scale,
        "display_vmax_raw": display_scale,
        "title_or_label_clipping": False,
    }
    return fig, panel_check


def _save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], format="pdf", transparent=False)
    fig.savefig(paths["svg"], format="svg", transparent=False)
    fig.savefig(paths["png"], format="png", dpi=PNG_DPI, transparent=False)
    plt.close(fig)
    return {kind: str(path) for kind, path in paths.items()}


def _write_side_by_side_preview(
    left_png: Path,
    right_png: Path,
    output_path: Path,
) -> dict[str, object]:
    target_width = round(FIGSIZE_IN[0] * PREVIEW_DPI)
    target_height = round(FIGSIZE_IN[1] * PREVIEW_DPI)
    gutter = 24
    with Image.open(left_png) as left_source, Image.open(right_png) as right_source:
        left = left_source.convert("RGB").resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        right = right_source.convert("RGB").resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        preview = Image.new(
            "RGB",
            (2 * target_width + gutter, target_height),
            color="white",
        )
        preview.paste(left, (0, 0))
        preview.paste(right, (target_width + gutter, 0))
        preview.save(
            output_path,
            format="PNG",
            dpi=(PREVIEW_DPI, PREVIEW_DPI),
            optimize=True,
        )
    return {
        "preview_width_px": 2 * target_width + gutter,
        "preview_height_px": target_height,
        "preview_dpi": PREVIEW_DPI,
        "preview_gutter_px": gutter,
    }


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dense, sparse, verification = _load_verified_pair()
    display_scale = float(verification["shared_display_scale_max_abs_raw"])

    dense_fig, dense_check = _build_panel(
        dense,
        title="Selected block",
        mathematical_label=r"$H_{\mathrm{block}}\; (8\!\times\!8)$",
        display_scale=display_scale,
    )
    dense_outputs = _save_panel(dense_fig, output_dir, "figure1_hblock")

    sparse_fig, sparse_check = _build_panel(
        sparse,
        title="Sparse support",
        mathematical_label=r"$H_{\mathrm{sparse}}\; (8\!\times\!8)$",
        display_scale=display_scale,
    )
    sparse_outputs = _save_panel(sparse_fig, output_dir, "figure1_hsparse")

    preview_path = output_dir / "figure1_hblock_hsparse_preview.png"
    preview_check = _write_side_by_side_preview(
        Path(dense_outputs["png"]),
        Path(sparse_outputs["png"]),
        preview_path,
    )

    verification.update(
        {
            "figure_dimensions_in": list(FIGSIZE_IN),
            "png_dpi": PNG_DPI,
            "colormap": {
                "negative": NEGATIVE_COLOR,
                "zero": ZERO_COLOR,
                "positive": POSITIVE_COLOR,
            },
            "dense_panel": dense_check,
            "sparse_panel": sparse_check,
            "preview": preview_check,
            "outputs": {
                "dense": dense_outputs,
                "sparse": sparse_outputs,
                "side_by_side_preview": str(preview_path),
            },
        }
    )

    output_paths = [
        *(Path(path) for path in dense_outputs.values()),
        *(Path(path) for path in sparse_outputs.values()),
        preview_path,
    ]
    verification["output_file_sha256"] = {
        path.name: file_sha256(path) for path in output_paths
    }
    verification_path = output_dir / "figure1_hblock_hsparse_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    return {
        "outputs": verification["outputs"],
        "verification_file": str(verification_path),
        "source_hashes": {
            "dense": verification["dense_array_sha256"],
            "sparse": verification["sparse_array_sha256"],
        },
        "support_nnz": verification["sparse_nonzero_count"],
        "shared_display_scale_max_abs_raw": display_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "pdf",
        help="destination directory (default: repository output/pdf)",
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
