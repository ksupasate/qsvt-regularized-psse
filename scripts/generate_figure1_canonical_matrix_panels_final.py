#!/usr/bin/env python3
"""Final Figure-1 heatmaps for the canonical IEEE-14 selected block and its sparse support.

Both panels plot the *raw* repository values on one common symmetric scale
[-M, +M] with M = max |H_tilde_block| = 1347.042590759086, so dense and sparse
colours are directly comparable and no display normalization has to be
interpreted.  The dense block is re-verified as an exact submatrix of the
rebuilt 82x27 weighted Jacobian before anything is drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
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
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

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
PARENT_MATRIX_SHA256 = "c113dfa0ee5240182a77d2a845b0f4e950639c49884486d330a5c1351488fa5b"

# Widened relative to the display-normalized draft so the raw +/-1347 colorbar
# tick labels fit without shrinking the matrix cells.
FIGSIZE_IN = (2.62, 2.35)
MATRIX_RECT = (0.048, 0.055, 0.6698473282442748, 0.7468085106382979)
COLORBAR_RECT = (0.748, MATRIX_RECT[1], 0.0206, MATRIX_RECT[3])
TITLE_X = MATRIX_RECT[0] + MATRIX_RECT[2] / 2.0
PNG_DPI = 600
PREVIEW_DPI = 300

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

# Manuscript notation (manuscript/main.tex:33 \Hs, :202 and :209 \Hs_S).
BLOCK_LABEL = r"$\widetilde H_{B}\;(8\!\times\!8)$"
SPARSE_LABEL = r"$\widetilde H_{S}\;(8\!\times\!8)$"

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
        "svg.hashsalt": "figure1-canonical-matrix-panels-final",
        "path.simplify": False,
    }
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def array_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_verified_pair() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    dense = np.load(DENSE_SOURCE, allow_pickle=False)
    sparse = np.load(SPARSE_SOURCE, allow_pickle=False)
    if dense.dtype != np.float64 or sparse.dtype != np.float64:
        raise RuntimeError("canonical matrix artifacts must be float64")
    if dense.shape != (8, 8) or sparse.shape != (8, 8):
        raise RuntimeError("canonical matrix artifacts must both be 8x8")
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise RuntimeError("canonical matrix artifacts contain non-finite values")

    if not np.array_equal(np.loadtxt(DENSE_CSV_SOURCE, delimiter=","), dense):
        raise RuntimeError("dense CSV does not exactly match the dense NPY artifact")
    if not np.array_equal(np.loadtxt(SPARSE_CSV_SOURCE, delimiter=","), sparse):
        raise RuntimeError("sparse CSV does not exactly match the sparse NPY artifact")

    freeze = json.loads(CANDIDATE_FREEZE.read_text(encoding="utf-8"))
    dense_hash = array_sha256(dense)
    sparse_hash = array_sha256(sparse)
    if dense_hash != str(freeze["component_hashes"]["matrix_original"]):
        raise RuntimeError("dense block hash mismatch against the frozen candidate")
    if sparse_hash != str(freeze["component_hashes"]["matrix_supported_exact"]):
        raise RuntimeError("sparse block hash mismatch against the frozen candidate")

    reconstruction = json.loads(RECONSTRUCTION.read_text(encoding="utf-8"))
    if not np.array_equal(np.asarray(reconstruction["original_block"], dtype=np.float64), dense):
        raise RuntimeError("dense source differs from the frozen reconstruction")
    if not np.array_equal(np.asarray(reconstruction["sparsified_block"], dtype=np.float64), sparse):
        raise RuntimeError("sparse source differs from the frozen reconstruction")

    # Exact-submatrix test against the deterministically rebuilt parent system.
    from robust_qsvt_se.cross_case_validation.common import build_case_full_system

    parent = np.asarray(build_case_full_system("ieee14", seed=123).matrix, dtype=np.float64)
    if parent.shape != (82, 27):
        raise RuntimeError("rebuilt parent weighted Jacobian has an unexpected shape")
    parent_hash = array_sha256(parent)
    if parent_hash != PARENT_MATRIX_SHA256:
        raise RuntimeError(f"parent matrix hash mismatch: {parent_hash}")
    rows = [int(r) for r in freeze["global_rows"]]
    columns = [int(c) for c in freeze["global_columns"]]
    submatrix = parent[np.ix_(rows, columns)]
    max_abs_difference = float(np.max(np.abs(submatrix - dense)))
    if max_abs_difference != 0.0:
        raise RuntimeError(f"selected block is not an exact submatrix ({max_abs_difference})")

    support_mask = np.zeros((8, 8), dtype=bool)
    for row, column in freeze["support_coordinates"]:
        support_mask[int(row), int(column)] = True
    actual_support = sparse != 0.0
    if not np.array_equal(actual_support, support_mask):
        raise RuntimeError("sparse nonzero pattern differs from the frozen support coordinates")
    if int(actual_support.sum()) != 16:
        raise RuntimeError("canonical sparse support must contain exactly 16 entries")
    if int(actual_support.sum(axis=1).max()) > 2:
        raise RuntimeError("canonical sparse support exceeds two retained entries per row")
    if not np.array_equal(sparse[support_mask], dense[support_mask]):
        raise RuntimeError("retained sparse values do not equal the corresponding dense entries")
    if not np.all(sparse[~support_mask] == 0.0):
        raise RuntimeError("off-support sparse entries are not exactly zero")

    scale = float(np.max(np.abs(dense)))
    if scale != 1347.042590759086:
        raise RuntimeError(f"unexpected common raw colour scale: {scale!r}")
    if float(np.max(np.abs(sparse))) != scale:
        raise RuntimeError("dense and sparse blocks do not share the canonical maximum magnitude")

    verification: dict[str, object] = {
        "workload_id": str(freeze["workload_id"]),
        "parent_matrix": "IEEE-14 weighted Jacobian H_tilde (82x27), rebuilt from build_case_full_system('ieee14', seed=123)",
        "parent_matrix_sha256": parent_hash,
        "parent_matrix_min": float(parent.min()),
        "parent_matrix_max": float(parent.max()),
        "block_global_rows": rows,
        "block_global_columns": columns,
        "block_exact_submatrix": True,
        "block_max_abs_difference_vs_parent": max_abs_difference,
        "dense_source": _display_path(DENSE_SOURCE),
        "sparse_source": _display_path(SPARSE_SOURCE),
        "dense_array_sha256": dense_hash,
        "sparse_array_sha256": sparse_hash,
        "dense_min": float(dense.min()),
        "dense_max": float(dense.max()),
        "dense_max_abs": scale,
        "sparse_min": float(sparse.min()),
        "sparse_max": float(sparse.max()),
        "sparse_retained_entries": int(actual_support.sum()),
        "sparse_total_entries": int(sparse.size),
        "sparse_row_nonzeros": [int(v) for v in actual_support.sum(axis=1)],
        "sparse_column_nonzeros": [int(v) for v in actual_support.sum(axis=0)],
        "sparse_max_entries_per_row": int(actual_support.sum(axis=1).max()),
        "retained_values_exactly_equal_dense": True,
        "removed_values_exactly_zero": True,
        "support_generation": (
            "sparsify_block(keep_per_row=2) at src/robust_qsvt_se/qsvt/"
            "toy_sparse_oracle_block_encoding_v2.py:66, invoked with KEEP_PER_ROW=2 from "
            "src/robust_qsvt_se/paper/phase10_sparse_wrapper_8x8_complete.py:142; per-row "
            "retention of the two largest-magnitude entries, no column pruning, no threshold "
            "parameter, no one-swap refinement"
        ),
        "representation": "raw repository values (no display normalization)",
        "common_raw_colour_scale": [-scale, scale],
        "row_order_preserved": True,
        "column_order_preserved": True,
        "transposed": False,
        "interpolation": "none (vector pcolormesh cells)",
    }
    return dense, sparse, verification


def _artist_is_inside_figure(fig: plt.Figure, artist: object) -> bool:
    renderer = fig.canvas.get_renderer()
    bounds = artist.get_window_extent(renderer=renderer)
    frame = fig.bbox
    tolerance = 0.5
    return bool(
        bounds.x0 >= frame.x0 - tolerance
        and bounds.y0 >= frame.y0 - tolerance
        and bounds.x1 <= frame.x1 + tolerance
        and bounds.y1 <= frame.y1 + tolerance
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

    # The colormap is piecewise linear blue -> white -> red, so three Gouraud
    # nodes reproduce it exactly as vector geometry instead of a rasterized ramp.
    color_nodes = np.asarray([-display_scale, 0.0, display_scale], dtype=float)
    color_x, color_y = np.meshgrid(np.asarray([0.0, 1.0]), color_nodes)
    cax.pcolormesh(
        color_x,
        color_y,
        np.repeat(color_nodes[:, np.newaxis], 2, axis=1),
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
    raw_tick_labels = [r"$-1347$", r"$0$", r"$+1347$"]
    cax.set_yticklabels(raw_tick_labels)
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
        TITLE_X, 0.944, title, ha="center", va="center", fontsize=9.7, color=TEXT_COLOR
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
        "colorbar_tick_positions_bottom_to_top": [-display_scale, 0.0, display_scale],
        "colorbar_tick_labels_top_to_bottom": ["+1347", "0", "-1347"],
        "colorbar_label": None,
        "display_vmin_raw": -display_scale,
        "display_vmax_raw": display_scale,
        "display_normalization_applied": False,
        "title_or_label_clipping": False,
    }
    return fig, panel_check


def _save_panel(fig: plt.Figure, dirs: dict[str, Path], stem: str) -> dict[str, str]:
    paths = {
        "pdf": dirs["pdf"] / f"{stem}.pdf",
        "svg": dirs["svg"] / f"{stem}.svg",
        "png": dirs["png"] / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], format="pdf", transparent=False)
    fig.savefig(paths["svg"], format="svg", transparent=False)
    fig.savefig(paths["png"], format="png", dpi=PNG_DPI, transparent=False)
    plt.close(fig)
    return {kind: _display_path(path) for kind, path in paths.items()}


def _write_side_by_side_preview(left_png: Path, right_png: Path, output_path: Path) -> dict[str, object]:
    width = round(FIGSIZE_IN[0] * PREVIEW_DPI)
    height = round(FIGSIZE_IN[1] * PREVIEW_DPI)
    gutter = 24
    with Image.open(left_png) as left_source, Image.open(right_png) as right_source:
        left = left_source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        right = right_source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        preview = Image.new("RGB", (2 * width + gutter, height), color="white")
        preview.paste(left, (0, 0))
        preview.paste(right, (width + gutter, 0))
        preview.save(output_path, format="PNG", dpi=(PREVIEW_DPI, PREVIEW_DPI), optimize=True)
    return {
        "preview_width_px": 2 * width + gutter,
        "preview_height_px": height,
        "preview_dpi": PREVIEW_DPI,
        "preview_gutter_px": gutter,
    }


def generate(output_root: Path) -> dict[str, object]:
    dirs = {kind: output_root / kind for kind in ("pdf", "svg", "png")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    dense, sparse, verification = _load_verified_pair()
    display_scale = float(verification["dense_max_abs"])

    dense_fig, dense_check = _build_panel(
        dense,
        title="Selected block",
        mathematical_label=BLOCK_LABEL,
        display_scale=display_scale,
    )
    dense_outputs = _save_panel(dense_fig, dirs, "figure1_hblock_final")
    dense_png = dirs["png"] / "figure1_hblock_final.png"

    sparse_fig, sparse_check = _build_panel(
        sparse,
        title="Sparse support",
        mathematical_label=SPARSE_LABEL,
        display_scale=display_scale,
    )
    sparse_outputs = _save_panel(sparse_fig, dirs, "figure1_hsparse_final")
    sparse_png = dirs["png"] / "figure1_hsparse_final.png"

    preview_path = dirs["png"] / "figure1_block_sparse_pair_preview.png"
    preview_check = _write_side_by_side_preview(
        dense_png,
        sparse_png,
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
                "cell_boundary": GRID_COLOR,
            },
            "background": "white",
            "block_panel": dense_check,
            "sparse_panel": sparse_check,
            "pair_preview": dict(
                preview_check, path=_display_path(preview_path)
            ),
            "panels_share_one_scale": dense_check["display_vmax_raw"] == sparse_check["display_vmax_raw"],
        }
    )
    verification_path = dirs["pdf"] / "figure1_block_sparse_final_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    return {
        "block_outputs": dense_outputs,
        "sparse_outputs": sparse_outputs,
        "pair_preview": _display_path(preview_path),
        "verification_json": _display_path(verification_path),
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "output")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_root), indent=2))


if __name__ == "__main__":
    main()
