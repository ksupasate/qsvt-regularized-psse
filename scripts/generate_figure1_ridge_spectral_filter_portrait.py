#!/usr/bin/env python3
"""Generate the portrait Ridge/Tikhonov spectral-filter panel for Figure 1.

This layout-only variant imports the same exact filter functions and illustrative
alpha as the landscape source.  No experimental parameter is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Import the shared source first so it establishes writable Matplotlib/font caches.
from generate_figure1_ridge_spectral_filter import (
    ALPHA,
    ANNOTATION_COLOR,
    DECADE_EXPONENTS,
    N_SAMPLES,
    REFERENCE_COLOR,
    RIDGE_COLOR,
    SIGMA_MAX,
    SIGMA_MIN,
    _configure_style,
    pseudoinverse_filter,
    ridge_filter,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
from PIL import Image


PORTRAIT_FIGSIZE_IN = (2.40, 3.80)
PORTRAIT_DPI = 600
PREVIEW_WIDTH_IN = 1.80
PREVIEW_DPI = 300


def _build_portrait_figure(alpha: float) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")

    sigma = np.logspace(
        math.log10(SIGMA_MIN), math.log10(SIGMA_MAX), N_SAMPLES, dtype=float
    )
    ridge = ridge_filter(sigma, alpha)
    pinv = pseudoinverse_filter(sigma)

    sigma_star = math.sqrt(alpha)
    ridge_zero = float(ridge_filter(0.0, alpha))
    analytic_maximum = 1.0 / (2.0 * sigma_star)
    sampled_max_index = int(np.argmax(ridge))
    sampled_sigma_maximum = float(sigma[sampled_max_index])
    sampled_filter_maximum = float(ridge[sampled_max_index])

    # Scientific checks apply to the exact arrays passed to Matplotlib below.
    assert alpha == ALPHA
    assert ridge_zero == 0.0
    assert math.isclose(sigma_star, 0.1, rel_tol=0.0, abs_tol=2.0e-17)
    assert math.isclose(sampled_sigma_maximum, sigma_star, rel_tol=0.0, abs_tol=2.0e-15)
    assert math.isclose(
        sampled_filter_maximum, analytic_maximum, rel_tol=2.0e-15, abs_tol=2.0e-15
    )
    assert np.all(np.diff(ridge[: sampled_max_index + 1]) > 0.0)
    assert np.all(np.diff(ridge[sampled_max_index:]) < 0.0)
    assert np.all(np.diff(pinv) < 0.0)

    _configure_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 10.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 8.6,
        }
    )
    fig, ax = plt.subplots(figsize=PORTRAIT_FIGSIZE_IN)
    fig.subplots_adjust(left=0.260, right=0.940, bottom=0.150, top=0.980)

    ax.axvline(
        sigma_star,
        color=ANNOTATION_COLOR,
        linewidth=0.75,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.72,
        zorder=1,
    )
    (pinv_line,) = ax.plot(
        sigma,
        pinv,
        color=REFERENCE_COLOR,
        linewidth=1.00,
        linestyle=(0, (4.2, 2.7)),
        dash_capstyle="butt",
        zorder=2,
    )
    (ridge_line,) = ax.plot(
        sigma,
        ridge,
        color=RIDGE_COLOR,
        linewidth=1.70,
        solid_capstyle="round",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
    ax.set_ylim(0.0, 6.2)
    ax.set_xlabel(r"Singular value $\sigma$", labelpad=4.0)
    ax.set_ylabel("Filter value", labelpad=4.0)

    decade_ticks = np.power(10.0, np.asarray(DECADE_EXPONENTS, dtype=float))
    decade_labels = [rf"$10^{{{exponent}}}$" for exponent in DECADE_EXPONENTS]
    ax.xaxis.set_major_locator(FixedLocator(decade_ticks))
    ax.set_xticklabels(decade_labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 2.0, 4.0, 6.0])
    ax.tick_params(axis="x", which="major", pad=2.0)
    ax.tick_params(axis="y", which="major", pad=2.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.grid(False)

    ax.text(
        3.0e-3,
        1.05,
        r"$f_\alpha(\sigma)$",
        color=RIDGE_COLOR,
        fontsize=9.6,
        ha="left",
        va="bottom",
        zorder=4,
    )
    ax.text(
        2.8e-1,
        4.22,
        r"$1/\sigma$",
        color=REFERENCE_COLOR,
        fontsize=9.3,
        ha="left",
        va="bottom",
        zorder=4,
    )
    ax.annotate(
        r"$\sqrt{\alpha}$",
        xy=(sigma_star, 0.0),
        xytext=(5.0, 8.0),
        textcoords="offset points",
        color=ANNOTATION_COLOR,
        fontsize=8.8,
        ha="left",
        va="bottom",
        annotation_clip=False,
        zorder=4,
    )

    assert np.array_equal(np.asarray(ridge_line.get_xdata()), sigma)
    assert np.array_equal(np.asarray(ridge_line.get_ydata()), ridge)
    assert np.array_equal(np.asarray(pinv_line.get_xdata()), sigma)
    assert np.array_equal(np.asarray(pinv_line.get_ydata()), pinv)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    assert np.array_equal(ax.get_xticks(), decade_ticks)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == decade_labels
    assert ax.get_xlabel() == r"Singular value $\sigma$"
    assert ax.get_ylabel() == "Filter value"
    assert ax.get_title() == ""
    assert ax.get_legend() is None
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())

    verification: dict[str, object] = {
        "alpha": alpha,
        "alpha_role": "conceptual illustrative alpha",
        "sigma_star_analytic": sigma_star,
        "sigma_star_sampled": sampled_sigma_maximum,
        "ridge_maximum_analytic": analytic_maximum,
        "ridge_maximum_sampled": sampled_filter_maximum,
        "ridge_at_zero": ridge_zero,
        "ridge_increases_through_sigma_star": True,
        "ridge_decreases_after_sigma_star": True,
        "pseudoinverse_strictly_decreasing": True,
        "ridge_to_pseudoinverse_ratio_at_sigma_max": float(ridge[-1] / pinv[-1]),
        "all_curve_samples_formula_evaluated": True,
        "x_axis_logarithmic": True,
        "y_axis_linear": True,
        "x_tick_exponents": list(DECADE_EXPONENTS),
        "x_tick_labels_mathtext": decade_labels,
        "grid_enabled": False,
        "embedded_title_caption_or_legend": False,
        "figure_width_in": PORTRAIT_FIGSIZE_IN[0],
        "figure_height_in": PORTRAIT_FIGSIZE_IN[1],
        "width_to_height_ratio": PORTRAIT_FIGSIZE_IN[0] / PORTRAIT_FIGSIZE_IN[1],
    }
    return fig, ax, verification


def _write_preview(source_png: Path, preview_png: Path) -> tuple[int, int]:
    preview_width_px = round(PREVIEW_WIDTH_IN * PREVIEW_DPI)
    aspect = PORTRAIT_FIGSIZE_IN[1] / PORTRAIT_FIGSIZE_IN[0]
    preview_height_px = round(preview_width_px * aspect)
    with Image.open(source_png) as image:
        reduced = image.convert("RGB").resize(
            (preview_width_px, preview_height_px), Image.Resampling.LANCZOS
        )
        reduced.save(
            preview_png,
            format="PNG",
            dpi=(PREVIEW_DPI, PREVIEW_DPI),
            optimize=True,
        )
    return preview_width_px, preview_height_px


def generate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure1_ridge_spectral_filter_portrait"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    preview_path = output_dir / f"{stem}_preview.png"
    verification_path = output_dir / f"{stem}_verification.json"

    fig, _, verification = _build_portrait_figure(ALPHA)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=PORTRAIT_DPI)
    plt.close(fig)

    preview_width_px, preview_height_px = _write_preview(png_path, preview_path)
    verification.update(
        {
            "main_png_dpi": PORTRAIT_DPI,
            "preview_dpi": PREVIEW_DPI,
            "preview_width_in": PREVIEW_WIDTH_IN,
            "preview_width_px": preview_width_px,
            "preview_height_px": preview_height_px,
            "pdf_saved_by_vector_backend": True,
            "svg_saved_by_vector_backend": True,
        }
    )
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    return {
        "classification": "conceptual illustrative alpha",
        "outputs": {
            "pdf": str(pdf_path),
            "svg": str(svg_path),
            "png_600_dpi": str(png_path),
            "preview": str(preview_path),
            "verification": str(verification_path),
        },
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf"),
        help="destination directory (default: output/pdf)",
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
