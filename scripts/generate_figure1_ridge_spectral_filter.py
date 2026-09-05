#!/usr/bin/env python3
"""Generate the conceptual Ridge/Tikhonov spectral-filter panel for Figure 1.

The plotted samples are evaluated directly from

    f_alpha(sigma) = sigma / (sigma**2 + alpha)

and the pseudoinverse reference is evaluated directly from 1 / sigma.  The
default alpha is deliberately illustrative; it is not imported from an
experimental configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "qsvt_figure1_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
from PIL import Image


ALPHA = 1.0e-2
SIGMA_MIN = 1.0e-3
SIGMA_MAX = 1.0e3
N_SAMPLES = 6001
MAIN_FIGSIZE_IN = (3.35, 2.35)
MAIN_DPI = 600
PREVIEW_WIDTH_IN = 2.20
PREVIEW_DPI = 300
DECADE_EXPONENTS = tuple(range(-3, 4))
RIDGE_COLOR = "#2F6F9F"
REFERENCE_COLOR = "#3F3F3F"
ANNOTATION_COLOR = "#6A6A6A"


def ridge_filter(sigma: np.ndarray | float, alpha: float) -> np.ndarray:
    """Return the exact Ridge/Tikhonov spectral response."""

    sigma_array = np.asarray(sigma, dtype=float)
    return sigma_array / (sigma_array**2 + alpha)


def pseudoinverse_filter(sigma: np.ndarray | float) -> np.ndarray:
    """Return the exact positive-singular-value pseudoinverse response."""

    sigma_array = np.asarray(sigma, dtype=float)
    return 1.0 / sigma_array


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "path.simplify": False,
        }
    )


def _build_figure(alpha: float) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
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

    # Numerical scientific checks, including the actual sampled arrays that are plotted.
    assert ridge_zero == 0.0
    if math.isclose(alpha, ALPHA, rel_tol=0.0, abs_tol=0.0):
        assert math.isclose(sigma_star, 0.1, rel_tol=0.0, abs_tol=2.0e-17)
    assert math.isclose(sampled_sigma_maximum, sigma_star, rel_tol=0.0, abs_tol=2.0e-15)
    assert math.isclose(
        sampled_filter_maximum, analytic_maximum, rel_tol=2.0e-15, abs_tol=2.0e-15
    )
    assert np.all(np.diff(ridge[: sampled_max_index + 1]) > 0.0)
    assert np.all(np.diff(ridge[sampled_max_index:]) < 0.0)
    assert np.all(np.diff(pinv) < 0.0)
    assert math.isclose(
        float(ridge[-1] / pinv[-1]),
        SIGMA_MAX**2 / (SIGMA_MAX**2 + alpha),
        rel_tol=2.0e-15,
        abs_tol=0.0,
    )

    _configure_style()
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE_IN)
    fig.subplots_adjust(left=0.185, right=0.950, bottom=0.245, top=0.975)

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
        linewidth=1.05,
        linestyle=(0, (4.2, 2.7)),
        dash_capstyle="butt",
        zorder=2,
    )
    (ridge_line,) = ax.plot(
        sigma,
        ridge,
        color=RIDGE_COLOR,
        linewidth=1.75,
        solid_capstyle="round",
        zorder=3,
    )

    # Keep the y-axis linear and focused on the regularized response.  The exact
    # pseudoinverse samples above the visible range remain in the plotted line and
    # are clipped by the axes, correctly communicating its divergence as sigma -> 0.
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
    ax.set_ylim(0.0, 6.2)
    ax.set_xlabel(r"Singular value $\sigma$", labelpad=3.5)
    ax.set_ylabel("Filter value", labelpad=4.0)

    decade_ticks = np.power(10.0, np.asarray(DECADE_EXPONENTS, dtype=float))
    decade_labels = [rf"$10^{{{exponent}}}$" for exponent in DECADE_EXPONENTS]
    ax.xaxis.set_major_locator(FixedLocator(decade_ticks))
    ax.set_xticklabels(decade_labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 2.0, 4.0, 6.0])
    ax.tick_params(axis="both", which="major", pad=2.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.grid(False)

    ax.text(
        5.2e-3,
        1.05,
        r"$f_\alpha(\sigma)$",
        color=RIDGE_COLOR,
        fontsize=9.6,
        ha="left",
        va="bottom",
        zorder=4,
    )
    ax.text(
        2.7e-1,
        4.18,
        r"$1/\sigma$",
        color=REFERENCE_COLOR,
        fontsize=9.4,
        ha="left",
        va="bottom",
        zorder=4,
    )
    ax.annotate(
        r"$\sqrt{\alpha}$",
        xy=(sigma_star, 0.0),
        xytext=(5.0, 6.0),
        textcoords="offset points",
        color=ANNOTATION_COLOR,
        fontsize=8.8,
        ha="left",
        va="bottom",
        annotation_clip=False,
        zorder=4,
    )

    # Confirm that Matplotlib received the exact arrays rather than a drawn surrogate.
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
        "alpha_role_conceptual_illustrative": True,
        "sigma_star_analytic": sigma_star,
        "sigma_star_sampled": sampled_sigma_maximum,
        "ridge_maximum_analytic": analytic_maximum,
        "ridge_maximum_sampled": sampled_filter_maximum,
        "ridge_at_zero": ridge_zero,
        "ridge_at_sigma_min": float(ridge[0]),
        "ridge_to_pseudoinverse_ratio_at_sigma_max": float(ridge[-1] / pinv[-1]),
        "ridge_increases_through_sigma_star": True,
        "ridge_decreases_after_sigma_star": True,
        "pseudoinverse_strictly_decreasing": True,
        "x_axis_logarithmic": True,
        "y_axis_linear": True,
        "x_tick_exponents": list(DECADE_EXPONENTS),
        "x_tick_labels_mathtext": decade_labels,
        "x_axis_label": r"Singular value $\sigma$",
        "y_axis_label": "Filter value",
        "grid_enabled": False,
        "all_curve_samples_formula_evaluated": True,
        "embedded_title_or_caption": False,
    }
    return fig, ax, verification


def _write_preview(source_png: Path, preview_png: Path) -> tuple[int, int]:
    preview_width_px = round(PREVIEW_WIDTH_IN * PREVIEW_DPI)
    aspect = MAIN_FIGSIZE_IN[1] / MAIN_FIGSIZE_IN[0]
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


def generate(output_dir: Path, alpha: float = ALPHA) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure1_ridge_spectral_filter"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    preview_path = output_dir / f"{stem}_preview.png"
    verification_path = output_dir / f"{stem}_verification.json"

    fig, _, verification = _build_figure(alpha)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=MAIN_DPI)
    plt.close(fig)

    preview_width_px, preview_height_px = _write_preview(png_path, preview_path)
    verification.update(
        {
            "main_png_dpi": MAIN_DPI,
            "main_figure_width_in": MAIN_FIGSIZE_IN[0],
            "main_figure_height_in": MAIN_FIGSIZE_IN[1],
            "preview_dpi": PREVIEW_DPI,
            "preview_width_in": PREVIEW_WIDTH_IN,
            "preview_width_px": preview_width_px,
            "preview_height_px": preview_height_px,
            "pdf_is_vector_target": True,
            "svg_is_vector_target": True,
        }
    )
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    result: dict[str, object] = {
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf"),
        help="destination directory (default: output/pdf)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="positive conceptual alpha (default: 1e-2)",
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.alpha), indent=2))


if __name__ == "__main__":
    main()
