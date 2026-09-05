#!/usr/bin/env python3
"""Four-condition evidence-framework diagram.

Script-generated schematic of logically ordered requirements evaluated through linked evidence
families: support preservation, truth-referenced usefulness, matched-filter implementation, and
access/readout cost. Deliberately number-free: quantitative values live in the generated tables
the caption cross-references. Palette and box style follow the existing workflow diagram
(fig_generic_sparse_qsvt_compiler.pdf); rendered with matplotlib like the repository's other
figure-generation scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "manuscript" / "figures" / "fig_four_test_decision_diagram.pdf"

NAVY = "#183B56"
INK = "#17212B"
MUTED = "#506273"
BOUNDARY = "#EFEFEF"

TESTS = [
    (
        "#DDEBF7",
        "Condition 1",
        "Support preservation",
        "Does retained support\nreproduce the full-block Ridge\n"
        "selected output? (support error)",
    ),
    (
        "#DDF2EE",
        "Condition 2",
        "Truth-referenced\nusefulness",
        "Does the surrogate track the\nfull-system benchmark-\n"
        "reference selected output?\n(reference error)",
    ),
    (
        "#F7ECD1",
        "Condition 3",
        "Matched-filter\nimplementation",
        "Can a bounded odd polynomial\nand synthesized phases realize\n"
        "the filter? (QSVT error)",
    ),
    (
        "#F5E1E5",
        "Condition 4",
        "Access / readout\ncost",
        "What do access, postselection,\nshots, and classical baselines\ncost? (resource ledgers)",
    ),
]


def build_diagram(path: Path = FIGURE_PATH) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    # Header band.
    ax.add_patch(
        FancyBboxPatch(
            (0, 33.4), 100, 4.6, boxstyle="square,pad=0", facecolor=NAVY, edgecolor="none"
        )
    )
    ax.text(
        1.5,
        35.7,
        "Four-condition evidence framework for candidate\nselected-output QSVT computations",
        color="white",
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )

    box_w, box_h, gap = 22.6, 15.5, 2.6
    left = (100 - 4 * box_w - 3 * gap) / 2
    box_y = 15.5
    centers = []
    for index, (color, tag, title, body) in enumerate(TESTS):
        x = left + index * (box_w + gap)
        centers.append(x + box_w / 2)
        ax.add_patch(
            FancyBboxPatch(
                (x, box_y),
                box_w,
                box_h,
                boxstyle="round,pad=0,rounding_size=1.1",
                facecolor=color,
                edgecolor=NAVY,
                linewidth=1.0,
            )
        )
        ax.text(
            x + box_w / 2,
            box_y + box_h - 2.0,
            tag,
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            color=MUTED,
        )
        ax.text(
            x + box_w / 2,
            box_y + box_h - 4.4,
            title,
            ha="center",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + box_w / 2,
            box_y + box_h / 2 - 3.2,
            body,
            ha="center",
            va="center",
            fontsize=5.4,
            color=INK,
            linespacing=1.4,
        )
        if index < 3:
            x1 = x + box_w
            ax.add_patch(
                FancyArrowPatch(
                    (x1 + 0.3, box_y + box_h / 2),
                    (x1 + gap - 0.3, box_y + box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=NAVY,
                    linewidth=1.2,
                )
            )
    # Shared boundary lane: linked evidence families retain every registered outcome.
    lane_y, lane_h = 4.6, 6.8
    ax.add_patch(
        FancyBboxPatch(
            (left, lane_y),
            4 * box_w + 3 * gap,
            lane_h,
            boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=BOUNDARY,
            edgecolor=MUTED,
            linewidth=0.9,
        )
    )
    ax.text(
        50,
        lane_y + lane_h - 1.9,
        "Linked evidence families (failures retained, never substituted)",
        ha="center",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        50,
        lane_y + 1.9,
        "Each outcome, positive or negative, becomes a registered table row with its "
        "evidence tier",
        ha="center",
        fontsize=6.2,
        color=INK,
    )
    for center in centers:
        ax.add_patch(
            FancyArrowPatch(
                (center, box_y - 0.2),
                (center, lane_y + lane_h + 0.3),
                arrowstyle="-|>",
                mutation_scale=8,
                color=MUTED,
                linewidth=1.0,
            )
        )

    ax.text(
        1.5,
        1.2,
        "Evidence from later conditions does not override an earlier application failure; "
        "the arrows show logical order, not one exhaustive joint grid.",
        fontsize=6.7,
        color=MUTED,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_diagram()
    print(f"wrote {FIGURE_PATH.relative_to(ROOT)}")
