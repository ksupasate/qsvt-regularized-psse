#!/usr/bin/env python3
"""Build the manuscript architecture figure from frozen compiler ledgers.

The output is a manuscript-facing vector diagram.  It reads and validates the
canonical IEEE-14 and IEEE-30 register/resource ledgers and never changes the
compiler, circuit, configuration, or scientific evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTER_LEDGER = (
    ROOT
    / "outputs"
    / "generic_sparse_qsvt_compiler"
    / "canonical_register_ledger_generic.csv"
)
TRANSFER_REGISTER_LEDGER = (
    ROOT
    / "outputs"
    / "generic_sparse_qsvt_compiler"
    / "second_workload_register_ledger.csv"
)
RESOURCE_LEDGER = (
    ROOT
    / "outputs"
    / "generic_sparse_qsvt_compiler"
    / "canonical_resource_ledger_generic.csv"
)
SPECIFICATION = (
    ROOT
    / "outputs"
    / "generic_sparse_qsvt_compiler"
    / "compiler_specification.md"
)
OUTPUT = ROOT / "manuscript" / "figures" / "fig_sparse_selected_output_architecture.pdf"
SOURCE_RECORD = (
    ROOT
    / "outputs"
    / "final_repository_backed_manuscript_completion"
    / "architecture_figure_sources.json"
)

EXPECTED_REGISTERS = {
    "index": ("0;1;2", 3),
    "slot": ("3;4", 2),
    "rotation_ancilla": ("5", 1),
    "postselection_flag": ("6", 1),
    "signed_readout": ("7", 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ledgers() -> dict[str, int]:
    canonical = pd.read_csv(REGISTER_LEDGER)
    transfer = pd.read_csv(TRANSFER_REGISTER_LEDGER)
    for label, frame in (("IEEE-14", canonical), ("IEEE-30", transfer)):
        actual = {
            str(row.register_name): (str(row.qubit_indices), int(row.count))
            for row in frame.itertuples(index=False)
        }
        if actual != EXPECTED_REGISTERS:
            raise RuntimeError(f"{label} register ledger does not match declared layout: {actual}")
        if not frame["simultaneously_live"].astype(bool).all():
            raise RuntimeError(f"{label} ledger contains a non-live declared register")

    resources = pd.read_csv(RESOURCE_LEDGER).iloc[0]
    if int(resources["total_simultaneously_live_qubits"]) != 8:
        raise RuntimeError("canonical final circuit does not report eight live qubits")
    if int(resources["qsvt_signal_calls"]) != 31:
        raise RuntimeError("canonical final circuit does not report 31 QSVT signal calls")
    if int(resources["phase_operations"]) != 32:
        raise RuntimeError("canonical final circuit does not report 32 phase operations")
    if bool(resources["dense_fallback_used"]):
        raise RuntimeError("dense fallback is marked as used")
    return {
        "qubits": int(resources["total_simultaneously_live_qubits"]),
        "signal_calls": int(resources["qsvt_signal_calls"]),
        "phase_operations": int(resources["phase_operations"]),
    }


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    title: str,
    body: str,
    *,
    category: str,
    width: float = 2.15,
    height: float = 1.25,
) -> FancyBboxPatch:
    palette = {
        "classical": ("#E8EEF5", "#315A7D"),
        "coherent": ("#E7F3EC", "#2F6B48"),
        "measurement": ("#FFF1E6", "#B85C1E"),
    }
    face, edge = palette[category]
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.055,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.23,
        title,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#17212B",
        linespacing=1.05,
    )
    axis.text(
        x + width / 2,
        y + 0.43,
        body,
        ha="center",
        va="center",
        fontsize=6.25,
        color="#263442",
        linespacing=1.18,
    )
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    style: str = "solid",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            linestyle=style,
            color="#4B5563",
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_figure(counts: dict[str, int]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.15, 5.0))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 7.0)
    axis.axis("off")

    axis.text(
        5.1,
        6.72,
        "Sparse selected-output QSVT: small-scale simulated-circuit workflow",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#17212B",
    )
    axis.text(
        5.1,
        6.42,
        "Executed evidence: two IEEE-derived 8×8 workloads; no IEEE-scale reversible access or hardware execution",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#7A3417",
    )

    x_positions = (0.25, 2.75, 5.25, 7.75)
    top_y = 4.82
    bottom_y = 3.05
    boxes = [
        add_box(
            axis,
            (x_positions[0], top_y),
            "1  Classical workload\nconstruction",
            "weighted residual + sparse support\nslot assignment, quantization,\n$\\beta$, phases, and functional",
            category="classical",
        ),
        add_box(
            axis,
            (x_positions[1], top_y),
            "2  Residual-state\npreparation",
            "$|r\\rangle=r/\\|r\\|$\non index $q_0$--$q_2$\n(controlled by readout branch)",
            category="coherent",
        ),
        add_box(
            axis,
            (x_positions[2], top_y),
            "3  Sparse block-encoding\naccess",
            "slot diffusion + stored signed\nvalue rotations + in-place index\npermutations; wrapper $U/U^\\dagger$",
            category="coherent",
        ),
        add_box(
            axis,
            (x_positions[3], top_y),
            "4  QSVT phase\nsequence",
            f"{counts['signal_calls']} signal calls and\n{counts['phase_operations']} projector phases\non the encoded signal subspace",
            category="coherent",
        ),
        add_box(
            axis,
            (x_positions[0], bottom_y),
            "8  Physical\nrescaling",
            "$\\widehat y_\\ell=(C/\\beta)\\|r\\|\\|\\ell\\|$\n$\\times(N_{00}-N_{10})/N_{\\rm attempted}$\n(classical postprocessing)",
            category="classical",
        ),
        add_box(
            axis,
            (x_positions[1], bottom_y),
            "7  Signed\nreadout",
            "closing Hadamard on $q_7$;\njoint readout/postselection counts\n$c_1c_0$",
            category="measurement",
        ),
        add_box(
            axis,
            (x_positions[2], bottom_y),
            "6  Functional-state\npreparation",
            "$|\\ell\\rangle=\\ell/\\|\\ell\\|$\non the index register;\nreal interference reference",
            category="coherent",
        ),
        add_box(
            axis,
            (x_positions[3], bottom_y),
            "5  Signal/postselection\nprojection",
            "encoded work-zero projector;\naggregate flag $q_6$ and\npostselection measurement $c_0$",
            category="measurement",
        ),
    ]
    for left, right in zip(boxes[:3], boxes[1:4], strict=True):
        arrow(
            axis,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2),
            (right.get_x(), right.get_y() + right.get_height() / 2),
        )
    arrow(
        axis,
        (
            boxes[3].get_x() + boxes[3].get_width() / 2,
            boxes[3].get_y(),
        ),
        (
            boxes[7].get_x() + boxes[7].get_width() / 2,
            boxes[7].get_y() + boxes[7].get_height(),
        ),
    )
    for right, left in zip(boxes[4:7], boxes[5:8], strict=True):
        arrow(
            axis,
            (left.get_x(), left.get_y() + left.get_height() / 2),
            (
                right.get_x() + right.get_width(),
                right.get_y() + right.get_height() / 2,
            ),
        )

    register_y = 1.36
    axis.text(
        0.25,
        register_y + 0.93,
        "Declared live register layout (both executed workloads)",
        ha="left",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#17212B",
    )
    register_items = [
        ("Index", "$q_0$--$q_2$", "3 qubits"),
        ("Slot", "$q_3,q_4$", "2 qubits"),
        ("Value rotation", "$q_5$", "1 ancilla"),
        ("Postselection", "$q_6$", "1 flag"),
        ("Readout", "$q_7$", "1 qubit"),
    ]
    for index, (name, indices, count) in enumerate(register_items):
        x = 0.25 + index * 1.62
        axis.add_patch(
            FancyBboxPatch(
                (x, register_y),
                1.48,
                0.7,
                boxstyle="round,pad=0.035,rounding_size=0.04",
                facecolor="#F7F8FA",
                edgecolor="#697586",
                linewidth=0.8,
            )
        )
        axis.text(x + 0.74, register_y + 0.49, name, ha="center", va="center", fontsize=7.3, fontweight="bold")
        axis.text(x + 0.74, register_y + 0.23, f"{indices}; {count}", ha="center", va="center", fontsize=6.6)

    axis.add_patch(
        FancyBboxPatch(
            (8.42, register_y),
            1.52,
            0.7,
            boxstyle="round,pad=0.035,rounding_size=0.04",
            facecolor="#F2F0FA",
            edgecolor="#665A91",
            linewidth=0.8,
        )
    )
    axis.text(9.18, register_y + 0.49, "Signal subspace", ha="center", va="center", fontsize=7.3, fontweight="bold")
    axis.text(
        9.18,
        register_y + 0.22,
        "slot=$00$, $q_5=0$\n(no separate signal qubit)",
        ha="center",
        va="center",
        fontsize=6.2,
        linespacing=1.05,
    )

    legend_y = 0.54
    legend = [
        ("#E8EEF5", "#315A7D", "Classical preprocessing/postprocessing"),
        ("#E7F3EC", "#2F6B48", "Coherent circuit operation"),
        ("#FFF1E6", "#B85C1E", "Projection/measurement evidence"),
    ]
    for index, (face, edge, label) in enumerate(legend):
        x = 0.25 + index * 3.02
        axis.add_patch(
            FancyBboxPatch(
                (x, legend_y),
                0.34,
                0.24,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        axis.text(x + 0.46, legend_y + 0.12, label, ha="left", va="center", fontsize=6.7)
    axis.text(
        5.1,
        0.18,
        "Modeled IEEE-scale access is a separate access-cost model and is not depicted as an executed circuit.",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#7A3417",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#B85C1E",
            "linestyle": "--",
            "linewidth": 0.8,
        },
    )

    figure.tight_layout(pad=0.3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", metadata={"Creator": __file__})
    plt.close(figure)


def main() -> None:
    counts = validate_ledgers()
    build_figure(counts)
    record = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in (
                REGISTER_LEDGER,
                TRANSFER_REGISTER_LEDGER,
                RESOURCE_LEDGER,
                SPECIFICATION,
            )
        ],
        "validated_registers": EXPECTED_REGISTERS,
        "validated_resource_counts": counts,
        "scope": (
            "small-scale simulated-circuit workflow; no IEEE-scale reversible "
            "access or hardware execution"
        ),
    }
    SOURCE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"wrote {SOURCE_RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
