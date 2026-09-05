from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_GRID_COLUMNS = {
    "backend_name",
    "candidate_name",
    "evaluation_domain",
    "sigma_normalized",
    "target_value",
    "phase_response_value",
    "phase_response_abs_error",
}
REQUIRED_SUMMARY_COLUMNS = {
    "backend_name",
    "candidate_name",
    "alpha",
    "degree",
    "phase_count",
    "phase_response_max_error_full_domain",
    "phase_response_max_error_actual_singular_values_if_available",
    "passed_1e_minus_3_full_domain",
    "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript Figure 2 from the finalized real pyqsp/QSVT "
            "phase-validation pointwise output."
        )
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(
            "outputs/qsvt_external_backend_phase_validation/external_backend_phase_error_grid.csv"
        ),
        help="CSV with sigma grid, bounded target, phase response, and pointwise error.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/qsvt_external_backend_phase_validation/"
            "external_backend_phase_validation_summary.csv"
        ),
        help="CSV summary used to select the passing pyqsp validation row.",
    )
    parser.add_argument(
        "--phase-angles",
        type=Path,
        default=Path(
            "outputs/qsvt_external_backend_phase_validation/external_backend_phase_angles.csv"
        ),
        help="CSV of synthesized phase angles, used to verify the phase count.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name to plot. Defaults to the first passing full-domain row.",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate name to plot. Defaults to the first passing full-domain row.",
    )
    parser.add_argument(
        "--evaluation-domain",
        default="full_domain",
        choices=["full_domain", "actual_singular_values"],
        help="Validation domain to read from the pointwise grid.",
    )
    parser.add_argument(
        "--positive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For full-domain validation, plot only sigma >= 0 as singular values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for the generated manuscript figure.",
    )
    parser.add_argument(
        "--basename",
        default="figure2_qsvt_phase_validation",
        help="Output basename without extension.",
    )
    return parser.parse_args()


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return data


def select_summary_row(
    summary: pd.DataFrame, *, backend: str | None, candidate: str | None
) -> pd.Series:
    if backend is None and candidate is None:
        passed = summary[
            (summary["passed_1e_minus_3_full_domain"].astype(str).str.lower() == "true")
            & (summary["status"].astype(str) == "passed")
        ]
        if passed.empty:
            raise ValueError("No passing full-domain phase-validation row found")
        return passed.sort_values(
            ["phase_response_max_error_full_domain", "degree"],
            kind="mergesort",
        ).iloc[0]

    selected = summary
    if backend is not None:
        selected = selected[selected["backend_name"].astype(str) == backend]
    if candidate is not None:
        selected = selected[selected["candidate_name"].astype(str) == candidate]
    if selected.empty:
        raise ValueError(f"No summary row found for backend={backend!r}, candidate={candidate!r}")
    return selected.iloc[0]


def select_grid_rows(
    grid: pd.DataFrame,
    *,
    row: pd.Series,
    evaluation_domain: str,
    positive_only: bool,
) -> pd.DataFrame:
    selected = grid[
        (grid["backend_name"].astype(str) == str(row["backend_name"]))
        & (grid["candidate_name"].astype(str) == str(row["candidate_name"]))
        & (grid["evaluation_domain"].astype(str) == evaluation_domain)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No pointwise grid rows found for "
            f"{row['backend_name']} / {row['candidate_name']} / {evaluation_domain}"
        )
    if positive_only:
        selected = selected[selected["sigma_normalized"] >= 0.0]
    selected = selected.sort_values("sigma_normalized", kind="mergesort").reset_index(drop=True)
    if selected.empty:
        raise ValueError("No pointwise grid rows remain after positive-domain filtering")
    if (selected["phase_response_abs_error"] <= 0).any():
        raise ValueError("phase_response_abs_error must be positive for log-scale plotting")
    return selected


def phase_count_for_row(phase_angles: pd.DataFrame, row: pd.Series) -> int:
    selected = phase_angles[
        (phase_angles["backend_name"].astype(str) == str(row["backend_name"]))
        & (phase_angles["candidate_name"].astype(str) == str(row["candidate_name"]))
    ]
    if selected.empty:
        raise ValueError(
            f"No phase-angle rows found for {row['backend_name']} / {row['candidate_name']}"
        )
    return len(selected)


def make_figure(grid: pd.DataFrame, *, row: pd.Series, phase_count: int) -> plt.Figure:
    alpha = float(row["alpha"])
    degree = int(row["degree"])
    sigma_min = float(grid["sigma_normalized"].min())
    sigma_max = float(grid["sigma_normalized"].max())
    reported_max_error = float(row["phase_response_max_error_full_domain"])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        grid["sigma_normalized"],
        grid["target_value"],
        color="black",
        linewidth=1.8,
        label=r"Bounded target $f_{\alpha,\mathrm{bounded}}(\sigma)$",
    )
    ax.plot(
        grid["sigma_normalized"],
        grid["phase_response_value"],
        color="#2f6f9f",
        linewidth=1.6,
        linestyle="--",
        label="pyqsp/QSVT phase response",
    )
    ax.set_xlabel(r"Normalized singular value $\sigma$")
    ax.set_ylabel("Bounded filter response")
    ax.set_title("Target and phase response")
    ax.set_xlim(sigma_min, sigma_max)
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.96,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(
        grid["sigma_normalized"],
        grid["phase_response_abs_error"],
        color="#8a3b12",
        linewidth=1.6,
        label=r"$|f_{\rm QSVT}-f_{\rm target}|$",
    )
    ax.axhline(
        reported_max_error,
        color="0.25",
        linewidth=0.9,
        linestyle=":",
        label=rf"max = {reported_max_error:.2e}",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized singular value $\sigma$")
    ax.set_ylabel("Absolute error")
    ax.set_title("Pointwise approximation error")
    ax.set_xlim(sigma_min, sigma_max)
    ax.grid(True, which="both", color="0.85", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.96,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    fig.suptitle(
        rf"QSVT phase validation: $\alpha={alpha:g}$, degree {degree}, "
        rf"{phase_count} phases, $\sigma\in[{sigma_min:.3g},{sigma_max:g}]$",
        fontsize=10,
    )
    return fig


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    grid: pd.DataFrame,
    row: pd.Series,
    phase_count: int,
    pdf_path: Path,
    png_path: Path,
) -> None:
    metadata: dict[str, Any] = {
        "figure": "Figure 2 QSVT phase validation",
        "plotted_curve_type": "real pyqsp/QSVT phase response from repository output",
        "source_grid_csv": str(args.grid),
        "source_summary_csv": str(args.summary),
        "source_phase_angles_csv": str(args.phase_angles),
        "selected_backend": str(row["backend_name"]),
        "selected_candidate": str(row["candidate_name"]),
        "evaluation_domain": args.evaluation_domain,
        "plotted_domain": "sigma >= 0" if args.positive_only else "all grid values",
        "used_columns": [
            "sigma_normalized",
            "target_value",
            "phase_response_value",
            "phase_response_abs_error",
        ],
        "alpha": float(row["alpha"]),
        "singular_value_interval_plotted": [
            float(grid["sigma_normalized"].min()),
            float(grid["sigma_normalized"].max()),
        ],
        "polynomial_degree": int(row["degree"]),
        "phase_count": phase_count,
        "max_pointwise_error_plotted": float(grid["phase_response_abs_error"].max()),
        "mean_pointwise_error_plotted": float(grid["phase_response_abs_error"].mean()),
        "max_error_reported_full_domain": float(row["phase_response_max_error_full_domain"]),
        "max_error_reported_actual_singular_values": float(
            row["phase_response_max_error_actual_singular_values_if_available"]
        ),
        "status": str(row["status"]),
        "passed_1e_minus_3_full_domain": bool(row["passed_1e_minus_3_full_domain"]),
        "phase_convention": str(row.get("phase_convention", "")),
        "response_convention": str(row.get("response_convention", "")),
        "pdf_path": str(pdf_path),
        "png_path": str(png_path),
        "claim_boundary": (
            "Scalar phase-response validation only; this is not hardware execution, "
            "full IEEE-scale block encoding, quantum speedup, or a claim that QSVT "
            "outperforms ridge/Tikhonov."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_csv(args.summary, REQUIRED_SUMMARY_COLUMNS)
    grid = load_csv(args.grid, REQUIRED_GRID_COLUMNS)
    phase_angles = load_csv(
        args.phase_angles,
        {"backend_name", "candidate_name", "phase_index", "phase_angle"},
    )

    row = select_summary_row(summary, backend=args.backend, candidate=args.candidate)
    grid_rows = select_grid_rows(
        grid,
        row=row,
        evaluation_domain=args.evaluation_domain,
        positive_only=bool(args.positive_only),
    )
    phase_count = phase_count_for_row(phase_angles, row)
    reported_phase_count = int(row["phase_count"])
    if phase_count != reported_phase_count:
        raise ValueError(
            f"phase angle count {phase_count} does not match summary phase_count "
            f"{reported_phase_count}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{args.basename}.pdf"
    png_path = output_dir / f"{args.basename}.png"
    metadata_path = output_dir / f"{args.basename}_metadata.json"

    fig = make_figure(grid_rows, row=row, phase_count=phase_count)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_metadata(
        metadata_path,
        args=args,
        grid=grid_rows,
        row=row,
        phase_count=phase_count,
        pdf_path=pdf_path,
        png_path=png_path,
    )

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
