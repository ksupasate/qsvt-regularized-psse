from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import build_dense_block_encoding
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    DEFAULT_SUBPROBLEMS,
    SweepSubproblem,
    load_sweep_subproblem,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCK_RESULTS_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "original_matrix_shape",
    "padded_matrix_shape",
    "weighted_status",
    "gamma",
    "spectral_norm_A",
    "frobenius_norm_A",
    "condition_number_A",
    "spectral_norm_A_bar",
    "block_error_frobenius",
    "spectral_block_error",
    "unitarity_error_frobenius",
    "unitarity_error_spectral",
    "padded_dimension",
    "system_qubits",
    "block_encoding_ancilla_qubits",
    "total_qubits",
    "dense_unitary_dimension",
    "dense_unitary_parameter_count",
    "estimated_dense_decomposition_cost",
    "gate_decomposition_produced",
    "gate_decomposition_status",
    "matrix_original_path",
    "matrix_normalized_path",
    "matrix_padded_path",
    "unitary_path",
    "run_status",
    "failure_mode",
    "failure_reason",
    "claim_boundary_note",
]


@dataclass(frozen=True, slots=True)
class PaddedBlockEncoding:
    A: np.ndarray
    A_bar: np.ndarray
    A_bar_padded: np.ndarray
    U: np.ndarray
    gamma: float
    padded_dimension: int


def run_explicit_block_encoding_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = paths["block"]
    matrices_dir = ensure_directory(output_dir / "matrices")
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    rows: list[dict[str, Any]] = []
    for spec in resolved["subproblems"]:
        try:
            subproblem = load_sweep_subproblem(spec, seed=int(resolved["seed"]))
            row = evaluate_block_encoding_subproblem(
                subproblem=subproblem,
                matrices_dir=matrices_dir,
                save_unitary_dimension_limit=int(resolved["save_unitary_dimension_limit"]),
                gamma_safety_factor=float(resolved["gamma_safety_factor"]),
                tolerance=float(resolved["tolerance"]),
            )
        except Exception as exc:
            row = _failure_row(spec, exc)
        rows.append(row)

    results_frame = pd.DataFrame(rows, columns=BLOCK_RESULTS_COLUMNS)
    results_csv = output_dir / "block_encoding_demo_results.csv"
    metadata_json = output_dir / "block_encoding_demo_metadata.json"
    summary_csv = tables_dir / "table_block_encoding_resource_summary.csv"
    figure_path = figures_dir / "figure_block_encoding_errors.png"
    report_path = reports_dir / "explicit_block_encoding_demo_report.md"

    results_frame.to_csv(results_csv, index=False)
    results_frame.to_csv(summary_csv, index=False)
    _plot_block_errors(results_frame, figure_path)
    report_path.write_text(
        _block_report_markdown(resolved, results_frame, results_csv, summary_csv),
        encoding="utf-8",
    )

    ended_at = utc_timestamp()
    artifacts = {
        "results_csv": str(results_csv),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "figure": str(figure_path),
        "report": str(report_path),
        "matrices_dir": str(matrices_dir),
    }
    write_json(
        metadata_json,
        reproducibility_metadata(
            config=resolved,
            started_at=started_at,
            ended_at=ended_at,
            status="completed",
            command=current_command(),
            artifacts=artifacts,
        ),
    )
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results_frame,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def construct_padded_block_encoding(
    A: np.ndarray,
    *,
    gamma: float | None = None,
    gamma_safety_factor: float = 1.0,
    tolerance: float = 1.0e-10,
) -> PaddedBlockEncoding:
    matrix = np.asarray(A, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("A must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A entries must be finite")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    norm = float(singular_values[0]) if singular_values.size else 0.0
    if gamma_safety_factor < 1.0:
        raise ValueError("gamma_safety_factor must be at least 1")
    gamma_value = (
        max(norm * float(gamma_safety_factor), np.finfo(float).eps)
        if gamma is None
        else float(gamma)
    )
    if gamma_value < norm * (1.0 - 1.0e-12):
        raise ValueError("gamma must be at least ||A||_2")
    A_bar = matrix / gamma_value
    padded_dimension = next_power_of_two(max(matrix.shape))
    A_bar_padded = pad_to_square_power_of_two(A_bar, padded_dimension=padded_dimension)
    U = build_dense_block_encoding(A_bar_padded, tolerance=tolerance)
    return PaddedBlockEncoding(
        A=matrix,
        A_bar=A_bar,
        A_bar_padded=A_bar_padded,
        U=U,
        gamma=gamma_value,
        padded_dimension=padded_dimension,
    )


def pad_to_square_power_of_two(
    matrix: np.ndarray,
    *,
    padded_dimension: int | None = None,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    dimension = (
        next_power_of_two(max(values.shape)) if padded_dimension is None else int(padded_dimension)
    )
    if dimension < max(values.shape):
        raise ValueError("padded_dimension must be at least the larger matrix dimension")
    if not _is_power_of_two(dimension):
        raise ValueError("padded_dimension must be a power of two")
    padded = np.zeros((dimension, dimension), dtype=np.float64)
    padded[: values.shape[0], : values.shape[1]] = values
    return padded


def next_power_of_two(value: int) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError("value must be positive")
    return 1 << (integer - 1).bit_length()


def verify_padded_block_encoding(
    encoding: PaddedBlockEncoding,
) -> dict[str, float]:
    top_left = encoding.U[: encoding.padded_dimension, : encoding.padded_dimension]
    block_delta = top_left - encoding.A_bar_padded
    unitary_delta = encoding.U.conj().T @ encoding.U - np.eye(encoding.U.shape[0])
    spectral_norm_A_bar = float(np.linalg.svd(encoding.A_bar_padded, compute_uv=False)[0])
    return {
        "block_error_frobenius": float(np.linalg.norm(block_delta, ord="fro")),
        "spectral_block_error": float(np.linalg.norm(block_delta, ord=2)),
        "unitarity_error_frobenius": float(np.linalg.norm(unitary_delta, ord="fro")),
        "unitarity_error_spectral": float(np.linalg.norm(unitary_delta, ord=2)),
        "spectral_norm_A_bar": spectral_norm_A_bar,
    }


def evaluate_block_encoding_subproblem(
    *,
    subproblem: SweepSubproblem,
    matrices_dir: Path,
    save_unitary_dimension_limit: int,
    gamma_safety_factor: float,
    tolerance: float,
) -> dict[str, Any]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    metadata = subproblem.metadata
    case_name = str(metadata.get("case_name", "unknown"))
    size = int(metadata.get("subproblem_size", min(H.shape)))
    selection = str(metadata.get("selection_mode", "unknown"))
    label = _safe_label(case_name, size, selection)
    singular_values = np.linalg.svd(H, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else np.inf
    encoding = construct_padded_block_encoding(
        H,
        gamma_safety_factor=gamma_safety_factor,
        tolerance=tolerance,
    )
    verification = verify_padded_block_encoding(encoding)
    paths = _save_matrices(
        encoding=encoding,
        matrices_dir=matrices_dir,
        label=label,
        save_unitary_dimension_limit=save_unitary_dimension_limit,
    )
    system_qubits = int(np.ceil(np.log2(encoding.padded_dimension)))
    total_qubits = system_qubits + 1
    dense_dimension = int(encoding.U.shape[0])
    return {
        "case_name": case_name,
        "subproblem_size": size,
        "selection_criterion": selection,
        "original_matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "padded_matrix_shape": f"{encoding.padded_dimension}x{encoding.padded_dimension}",
        "weighted_status": "weighted_jacobian_R_minus_half_H",
        "gamma": encoding.gamma,
        "spectral_norm_A": float(singular_values[0]) if singular_values.size else 0.0,
        "frobenius_norm_A": float(np.linalg.norm(H, ord="fro")),
        "condition_number_A": condition,
        "spectral_norm_A_bar": verification["spectral_norm_A_bar"],
        "block_error_frobenius": verification["block_error_frobenius"],
        "spectral_block_error": verification["spectral_block_error"],
        "unitarity_error_frobenius": verification["unitarity_error_frobenius"],
        "unitarity_error_spectral": verification["unitarity_error_spectral"],
        "padded_dimension": encoding.padded_dimension,
        "system_qubits": system_qubits,
        "block_encoding_ancilla_qubits": 1,
        "total_qubits": total_qubits,
        "dense_unitary_dimension": dense_dimension,
        "dense_unitary_parameter_count": dense_dimension**2,
        "estimated_dense_decomposition_cost": f"O(4^{total_qubits}) generic dense unitary",
        "gate_decomposition_produced": False,
        "gate_decomposition_status": "skipped_dense_decomposition_not_forced",
        "matrix_original_path": paths["original"],
        "matrix_normalized_path": paths["normalized"],
        "matrix_padded_path": paths["padded"],
        "unitary_path": paths["unitary"],
        "run_status": "completed",
        "failure_mode": "",
        "failure_reason": "",
        "claim_boundary_note": (
            "Explicit dense selected-subproblem block encoding; not a scalable sparse oracle."
        ),
    }


def _save_matrices(
    *,
    encoding: PaddedBlockEncoding,
    matrices_dir: Path,
    label: str,
    save_unitary_dimension_limit: int,
) -> dict[str, str]:
    original_path = matrices_dir / f"{label}_A.npy"
    normalized_path = matrices_dir / f"{label}_A_bar.npy"
    padded_path = matrices_dir / f"{label}_A_bar_padded.npy"
    unitary_path = matrices_dir / f"{label}_U_A.npy"
    np.save(original_path, encoding.A)
    np.save(normalized_path, encoding.A_bar)
    np.save(padded_path, encoding.A_bar_padded)
    unitary_value = ""
    if encoding.U.shape[0] <= int(save_unitary_dimension_limit):
        np.save(unitary_path, encoding.U)
        unitary_value = str(unitary_path)
    return {
        "original": str(original_path),
        "normalized": str(normalized_path),
        "padded": str(padded_path),
        "unitary": unitary_value,
    }


def _failure_row(spec: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row = {key: np.nan for key in BLOCK_RESULTS_COLUMNS}
    row.update(
        {
            "case_name": str(spec.get("case_name", "unknown")),
            "subproblem_size": int(spec.get("subproblem_size", 0)),
            "selection_criterion": str(spec.get("selection_mode", "unknown")),
            "weighted_status": "weighted_jacobian_R_minus_half_H",
            "gate_decomposition_produced": False,
            "run_status": "failed",
            "failure_mode": "block_encoding_construction_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "claim_boundary_note": (
                "Failure recorded explicitly; no dense block-encoding claim for this row."
            ),
        }
    )
    return row


def _plot_block_errors(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    if completed.empty:
        ax.text(0.5, 0.5, "No completed block encodings", ha="center", va="center")
    else:
        labels = [
            f"{row.case_name}-{int(row.subproblem_size)}"
            for row in completed.itertuples(index=False)
        ]
        x = np.arange(len(completed))
        width = 0.36
        block_errors = completed["block_error_frobenius"].astype(float).to_numpy()
        unitary_errors = completed["unitarity_error_frobenius"].astype(float).to_numpy()
        ax.bar(x - width / 2, block_errors, width=width, label="block error")
        ax.bar(x + width / 2, unitary_errors, width=width, label="unitarity error")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("error (Frobenius norm)")
        ax.set_title("Explicit Dense Block-Encoding Verification Errors")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _block_report_markdown(
    config: dict[str, Any],
    results: pd.DataFrame,
    results_csv: Path,
    summary_csv: Path,
) -> str:
    status_counts = results["run_status"].value_counts().to_dict() if not results.empty else {}
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    if completed.empty:
        resource_lines = ["- No completed block encodings."]
    else:
        resource_lines = [
            "- Completed dense block encodings: " + str(len(completed)),
            f"- Maximum block Frobenius error: {completed['block_error_frobenius'].max():.3e}",
            "- Maximum unitarity Frobenius error: "
            f"{completed['unitarity_error_frobenius'].max():.3e}",
            "- Total qubit range: "
            f"{int(completed['total_qubits'].min())} to {int(completed['total_qubits'].max())}",
        ]
    subproblem_lines = [
        "- "
        f"{spec.get('case_name')}, size={spec.get('subproblem_size')}, "
        f"selection={spec.get('selection_mode', 'high_leverage')}"
        for spec in config["subproblems"]
    ]
    return "\n".join(
        [
            "# Explicit Block-Encoding Demo Report",
            "",
            "## Command Used",
            "",
            f"`{current_command()}`",
            "",
            "## Construction Formula",
            "",
            "For a normalized contraction A_bar, the dense unitary dilation is "
            "U = [[A_bar, sqrt(I - A_bar A_bar^dagger)], "
            "[sqrt(I - A_bar^dagger A_bar), -A_bar^dagger]].",
            "",
            "## Padding Convention",
            "",
            "Each selected block is zero-padded in the lower/right entries to a "
            "square power-of-two dimension before constructing the dilation.",
            "",
            "## Normalization Convention",
            "",
            "gamma is chosen as gamma_safety_factor * ||A||_2 with "
            "gamma_safety_factor >= 1, and A_bar = A / gamma.",
            "",
            "## Verification Tolerances",
            "",
            f"- Numerical tolerance passed to the dense dilation helper: {config['tolerance']}",
            "",
            "## Selected Subproblems",
            "",
            *subproblem_lines,
            "",
            "## Resource Table Summary",
            "",
            *resource_lines,
            "",
            "## Success, Failure, and Skipped Cases",
            "",
            f"- Status counts: {status_counts}",
            "- Gate decomposition is not forced; rows report dense explicit "
            "block-encoding estimates.",
            "",
            "## Claim-Safe Interpretation",
            "",
            "The explicit dense block encoding verifies that selected IEEE-derived "
            "weighted-Jacobian blocks can be embedded into a unitary model suitable "
            "for QSVT-style validation. This construction is not claimed to be a "
            "scalable oracle for full IEEE-scale PSSE.",
            "",
            "## Limitations",
            "",
            "- Dense dilation is a selected-subproblem proof of concept.",
            "- Sparse access oracles, state preparation, and full-scale readout are "
            "not constructed.",
            "- Generic dense unitary decomposition costs are reported as estimates, "
            "not transpiled circuits.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{results_csv}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _safe_label(case_name: str, size: int, selection: str) -> str:
    raw = f"{case_name}_{size}x{size}_{selection}"
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_" for character in raw
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_root": str(OUTPUT_ROOT),
        "seed": 123,
        "subproblems": DEFAULT_SUBPROBLEMS,
        "gamma_safety_factor": 1.0,
        "tolerance": 1.0e-10,
        "save_unitary_dimension_limit": 64,
    }
    if config:
        resolved.update(config)
    resolved["subproblems"] = [dict(value) for value in resolved["subproblems"]]
    if float(resolved["gamma_safety_factor"]) < 1.0:
        raise ValueError("gamma_safety_factor must be at least 1")
    if float(resolved["tolerance"]) <= 0.0:
        raise ValueError("tolerance must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TQE explicit block-encoding demo")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = run_explicit_block_encoding_demo({"output_root": args.output_root})
    print(f"TQE explicit block-encoding demo complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
