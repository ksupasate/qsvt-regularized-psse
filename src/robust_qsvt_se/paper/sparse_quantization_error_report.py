"""Sparse value-oracle quantization error: normalized and output-level propagation.

Table VIII (``revision_sparse_oracle_validation``) shows that the compiled O_val
lookup circuit is *logically* exact for its six-bit sign-magnitude representation
and reports one *absolute* error relative to the unquantized block. That table
alone does not say whether a six-bit representation is numerically adequate. This
module adds the normalized/relative errors a reviewer needs to judge that, and
propagates the same quantization through the matched-alpha Ridge/QSVT update
(eq. ``selected_functional``: ``Delta x_alpha = (H^T H + alpha I)^-1 H^T r``) to
one selected output, so the report separates *logic correctness* (already exact)
from *representation error* (this report).

Only the matrix entries loaded through ``O_val`` are quantized here, matching the
sparse-access study: the residual state uses a separate, unquantized preparation
path, so ``r`` is unchanged. This audit re-derives the same deterministic block
and the same 6-bit sign-magnitude quantization used by
``tqe_revision_sparse_oracle_demo`` (same scale convention:
``scale = max(|H_ij|)``) so its ``max_absolute_quantization_error`` reproduces the
existing Table VIII number as a self-consistency check; it does not change any
existing artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.reversible_sparse_oracle import decode_value, quantize_value
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    EXPERIMENTS_CLAIM_BOUNDARY,
    assert_safe,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/sparse_quantization_error_report")
VALUE_BITS = 6  # matches the compiled O_val circuits validated in Table VIII
MATCHED_ALPHA = 1.0e-4  # benchmark Ridge default (Section VI-C); not tuned per block
DEFAULT_OUTPUT_INDEX = 0  # first selected-block state coordinate, y_l = e_1^T Delta x_alpha

REPORT_COLUMNS = [
    "block",
    "matrix_shape",
    "value_bits",
    "matrix_norm_frobenius",
    "matrix_norm_max_abs_entry",
    "max_absolute_quantization_error",
    "relative_frobenius_error",
    "relative_spectral_error",
    "alpha_used",
    "selected_output_index",
    "true_selected_output",
    "quantized_selected_output",
    "selected_output_error_abs_delta_y",
    "relative_selected_output_error",
    "update_relative_error_l2",
    "interpretation",
]


def _quantize_matrix(matrix: np.ndarray, value_bits: int) -> tuple[np.ndarray, float, float]:
    """Six-bit sign-magnitude quantization, same scale convention as build_value_oracle."""

    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).eps)
    quantized = np.zeros_like(matrix, dtype=np.float64)
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            sign_bit, magnitude = quantize_value(float(matrix[i, j]), scale, value_bits)
            quantized[i, j] = decode_value(sign_bit, magnitude, scale, value_bits)
    levels = (1 << value_bits) - 1
    step = scale / levels if levels > 0 else 0.0
    return quantized, scale, step


def _ridge_update(matrix: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    """Delta x_alpha = (H^T H + alpha I)^-1 H^T r, matching eq. selected_functional."""

    n = matrix.shape[1]
    gram = matrix.T @ matrix + alpha * np.eye(n)
    return np.linalg.solve(gram, matrix.T @ residual)


def _block_for(case: str, seed: int, size: int) -> tuple[np.ndarray, np.ndarray, str]:
    system, _ = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    block, residual, _rows, _cols = select_deterministic_block(
        H_full, r_full, row_count=int(size), col_count=int(size), policy="largest_row_col_norms"
    )
    label = f"{case}_{size}x{size}_weighted_jacobian_block"
    return np.asarray(block, dtype=np.float64), np.asarray(residual, dtype=np.float64), label


def _interpretation(block_label: str) -> str:
    return (
        "The lookup circuit is logically exact for its six-bit representation "
        "(Table VIII); this row separates that logic correctness from the numerical "
        "representation error, reported both as a matrix-level relative error and "
        f"as its propagated effect on one matched-alpha selected output for {block_label}."
    )


def build_sparse_quantization_error_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "case": "ieee14",
        "seed": 123,
        "sizes": [4, 8],
        "value_bits": VALUE_BITS,
        "alpha": MATCHED_ALPHA,
        "selected_output_index": DEFAULT_OUTPUT_INDEX,
        "command": "run_sparse_quantization_error_report",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    value_bits = int(resolved["value_bits"])
    alpha = float(resolved["alpha"])
    output_index = int(resolved["selected_output_index"])

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for size in resolved["sizes"]:
        size = int(size)
        matrix, residual, label = _block_for(resolved["case"], int(resolved["seed"]), size)
        quantized, scale, _step = _quantize_matrix(matrix, value_bits)

        diff = quantized - matrix
        frob_norm_true = float(np.linalg.norm(matrix))
        frob_error = float(np.linalg.norm(diff))
        spectral_norm_true = float(np.linalg.svd(matrix, compute_uv=False)[0])
        spectral_error = float(np.linalg.svd(diff, compute_uv=False)[0])
        max_abs_error = float(np.max(np.abs(diff)))

        if output_index >= size:
            failures.append(
                {
                    "block": label,
                    "issue": (
                        f"selected_output_index {output_index} >= block size {size}; "
                        "output propagation skipped for this block"
                    ),
                }
            )
            continue

        delta_true = _ridge_update(matrix, residual, alpha)
        delta_quant = _ridge_update(quantized, residual, alpha)
        y_true = float(delta_true[output_index])
        y_quant = float(delta_quant[output_index])
        abs_delta_y = abs(y_quant - y_true)
        update_norm_true = float(np.linalg.norm(delta_true))

        rows.append(
            {
                "block": label,
                "matrix_shape": f"{size}x{size}",
                "value_bits": value_bits,
                "matrix_norm_frobenius": frob_norm_true,
                "matrix_norm_max_abs_entry": scale,
                "max_absolute_quantization_error": max_abs_error,
                "relative_frobenius_error": (
                    frob_error / frob_norm_true if frob_norm_true > 0 else float("nan")
                ),
                "relative_spectral_error": (
                    spectral_error / spectral_norm_true if spectral_norm_true > 0 else float("nan")
                ),
                "alpha_used": alpha,
                "selected_output_index": output_index,
                "true_selected_output": y_true,
                "quantized_selected_output": y_quant,
                "selected_output_error_abs_delta_y": abs_delta_y,
                "relative_selected_output_error": (
                    abs_delta_y / abs(y_true) if y_true != 0 else float("nan")
                ),
                "update_relative_error_l2": (
                    float(np.linalg.norm(delta_quant - delta_true) / update_norm_true)
                    if update_norm_true > 0
                    else float("nan")
                ),
                "interpretation": _interpretation(label),
            }
        )

    frame = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    artifacts = _write_outputs(output_dir, frame)
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="F_sparse_quantization_error_report",
        script_name="scripts/run_sparse_quantization_error_report.py",
        command=str(resolved["command"]),
        description=(
            "Normalized (relative Frobenius / spectral) sparse value-oracle quantization "
            "error and its propagation through the matched-alpha Ridge/QSVT update to one "
            "selected output, for the same IEEE-14-derived blocks validated in Table VIII."
        ),
        artifacts=artifacts,
        inputs_used=[f"build_engineering_system:{resolved['case']}:weighted_jacobian"],
        random_seeds={"demo_system_seed": int(resolved["seed"])},
        warnings=[
            "only the O_val-loaded matrix entries are quantized; the residual state "
            "preparation path is unquantized in this study",
        ],
        failures=failures,
        interpretation_boundary=(
            "This report separates logic correctness (bit-exact for the six-bit "
            "representation, Table VIII) from representation error (this report): "
            "relative Frobenius/spectral error bound the matrix-level effect, and the "
            "propagated selected-output error bounds the effect actually seen by a "
            "matched-alpha Ridge/QSVT consumer of this block. It does not claim the "
            "sparse-access oracle circuits are inexact; it quantifies the numerical cost "
            "of the fixed six-bit resolution they implement."
        ),
        extra={"claim_boundary": EXPERIMENTS_CLAIM_BOUNDARY},
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "report": frame, "artifacts": artifacts, "failures": failures}


def _summary_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Sparse Value-Oracle Quantization Error: Normalized and Propagated",
        "",
        EXPERIMENTS_CLAIM_BOUNDARY,
        "",
        "Only the matrix entries loaded through `O_val` are quantized (six-bit "
        "sign-magnitude, `scale = max(|H_ij|)`, matching the compiled circuits behind "
        "Table VIII); the residual state preparation path is unquantized in this study.",
        "",
        "| Block | shape | rel. Frobenius err. | rel. spectral err. | "
        "|dy_l| | rel. |dy_l| | update rel. err. (L2) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| `{row['block']}` | {row['matrix_shape']} | "
            f"{row['relative_frobenius_error']:.3e} | {row['relative_spectral_error']:.3e} | "
            f"{row['selected_output_error_abs_delta_y']:.3e} | "
            f"{row['relative_selected_output_error']:.3e} | "
            f"{row['update_relative_error_l2']:.3e} |"
        )
    lines += [
        "",
        "`y_l = e_1^T Delta x_alpha` is the first selected-block state coordinate of the "
        "matched-alpha Ridge/QSVT update `Delta x_alpha = (H^T H + alpha I)^-1 H^T r`, "
        f"evaluated at the benchmark `alpha = {MATCHED_ALPHA:g}`.",
        "",
        "## Interpretation",
        "",
        "The lookup circuit is logically exact for its six-bit representation (Table VIII); "
        "this report separates that logic correctness from the numerical representation "
        "error, so a reviewer can judge whether six-bit resolution is numerically adequate "
        "for a given selected output rather than only knowing the reconstruction is "
        "bit-exact for its own quantized reference.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _write_outputs(output_dir: Path, frame: pd.DataFrame) -> dict[str, Path]:
    csv_path = output_dir / "quantization_error_report.csv"
    frame.to_csv(csv_path, index=False)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(_summary_markdown(frame), encoding="utf-8")
    return {"quantization_error_report_csv": csv_path, "summary_md": summary_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Normalized and output-propagated sparse value-oracle quantization error."
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--value-bits", type=int, default=VALUE_BITS)
    parser.add_argument("--alpha", type=float, default=MATCHED_ALPHA)
    parser.add_argument("--selected-output-index", type=int, default=DEFAULT_OUTPUT_INDEX)
    args = parser.parse_args(argv)
    run = build_sparse_quantization_error_report(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "seed": args.seed,
            "sizes": args.sizes,
            "value_bits": args.value_bits,
            "alpha": args.alpha,
            "selected_output_index": args.selected_output_index,
            "command": "scripts/run_sparse_quantization_error_report.py " + " ".join(argv or []),
        }
    )
    csv_path = run["artifacts"]["quantization_error_report_csv"]
    print(f"Sparse quantization error report complete: {csv_path}")
    print(run["report"].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
