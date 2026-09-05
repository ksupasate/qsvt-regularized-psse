from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.sparse_access_oracle import build_sparse_access_oracle
from robust_qsvt_se.utils.io import ensure_directory, write_json

TOY_SPARSE_ORACLE_LIMITATION = (
    "The toy sparse-oracle circuit demonstrates oracle structure on a tiny "
    "instance only. It does not implement the IEEE-scale sparse oracle as a "
    "physical circuit."
)


def build_toy_sparse_oracle_circuit_demo(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_toy_sparse_oracle_circuit",
        "degree": 7,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    matrix = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.1],
            [0.0, -0.4, 0.2, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2, 0.4],
        ],
        dtype=np.float64,
    )
    beta = max(float(np.linalg.svd(matrix, compute_uv=False)[0]), 1.0)
    oracle = build_sparse_access_oracle(matrix, normalization_beta=beta)
    normalized = matrix / beta
    block = canonical_square_block_encoding(normalized)
    block_report = validate_block_encoding(block, beta=beta)
    oracle_rows = []
    for row in range(oracle.shape[0]):
        columns, values = oracle.get_row_entries(row)
        for local_index, (col, value) in enumerate(zip(columns, values, strict=True)):
            oracle_rows.append(
                {
                    "row": row,
                    "local_nonzero_index": int(local_index),
                    "column": int(col),
                    "value": float(value),
                    "oracle_F_mapping": f"|{row},{local_index}> -> |{row},{int(col)}>",
                }
            )
    summary_rows = [
        {
            "matrix_size": "4x4",
            "nnz": oracle.nnz,
            "max_row_sparsity": oracle.max_row_sparsity,
            "max_col_sparsity": oracle.max_col_sparsity,
            "normalization_beta": beta,
            "qsvt_degree": int(resolved["degree"]),
            "qsvt_query_count_proxy": 2 * int(resolved["degree"]) + 1,
            "block_encoding_unitarity_error": block_report["unitarity_error"],
            "top_left_block_error": block_report["top_left_block_error"],
            "limitation": TOY_SPARSE_ORACLE_LIMITATION,
        }
    ]
    summary_csv = output_dir / "toy_sparse_oracle_summary.csv"
    diagnostics_json = output_dir / "toy_sparse_oracle_circuit_diagnostics.json"
    limitations_md = output_dir / "toy_sparse_oracle_limitations.md"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    diagnostics = {
        "oracle_table": oracle_rows,
        "block_encoding_report": block_report,
        "query_counting": {
            "degree": int(resolved["degree"]),
            "qsvt_query_count_proxy": 2 * int(resolved["degree"]) + 1,
            "oracle_calls_are_counted_not_synthesized": True,
        },
        "limitation": TOY_SPARSE_ORACLE_LIMITATION,
    }
    write_json(diagnostics_json, diagnostics)
    limitations_md.write_text(
        "\n".join(
            [
                "# Toy Sparse-Oracle Circuit Limitation",
                "",
                TOY_SPARSE_ORACLE_LIMITATION,
                "",
                "- The oracle table is explicit only because the matrix is 4x4.",
                "- The block encoding is dense and used only as a comparison diagnostic.",
                "- Query counts are QSVT-compatible accounting proxies.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "toy_sparse_oracle_summary": str(summary_csv),
            "toy_sparse_oracle_circuit_diagnostics": str(diagnostics_json),
            "toy_sparse_oracle_limitations": str(limitations_md),
        },
        input_config=resolved,
        claim_boundary=TOY_SPARSE_ORACLE_LIMITATION,
    )
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame(summary_rows),
        "diagnostics": diagnostics,
        "artifacts": {
            "manifest": manifest,
            "toy_sparse_oracle_summary": summary_csv,
            "toy_sparse_oracle_circuit_diagnostics": diagnostics_json,
            "toy_sparse_oracle_limitations": limitations_md,
        },
    }
