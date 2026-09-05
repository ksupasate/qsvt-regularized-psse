from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import (
    build_dense_block_encoding,
    normalize_for_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    build_engineering_system,
    select_rectangular_submatrix,
    singular_summary,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json


def run_block_encoding_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    full_matrix = np.asarray(system.H_tilde, dtype=np.float64)

    rows = [
        _validate_matrix(
            full_matrix,
            case_name=str(system.metadata.get("case_name", "unknown")),
            matrix_source=matrix_source,
            matrix_label="full_weighted_jacobian",
        )
    ]
    submatrix, selected_rows, selected_columns = select_rectangular_submatrix(
        full_matrix,
        row_count=int(resolved["rectangular_shape"][0]),
        column_count=int(resolved["rectangular_shape"][1]),
    )
    rows.append(
        _validate_matrix(
            submatrix,
            case_name=str(system.metadata.get("case_name", "unknown")),
            matrix_source=matrix_source,
            matrix_label="deterministic_rectangular_submatrix",
            selected_rows=selected_rows.tolist(),
            selected_columns=selected_columns.tolist(),
        )
    )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "block_encoding_summary.csv"
    json_path = output_dir / "block_encoding_summary.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "block_encoding_summary_csv": str(csv_path),
            "block_encoding_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "block_encoding_summary_csv": csv_path,
            "block_encoding_summary_json": json_path,
            "manifest": manifest_path,
        },
    }


def _validate_matrix(
    matrix: np.ndarray,
    *,
    case_name: str,
    matrix_source: str,
    matrix_label: str,
    selected_rows: list[int] | None = None,
    selected_columns: list[int] | None = None,
) -> dict[str, Any]:
    original = singular_summary(matrix)
    normalized, beta = normalize_for_block_encoding(matrix)
    unitary = build_dense_block_encoding(normalized)
    validation = validate_block_encoding(normalized, unitary, beta=beta)
    normalized_summary = singular_summary(normalized)
    return {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "matrix_label": matrix_label,
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "beta": beta,
        "spectral_norm_original": original["spectral_norm"],
        "spectral_norm_normalized": normalized_summary["spectral_norm"],
        "encoded_block_error": validation["encoded_block_error"],
        "unitarity_error": validation["unitarity_error"],
        "rank": original["rank"],
        "condition_number_before_normalization": original["condition_number"],
        "passed": bool(validation["passed"]),
        "selected_rows": "" if selected_rows is None else ",".join(map(str, selected_rows)),
        "selected_columns": (
            "" if selected_columns is None else ",".join(map(str, selected_columns))
        ),
        "scope_note": (
            "Dense Julia block-encoding validation for a normalized weighted matrix; "
            "not a scalable hardware-native oracle decomposition."
        ),
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_block_encoding",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "rectangular_shape": [6, 4],
    }
    if config:
        resolved.update(config)
    rectangular_shape = list(resolved["rectangular_shape"])
    if len(rectangular_shape) != 2 or any(int(value) <= 0 for value in rectangular_shape):
        raise ValueError("rectangular_shape must contain two positive integers")
    resolved["rectangular_shape"] = [int(rectangular_shape[0]), int(rectangular_shape[1])]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dense QSVT block-encoding validation")
    parser.parse_args(argv)
    run = run_block_encoding_demo()
    print(f"QSVT block-encoding demo complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
