from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import spectral_norm_bound
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, singular_summary
from robust_qsvt_se.utils.io import ensure_directory, write_json

DENSE_BLOCK_ENCODING_CAVEAT = (
    "Dense block encodings are validation prototypes, not scalable sparse-access "
    "oracles for full power-system matrices."
)


def build_block_encoding_scalability_report(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = [_case_scalability_row(case, resolved) for case in resolved["cases"]]
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "scalability_summary.csv"
    json_path = output_dir / "scalability_summary.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "scalability_summary_csv": str(csv_path),
            "scalability_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "scalability_summary_csv": csv_path,
            "scalability_summary_json": json_path,
            "manifest": manifest_path,
        },
    }


def _case_scalability_row(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = _case_config(case, config)
    case_name = str(case_config.get("case_name", "unknown"))
    try:
        system, matrix_source = build_engineering_system(case_config)
        H = np.asarray(system.H_tilde, dtype=np.float64)
        summary = singular_summary(H)
        nonzeros = int(np.count_nonzero(np.abs(H) > 1.0e-12))
        density = float(nonzeros / H.size)
        dense_dimension = int(H.shape[0] + H.shape[1])
        return {
            "case_name": case_name,
            "matrix_source": matrix_source,
            "status": "ok",
            "m": int(H.shape[0]),
            "n": int(H.shape[1]),
            "nonzeros": nonzeros,
            "density": density,
            "sparsity": float(1.0 - density),
            "state_dimension": int(H.shape[1]),
            "row_count": int(H.shape[0]),
            "estimated_dense_encoding_dimension": dense_dimension,
            "estimated_index_qubits": int(np.ceil(np.log2(max(dense_dimension, 2)))),
            "beta": spectral_norm_bound(H),
            "kappa": summary["condition_number"],
            "rank": summary["rank"],
            "scalability_caveat": DENSE_BLOCK_ENCODING_CAVEAT,
            "failure_reason_if_any": "",
        }
    except Exception as exc:
        return {
            "case_name": case_name,
            "matrix_source": str(case_config.get("matrix_source", "pypower_ac_weighted_jacobian")),
            "status": "failed",
            "m": None,
            "n": None,
            "nonzeros": None,
            "density": None,
            "sparsity": None,
            "state_dimension": None,
            "row_count": None,
            "estimated_dense_encoding_dimension": None,
            "estimated_index_qubits": None,
            "beta": None,
            "kappa": None,
            "rank": None,
            "scalability_caveat": DENSE_BLOCK_ENCODING_CAVEAT,
            "failure_reason_if_any": str(exc),
        }


def _case_config(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = {
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": config["case_source"],
        "seed": config["seed"],
    }
    if isinstance(case, dict):
        case_config.update(case)
    elif case == "synthetic":
        case_config.update({"matrix_source": "synthetic", "case_name": "synthetic"})
    else:
        case_config.update({"case_name": str(case)})
    return case_config


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_block_encoding_scalability",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
    }
    if config:
        resolved.update(config)
    if not list(resolved["cases"]):
        raise ValueError("cases must be non-empty")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT block-encoding scalability report")
    parser.parse_args(argv)
    run = build_block_encoding_scalability_report()
    print(f"QSVT block-encoding scalability report complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
