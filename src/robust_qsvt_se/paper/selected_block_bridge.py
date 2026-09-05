"""Bridge audit between full-system and selected-submatrix Ridge functionals.

The executed QSVT workloads use square submatrices selected from generated
weighted PSSE Jacobians.  This audit quantifies, without claiming equivalence,
how the first local coordinate and the full selected-coordinate vector differ
from the matched-alpha Ridge update of the original rectangular system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import array_checksum
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_COLUMNS = [
    "workload_id",
    "case_name",
    "full_shape",
    "block_shape",
    "selected_rows",
    "selected_cols",
    "alpha",
    "lambda_alpha_over_block_beta2",
    "full_selected_functional",
    "block_selected_functional",
    "absolute_discrepancy_delta_l",
    "relative_discrepancy_vs_full",
    "selected_coordinate_vector_relative_discrepancy",
    "full_matrix_checksum",
    "block_checksum",
    "interpretation",
]


def build_bridge_rows(seed: int = 123) -> list[dict[str, Any]]:
    anchor_system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    anchor_block, _, _, _ = select_deterministic_block(
        np.asarray(anchor_system.H_tilde),
        np.asarray(anchor_system.r_tilde),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    anchor_singular = np.linalg.svd(anchor_block, compute_uv=False)
    anchor_lambda = 4.0 * float(anchor_singular[-1]) ** 2 / float(anchor_singular[0]) ** 2
    specs = [
        ("ieee14_4x4_primary_anchor", "ieee14", 4, anchor_lambda),
        ("ieee14_8x8_lambda_matched_anchor", "ieee14", 8, anchor_lambda),
        ("ieee30_16x16_raw", "ieee30", 16, 0.069),
    ]
    rows: list[dict[str, Any]] = []
    for workload_id, case_name, size, lam in specs:
        system, _ = build_engineering_system(
            {
                "case_name": case_name,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        full = np.asarray(system.H_tilde, dtype=np.float64)
        residual = np.asarray(system.r_tilde, dtype=np.float64)
        block, block_residual, selected_rows, selected_cols = select_deterministic_block(
            full,
            residual,
            row_count=size,
            col_count=size,
            policy="largest_row_col_norms",
        )
        beta = float(np.linalg.svd(block, compute_uv=False)[0])
        alpha = float(lam) * beta**2
        full_update = ridge_svd_solution(full, residual, alpha=alpha)
        block_update = ridge_svd_solution(block, block_residual, alpha=alpha)
        full_value = float(full_update[int(selected_cols[0])])
        block_value = float(block_update[0])
        absolute = abs(full_value - block_value)
        relative = absolute / max(abs(full_value), np.finfo(float).tiny)
        vector_relative = float(
            np.linalg.norm(full_update[selected_cols] - block_update)
            / max(np.linalg.norm(full_update[selected_cols]), np.finfo(float).tiny)
        )
        rows.append(
            {
                "workload_id": workload_id,
                "case_name": case_name,
                "full_shape": f"{full.shape[0]}x{full.shape[1]}",
                "block_shape": f"{size}x{size}",
                "selected_rows": " ".join(str(int(v)) for v in selected_rows),
                "selected_cols": " ".join(str(int(v)) for v in selected_cols),
                "alpha": alpha,
                "lambda_alpha_over_block_beta2": float(lam),
                "full_selected_functional": full_value,
                "block_selected_functional": block_value,
                "absolute_discrepancy_delta_l": absolute,
                "relative_discrepancy_vs_full": relative,
                "selected_coordinate_vector_relative_discrepancy": vector_relative,
                "full_matrix_checksum": array_checksum(full),
                "block_checksum": array_checksum(block),
                "interpretation": (
                    "selected-submatrix surrogate; not the selected functional of the "
                    "full-system Ridge update"
                ),
            }
        )
    return rows


def run_selected_block_bridge(config: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(config or {})
    output_dir = ensure_directory(Path(options.get("output_dir", "outputs/selected_block_bridge")))
    seed = int(options.get("seed", 123))
    rows = build_bridge_rows(seed)
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    csv_path = output_dir / "selected_block_full_system_bridge.csv"
    json_path = output_dir / "selected_block_full_system_bridge.json"
    tex_path = Path(options.get("table_path", "manuscript/tables/selected_block_bridge.tex"))
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(_table_tex(frame), encoding="utf-8")
    manifest = {
        "artifact_name": "selected_block_full_system_bridge",
        "seed_provenance": {"status": "recorded", "seeds": {"system_seed": seed}},
        "regeneration_command": ".venv/bin/python scripts/run_selected_block_bridge.py",
        "outputs": [str(csv_path), str(json_path), str(tex_path)],
        "claim_boundary": (
            "The executed circuits are selected-submatrix QSVT boundary tests derived from "
            "PSSE Jacobians, not direct selected-output evaluations of the full PSSE update."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"frame": frame, "artifacts": [csv_path, json_path, tex_path, manifest_path]}


def _sci(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value)))) if value else 0
    mantissa = value / (10.0**exponent) if value else 0.0
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def _table_tex(frame: pd.DataFrame) -> str:
    lines = [
        "% Source: outputs/selected_block_bridge/selected_block_full_system_bridge.csv",
        "% Regenerate: .venv/bin/python scripts/run_selected_block_bridge.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Bridge audit between each raw executed selected submatrix and the source "
        r"full weighted PSSE system at the same $\alpha$. Here $\ell_B=e_1$ and $\ell$ selects "
        r"the corresponding full-system state coordinate. $\Delta_\ell=|\ell^T\Delta "
        r"x_{\rm full}-\ell_B^T\Delta x_B|$. The nonzero discrepancies show that the "
        r"selected-submatrix circuits are surrogate boundary tests, not direct evaluations "
        r"of selected outputs of the full-system Ridge update.}",
        r"\label{tab:selected_block_bridge}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Workload & Full / block shape & $\alpha$ & $\ell^T\Delta x_{\rm full}$ & "
        r"$\ell_B^T\Delta x_B$ & $\Delta_\ell$ & rel. $\Delta_\ell$ / vector rel. \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        label = str(row["workload_id"]).replace("_", r"\_")
        lines.append(
            f"{label} & {row['full_shape']} / {row['block_shape']} & {_sci(float(row['alpha']))} & "
            f"{_sci(float(row['full_selected_functional']))} & "
            f"{_sci(float(row['block_selected_functional']))} & "
            f"{_sci(float(row['absolute_discrepancy_delta_l']))} & "
            f"{float(row['relative_discrepancy_vs_full']):.3f} / "
            f"{float(row['selected_coordinate_vector_relative_discrepancy']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)
