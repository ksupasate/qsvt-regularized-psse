"""Work Package A: tight normalization audit.

For each IEEE case compute the spectral norm (sigma_max = beta), the beta actually used
(block_encoding.spectral_norm_bound with safety_factor=1.0 => beta = sigma_max), Frobenius,
max row/col norms, row-sum (inf-norm) and col-sum (1-norm) bounds, sparse-access
normalization, and rho_beta = beta_used / sigma_max. Then verify beta cannot be reduced
below sigma_max (block_encoding.normalize_for_block_encoding rejects it) and that the
physical Ridge solution is invariant to beta (beta only sets the QSVT contraction, not the
estimator).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.block_encoding import (  # noqa: E402
    normalize_for_block_encoding,
    spectral_norm_bound,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system  # noqa: E402
from robust_qsvt_se.qsvt.filters import ridge_filter  # noqa: E402

OUT = ROOT / "outputs" / "final_qsvt_feasibility_push"
CASES = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]


def ridge_solve(H, r, alpha):
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt.T @ (ridge_filter(s, alpha=alpha) * (U.T @ r))


def main() -> None:
    rows = []
    equiv = []
    for case in CASES:
        try:
            sys_, _src = build_engineering_system(
                {"case_name": case, "case_source": "pypower", "seed": 123}
            )
        except Exception as exc:
            rows.append({"case": case, "status": f"load_failed:{str(exc)[:60]}"})
            continue
        H = np.asarray(sys_.H_tilde, dtype=np.float64)
        sv = np.linalg.svd(H, compute_uv=False)
        sigma_max = float(sv.max())
        sigma_min = float(sv.min())
        beta_used = float(spectral_norm_bound(H, safety_factor=1.0))  # current convention
        fro = float(np.linalg.norm(H, ord="fro"))
        row_norms = np.linalg.norm(H, axis=1)
        col_norms = np.linalg.norm(H, axis=0)
        max_row = float(row_norms.max())
        max_col = float(col_norms.max())
        rowsum = float(np.max(np.sum(np.abs(H), axis=1)))  # inf-norm
        colsum = float(np.max(np.sum(np.abs(H), axis=0)))  # 1-norm
        rho_beta = beta_used / sigma_max
        # attempt to reduce beta below sigma_max -> must be rejected
        try:
            normalize_for_block_encoding(H, beta=sigma_max * (1.0 - 1e-3))
            reducible = "BUG_not_rejected"
        except ValueError:
            reducible = "rejected (beta>=sigma_max enforced)"
        # sparse-access normalization (max column l2 for sparse block encoding scaling proxy)
        sparse_norm = max_col
        rows.append(
            {
                "case": case,
                "shape": f"{H.shape[0]}x{H.shape[1]}",
                "sigma_max": sigma_max,
                "sigma_min": sigma_min,
                "kappa": sigma_max / sigma_min,
                "beta_used": beta_used,
                "safety_factor": 1.0,
                "rho_beta_used_over_sigma_max": rho_beta,
                "frobenius": fro,
                "max_row_norm": max_row,
                "max_col_norm": max_col,
                "rowsum_inf_norm": rowsum,
                "colsum_1_norm": colsum,
                "sparse_access_col_norm": sparse_norm,
                "beta_reducible_below_sigma_max": reducible,
                "status": "ok",
            }
        )
        # estimator invariance: Ridge solution does not depend on beta (beta is a QSVT scaling only)
        r = np.asarray(sys_.r_tilde, dtype=np.float64)
        x1 = ridge_solve(H, r, 1.0e-4)
        x2 = ridge_solve(H / beta_used, r, 1.0e-4 / beta_used**2)  # same estimator, re-normalized
        equiv.append(
            {
                "case": case,
                "alpha": 1.0e-4,
                "lambda_at_beta": 1.0e-4 / beta_used**2,
                "ridge_solution_invariant_to_beta": bool(np.allclose(x1, x2, atol=1e-10)),
                "max_abs_diff": float(np.max(np.abs(x1 - x2))),
            }
        )

    with (OUT / "normalization_audit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "normalization_equivalence_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(equiv[0].keys()))
        w.writeheader()
        w.writerows(equiv)
    print(json.dumps(rows, indent=2))
    print("--- equivalence ---")
    print(json.dumps(equiv, indent=2))


if __name__ == "__main__":
    main()
