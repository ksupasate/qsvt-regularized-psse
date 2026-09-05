"""WP C matrix-action validation: apply the PHASE-REALIZED QSVT polynomial to the
full-rectangular system via its SVD and recover the Ridge update.

This tests the full chain  phases -> ComputeQSPResponse -> p(s_i) -> p(A)|r> -> physical
recovery (C/beta) -> comparison to exact Ridge solve. It is NOT a polynomial-only result:
p(s_i) is read from the synthesized phases (imag(Wx/x/sym_qsp)), so a correct update
recovery proves the synthesized phases drive a QSVT action that realizes the Ridge filter.

Sanity anchor: IEEE-14 lambda~0.068, degree 31 (manuscript update_err ~7.09e-4).
Target: IEEE-14 alpha=1e-4 (benchmark), degrees 127/191/255.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import probe_high_degree_lambda_feasibility as P  # noqa: E402
from pyqsp.response import ComputeQSPResponse  # noqa: E402

from robust_qsvt_se.qsvt.engineering_utils import (  # noqa: E402
    build_engineering_system,
    ridge_svd_solution,
)

OUT = ROOT / "outputs" / "final_qsvt_feasibility_push"


def qsvt_matrix_action(case: str, alpha: float, degree: int) -> dict:
    sys_, _src = build_engineering_system(
        {"case_name": case, "case_source": "pypower", "seed": 123}
    )
    H = np.asarray(sys_.H_tilde, dtype=np.float64)
    r = np.asarray(sys_.r_tilde, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(H, full_matrices=False)
    beta = float(sv.max())
    s_min = float(sv.min() / beta)
    lam = alpha / beta**2
    g = lambda s, lam=lam: s / (s**2 + lam)  # noqa: E731
    mm = P.odd_minimax_cheb_coeffs(g, s_min, degree)
    if mm is None:
        return {"case": case, "alpha": alpha, "degree": degree, "status": "minimax_failed"}
    coeffs, _t = mm
    fg = np.linspace(-1, 1, 8193)
    cg = float(np.max(np.abs(chebval(fg, coeffs))))
    qspc = coeffs / cg
    phases, _recon, st, par, err = P.pyqsp_synthesize_and_reconstruct(qspc, np.array(sv / beta))
    if st != "ok":
        return {
            "case": case,
            "alpha": alpha,
            "degree": degree,
            "status": "phase_failed",
            "err": str(err)[:120],
        }
    s_norm = np.asarray(sv / beta, dtype=np.float64)
    p_real = np.imag(
        np.asarray(
            ComputeQSPResponse(s_norm, phases, signal_operator="Wx", measurement="x", sym_qsp=True)[
                "pdat"
            ]
        )
    )
    dx_exact = ridge_svd_solution(H, r, alpha=float(alpha))
    recovered = (cg / beta) * (Vt.T @ (p_real * (U.T @ r)))
    rel = float(np.linalg.norm(recovered - dx_exact) / np.linalg.norm(dx_exact))
    rr = float(np.dot(r, r))
    psucc = float(np.sum((p_real * (U.T @ r)) ** 2) / (cg**2 * rr)) if rr > 0 else float("nan")
    return {
        "case": case,
        "alpha": alpha,
        "degree": degree,
        "status": "ok",
        "lambda": lam,
        "C_global": cg,
        "s_min": s_min,
        "kappa": float(sv.max() / sv.min()),
        "phase_realized_occ_err": float(np.max(np.abs(p_real - g(s_norm) / cg))),
        "update_rel_err_vs_ridge": rel,
        "postselection_proxy": psucc,
        "n_phases": int(phases.size),
        "parity": par,
    }


def main() -> None:
    rows = []
    print("SANITY ieee14 lambda~0.068 deg31 (expect ~7e-4):", flush=True)
    r0 = qsvt_matrix_action("ieee14", 522731.33058420924, 31)
    print("  " + json.dumps(r0), flush=True)
    rows.append(r0)
    print("TARGET ieee14 alpha=1e-4 (benchmark):", flush=True)
    for d in [127, 191, 255]:
        r = qsvt_matrix_action("ieee14", 1.0e-4, d)
        print(f"  deg{d}: " + json.dumps(r), flush=True)
        rows.append(r)
    print("EXTEND ieee30 alpha=1e-4 (needs >255):", flush=True)
    for d in [255, 511]:
        r = qsvt_matrix_action("ieee30", 1.0e-4, d)
        print(f"  deg{d}: " + json.dumps(r), flush=True)
        rows.append(r)
    import csv

    with (OUT / "phase_reconstruction_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("WROTE outputs/final_qsvt_feasibility_push/phase_reconstruction_checks.csv", flush=True)


if __name__ == "__main__":
    main()
