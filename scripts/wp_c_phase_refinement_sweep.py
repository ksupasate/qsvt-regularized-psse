"""WP C refinement: sweep degrees 191-255 x {sym_qsp, laurent} x eps for IEEE-14 alpha=1e-4
to find the reliable pyqsp convergence frontier and the smallest update error vs Ridge.
Also records pyqsp convergence status (max-iteration / failure) honestly.
"""

from __future__ import annotations

import csv
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import probe_high_degree_lambda_feasibility as P  # noqa: E402
from pyqsp.angle_sequence import QuantumSignalProcessingPhases  # noqa: E402
from pyqsp.response import ComputeQSPResponse  # noqa: E402

from robust_qsvt_se.qsvt.engineering_utils import (  # noqa: E402
    build_engineering_system,
    ridge_svd_solution,
)

OUT = ROOT / "outputs" / "final_qsvt_feasibility_push"


def pyqsp_try(coeffs, method, eps, tolerance):
    try:
        buf = StringIO()
        with redirect_stdout(buf):
            res = QuantumSignalProcessingPhases(
                np.asarray(coeffs, dtype=np.float64),
                method=method,
                chebyshev_basis=(method == "sym_qsp"),
                eps=eps,
                tolerance=tolerance,
            )
        phases = np.asarray(res[0], dtype=np.float64)
        log = buf.getvalue()
        max_iter = "Max iteration" in log
        return phases, ("converged" if not max_iter else "max_iter_reached"), log
    except Exception as exc:
        return None, "failed", str(exc)[:100]


def main() -> None:
    sys_, _ = build_engineering_system(
        {"case_name": "ieee14", "case_source": "pypower", "seed": 123}
    )
    H = np.asarray(sys_.H_tilde, dtype=np.float64)
    r = np.asarray(sys_.r_tilde, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(H, full_matrices=False)
    beta = float(sv.max())
    s_min = float(sv.min() / beta)
    alpha = 1.0e-4
    lam = alpha / beta**2
    g = lambda s: s / (s**2 + lam)  # noqa: E731
    dx_exact = ridge_svd_solution(H, r, alpha=float(alpha))
    s_norm = sv / beta
    rows = []
    for degree in [191, 207, 223, 239, 255]:
        mm = P.odd_minimax_cheb_coeffs(g, s_min, degree)
        if mm is None:
            rows.append({"degree": degree, "method": "minimax", "status": "minimax_failed"})
            continue
        coeffs, _ = mm
        cg = float(np.max(np.abs(chebval(np.linspace(-1, 1, 8193), coeffs))))
        qspc = coeffs / cg
        for method in ["sym_qsp", "laurent"]:
            for eps in [1e-4, 1e-3]:
                phases, status, _log = pyqsp_try(qspc, method, eps, 1e-8)
                if phases is None:
                    rows.append({"degree": degree, "method": method, "eps": eps, "status": status})
                    continue
                p_real = (
                    np.imag(
                        np.asarray(
                            ComputeQSPResponse(
                                s_norm, phases, signal_operator="Wx", measurement="x", sym_qsp=True
                            )["pdat"]
                        )
                    )
                    if method == "sym_qsp"
                    else np.real(
                        np.asarray(
                            ComputeQSPResponse(
                                s_norm, phases, signal_operator="Wz", measurement="z"
                            )["pdat"]
                        )
                    )
                )
                recovered = (cg / beta) * (Vt.T @ (p_real * (U.T @ r)))
                rel = float(np.linalg.norm(recovered - dx_exact) / np.linalg.norm(dx_exact))
                occ = float(np.max(np.abs(p_real - g(s_norm) / cg)))
                rows.append(
                    {
                        "degree": degree,
                        "method": method,
                        "eps": eps,
                        "status": status,
                        "C_global": cg,
                        "phase_realized_occ_err": occ,
                        "update_rel_err_vs_ridge": rel,
                        "n_phases": int(phases.size),
                        "passes_1e-2": bool(rel <= 1e-2 and occ <= 1e-2),
                        "passes_1e-3": bool(rel <= 1e-3 and occ <= 1e-3),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    with (OUT / "phase_synthesis_refinement_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE phase_synthesis_refinement_sweep.csv", flush=True)


if __name__ == "__main__":
    main()
