"""IEEE application generalization harness (WP-F/G/H/I).

Tests whether the *useful-overlap* (application usefulness AND executable QSVT
feasibility with the validated convention) generalizes beyond the single
IEEE-14 headline configuration:

  WP-F  IEEE-14, three pre-selected outputs (theta_2, V_1, area-aggregate angle)
  WP-G  IEEE-14 robustness over Gaussian / bad-data / missing-measurement seeds
  WP-H  IEEE-30 useful-overlap search, staged degree escalation 31->63->127->255
  WP-I  IEEE-57 escalation; record the exact blocker if execution is infeasible

Convention correctness here is the SAME machine-precision block action validated
in run_generalized_rectangular_convention.py; this harness adds the APPLICATION
layer (RMSE ratio vs Ridge and ground truth, selected outputs, postselection,
resource cost). No ground-truth RMSE is ever used as a parameter selector.

Outputs under outputs/generalized_rectangular_qsvt/:
  ieee14_multioutput_statevector.csv, ieee14_multioutput_report.md
  ieee14_robustness_results.csv, ieee14_robustness_summary.csv, ieee14_robustness_report.md
  ieee30_useful_overlap_search.csv, ieee30_statevector_results.csv, ieee30_useful_overlap_report.md
  ieee57_escalation_results.csv, ieee57_escalation_report.md
"""

# ruff: noqa: E501,E741,RUF046

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev

from robust_qsvt_se.generalized.convention_api import (
    convert_pyqsp_to_production,
    make_request_from_phases,
)
from robust_qsvt_se.measurement.perturbations import (
    add_bad_data_outliers,
    add_gaussian_noise,
    remove_random_rows,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.rectangular_convention import (
    pcphase_qsvt_top_block,
    production_scalar_response,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
    fit_bounded_odd_chebyshev,
    synthesize_pyqsp_sym_qsp_phases,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "generalized_rectangular_qsvt"

# Pre-registered application threshold (mirror preregistered_criteria.yaml).
APP_RMSE_RATIO_THRESHOLD = 1.25
CONVENTION_TOL_HIGH = 1e-6


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def build_julia_square(M: np.ndarray) -> np.ndarray:
    """Julia-dilate an already-square matrix M to 2N x 2N."""

    N = M.shape[0]
    I = np.eye(N, dtype=M.dtype)
    sL = _psd_sqrt(I - M @ M.conj().T)
    sR = _psd_sqrt(I - M.conj().T @ M)
    return np.block([[M, sL], [sR, -M.conj().T]])


def apply_component(block: np.ndarray, component: str) -> np.ndarray:
    if component == "imag":
        return np.imag(block)
    if component == "neg_imag":
        return -np.imag(block)
    raise ValueError(component)


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# --------------------------------------------------------------------------- #
# core: evaluate one (case, degree, lambda, perturbation) configuration
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    ok: bool
    stage: str
    detail: dict


# Phase synthesis depends only on the polynomial fit (s_min, lam, degree), not on
# the matrix or perturbation. Cache it so a sweep over perturbations/seeds reuses
# one synthesis per (s_min, lam, degree). pyqsp sym_qsp at degree 255 is ~26s.
_PHASE_CACHE: dict[tuple, object] = {}


def evaluate_configuration(
    *,
    case: str,
    seed: int,
    degree: int,
    lam: float,
    perturbation: dict | None = None,
    selected_outputs: list[dict] | None = None,
) -> EvalResult:
    """Run the full useful-overlap evaluation for one configuration.

    Returns stage-tagged detail. Stops at the first failing gate and labels it
    (POLYNOMIAL_FAILED / PHASE_FAILED / RECTANGULAR_ACTION_FAILED /
    APPLICATION_FAILED / STATEVECTOR_PASSED / RESOURCE_LIMIT).
    """

    global _PHASE_CACHE

    system, _ = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": seed,
        }
    )
    if perturbation:
        rng = np.random.default_rng(perturbation.get("seed", seed))
        if perturbation["family"] == "gaussian":
            system = add_gaussian_noise(system, noise_std=perturbation["magnitude"], rng=rng)
        elif perturbation["family"] == "bad_data":
            cfg = {
                "enabled": True,
                "ratio": perturbation["ratio"],
                "magnitude": perturbation["magnitude"],
                "target": perturbation["target"],
            }
            system = add_bad_data_outliers(system, bad_data_config=cfg, rng=rng)
        elif perturbation["family"] == "missing":
            system = remove_random_rows(system, missing_ratio=perturbation["ratio"], rng=rng)

    H = np.asarray(system.H_tilde, dtype=float)
    r = np.asarray(system.r_tilde, dtype=float)
    m, n = H.shape
    beta = float(np.linalg.svd(H, compute_uv=False)[0])
    s_min = float(np.linalg.svd(H, compute_uv=False)[-1] / beta)
    cond = float(np.linalg.cond(H))
    alpha = lam * beta * beta
    x_true = np.asarray(system.metadata["true_state"], dtype=float)
    # ground-truth state perturbation: residual r = W^{1/2}(z_true - z0) corresponds
    # to dx_true = x_true - x0_lin (true state minus linearization point).
    lin_state = np.asarray(system.metadata.get("linearization_state", x_true), dtype=float)
    dx_true = x_true - lin_state

    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.T
    rr = min(m, n)

    # --- gate 1: polynomial boundedness (the BOUNDED Chebyshev coeffs must satisfy |P|<=1) ---
    ck = (round(s_min, 10), float(lam), int(degree))
    if ck in _PHASE_CACHE:
        bop, phases = _PHASE_CACHE[ck]
    else:
        try:
            bop = fit_bounded_odd_chebyshev(s_min=s_min, lam=lam, degree=degree)
        except Exception as exc:
            return EvalResult(
                False,
                "POLYNOMIAL_FAILED",
                {"error": str(exc)[:160], "case": case, "degree": degree},
            )
        Pcheb = Chebyshev(bop.chebyshev_coeffs, domain=[-1, 1])
        _grid = np.linspace(-1.0, 1.0, max(8193, 32 * degree + 1))
        bounded_global_max = float(np.max(np.abs(Pcheb(_grid))))
        if not (bounded_global_max <= 1.0 + 1e-9):
            return EvalResult(
                False,
                "POLYNOMIAL_FAILED",
                {"case": case, "degree": degree, "bounded_global_max": bounded_global_max},
            )
        try:
            phases = synthesize_pyqsp_sym_qsp_phases(bop.chebyshev_coeffs)
        except Exception as exc:
            return EvalResult(
                False, "PHASE_FAILED", {"error": str(exc)[:160], "case": case, "degree": degree}
            )
        _PHASE_CACHE[ck] = (bop, phases)

    # --- gate 3: convention conversion + rectangular block action ---
    req = make_request_from_phases(
        phases, degree=degree, configuration_id=f"{case}::d{degree}::lam{lam}"
    )
    res = convert_pyqsp_to_production(req)
    prod = res.phases
    comp = res.extraction_component
    A = H / beta
    pad = next_pow2(max(m, n))
    M = np.zeros((pad, pad))
    M[:m, :n] = A
    try:
        W = build_julia_square(M)
        top = pcphase_qsvt_top_block(W, prod, encoded_dimension=pad)
    except Exception as exc:
        return EvalResult(
            False,
            "RESOURCE_LIMIT",
            {"error": str(exc)[:160], "case": case, "degree": degree, "pad": pad},
        )
    B_prod = apply_component(top[:pad, :pad], comp)[:m, :n]  # convention polynomial block (m x n)

    # convention error vs exact-SVD of the ENCODED polynomial (isolates convention).
    # The convention acts on A=H/beta, so the polynomial is evaluated at NORMALIZED
    # singular values sigma_i = sv_i / beta (sv are H's singular values, same vectors).
    sigma_norm = sv / beta
    penc_sv = np.array(
        [
            production_scalar_response(min(1.0, max(-1.0, s)), prod, component=comp)
            for s in sigma_norm
        ]
    )
    B_exact = (U * penc_sv) @ Vh  # U diag(penc(sigma_i)) V^T  (m x n)
    conv_err = float(np.max(np.abs(B_prod - B_exact)) / max(np.linalg.norm(B_exact), 1e-300))
    if conv_err > CONVENTION_TOL_HIGH:
        return EvalResult(
            False,
            "RECTANGULAR_ACTION_FAILED",
            {"case": case, "degree": degree, "conv_err": conv_err},
        )

    # --- gate 4: application usefulness (RMSE ratio vs Ridge, ground truth) ---
    # physical QSVT update: dx_qsvt = (C_global/beta) * B_prod^T @ r  ;  Ridge update explicit.
    dx_qsvt = (bop.C_global / beta) * (B_prod.T @ r)
    ridge_filter = sv[:rr] / (sv[:rr] ** 2 + alpha)
    dx_ridge = V @ (ridge_filter * (U.T @ r))
    rmse_qsvt = float(np.linalg.norm(dx_qsvt - dx_true))
    rmse_ridge = float(np.linalg.norm(dx_ridge - dx_true))
    rmse_ratio = rmse_qsvt / max(rmse_ridge, 1e-300)
    app_pass = rmse_ratio <= APP_RMSE_RATIO_THRESHOLD

    # postselection success probability (state-space polynomial-action norm^2, estimated)
    r_hat = r / max(np.linalg.norm(r), 1e-300)
    p_succ_est = float(np.linalg.norm(B_prod.T @ r_hat) ** 2)

    # selected outputs
    sel_rows = []
    if selected_outputs:
        for so in selected_outputs:
            ell = np.zeros(n)
            signs = so.get("signs", [1.0] * len(so["indices"]))
            for idx, sg in zip(so["indices"], signs, strict=False):
                ell[idx] += sg
            y_qsvt = float(ell @ dx_qsvt)
            y_ridge = float(ell @ dx_ridge)
            y_exact = float(ell @ ((bop.C_global / beta) * (B_exact.T @ r)))
            sel_rows.append(
                {
                    "output": so["name"],
                    "y_qsvt": y_qsvt,
                    "y_ridge": y_ridge,
                    "y_exact_svd": y_exact,
                    "selected_rel_err_vs_exact": abs(y_qsvt - y_exact) / max(abs(y_exact), 1e-300),
                    "selected_rel_err_vs_ridge": abs(y_qsvt - y_ridge) / max(abs(y_ridge), 1e-300),
                }
            )

    detail = {
        "case": case,
        "seed": seed,
        "degree": degree,
        "lambda": lam,
        "alpha": alpha,
        "matrix_shape": f"{m}x{n}",
        "beta": beta,
        "s_min": s_min,
        "condition": cond,
        "C_global": float(bop.C_global),
        "global_max_abs": float(bop.global_max_abs),
        "phase_count": int(prod.size),
        "component": comp,
        "convention_block_error_vs_exact_svd": conv_err,
        "rmse_qsvt": rmse_qsvt,
        "rmse_ridge": rmse_ridge,
        "rmse_ratio": rmse_ratio,
        "postselection_probability_est": p_succ_est,
        "pad_dim": pad,
        "dilation_dim": 2 * pad,
        "qubits": int(math.ceil(math.log2(2 * pad))),
        "selected": sel_rows,
        "perturbation": perturbation["family"] if perturbation else "none",
    }
    stage = "STATEVECTOR_PASSED" if app_pass else "APPLICATION_FAILED"
    return EvalResult(app_pass, stage, detail)


# --------------------------------------------------------------------------- #
# WP-F: IEEE-14 multi-output (three pre-selected outputs)
# --------------------------------------------------------------------------- #
def ieee14_multioutput() -> None:
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    meta = system.metadata
    ang = list(meta["angle_state_indices"])  # theta columns (buses 2..14)
    vol = list(meta["voltage_magnitude_state_indices"])  # V columns (buses 1..14)
    # pre-selected (NOT post-hoc): non-slack angle theta_2, voltage magnitude V_1, area aggregate
    outputs = [
        {"name": "theta_2 (bus 2 angle, non-slack)", "indices": [ang[0]], "signs": [1.0]},
        {"name": "V_1 (bus 1 voltage magnitude)", "indices": [vol[0]], "signs": [1.0]},
        {
            "name": "area_aggregate_angle (buses 12,13,14 mean)",
            "indices": [ang[-3], ang[-2], ang[-1]],
            "signs": [1.0 / 3] * 3,
        },
    ]
    res = evaluate_configuration(
        case="ieee14", seed=123, degree=255, lam=1e-5, selected_outputs=outputs
    )
    rows = []
    if res.ok and res.detail.get("selected"):
        for s in res.detail["selected"]:
            rows.append(
                {
                    **{
                        k: res.detail[k]
                        for k in [
                            "case",
                            "degree",
                            "lambda",
                            "convention_block_error_vs_exact_svd",
                            "rmse_ratio",
                            "postselection_probability_est",
                        ]
                    },
                    **s,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "ieee14_multioutput_statevector.csv", index=False)
    # report
    lines = [
        "# IEEE-14 Multi-Output Validation (WP-F)",
        "",
        f"Configuration: degree {res.detail.get('degree')}, lambda {res.detail.get('lambda')}, "
        f"stage {res.stage}.",
        "",
        "Pre-selected outputs (chosen BEFORE inspecting errors):",
        "  1. theta_2 (bus 2 voltage angle, non-slack) -- the headline output",
        "  2. V_1 (bus 1 voltage magnitude)",
        "  3. area_aggregate_angle (mean of buses 12,13,14 angles)",
        "",
        "| output | y_QSVT | y_Ridge | y_exact-SVD | rel err vs exact | rel err vs Ridge |",
        "|---|---|---|---|---|---|",
    ]
    for s in res.detail.get("selected") or []:
        lines.append(
            f"| {s['output']} | {s['y_qsvt']:.6e} | {s['y_ridge']:.6e} | {s['y_exact_svd']:.6e} "
            f"| {s['selected_rel_err_vs_exact']:.3e} | {s['selected_rel_err_vs_ridge']:.3e} |"
        )
    lines += [
        "",
        f"Convention block error vs exact-SVD: {res.detail.get('convention_block_error_vs_exact_svd'):.3e}",
        f"Full-vector RMSE ratio (QSVT/Ridge vs ground truth): {res.detail.get('rmse_ratio'):.6f}",
        f"Postselection probability (estimated): {res.detail.get('postselection_probability_est'):.3e}",
    ]
    (OUT / "ieee14_multioutput_report.md").write_text("\n".join(lines))
    print(
        f"[WP-F] ieee14 multi-output: stage={res.stage}, rmse_ratio={res.detail.get('rmse_ratio'):.4f}"
    )


# --------------------------------------------------------------------------- #
# WP-G: IEEE-14 robustness over perturbation families x seeds
# --------------------------------------------------------------------------- #
def ieee14_robustness() -> None:
    families = {
        "gaussian": [
            {"family": "gaussian", "magnitude": m, "seed": s}
            for m in (0.01, 0.05)
            for s in range(10)
        ],
        "bad_data": [
            {"family": "bad_data", "ratio": 0.05, "magnitude": 5.0, "target": "random", "seed": s}
            for s in range(10)
        ],
        "missing": [{"family": "missing", "ratio": 0.10, "seed": s} for s in range(10)],
    }
    rows = []
    for fam, perts in families.items():
        for p in perts:
            res = evaluate_configuration(
                case="ieee14", seed=123, degree=255, lam=1e-5, perturbation=p
            )
            d = res.detail
            d["family"] = fam
            d["status"] = res.stage
            d["perturb_seed"] = p["seed"]
            rows.append(d)
    df = pd.DataFrame(
        [
            {
                k: r.get(k)
                for k in [
                    "family",
                    "perturb_seed",
                    "matrix_shape",
                    "condition",
                    "alpha",
                    "lambda",
                    "degree",
                    "global_max_abs",
                    "C_global",
                    "phase_count",
                    "convention_block_error_vs_exact_svd",
                    "rmse_qsvt",
                    "rmse_ridge",
                    "rmse_ratio",
                    "postselection_probability_est",
                    "status",
                ]
            }
            for r in rows
        ]
    )
    df.to_csv(OUT / "ieee14_robustness_results.csv", index=False)
    summ = (
        df.groupby("family")
        .agg(
            n=("status", "size"),
            statevector_passed=("status", lambda s: (s == "STATEVECTOR_PASSED").sum()),
            rmse_ratio_mean=("rmse_ratio", "mean"),
            rmse_ratio_max=("rmse_ratio", "max"),
            conv_err_max=("convention_block_error_vs_exact_svd", "max"),
        )
        .reset_index()
    )
    summ.to_csv(OUT / "ieee14_robustness_summary.csv", index=False)
    lines = [
        "# IEEE-14 Robustness Study (WP-G)",
        "",
        "Controlled perturbation families (NOT field-calibrated PMU/SCADA statistics):",
        "Gaussian residual perturbation, bad-data outliers, missing measurements.",
        "10 seeds per family. Degree 255, lambda 1e-5.",
        "",
        "| family | n | passed | RMSE ratio mean | RMSE ratio max | conv err max |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {r['family']} | {r['n']} | {r['statevector_passed']} | {r['rmse_ratio_mean']:.4f} "
            f"| {r['rmse_ratio_max']:.4f} | {r['conv_err_max']:.2e} |"
        )
    (OUT / "ieee14_robustness_report.md").write_text("\n".join(lines))
    print(f"[WP-G] ieee14 robustness: {len(df)} configs, {summ['statevector_passed'].sum()} passed")


# --------------------------------------------------------------------------- #
# WP-H: IEEE-30 useful-overlap search (staged degree escalation)
# --------------------------------------------------------------------------- #
def ieee30_search() -> None:
    degrees = [31, 63, 127, 255]
    lams = [1e-3, 1e-4, 1e-5]
    rows = []
    best = None
    for lam in lams:
        for deg in degrees:  # staged: do not jump to 255 first
            res = evaluate_configuration(case="ieee30", seed=123, degree=deg, lam=lam)
            d = res.detail
            d["status"] = res.stage
            rows.append(d)
            if res.ok and res.stage == "STATEVECTOR_PASSED" and best is None:
                best = d
    df = pd.DataFrame(
        [
            {
                k: r.get(k)
                for k in [
                    "lambda",
                    "degree",
                    "matrix_shape",
                    "beta",
                    "s_min",
                    "condition",
                    "C_global",
                    "global_max_abs",
                    "phase_count",
                    "convention_block_error_vs_exact_svd",
                    "rmse_ratio",
                    "postselection_probability_est",
                    "qubits",
                    "status",
                ]
            }
            for r in rows
        ]
    )
    df.to_csv(OUT / "ieee30_useful_overlap_search.csv", index=False)
    # statevector results for passing configs
    pd.DataFrame([r for r in rows if r["status"] == "STATEVECTOR_PASSED"]).to_csv(
        OUT / "ieee30_statevector_results.csv", index=False
    )
    lines = [
        "# IEEE-30 Useful-Overlap Search (WP-H)",
        "",
        "Staged degree escalation 31->63->127->255 over a preregistered lambda grid.",
        "No ground-truth RMSE used as a selector. Every candidate is recorded.",
        "",
        f"Total candidates: {len(df)}; STATEVECTOR_PASSED: {(df.status == 'STATEVECTOR_PASSED').sum()}",
        "",
        "| lambda | degree | rmse_ratio | conv_err | status |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['lambda']} | {r['degree']} | {r['rmse_ratio']:.4f} | {r['convention_block_error_vs_exact_svd']:.2e} | {r['status']} |"
        )
    if best:
        lines += [
            "",
            f"First passing configuration: lambda={best['lambda']}, degree={best['degree']}, "
            f"rmse_ratio={best['rmse_ratio']:.4f}, qubits={best['qubits']}.",
        ]
    else:
        lines += [
            "",
            "No configuration reached STATEVECTOR_PASSED (useful overlap NOT verified for IEEE-30).",
        ]
    (OUT / "ieee30_useful_overlap_report.md").write_text("\n".join(lines))
    print(
        f"[WP-H] ieee30 search: {len(df)} candidates, {(df.status == 'STATEVECTOR_PASSED').sum()} passed"
    )


# --------------------------------------------------------------------------- #
# WP-I: IEEE-57 escalation (record exact blocker)
# --------------------------------------------------------------------------- #
def ieee57_escalation() -> None:
    rows = []
    blocker = None
    # Low-degree failure rows plus the reported high-degree headline rows. The
    # high-degree rows are intentionally limited to lambda=1e-3, matching the
    # preregistered/reportable useful-overlap configuration.
    candidates = [
        (31, 1e-3),
        (31, 1e-4),
        (63, 1e-3),
        (63, 1e-4),
        (127, 1e-3),
        (255, 1e-3),
    ]
    for deg, lam in candidates:
        import time

        t0 = time.perf_counter()
        try:
            res = evaluate_configuration(case="ieee57", seed=123, degree=deg, lam=lam)
            dt = time.perf_counter() - t0
            d = res.detail
            d["status"] = res.stage
            d["wall_time_s"] = dt
            rows.append(d)
        except Exception as exc:
            dt = time.perf_counter() - t0
            blocker = f"{type(exc).__name__}: {str(exc)[:160]}"
            rows.append(
                {
                    "case": "ieee57",
                    "degree": deg,
                    "lambda": lam,
                    "status": "RESOURCE_LIMIT",
                    "wall_time_s": dt,
                    "blocker": blocker,
                }
            )
            break
        if blocker:
            break
    pd.DataFrame(rows).to_csv(OUT / "ieee57_escalation_results.csv", index=False)
    lines = [
        "# IEEE-57 Escalation (WP-I)",
        "",
        "IEEE-57 weighted Jacobian is [331,113] -> padded 512 -> dilation 1024 (10 qubits).",
        "Attempted staged low-degree execution first, then the reported high-degree",
        "lambda=1e-3 useful-overlap rows.",
        "",
    ]
    for r in rows:
        lines.append(
            f"- degree {r.get('degree')}, lambda {r.get('lambda')}: status={r.get('status')}, "
            f"wall={r.get('wall_time_s'):.1f}s"
            + (f", blocker={r.get('blocker')}" if r.get("blocker") else "")
        )
    if blocker:
        lines += [
            "",
            f"Measured blocker: {blocker}",
            "A measured blocker is an acceptable result (per protocol section 15).",
        ]
    else:
        passed = [r for r in rows if r.get("status") == "STATEVECTOR_PASSED"]
        lines += [
            "",
            f"STATEVECTOR_PASSED rows: {len(passed)}.",
            "IEEE-57 useful-overlap status is based only on rows satisfying RMSE ratio <= 1.25.",
        ]
    (OUT / "ieee57_escalation_report.md").write_text("\n".join(lines))
    print(f"[WP-I] ieee57: {len(rows)} configs attempted, blocker={blocker}")


def main() -> int:
    ieee14_multioutput()
    ieee14_robustness()
    ieee30_search()
    ieee57_escalation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
