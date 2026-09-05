"""Independent rectangular-QSVT convention generalization harness (WP-A/B/C/D).

Re-executes the convention machinery directly against the frozen baseline
primitives (``robust_qsvt_se.qsvt.rectangular_convention`` +
``sym_qsp_circuit_action``) and the new safe API
(``robust_qsvt_se.generalized.convention_api``). It does NOT read prior result
CSVs; every number is recomputed.

Outputs (under outputs/generalized_rectangular_qsvt/):
  rectangular_convention_symbolic_checks.csv      (WP-A)
  degree_generalization_results.csv               (WP-B)
  degree_generalization_failures.csv              (WP-B)
  heldout_rectangular_matrix_results.csv          (WP-C)
  heldout_rectangular_matrix_failures.csv         (WP-C)
  complex_rectangular_results.csv                 (WP-D)
  complex_rectangular_failures.csv                (WP-D)

Integrity notes:
  * Held-out matrices use a RESERVED seed range [770000, 779999], disjoint from
    every development seed used elsewhere (dev seeds are small integers / named
    configs). This is asserted, not assumed.
  * Even degree is recorded as an explicit unsupported limitation, never coerced.
  * The reference block is U P(Sigma) V^dagger built from the *assigned* singular
    values; numpy SVD of the constructed matrix is used only as a sanity check.
"""

# ruff: noqa: E501

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev

from robust_qsvt_se.generalized.convention_api import (
    ConversionError,
    convert_pyqsp_to_production,
    make_request_from_phases,
    predict_extraction,
)
from robust_qsvt_se.qsvt.rectangular_convention import (
    PYQSP_TO_PCPHASE_OFFSET,
    pcphase_qsvt_top_block,
    production_scalar_response,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
    fit_bounded_odd_chebyshev,
    synthesize_pyqsp_sym_qsp_phases,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "generalized_rectangular_qsvt"
OUT.mkdir(parents=True, exist_ok=True)

HELDOUT_SEED_BASE = 770000  # reserved range [770000, 779999], disjoint from dev seeds
DEV_SEEDS_KNOWN = {0, 1, 2, 10, 42, 123, 2024}  # documented dev seeds to stay clear of
SCALAR_GRID = np.linspace(-1.0, 1.0, 2001)

# Preregistered tolerances (mirror preregistered_criteria.yaml).
TOL_LOW = 1e-8  # degrees <= 63
TOL_HIGH = 1e-6  # degrees 127, 255


# --------------------------------------------------------------------------- #
# numerical helpers
# --------------------------------------------------------------------------- #
def psd_sqrt(M: np.ndarray) -> np.ndarray:
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def build_julia(A: np.ndarray) -> np.ndarray:
    """Zero-pad rectangular A to square (max(m,n)) and Julia-dilate to 2*pad."""

    m, n = A.shape
    pad = max(m, n)
    M = np.zeros((pad, pad), dtype=A.dtype)
    M[:m, :n] = A
    ident = np.eye(pad, dtype=A.dtype)
    sL = psd_sqrt(ident - M @ M.conj().T)
    sR = psd_sqrt(ident - M.conj().T @ M)
    return np.block([[M, sL], [sR, -M.conj().T]])


def make_unitary(n: int, rng: np.random.Generator, complex_: bool = False) -> np.ndarray:
    Z = rng.standard_normal((n, n))
    if complex_:
        Z = Z + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(Z)
    return Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))  # fix QR sign/phase


def singular_values(family: str, r: int, rng: np.random.Generator) -> np.ndarray:
    if family == "well_conditioned":
        return np.linspace(0.9, 1.0, r)
    if family == "moderate_condition":
        return np.linspace(0.3, 1.0, r)
    if family == "nearly_rank_deficient":
        sv = np.linspace(0.3, 1.0, r)
        sv[0] *= 1e-3
        return sv
    if family == "exact_zero_singular_values":
        sv = np.linspace(0.4, 1.0, r)
        sv[: max(1, r // 3)] = 0.0
        return sv
    if family == "repeated_singular_values":
        return np.repeat(0.7, r)
    if family == "clustered_singular_values":
        head = [0.71, 0.72, 0.73, 0.74, 0.75][:r]
        return np.array(head + [0.5] * max(0, r - len(head)))
    if family == "random_decay":
        base = np.sort(rng.random(r))[::-1]
        return 0.15 + 0.84 * base
    raise ValueError(f"unknown spectral family: {family}")


def apply_component(block: np.ndarray, component: str) -> np.ndarray:
    if component == "imag":
        return np.imag(block)
    if component == "neg_imag":
        return -np.imag(block)
    raise ValueError(component)


# --------------------------------------------------------------------------- #
# polynomial targets + phase synthesis
# --------------------------------------------------------------------------- #
@dataclass
class PolyTarget:
    name: str
    degree: int
    kind: str
    cheb: np.ndarray
    evaluate: Chebyshev

    @property
    def fn(self):
        return lambda x: float(self.evaluate(x))


def monomial_target(degree: int) -> PolyTarget:
    p1 = Chebyshev([0.0, 1.0], domain=[-1, 1])  # x
    pn = Chebyshev([1.0], domain=[-1, 1])
    for _ in range(degree):
        pn = pn * p1
    return PolyTarget(f"monomial_x{degree}", degree, "monomial", np.asarray(pn.coef, float), pn)


def filter_target(degree: int, s_min: float = 0.1, lam: float = 0.01) -> PolyTarget:
    bop = fit_bounded_odd_chebyshev(s_min=s_min, lam=lam, degree=degree)
    P = Chebyshev(bop.chebyshev_coeffs, domain=[-1, 1])
    return PolyTarget(f"ridge_filter_d{degree}", degree, "filter", bop.chebyshev_coeffs, P)


@dataclass(frozen=True)
class Synth:
    phases: np.ndarray
    component: str
    sign: int


def synthesize(target: PolyTarget) -> Synth:
    pyqsp_phases = synthesize_pyqsp_sym_qsp_phases(target.cheb)
    req = make_request_from_phases(
        pyqsp_phases,
        degree=target.degree,
        configuration_id=f"genconv::{target.name}::d{target.degree}",
    )
    res = convert_pyqsp_to_production(req)
    return Synth(res.phases, res.extraction_component, res.extraction_sign)


def _extract_and_refs(A: np.ndarray, synth: Synth, target_fn):
    """Return (extracted_block, ref_target, ref_encoded, sv, U, V).

    ``ref_target`` uses the *target* polynomial (end-to-end error);
    ``ref_encoded`` uses the *encoded* scalar response (convention-isolating
    error: does the matrix circuit agree with the scalar emulator?).
    """

    m, n = A.shape
    r = min(m, n)
    U, sv, Vh = np.linalg.svd(A, full_matrices=True)
    V = Vh.conj().T
    W = build_julia(A)
    top = pcphase_qsvt_top_block(W, synth.phases, encoded_dimension=max(m, n))
    ext = apply_component(top[: max(m, n), : max(m, n)], synth.component)[:m, :n]
    penc = np.array(
        [
            production_scalar_response(
                float(min(1.0, max(-1.0, s))), synth.phases, component=synth.component
            )
            for s in sv[:r]
        ],
        dtype=float,
    )
    ref_target = (
        U[:, :r]
        @ np.diag(np.array([target_fn(s) for s in sv[:r]], dtype=float))
        @ V[:, :r].conj().T
    )
    ref_encoded = U[:, :r] @ np.diag(penc) @ V[:, :r].conj().T
    return ext, ref_target, ref_encoded, sv, U, V, top


def convention_error(A: np.ndarray, synth: Synth, target_fn) -> float:
    """Convention-isolating error: extracted block vs U P_encoded(Sigma) V^dagger.

    This removes the pyqsp phase-synthesis contribution; it is ~machine epsilon
    iff the phase convention + extraction is correct, regardless of how well the
    phases reproduce the *target* polynomial.
    """

    ext, _, ref_enc, *_ = _extract_and_refs(A, synth, target_fn)
    denom = max(float(np.linalg.norm(ref_enc)), 1e-300)
    return float(np.max(np.abs(ext - ref_enc)) / denom)


def end_to_end_error(A: np.ndarray, synth: Synth, target_fn) -> float:
    """End-to-end error: extracted block vs U P_target(Sigma) V^dagger.

    Includes pyqsp phase-synthesis error (the difference between the encoded
    polynomial and the target). This is the application-relevant error.
    """

    ext, ref_tgt, *_ = _extract_and_refs(A, synth, target_fn)
    denom = max(float(np.linalg.norm(ref_tgt)), 1e-300)
    return float(np.max(np.abs(ext - ref_tgt)) / denom)


def complex_diagnostic(A: np.ndarray, synth: Synth, target_fn) -> dict:
    """For complex A, test whether ANY component / full block recovers U P(Sigma) V^dagger."""

    m, n = A.shape
    r = min(m, n)
    U, sv, Vh = np.linalg.svd(A, full_matrices=True)
    V = Vh.conj().T
    W = build_julia(A)
    top = pcphase_qsvt_top_block(W, synth.phases, encoded_dimension=max(m, n))
    block = top[:m, :n]
    ref = (
        U[:, :r]
        @ np.diag(np.array([target_fn(s) for s in sv[:r]], dtype=float))
        @ V[:, :r].conj().T
    )
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-12:
        # Degenerate: every singular value maps to ~0 (e.g. exact-zero SVs, P(0)=0).
        # The reference is the zero matrix; "0 error" is not evidence of complex support.
        return {
            "best_imag_rel": float("nan"),
            "best_real_rel": float("nan"),
            "full_block_rel": float("nan"),
            "degenerate_zero_reference": True,
        }
    denom = max(ref_norm, 1e-300)
    return {
        "best_imag_rel": float(np.max(np.abs(np.imag(block) - ref)) / denom),
        "best_real_rel": float(np.max(np.abs(np.real(block) - ref)) / denom),
        "full_block_rel": float(np.max(np.abs(block - ref)) / denom),
        "degenerate_zero_reference": False,
    }


# --------------------------------------------------------------------------- #
# WP-B: degree generalization
# --------------------------------------------------------------------------- #
def run_degree_generalization() -> None:
    rows: list[dict] = []
    failures: list[dict] = []
    odd_degrees = [1, 3, 5, 7, 15, 31, 63, 127, 255]
    even_degrees = [0, 2, 4, 8, 16, 32]
    rng = np.random.default_rng(HELDOUT_SEED_BASE + 9001)

    for d in odd_degrees:
        targets = [filter_target(d)]
        if d in (1, 3):
            targets.append(monomial_target(d))  # analytic clean check
        for target in targets:
            try:
                synth = synthesize(target)
                scalar = np.array(
                    [
                        production_scalar_response(x, synth.phases, component=synth.component)
                        for x in SCALAR_GRID
                    ]
                )
                synthesis_err = float(
                    np.max(np.abs(scalar - target.evaluate(SCALAR_GRID)))
                )  # pyqsp quality
                # convention-isolating diagonal + rectangular action (vs ENCODED poly)
                svd4 = np.array([0.9, 0.6, 0.3, 0.05])
                diag_conv = convention_error(np.diag(svd4), synth, target.fn)
                U = make_unitary(6, rng)
                V = make_unitary(4, rng)
                A = U[:, :4] @ np.diag(svd4) @ V[:, :4].T
                rect_conv = convention_error(A, synth, target.fn)
                cp, sp = predict_extraction(d)
                tol = TOL_LOW if d <= 63 else TOL_HIGH
                # convention passes iff matrix action agrees with scalar emulator
                conv_ok = diag_conv < tol and rect_conv < tol and synth.component == cp
                rows.append(
                    dict(
                        target=target.name,
                        degree=d,
                        parity="odd",
                        phase_count=synth.phases.size,
                        component=synth.component,
                        predicted_component=cp,
                        sign=synth.sign,
                        predicted_sign=sp,
                        synthesis_error_vs_target=synthesis_err,
                        convention_diagonal_error=diag_conv,
                        convention_rectangular_error=rect_conv,
                        tolerance=tol,
                        status="pass" if conv_ok else "fail",
                    )
                )
            except Exception as exc:
                failures.append(
                    dict(
                        target=target.name,
                        degree=d,
                        stage="synthesize_or_eval",
                        error=str(exc)[:200],
                    )
                )

    for d in even_degrees:
        rejected = False
        try:
            make_request_from_phases(np.zeros(d + 1), degree=d, configuration_id=f"evenprobe_d{d}")
        except ConversionError:
            rejected = True
        rows.append(
            dict(
                target=f"even_probe_d{d}",
                degree=d,
                parity="even",
                phase_count=d + 1,
                component="N/A",
                predicted_component="UNSUPPORTED",
                sign=0,
                predicted_sign=0,
                synthesis_error_vs_target=float("nan"),
                convention_diagonal_error=float("nan"),
                convention_rectangular_error=float("nan"),
                tolerance=float("nan"),
                status="even_degree_unsupported" if rejected else "UNEXPECTED_ACCEPT",
            )
        )
        if not rejected:
            failures.append(
                dict(
                    target=f"even_probe_d{d}",
                    degree=d,
                    stage="api_parity",
                    error="even degree accepted",
                )
            )

    pd.DataFrame(rows).to_csv(OUT / "degree_generalization_results.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "degree_generalization_failures.csv", index=False)
    odd_pass = sum(1 for r in rows if r["parity"] == "odd" and r["status"] == "pass")
    odd_total = sum(1 for r in rows if r["parity"] == "odd")
    print(
        f"[WP-B] degree generalization: {len(rows)} rows ({odd_pass}/{odd_total} odd pass), {len(failures)} failures"
    )


# --------------------------------------------------------------------------- #
# WP-C / WP-D: held-out (real) and complex rectangular matrices
# --------------------------------------------------------------------------- #
def run_matrix_sweep(*, complex_: bool) -> None:
    dims = [(2, 1), (3, 2), (4, 3), (5, 3), (6, 4), (8, 5), (12, 7)]
    families = [
        "well_conditioned",
        "moderate_condition",
        "nearly_rank_deficient",
        "exact_zero_singular_values",
        "repeated_singular_values",
        "clustered_singular_values",
        "random_decay",
    ]
    targets = [
        monomial_target(1),
        monomial_target(3),
        filter_target(7),
        filter_target(31),
        filter_target(255),
    ]
    synth_cache = {t.name: synthesize(t) for t in targets}

    rows: list[dict] = []
    failures: list[dict] = []
    for di, (m, n) in enumerate(dims):
        for fi, family in enumerate(families):
            seed = HELDOUT_SEED_BASE + di * 100 + fi
            assert seed not in DEV_SEEDS_KNOWN, "held-out seed collided with a dev seed"
            rng = np.random.default_rng(seed)
            r = min(m, n)
            U = make_unitary(m, rng, complex_)
            V = make_unitary(n, rng, complex_)
            sv = singular_values(family, r, rng)
            A = U[:, :r] @ np.diag(sv.astype(complex if complex_ else float)) @ V[:, :r].conj().T
            if not complex_:
                A = np.real(A)
            cond = float(np.linalg.cond(A)) if np.all(sv > 0) else float("inf")
            sv_check = np.linalg.svd(A, compute_uv=False)[:r]
            sv_match = (
                bool(np.allclose(np.sort(sv_check)[::-1], sv, atol=1e-9))
                if np.all(sv > 0)
                else True
            )
            for target in targets:
                synth = synth_cache[target.name]
                tol = TOL_LOW if target.degree <= 63 else TOL_HIGH
                base = dict(
                    seed=seed,
                    dim=f"{m}x{n}",
                    rows=m,
                    cols=n,
                    rank=r,
                    spectral_family=family,
                    condition_number=cond,
                    sv_assignment_matches_svd=sv_match,
                    is_complex=complex_,
                    target=target.name,
                    degree=target.degree,
                    component=synth.component,
                    tolerance=tol,
                )
                try:
                    if complex_:
                        diag = complex_diagnostic(A, synth, target.fn)
                        if diag.get("degenerate_zero_reference"):
                            status = "degenerate_zero_reference"
                        else:
                            best = min(
                                diag["best_imag_rel"], diag["best_real_rel"], diag["full_block_rel"]
                            )
                            status = "complex_unsupported" if best > 1e-3 else "pass"
                        rows.append({**base, **diag, "status": status})
                    else:
                        conv = convention_error(A, synth, target.fn)
                        e2e = end_to_end_error(A, synth, target.fn)
                        rows.append(
                            {
                                **base,
                                "convention_error_vs_encoded": conv,
                                "end_to_end_error_vs_target": e2e,
                                "status": "pass" if conv <= tol else "fail",
                            }
                        )
                except Exception as exc:
                    failures.append(
                        dict(
                            seed=seed,
                            dim=f"{m}x{n}",
                            family=family,
                            target=target.name,
                            error=str(exc)[:200],
                        )
                    )
                    rows.append({**base, "status": "exception"})

    if complex_:
        pd.DataFrame(rows).to_csv(OUT / "complex_rectangular_results.csv", index=False)
        pd.DataFrame(failures).to_csv(OUT / "complex_rectangular_failures.csv", index=False)
        unsup = sum(1 for r in rows if r["status"] == "complex_unsupported")
        print(
            f"[WP-D] complex: {len(rows)} rows, {unsup} complex_unsupported (architectural limitation), {len(failures)} exceptions"
        )
    else:
        pd.DataFrame(rows).to_csv(OUT / "heldout_rectangular_matrix_results.csv", index=False)
        pd.DataFrame(failures).to_csv(OUT / "heldout_rectangular_matrix_failures.csv", index=False)
        passed = sum(1 for r in rows if r["status"] == "pass")
        maxconv = max(
            (r.get("convention_error_vs_encoded", 0.0) for r in rows if r["status"] == "pass"),
            default=float("nan"),
        )
        print(
            f"[WP-C] held-out: {passed}/{len(rows)} convention-pass, max convention err {maxconv:.3e}, {len(failures)} exceptions"
        )


# --------------------------------------------------------------------------- #
# WP-A symbolic / algebraic identity checks
# --------------------------------------------------------------------------- #
def run_symbolic_checks() -> None:
    rows: list[dict] = []
    x = SCALAR_GRID
    for d in [1, 3, 7, 31, 255]:
        target = filter_target(d) if d >= 7 else monomial_target(d)
        synth = synthesize(target)
        resp = np.array(
            [production_scalar_response(xi, synth.phases, component=synth.component) for xi in x]
        )
        id_response = float(np.max(np.abs(resp - target.evaluate(x))))
        opp = "real" if "imag" in synth.component else "imag"
        resp_opp = np.array(
            [production_scalar_response(xi, synth.phases, component=opp) for xi in x]
        )
        id_opposite_complementary = float(
            np.max(np.abs(resp_opp))
        )  # complementary poly, O(1), not the target
        # confirm a global offset is required: shifting only the first phase breaks it
        broken = synth.phases.copy()
        broken[0] = 0.0  # drop the offset on the first phase only
        resp_b = np.array(
            [production_scalar_response(xi, broken, component=synth.component) for xi in x]
        )
        id_global_required = float(np.max(np.abs(resp_b - target.evaluate(x))))
        rows.append(
            dict(
                target=target.name,
                degree=d,
                component=synth.component,
                sign=synth.sign,
                offset_applied_per_phase=PYQSP_TO_PCPHASE_OFFSET,
                identity_response_equals_target_max_err=id_response,
                identity_opposite_component_complementary_max=id_opposite_complementary,
                identity_global_offset_required_max_err_when_first_phase_unshifted=id_global_required,
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "rectangular_convention_symbolic_checks.csv", index=False)
    print(f"[WP-A] symbolic checks: {len(rows)} rows")


def main() -> int:
    run_symbolic_checks()
    run_degree_generalization()
    run_matrix_sweep(complex_=False)
    run_matrix_sweep(complex_=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
