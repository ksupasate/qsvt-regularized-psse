"""Tests for degree generalization of the rectangular convention (WP-B).

LIVE test: for every odd degree in {1,3,5,7,15,31,63,127,255} the convention
isolated error (matrix action vs scalar emulator) is at machine precision, and
the component matches the predicted rule. Even degree is rejected.

Integrity invariant: the rule must NOT be degree-255-specific. The test asserts
the SAME convention works at degree 1 and degree 255, with the predicted
component, so a degree-255-specific hack would fail the low-degree cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.generalized.convention_api import (
    ConversionError,
    convert_pyqsp_to_production,
    make_request_from_phases,
    predict_extraction,
)
from robust_qsvt_se.qsvt.rectangular_convention import (
    pcphase_qsvt_top_block,
    production_scalar_response,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
    fit_bounded_odd_chebyshev,
    synthesize_pyqsp_sym_qsp_phases,
)


def _psd_sqrt(M):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _julia(A):
    m, n = A.shape
    pad = max(m, n)
    M = np.zeros((pad, pad))
    M[:m, :n] = A
    eye = np.eye(pad)
    sL = _psd_sqrt(eye - M @ M.T)
    sR = _psd_sqrt(eye - M.T @ M)
    return np.block([[M, sL], [sR, -M.T]])


def _convention_error(A, phases, comp):
    m, n = A.shape
    U, sv, Vh = np.linalg.svd(A, full_matrices=False)
    W = _julia(A)
    top = pcphase_qsvt_top_block(W, phases, encoded_dimension=max(m, n))
    ext = (
        np.imag(top[: max(m, n), : max(m, n)])
        if comp == "imag"
        else -np.imag(top[: max(m, n), : max(m, n)])
    )[:m, :n]
    penc = np.array([production_scalar_response(min(1.0, s), phases, component=comp) for s in sv])
    ref = (U * penc) @ Vh
    return float(np.max(np.abs(ext - ref)) / max(np.linalg.norm(ref), 1e-300))


@pytest.mark.parametrize("degree", [1, 3, 5, 7, 15, 31, 63, 127, 255])
def test_odd_degree_convention_is_exact(degree):
    bop = fit_bounded_odd_chebyshev(s_min=0.1, lam=0.01, degree=degree)
    phases = synthesize_pyqsp_sym_qsp_phases(bop.chebyshev_coeffs)
    res = convert_pyqsp_to_production(
        make_request_from_phases(phases, degree=degree, configuration_id=f"t::d{degree}")
    )
    assert res.extraction_component == predict_extraction(degree)[0]
    rng = np.random.default_rng(770500 + degree)
    U = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    V = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    A = U[:, :4] @ np.diag([0.9, 0.6, 0.3, 0.0][:4]) @ V[:, :4].T
    err = _convention_error(A, res.phases, res.extraction_component)
    tol = 1e-8 if degree <= 63 else 1e-6
    assert err < tol, f"degree {degree}: convention error {err} > {tol}"


@pytest.mark.parametrize("degree", [0, 2, 4, 8, 16, 32])
def test_even_degree_rejected(degree):
    with pytest.raises(ConversionError):
        make_request_from_phases(
            np.zeros(degree + 1), degree=degree, configuration_id=f"even::{degree}"
        )


def test_low_degree_and_high_degree_use_same_rule():
    """A degree-255-specific rule would make degree 1 fail this consistency check."""

    for d in (1, 255):
        comp, _sign = predict_extraction(d)
        expected = "imag" if (((d + 1) // 2) % 2 == 0) else "neg_imag"
        assert comp == expected
