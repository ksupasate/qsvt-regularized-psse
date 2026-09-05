"""Tests for complex-matrix support of the rectangular convention (WP-D).

The convention (Julia dilation + imag extraction) is architecturally
real-matrix-specific. This test enforces the documented limitation: complex
matrices must NOT be reported as passing. It verifies both the artifact
(complex_rectangular_results.csv has no false 'pass' rows) and the live
diagnostic (for a complex matrix, no component / full block recovers
U P(Sigma) V^dagger).

Integrity invariant: complex support must not be claimed falsely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev

from robust_qsvt_se.generalized.convention_api import (
    convert_pyqsp_to_production,
    make_request_from_phases,
)
from robust_qsvt_se.qsvt.rectangular_convention import pcphase_qsvt_top_block
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import synthesize_pyqsp_sym_qsp_phases

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "complex_rectangular_results.csv"


def test_complex_results_document_unsupported():
    df = pd.read_csv(CSV)
    # No row may claim a real complex pass (status in the unsupported/degenerate set).
    false_passes = df[~df.status.isin(["complex_unsupported", "degenerate_zero_reference"])]
    assert len(false_passes) == 0, (
        f"complex rows falsely claimed support: {false_passes['status'].unique()}"
    )
    assert (df.status == "complex_unsupported").sum() >= 200


def _psd_sqrt(M):
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def test_live_complex_matrix_fails_all_components():
    """For a complex matrix, imag/real/full block all fail to recover U P(Sigma) Vt."""

    p1 = Chebyshev([0, 1], domain=[-1, 1])
    pn = Chebyshev([1], domain=[-1, 1])
    for _ in range(3):
        pn = pn * p1
    phases = synthesize_pyqsp_sym_qsp_phases(np.asarray(pn.coef, float))
    res = convert_pyqsp_to_production(
        make_request_from_phases(phases, degree=3, configuration_id="complex::d3")
    )
    rng = np.random.default_rng(31337)
    Z = rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3))
    A = Z[:, :3]
    A = A / np.linalg.svd(A, compute_uv=False)[0]  # normalize ||A||<=1
    m, n = A.shape
    pad = max(m, n)
    M = np.zeros((pad, pad), dtype=complex)
    M[:m, :n] = A
    eye = np.eye(pad, dtype=complex)
    W = np.block(
        [
            [M, _psd_sqrt(eye - M @ M.conj().T)],
            [_psd_sqrt(eye - M.conj().T @ M), -M.conj().T],
        ]
    )
    top = pcphase_qsvt_top_block(W, res.phases, encoded_dimension=pad)
    block = top[:m, :n]
    Ua, sv, Vh = np.linalg.svd(A, full_matrices=False)
    ref = (Ua[:, :3] * np.array([float(pn(s)) for s in sv[:3]])) @ Vh[:3]
    denom = max(np.linalg.norm(ref), 1e-300)
    err_imag = np.max(np.abs(np.imag(block) - ref)) / denom
    err_real = np.max(np.abs(np.real(block) - ref)) / denom
    err_full = np.max(np.abs(block - ref)) / denom
    # none of the components should recover the complex reference
    assert min(err_imag, err_real, err_full) > 1e-3
