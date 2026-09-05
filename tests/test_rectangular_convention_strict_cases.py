"""Strict cases required by the blocking-revision convention audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.generalized.convention_api import (
    ConversionError,
    convert_pyqsp_to_production,
    make_request_from_phases,
    predict_extraction,
)
from robust_qsvt_se.qsvt.rectangular_convention import (
    extract_component,
    pcphase_qsvt_top_block,
    production_scalar_response,
    validate_real_rectangular_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(hermitian)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def _julia(matrix: np.ndarray) -> np.ndarray:
    matrix = validate_real_rectangular_matrix(matrix)
    rows, columns = matrix.shape
    pad = max(rows, columns)
    padded = np.zeros((pad, pad), dtype=np.float64)
    padded[:rows, :columns] = matrix
    identity = np.eye(pad)
    return np.block(
        [
            [padded, _psd_sqrt(identity - padded @ padded.T)],
            [_psd_sqrt(identity - padded.T @ padded), -padded.T],
        ]
    )


def _matrix_with_spectrum(
    shape: tuple[int, int], singular_values: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, columns = shape
    left = np.linalg.qr(rng.normal(size=(rows, rows)))[0]
    right = np.linalg.qr(rng.normal(size=(columns, columns)))[0]
    sigma = np.zeros(shape)
    rank = min(shape)
    sigma[:rank, :rank] = np.diag(singular_values[:rank])
    return left @ sigma @ right.T


@pytest.mark.parametrize("degree", [1, 3, 5, 7])
@pytest.mark.parametrize(
    "shape,singular_values",
    [
        ((6, 4), np.array([0.82, 0.82, 1.0e-8, 0.0])),
        ((4, 6), np.array([0.91, 0.43, 1.0e-10, 0.0])),
    ],
)
def test_svd_lift_for_tall_and_wide_real_matrices(
    degree: int, shape: tuple[int, int], singular_values: np.ndarray
) -> None:
    """The rectangular lift holds for arbitrary phases and distinct U/V spaces."""

    rng = np.random.default_rng(81_000 + 101 * degree + shape[0])
    phases = rng.uniform(-np.pi, np.pi, degree + 1)
    matrix = _matrix_with_spectrum(shape, singular_values, seed=91_000 + degree)
    rows, columns = shape
    pad = max(shape)
    top = pcphase_qsvt_top_block(_julia(matrix), phases, encoded_dimension=pad)
    component = predict_extraction(degree)[0]
    extracted = extract_component(top, component)[:rows, :columns]

    left, values, right_t = np.linalg.svd(matrix, full_matrices=False)
    scalar_values = np.array(
        [production_scalar_response(value, phases, component=component) for value in values]
    )
    reference = (left * scalar_values) @ right_t
    assert np.max(np.abs(extracted - reference)) < 2.0e-12


def test_complex_matrix_is_rejected_by_claim_scope_guard() -> None:
    with pytest.raises(ValueError, match="complex matrices are unsupported"):
        validate_real_rectangular_matrix(np.eye(3, dtype=np.complex128))


def test_nonfinite_matrix_is_rejected_by_claim_scope_guard() -> None:
    matrix = np.eye(3)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_real_rectangular_matrix(matrix)


def test_nonfinite_phase_is_rejected_by_low_level_converter() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        convert_pyqsp_to_production(
            make_request_from_phases(
                np.array([0.0, np.inf]), degree=1, configuration_id="strict::nonfinite"
            )
        )


def test_invalid_phase_length_is_rejected() -> None:
    request = make_request_from_phases(
        np.zeros(4), degree=3, configuration_id="strict::bad-length"
    )
    request = type(request)(
        source_convention=request.source_convention,
        target_convention=request.target_convention,
        degree=request.degree,
        phases=np.zeros(5),
        expected_phase_count=request.expected_phase_count,
        extraction_component=request.extraction_component,
        extraction_sign=request.extraction_sign,
        configuration_id=request.configuration_id,
    )
    with pytest.raises(ConversionError, match="phase count"):
        convert_pyqsp_to_production(request)


def test_convention_summary_selects_exactly_one_allowed_status() -> None:
    summary = (
        ROOT / "outputs" / "tqe_blocking_revision" / "convention_validation_summary.md"
    ).read_text("utf-8")
    allowed = {
        "formally_derived_and_independently_validated",
        "empirically_validated_with_narrowed_claim",
        "unresolved_and_removed_from_headline_contribution",
    }
    present = {status for status in allowed if f"`{status}`" in summary}
    assert present == {"formally_derived_and_independently_validated"}
    frame = pd.read_csv(
        ROOT / "outputs" / "tqe_blocking_revision" / "convention_validation.csv",
        keep_default_na=False,
    )
    strict = frame[frame["campaign"] == "strict_live"]
    assert len(strict) >= 13
    assert set(strict["status"]) == {"pass"}
    assert {"4x6", "6x4"} <= set(strict["shape"])
