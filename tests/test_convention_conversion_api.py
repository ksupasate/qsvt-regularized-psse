"""Tests for the safe convention-conversion API (WP-E).

Asserts every error path raises ConversionError and the success path returns the
correct metadata. Mirrors production_convention_api_validation.csv.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import Chebyshev

from robust_qsvt_se.generalized.convention_api import (
    ConversionError,
    ConversionRequest,
    convert_pyqsp_to_production,
    make_request_from_phases,
    predict_extraction,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import synthesize_pyqsp_sym_qsp_phases


@pytest.fixture(scope="module")
def phases():
    p1 = Chebyshev([0, 1], domain=[-1, 1])
    pn = Chebyshev([1], domain=[-1, 1])
    for _ in range(3):
        pn = pn * p1
    return synthesize_pyqsp_sym_qsp_phases(np.asarray(pn.coef, float))


def _good(base_phases, **ov):
    kw = dict(
        source_convention="pyqsp_sym_qsp_plus_i",
        target_convention="dense_julia_pcphase",
        degree=3,
        phases=base_phases,
        expected_phase_count=4,
        extraction_component="imag",
        extraction_sign=1,
        configuration_id="t::d3",
    )
    kw.update(ov)
    return ConversionRequest(**kw)


def test_success_returns_metadata(phases):
    res = convert_pyqsp_to_production(_good(phases))
    assert res.degree == 3
    assert res.extraction_component == "imag"
    assert res.extraction_sign == 1
    assert abs(res.applied_offset - np.pi / 2) < 1e-12
    assert res.conversion_checksum
    assert res.phase_mapping and res.phase_ordering


def test_reject_unknown_source(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, source_convention="bogus"))


def test_reject_unknown_target(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, target_convention="bogus"))


def test_reject_wrong_phase_count(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, phases=np.zeros(5), expected_phase_count=5))


def test_reject_inconsistent_degree(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(
            _good(
                phases,
                degree=5,
                expected_phase_count=6,
                phases=np.zeros(6),
                extraction_component="imag",
                extraction_sign=1,
            )
        )


def test_reject_expected_count_mismatch(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, expected_phase_count=99))


def test_reject_double_conversion(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, already_converted=True))


def test_reject_ambiguous_component(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, extraction_component="neg_imag"))


def test_reject_wrong_sign(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, extraction_sign=-1))


def test_reject_even_degree():
    with pytest.raises(ConversionError):
        make_request_from_phases(np.zeros(5), degree=4, configuration_id="even")


def test_reject_missing_config_id(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, configuration_id=""))


def test_reject_non_finite_phases(phases):
    with pytest.raises(ConversionError):
        convert_pyqsp_to_production(_good(phases, phases=np.array([np.nan, 0.0, 0.0, 0.0])))


@pytest.mark.parametrize("degree,expected", [(1, ("neg_imag", -1)), (255, ("imag", 1))])
def test_predict_extraction(degree, expected):
    assert predict_extraction(degree) == expected
