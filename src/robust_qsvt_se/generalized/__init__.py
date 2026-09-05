"""Generalized rectangular-QSVT study package.

New code added during the generalization research phase (branch
``research/generalized-rectangular-qsvt``). It builds on, but does not mutate, the
frozen baseline convention primitives in
``robust_qsvt_se.qsvt.rectangular_convention``.
"""

from __future__ import annotations

from robust_qsvt_se.generalized.convention_api import (
    SUPPORTED_SOURCE_CONVENTIONS,
    SUPPORTED_TARGET_CONVENTIONS,
    ConversionError,
    ConversionRequest,
    ConversionResult,
    convert_pyqsp_to_production,
    predict_extraction,
)

__all__ = [
    "SUPPORTED_SOURCE_CONVENTIONS",
    "SUPPORTED_TARGET_CONVENTIONS",
    "ConversionError",
    "ConversionRequest",
    "ConversionResult",
    "convert_pyqsp_to_production",
    "predict_extraction",
]
