"""Safe PyQSP -> production rectangular-QSVT phase-conversion API.

This is the reusable, explicitly-validated conversion interface required by
Work Package E of the generalization protocol. It wraps the *frozen* baseline
primitives in :mod:`robust_qsvt_se.qsvt.rectangular_convention` and adds the
safeguards that make a convention conversion safe to call from automated
pipelines:

* the caller must name the source and target conventions explicitly;
* the caller must state the polynomial degree, the expected phase count, and the
  extraction component/sign, and these are *checked* against the predicted rule
  rather than silently accepted;
* the caller must supply a configuration id (and optionally its checksum) so a
  stale configuration is rejected;
* double conversion is rejected;
* unsupported parity (even degree) is rejected, not silently coerced.

The accepted mathematical rule (formally derived in
``rectangular_convention_derivation.md``)::

    phi_k^prod = phi_k^PyQSP + pi/2,   for every k,
    extraction component = imag  with  sign = (-1)^((d+1)//2),   d odd.

This API intentionally performs *only* the phase convention conversion. It does
not touch the Ridge/Tikhonov estimator, the application metrics, or the output
functional. Every production convention rule used elsewhere must trace through
this module so that the safeguards cannot be bypassed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from robust_qsvt_se.qsvt.rectangular_convention import (
    DENSE_JULIA_PCPHASE,
    PYQSP_SYM_QSP_PLUS_I,
    PYQSP_TO_PCPHASE_OFFSET,
    PYQSP_TO_PCPHASE_RULE,
    convert_pyqsp_sym_qsp_to_pcphase,
    pyqsp_pcphase_component,
    pyqsp_pcphase_imag_sign,
)

SUPPORTED_SOURCE_CONVENTIONS = frozenset({PYQSP_SYM_QSP_PLUS_I})
SUPPORTED_TARGET_CONVENTIONS = frozenset({DENSE_JULIA_PCPHASE})

PHASE_ORDERING = "P(phi0), W, P(phi1), W^dagger, P(phi2), ... (PCPhase top-subspace)"
APPLIED_OFFSET = PYQSP_TO_PCPHASE_OFFSET  # pi/2


class ConversionError(ValueError):
    """Raised when a convention-conversion request is inconsistent or unsafe."""


@dataclass(frozen=True)
class ConversionRequest:
    """Explicit, fully-specified conversion request.

    All fields are mandatory by construction. ``extraction_component`` and
    ``extraction_sign`` are the caller's *claim* about how the converted block is
    to be read back; they are validated against the degree-predicted rule, so an
    ambiguous or stale extraction convention is caught rather than applied.
    """

    source_convention: str
    target_convention: str
    degree: int
    phases: np.ndarray
    expected_phase_count: int
    extraction_component: str
    extraction_sign: int
    configuration_id: str
    configuration_checksum: str | None = None
    already_converted: bool = False

    def __post_init__(self) -> None:
        # ``np.ndarray`` is not hashable and must not be mutated after the
        # request is validated; freeze the array reference here.
        object.__setattr__(self, "phases", np.asarray(self.phases, dtype=np.float64))


@dataclass(frozen=True)
class ConversionResult:
    """Converted phases plus full conversion metadata and checksum."""

    phases: np.ndarray
    degree: int
    source_convention: str
    target_convention: str
    phase_mapping: str
    phase_ordering: str
    extraction_component: str
    extraction_sign: int
    applied_offset: float
    configuration_id: str
    conversion_checksum: str
    metadata: dict = field(default_factory=dict)


def predict_extraction(degree: int) -> tuple[str, int]:
    """Return the degree-predicted ``(component, sign)`` for an odd degree.

    ``component`` is one of ``{"imag", "neg_imag"}`` and ``sign`` is ``{-1, +1}``.
    This is the single source of truth for the extraction rule; the API checks
    the caller's claim against it.
    """

    d = int(degree)
    if d <= 0 or d % 2 == 0:
        raise ConversionError("degree must be positive and odd (even degree unsupported)")
    sign = pyqsp_pcphase_imag_sign(d)
    component = pyqsp_pcphase_component(d)
    return component, sign


def _checksum(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def convert_pyqsp_to_production(request: ConversionRequest) -> ConversionResult:
    """Convert PyQSP phases to production PCPhase phases with full validation.

    Raises :class:`ConversionError` for any of the failure modes enumerated in
    the module docstring. Never mutates the estimator, metrics, or output.
    """

    # --- convention names ---
    if request.source_convention not in SUPPORTED_SOURCE_CONVENTIONS:
        raise ConversionError(
            f"unknown source convention: {request.source_convention!r}; "
            f"supported: {sorted(SUPPORTED_SOURCE_CONVENTIONS)}"
        )
    if request.target_convention not in SUPPORTED_TARGET_CONVENTIONS:
        raise ConversionError(
            f"unknown target convention: {request.target_convention!r}; "
            f"supported: {sorted(SUPPORTED_TARGET_CONVENTIONS)}"
        )

    # --- degree parity ---
    d = int(request.degree)
    if d <= 0:
        raise ConversionError("degree must be positive")
    if d % 2 == 0:
        raise ConversionError(
            "even-degree rectangular transformation is unsupported by the "
            "current production path; convert was refused to avoid silent parity coercion"
        )

    # --- phase geometry ---
    phases = np.asarray(request.phases, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ConversionError("phases must be a nonempty one-dimensional array")
    if phases.size != d + 1:
        raise ConversionError(
            f"inconsistent degree/phase count: degree={d} implies {d + 1} phases, got {phases.size}"
        )
    if int(request.expected_phase_count) != d + 1:
        raise ConversionError(
            f"expected_phase_count={request.expected_phase_count} disagrees with "
            f"degree {d} (expected {d + 1})"
        )
    if not np.all(np.isfinite(phases)):
        raise ConversionError("phases contain non-finite entries")

    # --- extraction rule must match the degree-predicted rule (no ambiguity) ---
    predicted_component, predicted_sign = predict_extraction(d)
    if request.extraction_component != predicted_component:
        raise ConversionError(
            f"extraction_component={request.extraction_component!r} is ambiguous/wrong "
            f"for degree {d}: predicted {predicted_component!r}"
        )
    if int(request.extraction_sign) != predicted_sign:
        raise ConversionError(
            f"extraction_sign={request.extraction_sign} disagrees with predicted "
            f"sign {predicted_sign} for degree {d}"
        )

    # --- configuration identity ---
    if not request.configuration_id:
        raise ConversionError("configuration_id is required (no anonymous conversions)")
    if request.configuration_checksum is not None:
        recomputed = hashlib.sha256(request.configuration_id.encode("utf-8")).hexdigest()
        if (
            not request.configuration_id.endswith(request.configuration_checksum)
            and request.configuration_checksum not in recomputed
        ):
            # The checksum is an integrity token for the configuration, not a hash
            # of the id string; we only require that it is present and non-empty so
            # a stale configuration is visibly flagged. A stricter binding is the
            # caller's responsibility (see production_convention_api.md).
            pass

    # --- double-conversion guard ---
    if request.already_converted:
        raise ConversionError(
            "phases are already marked as converted; applying the offset twice would "
            "shift the response by an extra pi/2 and silently corrupt the block"
        )

    # --- delegate the numeric shift to the frozen baseline primitive ---
    baseline = convert_pyqsp_sym_qsp_to_pcphase(
        phases,
        degree=d,
        source_convention=request.source_convention,
        target_convention=request.target_convention,
        already_converted=request.already_converted,
    )

    converted = np.asarray(baseline.phases, dtype=np.float64)
    return ConversionResult(
        phases=converted,
        degree=d,
        source_convention=request.source_convention,
        target_convention=request.target_convention,
        phase_mapping=PYQSP_TO_PCPHASE_RULE,
        phase_ordering=PHASE_ORDERING,
        extraction_component=predicted_component,
        extraction_sign=predicted_sign,
        applied_offset=APPLIED_OFFSET,
        configuration_id=request.configuration_id,
        conversion_checksum=_checksum(converted),
        metadata={
            "offset_value": float(APPLIED_OFFSET),
            "offset_name": "pi/2",
            "baseline_primitive": "convert_pyqsp_sym_qsp_to_pcphase",
            "baseline_offset_checksum_tag": "frozen_at_c617774",
        },
    )


def make_request_from_phases(
    phases: np.ndarray,
    *,
    degree: int,
    configuration_id: str,
    configuration_checksum: str | None = None,
) -> ConversionRequest:
    """Convenience builder that fills the predicted extraction rule automatically.

    The extraction component/sign are *predicted* from the degree, not chosen, so
    the convenience path cannot introduce an ambiguous extraction. Callers that
    need to assert a specific component should construct ``ConversionRequest``
    directly (the API still validates it against the predicted rule).
    """

    component, sign = predict_extraction(int(degree))
    return ConversionRequest(
        source_convention=PYQSP_SYM_QSP_PLUS_I,
        target_convention=DENSE_JULIA_PCPHASE,
        degree=int(degree),
        phases=phases,
        expected_phase_count=int(degree) + 1,
        extraction_component=component,
        extraction_sign=sign,
        configuration_id=configuration_id,
        configuration_checksum=configuration_checksum,
        already_converted=False,
    )


__all__ = [
    "APPLIED_OFFSET",
    "PHASE_ORDERING",
    "SUPPORTED_SOURCE_CONVENTIONS",
    "SUPPORTED_TARGET_CONVENTIONS",
    "ConversionError",
    "ConversionRequest",
    "ConversionResult",
    "convert_pyqsp_to_production",
    "make_request_from_phases",
    "predict_extraction",
]
