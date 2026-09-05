from __future__ import annotations

from robust_qsvt_se.qsvt.external_backend_sanity import _sanity_candidates
from robust_qsvt_se.qsvt.phase_backend_adapters import (
    VALID_BACKEND_STATUSES,
    available_backend_adapters,
)


def test_phase_backend_adapter_statuses_are_valid() -> None:
    candidate = _sanity_candidates(21)[0]

    for adapter in available_backend_adapters(enable_local_optimization=True):
        result = adapter.synthesize(candidate)
        assert result.status in VALID_BACKEND_STATUSES
        assert result.phase_count == (0 if result.phases is None else len(result.phases))


def test_pyqsp_adapter_accepts_chebyshev_when_available() -> None:
    candidate = _sanity_candidates(21)[0]
    pyqsp = next(
        adapter
        for adapter in available_backend_adapters(enable_local_optimization=False)
        if adapter.backend_name == "pyqsp_sym_qsp"
    )
    result = pyqsp.synthesize(candidate)

    if result.status == "passed_synthesis":
        assert result.input_basis == "chebyshev_T_low_to_high"
        assert result.phase_count == candidate.degree + 1
    else:
        assert result.status == "skipped_backend_unavailable"
