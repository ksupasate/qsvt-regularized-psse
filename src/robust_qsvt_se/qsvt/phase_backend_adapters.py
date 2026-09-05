from __future__ import annotations

import importlib.util
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Protocol

import numpy as np
from scipy.optimize import least_squares

from robust_qsvt_se.qsvt.external_phase_candidates import ExternalPhaseCandidate
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_synthesis import qsp_response, validate_qsvt_polynomial

VALID_BACKEND_STATUSES = {
    "passed_synthesis",
    "failed_synthesis",
    "skipped_backend_unavailable",
    "skipped_unsupported_basis",
    "skipped_unstable_candidate",
    "failed_response_validation",
}


@dataclass(frozen=True, slots=True)
class PhaseBackendResult:
    backend_name: str
    status: str
    phases: list[float] | None
    phase_count: int
    convention: str
    input_basis: str
    error_message: str | None
    metadata: dict[str, Any]


class PhaseBackend(Protocol):
    backend_name: str
    input_basis: str
    response_convention: str

    def synthesize(self, candidate: ExternalPhaseCandidate) -> PhaseBackendResult: ...

    def evaluate_response(
        self,
        points: np.ndarray,
        phases: np.ndarray,
        candidate: ExternalPhaseCandidate,
    ) -> np.ndarray: ...


class PennyLanePolyToAnglesAdapter:
    backend_name = "pennylane_poly_to_angles"
    input_basis = "monomial_power_low_to_high"
    response_convention = "pennylane_rx_pcphase/original/phi/none/real_u00"

    def synthesize(self, candidate: ExternalPhaseCandidate) -> PhaseBackendResult:
        if importlib.util.find_spec("pennylane") is None:
            return _skip(self, "skipped_backend_unavailable", "PennyLane is unavailable")
        if candidate.monomial_coefficients is None:
            return _skip(
                self,
                "skipped_unsupported_basis",
                "candidate has no monomial coefficients",
            )
        try:
            import pennylane as qml  # type: ignore[import-not-found]

            validate_qsvt_polynomial(candidate.monomial_coefficients, parity="odd")
            phases = np.asarray(
                qml.poly_to_angles(
                    candidate.monomial_coefficients,
                    "QSVT",
                    angle_solver="root-finding",
                ),
                dtype=np.float64,
            )
            return PhaseBackendResult(
                backend_name=self.backend_name,
                status="passed_synthesis",
                phases=[float(value) for value in phases],
                phase_count=int(phases.size),
                convention="PennyLane QSVT poly_to_angles",
                input_basis=self.input_basis,
                error_message=None,
                metadata={"version": qml.__version__},
            )
        except Exception as exc:
            return _failed(self, str(exc))

    def evaluate_response(
        self,
        points: np.ndarray,
        phases: np.ndarray,
        candidate: ExternalPhaseCandidate,
    ) -> np.ndarray:
        return pennylane_qsvt_response(
            np.asarray(points, dtype=np.float64),
            np.asarray(phases, dtype=np.float64),
            phase_order="original",
            phase_sign="phi",
            phase_offset_rule="none",
            signal_operator_convention="pennylane_rx_pcphase",
            response_component="real_u00",
        )


class PyQspSymQspAdapter:
    backend_name = "pyqsp_sym_qsp"
    input_basis = "chebyshev_T_low_to_high"
    response_convention = "pyqsp_sym_qsp/Wx/x/imag"

    def synthesize(self, candidate: ExternalPhaseCandidate) -> PhaseBackendResult:
        if importlib.util.find_spec("pyqsp") is None:
            return _skip(self, "skipped_backend_unavailable", "pyqsp is unavailable")
        if candidate.chebyshev_coefficients is None:
            return _skip(
                self,
                "skipped_unsupported_basis",
                "candidate has no Chebyshev coefficients",
            )
        try:
            from pyqsp.angle_sequence import (  # type: ignore[import-not-found]
                QuantumSignalProcessingPhases,
            )

            buffer = StringIO()
            with redirect_stdout(buffer):
                result = QuantumSignalProcessingPhases(
                    np.asarray(candidate.chebyshev_coefficients, dtype=np.float64),
                    method="sym_qsp",
                    chebyshev_basis=True,
                )
            phases = np.asarray(result[0], dtype=np.float64)
            return PhaseBackendResult(
                backend_name=self.backend_name,
                status="passed_synthesis",
                phases=[float(value) for value in phases],
                phase_count=int(phases.size),
                convention="pyqsp symmetric QSP full phases",
                input_basis=self.input_basis,
                error_message=None,
                metadata={
                    "version": _version("pyqsp"),
                    "reduced_phase_count": len(result[1]),
                    "parity": int(result[2]),
                    "synthesis_log": buffer.getvalue(),
                },
            )
        except Exception as exc:
            return _failed(self, str(exc))

    def evaluate_response(
        self,
        points: np.ndarray,
        phases: np.ndarray,
        candidate: ExternalPhaseCandidate,
    ) -> np.ndarray:
        from pyqsp.response import ComputeQSPResponse  # type: ignore[import-not-found]

        response = ComputeQSPResponse(
            np.asarray(points, dtype=np.float64),
            np.asarray(phases, dtype=np.float64),
            signal_operator="Wx",
            measurement="x",
            sym_qsp=True,
        )["pdat"]
        return np.imag(response)


class QspPackAdapter:
    backend_name = "qsppack_if_available"
    input_basis = "unavailable"
    response_convention = "unavailable"

    def synthesize(self, candidate: ExternalPhaseCandidate) -> PhaseBackendResult:
        return _skip(
            self,
            "skipped_backend_unavailable",
            "No directly callable Python QSPPACK backend is available",
        )

    def evaluate_response(
        self,
        points: np.ndarray,
        phases: np.ndarray,
        candidate: ExternalPhaseCandidate,
    ) -> np.ndarray:
        raise RuntimeError("QSPPACK response evaluator is unavailable")


class LocalOptimizationQspAdapter:
    backend_name = "local_optimization_qsp"
    input_basis = "function_values"
    response_convention = "repository_qsp_response/real_u00"

    def __init__(
        self,
        *,
        max_nfev: int = 2000,
        seed: int = 123,
        enabled: bool = True,
    ) -> None:
        self.max_nfev = int(max_nfev)
        self.seed = int(seed)
        self.enabled = bool(enabled)

    def synthesize(self, candidate: ExternalPhaseCandidate) -> PhaseBackendResult:
        if not self.enabled:
            return _skip(self, "skipped_backend_unavailable", "local optimizer disabled")
        if importlib.util.find_spec("scipy") is None:
            return _skip(self, "skipped_backend_unavailable", "SciPy is unavailable")
        if not candidate.supports_function_values:
            return _skip(self, "skipped_unsupported_basis", "candidate lacks function samples")
        grid = np.asarray(candidate.full_domain_grid, dtype=np.float64)
        target = np.asarray(candidate.full_domain_target, dtype=np.float64)
        degree = int(candidate.degree)
        rng = np.random.default_rng(self.seed)
        initializations = [
            ("zeros", np.zeros(degree + 1, dtype=np.float64)),
            ("small_random", rng.normal(0.0, 0.05, size=degree + 1)),
        ]
        start = time.perf_counter()
        best: tuple[float, Any, str] | None = None
        try:
            for label, initial in initializations:
                result = least_squares(
                    lambda phases: qsp_response(grid, phases) - target,
                    initial,
                    max_nfev=self.max_nfev,
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-10,
                )
                max_error = float(np.max(np.abs(qsp_response(grid, result.x) - target)))
                if best is None or max_error < best[0]:
                    best = (max_error, result, label)
            if best is None:
                return _failed(self, "local optimization did not run")
            max_error, result, label = best
            phases = np.asarray(result.x, dtype=np.float64)
            return PhaseBackendResult(
                backend_name=self.backend_name,
                status="passed_synthesis",
                phases=[float(value) for value in phases],
                phase_count=int(phases.size),
                convention="optimization-based scalar QSP phases",
                input_basis=self.input_basis,
                error_message=None,
                metadata={
                    "optimizer": "scipy.optimize.least_squares",
                    "initialization": label,
                    "iterations": int(result.nfev),
                    "success": bool(result.success),
                    "final_loss": float(result.cost),
                    "max_error": max_error,
                    "runtime_seconds": float(time.perf_counter() - start),
                    "caveat": (
                        "Optimization-based scalar phase-response validation, not "
                        "certified theorem-level phase synthesis."
                    ),
                },
            )
        except Exception as exc:
            return _failed(self, str(exc))

    def evaluate_response(
        self,
        points: np.ndarray,
        phases: np.ndarray,
        candidate: ExternalPhaseCandidate,
    ) -> np.ndarray:
        return qsp_response(
            np.asarray(points, dtype=np.float64),
            np.asarray(phases, dtype=np.float64),
        )


def available_backend_adapters(*, enable_local_optimization: bool = False) -> list[PhaseBackend]:
    return [
        PennyLanePolyToAnglesAdapter(),
        PyQspSymQspAdapter(),
        QspPackAdapter(),
        LocalOptimizationQspAdapter(enabled=enable_local_optimization),
    ]


def _skip(adapter: Any, status: str, message: str) -> PhaseBackendResult:
    return PhaseBackendResult(
        backend_name=adapter.backend_name,
        status=status,
        phases=None,
        phase_count=0,
        convention=getattr(adapter, "response_convention", ""),
        input_basis=getattr(adapter, "input_basis", ""),
        error_message=message,
        metadata={},
    )


def _failed(adapter: Any, message: str) -> PhaseBackendResult:
    return PhaseBackendResult(
        backend_name=adapter.backend_name,
        status="failed_synthesis",
        phases=None,
        phase_count=0,
        convention=getattr(adapter, "response_convention", ""),
        input_basis=getattr(adapter, "input_basis", ""),
        error_message=message,
        metadata={},
    )


def _version(package: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return ""
