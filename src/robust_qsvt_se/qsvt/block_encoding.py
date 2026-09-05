from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class BlockEncodingResult:
    source_matrix: np.ndarray
    unitary: np.ndarray
    summary: dict[str, Any]


def spectral_norm_bound(A: ArrayLike, safety_factor: float = 1.0) -> float:
    """Return a positive ``beta`` satisfying ``beta >= ||A||_2``.

    The returned value is suitable for normalizing weighted Jacobian matrices
    before dense block-encoding validation. A zero matrix receives a tiny
    positive bound so downstream normalization remains well-defined.
    """

    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least 1")
    values = _as_matrix(A)
    singular_values = np.linalg.svd(values, compute_uv=False)
    norm = float(singular_values[0]) if singular_values.size else 0.0
    return max(norm * float(safety_factor), np.finfo(float).eps)


def normalize_for_block_encoding(
    H_tilde: ArrayLike,
    beta: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``A = H_tilde / beta`` with ``||A||_2 <= 1``.

    ``beta`` may be supplied when the normalization is fixed by a caller. When
    omitted, the spectral norm is used. Supplied bounds below the matrix
    spectral norm are rejected because they would not define a contraction.
    """

    values = _as_matrix(H_tilde)
    norm = spectral_norm_bound(values)
    beta_value = norm if beta is None else float(beta)
    if beta_value <= 0.0 or not np.isfinite(beta_value):
        raise ValueError("beta must be positive and finite")
    if beta_value < norm * (1.0 - 1.0e-12):
        raise ValueError("beta must be at least the spectral norm of H_tilde")
    return values / beta_value, beta_value


def build_dense_block_encoding(A: ArrayLike, *, tolerance: float = 1.0e-9) -> np.ndarray:
    """Build a dense Julia block-encoding unitary for a contraction.

    For an ``m x n`` matrix ``A`` with ``||A||_2 <= 1``, this returns the
    ``(m+n) x (m+n)`` unitary

    ``[[A, sqrt(I_m - A A*)], [sqrt(I_n - A* A), -A*]]``.

    The encoded block is the top-left ``m x n`` subblock. This is a dense
    validation prototype, not an oracle decomposition for large systems.
    """

    values = _as_matrix(A, dtype=np.complex128)
    singular_values = np.linalg.svd(values, compute_uv=False)
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    if spectral_norm > 1.0 + tolerance:
        raise ValueError(f"matrix spectral norm exceeds 1: {spectral_norm}")

    rows, columns = values.shape
    left_identity = np.eye(rows, dtype=np.complex128)
    right_identity = np.eye(columns, dtype=np.complex128)
    left_defect = _sqrt_psd(left_identity - values @ values.conj().T, tolerance=tolerance)
    right_defect = _sqrt_psd(
        right_identity - values.conj().T @ values,
        tolerance=tolerance,
    )
    return np.block(
        [
            [values, left_defect],
            [right_defect, -values.conj().T],
        ]
    )


def extract_encoded_block(U: ArrayLike, shape: tuple[int, int]) -> np.ndarray:
    """Extract the encoded top-left block with the documented target shape."""

    unitary = _as_matrix(U, dtype=np.complex128)
    rows, columns = _validate_shape(shape)
    if unitary.shape[0] < rows or unitary.shape[1] < columns:
        raise ValueError("unitary is too small for requested encoded block shape")
    return unitary[:rows, :columns]


def canonical_square_block_encoding(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> BlockEncodingResult:
    """Build a canonical block encoding for a square contraction.

    For a square matrix ``A`` with ``||A||_2 <= 1``, this returns

    ``[[A, sqrt(I - A A*)], [sqrt(I - A* A), -A*]]``.

    The top-left block is exactly the normalized research matrix up to
    numerical roundoff. This primitive is intentionally dense and small-scale;
    it is used as the block-encoding primitive inside an explicit QSVT sequence.
    """
    values = np.asarray(matrix, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("canonical block encoding requires a square matrix")
    dimension = int(values.shape[0])
    if not _is_power_of_two(dimension):
        raise ValueError("hardware QSVT prototype requires power-of-two matrix dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix entries must be finite")
    singular_values = np.linalg.svd(values, compute_uv=False)
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    if spectral_norm > 1.0 + tolerance:
        raise ValueError(f"matrix spectral norm exceeds 1: {spectral_norm}")

    identity = np.eye(dimension, dtype=np.complex128)
    left_defect = _sqrt_psd(identity - values @ values.conj().T, tolerance=tolerance)
    right_defect = _sqrt_psd(identity - values.conj().T @ values, tolerance=tolerance)
    unitary = np.block(
        [
            [values, left_defect],
            [right_defect, -values.conj().T],
        ]
    )
    top_left_error = float(np.max(np.abs(unitary[:dimension, :dimension] - values)))
    unitarity_error = float(
        np.max(np.abs(unitary.conj().T @ unitary - np.eye(2 * dimension, dtype=np.complex128)))
    )
    return BlockEncodingResult(
        source_matrix=values,
        unitary=unitary,
        summary={
            "block_encoding_type": "canonical_square_contraction",
            "matrix_dimension": dimension,
            "unitary_dimension": int(unitary.shape[0]),
            "spectral_norm": spectral_norm,
            "top_left_block_error": top_left_error,
            "unitarity_error": unitarity_error,
            "is_unitary": bool(unitarity_error <= 10.0 * tolerance),
            "uses_dense_block_encoding_gate": True,
            "scope_note": (
                "Dense canonical block-encoding primitive for a small research-derived "
                "matrix; not an oracle decomposition for full IEEE systems."
            ),
        },
    )


def validate_block_encoding(
    A: BlockEncodingResult | ArrayLike,
    U: ArrayLike | None = None,
    shape: tuple[int, int] | None = None,
    *,
    beta: float | None = None,
    atol: float | None = None,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Validate a dense block encoding.

    This function accepts the historical ``BlockEncodingResult`` object or the
    newer explicit ``(A, U, shape)`` form used by rectangular engineering demos.
    """

    threshold = float(tolerance if atol is None else atol)
    if isinstance(A, BlockEncodingResult):
        matrix = np.asarray(A.source_matrix, dtype=np.complex128)
        unitary = np.asarray(A.unitary, dtype=np.complex128)
        target_shape = matrix.shape
    else:
        if U is None:
            raise ValueError("U must be supplied when validating an explicit matrix")
        matrix = _as_matrix(A, dtype=np.complex128)
        unitary = _as_matrix(U, dtype=np.complex128)
        target_shape = matrix.shape if shape is None else _validate_shape(shape)
        if target_shape != matrix.shape:
            raise ValueError("shape must match A.shape")

    encoded = extract_encoded_block(unitary, target_shape)
    encoded_block_error = float(np.max(np.abs(encoded - matrix)))
    unitarity_error = float(
        np.max(np.abs(unitary.conj().T @ unitary - np.eye(unitary.shape[0], dtype=np.complex128)))
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    condition = _condition_number(singular_values)
    rank = int(np.linalg.matrix_rank(matrix, tol=threshold))
    passed = bool(encoded_block_error <= threshold and unitarity_error <= threshold)
    return {
        "matrix_shape": [int(target_shape[0]), int(target_shape[1])],
        "beta": None if beta is None else float(beta),
        "encoded_block_error": encoded_block_error,
        "top_left_block_error": encoded_block_error,
        "unitarity_error": unitarity_error,
        "spectral_norm_A": spectral_norm,
        "rank": rank,
        "condition_number": condition,
        "passed": passed,
        "top_left_block_valid": bool(encoded_block_error <= threshold),
        "unitarity_valid": bool(unitarity_error <= threshold),
        "validation_tolerance": threshold,
    }


def _as_matrix(
    values: ArrayLike,
    *,
    dtype: type[np.complex128] | type[np.float64] = np.float64,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=dtype)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must have nonzero dimensions")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix entries must be finite")
    return matrix


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain two dimensions")
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError("shape dimensions must be positive")
    return rows, columns


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(np.max(positive) / np.min(positive))


def _sqrt_psd(matrix: np.ndarray, *, tolerance: float) -> np.ndarray:
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < -10.0 * tolerance:
        raise ValueError(f"defect matrix is not positive semidefinite: {min_eigenvalue}")
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0
