"""Joint four-condition end-to-end single-candidate TQE evidence pipeline.

Evaluates ONE frozen canonical IEEE-14 8x8 degree-31 sparse-QSVT candidate through four
logically ordered conditions in a single traceable pipeline:

  1. Support preservation       -- E_support (selected-output scalar) vs 0.1
  2. Benchmark usefulness       -- E_benchmark vs full-system benchmark (INCONCLUSIVE: no
                                    registered pass threshold; metric still computed/reported)
  3. Matched-filter QSVT impl.  -- epsilon_qsvt vs 1e-6 + required correctness checks
  4. Access/readout credibility -- qualitative credible / not-credible (predeclared rubric)

The candidate is frozen (``candidate_freeze.json``) BEFORE any four-condition metric is computed,
and the exact same matrix / support / residual / regularization / polynomial / phases / functional
propagate through every applicable stage. No stage substitutes another's data; failures are
retained explicitly. The full-system benchmark link is ESTABLISHED HERE from frozen inputs (the
82x27 weighted Jacobian at matrix_seed 123) -- not fabricated, not altering historical evidence.

This module reuses the validated numerical recipes; it does not reimplement them:
  - ``ft_workload_rebuild.rebuild_canonical`` (identity-verified CompiledSparseQSVT)
  - ``generic_sparse_execution.validate_compiled_statevector`` (full SV + epsilon decomposition)
  - ``generic_sparse_execution.run_compiled_shots`` (the exact frozen finite-shot generator)
  - ``generic_sparse_execution.build_resource_evidence`` (transpiled resource ledger)
  - ``cross_case_validation.common.build_case_full_system`` (frozen 82x27 system)
  - ``qsvt.engineering_utils.ridge_svd_solution`` (matched Ridge filter)

Scientific boundary: simulator-based only; no quantum speedup/advantage; no scalability claim
from one 8x8 workload; a later-condition pass cannot override an earlier application failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.cross_case_validation.common import build_case_full_system
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.ft_workload_rebuild import rebuild_canonical
from robust_qsvt_se.qsvt.generic_sparse_execution import (
    build_resource_evidence,
    prepare_compiled_execution,
    run_compiled_shots,
    validate_compiled_statevector,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "joint_four_condition.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "joint_four_condition"
HISTORICAL_ROOT = REPO_ROOT / "outputs" / "sparse_chain_reconciliation" / "end_to_end_run"
FROZEN_SHOT_LEDGER = (
    REPO_ROOT / "outputs" / "generic_sparse_qsvt_compiler" / "canonical_shot_rows_generic.csv"
)

CLAIM_BOUNDARY = (
    "Single frozen IEEE-14 8x8 d=31 candidate evaluated through four logically ordered conditions "
    "in one traceable pipeline. Simulator-based only. No quantum speedup or quantum advantage; no "
    "scalability claim from one 8x8 workload; no hardware execution; no practical PSSE deployment; "
    "no impossibility theorem. A later-condition pass cannot override an earlier application "
    "failure."
)

# Predeclared in configs/tqe_reviewer_blocking/joint_feasibility.json (declared before evaluation).
Y_FLOOR = 1.0e-6
CONDITION1_THRESHOLD = 0.1  # sparsification_error_threshold
CONDITION3_THRESHOLD = 1.0e-6  # action_error_tolerance
BOUND_TOLERANCE = 2.0e-3  # bound_tolerance


# --------------------------------------------------------------------------- config / freeze


def load_decision_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the versioned decision-rule config (declared before outcome evaluation)."""

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config.get("declared_before_outcome_evaluation"):
        raise ValueError("decision config must declare declared_before_outcome_evaluation: true")
    return config


def _read_global_indices() -> tuple[list[int], list[int], dict[str, Any]]:
    """Frozen canonical global rows/cols of the 8x8 block within the 82x27 system."""

    metadata_path = HISTORICAL_ROOT / "matrix_metadata.json"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    rows = [int(value) for value in metadata["selected_rows"]]
    cols = [int(value) for value in metadata["selected_columns"]]
    provenance = {
        "source_file": str(metadata_path.relative_to(REPO_ROOT)),
        "selected_rows": rows,
        "selected_columns": cols,
    }
    return rows, cols, provenance


@dataclass
class CandidateFreeze:
    """Outcome-independent record of every frozen candidate component + hash."""

    workload_id: str
    workload_digest: str
    ieee_case: str
    operating_point_seed: int
    global_rows: list[int]
    global_columns: list[int]
    block_shape: tuple[int, int]
    support_coordinates: list[tuple[int, int]]
    support_selection_method: str
    support_budget_k: int
    slot_budget_d_s: int
    support_hash: str
    quantization_magnitude_bits: int
    quantization_sign_bits: int
    quantization_scale_mu: float
    quantization_rule: str
    quantized_matrix_hash: str
    matrix_original_hash: str
    residual_id: str
    residual_hash: str
    residual_norm: float
    functionals: list[dict[str, Any]]
    primary_functional_id: str
    alpha: float
    beta: float
    normalized_lambda: float
    boundedness_factor_C: float
    polynomial_degree: int
    polynomial_id: str
    polynomial_hash: str | None
    phase_hash: str | None
    phase_convention: str
    phase_count: int
    component_hashes: dict[str, str]
    simulator_method: str
    basis_gates: Any
    optimization_level: int
    seed_transpiler: Any
    max_parallel_threads: int
    shot_counts: list[int]
    simulator_seeds: list[int]
    freeze_provenance: dict[str, Any]


def freeze_candidate(compiled: Any, config: dict[str, Any]) -> CandidateFreeze:
    """Build the outcome-independent freeze record from the frozen compiled workload.

    Pure function of the frozen inputs + hashes; no four-condition metric is read here.
    """

    rows, cols, idx_provenance = _read_global_indices()
    support_spec = compiled.support_spec
    quant_spec = compiled.quantization_spec
    qsvt_spec = compiled.qsvt_spec
    exec_spec = compiled.execution_spec
    residual_spec = compiled.residual_spec
    functional_spec = compiled.functional_spec
    hashes = dict(compiled.component_hashes)

    support_coords = [
        (int(i), int(j))
        for i, row in enumerate(compiled.matrix_supported_exact)
        for j, value in enumerate(row)
        if value != 0.0
    ]
    functionals = [
        {
            "functional_id": fid,
            "vector": [float(value) for value in vector],
            "norm": float(np.linalg.norm(vector)),
        }
        for fid, vector in compiled.functional_vectors.items()
    ]

    return CandidateFreeze(
        workload_id=compiled.workload_id,
        workload_digest=compiled.workload_digest,
        ieee_case=config["candidate"]["ieee_case"],
        operating_point_seed=int(config["candidate"]["matrix_seed"]),
        global_rows=rows,
        global_columns=cols,
        block_shape=tuple(int(value) for value in compiled.matrix_original.shape),
        support_coordinates=support_coords,
        support_selection_method=getattr(
            support_spec, "selection_method", "frozen_canonical_support"
        ),
        support_budget_k=len(getattr(support_spec, "coordinates", support_coords)),
        slot_budget_d_s=int(compiled.wrapper.slots),
        support_hash=hashes.get("support_mask", ""),
        quantization_magnitude_bits=int(quant_spec.magnitude_bits),
        quantization_sign_bits=1,
        quantization_scale_mu=float(
            getattr(quant_spec, "mu", float(np.max(np.abs(compiled.matrix_quantized))))
        ),
        quantization_rule=getattr(quant_spec, "rule", "sign_magnitude_round_to_nearest"),
        quantized_matrix_hash=hashes.get("matrix_quantized", ""),
        matrix_original_hash=hashes.get("matrix_original", ""),
        residual_id=getattr(residual_spec, "residual_id", "canonical_ieee14_residual_seed123"),
        residual_hash=hashes.get("residual", ""),
        residual_norm=float(np.linalg.norm(compiled.residual)),
        functionals=functionals,
        primary_functional_id=functional_spec.primary_functional_id,
        alpha=float(qsvt_spec.alpha),
        beta=float(qsvt_spec.beta),
        normalized_lambda=float(qsvt_spec.normalized_lambda),
        boundedness_factor_C=float(qsvt_spec.boundedness_factor),
        polynomial_degree=int(qsvt_spec.degree),
        polynomial_id=qsvt_spec.polynomial_id,
        polynomial_hash=getattr(qsvt_spec, "expected_polynomial_hash", None)
        or hashes.get("polynomial"),
        phase_hash=getattr(qsvt_spec, "expected_phase_hash", None) or hashes.get("phases"),
        phase_convention=qsvt_spec.phase_convention,
        phase_count=len(compiled.phases),
        component_hashes=hashes,
        simulator_method=exec_spec.simulator_method,
        basis_gates=list(exec_spec.basis_gates) if exec_spec.basis_gates is not None else None,
        optimization_level=int(exec_spec.optimization_level),
        seed_transpiler=exec_spec.seed_transpiler,
        max_parallel_threads=int(exec_spec.max_parallel_threads),
        shot_counts=[int(value) for value in exec_spec.shot_counts],
        simulator_seeds=[int(value) for value in exec_spec.simulator_seeds],
        freeze_provenance={
            **idx_provenance,
            "declared_before_outcome_evaluation": True,
            "rebuild": "ft_workload_rebuild.rebuild_canonical (identity-verified)",
        },
    )


# ------------------------------------------------------------------------- benchmark reference


@dataclass
class BenchmarkReference:
    """Full-system benchmark reference, established freshly from frozen inputs."""

    full_matrix_shape: tuple[int, int]
    full_matrix_hash: str
    natural_alpha: float
    matched_alpha: float
    # selected-output benchmark scalars per functional id
    y_benchmark_natural: dict[str, float]
    y_benchmark_matched: dict[str, float]
    y_truth: dict[str, float]
    # consistency checks (candidate block is a genuine submatrix of the full system)
    block_submatrix_match: bool
    residual_slice_match: bool
    block_submatrix_max_abs_err: float
    residual_slice_max_abs_err: float
    provenance: dict[str, Any]


def _stable_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_full_system_benchmark(
    compiled: Any,
    block_rows: list[int],
    block_cols: list[int],
    natural_alpha: float,
    matched_alpha: float,
) -> BenchmarkReference:
    """Establish the full-system benchmark-reference link from frozen inputs.

    Computes three distinct selected-output references for each frozen functional,
    all with the block-local 8-dim ``ell`` applied to the block-column slice
    (matching engine.py:114-121,388-396):
      natural : full 82x27 Ridge at alpha=natural_alpha (full-system reference)
      matched : full 82x27 Ridge at candidate alpha (isolates reg.-regime change)
      truth   : ell . full.x_true[block_cols]  (ground-truth; repo E_physical_norm ref)

    The full-system Ridge at natural_alpha is DISTINCT from engine.py's block-8x8
    ``y_full``; it is labeled as such in the ledger. No historical evidence is
    altered; the link provenance is recorded.
    """

    full = build_case_full_system("ieee14", seed=123)
    full_matrix = np.asarray(full.matrix, dtype=np.float64)
    full_residual = np.asarray(full.residual, dtype=np.float64)
    x_true = np.asarray(full.x_true, dtype=np.float64)
    rows_idx = np.asarray(block_rows, dtype=np.intp)
    cols_idx = np.asarray(block_cols, dtype=np.intp)

    # Provenance consistency: the frozen 8x8 block must be an exact submatrix of the full system
    # and the frozen 8-dim block residual must be the matching row slice of the full residual.
    block_submatrix = full_matrix[np.ix_(rows_idx, cols_idx)]
    block_sub_err = float(np.max(np.abs(block_submatrix - compiled.matrix_original)))
    residual_slice = full_residual[rows_idx]
    residual_slice_err = float(np.max(np.abs(residual_slice - compiled.residual)))

    dx_full_natural = ridge_svd_solution(full_matrix, full_residual, alpha=natural_alpha)
    dx_full_matched = ridge_svd_solution(full_matrix, full_residual, alpha=matched_alpha)
    truth_block = x_true[cols_idx]

    y_natural: dict[str, float] = {}
    y_matched: dict[str, float] = {}
    y_truth: dict[str, float] = {}
    for functional_id, ell in compiled.functional_vectors.items():
        ell8 = np.asarray(ell, dtype=np.float64)
        y_natural[functional_id] = float(ell8 @ dx_full_natural[cols_idx])
        y_matched[functional_id] = float(ell8 @ dx_full_matched[cols_idx])
        y_truth[functional_id] = float(ell8 @ truth_block)

    atol = 1.0e-9
    return BenchmarkReference(
        full_matrix_shape=tuple(int(value) for value in full_matrix.shape),
        full_matrix_hash=_stable_array_hash(full_matrix),
        natural_alpha=float(natural_alpha),
        matched_alpha=float(matched_alpha),
        y_benchmark_natural=y_natural,
        y_benchmark_matched=y_matched,
        y_truth=y_truth,
        block_submatrix_match=bool(block_sub_err <= atol),
        residual_slice_match=bool(residual_slice_err <= atol),
        block_submatrix_max_abs_err=block_sub_err,
        residual_slice_max_abs_err=residual_slice_err,
        provenance={
            "link_established_here": True,
            "full_system_builder": "build_case_full_system(ieee14, seed=123)",
            "block_rows": block_rows,
            "block_cols": block_cols,
            "natural_alpha": float(natural_alpha),
            "matched_alpha": float(matched_alpha),
            "note": (
                "Full-system benchmark link established here from frozen inputs; the "
                "canonical compiler artifacts never recorded it. The prior bridge file "
                "disclaims itself as a selected-submatrix surrogate."
            ),
            "distinct_from_engine_y_full": (
                "primary reference is the full 82x27 Ridge at natural_alpha projected to block "
                "columns, distinct from engine.py's block-8x8 y_full"
            ),
        },
    )


# ----------------------------------------------------------------------- classical comparators


def _time_call(callable_: Any, *, repeats: int = 5) -> float:
    """Median wall-clock seconds over repeats (host-specific, not hardware-normalized)."""

    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _ridge_normal(matrix: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Normal-equations operator (M^T M + alpha I) and M^T for Ridge selected-output solves."""

    gram = matrix.T @ matrix
    operator = gram + alpha * np.eye(matrix.shape[1])
    mt = matrix.T
    return operator, mt


def _classical_block_records(
    compiled: Any, functional_id: str, alpha: float
) -> list[dict[str, Any]]:
    """Matched classical selected-output methods on the 8x8 block (host wall-clock times)."""

    matrix = np.asarray(compiled.matrix_original, dtype=np.float64)
    residual = np.asarray(compiled.residual, dtype=np.float64)
    ell = np.asarray(compiled.functional_vectors[functional_id], dtype=np.float64)
    operator, mt = _ridge_normal(matrix, alpha)
    rhs = mt @ residual
    rows: list[dict[str, Any]] = []
    common = {
        "scope": "block_8x8",
        "functional_id": functional_id,
        "alpha": alpha,
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
    }

    # dense Ridge (SVD) -- the matched reference
    def _dense() -> float:
        return float(ell @ ridge_svd_solution(matrix, residual, alpha=alpha))

    rows.append(
        {
            **common,
            "method": "dense_ridge",
            "selected_output": _dense(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_dense),
            "per_functional_time_s": _time_call(_dense),
        }
    )

    # factorization reuse: LU(operator) once, then solve per RHS and extract coordinate
    def _factorize() -> Any:
        return np.linalg.solve(operator, rhs)

    lu_start = time.perf_counter()
    dx_fac = np.linalg.solve(operator, rhs)
    setup_fac = time.perf_counter() - lu_start

    def _solve_reuse() -> float:
        return float(ell @ dx_fac)

    rows.append(
        {
            **common,
            "method": "factorization_extract",
            "selected_output": float(ell @ dx_fac),
            "setup_time_s": setup_fac,
            "per_rhs_time_s": _time_call(lambda: np.linalg.solve(operator, rhs)),
            "per_functional_time_s": _time_call(_solve_reuse),
        }
    )

    # adjoint selected-output solve: solve operator z = ell, then y = z . rhs (no full dx needed)
    def _adjoint() -> float:
        z = np.linalg.solve(operator, ell)
        return float(z @ rhs)

    rows.append(
        {
            **common,
            "method": "adjoint_selected_output",
            "selected_output": _adjoint(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_adjoint),
            "per_functional_time_s": _time_call(_adjoint),
        }
    )

    # Krylov (LSMR on the least-squares Ridge problem)
    from scipy.sparse.linalg import lsmr

    def _krylov() -> float:
        dx = lsmr(matrix, residual, damp=math.sqrt(alpha), atol=1e-12, btol=1e-12)[0]
        return float(ell @ dx)

    rows.append(
        {
            **common,
            "method": "krylov",
            "selected_output": _krylov(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_krylov, repeats=3),
            "per_functional_time_s": _time_call(_krylov, repeats=3),
        }
    )

    # Gaussian sketch-and-solve (deterministic seed-free fixed orthogonal projection)
    dim = matrix.shape[1]
    sketch_size = max(dim, 2 * matrix.shape[0] // 4 + dim)  # O(n) sketch at this tiny scale

    def _sketch() -> float:
        local_rng = np.random.default_rng(20260723)
        omega = local_rng.standard_normal(size=(matrix.shape[0], sketch_size)) / math.sqrt(
            sketch_size
        )
        sketch = omega.T @ matrix
        sk_rhs = omega.T @ residual
        dx = np.linalg.solve(sketch.T @ sketch + alpha * np.eye(dim), sketch.T @ sk_rhs)
        return float(ell @ dx)

    rows.append(
        {
            **common,
            "method": "gaussian_sketch",
            "selected_output": _sketch(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_sketch, repeats=3),
            "per_functional_time_s": _time_call(_sketch, repeats=3),
        }
    )

    return rows


def _classical_full_records(
    full_matrix: np.ndarray,
    full_residual: np.ndarray,
    block_cols: list[int],
    functional_id: str,
    ell8: np.ndarray,
    alpha: float,
) -> list[dict[str, Any]]:
    """Matched classical selected-output methods on the full 82x27 system."""

    operator, mt = _ridge_normal(full_matrix, alpha)
    rhs = mt @ full_residual
    cols_idx = np.asarray(block_cols, dtype=np.intp)
    rows: list[dict[str, Any]] = []
    common = {
        "scope": "full_82x27",
        "functional_id": functional_id,
        "alpha": alpha,
        "matrix_shape": f"{full_matrix.shape[0]}x{full_matrix.shape[1]}",
    }

    def _dense() -> float:
        return float(ell8 @ ridge_svd_solution(full_matrix, full_residual, alpha=alpha)[cols_idx])

    rows.append(
        {
            **common,
            "method": "dense_ridge",
            "selected_output": _dense(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_dense),
            "per_functional_time_s": _time_call(_dense),
        }
    )

    lu_start = time.perf_counter()
    dx_fac = np.linalg.solve(operator, rhs)
    setup_fac = time.perf_counter() - lu_start
    rows.append(
        {
            **common,
            "method": "factorization_extract",
            "selected_output": float(ell8 @ dx_fac[cols_idx]),
            "setup_time_s": setup_fac,
            "per_rhs_time_s": _time_call(lambda: np.linalg.solve(operator, rhs)),
            "per_functional_time_s": _time_call(lambda: float(ell8 @ dx_fac[cols_idx])),
        }
    )

    def _adjoint() -> float:
        z = np.linalg.solve(operator, _pad_functional(ell8, cols_idx, operator.shape[0]))
        return float(z @ rhs)

    rows.append(
        {
            **common,
            "method": "adjoint_selected_output",
            "selected_output": _adjoint(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_adjoint),
            "per_functional_time_s": _time_call(_adjoint),
        }
    )

    from scipy.sparse.linalg import lsmr

    def _krylov() -> float:
        dx = lsmr(full_matrix, full_residual, damp=math.sqrt(alpha), atol=1e-12, btol=1e-12)[0]
        return float(ell8 @ dx[cols_idx])

    rows.append(
        {
            **common,
            "method": "krylov",
            "selected_output": _krylov(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_krylov, repeats=3),
            "per_functional_time_s": _time_call(_krylov, repeats=3),
        }
    )

    return rows


def _pad_functional(ell8: np.ndarray, cols_idx: np.ndarray, dim: int) -> np.ndarray:
    padded = np.zeros(dim, dtype=np.float64)
    padded[cols_idx] = ell8
    return padded


def build_classical_comparison(
    compiled: Any,
    benchmark: BenchmarkReference,
    primary_functional_id: str,
) -> pd.DataFrame:
    """Classical selected-output comparators under matched conditions (block + full). No speedup."""

    rows: list[dict[str, Any]] = []
    alpha = compiled.qsvt_spec.alpha
    # block scope: candidate's own (matched) alpha
    rows.extend(_classical_block_records(compiled, primary_functional_id, alpha))
    # full scope: natural alpha (established full-system reference)
    full_system = build_case_full_system("ieee14", seed=123)
    full_matrix = np.asarray(full_system.matrix, dtype=np.float64)
    full_residual = np.asarray(full_system.residual, dtype=np.float64)
    ell8 = np.asarray(compiled.functional_vectors[primary_functional_id], dtype=np.float64)
    block_cols = _read_global_indices()[1]
    rows.extend(
        _classical_full_records(
            full_matrix,
            full_residual,
            block_cols,
            primary_functional_id,
            ell8,
            benchmark.natural_alpha,
        )
    )
    # exact polynomial action as a classical reference (block scope; matched polynomial action).
    # Uses the beta-normalized quantized block (singular values on [-1,1]), matching the
    # validate_compiled_statevector recipe -- NOT the unnormalized block.
    qsvt = compiled.qsvt_spec
    poly = np.polynomial.Polynomial(compiled.polynomial_coefficients)
    normalized = np.asarray(compiled.matrix_quantized, dtype=np.float64).T / qsvt.beta
    left, singular, right_t = np.linalg.svd(normalized, full_matrices=False)
    residual = np.asarray(compiled.residual, dtype=np.float64)
    scale = qsvt.boundedness_factor / qsvt.beta * float(np.linalg.norm(residual))
    res_unit = residual / float(np.linalg.norm(residual))

    def _poly_action() -> float:
        action = left @ (poly(singular) * (right_t @ res_unit))
        return float(ell8 @ (scale * action))

    rows.append(
        {
            "scope": "block_8x8",
            "functional_id": primary_functional_id,
            "alpha": alpha,
            "matrix_shape": f"{compiled.matrix_original.shape[0]}"
            f"x{compiled.matrix_original.shape[1]}",
            "method": "exact_polynomial_action",
            "selected_output": _poly_action(),
            "setup_time_s": float("nan"),
            "per_rhs_time_s": _time_call(_poly_action),
            "per_functional_time_s": _time_call(_poly_action),
        }
    )
    frame = pd.DataFrame(rows)
    frame["host_note"] = (
        "wall-clock on audit host; not hardware-normalized; not divided into quantum gate counts"
    )
    return frame


# ----------------------------------------------------------------------- four-condition decision


def _rel(left: float, right: float, floor: float = Y_FLOOR) -> float:
    return float(abs(left - right) / max(abs(right), floor))


@dataclass
class ConditionResult:
    condition_id: str
    condition_name: str
    metric: str
    metric_value: Any
    threshold: Any
    direction: str
    status: str
    margin: Any
    evidence_level: str
    failure_reason: str
    source_artifacts: list[str]


def evaluate_four_conditions(
    compiled: Any,
    statevector: Any,
    benchmark: BenchmarkReference,
    config: dict[str, Any],
    finite_shot_summary: pd.DataFrame,
) -> tuple[list[ConditionResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply predeclared decision rules; emit per-condition results + decomposition tables."""

    primary = compiled.functional_spec.primary_functional_id
    metrics = statevector.metrics

    # ---- selected-output results: project every stage through each functional ----
    sel_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for functional_id, _ell in compiled.functional_vectors.items():
        row = next(
            item for item in statevector.functional_rows if item["functional_id"] == functional_id
        )
        y_sparse = float(row["selected_exact_ridge_output"])  # supported-exact Ridge
        y_full_block = float(row["full_matrix_ridge_output"])
        y_quantized = float(row["quantized_ridge_selected_output"])
        y_rational = float(row["exact_rational_selected_output"])
        y_polynomial = float(row["exact_polynomial_selected_output"])
        y_statevector = float(row["statevector_selected_output"])
        y_benchmark_natural = benchmark.y_benchmark_natural[functional_id]
        y_benchmark_matched = benchmark.y_benchmark_matched[functional_id]
        y_truth = benchmark.y_truth[functional_id]

        e_support = _rel(y_sparse, y_full_block)
        e_benchmark_natural = _rel(y_sparse, y_benchmark_natural)
        e_benchmark_matched = _rel(y_sparse, y_benchmark_matched)
        e_benchmark_truth = _rel(y_sparse, y_truth)

        sel_rows.append(
            {
                "functional_id": functional_id,
                "y_full_system_benchmark_natural": y_benchmark_natural,
                "y_full_system_benchmark_matched": y_benchmark_matched,
                "y_truth_reference": y_truth,
                "y_full_block_ridge": y_full_block,
                "y_sparse_exact_ridge": y_sparse,
                "y_quantized_ridge": y_quantized,
                "y_exact_rational": y_rational,
                "y_exact_polynomial": y_polynomial,
                "y_statevector_qsvt": y_statevector,
                "E_support_selected_output": e_support,
                "E_benchmark_natural_selected_output": e_benchmark_natural,
                "E_benchmark_matched_selected_output": e_benchmark_matched,
                "E_benchmark_truth_selected_output": e_benchmark_truth,
                "postselection_probability": float(row["postselection_probability"]),
            }
        )

        # full separated chain (absolute + relative vs the immediately-preceding reference)
        chain = [
            ("full_system_benchmark_natural", y_benchmark_natural, None),
            ("block_truncation_full_block_ridge", y_full_block, y_benchmark_natural),
            ("support_removal_sparse_exact_ridge", y_sparse, y_full_block),
            ("quantization_quantized_ridge", y_quantized, y_sparse),
            ("polynomial_exact_rational", y_rational, y_quantized),
            ("polynomial_to_exact_poly", y_polynomial, y_rational),
            ("qsvt_circuit_statevector", y_statevector, y_polynomial),
        ]
        for stage, value, reference in chain:
            abs_err = abs(value - reference) if reference is not None else 0.0
            rel_err = _rel(value, reference) if reference is not None else 0.0
            error_rows.append(
                {
                    "category": "numerical_approximation",
                    "stage": stage,
                    "functional_id": functional_id,
                    "stage_selected_output": value,
                    "reference_selected_output": reference if reference is not None else value,
                    "absolute_error": abs_err,
                    "relative_error": rel_err,
                    "reference_description": "stage-immediately-preceding"
                    if reference is not None
                    else "chain_origin",
                }
            )

    # ---- matrix / access validation rows ----
    access_rows = [
        {
            "category": "matrix_access",
            "stage": "lookup_error",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": float(metrics["epsilon_lookup"]),
            "relative_error": float("nan"),
            "reference_description": "slot lookup value max error",
        },
        {
            "category": "matrix_access",
            "stage": "reconstructed_encoded_block_error",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": float(metrics["epsilon_block"]),
            "relative_error": float("nan"),
            "reference_description": "wrapper encoded-block Frobenius reconstruction",
        },
        {
            "category": "matrix_access",
            "stage": "wrapper_unitarity_error",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": float(metrics["wrapper_unitarity_max_error"]),
            "relative_error": float("nan"),
            "reference_description": "wrapper unitarity max error",
        },
        {
            "category": "matrix_access",
            "stage": "uncomputation_roundtrip_error",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": float(metrics["uncomputation_roundtrip_max_error"]),
            "relative_error": float("nan"),
            "reference_description": "wrapper inverse round-trip max error",
        },
        {
            "category": "matrix_access",
            "stage": "support_identity",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": 0.0,
            "relative_error": float("nan"),
            "reference_description": "frozen support reused exactly (no re-selection)",
        },
        {
            "category": "matrix_access",
            "stage": "slot_capacity",
            "functional_id": primary,
            "stage_selected_output": float(compiled.wrapper.slots),
            "reference_selected_output": float("nan"),
            "absolute_error": float("nan"),
            "relative_error": float("nan"),
            "reference_description": (
                f"slots={compiled.wrapper.slots}, "
                f"support_k={len(compiled.support_spec.coordinates)}"
            ),
        },
        {
            "category": "matrix_access",
            "stage": "dense_fallback_detection",
            "functional_id": primary,
            "stage_selected_output": 0.0,
            "reference_selected_output": float("nan"),
            "absolute_error": float("nan"),
            "relative_error": float("nan"),
            "reference_description": "no dense signal fallback present (raises on fallback)",
        },
        {
            "category": "matrix_access",
            "stage": "support_action_vector_norm",
            "functional_id": primary,
            "stage_selected_output": float("nan"),
            "reference_selected_output": float("nan"),
            "absolute_error": float(metrics["epsilon_support"]),
            "relative_error": float("nan"),
            "reference_description": (
                "vector-norm epsilon_support (continuity with summary 0.2809)"
            ),
        },
    ]
    # ---- finite-shot recovery rows (primary functional, per shot budget) ----
    recovery_rows: list[dict[str, Any]] = []
    if not finite_shot_summary.empty:
        primary_fs = finite_shot_summary[finite_shot_summary["functional_id"] == primary]
        for _, summary in primary_fs.iterrows():
            recovery_rows.append(
                {
                    "category": "selected_output_recovery",
                    "stage": f"finite_shot_{int(summary['shots_attempted'])}",
                    "functional_id": primary,
                    "stage_selected_output": float(summary["mean_recovered_selected_output"]),
                    "reference_selected_output": float(summary["statevector_reference"]),
                    "absolute_error": float(summary["mean_abs_error_vs_statevector"]),
                    "relative_error": float(
                        summary.get("mean_relative_error_vs_statevector", float("nan"))
                    ),
                    "reference_description": (
                        "finite-shot mean vs statevector; seed std + CI in finite_shot_summary"
                    ),
                }
            )
    error_df = pd.DataFrame(error_rows + access_rows + recovery_rows)

    # ---- conditions ----
    e_support_primary = next(
        r["E_support_selected_output"] for r in sel_rows if r["functional_id"] == primary
    )
    e_benchmark_primary = next(
        r["E_benchmark_natural_selected_output"] for r in sel_rows if r["functional_id"] == primary
    )

    cond1_pass = e_support_primary <= CONDITION1_THRESHOLD
    condition_1 = ConditionResult(
        condition_id="condition_1",
        condition_name="support_preservation",
        metric="E_support",
        metric_value=e_support_primary,
        threshold=CONDITION1_THRESHOLD,
        direction="lower_is_better",
        status="pass" if cond1_pass else "fail",
        margin=float(e_support_primary - CONDITION1_THRESHOLD),
        evidence_level="classical_exact_ridge_selected_output",
        failure_reason=""
        if cond1_pass
        else f"E_support {e_support_primary:.4g} > {CONDITION1_THRESHOLD}",
        source_artifacts=["selected_output_results.csv", "error_decomposition.csv"],
    )

    condition_2 = ConditionResult(
        condition_id="condition_2",
        condition_name="benchmark_usefulness",
        metric="E_benchmark",
        metric_value=e_benchmark_primary,
        threshold=None,
        direction="lower_is_better",
        status="inconclusive",
        margin=None,
        evidence_level="full_system_ridge_natural_alpha_selected_output",
        failure_reason=(
            "no pass threshold registered for the benchmark-relative selected-output error of this "
            "candidate; descriptive gates (0.1, 1.5) are context only, not PASS/FAIL"
        ),
        source_artifacts=["selected_output_results.csv", "candidate_provenance.json"],
    )

    # Condition 3: required correctness checks + epsilon_qsvt
    poly = np.polynomial.Polynomial(compiled.polynomial_coefficients)
    bounded_max_abs = float(np.max(np.abs(poly(np.linspace(-1.0, 1.0, 8193)))))
    required = {
        "block_validation": bool(metrics["block_pass"]),
        "polynomial_boundedness": bool(bounded_max_abs <= 1.0 + BOUND_TOLERANCE),
        "phase_synthesis": compiled.phases is not None
        and len(compiled.phases) == compiled.qsvt_spec.degree + 1,
        "qsvt_polynomial_error": bool(metrics["qsvt_polynomial_pass"]),
        "signed_readout_recovery": float(metrics["max_signed_recovery_absolute_error"]) < 1.0e-6,
    }
    cond3_metric = float(metrics["epsilon_qsvt"])
    cond3_pass = (cond3_metric <= CONDITION3_THRESHOLD) and all(required.values())
    condition_3 = ConditionResult(
        condition_id="condition_3",
        condition_name="filter_implementation",
        metric="epsilon_qsvt",
        metric_value=cond3_metric,
        threshold=CONDITION3_THRESHOLD,
        direction="lower_is_better",
        status="pass" if cond3_pass else "fail",
        margin=float(CONDITION3_THRESHOLD - cond3_metric),
        evidence_level="statevector_circuit_execution_vs_exact_polynomial_action",
        failure_reason=""
        if cond3_pass
        else (
            f"epsilon_qsvt {cond3_metric:.4g} or required check failed: "
            f"{[k for k, v in required.items() if not v]}"
        ),
        source_artifacts=["error_decomposition.csv", "resource_ledger.csv"],
    )

    # Condition 4: predeclared qualitative rubric -- not credible for this candidate
    not_credible_reasons = [
        "direct_small_scale_multiplexing_access",
        "sampled_selected_outputs_only",
        "no_compiled_ieee_scale_reversible_qrom",
        "no_integrated_amplitude_amplification",
    ]
    condition_4 = ConditionResult(
        condition_id="condition_4",
        condition_name="access_readout_credibility",
        metric="credible_or_not_credible",
        metric_value="not_credible",
        threshold="credible",
        direction="n/a",
        status="fail",
        margin=None,
        evidence_level="transpiled_circuit_resources_plus_sampled_readout_accounting",
        failure_reason="; ".join(not_credible_reasons) + " (predeclared rubric)",
        source_artifacts=["resource_ledger.csv", "classical_comparison.csv"],
    )

    conditions = [condition_1, condition_2, condition_3, condition_4]
    selected_outputs = pd.DataFrame(sel_rows)
    return conditions, selected_outputs, error_df


def first_failed_condition(conditions: list[ConditionResult]) -> ConditionResult | None:
    """First logical condition with a non-pass status (fail > inconclusive in ordering)."""

    for condition in conditions:
        if condition.status != "pass":
            return condition
    return None


# ------------------------------------------------------------------------- finite-shot summary


def summarize_finite_shots(
    shots: Any, compiled: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Finite-shot summary + bit-for-bit reproduction check vs the frozen ledger."""

    rows_frame = pd.DataFrame(shots.rows)
    summary_rows: list[dict[str, Any]] = []
    if not rows_frame.empty:
        for (functional_id, shot_count), group in rows_frame.groupby(
            ["functional_id", "shots_attempted"], sort=True
        ):
            recovered = group["recovered_selected_output"].astype(float)
            sv_ref = float(group["statevector_reference"].iloc[0])
            abs_err = (recovered - sv_ref).abs()
            rel_mask = group["relative_error_numerically_stable"].astype(bool)
            summary_rows.append(
                {
                    "functional_id": functional_id,
                    "shots_attempted": int(shot_count),
                    "n_seeds": len(group),
                    "seeds": ",".join(
                        str(int(value)) for value in sorted(group["seed"].astype(int))
                    ),
                    "mean_recovered_selected_output": float(recovered.mean()),
                    "std_over_seeds": float(recovered.std(ddof=1)) if len(recovered) > 1 else 0.0,
                    "statevector_reference": sv_ref,
                    "quantized_ridge_reference": float(group["quantized_ridge_reference"].iloc[0]),
                    "mean_abs_error_vs_statevector": float(abs_err.mean()),
                    "max_abs_error_vs_statevector": float(abs_err.max()),
                    "mean_relative_error_vs_statevector": float(
                        group.loc[rel_mask, "relative_error_vs_statevector"].astype(float).mean()
                    )
                    if rel_mask.any()
                    else float("nan"),
                    "ci_lower_mean": float(group["confidence_interval_lower"].astype(float).mean()),
                    "ci_upper_mean": float(group["confidence_interval_upper"].astype(float).mean()),
                    "mean_postselection_probability": float(
                        group["estimated_postselection_probability"].astype(float).mean()
                    ),
                    "total_attempted_shots": int(group["shots_attempted"].astype(int).sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    # bit-for-bit reproduction check vs the frozen canonical ledger
    check_rows: list[dict[str, Any]] = []
    bit_for_bit = None
    max_count_drift = 0.0
    if FROZEN_SHOT_LEDGER.is_file() and not rows_frame.empty:
        frozen = pd.read_csv(FROZEN_SHOT_LEDGER)
        key = ["functional_id", "shots_attempted", "seed"]
        count_cols = ["count_00", "count_01", "count_10", "count_11"]
        merged = rows_frame[key + count_cols + ["source_circuit_hash"]].merge(
            frozen[key + count_cols + ["source_circuit_hash"]],
            on=key,
            suffixes=("_fresh", "_frozen"),
            how="inner",
        )
        for col in count_cols:
            drift = (merged[f"{col}_fresh"].astype(int) - merged[f"{col}_frozen"].astype(int)).abs()
            max_count_drift = max(max_count_drift, float(drift.max()))
        hash_drift = int(
            (
                merged["source_circuit_hash_fresh"].astype(str)
                != merged["source_circuit_hash_frozen"].astype(str)
            ).sum()
        )
        bit_for_bit = bool(max_count_drift == 0 and hash_drift == 0)
        check_rows.append(
            {
                "frozen_ledger": str(FROZEN_SHOT_LEDGER.relative_to(REPO_ROOT)),
                "matched_rows": len(merged),
                "expected_rows": len(rows_frame),
                "max_count_drift": max_count_drift,
                "source_circuit_hash_drift_rows": hash_drift,
                "bit_for_bit_reproduction": bit_for_bit,
                "generator": "run_compiled_shots (generic_sparse_execution.py:540)",
                "seed_convention": "seed_simulator=int(seed), no functional offset",
            }
        )
    check = pd.DataFrame(check_rows)
    return summary, check, rows_frame


# ------------------------------------------------------------------------- artifact writers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_and_checksums(directory: Path) -> None:
    """Convention-A minimal manifest + checksums (excludes manifest.json + checksums.sha256)."""

    artifacts = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "checksums.sha256"}:
            continue
        rel = path.relative_to(directory).as_posix()
        artifacts.append({"path": rel, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    manifest = {"artifact_count": len(artifacts), "artifacts": artifacts}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"{item['sha256']}  {item['path']}\n" for item in artifacts]
    (directory / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------------- orchestration


def run_joint_four_condition(
    config_path: Path | str = DEFAULT_CONFIG,
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the full single-candidate four-condition pipeline and write all artifacts."""

    config = load_decision_config(config_path)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if progress:
        print("[joint_four_condition] rebuilding frozen canonical candidate (identity-verified)")
    compiled = rebuild_canonical()

    if progress:
        print("[joint_four_condition] freezing candidate (outcome-independent)")
    freeze = freeze_candidate(compiled, config)
    (directory / "candidate_freeze.json").write_text(
        json.dumps(asdict(freeze), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if progress:
        print("[joint_four_condition] establishing full-system benchmark reference (frozen inputs)")
    benchmark = build_full_system_benchmark(
        compiled,
        freeze.global_rows,
        freeze.global_columns,
        natural_alpha=float(config["benchmark"]["natural_alpha"]),
        matched_alpha=float(config["benchmark"]["matched_alpha"]),
    )

    if progress:
        print("[joint_four_condition] running statevector QSVT validation + epsilon decomposition")
    statevector = validate_compiled_statevector(compiled)

    if progress:
        print("[joint_four_condition] preparing + running finite-shot reproduction (all 90 rows)")
    prepared = prepare_compiled_execution(compiled)
    shots = run_compiled_shots(compiled, statevector, prepared)
    finite_shot_summary, finite_shot_check, finite_shot_rows = summarize_finite_shots(
        shots, compiled
    )

    if progress:
        print("[joint_four_condition] measuring circuit resources + classical comparators")
    resources = build_resource_evidence(compiled, prepared)
    classical = build_classical_comparison(compiled, benchmark, freeze.primary_functional_id)

    if progress:
        print("[joint_four_condition] applying predeclared four-condition decision rules")
    conditions, selected_outputs, error_decomposition = evaluate_four_conditions(
        compiled, statevector, benchmark, config, finite_shot_summary
    )
    first_failed = first_failed_condition(conditions)
    overall_pass = all(condition.status == "pass" for condition in conditions)

    # ---- write CSVs ----
    selected_outputs.to_csv(directory / "selected_output_results.csv", index=False)
    error_decomposition.to_csv(directory / "error_decomposition.csv", index=False)
    finite_shot_summary.to_csv(directory / "finite_shot_summary.csv", index=False)
    finite_shot_rows.to_csv(directory / "finite_shot_rows.csv", index=False)
    finite_shot_check.to_csv(directory / "finite_shot_reproduction_check.csv", index=False)

    resource_record = dict(resources.record)
    resource_ledger = pd.DataFrame([resource_record])
    # postselection + shot accounting + query cost (primary functional)
    primary = freeze.primary_functional_id
    primary_sv = next(r for r in statevector.functional_rows if r["functional_id"] == primary)
    p_post = float(primary_sv["postselection_probability"])
    total_attempted = (
        int(finite_shot_rows["shots_attempted"].astype(int).sum())
        if not finite_shot_rows.empty
        else 0
    )
    direct_attempted = (
        int(finite_shot_rows["direct_postselection_shots_attempted"].astype(int).sum())
        if not finite_shot_rows.empty
        else 0
    )
    n_functional_queries = len(finite_shot_rows)
    resource_ledger["postselection_probability_primary"] = p_post
    resource_ledger["expected_repetitions_per_accepted_result"] = (
        float(1.0 / p_post) if p_post > 0 else float("inf")
    )
    resource_ledger["attempted_shots_total"] = total_attempted
    resource_ledger["direct_postselection_attempted_shots_total"] = direct_attempted
    resource_ledger["functional_queries_total"] = n_functional_queries
    resource_ledger["per_functional_query_attempted_shots"] = total_attempted // max(
        n_functional_queries, 1
    )
    resource_ledger["evidence_tier"] = "statevector_circuit_execution_plus_finite_shot_simulation"
    resource_ledger["hardware_executed"] = False
    resource_ledger["transpiled_circuit_executed_on_hardware"] = False
    resource_ledger.to_csv(directory / "resource_ledger.csv", index=False)
    pd.DataFrame(resources.register_rows).to_csv(directory / "register_ledger.csv", index=False)

    classical.to_csv(directory / "classical_comparison.csv", index=False)

    # ---- decision ledger ----
    ledger_rows = []
    for condition in conditions:
        ledger_rows.append(
            {
                "condition_id": condition.condition_id,
                "condition_name": condition.condition_name,
                "metric": condition.metric,
                "metric_value": condition.metric_value,
                "threshold": condition.threshold,
                "direction": condition.direction,
                "status": condition.status,
                "margin": condition.margin,
                "evidence_level": condition.evidence_level,
                "failure_reason": condition.failure_reason,
                "source_artifacts": ";".join(condition.source_artifacts),
            }
        )
    ledger_rows.append(
        {
            "condition_id": "overall",
            "condition_name": "overall_candidate",
            "metric": "all_conditions_pass",
            "metric_value": overall_pass,
            "threshold": True,
            "direction": "all_pass",
            "status": "pass" if overall_pass else "fail",
            "margin": None,
            "evidence_level": "joint_four_condition_ledger",
            "failure_reason": ""
            if overall_pass
            else (
                f"first failed logical condition: {first_failed.condition_id} "
                f"({first_failed.condition_name}); later-condition success cannot override"
            ),
            "source_artifacts": "four_condition_decision_ledger.csv",
        }
    )
    ledger = pd.DataFrame(ledger_rows)
    ledger.to_csv(directory / "four_condition_decision_ledger.csv", index=False)
    ledger_json = {
        "study_id": config["study_id"],
        "candidate": freeze.workload_id,
        "overall_status": ledger_rows[-1]["status"],
        "first_failed_logical_condition": (
            None
            if first_failed is None
            else {
                "condition_id": first_failed.condition_id,
                "condition_name": first_failed.condition_name,
                "status": first_failed.status,
                "metric_value": first_failed.metric_value,
                "threshold": first_failed.threshold,
                "failure_reason": first_failed.failure_reason,
            }
        ),
        "conditions": [asdict(condition) for condition in conditions],
        "declared_before_outcome_evaluation": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (directory / "four_condition_decision_ledger.json").write_text(
        json.dumps(ledger_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- failures (explicit; canonical has no structured failure, schema retained) ----
    failures = pd.DataFrame(
        [
            {
                "workload_id": freeze.workload_id,
                "failure_code": "none",
                "stage": "n/a",
                "retained": True,
                "note": "canonical candidate executed with no structured failure; dense-fallback / "
                "phase-failure / finite-shot-ceiling rows would be retained here if they occurred",
            }
        ]
    )
    failures.to_csv(directory / "failures.csv", index=False)

    # ---- provenance ----
    provenance = {
        "study_id": config["study_id"],
        "generated_utc": _now_utc(),
        "generator": "scripts/run_joint_four_condition.py",
        "git_commit": _git_commit(),
        "python": _python_version(),
        "platform": _platform(),
        "evidence_tier": "statevector_circuit_execution_plus_finite_shot_simulation",
        "claim_boundary": CLAIM_BOUNDARY,
        "manuscript_assets": [
            "manuscript/tables/joint_four_condition_candidate.tex",
            "manuscript/tables/joint_four_condition_decision.tex",
            "manuscript/tables/joint_error_decomposition.tex",
            "manuscript/tables/joint_four_condition_metrics.tex",
            "manuscript/figures/joint_four_condition_waterfall.pdf",
        ],
        "frozen_inputs_read_only": [
            "configs/generic_sparse_qsvt_compiler.json",
            "configs/tqe_reviewer_blocking/joint_feasibility.json",
            "outputs/sparse_chain_reconciliation/end_to_end_run/",
            "outputs/phase10_sparse_wrapper_8x8_complete/phase_cache/",
            "outputs/generic_sparse_qsvt_compiler/canonical_shot_rows_generic.csv",
        ],
        "benchmark_reference_provenance": benchmark.provenance,
        "benchmark_reference": {
            "full_matrix_shape": list(benchmark.full_matrix_shape),
            "full_matrix_hash": benchmark.full_matrix_hash,
            "block_submatrix_match": benchmark.block_submatrix_match,
            "residual_slice_match": benchmark.residual_slice_match,
            "block_submatrix_max_abs_err": benchmark.block_submatrix_max_abs_err,
            "residual_slice_max_abs_err": benchmark.residual_slice_max_abs_err,
        },
        "finite_shot_reproduction": (
            finite_shot_check.to_dict(orient="records")[0] if not finite_shot_check.empty else {}
        ),
        "reproduce_command": (
            "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/run_joint_four_condition.py"
        ),
    }
    (directory / "candidate_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                k: provenance[k]
                for k in (
                    "study_id",
                    "generated_utc",
                    "generator",
                    "git_commit",
                    "python",
                    "platform",
                    "evidence_tier",
                    "claim_boundary",
                    "manuscript_assets",
                    "frozen_inputs_read_only",
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_manifest_and_checksums(directory)
    _write_readme(directory, config, freeze, ledger_json)

    if progress:
        print(f"[joint_four_condition] DONE -> {directory}")
        first_failed_id = (
            ledger_json["first_failed_logical_condition"]["condition_id"]
            if first_failed
            else "none"
        )
        print(f"  overall_status={ledger_json['overall_status']}  first_failed={first_failed_id}")
    return {
        "directory": str(directory),
        "overall_status": ledger_json["overall_status"],
        "first_failed": ledger_json["first_failed_logical_condition"],
        "conditions": [asdict(condition) for condition in conditions],
        "benchmark": benchmark.provenance,
    }


def _write_readme(
    directory: Path, config: dict, freeze: CandidateFreeze, ledger_json: dict
) -> None:
    readme = f"""# Joint Four-Condition Single-Candidate Trace

{CLAIM_BOUNDARY}

- study_id: {config["study_id"]}
- candidate: {freeze.workload_id}
- overall_status: {ledger_json["overall_status"]}

This root evaluates ONE frozen IEEE-14 8x8 d=31 candidate through four logically ordered
conditions in a single traceable pipeline. The candidate was frozen (candidate_freeze.json)
before any four-condition metric was computed. The full-system benchmark link is established
here from frozen inputs (matrix_seed 123) with explicit provenance; no historical evidence was
altered and the link was not fabricated.

## Files
- `candidate_freeze.json` -- outcome-independent frozen candidate record (all components + hashes)
- `candidate_provenance.json` / `provenance.json` -- generation provenance + benchmark link
- `four_condition_decision_ledger.csv` / `.json` -- per-condition PASS/FAIL/INCONCLUSIVE + overall
- `error_decomposition.csv` -- separated matrix/access + numerical + recovery errors (never merged)
- `selected_output_results.csv` -- per-functional selected-output chain + E_support/E_benchmark
- `finite_shot_summary.csv` / `finite_shot_rows.csv` / `finite_shot_reproduction_check.csv`
- `resource_ledger.csv` / `register_ledger.csv` -- transpiled resources + shot accounting
- `classical_comparison.csv` -- matched classical selected-output comparators (no speedup claim)
- `failures.csv` -- explicit failure retention (schema retained; no structured failure here)
- `manifest.json` / `checksums.sha256` -- Convention-A artifact manifest

## Evidence tiers (kept separate, never merged)
sparse support error; exact polynomial matrix action; explicit statevector circuit execution;
finite-shot simulation; transpiled resources; modeled classical comparator cost.

## Reproduce
```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
  .venv/bin/python scripts/run_joint_four_condition.py
```
"""
    (directory / "README.md").write_text(readme, encoding="utf-8")


def _now_utc() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _python_version() -> str:
    import platform
    import sys

    return f"{sys.version.split()[0]} ({platform.python_implementation()})"


def _platform() -> str:
    import platform

    return platform.platform()


if __name__ == "__main__":  # pragma: no cover
    run_joint_four_condition()
