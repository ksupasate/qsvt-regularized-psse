"""Frozen workload adapters for the generic sparse-QSVT compiler.

Scientific constants and repository artifact paths belong here, not in the generic
compiler. Each adapter validates its frozen source registries before constructing the
seven typed compiler input records.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import (
    build_case_design,
    generate_case_residual,
)
from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import _build_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompilerInputBundle,
    ExecutionSpec,
    FunctionalDefinition,
    FunctionalSpec,
    MatrixSpec,
    QSVTSpec,
    QuantizationSpec,
    ResidualSpec,
    SparseCompilerError,
    SupportSpec,
)
from robust_qsvt_se.qsvt.phase_synthesis import _phase_cache_key
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    PHASE_CONVENTION,
    stable_array_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/generic_sparse_qsvt_compiler.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SparseCompilerError(
            "frozen_registry_mismatch",
            "workload_adapter",
            f"{name} disagrees with the frozen workload registry",
            expected=expected,
            actual=actual,
        )


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise SparseCompilerError(
            "frozen_registry_mismatch",
            "workload_adapter",
            f"{name} disagrees with the frozen workload registry",
            expected=expected,
            actual=actual,
        )


def load_generic_experiment_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = _load_json(_resolve(path))
    _assert_equal("study_id", config.get("study_id"), "generic_sparse_qsvt_compiler_v1")
    return config


def _execution_spec(config: dict[str, Any], workload: dict[str, Any]) -> ExecutionSpec:
    transpilation = config["transpilation"]
    basis = transpilation.get("basis_gates")
    return ExecutionSpec(
        shot_counts=tuple(int(value) for value in workload["shot_counts"]),
        simulator_seeds=tuple(int(value) for value in workload["seeds"]),
        simulator_method="statevector",
        basis_gates=None if basis is None else tuple(str(value) for value in basis),
        optimization_level=int(transpilation["optimization_level"]),
        seed_transpiler=(
            None
            if transpilation.get("seed_transpiler") is None
            else int(transpilation["seed_transpiler"])
        ),
        execute_statevector=True,
        execute_finite_shots=True,
        max_parallel_threads=int(transpilation["max_parallel_threads"]),
    )


def build_canonical_compiler_inputs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> CompilerInputBundle:
    """Bind the historical canonical workload to the generic input contract."""

    config = load_generic_experiment_config(config_path)
    frozen = config["canonical"]
    historical_root = _resolve(frozen["historical_root"])
    configuration = _load_json(historical_root / "configuration.json")
    matrix_metadata = _load_json(historical_root / "matrix_metadata.json")
    functionals_payload = _load_json(historical_root / "selected_functionals.json")
    quantized_reference = np.load(historical_root / "matrix_quantized.npy")
    residual = np.load(historical_root / "residual.npy")
    coefficients = np.load(historical_root / "polynomial_coefficients.npy")
    phases = np.load(historical_root / "phases.npy")
    source = _build_block(int(frozen["matrix_seed"]))
    matrix_original = np.asarray(source["H_block"], dtype=np.float64)

    _assert_equal(
        "canonical original matrix hash",
        stable_array_fingerprint(matrix_original),
        frozen["expected_original_matrix_hash"],
    )
    _assert_equal(
        "canonical quantized matrix hash",
        stable_array_fingerprint(quantized_reference),
        frozen["expected_quantized_matrix_hash"],
    )
    _assert_equal(
        "canonical residual hash",
        stable_array_fingerprint(residual),
        frozen["expected_residual_hash"],
    )
    coordinates = tuple(
        (int(i), int(j)) for i, j in zip(*np.nonzero(quantized_reference), strict=True)
    )
    support_mask = np.zeros(quantized_reference.shape, dtype=np.float64)
    for coordinate in coordinates:
        support_mask[coordinate] = 1.0

    definitions = tuple(
        FunctionalDefinition(
            functional_id=str(record["functional_id"]),
            vector=np.asarray(record["vector"], dtype=np.float64),
            kind=str(record["kind"]),
            metadata={
                "selection_timing": functionals_payload["selection_timing"],
                "physical_semantics": functionals_payload["physical_semantics"],
            },
        )
        for record in functionals_payload["functionals"]
    )
    bundle = CompilerInputBundle(
        matrix_spec=MatrixSpec(
            values=matrix_original,
            shape=tuple(int(value) for value in matrix_original.shape),
            matrix_id="canonical_ieee14_selected_weighted_jacobian_block",
            source=str(configuration["matrix_source"]),
            workload_id=str(frozen["workload_id"]),
            expected_hash=str(frozen["expected_original_matrix_hash"]),
            metadata={
                "case_name": configuration["case_name"],
                "case_source": configuration["case_source"],
                "selected_rows": configuration["selected_rows"],
                "selected_columns": configuration["selected_columns"],
                "historical_root": str(historical_root.relative_to(REPO_ROOT)),
            },
        ),
        support_spec=SupportSpec(
            coordinates=coordinates,
            support_id=str(matrix_metadata["matrix_fingerprint_quantized"]),
            slots=int(matrix_metadata["slots"]),
            expected_support_hash=stable_array_fingerprint(support_mask),
            provenance={
                "policy": "frozen canonical support",
                "historical_matrix_metadata": str(
                    (historical_root / "matrix_metadata.json").relative_to(REPO_ROOT)
                ),
            },
        ),
        quantization_spec=QuantizationSpec(
            magnitude_bits=int(frozen["value_bits"]),
            sign_representation="sign_magnitude",
            rule="sign_magnitude_round_to_nearest",
            scale=float(matrix_metadata["mu"]),
            expected_quantized_hash=str(frozen["expected_quantized_matrix_hash"]),
        ),
        qsvt_spec=QSVTSpec(
            alpha=float(configuration["alpha"]),
            beta=float(configuration["beta"]),
            normalized_lambda=float(configuration["normalized_lambda"]),
            boundedness_factor=float(configuration["contraction_c"]),
            polynomial_coefficients=np.asarray(coefficients, dtype=np.float64),
            polynomial_id="codesigned_bounded_ridge_d31",
            phases=np.asarray(phases, dtype=np.float64),
            degree=int(configuration["polynomial_degree"]),
            parity="odd",
            phase_convention=str(configuration["phase_convention"]),
            require_uncomputation=True,
            bound_tolerance=2.0e-3,
            expected_polynomial_hash=stable_array_fingerprint(coefficients),
            expected_phase_hash=str(matrix_metadata["phase_fingerprint"]),
            provenance={
                "polynomial_path": str(
                    (historical_root / "polynomial_coefficients.npy").relative_to(REPO_ROOT)
                ),
                "phase_path": str((historical_root / "phases.npy").relative_to(REPO_ROOT)),
                "historical_phase_path": configuration["phase_path"],
            },
        ),
        residual_spec=ResidualSpec(
            vector=np.asarray(residual, dtype=np.float64),
            residual_id="canonical_ieee14_residual_seed123",
            data_split="frozen_canonical",
            expected_hash=str(frozen["expected_residual_hash"]),
            provenance={"path": str((historical_root / "residual.npy").relative_to(REPO_ROOT))},
        ),
        functional_spec=FunctionalSpec(
            functionals=definitions,
            primary_functional_id=str(configuration["selected_output_name"]),
            postselection_convention="flag_zero_iff_sparse_work_zero",
            readout_convention="real_branch_hadamard_c1c0",
        ),
        execution_spec=_execution_spec(config, frozen),
        provenance={
            "adapter": "canonical_frozen_reconciliation",
            "configuration_file_sha256": _sha256_file(historical_root / "configuration.json"),
            "matrix_metadata_file_sha256": _sha256_file(historical_root / "matrix_metadata.json"),
        },
    )
    return bundle


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _select_second_qsvt_row(
    frame: pd.DataFrame,
    workload: dict[str, Any],
) -> pd.Series:
    rule = workload["qsvt_rule"]
    subset = frame[
        (frame["selector"] == workload["selector"])
        & (frame["k_budget"] == int(workload["support_budget"]))
        & (frame["slot_budget"] == int(workload["slot_budget"]))
    ].copy()
    subset["configuration_id"] = subset.apply(
        lambda row: (
            f"{row['selector']}:k{int(row['k_budget'])}:s{int(row['slot_budget'])}:"
            f"alpha{float(row['alpha']).hex()}:d{int(row['degree'])}"
        ),
        axis=1,
    )
    eligible = subset[
        _bool_series(subset, "boundedness_ok")
        & (pd.to_numeric(subset["uniform_fit_error"]) <= float(rule["uniform_fit_tolerance"]))
        & (subset["phase_synthesis_status"] == "synthesized")
        & (pd.to_numeric(subset["phase_count"]) == pd.to_numeric(subset["degree"]) + 1)
        & (
            subset["statevector_action_error"].isna()
            | (
                pd.to_numeric(subset["statevector_action_error"])
                <= float(rule["action_tolerance"])
            )
        )
    ].copy()
    if eligible.empty:
        raise SparseCompilerError(
            "no_predeclared_qsvt_operating_point",
            "workload_adapter",
            "no frozen QSVT grid row passes the outcome-independent rule",
        )
    eligible = eligible.sort_values(
        ["degree", "normalized_lambda", "configuration_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0]


def build_second_ieee30_compiler_inputs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> CompilerInputBundle:
    """Bind the frozen IEEE-30 workload without consulting held-out outputs."""

    config = load_generic_experiment_config(config_path)
    frozen = config["second_workload"]
    block_registry_path = _resolve(frozen["block_registry"])
    block_registry = _load_json(block_registry_path)
    _assert_equal("block case", block_registry["case_name"], frozen["case_name"])
    _assert_equal("block matrix seed", block_registry["matrix_seed"], frozen["matrix_seed"])
    _assert_equal("block shape", block_registry["block_shape"], [8, 8])
    _assert_equal("block matrix hash", block_registry["matrix_fingerprint"], frozen["expected_matrix_hash"])

    design = build_case_design(
        str(frozen["case_name"]),
        int(frozen["matrix_seed"]),
        dimension=int(frozen["dimension"]),
    )
    matrix = np.asarray(design.small.matrix, dtype=np.float64)
    _assert_equal("rebuilt block hash", stable_array_fingerprint(matrix), frozen["expected_matrix_hash"])
    _assert_equal("rebuilt rows", list(design.small.selected_rows), block_registry["selected_global_rows"])
    _assert_equal("rebuilt columns", list(design.small.selected_columns), block_registry["selected_global_columns"])

    support_registry_path = _resolve(frozen["support_registry"])
    support_registry = _load_json(support_registry_path)
    support_id = str(frozen["support_id"])
    if support_id not in support_registry:
        raise SparseCompilerError(
            "frozen_support_missing",
            "workload_adapter",
            "the predeclared IEEE-30 support is missing",
            support_id=support_id,
        )
    coordinates = tuple((int(i), int(j)) for i, j in support_registry[support_id])
    mask = np.zeros(matrix.shape, dtype=np.float64)
    for coordinate in coordinates:
        mask[coordinate] = 1.0
    _assert_equal("support hash", stable_array_fingerprint(mask), frozen["expected_support_hash"])
    exact_sparse = np.where(mask.astype(bool), matrix, 0.0)
    _assert_equal(
        "exact sparse matrix hash",
        stable_array_fingerprint(exact_sparse),
        frozen["expected_exact_sparse_hash"],
    )

    cross_config = _load_json(_resolve(frozen["cross_case_config"]))
    held_out_seed = min(int(value) for value in cross_config["held_out_seed_ids"])
    _assert_equal("held-out residual seed", held_out_seed, frozen["expected_residual_seed"])
    residual = generate_case_residual(
        str(frozen["case_name"]), held_out_seed, design.small.selected_rows
    )
    _assert_equal(
        "held-out residual hash",
        stable_array_fingerprint(residual),
        frozen["expected_residual_hash"],
    )

    grid_path = _resolve(frozen["qsvt_grid"])
    selected = _select_second_qsvt_row(pd.read_csv(grid_path), frozen)
    _assert_equal("selected degree", int(selected["degree"]), frozen["expected_selected_degree"])
    _assert_close("selected alpha", float(selected["alpha"]), frozen["expected_selected_alpha"])
    _assert_close("selected beta", float(selected["beta"]), frozen["expected_selected_beta"])
    _assert_close(
        "selected normalized lambda",
        float(selected["normalized_lambda"]),
        frozen["expected_selected_normalized_lambda"],
    )
    _assert_close("selected C", float(selected["contraction_c"]), frozen["expected_selected_C"])

    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    degree = int(selected["degree"])
    positive = np.linalg.svd(exact_sparse, compute_uv=False)
    positive = positive[positive > 1.0e-10]
    domain_min = float(np.clip(0.9 * positive.min() / beta, 1.0e-4, 0.999))
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=alpha,
        domain_min=domain_min,
        domain_max=1.0,
        degree=degree,
        margin=1.05,
    )
    _assert_close("reconstructed C", target.bound_C, float(selected["contraction_c"]))
    _assert_close(
        "reconstructed fit error", target.fit_max_abs_error, float(selected["uniform_fit_error"])
    )
    coefficients = np.asarray(target.coefficients, dtype=np.float64)
    cache_metadata = {
        "study_id": str(frozen["phase_cache_study_id"]),
        "degree": degree,
        "beta": beta,
    }
    cache_key = _phase_cache_key(
        coefficients, angle_solver="iterative", metadata=cache_metadata
    )
    _assert_equal("phase cache key", cache_key, frozen["expected_phase_cache_key"])
    phase_path = _resolve(frozen["phase_cache"]) / f"{cache_key}_phase_angles.csv"
    metadata_path = _resolve(frozen["phase_cache"]) / f"{cache_key}_metadata.json"
    if not phase_path.is_file() or not metadata_path.is_file():
        raise SparseCompilerError(
            "frozen_phase_missing",
            "workload_adapter",
            "the selected existing phase sequence is unavailable",
            phase_path=str(phase_path),
        )
    _assert_equal("phase file hash", _sha256_file(phase_path), frozen["expected_phase_file_sha256"])
    phase_frame = pd.read_csv(phase_path)
    phases = phase_frame["phase_angle"].to_numpy(dtype=np.float64)

    physical_records = [
        record for record in design.functional_records if record.family != "legacy_predetermined"
    ]
    coordinates_records = sorted(
        (record for record in physical_records if record.family == "coordinate"),
        key=lambda record: (record.global_state_indices, record.functional_id),
    )
    branch_records = sorted(
        (record for record in physical_records if record.family == "branch_angle_difference"),
        key=lambda record: (
            tuple(int(value) for value in record.branch_id.split("-") if value),
            record.functional_id,
        ),
    )
    if not coordinates_records or not branch_records:
        raise SparseCompilerError(
            "metadata_functional_missing",
            "workload_adapter",
            "the frozen metadata does not provide both requested available functionals",
        )
    coordinate = coordinates_records[0]
    branch = branch_records[0]
    unavailable = [
        record
        for record in design.unavailable
        if record.requested_functional_id == "connected_block_angle_area_aggregate"
    ]
    if not unavailable:
        raise SparseCompilerError(
            "metadata_functional_missing",
            "workload_adapter",
            "the requested unavailable connected-area functional is missing from the registry",
        )
    definitions = tuple(
        FunctionalDefinition(
            functional_id=record.functional_id,
            vector=np.asarray(record.vector, dtype=np.float64),
            kind=record.family,
            metadata=asdict(record),
        )
        for record in (coordinate, branch)
    )
    value_scale = float(np.max(np.abs(exact_sparse)))
    bundle = CompilerInputBundle(
        matrix_spec=MatrixSpec(
            values=matrix,
            shape=tuple(int(value) for value in matrix.shape),
            matrix_id=str(frozen["expected_block_id"]),
            source="ieee30_pypower_ac_weighted_jacobian",
            workload_id=str(frozen["workload_id"]),
            expected_hash=str(frozen["expected_matrix_hash"]),
            metadata={
                "case_name": frozen["case_name"],
                "case_source": frozen["case_source"],
                "matrix_seed": frozen["matrix_seed"],
                "selected_rows": list(design.small.selected_rows),
                "selected_columns": list(design.small.selected_columns),
                "rank": int(np.linalg.matrix_rank(matrix)),
                "nonzeros": int(np.count_nonzero(matrix)),
                "block_registry": str(block_registry_path.relative_to(REPO_ROOT)),
            },
        ),
        support_spec=SupportSpec(
            coordinates=coordinates,
            support_id=support_id,
            slots=int(frozen["slot_budget"]),
            expected_support_hash=str(frozen["expected_support_hash"]),
            provenance={
                "selector": frozen["selector"],
                "support_budget": frozen["support_budget"],
                "slot_budget": frozen["slot_budget"],
                "training_seed_ids": cross_config["training_seed_ids"],
                "support_registry": str(support_registry_path.relative_to(REPO_ROOT)),
                "held_out_or_truth_used": False,
            },
        ),
        quantization_spec=QuantizationSpec(
            magnitude_bits=int(frozen["value_bits"]),
            sign_representation="sign_magnitude",
            rule="sign_magnitude_round_to_nearest",
            scale=value_scale,
            expected_quantized_hash=str(frozen["expected_quantized_sparse_hash"]),
        ),
        qsvt_spec=QSVTSpec(
            alpha=alpha,
            beta=beta,
            normalized_lambda=float(selected["normalized_lambda"]),
            boundedness_factor=float(selected["contraction_c"]),
            polynomial_coefficients=coefficients,
            polynomial_id=f"frozen_grid_{cache_key}",
            phases=phases,
            degree=degree,
            parity="odd",
            phase_convention=PHASE_CONVENTION,
            require_uncomputation=True,
            bound_tolerance=float(cross_config["joint_feasibility"]["bound_tolerance"]),
            expected_polynomial_hash=stable_array_fingerprint(coefficients),
            expected_phase_hash=stable_array_fingerprint(phases),
            provenance={
                "qsvt_grid": str(grid_path.relative_to(REPO_ROOT)),
                "selection_rule": frozen["qsvt_rule"],
                "selected_configuration_id": str(selected["configuration_id"]),
                "phase_path": str(phase_path.relative_to(REPO_ROOT)),
                "phase_metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
                "phase_cache_key": cache_key,
                "domain_min": domain_min,
                "output_metrics_used_for_selection": False,
            },
        ),
        residual_spec=ResidualSpec(
            vector=np.asarray(residual, dtype=np.float64),
            residual_id=f"ieee30_held_out_seed_{held_out_seed}",
            data_split="held_out",
            expected_hash=str(frozen["expected_residual_hash"]),
            provenance={
                "seed": held_out_seed,
                "selection_rule": frozen["residual_rule"],
                "selected_rows": list(design.small.selected_rows),
            },
        ),
        functional_spec=FunctionalSpec(
            functionals=definitions,
            primary_functional_id=coordinate.functional_id,
            postselection_convention="flag_zero_iff_sparse_work_zero",
            readout_convention="real_branch_hadamard_c1c0",
            unavailable_requests=tuple(asdict(record) for record in unavailable),
        ),
        execution_spec=_execution_spec(config, frozen),
        provenance={
            "adapter": "frozen_ieee30_cross_case_transfer",
            "block_registry_sha256": _sha256_file(block_registry_path),
            "support_registry_sha256": _sha256_file(support_registry_path),
            "qsvt_grid_sha256": _sha256_file(grid_path),
            "functional_registry_sha256": _sha256_file(_resolve(frozen["functional_registry"])),
            "selection_protocol": "outputs/generic_sparse_qsvt_compiler/second_workload_selection_protocol.md",
        },
    )
    return bundle


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "build_canonical_compiler_inputs",
    "build_second_ieee30_compiler_inputs",
    "load_generic_experiment_config",
]
