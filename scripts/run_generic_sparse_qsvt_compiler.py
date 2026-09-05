#!/usr/bin/env python3
"""Generate non-overwriting generic sparse-QSVT compiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.generic_sparse_compiler import CompiledSparseQSVT, compile_from_bundle
from robust_qsvt_se.qsvt.generic_sparse_execution import (
    ResourceEvidence,
    ShotEvidence,
    StatevectorEvidence,
    build_resource_evidence,
    prepare_compiled_execution,
    run_compiled_shots,
    validate_compiled_statevector,
)
from robust_qsvt_se.qsvt.generic_sparse_scaling import run_compiled_scaling_study
from robust_qsvt_se.qsvt.generic_sparse_workloads import (
    REPO_ROOT,
    build_canonical_compiler_inputs,
    build_second_ieee30_compiler_inputs,
    load_generic_experiment_config,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint


DEFAULT_OUTPUT = REPO_ROOT / "outputs/generic_sparse_qsvt_compiler"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_qpy(path: Path, circuit: Any) -> None:
    from qiskit import qpy

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)


def _save_compiled_data(output: Path, label: str, compiled: CompiledSparseQSVT) -> None:
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    arrays = {
        "matrix_original": compiled.matrix_original,
        "matrix_supported_exact": compiled.matrix_supported_exact,
        "matrix_quantized": compiled.matrix_quantized,
        "residual": compiled.residual,
        "polynomial_coefficients": compiled.polynomial_coefficients,
        "phases": compiled.phases,
    }
    for name, values in arrays.items():
        np.save(data / f"{label}_{name}.npy", np.asarray(values))
    for functional_id, vector in compiled.functional_vectors.items():
        np.save(data / f"{label}_functional_{functional_id}.npy", vector)


def _save_circuits(
    output: Path,
    label: str,
    compiled: CompiledSparseQSVT,
    prepared: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    circuit_root = output / "circuits"
    for functional_id, bundle in compiled.functional_circuits.items():
        source_path = circuit_root / f"{label}_{functional_id}_source_final.qpy"
        transpiled_path = circuit_root / f"{label}_{functional_id}_transpiled_final.qpy"
        _dump_qpy(source_path, bundle.circuit)
        _dump_qpy(transpiled_path, prepared.compiled_functional_circuits[functional_id])
        records.append(
            {
                "workload_id": compiled.workload_id,
                "functional_id": functional_id,
                "source_final_qpy": str(source_path.relative_to(REPO_ROOT)),
                "source_final_sha256": _sha256(source_path),
                "source_component_hash": compiled.component_hashes[
                    f"source_final_circuit:{functional_id}"
                ],
                "transpiled_final_qpy": str(transpiled_path.relative_to(REPO_ROOT)),
                "transpiled_final_sha256": _sha256(transpiled_path),
                "same_source_object_used_for_transpilation_and_shots": True,
            }
        )
    direct_source = circuit_root / f"{label}_direct_postselection_source.qpy"
    direct_transpiled = circuit_root / f"{label}_direct_postselection_transpiled.qpy"
    wrapper_path = circuit_root / f"{label}_sparse_wrapper.qpy"
    _dump_qpy(direct_source, compiled.direct_postselection_circuit)
    _dump_qpy(direct_transpiled, prepared.compiled_direct_postselection_circuit)
    _dump_qpy(wrapper_path, compiled.sparse_block_encoding_wrapper)
    records.append(
        {
            "workload_id": compiled.workload_id,
            "functional_id": "direct_postselection_companion",
            "source_final_qpy": str(direct_source.relative_to(REPO_ROOT)),
            "source_final_sha256": _sha256(direct_source),
            "source_component_hash": compiled.component_hashes[
                "direct_postselection_circuit"
            ],
            "transpiled_final_qpy": str(direct_transpiled.relative_to(REPO_ROOT)),
            "transpiled_final_sha256": _sha256(direct_transpiled),
            "same_source_object_used_for_transpilation_and_shots": True,
            "wrapper_qpy": str(wrapper_path.relative_to(REPO_ROOT)),
            "wrapper_qpy_sha256": _sha256(wrapper_path),
        }
    )
    return records


def _canonical_reproduction_rows(
    compiled: CompiledSparseQSVT,
    statevector: StatevectorEvidence,
    resource: ResourceEvidence,
    shots: ShotEvidence,
) -> list[dict[str, Any]]:
    historical_root = REPO_ROOT / "outputs/sparse_chain_reconciliation/end_to_end_run"
    historical_state = pd.read_csv(historical_root / "statevector_validation.csv")
    historical_resource = pd.read_csv(historical_root / "resource_ledger.csv").iloc[0]
    historical_shots = pd.read_csv(historical_root / "finite_shot_results.csv")
    historical_shots = historical_shots[historical_shots["chain_type"] == "sparse"].copy()
    historical_quantized = np.load(historical_root / "matrix_quantized.npy")
    historical_support = (historical_quantized != 0.0).astype(np.float64)
    rows: list[dict[str, Any]] = []

    def add(
        criterion: str,
        evidence_type: str,
        historical: Any,
        generic: Any,
        tolerance: float = 0.0,
        comparison: str = "exact",
        note: str = "",
    ) -> None:
        if isinstance(historical, (int, float, np.integer, np.floating)) and isinstance(
            generic, (int, float, np.integer, np.floating)
        ):
            difference = abs(float(generic) - float(historical))
            passed = difference <= tolerance
        else:
            difference = 0.0 if generic == historical else float("nan")
            passed = generic == historical
        rows.append(
            {
                "criterion": criterion,
                "evidence_type": evidence_type,
                "historical_value": historical,
                "generic_compiler_value": generic,
                "comparison": comparison,
                "tolerance": tolerance,
                "absolute_difference": difference,
                "pass": bool(passed),
                "note": note,
            }
        )

    config = load_generic_experiment_config()
    frozen = config["canonical"]
    add("workload_id", "identifier", frozen["workload_id"], compiled.workload_id)
    add(
        "original_matrix_hash",
        "hash",
        frozen["expected_original_matrix_hash"],
        compiled.component_hashes["matrix_original"],
    )
    add(
        "quantized_matrix_hash",
        "hash",
        stable_array_fingerprint(historical_quantized),
        compiled.component_hashes["matrix_quantized"],
    )
    add(
        "support_hash",
        "hash_derived_from_frozen_historical_matrix",
        stable_array_fingerprint(historical_support),
        compiled.component_hashes["support_mask"],
    )
    add(
        "residual_hash",
        "hash",
        frozen["expected_residual_hash"],
        compiled.component_hashes["residual"],
    )
    historical_coefficients = np.load(historical_root / "polynomial_coefficients.npy")
    historical_phases = np.load(historical_root / "phases.npy")
    add(
        "polynomial_hash", "hash", stable_array_fingerprint(historical_coefficients),
        compiled.component_hashes["polynomial_coefficients"],
    )
    add(
        "phase_hash", "hash", stable_array_fingerprint(historical_phases),
        compiled.component_hashes["phases"],
    )
    primary_historical = historical_state.iloc[0]
    metric_map = {
        "block_reconstruction_relative_fro_error": "epsilon_block",
        "sparse_dense_action_relative_l2_error": "sparse_dense_action_relative_error",
        "qsvt_exact_polynomial_svt_relative_l2_error": "epsilon_qsvt",
        "qsvt_quantized_ridge_relative_l2_error": "qsvt_quantized_ridge_relative_error",
        "sparse_postselection_probability": "sparse_postselection_probability",
        "dense_postselection_probability": "dense_postselection_probability",
        "physical_rescaling_factor_C_over_beta": "physical_rescaling_factor_C_over_beta",
    }
    for historical_name, generic_name in metric_map.items():
        add(
            historical_name,
            "floating_statevector",
            float(primary_historical[historical_name]),
            float(statevector.metrics[generic_name]),
            5.0e-15,
            "absolute_tolerance",
        )
    generic_by_functional = {
        row["functional_id"]: row for row in statevector.functional_rows
    }
    for _, historical_row in historical_state.iterrows():
        functional_id = str(historical_row["functional_id"])
        generic = generic_by_functional[functional_id]
        add(
            f"statevector_selected_output:{functional_id}",
            "floating_statevector",
            float(historical_row["sparse_statevector_selected_output"]),
            float(generic["statevector_selected_output"]),
            5.0e-15,
            "absolute_tolerance",
        )
        add(
            f"quantized_ridge_selected_output:{functional_id}",
            "floating_reference",
            float(historical_row["quantized_ridge_selected_output"]),
            float(generic["quantized_ridge_selected_output"]),
            5.0e-15,
            "absolute_tolerance",
        )
    resource_map = {
        "total_logical_qubits": "total_simultaneously_live_qubits",
        "transpiled_gate_count": "transpiled_gate_count",
        "transpiled_depth": "transpiled_depth",
        "toffoli_count": "toffoli_count",
        "controlled_rotation_count": "controlled_rotation_count",
        "signal_unitary_calls_per_attempt": "qsvt_signal_calls",
        "projector_phase_operations_per_attempt": "phase_operations",
    }
    for historical_name, generic_name in resource_map.items():
        add(
            historical_name,
            "deterministic_integer_resource",
            int(historical_resource[historical_name]),
            int(resource.record[generic_name]),
        )
    add("live_register_qubit_sum", "deterministic_integer_resource", 8, resource.record["register_sum"])
    add("live_register_group_count", "deterministic_integer_resource", 5, len(resource.register_rows))
    if shots.rows:
        generic_shots = pd.DataFrame(shots.rows).set_index(
            ["functional_id", "shots_attempted", "seed"]
        )
        historical_indexed = historical_shots.set_index(
            ["functional_id", "shots_attempted", "seed"]
        )
        add("finite_shot_row_count", "deterministic_integer_shots", len(historical_indexed), len(generic_shots))
        exact_columns = {
            "direct_postselection_shots_attempted": "direct_postselection_shots_attempted",
            "postselection_accepted": "postselection_accepted_shots",
            "readout_accepted": "interference_branch_accepted_shots",
        }
        for historical_name, generic_name in exact_columns.items():
            mismatch = sum(
                int(historical_indexed.loc[key, historical_name])
                != int(generic_shots.loc[key, generic_name])
                for key in historical_indexed.index
            )
            add(
                f"finite_shot_all_rows_exact:{historical_name}",
                "deterministic_integer_shots",
                0,
                mismatch,
            )
        max_difference = max(
            abs(
                float(historical_indexed.loc[key, "selected_output_estimate"])
                - float(generic_shots.loc[key, "recovered_selected_output"])
            )
            for key in historical_indexed.index
        )
        add(
            "finite_shot_selected_output_all_seeds_max_difference",
            "floating_seed_reproduction",
            0.0,
            max_difference,
            5.0e-15,
            "absolute_tolerance",
        )
    else:
        add(
            "finite_shot_seed_reproduction",
            "not_executed",
            "historical rows available",
            "not executed",
            note="--skip-shots was requested",
        )
    return rows


def _compiler_validation_rows(
    compiled: CompiledSparseQSVT, evidence: StatevectorEvidence
) -> list[dict[str, Any]]:
    definitions = (
        ("lookup logical reconstruction", "epsilon_lookup", 1.0e-14),
        ("sparse-wrapper top block", "epsilon_block", 1.0e-9),
        ("sparse-wrapper unitarity", "wrapper_unitarity_max_error", 1.0e-9),
        ("inverse and uncomputation roundtrip", "uncomputation_roundtrip_max_error", 1.0e-9),
        ("sparse versus dense encoded action", "sparse_dense_action_relative_error", 1.0e-9),
        ("QSVT circuit versus exact polynomial action", "epsilon_qsvt", 1.0e-6),
    )
    rows = [
        {
            "workload_id": compiled.workload_id,
            "validation_stage": name,
            "metric": metric,
            "value": evidence.metrics[metric],
            "tolerance": tolerance,
            "pass": float(evidence.metrics[metric]) <= tolerance,
            "evidence_status": "statevector executed",
        }
        for name, metric, tolerance in definitions
    ]
    for metric in ("epsilon_poly", "epsilon_quant", "epsilon_support"):
        rows.append(
            {
                "workload_id": compiled.workload_id,
                "validation_stage": "scientific approximation decomposition",
                "metric": metric,
                "value": evidence.metrics[metric],
                "tolerance": np.nan,
                "pass": np.nan,
                "evidence_status": "reported, not an implementation pass/fail metric",
            }
        )
    return rows


def _second_metadata(
    compiled: CompiledSparseQSVT,
    statevector: StatevectorEvidence,
    resource: ResourceEvidence,
    circuit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "study_id": "generic_sparse_qsvt_compiler_v1",
        "workload_id": compiled.workload_id,
        "selection_was_frozen_before_output_evaluation": True,
        "case": {
            "case_name": "ieee30",
            "case_source": "pypower",
            "measurement_rows": "generated from the frozen network model",
            "field_measurements": False,
        },
        "block": {
            "block_id": compiled.matrix_spec.matrix_id,
            "source_registry": compiled.matrix_spec.metadata["block_registry"],
            "selected_rows": compiled.matrix_spec.metadata["selected_rows"],
            "selected_columns": compiled.matrix_spec.metadata["selected_columns"],
            "shape": list(compiled.matrix_original.shape),
            "rank": int(np.linalg.matrix_rank(compiled.matrix_original)),
            "nonzero_count": int(np.count_nonzero(compiled.matrix_original)),
            "matrix_hash": compiled.component_hashes["matrix_original"],
        },
        "support": {
            "support_id": compiled.support_spec.support_id,
            "coordinates": [list(pair) for pair in compiled.support_spec.coordinates],
            "support_hash": compiled.component_hashes["support_mask"],
            "exact_sparse_hash": compiled.component_hashes["matrix_supported_exact"],
            "quantized_sparse_hash": compiled.component_hashes["matrix_quantized"],
            "slot_assignment_hash": compiled.component_hashes["slot_assignment"],
            "selector": compiled.support_spec.provenance["selector"],
            "training_seed_ids": compiled.support_spec.provenance["training_seed_ids"],
            "held_out_or_truth_used": False,
        },
        "residual": {
            "residual_id": compiled.residual_spec.residual_id,
            "hash": compiled.component_hashes["residual"],
            "data_split": compiled.residual_spec.data_split,
            "provenance": compiled.residual_spec.provenance,
        },
        "qsvt_operating_point": {
            "alpha": compiled.qsvt_spec.alpha,
            "beta": compiled.qsvt_spec.beta,
            "normalized_lambda": compiled.qsvt_spec.normalized_lambda,
            "boundedness_factor_C": compiled.qsvt_spec.boundedness_factor,
            "degree": compiled.qsvt_spec.degree,
            "phase_count": len(compiled.phases),
            "polynomial_hash": compiled.component_hashes["polynomial_coefficients"],
            "phase_hash": compiled.component_hashes["phases"],
            "selection_rule": compiled.qsvt_spec.provenance["selection_rule"],
            "phase_cache_key": compiled.qsvt_spec.provenance["phase_cache_key"],
            "output_metrics_used_for_selection": False,
        },
        "compiler": {
            "workload_digest": compiled.workload_digest,
            "validated_input_metadata": compiled.validated_input_metadata,
            "padded_dimensions": compiled.padded_dimensions,
            "register_allocation": compiled.register_allocation,
            "component_hashes": compiled.component_hashes,
            "qsvt_sequence": compiled.qsvt_sequence,
            "postselection_logic": compiled.postselection_logic,
            "signed_readout_logic": compiled.signed_readout_logic,
            "inverse_uncomputation_path": compiled.inverse_uncomputation_path,
            "dense_fallback_used": False,
            "direct_output_state_preparation_used": False,
        },
        "statevector_metrics": statevector.metrics,
        "resource_record": resource.record,
        "circuit_artifacts": circuit_records,
        "claim_boundary": (
            "Small-scale classical statevector and Aer sampling evidence only. "
            "No hardware, scalable sparse oracle, fault-tolerant resource, speedup, "
            "advantage, or practical-competitiveness claim."
        ),
    }


def _functional_registry(compiled: CompiledSparseQSVT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = {item.functional_id: item for item in compiled.functional_spec.functionals}
    for functional_id, vector in compiled.functional_vectors.items():
        definition = definitions[functional_id]
        rows.append(
            {
                "request_order": len(rows) + 1,
                "request": definition.kind,
                "functional_id": functional_id,
                "availability": "available",
                "reason": "constructed from frozen state/topology metadata",
                "vector_json": json.dumps(vector.tolist()),
                "vector_hash": compiled.component_hashes[f"functional:{functional_id}"],
                "metadata_json": json.dumps(_json_ready(definition.metadata), sort_keys=True),
                "proxy_substituted": False,
                "output_metrics_used_for_selection": False,
            }
        )
    for request in compiled.functional_spec.unavailable_requests:
        rows.append(
            {
                "request_order": len(rows) + 1,
                "request": request.get("family", "connected_area_aggregate"),
                "functional_id": request.get(
                    "requested_functional_id", "connected_block_angle_area_aggregate"
                ),
                "availability": "unavailable",
                "reason": request.get("reason", "not representable in the frozen block"),
                "vector_json": "",
                "vector_hash": "",
                "metadata_json": json.dumps(_json_ready(request), sort_keys=True),
                "proxy_substituted": False,
                "output_metrics_used_for_selection": False,
            }
        )
    return rows


def _write_core_reports(
    output: Path,
    canonical_rows: list[dict[str, Any]],
    canonical: CompiledSparseQSVT,
    canonical_state: StatevectorEvidence,
    canonical_resource: ResourceEvidence,
    second: CompiledSparseQSVT,
    second_state: StatevectorEvidence,
    second_resource: ResourceEvidence,
    second_shots: ShotEvidence,
    scaling: Any | None,
) -> None:
    failures = [row for row in canonical_rows if not bool(row["pass"])]
    report = f"""# Canonical reproduction report

The generic compiler reconstructed `{canonical.workload_id}` from structured inputs. {len(canonical_rows) - len(failures)} of {len(canonical_rows)} registered comparisons passed. The comparison used exact equality for identifiers, hashes, counts, and integer resources, and an absolute tolerance of `5e-15` for deterministic floating-point serialization.

## Numerical agreement

- Block reconstruction relative error: `{canonical_state.metrics['epsilon_block']:.17g}`.
- Sparse-versus-dense action error: `{canonical_state.metrics['sparse_dense_action_relative_error']:.17g}`.
- QSVT-versus-polynomial action error: `{canonical_state.metrics['epsilon_qsvt']:.17g}`.
- QSVT-versus-quantized-Ridge action error: `{canonical_state.metrics['qsvt_quantized_ridge_relative_error']:.17g}`.
- Postselection probability: `{canonical_state.metrics['sparse_postselection_probability']:.17g}`.

## Circuit and resource agreement

The compiler circuit has `{canonical_resource.record['total_simultaneously_live_qubits']}` simultaneously live qubits, `{canonical_resource.record['transpiled_gate_count']}` transpiled gates, depth `{canonical_resource.record['transpiled_depth']}`, `{canonical_resource.record['toffoli_count']}` Toffoli gates, and `{canonical_resource.record['controlled_rotation_count']}` controlled rotations. These integer quantities match the protected historical ledger.

## Differences

{('No registered scientific or circuit-resource mismatch was observed.' if not failures else 'Blocking mismatches: ' + '; '.join(str(row['criterion']) for row in failures))}
"""
    (output / "canonical_reproduction_report.md").write_text(report, encoding="utf-8")

    implementation = """# Implementation change log

## Additive compiler implementation

- `src/robust_qsvt_se/qsvt/generic_sparse_compiler.py` defines the typed seven-record contract, structured failures, deterministic hashes, sparse wrapper construction, QSVT composition, postselection, signed readout, and construction-only result.
- `src/robust_qsvt_se/qsvt/generic_sparse_workloads.py` isolates frozen workload registries from the generic compiler and enforces the outcome-independent IEEE-30 selection rules.
- `src/robust_qsvt_se/qsvt/generic_sparse_execution.py` performs ordered statevector validation, same-source-circuit transpilation, resource accounting, and Aer count sampling.
- `src/robust_qsvt_se/qsvt/generic_sparse_scaling.py` implements the one-factor-at-a-time compiled resource study and retains infeasible rows.
- `configs/generic_sparse_qsvt_compiler.json` records frozen hashes, selection rules, seeds, shot budgets, and the scaling grid.
- `scripts/run_generic_sparse_qsvt_compiler.py` generates only the new evidence root.

## Compatibility boundary

No historical public function was removed or changed. The legacy canonical generator remains callable. Scientific constants remain in workload adapters rather than the compiler core. The compiler supports real square power-of-two matrices and returns a structured rejection for rectangular matrices.
"""
    (output / "implementation_change_log.md").write_text(implementation, encoding="utf-8")

    if scaling is not None:
        def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
            def cell(value: Any) -> str:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return ""
                return str(value).replace("|", "\\|")

            header = "| " + " | ".join(columns) + " |"
            separator = "| " + " | ".join("---" for _ in columns) + " |"
            body = [
                "| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |"
                for row in rows
            ]
            return "\n".join([header, separator, *body])

        scaling_report = f"""# Compiled resource-scaling summary

Every nonfailed resource row below comes from a final measured circuit constructed by the generic compiler. The labels distinguish statevector execution, finite-shot execution, and transpilation-only evidence. No row is analytically modeled and mislabeled as compiled.

## Dimension

{table(scaling.dimension_rows, ['level', 'evidence_status', 'total_simultaneously_live_qubits', 'transpiled_gate_count', 'transpiled_depth', 'toffoli_count', 'controlled_rotation_count'])}

At fixed `s=3`, `b_v=6`, and `d=31`, compiled gates rise from 4 to 8 to 16 dimensions. The 16-dimensional row is transpilation-only.

## Slots

{table(scaling.slot_rows, ['level', 'evidence_status', 'failure_code', 'transpiled_gate_count', 'transpiled_depth', 'controlled_rotation_count'])}

The frozen canonical support requires three slots. Its `s=2` row is retained as infeasible. Four slots compile with additional lookup rotations and permutation cost.

## Value precision

{table(scaling.precision_rows, ['level', 'evidence_status', 'transpiled_gate_count', 'transpiled_depth', 'matrix_quantization_relative_error', 'selected_output_quantization_absolute_error'])}

The current direct-multiplexed access architecture compiles values directly into rotation angles, so changing the declared magnitude precision changes quantization error but not gate topology or gate count. This is a small-scale access limitation, not a precision-independent scalable oracle claim.

## Degree

{table(scaling.degree_rows, ['level', 'evidence_status', 'failure_code', 'transpiled_gate_count', 'transpiled_depth', 'qsvt_signal_calls', 'phase_count', 'polynomial_uniform_fit_error'])}

The compiled degree-15 and degree-31 points show approximately degree-proportional signal-wrapper repetition. Degree 63 is retained as failed because the existing polynomial construction violated the required global boundedness check. Three points do not establish an asymptotic theorem.

## Interpretation boundary

The measured relationship is descriptive: `G_total` consists mainly of repeated signal-wrapper calls plus preparation and postselection/readout overhead. These simulator and transpilation data do not establish scalable IEEE-size access, hardware feasibility, speedup, advantage, or practical competitiveness.
"""
        (output / "scaling_summary.md").write_text(scaling_report, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_root).resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"output root must be the protected non-overwriting target: {DEFAULT_OUTPUT}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output / "logs/core_execution.log", mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
    started = time.perf_counter()
    logging.info("compile canonical workload")
    canonical = compile_from_bundle(build_canonical_compiler_inputs())
    logging.info("compile frozen second workload")
    second = compile_from_bundle(build_second_ieee30_compiler_inputs())
    logging.info("statevector validate canonical")
    canonical_state = validate_compiled_statevector(canonical)
    logging.info("statevector validate second workload")
    second_state = validate_compiled_statevector(second)
    logging.info("transpile stored final circuits")
    canonical_prepared = prepare_compiled_execution(canonical)
    second_prepared = prepare_compiled_execution(second)
    canonical_resource = build_resource_evidence(canonical, canonical_prepared)
    second_resource = build_resource_evidence(second, second_prepared)
    canonical_checkpoint = output / "logs/canonical_shot_checkpoint.csv"
    canonical_summary_checkpoint = output / "logs/canonical_shot_summary_checkpoint.csv"
    second_checkpoint = output / "logs/second_shot_checkpoint.csv"
    second_summary_checkpoint = output / "logs/second_shot_summary_checkpoint.csv"
    if args.skip_shots:
        canonical_shots = ShotEvidence(canonical.workload_id, [], [])
        second_shots = ShotEvidence(second.workload_id, [], [])
    elif args.reuse_shot_checkpoints and all(
        path.is_file()
        for path in (
            canonical_checkpoint,
            canonical_summary_checkpoint,
            second_checkpoint,
            second_summary_checkpoint,
        )
    ):
        logging.info("reuse complete finite-shot checkpoints")
        canonical_shots = ShotEvidence(
            canonical.workload_id,
            pd.read_csv(canonical_checkpoint).to_dict(orient="records"),
            pd.read_csv(canonical_summary_checkpoint).to_dict(orient="records"),
        )
        second_shots = ShotEvidence(
            second.workload_id,
            pd.read_csv(second_checkpoint).to_dict(orient="records"),
            pd.read_csv(second_summary_checkpoint).to_dict(orient="records"),
        )
    else:
        logging.info("run canonical finite-shot grid")
        canonical_shots = run_compiled_shots(
            canonical, canonical_state, canonical_prepared
        )
        _write_csv(canonical_checkpoint, canonical_shots.rows)
        _write_csv(canonical_summary_checkpoint, canonical_shots.summary_rows)
        logging.info("run second-workload finite-shot grid")
        second_shots = run_compiled_shots(second, second_state, second_prepared)
        _write_csv(second_checkpoint, second_shots.rows)
        _write_csv(second_summary_checkpoint, second_shots.summary_rows)

    canonical_circuit_records = _save_circuits(
        output, "canonical", canonical, canonical_prepared
    )
    second_circuit_records = _save_circuits(output, "second", second, second_prepared)
    _write_csv(output / "circuit_artifact_registry.csv", canonical_circuit_records + second_circuit_records)
    _save_compiled_data(output, "canonical", canonical)
    _save_compiled_data(output, "second", second)

    canonical_rows = _canonical_reproduction_rows(
        canonical, canonical_state, canonical_resource, canonical_shots
    )
    _write_csv(output / "canonical_reproduction.csv", canonical_rows)
    _write_json(
        output / "second_workload_metadata.json",
        _second_metadata(second, second_state, second_resource, second_circuit_records),
    )
    functional_rows = _functional_registry(second)
    _write_csv(output / "second_workload_functional_registry.csv", functional_rows)
    _write_csv(output / "second_workload_statevector_validation.csv", second_state.functional_rows)
    _write_csv(output / "second_workload_shot_rows.csv", second_shots.rows)
    _write_csv(output / "second_workload_shot_summary.csv", second_shots.summary_rows)
    _write_csv(output / "second_workload_resource_ledger.csv", [second_resource.record])
    _write_csv(output / "second_workload_register_ledger.csv", second_resource.register_rows)
    _write_csv(output / "canonical_shot_rows_generic.csv", canonical_shots.rows)
    _write_csv(output / "canonical_shot_summary_generic.csv", canonical_shots.summary_rows)
    _write_csv(output / "canonical_resource_ledger_generic.csv", [canonical_resource.record])
    _write_csv(output / "canonical_register_ledger_generic.csv", canonical_resource.register_rows)
    _write_csv(
        output / "generic_compiler_validation.csv",
        _compiler_validation_rows(canonical, canonical_state)
        + _compiler_validation_rows(second, second_state),
    )

    scaling = None
    if not args.skip_scaling:
        logging.info("run compiled one-factor-at-a-time scaling study")
        scaling = run_compiled_scaling_study(
            canonical,
            canonical_resource,
            canonical_state,
            output_root=output,
        )
        _write_csv(output / "dimension_scaling.csv", scaling.dimension_rows)
        _write_csv(output / "slot_scaling.csv", scaling.slot_rows)
        _write_csv(output / "value_precision_scaling.csv", scaling.precision_rows)
        _write_csv(output / "degree_scaling.csv", scaling.degree_rows)

    failure_rows: list[dict[str, Any]] = [] if scaling is None else list(scaling.failure_rows)
    for row in functional_rows:
        if row["availability"] == "unavailable":
            failure_rows.append(
                {
                    "workstream": "second_workload_functionals",
                    "scaling_factor": "",
                    "level": "",
                    "failure_code": "metadata_functional_unavailable",
                    "stage": "functional_registry",
                    "reason": row["reason"],
                    "details_json": row["metadata_json"],
                    "retained": True,
                }
            )
    _write_csv(output / "failure_registry.csv", failure_rows)

    evidence_status = [
        {
            "evidence_id": "canonical_generic_reproduction",
            "status": "statevector and finite-shot executed" if not args.skip_shots else "statevector executed",
            "workload_id": canonical.workload_id,
            "artifact": "canonical_reproduction.csv",
        },
        {
            "evidence_id": "second_generic_end_to_end",
            "status": "statevector and finite-shot executed" if not args.skip_shots else "statevector executed",
            "workload_id": second.workload_id,
            "artifact": "second_workload_statevector_validation.csv",
        },
    ]
    if scaling is not None:
        for filename, rows in (
            ("dimension_scaling.csv", scaling.dimension_rows),
            ("slot_scaling.csv", scaling.slot_rows),
            ("value_precision_scaling.csv", scaling.precision_rows),
            ("degree_scaling.csv", scaling.degree_rows),
        ):
            for row in rows:
                evidence_status.append(
                    {
                        "evidence_id": f"{row['scaling_factor']}:{row['level']}",
                        "status": row["evidence_status"],
                        "workload_id": row.get("workload_id", ""),
                        "artifact": filename,
                    }
                )
    _write_csv(output / "evidence_status_registry.csv", evidence_status)
    _write_core_reports(
        output,
        canonical_rows,
        canonical,
        canonical_state,
        canonical_resource,
        second,
        second_state,
        second_resource,
        second_shots,
        scaling,
    )
    elapsed = time.perf_counter() - started
    _write_json(
        output / "execution_provenance.json",
        {
            "command": " ".join(sys.argv),
            "elapsed_seconds": elapsed,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "canonical_workload_digest": canonical.workload_digest,
            "second_workload_digest": second.workload_digest,
            "shot_grid_executed": not args.skip_shots,
            "scaling_grid_executed": not args.skip_scaling,
            "same_stored_source_circuit_for_statevector_distribution_transpilation_resources_and_shots": True,
            "dense_fallback_used": False,
            "direct_output_state_preparation_used": False,
        },
    )
    logging.info("core evidence complete in %.3f seconds", elapsed)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--reuse-shot-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
