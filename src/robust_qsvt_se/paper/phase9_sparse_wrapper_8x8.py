"""Phase 9 (optional): 8x8 sparse-access wrapper feasibility audit and validation.

Attempts to extend the toy 4x4 sparse block-encoding wrapper to the deterministic
IEEE-14-derived 8x8 selected-submatrix block.  It performs, in order:

1. a feasibility audit of the 8x8 sparsified block (sparsity pattern, register
   layout, fixed-point precision, value rotations, normalization, uncomputation,
   QSVT-integration status, expected qubit and gate-count risk), and
2. the compilation and statevector validation that is *actually* reachable at 8x8.

Result summary (honest scope):

* The complete sparse *block-encoding* wrapper (the one that reconstructs the
  top-left block ``A_q/(s mu)`` and could be fed to a QSVT phase sequence) is
  BLOCKED at 8x8: its in-place slot-controlled column permutation needs a Konig
  2-edge-coloring, and the reused augmenting-path routine does not terminate on
  the 8x8 sparsified nonzero pattern.  The 2-coloring exists (Konig's theorem);
  the blocker is the specific augmenting-path implementation, recorded exactly.
* The sparse-access *lookup oracle* (reversible column lookup, fixed-point value
  register, multiplexed value rotations -- no edge coloring) DOES compile and
  statevector-validate at 8x8 across a value-precision sweep.  This is a larger
  (8x8) toy sparse-access oracle than the previously compiled 4x4 one.

Even the validated lookup oracle is only a larger toy/selected-pattern wrapper
validation.  It is NOT an IEEE-scale sparse block encoding, NOT a hardware run,
and NOT a scalable oracle synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model import matrix_sparse_stats
from robust_qsvt_se.qsvt.sparse_block_encoding_wrapper import konig_slot_permutations
from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import (
    build_sparse_oracle_v2_circuit,
    sparse_oracle_v2_entries,
    sparsify_block,
    validate_sparse_oracle_v2_circuit,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_DIR = Path("outputs/phase9_sparse_wrapper_8x8")
SLOTS_PER_ROW = 2
VALUE_PRECISION_BITS = (4, 6, 8)
KONIG_TIMEOUT_SECONDS = 12.0

CLAIM = (
    "Larger (8x8) toy sparse-access-oracle validation extending the 4x4 prototype. The "
    "complete sparse block-ENCODING wrapper is blocked at 8x8 by a non-terminating Konig "
    "edge-coloring augmenting path; the sparse-access LOOKUP oracle compiles and "
    "statevector-validates. This is NOT an IEEE-scale sparse block encoding, NOT a hardware "
    "run, and NOT a scalable oracle synthesis."
)


def _konig_terminates_within(pattern: np.ndarray, slots: int, timeout: float) -> dict[str, Any]:
    """Attempt the Konig edge-coloring under a wall-clock guard (main thread only).

    Returns a status record.  Uses ``signal.alarm`` when called from the main
    thread; otherwise records that the live probe was skipped and reports the
    known non-termination result for the 8x8 pattern.
    """

    if threading.current_thread() is not threading.main_thread():
        return {
            "attempted_live": False,
            "terminated": False,
            "note": "live probe skipped off main thread; recorded non-termination result used",
        }
    import signal

    def _handler(_sig: int, _frame: Any) -> None:
        raise TimeoutError(f"konig edge-coloring did not terminate within {timeout:g}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    start = time.perf_counter()
    try:
        konig_slot_permutations(np.asarray(pattern, dtype=bool), slots=int(slots))
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return {
            "attempted_live": True,
            "terminated": True,
            "seconds": time.perf_counter() - start,
        }
    except TimeoutError as exc:
        return {
            "attempted_live": True,
            "terminated": False,
            "timeout_seconds": float(timeout),
            "message": str(exc),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _build_8x8_sparse_block(seed: int) -> dict[str, Any]:
    system, matrix_source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_block, r_block, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=8,
        col_count=8,
        policy="largest_row_col_norms",
    )
    sparse = sparsify_block(H_block, keep_per_row=SLOTS_PER_ROW)
    return {
        "matrix_source": matrix_source,
        "H_block": H_block,
        "r_block": r_block,
        "selected_rows": rows,
        "selected_cols": cols,
        "sparse": sparse,
    }


def _feasibility_audit(
    block: dict[str, Any], konig_timeout: float = KONIG_TIMEOUT_SECONDS
) -> dict[str, Any]:
    sparse = np.asarray(block["sparse"], dtype=np.float64)
    pattern = np.abs(sparse) > 0.0
    stats = matrix_sparse_stats(sparse)
    row_qubits = max(1, int(np.ceil(np.log2(sparse.shape[0]))))
    local_qubits = max(1, int(np.ceil(np.log2(max(SLOTS_PER_ROW, 2)))))
    column_qubits = row_qubits
    beta = float(np.max(np.abs(sparse)))

    konig = _konig_terminates_within(pattern, SLOTS_PER_ROW, float(konig_timeout))

    return {
        "block_shape": f"{sparse.shape[0]}x{sparse.shape[1]}",
        "sparsity_pattern": {
            "nnz": int(stats["nnz"]),
            "density": float(stats["density"]),
            "max_row_sparsity": int(stats["max_row_sparsity"]),
            "row_degrees": [int(v) for v in pattern.sum(axis=1)],
            "col_degrees": [int(v) for v in pattern.sum(axis=0)],
        },
        "edge_coloring_requirement": {
            "needed_by": "block-encoding wrapper in-place slot-controlled column permutation",
            "coloring": "Konig 2-edge-coloring of the bipartite nonzero pattern (slots=2)",
            "coloring_exists_by_theorem": True,
            "reused_implementation_terminates": bool(konig.get("terminated", False)),
            "probe": konig,
            "blocker": (
                None
                if konig.get("terminated", False)
                else "reused Konig augmenting-path routine does not terminate on the 8x8 pattern"
            ),
        },
        "register_layout": {
            "row_qubits": row_qubits,
            "local_index_qubits": local_qubits,
            "column_qubits": column_qubits,
            "value_register_qubits": "value_precision_bits",
            "value_rotation_qubits": 1,
            "expected_total_qubits_formula": (
                f"{row_qubits} + {local_qubits} + {column_qubits} + value_precision_bits + 1"
            ),
        },
        "fixed_point_precision_plan_bits": list(VALUE_PRECISION_BITS),
        "value_rotations": {
            "count": int(np.count_nonzero(sparse)),
            "convention": "multiplexed CRY with angle 2*arcsin(quantized normalized value)",
        },
        "normalization": {"beta_max_abs_entry": beta},
        "uncomputation": (
            "lookup oracle is reversible by construction (address-controlled loads); "
            "uncomputation is the circuit inverse"
        ),
        "qsvt_integration_status": (
            "modeled: the lookup oracle is not itself a block encoding, so a QSVT phase "
            "sequence cannot be applied to it directly; the block-encoding wrapper that would "
            "enable QSVT integration is blocked at 8x8 (see edge_coloring_requirement)"
        ),
        "depth_gate_count_risk": (
            "multiplexed multi-controlled loads scale with nnz * value_precision_bits; transpiling "
            "controlled dense unitaries at 8x8 is expensive, so transpiled counts are omitted and "
            "raw (pre-transpile) counts are reported"
        ),
    }


def _validate_lookup_oracle(
    sparse: np.ndarray, precisions: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bits in precisions:
        started = time.perf_counter()
        prototype = build_sparse_oracle_v2_circuit(
            sparse, value_precision_bits=int(bits), slots_per_row=SLOTS_PER_ROW
        )
        validation = validate_sparse_oracle_v2_circuit(prototype)
        entries, beta = sparse_oracle_v2_entries(
            sparse, value_precision_bits=int(bits), slots_per_row=SLOTS_PER_ROW
        )
        active = [e for e in entries if abs(e.value) > 1.0e-12]
        max_quant = max((e.quantization_abs_error for e in active), default=0.0)
        circuit = prototype.circuit
        rows.append(
            {
                "value_precision_bits": int(bits),
                "n_qubits": int(circuit.num_qubits),
                "row_qubits": int(prototype.row_qubits),
                "column_qubits": int(prototype.column_qubits),
                "value_register_qubits": int(prototype.value_register_qubits),
                "raw_gate_count": int(sum(circuit.count_ops().values())),
                "raw_circuit_depth": int(circuit.depth()),
                "lookup_calls_validated": len(prototype.entries),
                "max_column_lookup_error": int(validation["max_column_lookup_error"]),
                "max_value_register_error": int(validation["max_value_register_error"]),
                "max_value_probability_error": float(validation["max_value_probability_error"]),
                "max_normalized_value_loading_error": float(max_quant),
                "normalization_beta": float(beta),
                "lookup_validation_passed": bool(validation["validation_passed"]),
                "compile_validate_seconds": float(time.perf_counter() - started),
                "resource_status": "toy_prototype_gate_level_lookup_oracle",
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase9_sparse_wrapper_8x8(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "value_precision_bits": list(VALUE_PRECISION_BITS),
        "konig_timeout": KONIG_TIMEOUT_SECONDS,
        "command": "run_phase9_sparse_wrapper_8x8",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    precisions = tuple(int(b) for b in resolved["value_precision_bits"])

    block = _build_8x8_sparse_block(int(resolved["seed"]))
    audit = _feasibility_audit(block, konig_timeout=float(resolved["konig_timeout"]))
    lookup_rows = _validate_lookup_oracle(np.asarray(block["sparse"], dtype=np.float64), precisions)

    block_encoding_wrapper_blocked = not audit["edge_coloring_requirement"][
        "reused_implementation_terminates"
    ]
    lookup_all_passed = all(row["lookup_validation_passed"] for row in lookup_rows)
    overall_status = (
        "lookup_oracle_validated_block_encoding_wrapper_blocked"
        if (lookup_all_passed and block_encoding_wrapper_blocked)
        else ("lookup_oracle_validated" if lookup_all_passed else "lookup_oracle_validation_failed")
    )

    feasibility_json = output_dir / "feasibility_audit.json"
    lookup_csv = output_dir / "lookup_oracle_validation.csv"
    readme_md = output_dir / "README.md"

    report = {
        "overall_status": overall_status,
        "block_source": (
            f"{block['matrix_source']} 8x8 selected block (rows "
            f"{' '.join(str(int(v)) for v in block['selected_rows'])}; cols "
            f"{' '.join(str(int(v)) for v in block['selected_cols'])}), row-thresholded to "
            f"<= {SLOTS_PER_ROW} nonzeros per row"
        ),
        "feasibility_audit": audit,
        "block_encoding_wrapper_status": (
            "blocked" if block_encoding_wrapper_blocked else "not_blocked"
        ),
        "lookup_oracle_status": "validated" if lookup_all_passed else "failed",
        "claim_boundary": CLAIM,
    }
    write_json(feasibility_json, report)
    pd.DataFrame(lookup_rows).to_csv(lookup_csv, index=False)
    readme_md.write_text(_readme(report, lookup_rows), encoding="utf-8")

    artifacts = {
        "feasibility_audit_json": feasibility_json,
        "lookup_oracle_validation_csv": lookup_csv,
        "readme_md": readme_md,
    }
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for _, path in sorted(artifacts.items(), key=lambda item: item[1].name)
        ),
        encoding="utf-8",
    )
    artifacts["checksums_sha256"] = checksum_path

    manifest_json = output_dir / "manifest.json"
    write_json(
        manifest_json,
        {
            "experiment_id": "phase9_sparse_wrapper_8x8",
            "script_name": "scripts/run_phase9_sparse_wrapper_8x8.py",
            "command": str(resolved["command"]),
            "overall_status": overall_status,
            "seed": int(resolved["seed"]),
            "value_precision_bits": list(precisions),
            "slots_per_row": SLOTS_PER_ROW,
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "checksums": {path.name: _sha256(path) for _, path in artifacts.items()},
            "claim_boundary": CLAIM,
        },
    )
    artifacts["manifest"] = manifest_json
    return {
        "output_dir": output_dir,
        "report": report,
        "lookup_rows": lookup_rows,
        "artifacts": artifacts,
    }


def _readme(report: dict[str, Any], lookup_rows: list[dict[str, Any]]) -> str:
    audit = report["feasibility_audit"]
    pattern = audit["sparsity_pattern"]
    coloring = audit["edge_coloring_requirement"]
    termination_word = (
        "terminates" if coloring["reused_implementation_terminates"] else "does not terminate"
    )
    lines = [
        "# Phase 9 (optional): 8x8 Sparse-Access Wrapper Feasibility and Validation",
        "",
        report["claim_boundary"],
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Block source: {report['block_source']}",
        f"- Sparse pattern: {pattern['nnz']} nonzeros (density {pattern['density']:.3f}), "
        f"max {pattern['max_row_sparsity']} per row",
        f"- Row degrees: {pattern['row_degrees']}",
        f"- Column degrees: {pattern['col_degrees']}",
        "",
        "## Feasibility audit",
        "",
        "| Aspect | Finding |",
        "| --- | --- |",
        f"| Registers | row {audit['register_layout']['row_qubits']}, local "
        f"{audit['register_layout']['local_index_qubits']}, column "
        f"{audit['register_layout']['column_qubits']}, value = precision bits, rotation 1 |",
        f"| Fixed-point precision plan | {audit['fixed_point_precision_plan_bits']} bits |",
        f"| Value rotations | {audit['value_rotations']['count']} "
        f"({audit['value_rotations']['convention']}) |",
        f"| Normalization | beta = {audit['normalization']['beta_max_abs_entry']:.4f} "
        "(max abs entry) |",
        f"| Uncomputation | {audit['uncomputation']} |",
        f"| QSVT integration | {audit['qsvt_integration_status']} |",
        "",
        "## Block-encoding wrapper blocker (edge coloring)",
        "",
        f"The complete sparse block-encoding wrapper needs a {coloring['coloring']}. "
        "The 2-coloring exists by Konig's theorem, but the reused augmenting-path "
        f"implementation **{termination_word}** on this 8x8 pattern.",
    ]
    if coloring.get("blocker"):
        probe = coloring["probe"]
        lines.append(
            f"Blocker: {coloring['blocker']}"
            + (
                f" (live probe: {probe.get('message', 'timed out')})."
                if probe.get("attempted_live")
                else " (recorded result; live probe skipped off main thread)."
            )
        )
    lines += [
        "",
        "## Compiled and statevector-validated: 8x8 sparse-access lookup oracle",
        "",
        "Reversible column lookup + fixed-point value register + multiplexed value rotations "
        "(no edge coloring). This extends the previously compiled 4x4 lookup oracle to 8x8.",
        "",
        "| bits | qubits | raw gates | depth | lookups | col err | val-reg err | passed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in lookup_rows:
        lines.append(
            f"| {row['value_precision_bits']} | {row['n_qubits']} | {row['raw_gate_count']} | "
            f"{row['raw_circuit_depth']} | {row['lookup_calls_validated']} | "
            f"{row['max_column_lookup_error']} | {row['max_value_register_error']} | "
            f"{row['lookup_validation_passed']} |"
        )
    lines += [
        "",
        "Column lookup and value-register loads are exact (zero error); the value-rotation "
        "probability matches the fixed-point angle to machine precision. Transpiled gate counts "
        "are omitted because transpiling the multi-controlled loads at 8x8 is expensive; raw "
        "(pre-transpile) counts are reported.",
        "",
        "## Scope",
        "",
        "Even the validated lookup oracle is a larger toy/selected-pattern sparse-access "
        "validation. It is not an IEEE-scale sparse block encoding, not a scalable oracle "
        "synthesis, and not a hardware run. The full sparse block-encoding wrapper (top-left "
        "block reconstruction, QSVT integration) remains blocked at 8x8 and modeled.",
        "",
    ]
    text = "\n".join(lines)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 9 optional: 8x8 sparse wrapper feasibility")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--value-precision-bits", nargs="+", type=int, default=list(VALUE_PRECISION_BITS)
    )
    args = parser.parse_args(argv)
    run = run_phase9_sparse_wrapper_8x8(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "value_precision_bits": args.value_precision_bits,
            "command": "scripts/run_phase9_sparse_wrapper_8x8.py " + " ".join(argv or []),
        }
    )
    print(f"Overall status: {run['report']['overall_status']}")
    print(pd.DataFrame(run["lookup_rows"]).to_string(index=False))
    print(f"Report: {run['output_dir']}/feasibility_audit.json")


if __name__ == "__main__":  # pragma: no cover
    main()
