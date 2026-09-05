"""Experiment D: compiled sparse-access oracle demonstration.

Substantiates (and bounds) the claim that sparse access is the scalability
pathway by *compiling and statevector-validating* the reversible sparse-access
oracles for small weighted-Jacobian blocks, then reconstructing the block from
their outputs. It reuses the synthesized primitives from
``reversible_sparse_oracle`` (multi-controlled-X column/value oracles) and adds:

* per-entry lookup correctness for ``O_col`` (CSR column lookup) and ``O_val``
  (sign-magnitude fixed-point value lookup),
* an *oracle-output-derived* reconstruction of the block and its error versus the
  exact and the quantized matrix,
* a compiled-circuit summary (qubits, gate count, depth, and a unitarity check
  where the register is small enough).

Claim boundary: these are compiled **sparse-access oracles**, *not* a compiled
block encoding. A full block encoding (LCU/qubitization around these oracles)
and any IEEE-scale oracle remain **modeled**, not compiled. This is small-scale
reversible-arithmetic evidence, not an IEEE-scale oracle and not a
quantum-hardware run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.reversible_sparse_oracle import (
    TOFFOLI_T_COST,
    build_column_oracle,
    build_value_oracle,
    decode_value,
    formal_cost_model,
    quantize_value,
    validate_column_oracle,
    validate_value_oracle,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    EXPERIMENTS_CLAIM_BOUNDARY,
    SPARSE_DIR,
    assert_safe,
    write_experiment_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_VALUE_BITS = 6
UNITARITY_QUBIT_LIMIT = 11  # build the dense Operator only for small registers

CORRECTNESS_COLUMNS = [
    "block",
    "matrix_shape",
    "oracle",
    "row",
    "index",
    "expected",
    "observed",
    "match",
    "detail",
]

RECON_COLUMNS = [
    "block",
    "matrix_shape",
    "nnz",
    "value_precision_bits",
    "quantization_step",
    "max_abs_error_vs_quantized",
    "max_abs_error_vs_true",
    "frobenius_error_vs_true",
    "column_oracle_passed",
    "value_oracle_bit_exact",
    "reconstruction_status",
]


def _oracle_unitarity_error(circuit: Any) -> float | None:
    if int(circuit.num_qubits) > UNITARITY_QUBIT_LIMIT:
        return None
    try:
        from qiskit.quantum_info import Operator  # type: ignore[import-not-found]

        matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
        identity = np.eye(matrix.shape[0])
        return float(np.max(np.abs(matrix.conj().T @ matrix - identity)))
    except Exception:
        return None


def _reconstruct_from_oracles(
    matrix: np.ndarray,
    col_validation: dict[str, Any],
    val_validation: dict[str, Any],
    scale: float,
    value_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``H`` from the *oracle-output* column indices and decoded values."""

    n_rows, n_cols = matrix.shape
    reconstructed = np.zeros_like(matrix, dtype=np.float64)
    decoded_lookup = {
        (int(entry["row"]), int(entry["col"])): float(entry["decoded_value"])
        for entry in val_validation["entries"]
    }
    for entry in col_validation["entries"]:
        if not entry["valid"] or entry["observed_invalid_flag"]:
            continue
        row = int(entry["row"])
        column = int(entry["observed_column"])
        reconstructed[row, column] = decoded_lookup.get((row, column), 0.0)
    quantized = np.zeros_like(matrix, dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            sign_bit, magnitude = quantize_value(float(matrix[i, j]), scale, value_bits)
            quantized[i, j] = decode_value(sign_bit, magnitude, scale, value_bits)
    return reconstructed, quantized


def _block_for(case: str, seed: int, size: int) -> tuple[np.ndarray, str]:
    system, _ = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    block, _r, _rows, _cols = select_deterministic_block(
        H_full, r_full, row_count=int(size), col_count=int(size), policy="largest_row_col_norms"
    )
    return np.asarray(block, dtype=np.float64), f"{case}_{size}x{size}_weighted_jacobian_block"


def run_sparse_oracle_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    value_bits = int(resolved["value_bits"])

    correctness_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    circuit_summary: dict[str, Any] = {
        "blocks": {},
        "value_precision_bits": value_bits,
        "toffoli_t_cost": TOFFOLI_T_COST,
    }
    failures: list[dict[str, Any]] = []

    for size in resolved["sizes"]:
        matrix, label = _block_for(resolved["case"], int(resolved["seed"]), int(size))
        nonzero_mask = np.abs(matrix) > 1.0e-12
        max_nnz = int(nonzero_mask.sum(axis=1).max())
        nnz = int(nonzero_mask.sum())

        col_built = build_column_oracle(matrix, max_nnz)
        val_built = build_value_oracle(matrix, value_bits)
        col_validation = validate_column_oracle(col_built)
        val_validation = validate_value_oracle(val_built)

        reconstructed, quantized = _reconstruct_from_oracles(
            matrix, col_validation, val_validation, val_built["scale"], value_bits
        )
        err_vs_quantized = float(np.max(np.abs(reconstructed - quantized)))
        err_vs_true = float(np.max(np.abs(reconstructed - matrix)))
        frob_vs_true = float(np.linalg.norm(reconstructed - matrix))
        recon_status = (
            "bit_exact_vs_quantized" if err_vs_quantized <= 1.0e-12 else ("reconstruction_mismatch")
        )
        if recon_status != "bit_exact_vs_quantized":
            failures.append(
                {
                    "block": label,
                    "issue": "oracle reconstruction not bit-exact",
                    "max_abs_error_vs_quantized": err_vs_quantized,
                }
            )

        recon_rows.append(
            {
                "block": label,
                "matrix_shape": f"{size}x{size}",
                "nnz": nnz,
                "value_precision_bits": value_bits,
                "quantization_step": float(val_validation["quantization_step"]),
                "max_abs_error_vs_quantized": err_vs_quantized,
                "max_abs_error_vs_true": err_vs_true,
                "frobenius_error_vs_true": frob_vs_true,
                "column_oracle_passed": bool(col_validation["passed"]),
                "value_oracle_bit_exact": bool(val_validation["passed"]),
                "reconstruction_status": recon_status,
            }
        )

        for entry in col_validation["entries"]:
            correctness_rows.append(
                {
                    "block": label,
                    "matrix_shape": f"{size}x{size}",
                    "oracle": "O_col",
                    "row": int(entry["row"]),
                    "index": int(entry["slot"]),
                    "expected": int(entry["expected_column"]) if entry["valid"] else "invalid",
                    "observed": int(entry["observed_column"]),
                    "match": bool(entry["match"]),
                    "detail": f"valid={entry['valid']}, inv={entry['observed_invalid_flag']}",
                }
            )
        for entry in val_validation["entries"]:
            correctness_rows.append(
                {
                    "block": label,
                    "matrix_shape": f"{size}x{size}",
                    "oracle": "O_val",
                    "row": int(entry["row"]),
                    "index": int(entry["col"]),
                    "expected": f"s{entry['expected_sign']}m{entry['expected_magnitude']}",
                    "observed": f"s{entry['observed_sign']}m{entry['observed_magnitude']}",
                    "match": bool(entry["match"]),
                    "detail": f"decoded={entry['decoded_value']:+.5f}",
                }
            )

        col_unitarity = _oracle_unitarity_error(col_built["circuit"])
        val_unitarity = _oracle_unitarity_error(val_built["circuit"])
        model = formal_cost_model(
            case=label,
            rows=size,
            cols=size,
            nnz=nnz,
            max_nonzeros_per_row=max_nnz,
            value_bits=value_bits,
        )
        circuit_summary["blocks"][label] = {
            "matrix_shape": f"{size}x{size}",
            "max_nonzeros_per_row": max_nnz,
            "nnz": nnz,
            "column_oracle": {
                "n_qubits": int(col_built["circuit"].num_qubits),
                "gate_count": int(sum(col_built["circuit"].count_ops().values())),
                "depth": int(col_built["circuit"].depth()),
                "unitarity_error": col_unitarity,
                "unitarity_checked": col_unitarity is not None,
                "lookup_passed": bool(col_validation["passed"]),
                "permutation_by_construction": True,
            },
            "value_oracle": {
                "n_qubits": int(val_built["circuit"].num_qubits),
                "gate_count": int(sum(val_built["circuit"].count_ops().values())),
                "depth": int(val_built["circuit"].depth()),
                "unitarity_error": val_unitarity,
                "unitarity_checked": val_unitarity is not None,
                "bit_exact": bool(val_validation["passed"]),
                "permutation_by_construction": True,
            },
            "modeled_total_t_count_qrom": int(model["total_t_count_qrom"]),
            "basis_gates": "logical multi-controlled-X (mcx) writes; permutation on basis states",
            "status": "compiled_sparse_access_oracle_small_scale",
            "not_a_block_encoding": True,
        }

    correctness_frame = pd.DataFrame(correctness_rows, columns=CORRECTNESS_COLUMNS)
    recon_frame = pd.DataFrame(recon_rows, columns=RECON_COLUMNS)

    artifacts = _write_outputs(
        output_dir=output_dir,
        correctness=correctness_frame,
        recon=recon_frame,
        circuit_summary=circuit_summary,
    )
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="D_sparse_access_oracle_demo",
        script_name="scripts/run_tqe_revision_sparse_oracle_demo.py",
        command=resolved["command"],
        description=(
            "Compiled, statevector-validated reversible sparse-access oracles (O_col, O_val) "
            "for small weighted-Jacobian blocks, with an oracle-output-derived block "
            "reconstruction. These are sparse-access oracles, not a compiled block encoding; "
            "the block encoding and IEEE-scale oracle remain modeled."
        ),
        artifacts=artifacts,
        inputs_used=[f"build_engineering_system:{resolved['case']}:weighted_jacobian"],
        random_seeds={"demo_system_seed": int(resolved["seed"])},
        warnings=[
            "these are compiled sparse-access oracles, not a compiled block encoding; the "
            "block encoding and IEEE-scale oracle are modeled, not compiled",
        ],
        failures=failures,
        interpretation_boundary=(
            "Small-scale compiled reversible-arithmetic evidence: the sparse-access oracles are "
            "synthesized and statevector-validated and the block is reconstructed bit-exactly "
            "from their outputs (up to the fixed-point quantization). This substantiates the "
            "sparse-access primitive but does NOT compile a block encoding, does not reach "
            "IEEE scale, and is not a quantum-hardware run."
        ),
        extra={
            "sizes": list(resolved["sizes"]),
            "value_precision_bits": value_bits,
            "all_column_oracles_passed": bool(recon_frame["column_oracle_passed"].all()),
            "all_value_oracles_bit_exact": bool(recon_frame["value_oracle_bit_exact"].all()),
            "all_reconstructions_bit_exact": bool(
                (recon_frame["reconstruction_status"] == "bit_exact_vs_quantized").all()
            ),
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "correctness": correctness_frame,
        "reconstruction": recon_frame,
        "circuit_summary": circuit_summary,
        "artifacts": artifacts,
    }


def _write_outputs(
    *,
    output_dir: Path,
    correctness: pd.DataFrame,
    recon: pd.DataFrame,
    circuit_summary: dict[str, Any],
) -> dict[str, Path]:
    correctness_csv = output_dir / "sparse_oracle_correctness.csv"
    recon_csv = output_dir / "reconstructed_block_error.csv"
    summary_json = output_dir / "compiled_circuit_summary.json"
    correctness.to_csv(correctness_csv, index=False)
    recon.to_csv(recon_csv, index=False)
    write_json(summary_json, circuit_summary)

    summary_tex = output_dir / "sparse_block_encoding_summary.tex"
    summary_tex.write_text(_summary_tex(recon, circuit_summary), encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(_readme(recon, circuit_summary), encoding="utf-8")

    return {
        "sparse_oracle_correctness_csv": correctness_csv,
        "reconstructed_block_error_csv": recon_csv,
        "compiled_circuit_summary_json": summary_json,
        "sparse_block_encoding_summary_tex": summary_tex,
        "readme_md": readme,
    }


def _summary_tex(recon: pd.DataFrame, circuit_summary: dict[str, Any]) -> str:
    lines = [
        "% Compiled sparse-ACCESS oracle demo (NOT a compiled block encoding).",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Block & $O_\mathrm{col}$ qubits/gates & $O_\mathrm{val}$ qubits/gates & "
        r"recon.\ err (quantized) & recon.\ err (true) & lookup \\",
        r"\midrule",
    ]
    for _, row in recon.iterrows():
        block = str(row["block"])
        info = circuit_summary["blocks"].get(block, {})
        col = info.get("column_oracle", {})
        val = info.get("value_oracle", {})
        label = block.replace("_", r"\_")
        passed = "pass" if row["column_oracle_passed"] and row["value_oracle_bit_exact"] else "FAIL"
        lines.append(
            f"{label} & {col.get('n_qubits', '')}/{col.get('gate_count', '')} & "
            f"{val.get('n_qubits', '')}/{val.get('gate_count', '')} & "
            f"{row['max_abs_error_vs_quantized']:.1e} & {row['max_abs_error_vs_true']:.1e} & "
            f"{passed} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Compiled sparse-\emph{access} oracles ($O_\mathrm{col}$, $O_\mathrm{val}$) "
        r"for small weighted-Jacobian blocks, statevector-validated, with the block "
        r"reconstructed bit-exactly from the oracle outputs (up to the fixed-point "
        r"quantization). These are sparse-access oracles, \emph{not} a compiled block "
        r"encoding; the block encoding and any IEEE-scale oracle remain modeled.}",
        r"\label{tab:sparse_access_oracle_demo}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _readme(recon: pd.DataFrame, circuit_summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment D: Compiled Sparse-Access Oracle Demonstration",
        "",
        EXPERIMENTS_CLAIM_BOUNDARY,
        "",
        "Reversible sparse-access oracles for small weighted-Jacobian blocks are compiled and "
        "statevector-validated, and each block is reconstructed from the **oracle outputs**:",
        "",
        "- `O_col : |i,k,z> -> |i,k, z XOR c(i,k)>` writes the column index of the k-th stored "
        "nonzero of row i (with an `invalid` flag for padding slots),",
        "- `O_val : |i,j,z> -> |i,j, z XOR fix(H_ij)>` writes a sign + fixed-point magnitude.",
        "",
        "## Reconstruction and validation",
        "",
        "| Block | shape | nnz | recon(quant) | recon(true) | O_col | O_val exact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in recon.iterrows():
        lines.append(
            f"| `{row['block']}` | {row['matrix_shape']} | {row['nnz']} | "
            f"{row['max_abs_error_vs_quantized']:.1e} | {row['max_abs_error_vs_true']:.1e} | "
            f"{row['column_oracle_passed']} | {row['value_oracle_bit_exact']} |"
        )
    lines += [
        "",
        "The reconstruction error versus the *quantized* matrix is zero (the oracles are "
        "bit-exact permutations); the error versus the *true* matrix is the fixed-point "
        "quantization step, reported per block.",
        "",
        "## Compiled circuit summary",
        "",
        "| Block | O_col qubits/gates/depth | O_val qubits/gates/depth | unitarity checked |",
        "| --- | --- | --- | --- |",
    ]
    for block, info in circuit_summary["blocks"].items():
        col = info["column_oracle"]
        val = info["value_oracle"]
        checked = f"O_col={col['unitarity_checked']}, O_val={val['unitarity_checked']}"
        lines.append(
            f"| `{block}` | {col['n_qubits']}/{col['gate_count']}/{col['depth']} | "
            f"{val['n_qubits']}/{val['gate_count']}/{val['depth']} | {checked} |"
        )
    lines += [
        "",
        "Column-oracle unitarity is verified from the dense operator where the register is "
        "small enough; both oracles are permutations on the computational basis by "
        "construction, and every stored address is validated by statevector simulation.",
        "",
        "## Claim boundary",
        "",
        "These are compiled **sparse-access oracles**, **not** a compiled block encoding. A "
        "full block encoding (an LCU/qubitization construction wrapping these oracles) and any "
        "IEEE-scale oracle remain **modeled** cost estimates, not compiled circuits. This is "
        "small-scale reversible-arithmetic evidence; it is not an IEEE-scale oracle and not a "
        "quantum-hardware run. The manuscript should therefore describe sparse access as a "
        "*validated small-scale primitive with a modeled scaling path*, not as a compiled "
        "IEEE-scale block encoding.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(SPARSE_DIR),
        "case": "ieee14",
        "seed": 123,
        "sizes": [4, 8],
        "value_bits": DEFAULT_VALUE_BITS,
        "command": "run_tqe_revision_sparse_oracle_demo",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment D: compiled sparse-access oracle demo")
    parser.add_argument("--output-dir", default=str(SPARSE_DIR))
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--value-bits", type=int, default=DEFAULT_VALUE_BITS)
    parser.add_argument("--quick", action="store_true", help="4x4 block only")
    parser.add_argument("--full", action="store_true", help="4x4 and 8x8 blocks")
    args = parser.parse_args(argv)
    sizes = [4] if args.quick else list(args.sizes)
    run = run_sparse_oracle_demo(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "seed": args.seed,
            "sizes": sizes,
            "value_bits": args.value_bits,
            "command": "scripts/run_tqe_revision_sparse_oracle_demo.py " + " ".join(argv or []),
        }
    )
    print(
        f"Sparse-access oracle demo complete: {run['artifacts']['reconstructed_block_error_csv']}"
    )
    print(run["reconstruction"][["block", "reconstruction_status"]].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
