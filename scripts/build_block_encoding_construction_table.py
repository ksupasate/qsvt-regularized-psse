"""Emit the manuscript block-encoding construction-chain status table (Edit 5).

Each stage of the CSR-to-readout chain is labeled compiled, modeled, or excluded,
with the compiled scale taken from generated artifacts (lookup demo, wrapper demo,
anchor circuit).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WRAPPER = Path("outputs/sparse_block_encoding_wrapper_demo/wrapper_demo_results.csv")
LOOKUP = Path(
    "outputs/tqe_revision_experiments/sparse_block_encoding_demo/reconstructed_block_error.csv"
)
DENSE_BLOCKS = Path("outputs/ieee_qsvt_pipeline_boundary/block_encoding_report.csv")
WORKLOAD = Path("outputs/qsvt_selected_workload_extension/selected_workload_results.csv")
READOUT = Path("outputs/tqe_revision_experiments/readout_statistics/readout_seed_results.csv")
RESOURCE_LEDGER = Path("outputs/component_resource_ledger/component_resource_ledger.csv")
TARGET = Path("manuscript/tables/block_encoding_construction_status.tex")


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required construction-table evidence is missing: {path}")
    return pd.read_csv(path)


def _shape_key(shape: str) -> int:
    return int(shape.lower().split("x", maxsplit=1)[0])


def _tex_shape(shape: str) -> str:
    return "$" + shape.lower().replace("x", r"\times") + "$"


def main() -> None:
    wrapper = _read_required(WRAPPER).iloc[0]
    lookup = _read_required(LOOKUP)
    dense_blocks = _read_required(DENSE_BLOCKS)
    workload = _read_required(WORKLOAD)
    readout = _read_required(READOUT)
    resource_ledger = _read_required(RESOURCE_LEDGER)

    if not (lookup["column_oracle_passed"].all() and lookup["value_oracle_bit_exact"].all()):
        raise ValueError("Sparse lookup evidence contains a failed validation row")
    if not dense_blocks.loc[
        dense_blocks["path"] == "selected_block_executable", "validation_passed"
    ].all():
        raise ValueError("Dense block-encoding evidence contains a failed validation row")
    if not readout["sampling_mode"].str.contains("circuit_shot_sampling").all():
        raise ValueError("Readout evidence is not finite-shot circuit sampling")
    if not resource_ledger["component"].eq("sparse block-encoding wrapper").any():
        raise ValueError("Component resource ledger is missing the toy-wrapper row")

    lookup_shapes = sorted(lookup["matrix_shape"].astype(str).unique(), key=_shape_key)
    lookup_scale = "/".join(_tex_shape(shape) for shape in lookup_shapes)
    lookup_bits = int(lookup["value_precision_bits"].max())
    selected_dense = dense_blocks[dense_blocks["path"] == "selected_block_executable"]
    dense_shapes = sorted(selected_dense["block_shape"].astype(str).unique(), key=_shape_key)
    dense_scale = f"{_tex_shape(dense_shapes[0])}--{_tex_shape(dense_shapes[-1])}"
    anchor = workload[workload["workload_id"] == "anchor_4x4_benchmark_alpha_d31"].iloc[0]
    anchor_degree = int(anchor["degree_attempted"])

    wrapper_scale = (
        f"toy $4\\times4$ sparsified quantized block "
        f"({int(wrapper['qubits'])} qubits, {int(wrapper['transpiled_gate_count'])} u3/cx ops)"
    )
    wrapper_status = (
        "compiled (toy)"
        if str(wrapper["status"]) in ("compiled", "statevector_validated")
        else "modeled"
    )
    integration_ok = str(wrapper["qsvt_integration_status"]) == "statevector_validated"
    qsvt_integration = "compiled (toy)" if integration_ok else "modeled"

    rows = [
        ("CSR data (weighted Jacobian)", "classical input", "generated IEEE/PYPOWER rows"),
        (
            r"$O_{\rm col}$ index lookup",
            "compiled",
            lookup_scale + r" blocks (Table~\ref{tab:revision_sparse_oracle_validation})",
        ),
        (
            r"$O_{\rm val}$ value lookup",
            "compiled",
            f"{lookup_bits}-bit sign-magnitude words, bit-exact",
        ),
        (
            "fixed-point arithmetic pipeline",
            "modeled",
            "rotation angles enumerated at toy scale only",
        ),
        (
            "normalization logic ($s\\mu$, $\\beta$)",
            "recorded classically",
            "classical factors in every recovery",
        ),
        ("wrapper (diffusion + permutations + value rotations)", wrapper_status, wrapper_scale),
        (
            r"block encoding $U_A$",
            f"{wrapper_status} / dense instrument",
            f"dense dilation {dense_scale}; sparse wrapper at toy scale",
        ),
        (
            "QSVT phase sequence on $U_A$",
            qsvt_integration,
            f"degree {int(wrapper['qsvt_integration_degree'])} through the toy wrapper; "
            f"degree {anchor_degree} anchor circuit",
        ),
        ("postselection", "compiled (statevector)", r"$p_{\rm succ}$ measured per block/residual"),
        (
            "isolated signed-overlap readout",
            "executed with assumed output-state preparation",
            "direct StatePreparation of the classically computed postselected state",
        ),
        ("IEEE-scale sparse block encoding", "modeled", "QROM lookup cost model only"),
        ("full-vector readout", "excluded", "outside the selected-output scope"),
    ]

    lines = [
        "% Sources: wrapper_demo_results.csv; reconstructed_block_error.csv;",
        "% block_encoding_report.csv; selected_workload_results.csv;",
        "% readout_seed_results.csv; component_resource_ledger.csv",
        "% Regenerate: .venv/bin/python scripts/run_sparse_block_encoding_wrapper_demo.py"
        " && .venv/bin/python scripts/build_block_encoding_construction_table.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Construction and validation status for the selected-submatrix tests. "
        r"Each stage is"
        r" compiled, modeled, recorded classically, or excluded; compiled stages state their"
        r" validated scale. The toy wrapper makes the previously modeled wrapper stage"
        r" executable at $4\times4$ scale; the IEEE-scale sparse block encoding remains"
        r" modeled. The isolated signed-overlap shot experiment is not an integrated"
        r" residual-loading--QSVT--postselection--readout circuit.}",
        r"\label{tab:block_encoding_construction}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.36\columnwidth}p{0.20\columnwidth}p{0.32\columnwidth}}",
        r"\hline",
        r"Stage & Status & Scale / evidence \\",
        r"\hline",
    ]
    for stage, status, scale in rows:
        lines.append(f"{stage} & {status} & {scale} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
