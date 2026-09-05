"""Goal F: component-level resource ledger (not an end-to-end fault-tolerant estimate).

Each row is one workload component with its own scale, unit, and evidence status.
Wall-clock classical timings and quantum signal-unitary-call counts are different units and are
never merged into a single figure of merit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/component_resource_ledger")
RESOURCE = Path(
    "outputs/tqe_revision_experiments/end_to_end_resource_case/fixed_case_resource_ledger.csv"
)
ADJOINT = Path(
    "outputs/tqe_revision_experiments/end_to_end_resource_case/classical_adjoint_baseline.csv"
)
BASELINE = Path("outputs/classical_selected_observable_baseline/baseline_summary.csv")
WRAPPER = Path("outputs/sparse_block_encoding_wrapper_demo/wrapper_demo_results.csv")
ORACLE_TABLE = Path("outputs/sparse_oracle_assumption_ledger/ieee_resource_table.csv")

HEADLINE_NOTE = "This is not an end-to-end fault-tolerant resource estimate."

COLUMNS = [
    "component",
    "scale",
    "measured_or_modeled",
    "value",
    "unit",
    "repeated_per_attempt",
    "included_in_table_6_style_count",
    "excluded_from_count",
    "evidence_source",
    "limitation",
]


def _resource_lookup() -> dict[str, str]:
    if not RESOURCE.is_file():
        return {}
    frame = pd.read_csv(RESOURCE)
    return {str(row["field"]): str(row["value"]) for _, row in frame.iterrows()}


def _fmt_sci(raw: str | None, digits: int = 3) -> str:
    if raw is None:
        return "not available"
    try:
        return f"{float(raw):.{digits}e}"
    except (TypeError, ValueError):
        return str(raw)


def build_rows() -> list[dict[str, str]]:
    fixed = _resource_lookup()
    adjoint = pd.read_csv(ADJOINT) if ADJOINT.is_file() else None
    baseline = pd.read_csv(BASELINE) if BASELINE.is_file() else None
    wrapper = pd.read_csv(WRAPPER) if WRAPPER.is_file() else None
    oracle = pd.read_csv(ORACLE_TABLE) if ORACLE_TABLE.is_file() else None

    def baseline_time(case: str, method: str) -> str:
        if baseline is None:
            return "not available"
        rows = baseline[(baseline["case"] == case) & (baseline["method"] == method)]
        if not len(rows):
            return "not available"
        return f"{float(rows.iloc[0]['runtime_seconds']):.3e}"

    def adjoint_time(method: str) -> str:
        if adjoint is None:
            return "not available"
        rows = adjoint[adjoint["method"] == method]
        if not len(rows):
            return "not available"
        return f"{float(rows.iloc[0]['median_runtime_seconds']):.3e}"

    ieee300_tcount = "not available"
    if oracle is not None:
        row300 = oracle[oracle["case"] == "ieee300"]
        if len(row300):
            r = row300.iloc[0]
            ieee300_tcount = f"{7 * (int(r['nnz']) + int(r['max_row_sparsity']) * 1722):.3e}"

    rows = [
        {
            "component": "polynomial degree",
            "scale": "4x4 anchor / 8x8 secondary anchor",
            "measured_or_modeled": "measured",
            "value": fixed.get("degree", "not available"),
            "unit": "degree",
            "repeated_per_attempt": "n/a (fixed per workload)",
            "included_in_table_6_style_count": "yes (d signal calls)",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "degree-45 phase-synthesis ceiling in the tested pipeline",
        },
        {
            "component": "phase count",
            "scale": "4x4 anchor circuit",
            "measured_or_modeled": "measured",
            "value": fixed.get("phase_count", "not available"),
            "unit": "phases",
            "repeated_per_attempt": "applied once per attempt",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "PennyLane iterative angle solver",
        },
        {
            "component": "signal-unitary calls per attempt",
            "scale": "4x4 anchor circuit",
            "measured_or_modeled": "measured from circuit sequence",
            "value": fixed.get("signal_unitary_calls_per_attempt", "not available"),
            "unit": "U_A or U_A^dagger calls/attempt",
            "repeated_per_attempt": "yes (definitionally)",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "dense encoding gate at this scale, not a sparse oracle",
        },
        {
            "component": "postselection probability",
            "scale": "4x4 anchor circuit (prepared residual)",
            "measured_or_modeled": "measured (statevector)",
            "value": _fmt_sci(fixed.get("postselection_probability"), 4),
            "unit": "probability",
            "repeated_per_attempt": "multiplies attempts by 1/p_succ",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "block/residual specific; sweep uses a uniform-input proxy",
        },
        {
            "component": "isolated signed-overlap shots",
            "scale": "one signed functional at 1e-2 relative error",
            "measured_or_modeled": "measured under assumed output-state preparation",
            "value": _fmt_sci(fixed.get("shots_for_target_error")),
            "unit": "shots",
            "repeated_per_attempt": "defines the attempt count",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": (
                "per functional; grows linearly with q; not integrated with residual "
                "loading, QSVT, and postselection"
            ),
        },
        {
            "component": "total signal-unitary calls",
            "scale": "one functional, 4x4 anchor",
            "measured_or_modeled": "modeled from measured factors",
            "value": _fmt_sci(fixed.get("total_signal_unitary_calls_without_AA")),
            "unit": "U_A or U_A^dagger calls",
            "repeated_per_attempt": "aggregate over shots/p_succ",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "no amplitude amplification synthesized",
        },
        {
            "component": "sparse-oracle calls",
            "scale": "one functional, 4x4 anchor",
            "measured_or_modeled": "modeled (~2 per signal-unitary call)",
            "value": _fmt_sci(fixed.get("oracle_call_proxy")),
            "unit": "oracle calls",
            "repeated_per_attempt": "aggregate over shots/p_succ",
            "included_in_table_6_style_count": "yes",
            "excluded_from_count": "-",
            "evidence_source": str(RESOURCE),
            "limitation": "call-count proxy; wrapper compiled only at toy 4x4",
        },
        {
            "component": "residual-state preparation",
            "scale": "blocks: dense loading / full scale: model",
            "measured_or_modeled": "measured (blocks); modeled (full scale)",
            "value": "dense initialize (exact); O(2^{q_r}) generic proxy",
            "unit": "circuit/model",
            "repeated_per_attempt": "yes (every shot and postselection attempt)",
            "included_in_table_6_style_count": "yes (as repeated T_prep)",
            "excluded_from_count": "-",
            "evidence_source": "outputs/residual_loading_ledger/residual_loading_ledger.csv",
            "limitation": "no scalable structured residual loader compiled",
        },
        {
            "component": "sparse lookup forward T-count",
            "scale": "IEEE 300 full matrix",
            "measured_or_modeled": "modeled (QROM unary iteration)",
            "value": ieee300_tcount,
            "unit": "T gates (forward pair)",
            "repeated_per_attempt": "per block-encoding call",
            "included_in_table_6_style_count": "no",
            "excluded_from_count": "full-scale execution not compiled",
            "evidence_source": str(ORACLE_TABLE),
            "limitation": "forward lookup only; reverse/uncompute in per-call factor",
        },
        {
            "component": "sparse block-encoding wrapper",
            "scale": "toy 4x4 sparsified quantized block",
            "measured_or_modeled": "measured (toy compiled + statevector validated)",
            "value": (
                f"{int(wrapper.iloc[0]['qubits'])} qubits, "
                f"{int(wrapper.iloc[0]['transpiled_gate_count'])} u3/cx ops, depth "
                f"{int(wrapper.iloc[0]['transpiled_depth'])}"
                if wrapper is not None
                else "not available (wrapper demo not run)"
            ),
            "unit": "circuit",
            "repeated_per_attempt": "per block-encoding call",
            "included_in_table_6_style_count": "no",
            "excluded_from_count": "IEEE-scale wrapper not compiled",
            "evidence_source": str(WRAPPER),
            "limitation": "in-place permutations exist only for edge-colorable toy patterns",
        },
        {
            "component": "controlled readout",
            "scale": "4x4 anchor circuit",
            "measured_or_modeled": "isolated shot circuit; output state assumed prepared",
            "value": "1 readout ancilla; direct StatePreparation(output_state)",
            "unit": "circuit",
            "repeated_per_attempt": "yes (every shot)",
            "included_in_table_6_style_count": "yes (as T_read)",
            "excluded_from_count": "-",
            "evidence_source": (
                "outputs/tqe_revision_experiments/readout_statistics/readout_seed_results.csv"
            ),
            "limitation": (
                "not integrated with residual loading, QSVT, or postselection; "
                "not full-vector recovery"
            ),
        },
        {
            "component": "amplitude-estimation alternative",
            "scale": "discussed alternative",
            "measured_or_modeled": "modeled",
            "value": _fmt_sci(fixed.get("total_signal_unitary_calls_with_AA_proxy")),
            "unit": "controlled sequence uses (proxy)",
            "repeated_per_attempt": "coherent repetitions inside each AE call",
            "included_in_table_6_style_count": "no",
            "excluded_from_count": "controlled prep/QSVT/readout not synthesized",
            "evidence_source": str(RESOURCE),
            "limitation": "idealized 1/(eps sqrt(p_succ)) dependence",
        },
        {
            "component": "classical dense selected-output solve",
            "scale": "4x4 block (matched alpha)",
            "measured_or_modeled": "measured wall-clock",
            "value": adjoint_time("dense_direct_full_update"),
            "unit": "seconds (median, 30 repeats)",
            "repeated_per_attempt": "n/a (classical reference)",
            "included_in_table_6_style_count": "reference only (different unit)",
            "excluded_from_count": "-",
            "evidence_source": str(ADJOINT),
            "limitation": "diagnostic timing on one arm64 host; not hardware-normalized",
        },
        {
            "component": "classical sparse selected-output solve",
            "scale": "IEEE 300 full weighted system",
            "measured_or_modeled": "measured wall-clock",
            "value": baseline_time("ieee300", "sparse_factorized"),
            "unit": "seconds (median, 30 repeats)",
            "repeated_per_attempt": "n/a (classical reference)",
            "included_in_table_6_style_count": "reference only (different unit)",
            "excluded_from_count": "-",
            "evidence_source": str(BASELINE),
            "limitation": "shared factorization amortized across functionals",
        },
        {
            "component": "classical adjoint selected-output solve",
            "scale": "4x4 block (matched alpha, same functional)",
            "measured_or_modeled": "measured wall-clock",
            "value": adjoint_time("dense_adjoint_selected_observable"),
            "unit": "seconds (median, 30 repeats)",
            "repeated_per_attempt": "n/a (classical reference)",
            "included_in_table_6_style_count": "reference only (different unit)",
            "excluded_from_count": "-",
            "evidence_source": str(ADJOINT),
            "limitation": ("cheaper for the demonstrated fixed workload; not a scaling statement"),
        },
        {
            "component": "full error correction",
            "scale": "any",
            "measured_or_modeled": "excluded",
            "value": "not provided",
            "unit": "-",
            "repeated_per_attempt": "-",
            "included_in_table_6_style_count": "no",
            "excluded_from_count": "fault-tolerant/hardware-level estimate not provided",
            "evidence_source": "-",
            "limitation": "logical-level accounting only",
        },
    ]
    return rows


def _tex_escape(text: str) -> str:
    out = text.replace("{", r"\{").replace("}", r"\}")
    for char, repl in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("^", "\\textasciicircum{}"),
        ("~", "\\textasciitilde{}"),
    ]:
        out = out.replace(char, repl)
    return out


def _write_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"% Source: outputs/component_resource_ledger/component_resource_ledger.csv",
        r"% Regenerate: .venv/bin/python scripts/build_component_resource_ledger.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Component-level resource ledger for the selected-submatrix workload. "
        + HEADLINE_NOTE
        + r" Wall-clock classical timings (seconds) and quantum operation counts are different "
        r"units and are never merged; the classical adjoint row is a fixed-case reference, "
        r"not a scaling comparison.}",
        r"\label{tab:component_resource_ledger}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.155\textwidth}p{0.135\textwidth}p{0.115\textwidth}"
        r"p{0.145\textwidth}p{0.075\textwidth}p{0.10\textwidth}p{0.16\textwidth}}",
        r"\hline",
        r"Component & Scale & Status & Value & Unit & In aggregate? & Limitation \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        included = str(row["included_in_table_6_style_count"])
        excluded = str(row["excluded_from_count"])
        in_count = included if excluded in ("-", "") else f"no ({excluded})"
        cells = [
            _tex_escape(str(row["component"])),
            _tex_escape(str(row["scale"])),
            _tex_escape(str(row["measured_or_modeled"])),
            _tex_escape(str(row["value"])),
            _tex_escape(str(row["unit"])),
            _tex_escape(in_count),
            _tex_escape(str(row["limitation"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "/") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(build_rows(), columns=COLUMNS)
    frame.to_csv(OUTPUT_DIR / "component_resource_ledger.csv", index=False)
    _write_tex(frame, OUTPUT_DIR / "component_resource_ledger.tex")
    md = [
        "# Component-Level Resource Ledger (Goal F)",
        "",
        f"**{HEADLINE_NOTE}** Wall-clock classical timings and quantum signal-unitary-call",
        "counts are different units and are never merged. The fixed-case conclusion is that the",
        "classical adjoint solve is cheaper for the demonstrated fixed workload; this is",
        "not a scaling statement.",
        "",
        _to_markdown(frame),
        "",
    ]
    (OUTPUT_DIR / "component_resource_ledger.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "artifact_name": "component_resource_ledger",
        "headline_note": HEADLINE_NOTE,
        "sources": {
            "fixed_case_resource_ledger": str(RESOURCE),
            "classical_adjoint_baseline": str(ADJOINT),
            "classical_baseline_summary": str(BASELINE),
            "wrapper_demo": str(WRAPPER),
            "oracle_resource_table": str(ORACLE_TABLE),
        },
        "regeneration_command": ".venv/bin/python scripts/build_component_resource_ledger.py",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Component resource ledger written to {OUTPUT_DIR} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
