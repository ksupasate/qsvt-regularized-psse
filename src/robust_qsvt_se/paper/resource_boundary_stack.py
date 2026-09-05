"""Task 4: layered resource-boundary stack.

Assembles a single audit table that separates *implemented* circuit-level
components from *modeled* / *proxy* / *excluded* components of the
selected-observable QSVT pathway. Numbers are pulled from the artifacts produced
by Tasks 1-3 (demo circuit stats, normalization record, circuit-level readout,
reversible-oracle resource summary); symbolic quantities are labelled modeled or
proxy and never presented as execution on quantum hardware.

Implemented-status taxonomy:

* ``implemented``  - synthesized and validated at small simulator scale,
* ``proxy``        - a small-simulator stand-in (e.g. dense amplitude loading),
* ``modeled``      - a formal cost model / symbolic factor only,
* ``excluded``     - explicitly out of scope,
* ``future_work``  - acknowledged gap left for later work.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    DEMO_DIR,
    ORACLE_DIR,
    RESOURCE_STACK_DIR,
    assert_safe,
    write_demo_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory

STACK_COLUMNS = [
    "component",
    "role",
    "implemented_status",
    "evidence_file",
    "qubits",
    "gates",
    "depth",
    "oracle_calls",
    "shots",
    "postselection_factor",
    "modeled_factor",
    "remaining_gap",
    "safe_manuscript_wording",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def build_resource_stack(
    *, demo_dir: Path, oracle_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    circuit_stats = _load_json(demo_dir / "circuit_stats.json")
    normalization = _load_json(demo_dir / "normalization_record.json")
    pipeline = _load_json(demo_dir / "qsvt_pipeline_metadata.json")
    readout_meta = _load_json(demo_dir / "readout_metadata.json")
    oracle_meta = _load_json(oracle_dir / "oracle_metadata.json")

    readout_csv = demo_dir / "readout_diagnostics.csv"
    readout_shots = ""
    readout_backend = ""
    if readout_csv.is_file():
        try:
            readout_frame = pd.read_csv(readout_csv)
            if not readout_frame.empty and "shots" in readout_frame:
                readout_shots = int(readout_frame["shots"].max())
                readout_backend = str(readout_frame["backend"].iloc[0])
        except Exception:
            pass

    oracle_summary_csv = oracle_dir / "oracle_resource_summary.csv"
    oracle_synth_qubits = ""
    oracle_synth_gates = ""
    oracle_synth_depth = ""
    oracle_modeled_t = ""
    if oracle_summary_csv.is_file():
        try:
            oracle_frame = pd.read_csv(oracle_summary_csv)
            synth = oracle_frame[oracle_frame["status"] == "synthesized_small_scale"]
            if not synth.empty:
                oracle_synth_gates = int(synth["col_oracle_synth_gate_count"].iloc[0])
                oracle_synth_depth = int(synth["col_oracle_synth_depth"].iloc[0])
                oracle_synth_qubits = int(
                    synth["row_index_qubits"].iloc[0]
                    + synth["slot_index_qubits"].iloc[0]
                    + synth["column_register_qubits"].iloc[0]
                    + 1
                )
            modeled = oracle_frame[oracle_frame["status"] == "modeled"]
            if not modeled.empty:
                oracle_modeled_t = int(modeled["total_t_count_qrom"].max())
        except Exception:
            pass

    num_qubits = circuit_stats.get("num_qubits", "")
    raw_gates = _fmt(circuit_stats.get("raw_gate_count"))
    raw_depth = _fmt(circuit_stats.get("raw_circuit_depth"))
    degree = pipeline.get("degree", circuit_stats.get("degree", ""))
    phase_count = pipeline.get("phase_count", circuit_stats.get("phase_count", ""))
    success = normalization.get("postselection_probability", "")
    success_factor = ""
    amplification_repeats = ""
    if isinstance(success, (int, float)) and 0.0 < float(success) <= 1.0:
        success_factor = round(float(success), 4)
        amplification_repeats = round(1.0 / math.sqrt(float(success)), 3)

    demo_block = pipeline.get("block_shape", "")
    rel = demo_dir.name

    rows: list[dict[str, Any]] = [
        {
            "component": "dense block encoding",
            "role": "embed A = H_B^T/beta in a unitary (top-left block)",
            "implemented_status": "implemented",
            "evidence_file": f"{rel}/circuit_stats.json",
            "qubits": num_qubits,
            "gates": raw_gates,
            "depth": raw_depth,
            "oracle_calls": "",
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": "dense (non-scalable) primitive",
            "remaining_gap": "scalable sparse/oracle block-encoding is modeled, not compiled",
            "safe_manuscript_wording": (
                "Dense block-encoding gate validated at small simulator scale for the "
                f"{demo_block} block."
            ),
        },
        {
            "component": "QSVT phase synthesis",
            "role": "phases realizing the bounded odd filter polynomial",
            "implemented_status": "implemented",
            "evidence_file": f"{rel}/qsvt_pipeline_metadata.json",
            "qubits": num_qubits,
            "gates": _fmt(phase_count),
            "depth": "",
            "oracle_calls": _fmt(phase_count),
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": "",
            "remaining_gap": "high-degree (>~45) synthesis is the binding ceiling",
            "safe_manuscript_wording": (
                "PennyLane phase synthesis succeeded for the headline block; the "
                "high-degree synthesis ceiling bounds the 8x8 boundary case."
            ),
        },
        {
            "component": "polynomial degree",
            "role": "approximation degree of the regularized filter",
            "implemented_status": "implemented",
            "evidence_file": f"{rel}/qsvt_pipeline_metadata.json",
            "qubits": "",
            "gates": "",
            "depth": "",
            "oracle_calls": _fmt(degree),
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": "",
            "remaining_gap": "degree is the accuracy lever for ill-conditioned blocks",
            "safe_manuscript_wording": (
                f"Odd polynomial degree {degree} reproduces the matched Ridge filter on "
                "the headline block."
            ),
        },
        {
            "component": "residual state preparation",
            "role": "prepare |r_hat> = r_B/||r_B|| in the system register",
            "implemented_status": "proxy",
            "evidence_file": f"{rel}/circuit_stats.json",
            "qubits": num_qubits,
            "gates": "",
            "depth": "",
            "oracle_calls": "",
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": "dense amplitude loading",
            "remaining_gap": "an efficient scalable state-preparation is not proven",
            "safe_manuscript_wording": (
                "Residual state prepared by dense amplitude loading (small-simulator "
                "proxy, not an efficiency proof)."
            ),
        },
        {
            "component": "postselection",
            "role": "select the encoded success branch of the ancilla",
            "implemented_status": "implemented",
            "evidence_file": f"{rel}/normalization_record.json",
            "qubits": num_qubits,
            "gates": "",
            "depth": "",
            "oracle_calls": "",
            "shots": "",
            "postselection_factor": _fmt(success_factor),
            "modeled_factor": (
                f"~{amplification_repeats} repetitions without amplification"
                if amplification_repeats != ""
                else ""
            ),
            "remaining_gap": "amplitude amplification to boost success is modeled below",
            "safe_manuscript_wording": (
                "Postselection on the encoded branch is simulator-validated; the success "
                "probability is reported, not amplified."
            ),
        },
        {
            "component": "signed readout",
            "role": "estimate y_l = l^T dx_alpha (one selected functional)",
            "implemented_status": "implemented",
            "evidence_file": f"{rel}/readout_diagnostics.csv",
            "qubits": "" if num_qubits == "" else int(num_qubits) + 1,
            "gates": "",
            "depth": "",
            "oracle_calls": "",
            "shots": _fmt(readout_shots),
            "postselection_factor": "",
            "modeled_factor": f"backend: {readout_backend}" if readout_backend else "",
            "remaining_gap": "one functional per query; sampling cost ~ 1/eps^2",
            "safe_manuscript_wording": (
                "Circuit-level Hadamard-test overlap readout recovers the signed "
                "functional with shot sampling."
            ),
        },
        {
            "component": "sparse index/value access",
            "role": "reversible O_col / O_val sparse-access oracles",
            "implemented_status": "implemented",
            "evidence_file": f"{oracle_dir.name}/oracle_resource_summary.csv",
            "qubits": _fmt(oracle_synth_qubits),
            "gates": _fmt(oracle_synth_gates),
            "depth": _fmt(oracle_synth_depth),
            "oracle_calls": "",
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": (
                f"IEEE-scale modeled, up to ~{oracle_modeled_t} T (QROM)"
                if oracle_modeled_t != ""
                else "IEEE-scale modeled"
            ),
            "remaining_gap": "IEEE-scale oracle is a formal cost model, not compiled",
            "safe_manuscript_wording": (
                "Small reversible column/value oracles are synthesized and "
                "statevector-validated; IEEE-scale costs are modeled."
            ),
        },
        {
            "component": "amplitude amplification",
            "role": "boost the postselection success probability",
            "implemented_status": "modeled",
            "evidence_file": f"{rel}/normalization_record.json",
            "qubits": "",
            "gates": "",
            "depth": "",
            "oracle_calls": _fmt(amplification_repeats),
            "shots": "",
            "postselection_factor": _fmt(success_factor),
            "modeled_factor": (
                f"~{amplification_repeats} amplified rounds (modeled)"
                if amplification_repeats != ""
                else "O(1/sqrt(p_success)) rounds (modeled)"
            ),
            "remaining_gap": "not synthesized; symbolic O(1/sqrt(p_success)) factor",
            "safe_manuscript_wording": (
                "Amplitude amplification is a modeled O(1/sqrt(p_success)) factor, not "
                "an implemented circuit."
            ),
        },
        {
            "component": "full-vector recovery",
            "role": "recover every signed component of dx_alpha",
            "implemented_status": "excluded",
            "evidence_file": "n/a",
            "qubits": "",
            "gates": "",
            "depth": "",
            "oracle_calls": "",
            "shots": "",
            "postselection_factor": "",
            "modeled_factor": "one functional per coordinate",
            "remaining_gap": "out of scope; not attempted",
            "safe_manuscript_wording": (
                "Full-vector readout is excluded; each query returns one selected "
                "functional, not the whole update vector."
            ),
        },
    ]

    counts = {
        status: int(sum(1 for row in rows if row["implemented_status"] == status))
        for status in ("implemented", "proxy", "modeled", "excluded", "future_work")
    }
    provenance = {
        "demo_artifacts_present": bool(circuit_stats),
        "oracle_artifacts_present": bool(oracle_meta),
        "readout_artifacts_present": bool(readout_meta),
        "status_counts": counts,
    }
    return rows, provenance


def run_resource_boundary_stack(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows, provenance = build_resource_stack(
        demo_dir=Path(resolved["demo_dir"]), oracle_dir=Path(resolved["oracle_dir"])
    )
    frame = pd.DataFrame(rows, columns=STACK_COLUMNS)

    csv_path = output_dir / "resource_boundary_stack.csv"
    md_path = output_dir / "resource_boundary_stack.md"
    tex_path = output_dir / "resource_boundary_stack.tex"
    frame.to_csv(csv_path, index=False)
    md_text = _markdown(frame, provenance)
    assert_safe(md_text)
    md_path.write_text(md_text, encoding="utf-8")
    tex_text = _latex(frame)
    assert_safe(tex_text)
    tex_path.write_text(tex_text, encoding="utf-8")

    artifacts = {
        "resource_boundary_stack_csv": csv_path,
        "resource_boundary_stack_md": md_path,
        "resource_boundary_stack_tex": tex_path,
    }
    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="resource_boundary_stack",
        description=(
            "Layered resource-boundary stack separating implemented circuit-level "
            "components from modeled/proxy/excluded ones. Symbolic quantities are labelled "
            "modeled or proxy and are not presented as execution on quantum hardware."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[resolved["demo_dir"], resolved["oracle_dir"]],
        extra={"status_counts": provenance["status_counts"]},
        manifest_name="resource_boundary_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "stack": frame,
        "provenance": provenance,
        "artifacts": artifacts,
    }


def _markdown(frame: pd.DataFrame, provenance: dict[str, Any]) -> str:
    lines = [
        "# Layered Resource-Boundary Stack",
        "",
        "Separates **implemented** circuit-level components from **modeled**, **proxy**, "
        "and **excluded** ones for the selected-observable QSVT pathway. Symbolic resource "
        "quantities are labelled modeled or proxy and are **not** presented as "
        "execution on quantum hardware.",
        "",
        "Status counts: "
        + ", ".join(f"{key}={value}" for key, value in provenance["status_counts"].items()),
        "",
        "| Component | Role | Status | Evidence | Qubits | Gates | Depth | Shots | "
        "Postselect | Modeled factor | Remaining gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['component']} | {row['role']} | `{row['implemented_status']}` | "
            f"`{row['evidence_file']}` | {row['qubits']} | {row['gates']} | {row['depth']} | "
            f"{row['shots']} | {row['postselection_factor']} | {row['modeled_factor']} | "
            f"{row['remaining_gap']} |"
        )
    lines += [
        "",
        "## Safe manuscript wording (per component)",
        "",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"- **{row['component']}** (`{row['implemented_status']}`): "
            f"{row['safe_manuscript_wording']}"
        )
    lines.append("")
    return "\n".join(lines)


def _latex(frame: pd.DataFrame) -> str:
    def esc(text: Any) -> str:
        return (
            str(text)
            .replace("\\", " ")
            .replace("&", r"\&")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("^", " ")
            .replace("~", " ")
        )

    lines = [
        "% Layered resource-boundary stack (modeled/proxy quantities are labelled).",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Component & Status & Qubits & Gates/Depth & Remaining gap \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        gates_depth = f"{esc(row['gates'])}/{esc(row['depth'])}"
        lines.append(
            f"{esc(row['component'])} & {esc(row['implemented_status'])} & "
            f"{esc(row['qubits'])} & {gates_depth} & {esc(row['remaining_gap'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Layered resource-boundary stack for the selected-observable QSVT "
        r"pathway. Implemented rows are synthesized and validated at small simulator "
        r"scale; modeled and proxy rows are formal cost models or small-simulator "
        r"stand-ins, not execution on quantum hardware.}",
        r"\label{tab:resource_boundary_stack}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(RESOURCE_STACK_DIR),
        "demo_dir": str(DEMO_DIR),
        "oracle_dir": str(ORACLE_DIR),
        "command": "run_resource_boundary_stack",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the layered resource-boundary stack")
    parser.add_argument("--output-dir", default=str(RESOURCE_STACK_DIR))
    parser.add_argument("--demo-dir", default=str(DEMO_DIR))
    parser.add_argument("--oracle-dir", default=str(ORACLE_DIR))
    args = parser.parse_args(argv)
    run = run_resource_boundary_stack(
        {
            "output_dir": args.output_dir,
            "demo_dir": args.demo_dir,
            "oracle_dir": args.oracle_dir,
            "command": "scripts/build_resource_boundary_stack.py " + " ".join(argv or []),
        }
    )
    print(f"Resource boundary stack complete: {run['artifacts']['resource_boundary_stack_csv']}")
    print(f"Status counts: {run['provenance']['status_counts']}")


if __name__ == "__main__":  # pragma: no cover
    main()
