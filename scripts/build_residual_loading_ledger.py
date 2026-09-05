"""Goal D: residual-state loading ledger.

Makes explicit that residual loading is a repeated in-loop cost: it recurs inside
the sampling/postselection loop and, in nonlinear AC, after every residual and
Jacobian rebuild. Values are taken from generated artifacts only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/residual_loading_ledger")
STATE_PREP_BLOCKS = Path("outputs/ieee_qsvt_pipeline_boundary/state_preparation_report.csv")
STATE_PREP_MODELS = Path("outputs/qsvt_state_preparation_model/state_preparation_summary.csv")
RESOURCE = Path(
    "outputs/tqe_revision_experiments/end_to_end_resource_case/fixed_case_resource_ledger.csv"
)
NONLINEAR_TABLE = Path("manuscript/tables/nonlinear_convergence_revision.tex")

CAPTION_CONCLUSION = (
    "Residual loading would repeat inside an integrated sampling/postselection loop. "
    "The present finite-shot experiment directly prepares the computed output state, so "
    "this repetition is modeled, not executed; in nonlinear AC it would also repeat after "
    "each residual/Jacobian rebuild."
)

COLUMNS = [
    "loading_model",
    "scale",
    "implemented_status",
    "cost_model",
    "repeated_per_shot",
    "repeated_per_postselection_attempt",
    "repeated_per_nonlinear_iteration",
    "evidence_file",
    "limitation",
]


def build_rows() -> list[dict[str, str]]:
    blocks = pd.read_csv(STATE_PREP_BLOCKS) if STATE_PREP_BLOCKS.is_file() else None
    block_evidence = str(STATE_PREP_BLOCKS) if blocks is not None else "not available"
    block_gates = (
        f"{int(blocks['gate_count'].min())}--{int(blocks['gate_count'].max())} gates, "
        f"fidelity error {blocks['state_preparation_l2_error'].max():.1e} max"
        if blocks is not None
        else "not available"
    )
    models = pd.read_csv(STATE_PREP_MODELS) if STATE_PREP_MODELS.is_file() else None

    def model_row(name: str, column: str) -> str:
        if models is None:
            return "not available"
        rows = models[models["preparation_model"] == name]
        if not len(rows):
            return "not available"
        value = rows.iloc[0][column]
        return "not recorded" if pd.isna(value) else str(value)

    return [
        {
            "loading_model": "dense selected-block amplitude initialization",
            "scale": "4x4--16x16 selected blocks",
            "implemented_status": f"implemented + statevector validated ({block_gates})",
            "cost_model": "exact dense initialize, O(2^{q_r}) rotations",
            "repeated_per_shot": "yes",
            "repeated_per_postselection_attempt": "yes",
            "repeated_per_nonlinear_iteration": "yes (residual changes each rebuild)",
            "evidence_file": block_evidence,
            "limitation": "simulator instrument; not an efficient-preparation proof",
        },
        {
            "loading_model": "selected-block residual preparation fidelity",
            "scale": "4x4--16x16 selected blocks",
            "implemented_status": (
                "measured fidelity 1.0 "
                f"(L2 error {blocks['state_preparation_l2_error'].max():.1e} max)"
                if blocks is not None
                else "not available"
            ),
            "cost_model": "included in dense initialization",
            "repeated_per_shot": "yes",
            "repeated_per_postselection_attempt": "yes",
            "repeated_per_nonlinear_iteration": "yes",
            "evidence_file": block_evidence,
            "limitation": "verified against the normalized weighted residual only",
        },
        {
            "loading_model": "generic full-matrix amplitude loading proxy",
            "scale": "full IEEE weighted systems (m up to 1722)",
            "implemented_status": "modeled only (no compiled loader)",
            "cost_model": (
                "O(2^{q_r}) elementary rotations, q_r = ceil(log2 m); depth proxy "
                + model_row("exact_dense_amplitude_loading", "estimated_depth_proxy")
                + " (IEEE 14)"
            ),
            "repeated_per_shot": "yes",
            "repeated_per_postselection_attempt": "yes",
            "repeated_per_nonlinear_iteration": "yes",
            "evidence_file": str(STATE_PREP_MODELS),
            "limitation": "exponential without exploitable structure",
        },
        {
            "loading_model": "structured/qRAM-like residual access assumption",
            "scale": "full IEEE weighted systems",
            "implemented_status": "assumed access model; not implemented",
            "cost_model": (
                "oracle query cost "
                + model_row("qram_amplitude_oracle", "estimated_query_cost")
                + ", depth proxy "
                + model_row("qram_amplitude_oracle", "estimated_depth_proxy")
                + " (IEEE 14)"
            ),
            "repeated_per_shot": "yes",
            "repeated_per_postselection_attempt": "yes",
            "repeated_per_nonlinear_iteration": "yes",
            "evidence_file": str(STATE_PREP_MODELS),
            "limitation": "data-loading hardware is not synthesized",
        },
        {
            "loading_model": "nonlinear AC per-iteration residual recomputation",
            "scale": "IEEE 14/30/57 Gauss-Newton loops",
            "implemented_status": "classical loop implemented; QSVT not executed inside",
            "cost_model": "classical h(x_k), Jacobian rebuild each iteration",
            "repeated_per_shot": "n/a (classical)",
            "repeated_per_postselection_attempt": "n/a (classical)",
            "repeated_per_nonlinear_iteration": "yes (3--4 iterations measured)",
            "evidence_file": str(NONLINEAR_TABLE),
            "limitation": "interface check; residual changes invalidate any prepared state",
        },
        {
            "loading_model": "per-shot repetition of residual loading",
            "scale": "4x4 anchor resource case",
            "implemented_status": "counted in resource model",
            "cost_model": "T_prep multiplies shots x (1/p_succ) in Eq. (cost_parametric)",
            "repeated_per_shot": "yes (1.549e+05 shots at 1e-2 target)",
            "repeated_per_postselection_attempt": "yes (1/p_succ = 1.0097)",
            "repeated_per_nonlinear_iteration": "yes if embedded",
            "evidence_file": str(RESOURCE),
            "limitation": "state preparation is never amortized across shots",
        },
        {
            "loading_model": "amplitude-estimation variant",
            "scale": "discussed alternative",
            "implemented_status": "modeled only; not used in reported counts",
            "cost_model": "O(1/(eps sqrt(p_succ))) coherent repetitions of prep+QSVT+readout",
            "repeated_per_shot": "yes (coherent copies inside each AE call)",
            "repeated_per_postselection_attempt": "yes",
            "repeated_per_nonlinear_iteration": "yes if embedded",
            "evidence_file": str(RESOURCE),
            "limitation": "requires controlled prep/QSVT/readout and their inverses",
        },
    ]


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
        r"% Source: outputs/residual_loading_ledger/residual_loading_ledger.csv",
        r"% Regenerate: .venv/bin/python scripts/build_residual_loading_ledger.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Residual-state loading ledger. " + CAPTION_CONCLUSION + r"}",
        r"\label{tab:residual_loading_ledger}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.20\textwidth}p{0.17\textwidth}p{0.21\textwidth}ccc"
        r"p{0.17\textwidth}}",
        r"\hline",
        r"Loading model & Implemented status & Cost model & Per shot & Per attempt & "
        r"Per NL iter. & Limitation \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():

        def _short(text: str) -> str:
            value = str(text)
            lowered = value.lower()
            if lowered.startswith("yes"):
                return "yes"
            if lowered.startswith("n/a"):
                return "n/a"
            return _tex_escape(value)

        cells = [
            _tex_escape(str(row["loading_model"])),
            _tex_escape(str(row["implemented_status"])),
            _tex_escape(str(row["cost_model"])),
            _short(row["repeated_per_shot"]),
            _short(row["repeated_per_postselection_attempt"]),
            _short(row["repeated_per_nonlinear_iteration"]),
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
    frame.to_csv(OUTPUT_DIR / "residual_loading_ledger.csv", index=False)
    _write_tex(frame, OUTPUT_DIR / "residual_loading_ledger.tex")
    md = [
        "# Residual-State Loading Ledger (Goal D)",
        "",
        f"**Conclusion:** {CAPTION_CONCLUSION}",
        "",
        _to_markdown(frame),
        "",
    ]
    (OUTPUT_DIR / "residual_loading_ledger.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "artifact_name": "residual_loading_ledger",
        "sources": {
            "state_preparation_blocks": str(STATE_PREP_BLOCKS),
            "state_preparation_models": str(STATE_PREP_MODELS),
            "fixed_case_resource_ledger": str(RESOURCE),
            "nonlinear_table": str(NONLINEAR_TABLE),
        },
        "regeneration_command": ".venv/bin/python scripts/build_residual_loading_ledger.py",
        "caption_conclusion": CAPTION_CONCLUSION,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Residual loading ledger written to {OUTPUT_DIR} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
