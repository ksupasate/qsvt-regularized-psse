"""Goal E: output-count scaling ledger for selected-output readout.

Instantiates T_selected = O[(q / (eps^2 p_succ)) (T_prep + d T_BE + T_read)]
for growing functional counts q. Direct sampling repeats the complete
preparation/QSVT/readout sequence per functional, so cost grows linearly in q;
the ledger states explicitly that selected-output readout does not solve
full-state recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/selected_output_scaling")
RESOURCE = Path(
    "outputs/tqe_revision_experiments/end_to_end_resource_case/fixed_case_resource_ledger.csv"
)
BASELINE = Path("outputs/classical_selected_observable_baseline/baseline_summary.csv")

COLUMNS = [
    "q",
    "example_use_case",
    "direct_sampling_multiplier",
    "interpretation",
    "within_scope",
    "classical_comparator",
]


def _fixed_case_numbers() -> dict[str, float]:
    if not RESOURCE.is_file():
        return {}
    frame = pd.read_csv(RESOURCE)
    values: dict[str, float] = {}
    for field in ["shots_for_target_error", "total_signal_unitary_calls_without_AA"]:
        rows = frame[frame["field"] == field]
        if len(rows):
            values[field] = float(rows.iloc[0]["value"])
    return values


def _classical_comparators() -> dict[str, str]:
    if not BASELINE.is_file():
        return {}
    frame = pd.read_csv(BASELINE)
    adjoint_4x4 = frame[
        (frame["case"] == "selected_block_4x4") & (frame["method"] == "adjoint_functional")
    ]
    ieee300 = frame[(frame["case"] == "ieee300") & (frame["method"] == "adjoint_functional")]
    out: dict[str, str] = {}
    if len(adjoint_4x4):
        row = adjoint_4x4.iloc[0]
        out["block"] = (
            f"shared factorization + {row['query_median_seconds']:.1e} s per extra functional "
            f"(4x4 block, 30 repeats)"
        )
    if len(ieee300):
        row = ieee300.iloc[0]
        out["full"] = (
            f"IEEE 300 sparse adjoint: {row['runtime_seconds']:.2e} s total, "
            f"{row['query_median_seconds']:.1e} s per extra functional"
        )
    return out


def build_rows() -> list[dict[str, str]]:
    fixed = _fixed_case_numbers()
    classical = _classical_comparators()
    shots = fixed.get("shots_for_target_error")
    signal_calls = fixed.get("total_signal_unitary_calls_without_AA")
    base = (
        f"1x = {shots:.3e} isolated-overlap shots, {signal_calls:.3e} modeled "
        "signal-unitary calls at eps = 1e-2 (4x4 selected-submatrix anchor)"
        if shots is not None and signal_calls is not None
        else "1x (fixed-case numbers not available)"
    )
    block_comparator = classical.get("block", "not available")
    full_comparator = classical.get("full", "not available")

    return [
        {
            "q": "1",
            "example_use_case": "one bus voltage/angle correction",
            "direct_sampling_multiplier": base,
            "interpretation": "modeled composition; the finite-shot experiment directly "
            "prepares the computed output state and is not integrated with prep+QSVT",
            "within_scope": "modeled from 4x4 selected-submatrix factors",
            "classical_comparator": block_comparator,
        },
        {
            "q": "3",
            "example_use_case": "bus correction + branch-angle difference + area aggregate",
            "direct_sampling_multiplier": "3x the q = 1 sequence count",
            "interpretation": "each signed functional needs its own Hadamard-test campaign; "
            "no state reuse across functionals in direct sampling",
            "within_scope": "isolated overlap experiments only",
            "classical_comparator": "same shared factorization; ~3 extra adjoint solves",
        },
        {
            "q": "10",
            "example_use_case": "small predetermined monitoring set (e.g., corridor screen)",
            "direct_sampling_multiplier": "10x the q = 1 sequence count",
            "interpretation": "linear growth still holds; classical shadows/AE could amortize "
            "only under favorable norms and coherent access (modeled, not compiled)",
            "within_scope": "modeled only",
            "classical_comparator": "one factorization + 10 adjoint solves (microseconds each)",
        },
        {
            "q": "n (state dimension)",
            "example_use_case": "every state coordinate separately (599 for IEEE 300)",
            "direct_sampling_multiplier": "n x the q = 1 sequence count",
            "interpretation": "coordinate-wise recovery of the full update by repeated "
            "selected-output queries; eliminates the selected-output premise",
            "within_scope": "no (outside the selected-output formulation)",
            "classical_comparator": full_comparator,
        },
        {
            "q": "full vector / tomography",
            "example_use_case": "complete signed update vector as classical data",
            "direct_sampling_multiplier": "not modeled here",
            "interpretation": "requires classical reconstruction, coordinate-wise queries, "
            "or tomography; selected-output readout does not solve full-state recovery",
            "within_scope": "no (explicitly excluded)",
            "classical_comparator": "direct classical solve returns the full vector already",
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
        r"% Source: outputs/selected_output_scaling/selected_output_scaling.csv",
        r"% Regenerate: .venv/bin/python scripts/build_selected_output_scaling_table.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Modeled output-count scaling for selected-submatrix functionals, "
        r"following Eq.~\eqref{eq:cost_parametric}. The isolated finite-shot experiment "
        r"does not execute the complete preparation/QSVT/readout composition. Under the "
        r"model, cost grows linearly with $q$. Selected-output readout does not solve full-state "
        r"recovery; full-vector rows are out of scope.}",
        r"\label{tab:selected_output_scaling}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{p{0.06\textwidth}p{0.20\textwidth}p{0.20\textwidth}"
        r"p{0.26\textwidth}p{0.16\textwidth}}",
        r"\hline",
        r"$q$ & Example use case & Direct-sampling multiplier & Interpretation & "
        r"Classical comparator \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        interpretation = str(row["interpretation"])
        scope = str(row["within_scope"])
        cells = [
            _tex_escape(str(row["q"])),
            _tex_escape(str(row["example_use_case"])),
            _tex_escape(str(row["direct_sampling_multiplier"])),
            _tex_escape(f"{interpretation} (scope: {scope})"),
            _tex_escape(str(row["classical_comparator"])),
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
    frame.to_csv(OUTPUT_DIR / "selected_output_scaling.csv", index=False)
    _write_tex(frame, OUTPUT_DIR / "selected_output_scaling.tex")
    md = [
        "# Selected-Output Scaling Ledger (Goal E)",
        "",
        "Model: `T_selected = O[(q / (eps^2 p_succ)) (T_prep + d T_BE + T_read)]`.",
        "Direct sampling repeats the complete sequence per functional; selected-output",
        "readout does not solve full-state recovery.",
        "",
        _to_markdown(frame),
        "",
    ]
    (OUTPUT_DIR / "selected_output_scaling.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "artifact_name": "selected_output_scaling",
        "sources": {
            "fixed_case_resource_ledger": str(RESOURCE),
            "classical_baseline": str(BASELINE),
        },
        "regeneration_command": (".venv/bin/python scripts/build_selected_output_scaling_table.py"),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected-output scaling ledger written to {OUTPUT_DIR} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
