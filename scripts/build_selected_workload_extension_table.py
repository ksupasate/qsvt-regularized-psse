"""Emit the manuscript table for the Goal A selected-workload extension attempts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE = Path("outputs/qsvt_selected_workload_extension/selected_workload_results.csv")
TARGET = Path("manuscript/tables/selected_workload_extension.tex")

STATUS_LABEL = {
    "feasible": "feasible",
    "degree_limited": "degree-limited",
    "tolerance_missing": "tolerance-missing",
    "phase_failed": "phase-failed",
    "dimension_infeasible": "dimension-infeasible",
}


def _sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"


def main() -> None:
    frame = pd.read_csv(SOURCE).drop_duplicates("workload_id")
    lines = [
        "% Source: outputs/qsvt_selected_workload_extension/selected_workload_results.csv",
        "% Regenerate: .venv/bin/python scripts/run_selected_workload_extension.py"
        " && .venv/bin/python scripts/build_selected_workload_extension_table.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Selected-workload extension attempts on the deterministic IEEE-14-derived"
        r" blocks. Every row keeps the matched Ridge/Tikhonov reference at the same $\alpha$."
        r" The benchmark rule $\alpha=4\sigma_{\min}^2$ on the $8\times8$ block gives"
        r" $\lambda=2.3\times10^{-4}$ and fails inside the tested synthesis range (boundary"
        r" evidence); matching the anchor's $\lambda$ instead makes the same $8\times8$ block"
        r" a second phase-synthesized correctness anchor. Within these tested $8\times8$"
        r" rows and degree settings, the boundary is driven by normalized regularization"
        r" rather than block dimension. Failed rows are retained as"
        r" boundary evidence.}",
        r"\label{tab:selected_workload_extension}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"Attempt & Block & $\kappa$ & $\lambda=\alpha/\beta^2$ & Degree & Phases & "
        r"$p_{\rm succ}$ & Update rel.\ error & Status \\",
        r"\hline",
    ]
    names = {
        "anchor_4x4_benchmark_alpha_d31": r"$4\times4$ anchor, $\alpha=4\sigma_{\min}^2$",
        "8x8_benchmark_alpha_d31": r"$8\times8$, $\alpha=4\sigma_{\min}^2$",
        "8x8_benchmark_alpha_d45": r"$8\times8$, $\alpha=4\sigma_{\min}^2$ (ceiling)",
        "8x8_codesigned_lambda_matched_d31": r"$8\times8$, $\lambda$-matched co-design",
    }
    for _, row in frame.iterrows():
        p_succ = (
            f"{row['postselection_probability']:.4f}"
            if pd.notna(row["postselection_probability"])
            else "--"
        )
        update_error = (
            _sci(float(row["update_relative_error_vs_ridge"]))
            if pd.notna(row["update_relative_error_vs_ridge"])
            else "--"
        )
        lines.append(
            f"{names.get(row['workload_id'], row['workload_id'])} & "
            f"{row['block_size']} & {row['kappa_block']:.2f} & "
            f"{_sci(float(row['lambda_alpha_over_beta2']))} & "
            f"{int(row['degree_attempted'])} & "
            f"{int(row['phase_count']) if row['phase_count'] else '--'} & "
            f"{p_succ} & {update_error} & "
            f"{STATUS_LABEL.get(str(row['status']), str(row['status']))} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
