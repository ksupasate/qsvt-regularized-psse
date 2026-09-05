"""Generate explicit PSSE measurement, stress, and numerical-rank assumptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

CASES = ("ieee14", "ieee30", "ieee57", "ieee118", "ieee300")
RCOND = 1.0e-10


def rank_rows(seed: int = 123, rcond: float = RCOND) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name in CASES:
        system, _ = build_engineering_system(
            {
                "case_name": case_name,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
        singular = np.linalg.svd(system.H_tilde, compute_uv=False)
        threshold = float(rcond) * float(singular[0])
        retained = singular[singular > threshold]
        numerical_rank = int(retained.size)
        effective_condition = float(retained[0] / retained[-1]) if retained.size else float("inf")
        types, counts = np.unique(system.metadata["measurement_types"], return_counts=True)
        rows.append(
            {
                "case_name": case_name,
                "matrix_shape": f"{system.n_measurements}x{system.n_states}",
                "row_type_counts": "; ".join(
                    f"{name}:{int(count)}" for name, count in zip(types, counts, strict=True)
                ),
                "rank_relative_threshold": float(rcond),
                "absolute_singular_value_threshold": threshold,
                "pseudoinverse_rcond": float(rcond),
                "numerical_rank": numerical_rank,
                "state_dimension": int(system.n_states),
                "sigma_max": float(singular[0]),
                "sigma_min_retained": float(retained[-1]) if retained.size else 0.0,
                "effective_condition_number": effective_condition,
                "raw_numpy_condition_number": float(np.linalg.cond(system.H_tilde)),
                "seed": int(seed),
                "matrix_scope": (
                    "full generated weighted Jacobian used as the source for deterministic "
                    "selected-submatrix extraction"
                ),
            }
        )
    return rows


def parameter_rows() -> list[dict[str, str]]:
    return [
        {
            "item": "measurement rows",
            "value": "V, P injection, Q injection, directed P flow, directed Q flow",
            "scope": "QSVT source/nonlinear; classical PYPOWER linearized runs omit Q rows",
        },
        {
            "item": "row standard deviations",
            "value": "V: 0.01 pu; P/Q injection: 0.03 pu; P/Q flow: 0.02 pu",
            "scope": "diagonal R; power quantities use the case base-MVA per-unit basis",
        },
        {
            "item": "weak-area multiplier",
            "value": "10x main classical/nonlinear configs; 15x or 30x stress; 1x QSVT source",
            "scope": "multiplies sigma_i for rows incident on configured weak-area buses",
        },
        {
            "item": "missing rows",
            "value": "round(m rho) rows sampled uniformly without replacement",
            "scope": "rho in {0,0.1,0.2} main sweeps; retained rows must not be fewer than states",
        },
        {
            "item": "bad data",
            "value": "random sign +/-; magnitude m_b sigma_i in raw z, or +/-m_b in weighted r",
            "scope": "ratios {0,0.05,0.1}; m_b=5 main; weak-area eligible rows",
        },
        {
            "item": "seeds",
            "value": "QSVT extraction 123; main sweeps 101,202,...,909,1001",
            "scope": "all random selection, perturbation, and shot seeds are recorded",
        },
        {
            "item": "rank / pinv rule",
            "value": "retain sigma_i > 1e-10 sigma_max; pseudoinverse rcond=1e-10",
            "scope": "rank table below uses the same explicit relative cutoff",
        },
        {
            "item": "perturbation location",
            "value": "nonlinear: z=h(x_true)+e+b; single-step: perturb weighted residual",
            "scope": "QSVT is not executed in the nonlinear loop",
        },
    ]


def run_psse_assumption_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(config or {})
    output_dir = ensure_directory(Path(options.get("output_dir", "outputs/psse_assumption_audit")))
    table_path = Path(options.get("table_path", "manuscript/tables/psse_experiment_parameters.tex"))
    seed = int(options.get("seed", 123))
    rcond = float(options.get("rcond", RCOND))
    parameters = pd.DataFrame(parameter_rows())
    ranks = pd.DataFrame(rank_rows(seed=seed, rcond=rcond))
    parameters.to_csv(output_dir / "psse_experiment_parameters.csv", index=False)
    ranks.to_csv(output_dir / "psse_rank_diagnostics.csv", index=False)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_tex(parameters, ranks), encoding="utf-8")
    manifest = {
        "artifact_name": "psse_assumption_audit",
        "seed_provenance": {"status": "recorded", "seeds": {"qsvt_source_seed": seed}},
        "rank_relative_threshold": rcond,
        "pseudoinverse_rcond": rcond,
        "source_configs": [
            "configs/real_ieee14.yaml",
            "configs/nonlinear_ac_ieee14_seed10.yaml",
            "configs/qsvt_phase2_boundary.yaml",
        ],
        "regeneration_command": ".venv/bin/python scripts/run_psse_assumption_audit.py",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"parameters": parameters, "ranks": ranks, "table": table_path}


def _escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("+/-", r"$\pm$")


def _tex(parameters: pd.DataFrame, ranks: pd.DataFrame) -> str:
    lines = [
        "% Source: outputs/psse_assumption_audit/"
        "{psse_experiment_parameters.csv,psse_rank_diagnostics.csv}",
        "% Regenerate: .venv/bin/python scripts/run_psse_assumption_audit.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled PSSE benchmark parameters. These settings generate measurement "
        r"rows from IEEE/PYPOWER network models; they are not field-calibrated PMU/SCADA "
        r"statistics. Power quantities are per unit on each case base MVA.}",
        r"\label{tab:psse_parameters}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.16\textwidth}p{0.39\textwidth}p{0.39\textwidth}}",
        r"\hline",
        r"Item & Value/rule & Scope \\",
        r"\hline",
    ]
    for _, row in parameters.iterrows():
        lines.append(
            f"{_escape(str(row['item']))} & {_escape(str(row['value']))} & "
            f"{_escape(str(row['scope']))} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Numerical rank of the full generated weighted-Jacobian sources at seed "
            r"123. The retained set is $\sigma_i>10^{-10}\sigma_{\max}$, the same relative "
            r"cutoff used by the reported pseudoinverse. $\kappa_{\rm eff}$ uses the smallest "
            r"retained singular value; it is not the unthresholded floating-point condition "
            r"number.}",
            r"\label{tab:psse_rank}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Case & Shape & Rank / states & cutoff & $\kappa_{\rm eff}$ \\",
            r"\hline",
        ]
    )
    for _, row in ranks.iterrows():
        lines.append(
            f"{str(row['case_name']).upper()} & {row['matrix_shape']} & "
            f"{int(row['numerical_rank'])}/{int(row['state_dimension'])} & "
            f"{float(row['absolute_singular_value_threshold']):.2e} & "
            f"{float(row['effective_condition_number']):.2e} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)
