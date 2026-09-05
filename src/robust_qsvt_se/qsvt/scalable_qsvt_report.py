from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.hardware_resource_estimator import (
    build_oracle_model_resource_report,
    build_qubit_convention_audit,
)
from robust_qsvt_se.qsvt.matrix_free_qsvt_action import (
    build_matrix_free_action_outputs,
    run_matrix_free_ieee_experiments,
)
from robust_qsvt_se.qsvt.norm_success import (
    estimate_success_probability_by_sampling,
    estimate_success_probability_iterative_proxy,
)
from robust_qsvt_se.qsvt.sparse_access_oracle import build_sparse_oracle_outputs
from robust_qsvt_se.qsvt.state_preparation_model import build_state_preparation_outputs
from robust_qsvt_se.qsvt.toy_sparse_oracle_circuit import build_toy_sparse_oracle_circuit_demo
from robust_qsvt_se.utils.io import ensure_directory

SCALABLE_QSVT_CLAIM_BOUNDARY = (
    "This is a scalable QSVT implementation pathway with sparse-access oracle "
    "abstractions, matrix-free polynomial-action simulations, small explicit "
    "circuit demos, and IEEE-scale resource estimates. It does not demonstrate "
    "quantum speedup, full IEEE-scale quantum hardware execution, or QSVT "
    "numerical superiority over Ridge/Tikhonov."
)


def build_success_amplitude_proxy_outputs(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_success_amplitude_proxy",
        "success_probabilities": [0.01, 0.05, 0.1],
        "shots": [100, 1000, 10000],
        "max_queries": [100, 1000, 10000],
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sampling_rows: list[dict[str, Any]] = []
    amplitude_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for p_success in [float(value) for value in resolved["success_probabilities"]]:
        for shots in [int(value) for value in resolved["shots"]]:
            result = estimate_success_probability_by_sampling(
                p_success,
                shots,
                int(resolved["seed"]),
            )
            row = result.to_row()
            sampling_rows.append(row)
            norm_rows.append(_norm_rescaling_row(row, "sampling", shots))
        for queries in [int(value) for value in resolved["max_queries"]]:
            result = estimate_success_probability_iterative_proxy(
                p_success,
                queries,
                int(resolved["seed"]),
            )
            row = result.to_row()
            amplitude_rows.append(row)
            norm_rows.append(_norm_rescaling_row(row, "amplitude_proxy", queries))
    sampling_csv = output_dir / "success_sampling_summary.csv"
    amplitude_csv = output_dir / "amplitude_proxy_summary.csv"
    norm_csv = output_dir / "norm_rescaling_error_summary.csv"
    limitations_md = output_dir / "success_amplitude_proxy_limitations.md"
    pd.DataFrame(sampling_rows).to_csv(sampling_csv, index=False)
    pd.DataFrame(amplitude_rows).to_csv(amplitude_csv, index=False)
    pd.DataFrame(norm_rows).to_csv(norm_csv, index=False)
    limitations_md.write_text(_success_limitations_markdown(), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "success_sampling_summary": str(sampling_csv),
            "amplitude_proxy_summary": str(amplitude_csv),
            "norm_rescaling_error_summary": str(norm_csv),
            "success_amplitude_proxy_limitations": str(limitations_md),
        },
        input_config=resolved,
        claim_boundary=SCALABLE_QSVT_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "sampling": pd.DataFrame(sampling_rows),
        "amplitude": pd.DataFrame(amplitude_rows),
        "norm": pd.DataFrame(norm_rows),
        "artifacts": {
            "manifest": manifest,
            "success_sampling_summary": sampling_csv,
            "amplitude_proxy_summary": amplitude_csv,
            "norm_rescaling_error_summary": norm_csv,
            "success_amplitude_proxy_limitations": limitations_md,
        },
    }


def build_scalable_qsvt_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/scalable_qsvt_ieee_report",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "matrix_free_input_dirs": [
            "outputs/qsvt_matrix_free_ieee_experiments",
            "outputs/qsvt_matrix_free_ieee_resource_only",
        ],
        "generate_missing_matrix_free": True,
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    support_config = {"cases": list(resolved["cases"])}
    qubit = build_qubit_convention_audit(support_config)
    sparse_oracle = build_sparse_oracle_outputs(support_config)
    build_matrix_free_action_outputs({"case": next(iter(resolved["cases"]))})
    state_prep = build_state_preparation_outputs(support_config)
    success = build_success_amplitude_proxy_outputs()
    resources = build_oracle_model_resource_report(support_config)
    toy = build_toy_sparse_oracle_circuit_demo()
    matrix_free = _load_or_generate_matrix_free(resolved)

    files = {
        "scalable_qsvt_capability_matrix": output_dir / "scalable_qsvt_capability_matrix.csv",
        "sparse_oracle_summary": output_dir / "sparse_oracle_summary.csv",
        "matrix_free_ieee_summary": output_dir / "matrix_free_ieee_summary.csv",
        "state_preparation_summary": output_dir / "state_preparation_summary.csv",
        "success_amplitude_proxy_summary": output_dir / "success_amplitude_proxy_summary.csv",
        "oracle_model_resource_summary": output_dir / "oracle_model_resource_summary.csv",
        "toy_sparse_oracle_summary": output_dir / "toy_sparse_oracle_summary.csv",
        "claim_support_matrix": output_dir / "claim_support_matrix.csv",
        "scalable_qsvt_ieee_report": output_dir / "scalable_qsvt_ieee_report.md",
    }
    capability = _capability_matrix()
    claim_support = _claim_support_matrix()
    success_summary = _success_summary(success)

    capability.to_csv(files["scalable_qsvt_capability_matrix"], index=False)
    sparse_oracle["summary"].to_csv(files["sparse_oracle_summary"], index=False)
    matrix_free.to_csv(files["matrix_free_ieee_summary"], index=False)
    state_prep["summary"].to_csv(files["state_preparation_summary"], index=False)
    success_summary.to_csv(files["success_amplitude_proxy_summary"], index=False)
    resources["summary"].to_csv(files["oracle_model_resource_summary"], index=False)
    toy["summary"].to_csv(files["toy_sparse_oracle_summary"], index=False)
    claim_support.to_csv(files["claim_support_matrix"], index=False)
    files["scalable_qsvt_ieee_report"].write_text(
        _report_markdown(
            capability=capability,
            matrix_free=matrix_free,
            resources=resources["summary"],
            qubit=qubit["summary"],
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={key: str(value) for key, value in files.items()},
        input_config=resolved,
        claim_boundary=SCALABLE_QSVT_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "capability": capability,
        "matrix_free": matrix_free,
        "resources": resources["summary"],
        "artifacts": {**files, "manifest": manifest},
    }


def _load_or_generate_matrix_free(config: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in list(config["matrix_free_input_dirs"]):
        path = Path(directory) / "matrix_free_ieee_summary.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    if not bool(config["generate_missing_matrix_free"]):
        return pd.DataFrame()
    run = run_matrix_free_ieee_experiments(
        {
            "output_dir": str(Path(config["output_dir"]) / "_matrix_free_generated"),
            "cases": [next(iter(config.get("cases", ["ieee14"])))],
            "alphas": [1.0e-4],
            "degrees": [35],
            "seed": int(config["seed"]),
        }
    )
    return run["summary"]


def _norm_rescaling_row(row: dict[str, Any], model: str, budget: int) -> dict[str, Any]:
    true_p = float(row["true_success_probability"])
    estimate = float(
        row.get("shot_estimated_success_probability", row.get("amplitude_proxy_estimate"))
    )
    true_scale = np.sqrt(max(true_p, 0.0))
    estimated_scale = np.sqrt(max(estimate, 0.0))
    return {
        "model": model,
        "budget": int(budget),
        "true_success_probability": true_p,
        "estimated_success_probability": estimate,
        "norm_scale_absolute_error": abs(estimated_scale - true_scale),
        "relative_norm_scale_error": abs(estimated_scale - true_scale) / max(true_scale, 1.0e-15),
    }


def _success_summary(success: dict[str, Any]) -> pd.DataFrame:
    sampling = success["sampling"].copy()
    sampling["proxy_type"] = "sampling"
    amplitude = success["amplitude"].copy()
    amplitude["proxy_type"] = "amplitude_estimation_proxy"
    sampling = sampling.rename(columns={"shot_estimated_success_probability": "estimate"})
    amplitude = amplitude.rename(columns={"amplitude_proxy_estimate": "estimate"})
    common = [
        "proxy_type",
        "true_success_probability",
        "estimate",
        "absolute_error",
        "limitation",
    ]
    return pd.concat([sampling[common], amplitude[common]], ignore_index=True)


def _capability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "capability": "explicit dense QSVT simulation",
                "status": "implemented for small selected subproblems",
                "evidence_type": "executed dense simulation",
                "claim_boundary": "not scalable IEEE-scale hardware execution",
            },
            {
                "capability": "sparse-access oracle abstraction",
                "status": "implemented as classical CSR/CSC access interface",
                "evidence_type": "oracle-model software abstraction",
                "claim_boundary": "not a physical quantum oracle circuit",
            },
            {
                "capability": "matrix-free QSVT polynomial action",
                "status": "implemented as Chebyshev sparse matvec/rmatvec proxy",
                "evidence_type": "matrix-free polynomial-action simulation",
                "claim_boundary": "not a full dense QSVT circuit simulation",
            },
            {
                "capability": "residual state preparation",
                "status": "implemented as resource models",
                "evidence_type": "state-preparation estimate",
                "claim_boundary": "qRAM/dense loading assumptions are not hardware",
            },
            {
                "capability": "success and norm recovery",
                "status": "implemented as sampling and amplitude-estimation proxies",
                "evidence_type": "simulator proxy",
                "claim_boundary": "not a complete quantum norm-estimation routine",
            },
            {
                "capability": "IEEE-scale resource analysis",
                "status": "implemented under sparse-access oracle assumptions",
                "evidence_type": "resource estimate",
                "claim_boundary": "no full IEEE-scale quantum execution",
            },
        ]
    )


def _claim_support_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "small explicit dense block-encoded QSVT can be simulated",
                "support": "small dense simulation outputs",
                "category": "implemented dense simulation",
                "allowed": True,
            },
            {
                "claim": "IEEE matrices have sparse-access oracle resource estimates",
                "support": "oracle_model_resource_summary.csv",
                "category": "oracle-model resource estimate",
                "allowed": True,
            },
            {
                "claim": "matrix-free polynomial action can be evaluated with sparse matvecs",
                "support": "matrix_free_ieee_summary.csv",
                "category": "matrix-free proxy",
                "allowed": True,
            },
            {
                "claim": "QSVT gives quantum speedup or numerical superiority",
                "support": "none",
                "category": "unsupported",
                "allowed": False,
            },
            {
                "claim": "full IEEE-scale hardware execution is implemented",
                "support": "none",
                "category": "unsupported",
                "allowed": False,
            },
        ]
    )


def _report_markdown(
    *,
    capability: pd.DataFrame,
    matrix_free: pd.DataFrame,
    resources: pd.DataFrame,
    qubit: pd.DataFrame,
) -> str:
    cases = ", ".join(sorted(resources["case"].astype(str).unique()))
    implemented = matrix_free[
        matrix_free.get("implemented_or_estimated", pd.Series(dtype=str))
        == "matrix_free_polynomial_action_proxy"
    ]
    best_error = implemented.get("relative_error_vs_ridge", pd.Series(dtype=float)).min()
    best_error_text = "not available" if pd.isna(best_error) else f"{best_error:.6g}"
    return "\n".join(
        [
            "# Scalable QSVT IEEE Report",
            "",
            "## Scope and Claim Boundary",
            SCALABLE_QSVT_CLAIM_BOUNDARY,
            "",
            "## What Was Implemented Exactly",
            "- Sparse-access oracle interfaces over IEEE-derived weighted Jacobians.",
            "- Matrix-free Chebyshev polynomial action using sparse matvec/rmatvec calls.",
            "- State-preparation resource models and success-probability proxies.",
            "- A tiny sparse-oracle circuit-structure demo.",
            "",
            "## What Remains a Proxy",
            "- Matrix-free polynomial action is a QSVT-target proxy, not a full QSVT circuit.",
            "- Sampling and amplitude-estimation outputs are simulator proxies.",
            "",
            "## What Remains a Resource Estimate",
            "- IEEE118/IEEE300 rows are resource estimates unless explicitly run otherwise.",
            "- Sparse oracle circuits, qRAM loading, and norm estimation are not synthesized.",
            "",
            "## Sparse-Access Matrix Oracle",
            "The oracle model stores CSR/CSC nonzero positions and values and provides row "
            "access, value access, matvec, and rmatvec operations.",
            "",
            "## Residual State-Preparation Model",
            "Exact dense loading is simulator-only; qRAM and sparse residual loading are "
            "assumed access models.",
            "",
            "## Matrix-Free QSVT Polynomial Action",
            f"Implemented matrix-free rows: {len(implemented)}.",
            f"Best reported relative error vs Ridge where available: {best_error_text}.",
            "",
            "## Success Probability and Norm Rescaling",
            "Success-probability estimates are reported separately from normalized state "
            "directions. Norm recovery is not claimed as a solved hardware readout.",
            "",
            "## Partial-Observable Readout",
            "Observable rows report selected normalized-state quantities and direction "
            "diagnostics; they do not reconstruct the full update vector as a readout claim.",
            "",
            "## Circuit-Level Toy Sparse-Oracle Demo",
            "The toy 4x4 demo exposes an oracle table and compares against a dense "
            "block-encoding diagnostic.",
            "",
            "## IEEE-Scale Resource Estimates",
            f"Cases included: {cases}.",
            "",
            "## Alpha-Degree-Resource Tradeoff",
            "Degree controls both polynomial approximation cost and the QSVT query-count "
            "proxy 2d+1.",
            "",
            "## Bottlenecks for True Hardware Implementation",
            "- Sparse oracle synthesis",
            "- Efficient residual state preparation",
            "- Success-probability and norm estimation",
            "- Fault-tolerant compilation and readout budgeting",
            "",
            "## Safe Manuscript Wording",
            "We implement a scalable QSVT implementation pathway with sparse-access oracle "
            "abstractions and matrix-free polynomial-action simulations for IEEE-derived "
            "weighted Jacobians, complemented by small explicit dense simulations and "
            "resource estimates. These results do not establish quantum speedup or full "
            "IEEE-scale hardware execution.",
            "",
            "## Qubit Convention Audit",
            f"Rows audited: {len(qubit)}. Rectangular row/column registers and padded "
            "square-dimension registers are reported separately.",
            "",
            "## Capability Matrix",
            _markdown_table(capability),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _success_limitations_markdown() -> str:
    return "\n".join(
        [
            "# Success and Amplitude Proxy Limitations",
            "",
            SCALABLE_QSVT_CLAIM_BOUNDARY,
            "",
            "- Bernoulli sampling estimates success probability from simulator metadata.",
            "- The amplitude-estimation row is a statistical proxy, not a quantum circuit.",
            "- Norm rescaling errors are diagnostics, not a solved readout pipeline.",
            "",
        ]
    )
