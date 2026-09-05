from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory

ORACLE_LEDGER_CLAIM = (
    "Full IEEE-scale QSVT is treated as an oracle-model pathway. The current "
    "repository does not demonstrate full IEEE-scale hardware execution or full "
    "sparse-oracle gate synthesis."
)

REQUIRED_ASSUMPTIONS = [
    "sparse row/column access oracle",
    "matrix value-loading oracle",
    "value-rotation / fixed-point encoding",
    "residual state preparation",
    "QSVT block-encoding query model",
    "success probability estimation",
    "amplitude amplification",
    "norm recovery",
    "observable readout",
    "hardware-native oracle synthesis",
]


def build_sparse_oracle_assumption_ledger(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "alpha": 1.0e-4,
        "degree": 51,
        "output_dir": "outputs/sparse_oracle_assumption_ledger",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    ledger_rows = assumption_ledger_rows()
    resource_rows = [
        resource_table_row(case, resolved=resolved)
        for case in [str(value) for value in resolved["cases"]]
    ]
    artifacts = _write_outputs(output_dir, resolved, ledger_rows, resource_rows)
    return {
        "output_dir": output_dir,
        "ledger_rows": ledger_rows,
        "resource_rows": resource_rows,
        "artifacts": artifacts,
    }


def assumption_ledger_rows() -> list[dict[str, Any]]:
    rows = [
        _assumption(
            "sparse row/column access oracle",
            "Sparse block encoding of weighted Jacobian",
            "toy_gate_evidence_only",
            "Toy sparse-oracle gate demonstration for small matrix lookup",
            "IEEE matrix sparsity statistics and qubit counts",
            "Full row/column oracle synthesis for IEEE matrices",
            "high",
            "QSVT query model cannot be instantiated at full scale",
            "Implement and validate sparse access oracles on selected IEEE blocks",
            "oracle-model full IEEE-scale pathway",
            "full IEEE-scale hardware execution",
        ),
        _assumption(
            "matrix value-loading oracle",
            "Encoding H_tilde entries",
            "toy_gate_evidence_only",
            "Toy value lookup/rotation evidence",
            "Dense and sparse resource proxies",
            "Precision-controlled value loading for weighted entries",
            "high",
            "Resource estimates understate data-loading cost",
            "Specify fixed-point precision and synthesize value-loading blocks",
            "sparse-oracle assumption ledger",
            "hardware-native oracle synthesis complete",
        ),
        _assumption(
            "value-rotation / fixed-point encoding",
            "Block-encoding amplitude construction",
            "partial",
            "Toy rotation gate evidence",
            "Rotation-count proxies",
            "Fault-tolerant fixed-point rotation synthesis",
            "moderate",
            "Block-encoding normalization and precision may be inaccurate",
            "Add precision sweep for entry rotations",
            "oracle-model cost proxy",
            "precision bottleneck solved",
        ),
        _assumption(
            "residual state preparation",
            "Preparing |r_tilde>",
            "dense_initialize_only",
            "Qiskit Initialize on small residual vectors",
            "State-prep model labels in resource tables",
            "Scalable residual loading circuit",
            "high",
            "End-to-end query cost omits input-state loading",
            "Add sparse/loading model for residual vectors",
            "readout and norm-recovery bottleneck analysis",
            "efficient residual preparation demonstrated",
        ),
        _assumption(
            "QSVT block-encoding query model",
            "Applying polynomial singular-value filter",
            "implemented_small_dense",
            "Scalar, diagonal, tiny, and selected 4x4 gate-level tests",
            "Query-count estimates degree dependent",
            "Sparse block-encoding integration with QSVT phases",
            "moderate",
            "Full-scale estimates may not match circuit synthesis",
            "Integrate sparse oracle with QSVT sequence on small sparse block",
            "selected IEEE-derived gate-level QSVT update",
            "full IEEE-scale gate-level QSVT solver",
        ),
        _assumption(
            "success probability estimation",
            "Postselection and norm diagnostics",
            "simulator_metadata",
            "Statevector success probabilities for dense demos",
            "Success/amplification cost table",
            "Amplitude-estimation circuit or statistical estimator",
            "moderate",
            "Runtime can be dominated by postselection",
            "Implement explicit success-probability estimation primitive",
            "success probability proxy",
            "readout bottleneck solved",
        ),
        _assumption(
            "amplitude amplification",
            "Boosting postselection success",
            "not_implemented",
            "No hardware circuit; only cost proxy",
            "1/sqrt(p_success) query multiplier",
            "Controlled reflections and full amplified circuit",
            "high",
            "Postselection can be too costly for low success probabilities",
            "Prototype amplitude-amplification circuit on tiny solver",
            "amplitude-amplification cost proxy",
            "amplitude amplification implemented",
        ),
        _assumption(
            "norm recovery",
            "Recovering update magnitude",
            "simulator_metadata",
            "Norm/residual-gap audit",
            "Norm-success diagnostic rows",
            "Quantum norm-estimation or amplitude-estimation routine",
            "high",
            "Direction may be useful but residual accuracy remains unexplained",
            "Implement norm-estimation proxy with shots/query accounting",
            "norm and residual-gap audit",
            "norm recovery solved",
        ),
        _assumption(
            "observable readout",
            "Partial state-estimation quantities",
            "partial",
            "Component/subset and power-observable mapping",
            "Shot-scaling proxies",
            "Problem-specific measurement circuits for signed observables",
            "moderate",
            "Full update tomography remains expensive",
            "Add Hadamard-test circuits for selected signed observables",
            "partial observable readout path",
            "full readout problem solved",
        ),
        _assumption(
            "hardware-native oracle synthesis",
            "Compilation to target gates",
            "not_implemented",
            "Dense tiny transpilation and toy sparse gates",
            "Resource-model caveats",
            "Complete target-native sparse oracle decomposition",
            "high",
            "Gate counts may be unrealistic for hardware",
            "Synthesize oracle blocks under an explicit basis and precision",
            "oracle-model full IEEE-scale pathway",
            "hardware execution",
        ),
    ]
    return rows


def resource_table_row(case: str, *, resolved: dict[str, Any]) -> dict[str, Any]:
    try:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": str(resolved["case_source"]),
                "matrix_source": f"{case}_ac_weighted_jacobian",
                "seed": 123,
            }
        )
        matrix = np.asarray(system.H_tilde, dtype=np.float64)
        nnz = int(np.count_nonzero(np.abs(matrix) > 1.0e-12))
        row_sparsity = np.count_nonzero(np.abs(matrix) > 1.0e-12, axis=1)
        col_sparsity = np.count_nonzero(np.abs(matrix) > 1.0e-12, axis=0)
        rows, columns = matrix.shape
        success_proxy = _existing_success_proxy(case)
        return {
            "case": case,
            "matrix_shape": f"{rows}x{columns}",
            "nnz": nnz,
            "density": float(nnz / matrix.size),
            "max_row_sparsity": int(np.max(row_sparsity)) if row_sparsity.size else 0,
            "max_col_sparsity": int(np.max(col_sparsity)) if col_sparsity.size else 0,
            "row_qubits": _qubits(rows),
            "col_qubits": _qubits(columns),
            "padded_qubits": _qubits(rows) + _qubits(columns),
            "ancilla_qubits": _oracle_ancilla_estimate(rows, columns),
            "degree": int(resolved["degree"]),
            "phase_count": int(resolved["degree"]) + 1,
            "query_count": 2 * int(resolved["degree"]) + 1,
            "state_prep_model": "assumed sparse/loadable residual state",
            "matrix_oracle_model": "assumed sparse row/column/value oracle",
            "readout_model": "partial observable readout; full tomography not assumed",
            "success_probability_proxy_if_available": success_proxy,
            "resource_status": f"oracle_model_estimate_from_{matrix_source}",
        }
    except Exception as exc:  # pragma: no cover - case availability dependent
        return {
            "case": case,
            "matrix_shape": "",
            "nnz": np.nan,
            "density": np.nan,
            "max_row_sparsity": np.nan,
            "max_col_sparsity": np.nan,
            "row_qubits": np.nan,
            "col_qubits": np.nan,
            "padded_qubits": np.nan,
            "ancilla_qubits": np.nan,
            "degree": int(resolved["degree"]),
            "phase_count": int(resolved["degree"]) + 1,
            "query_count": 2 * int(resolved["degree"]) + 1,
            "state_prep_model": "unavailable",
            "matrix_oracle_model": "unavailable",
            "readout_model": "unavailable",
            "success_probability_proxy_if_available": np.nan,
            "resource_status": f"failed: {type(exc).__name__}: {exc}",
        }


def _assumption(
    assumption_name: str,
    required_for: str,
    current_status: str,
    implemented_evidence: str,
    estimated_evidence: str,
    missing_piece: str,
    risk_level: str,
    impact_if_false: str,
    next_action: str,
    allowed_claim: str,
    disallowed_claim: str,
) -> dict[str, str]:
    return {
        "assumption_name": assumption_name,
        "required_for": required_for,
        "current_status": current_status,
        "implemented_evidence": implemented_evidence,
        "estimated_evidence": estimated_evidence,
        "missing_piece": missing_piece,
        "risk_level": risk_level,
        "impact_if_false": impact_if_false,
        "next_action": next_action,
        "allowed_claim": allowed_claim,
        "disallowed_claim": disallowed_claim,
    }


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    ledger_path = output_dir / "oracle_assumption_ledger.csv"
    resource_path = output_dir / "ieee_resource_table.csv"
    cost_path = output_dir / "oracle_cost_model.md"
    risk_path = output_dir / "assumption_risk_ranking.md"
    pd.DataFrame(ledger_rows).to_csv(ledger_path, index=False)
    pd.DataFrame(resource_rows).to_csv(resource_path, index=False)
    cost_path.write_text(_cost_model_markdown(resource_rows), encoding="utf-8")
    risk_path.write_text(_risk_markdown(ledger_rows), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "oracle_assumption_ledger": str(ledger_path),
            "ieee_resource_table": str(resource_path),
            "oracle_cost_model": str(cost_path),
            "assumption_risk_ranking": str(risk_path),
        },
        input_config=resolved,
        claim_boundary=ORACLE_LEDGER_CLAIM,
    )
    return {
        "manifest": manifest,
        "oracle_assumption_ledger": ledger_path,
        "ieee_resource_table": resource_path,
        "oracle_cost_model": cost_path,
        "assumption_risk_ranking": risk_path,
    }


def _cost_model_markdown(resource_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Sparse-Oracle Cost Model",
            "",
            ORACLE_LEDGER_CLAIM,
            "",
            "- Query count is reported as `2*degree + 1`.",
            "- Row/column/value oracles, residual state preparation, success estimation, "
            "amplitude amplification, and norm recovery are not hardware-native "
            "implementations in this repository.",
            f"- Cases summarized: {', '.join(str(row['case']) for row in resource_rows)}",
            "",
        ]
    )


def _risk_markdown(ledger_rows: list[dict[str, Any]]) -> str:
    high = [row for row in ledger_rows if row["risk_level"] == "high"]
    lines = ["# Sparse-Oracle Assumption Risk Ranking", "", ORACLE_LEDGER_CLAIM, ""]
    for row in high:
        lines.append(f"- High risk: {row['assumption_name']} -> {row['missing_piece']}")
    return "\n".join(lines) + "\n"


def _existing_success_proxy(case: str) -> float:
    path = Path(f"outputs/gate_level_qsvt_{case}_solver/qsvt_solver_circuit_summary.json")
    if not path.is_file() and case == "ieee14":
        path = Path("outputs/gate_level_qsvt_ieee14_solver/qsvt_solver_circuit_summary.json")
    if not path.is_file():
        return float("nan")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("success_probability", np.nan))
    except Exception:
        return float("nan")


def _qubits(value: int) -> int:
    return int(np.ceil(np.log2(max(int(value), 2))))


def _oracle_ancilla_estimate(rows: int, columns: int) -> int:
    return 2 + _qubits(max(rows, columns))
