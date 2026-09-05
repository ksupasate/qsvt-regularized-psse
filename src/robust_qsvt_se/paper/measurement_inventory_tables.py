"""Phase 1: measurement model and benchmark inventory paper tables.

Converts the existing measurement-inventory artifacts into manuscript-ready
tables and figure-ready data. IEEE/PYPOWER cases are described as benchmark
network models and all measurement rows are described as generated from the
network model, never as real PMU/SCADA field records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

PYPOWER_BASE_MVA = 100.0

CASE_DIMENSION_COLUMNS = [
    "case",
    "base_mva",
    "n_bus",
    "n_branch",
    "n_gen",
    "state_dimension",
    "angle_states",
    "voltage_magnitude_states",
    "measurement_rows_total",
    "redundancy_ratio",
    "source",
    "notes",
]

MEASUREMENT_INVENTORY_COLUMNS = [
    "case",
    "workflow",
    "measurement_type",
    "row_count",
    "physical_or_diagnostic",
    "source_equation_group",
    "included_in_z",
    "included_in_hx",
    "included_in_jacobian",
    "notes",
]

MEASUREMENT_TYPE_COLUMNS = [
    "measurement_type",
    "symbol",
    "description",
    "nonlinear_ac_role",
    "linearized_role",
    "jacobian_rows",
    "typical_noise_assumption",
    "field_data_available",
    "notes",
]

WORKFLOW_TAXONOMY_COLUMNS = [
    "workflow",
    "uses_raw_measurements",
    "uses_weighted_residual",
    "uses_iterative_jacobian_rebuild",
    "perturbation_model",
    "primary_outputs",
    "claim_boundary",
    "notes",
]

FIGURE_DATA_COLUMNS = [
    "case",
    "state_dimension",
    "measurement_rows_total",
    "redundancy_ratio",
    "n_bus",
]

# Static measurement-type facts from the measurement model code (ac_linear / dc_linear).
MEASUREMENT_TYPES: tuple[dict[str, Any], ...] = (
    {
        "measurement_type": "voltage_magnitude",
        "symbol": r"$V_i$",
        "description": "Bus voltage magnitude",
        "nonlinear_ac_role": "z_i = V_i",
        "linearized_role": "identity row on the voltage state",
        "jacobian_rows": "1 per bus",
        "typical_noise_assumption": "sigma=0.01 (generated)",
        "field_data_available": "no",
        "notes": "Direct state observation.",
    },
    {
        "measurement_type": "p_injection",
        "symbol": r"$P_i$",
        "description": "Real power injection at a bus",
        "nonlinear_ac_role": "sum_j V_iV_j(G_ij cos + B_ij sin)",
        "linearized_role": "chain-rule rows on angle and voltage states",
        "jacobian_rows": "1 per bus",
        "typical_noise_assumption": "sigma=0.03 (generated)",
        "field_data_available": "no",
        "notes": "Nonlinear in the AC model.",
    },
    {
        "measurement_type": "q_injection",
        "symbol": r"$Q_i$",
        "description": "Reactive power injection at a bus",
        "nonlinear_ac_role": "sum_j V_iV_j(G_ij sin - B_ij cos)",
        "linearized_role": "chain-rule rows on angle and voltage states",
        "jacobian_rows": "1 per bus",
        "typical_noise_assumption": "sigma=0.03 (generated)",
        "field_data_available": "no",
        "notes": "Dropped in the half AC-linearized profile.",
    },
    {
        "measurement_type": "p_branch_flow",
        "symbol": r"$P_{ij}$",
        "description": "Real power flow on a branch",
        "nonlinear_ac_role": "V_i Re(conj(Y_ff)V_i + conj(Y_ft)V_j e^{j delta})",
        "linearized_role": "branch power chain-rule rows",
        "jacobian_rows": "1 per branch",
        "typical_noise_assumption": "sigma=0.02 (generated)",
        "field_data_available": "no",
        "notes": "From-bus branch power.",
    },
    {
        "measurement_type": "q_branch_flow",
        "symbol": r"$Q_{ij}$",
        "description": "Reactive power flow on a branch",
        "nonlinear_ac_role": "imaginary part of the branch power",
        "linearized_role": "branch power chain-rule rows",
        "jacobian_rows": "1 per branch",
        "typical_noise_assumption": "sigma=0.02 (generated)",
        "field_data_available": "no",
        "notes": "Dropped in the half AC-linearized profile.",
    },
    {
        "measurement_type": "dc_branch_flow",
        "symbol": r"$P^{DC}_{ij}$",
        "description": "DC branch active-power flow",
        "nonlinear_ac_role": "not applicable (DC model)",
        "linearized_role": "B_ij(theta_i - theta_j) row",
        "jacobian_rows": "1 per branch",
        "typical_noise_assumption": "sigma=0.02 (generated)",
        "field_data_available": "no",
        "notes": "DC linearized model only.",
    },
    {
        "measurement_type": "dc_bus_injection",
        "symbol": r"$P^{DC}_i$",
        "description": "DC bus active-power injection",
        "nonlinear_ac_role": "not applicable (DC model)",
        "linearized_role": "sum of incident-branch susceptance rows",
        "jacobian_rows": "1 per non-slack bus",
        "typical_noise_assumption": "sigma=0.03 (generated)",
        "field_data_available": "no",
        "notes": "DC linearized model only.",
    },
    {
        "measurement_type": "dc_angle",
        "symbol": r"$\theta_i$",
        "description": "Optional DC angle pseudo-measurement",
        "nonlinear_ac_role": "not applicable (DC model)",
        "linearized_role": "identity row on a non-slack angle state",
        "jacobian_rows": "1 per measured non-slack bus",
        "typical_noise_assumption": "sigma=0.005 (generated)",
        "field_data_available": "no",
        "notes": "Diagnostic angle anchor.",
    },
)

# Maps exact_ac_row_counts columns to measurement types.
_AC_ROW_COLUMNS = {
    "v_rows": "voltage_magnitude",
    "p_injection_rows": "p_injection",
    "q_injection_rows": "q_injection",
    "p_branch_flow_rows": "p_branch_flow",
    "q_branch_flow_rows": "q_branch_flow",
}
_DC_ROW_COLUMNS = {
    "branch_flow": "dc_branch_flow",
    "bus_injection": "dc_bus_injection",
    "angle_rows": "dc_angle",
}


def build_paper_measurement_inventory(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/final_manuscript_package/phase1_measurement_inventory",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    case_rows = case_dimension_rows(input_root)
    inventory_rows = measurement_inventory_rows(input_root)
    type_rows = [dict(t) for t in MEASUREMENT_TYPES]
    workflow_rows = workflow_taxonomy_rows(input_root)
    figure_rows = [
        {
            "case": r["case"],
            "state_dimension": r["state_dimension"],
            "measurement_rows_total": r["measurement_rows_total"],
            "redundancy_ratio": r["redundancy_ratio"],
            "n_bus": r["n_bus"],
        }
        for r in case_rows
    ]

    artifacts = _write_outputs(
        output_dir,
        resolved,
        case_rows=case_rows,
        inventory_rows=inventory_rows,
        type_rows=type_rows,
        workflow_rows=workflow_rows,
        figure_rows=figure_rows,
    )
    return {
        "output_dir": output_dir,
        "case_rows": case_rows,
        "inventory_rows": inventory_rows,
        "type_rows": type_rows,
        "workflow_rows": workflow_rows,
        "artifacts": artifacts,
    }


def case_dimension_rows(input_root: Path) -> list[dict[str, Any]]:
    ac = read_csv(input_root / "measurement_inventory" / "exact_ac_row_counts.csv")
    inventory = read_csv(input_root / "ieee_dataset_visualization" / "ieee_case_inventory.csv")
    topology = (
        {str(row["case"]).lower(): row for _, row in inventory.iterrows()}
        if not inventory.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    if ac.empty:
        return rows
    ac = ac.copy()
    ac["case_key"] = ac["case"].astype(str).str.lower()
    for case_key, group in ac.groupby("case_key", sort=True):
        # Prefer the full AC weighted-Jacobian (nonlinear AC) measurement set.
        preferred = group[group["experiment_group"].astype(str).str.contains("nonlinear")]
        chosen = preferred.iloc[0] if not preferred.empty else group.iloc[0]
        state_dim = int(chosen["state_dimension"])
        total_rows = int(chosen["total_rows"])
        topo = topology.get(case_key)
        n_bus = int(topo["buses"]) if topo is not None else (state_dim + 1) // 2
        n_branch = int(topo["branches"]) if topo is not None else ""
        n_gen = int(topo["generators"]) if topo is not None else ""
        redundancy = round(total_rows / state_dim, 6) if state_dim else ""
        rows.append(
            {
                "case": case_key,
                "base_mva": PYPOWER_BASE_MVA,
                "n_bus": n_bus,
                "n_branch": n_branch,
                "n_gen": n_gen,
                "state_dimension": state_dim,
                "angle_states": n_bus - 1,
                "voltage_magnitude_states": n_bus,
                "measurement_rows_total": total_rows,
                "redundancy_ratio": redundancy,
                "source": "measurement_inventory/exact_ac_row_counts.csv; "
                "ieee_dataset_visualization/ieee_case_inventory.csv",
                "notes": f"AC weighted state model ({chosen['experiment_group']}); "
                "benchmark network model with generated measurement rows.",
            }
        )
    return rows


def measurement_inventory_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ac = read_csv(input_root / "measurement_inventory" / "exact_ac_row_counts.csv")
    for _, record in ac.iterrows():
        workflow = str(record["experiment_group"])
        for column, mtype in _AC_ROW_COLUMNS.items():
            count = int(record.get(column, 0) or 0)
            if count == 0:
                continue
            rows.append(_inventory_row(record["case"], workflow, mtype, count, "AC power flow"))

    dc = read_csv(input_root / "measurement_inventory" / "exact_dc_row_counts.csv")
    for _, record in dc.iterrows():
        workflow = f"DC-linearized ({record.get('case_source', 'builtin')})"
        for column, mtype in _DC_ROW_COLUMNS.items():
            count = int(record.get(column, 0) or 0)
            if count == 0:
                continue
            rows.append(_inventory_row(record["case"], workflow, mtype, count, "DC power flow"))
    return rows


def _inventory_row(
    case: Any, workflow: str, mtype: str, count: int, equation_group: str
) -> dict[str, Any]:
    return {
        "case": str(case).lower(),
        "workflow": workflow,
        "measurement_type": mtype,
        "row_count": count,
        "physical_or_diagnostic": "diagnostic" if mtype == "dc_angle" else "physical",
        "source_equation_group": equation_group,
        "included_in_z": "yes",
        "included_in_hx": "yes",
        "included_in_jacobian": "yes",
        "notes": "Generated from the benchmark network model.",
    }


def workflow_taxonomy_rows(input_root: Path) -> list[dict[str, Any]]:
    taxonomy = read_csv(input_root / "paper_level_summary" / "experiment_taxonomy.csv")
    rows: list[dict[str, Any]] = []
    for _, record in taxonomy.iterrows():
        perturbation = str(record.get("perturbation_location", ""))
        model_type = str(record.get("model_type", ""))
        uses_raw = "yes" if "raw" in perturbation.lower() else "no"
        uses_weighted = "yes" if "weighted residual" in perturbation.lower() else "no"
        uses_iterative = (
            "yes"
            if "iterative" in model_type.lower() or "nonlinear" in model_type.lower()
            else "no"
        )
        rows.append(
            {
                "workflow": str(record.get("experiment_group", "")),
                "uses_raw_measurements": uses_raw,
                "uses_weighted_residual": uses_weighted,
                "uses_iterative_jacobian_rebuild": uses_iterative,
                "perturbation_model": perturbation or "n/a",
                "primary_outputs": str(record.get("main_metrics", "")),
                "claim_boundary": str(record.get("limitations", "")),
                "notes": str(record.get("paper_role", "")),
            }
        )
    return rows


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    case_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    workflow_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    case_path = rows_to_table(
        case_rows, output_dir / "paper_table_case_dimensions.csv", CASE_DIMENSION_COLUMNS
    )
    inventory_path = rows_to_table(
        inventory_rows,
        output_dir / "paper_table_measurement_inventory.csv",
        MEASUREMENT_INVENTORY_COLUMNS,
    )
    type_path = rows_to_table(
        type_rows, output_dir / "paper_table_measurement_types.csv", MEASUREMENT_TYPE_COLUMNS
    )
    workflow_path = rows_to_table(
        workflow_rows, output_dir / "paper_table_workflow_taxonomy.csv", WORKFLOW_TAXONOMY_COLUMNS
    )
    figure_path = rows_to_table(
        figure_rows, output_dir / "measurement_inventory_figure_data.csv", FIGURE_DATA_COLUMNS
    )
    summary_path = output_dir / "measurement_model_summary.md"
    summary_path.write_text(_summary_markdown(case_rows), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "paper_table_case_dimensions": str(case_path),
            "paper_table_measurement_inventory": str(inventory_path),
            "paper_table_measurement_types": str(type_path),
            "paper_table_workflow_taxonomy": str(workflow_path),
            "measurement_model_summary": str(summary_path),
            "measurement_inventory_figure_data": str(figure_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "paper_table_case_dimensions": case_path,
        "paper_table_measurement_inventory": inventory_path,
        "paper_table_measurement_types": type_path,
        "paper_table_workflow_taxonomy": workflow_path,
        "measurement_model_summary": summary_path,
        "measurement_inventory_figure_data": figure_path,
    }


def _summary_markdown(case_rows: list[dict[str, Any]]) -> str:
    cases = ", ".join(str(r["case"]) for r in case_rows) or "none"
    return "\n".join(
        [
            "# Measurement Model and Benchmark Construction",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "IEEE/PYPOWER cases provide benchmark network models. Measurement rows are "
            "generated from the network model; no real PMU/SCADA field records are used.",
            "",
            "## Nonlinear AC raw measurement model",
            "",
            r"\[",
            r"z = h(x_{\mathrm{true}}) + e + b.",
            r"\]",
            "",
            "## Linearized / single-step weighted residual model",
            "",
            r"\[",
            r"\tilde r_{\mathrm{perturbed}}",
            r"=",
            r"\tilde r_{\mathrm{clean}} + \tilde e + \tilde b.",
            r"\]",
            "",
            "## Diagonal covariance and weighting",
            "",
            r"\[",
            r"R_{ii} = \sigma_i^2,",
            r"\qquad",
            r"\tilde H = R^{-1/2}H,",
            r"\qquad",
            r"\tilde r = R^{-1/2}r.",
            r"\]",
            "",
            "## Coverage",
            f"- Cases summarized: {cases}.",
            "- AC measurement types: voltage magnitude, real/reactive power injection, and "
            "real/reactive branch flow. DC measurement types: branch flow, bus injection, and "
            "optional angle anchors.",
            "- The AC state vector concatenates non-slack bus angles and all bus voltage "
            "magnitudes, so the AC state dimension is 2*n_bus - 1.",
            "- Workflows separate the nonlinear AC raw-measurement perturbation (z = h(x) + e + b) "
            "from the single-step weighted residual perturbation (R^{-1/2} r).",
            "",
        ]
    )
