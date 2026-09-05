"""Task C: end-to-end QSVT-compatible cost accounting with excluded costs.

Builds a single component table for the modeled total cost

    T_total = T_access + T_prep + d T_U + (d+1) T_phase + T_post + T_amp + T_readout
              + T_full_vector_recovery (excluded),

and records, per component, whether it is validated/modeled/proxy/excluded,
whether it already appears in an existing resource table, whether it is a
dominant scaling risk, and the manuscript-safe vs. to-avoid wording.

Representative numeric values are *ingested* from existing resource outputs
(oracle-model resource summary, hardware-aware cost model, query-count summary).
Nothing is invented: a missing source leaves the representative value blank and
notes the absence. This is a cost-accounting proxy for feasibility discussion -
not a quantum-speedup claim, not full hardware execution, not field validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper._common import read_csv
from robust_qsvt_se.paper.tqe_revision_support_common import (
    REVISION_OUTPUT_ROOT,
    find_forbidden,
    write_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory

COST_ACCOUNTING_DIR = REVISION_OUTPUT_ROOT / "cost_accounting"

SAFE_WORDING_HEADER = (
    "Cost-accounting proxy for feasibility discussion under modeled sparse-access "
    "assumptions. This is not a speedup claim; it is simulator/resource-model level only "
    "(no hardware run) on benchmark generated measurements (not field PMU/SCADA data)."
)

COMPONENT_COLUMNS = [
    "component",
    "formula_term",
    "current_status",
    "included_in_existing_resource_table",
    "dominant_risk",
    "representative_value",
    "representative_unit",
    "representative_source",
    "signal_unitary_calls",
    "projector_phase_operations",
    "alternating_sequence_length",
    "manuscript_safe_wording",
    "avoid_wording",
    "notes",
]

PROVENANCE_COLUMNS = [
    "source",
    "case",
    "degree_d",
    "computed_signal_unitary_calls",
    "computed_projector_phase_operations",
    "computed_alternating_sequence_length",
    "reported_query_count",
    "reported_count_matches_signal_calls",
]

# Resource outputs ingested for representative values (read-only, fail-soft).
ORACLE_MODEL_CSV = "outputs/qsvt_oracle_model_resources/oracle_model_resource_summary.csv"
HARDWARE_AWARE_CSV = "outputs/hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv"
QUERY_SUMMARY_CSV = "outputs/full_qsvt_ieee_hardware_resources/qsvt_query_count_summary.csv"
RESOURCE_FULL_CSV = "outputs/qsvt_resource_full_ieee/qsvt_resource_estimates.csv"

REPRESENTATIVE_CASE = "ieee14"
REPRESENTATIVE_DEGREE_HINT = 51


@dataclass(frozen=True, slots=True)
class CostComponent:
    component: str
    formula_term: str
    current_status: str
    included_in_existing_resource_table: bool
    dominant_risk: str
    representative_key: str
    representative_unit: str
    manuscript_safe_wording: str
    avoid_wording: str
    notes: str


def query_count(degree: int) -> int:
    """Signal-unitary call count for the implemented degree-``d`` convention."""

    if int(degree) < 0:
        raise ValueError("degree must be nonnegative")
    return int(degree)


COST_COMPONENTS: tuple[CostComponent, ...] = (
    CostComponent(
        component="access",
        formula_term="T_access",
        current_status="modeled",
        included_in_existing_resource_table=True,
        dominant_risk="yes - sparse-access index/value oracle is modeled, not compiled "
        "to a reversible circuit",
        representative_key="block_encoding_query_cost",
        representative_unit="block-encoding query-cost proxy",
        manuscript_safe_wording="Sparse index/value oracle access (or dense block "
        "construction for selected small subproblems) under modeled sparse-access "
        "assumptions.",
        avoid_wording="implemented scalable reversible sparse oracle circuit",
        notes="Modeled via sparse-access oracle cost model; dense block encoding only "
        "for selected small subproblems.",
    ),
    CostComponent(
        component="state_prep",
        formula_term="T_prep",
        current_status="modeled",
        included_in_existing_resource_table=True,
        dominant_risk="yes - efficient state-preparation loader is assumed, not synthesized",
        representative_key="state_preparation_query_cost",
        representative_unit="state-preparation query-cost proxy",
        manuscript_safe_wording="Preparation of the weighted-residual state |r~> is an "
        "assumed/modeled amplitude or qRAM-style loader.",
        avoid_wording="efficient qRAM state-preparation loader implemented",
        notes="State preparation is assumed/modeled, not efficiently synthesized.",
    ),
    CostComponent(
        component="qsvt_signal_and_phase_sequence",
        formula_term="d T_U + (d+1) T_phase",
        current_status="validated",
        included_in_existing_resource_table=True,
        dominant_risk="no - signal calls and projector phases are counted separately",
        representative_key="qsvt_query_count",
        representative_unit="signal-unitary calls",
        manuscript_safe_wording="A degree-d sequence uses d calls to U_A or U_A^dagger "
        "and d+1 projector phases; 2d+1 is only the alternating sequence length.",
        avoid_wording="quantum speedup from fewer queries than classical",
        notes="Signal-unitary and phase-operation counts are kept separate.",
    ),
    CostComponent(
        component="postselection",
        formula_term="T_post",
        current_status="proxy",
        included_in_existing_resource_table=True,
        dominant_risk="yes - success probability can be small, inflating expected repeats",
        representative_key="success_probability_proxy",
        representative_unit="resource-model postselection success-probability proxy",
        manuscript_safe_wording="Postselection overhead is reported through a "
        "resource-model success-probability proxy (distinct from the gate-level "
        "selected-block success probability in the resource table); expected repeats "
        "scale as 1/p.",
        avoid_wording="deterministic single-shot QSVT output",
        notes="Success-probability proxy only; no measured hardware yield.",
    ),
    CostComponent(
        component="amplitude_amplification",
        formula_term="T_amp",
        current_status="modeled",
        included_in_existing_resource_table=True,
        dominant_risk="no - bounded O(1/sqrt(p)) overhead when applied",
        representative_key="amplitude_amplification_proxy",
        representative_unit="amplification rounds (proxy)",
        manuscript_safe_wording="Optional amplitude amplification contributes a modeled "
        "O(1/sqrt(p)) overhead when used.",
        avoid_wording="amplitude amplification demonstrates quantum advantage",
        notes="Modeled overhead; not a speedup claim.",
    ),
    CostComponent(
        component="readout",
        formula_term="T_readout",
        current_status="proxy",
        included_in_existing_resource_table=True,
        dominant_risk="yes - observable shot count is the dominant included runtime term",
        representative_key="observable_readout_shots",
        representative_unit="shots",
        manuscript_safe_wording="Selected-observable extraction (energy-style or the "
        "sign-aware selected-observable diagnostic) uses a shot budget; it is flagged "
        "as a dominant-cost risk in the existing model.",
        avoid_wording="full-vector readout solved",
        notes="Existing cost model flags readout as a dominant-cost risk.",
    ),
    CostComponent(
        component="full_vector_recovery",
        formula_term="n_states * T_readout(component)",
        current_status="excluded",
        included_in_existing_resource_table=False,
        dominant_risk="yes - full signed-vector recovery would dominate and is excluded from scope",
        representative_key="state_dimension",
        representative_unit="state components requiring separate readout",
        manuscript_safe_wording="Full signed-vector recovery is excluded: it would require "
        "a separate selected-observable readout per state component and remains out of scope.",
        avoid_wording="full-vector quantum state readout achieved",
        notes="Explicitly excluded cost; not implemented.",
    ),
)


def build_cost_accounting(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", COST_ACCOUNTING_DIR)))
    case = str(resolved.get("case", REPRESENTATIVE_CASE))

    ingested = _ingest_representative_values(case=case, degree_hint=REPRESENTATIVE_DEGREE_HINT)
    component_rows = [_component_row(component, ingested) for component in COST_COMPONENTS]
    component_frame = pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)
    provenance_frame = _query_provenance(case=case)

    _self_check_safe_fields(component_frame)

    component_csv = output_dir / "qsvt_cost_accounting.csv"
    provenance_csv = output_dir / "qsvt_cost_accounting_query_provenance.csv"
    summary_md = output_dir / "qsvt_cost_accounting.md"
    component_frame.to_csv(component_csv, index=False)
    provenance_frame.to_csv(provenance_csv, index=False)
    summary_md.write_text(
        _summary_markdown(component_frame, provenance_frame, ingested, case),
        encoding="utf-8",
    )

    artifacts = {
        "qsvt_cost_accounting_csv": component_csv,
        "qsvt_cost_accounting_query_provenance": provenance_csv,
        "qsvt_cost_accounting_md": summary_md,
    }
    manifest = write_manifest(
        output_dir=output_dir,
        artifact_name="qsvt_cost_accounting",
        description=(
            "End-to-end QSVT-compatible cost accounting separating included and excluded "
            "costs. Cost-accounting proxy; not a speedup claim and not hardware execution."
        ),
        artifacts=artifacts,
        extra={
            "representative_case": case,
            "representative_degree": ingested.get("degree"),
            "representative_signal_unitary_calls": (
                query_count(int(ingested["degree"])) if ingested.get("degree") is not None else None
            ),
            "ingested_sources": ingested.get("sources", {}),
            "excluded_components": ["full_vector_recovery"],
            "speedup_claim": False,
            "hardware_execution": False,
        },
        manifest_name="qsvt_cost_accounting_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "component_frame": component_frame,
        "provenance_frame": provenance_frame,
        "ingested": ingested,
        "artifacts": artifacts,
    }


def _ingest_representative_values(*, case: str, degree_hint: int) -> dict[str, Any]:
    oracle = read_csv(ORACLE_MODEL_CSV)
    hardware = read_csv(HARDWARE_AWARE_CSV)
    query_summary = read_csv(QUERY_SUMMARY_CSV)
    resource_full = read_csv(RESOURCE_FULL_CSV)

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    oracle_row = _row_for_case(oracle, case, "case")
    if oracle_row is not None:
        for key in (
            "degree",
            "qsvt_query_count",
            "block_encoding_query_cost",
            "state_preparation_query_cost",
            "readout_shots",
        ):
            value = _scalar(oracle_row, key)
            if value is not None:
                values[key] = value
                sources[key] = ORACLE_MODEL_CSV
        state_dim = _scalar(oracle_row, "matrix_cols")
        if state_dim is not None:
            values["state_dimension"] = state_dim
            sources["state_dimension"] = ORACLE_MODEL_CSV

    hardware_row = _hardware_representative_row(hardware, case=case, degree_hint=degree_hint)
    if hardware_row is not None:
        for key in (
            "success_probability_proxy",
            "amplitude_amplification_proxy",
            "observable_readout_shots",
            "dominant_cost_source",
            "estimated_total_query_cost",
        ):
            value = _scalar(hardware_row, key)
            if value is not None:
                values[key] = value
                sources[key] = HARDWARE_AWARE_CSV
        if "observable_readout_shots" in values:
            # Prefer the explicit shot budget from the hardware-aware model for readout.
            values["readout_shots"] = values["observable_readout_shots"]
            sources["readout_shots"] = HARDWARE_AWARE_CSV

    query_row = _row_for_case(query_summary, case, "case_name")
    if query_row is not None:
        for key in ("qsvt_degree", "query_count"):
            value = _scalar(query_row, key)
            if value is not None:
                values.setdefault("degree" if key == "qsvt_degree" else "qsvt_query_count", value)
                sources.setdefault(
                    "degree" if key == "qsvt_degree" else "qsvt_query_count", QUERY_SUMMARY_CSV
                )

    if "state_dimension" not in values:
        resource_row = _row_for_case(resource_full, case, "case_name")
        if resource_row is not None:
            state_dim = _scalar(resource_row, "matrix_columns")
            if state_dim is not None:
                values["state_dimension"] = state_dim
                sources["state_dimension"] = RESOURCE_FULL_CSV

    # Legacy resource files used ``2d+1`` as a generic sequence count. For the
    # implemented circuit convention, the signal-unitary call count is ``d``;
    # recompute it from the recorded degree instead of propagating the stale label.
    if values.get("degree") is not None:
        values["qsvt_query_count"] = query_count(int(values["degree"]))
        sources["qsvt_query_count"] = "recomputed from degree using implemented convention"

    values["sources"] = sources
    return values


def _component_row(component: CostComponent, ingested: dict[str, Any]) -> dict[str, Any]:
    representative_value = ingested.get(component.representative_key)
    representative_source = ingested.get("sources", {}).get(component.representative_key, "")
    notes = component.notes
    if representative_value is None:
        notes = f"{notes} (source output missing; representative value unavailable)"
    signal_calls: Any = ""
    projector_phases: Any = ""
    sequence_length: Any = ""
    if (
        component.component == "qsvt_signal_and_phase_sequence"
        and ingested.get("degree") is not None
    ):
        degree = int(ingested["degree"])
        signal_calls = query_count(degree)
        projector_phases = degree + 1
        sequence_length = 2 * degree + 1
    return {
        "component": component.component,
        "formula_term": component.formula_term,
        "current_status": component.current_status,
        "included_in_existing_resource_table": bool(component.included_in_existing_resource_table),
        "dominant_risk": component.dominant_risk,
        "representative_value": "" if representative_value is None else representative_value,
        "representative_unit": component.representative_unit,
        "representative_source": representative_source,
        "signal_unitary_calls": signal_calls,
        "projector_phase_operations": projector_phases,
        "alternating_sequence_length": sequence_length,
        "manuscript_safe_wording": component.manuscript_safe_wording,
        "avoid_wording": component.avoid_wording,
        "notes": notes,
    }


def _query_provenance(*, case: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = (
        ("oracle_model_resource_summary", ORACLE_MODEL_CSV, "case", "degree", "qsvt_query_count"),
        ("qsvt_query_count_summary", QUERY_SUMMARY_CSV, "case_name", "qsvt_degree", "query_count"),
        (
            "qsvt_resource_full_ieee",
            RESOURCE_FULL_CSV,
            "case_name",
            "degree",
            "estimated_qsvt_query_count",
        ),
    )
    for source, path, case_col, degree_col, query_col in specs:
        frame = read_csv(path)
        row = _row_for_case(frame, case, case_col)
        if row is None:
            rows.append(
                {
                    "source": source,
                    "case": case,
                    "degree_d": "",
                    "computed_signal_unitary_calls": "",
                    "computed_projector_phase_operations": "",
                    "computed_alternating_sequence_length": "",
                    "reported_query_count": "",
                    "reported_count_matches_signal_calls": "source_missing",
                }
            )
            continue
        degree = _scalar(row, degree_col)
        reported = _scalar(row, query_col)
        computed_signal = query_count(int(degree)) if degree is not None else None
        computed_phases = int(degree) + 1 if degree is not None else None
        computed_length = 2 * int(degree) + 1 if degree is not None else None
        if computed_signal is None or reported is None:
            matches: Any = "unknown"
        else:
            matches = bool(int(reported) == int(computed_signal))
        rows.append(
            {
                "source": source,
                "case": case,
                "degree_d": "" if degree is None else int(degree),
                "computed_signal_unitary_calls": (
                    "" if computed_signal is None else int(computed_signal)
                ),
                "computed_projector_phase_operations": (
                    "" if computed_phases is None else int(computed_phases)
                ),
                "computed_alternating_sequence_length": (
                    "" if computed_length is None else int(computed_length)
                ),
                "reported_query_count": "" if reported is None else int(reported),
                "reported_count_matches_signal_calls": matches,
            }
        )
    return pd.DataFrame(rows, columns=PROVENANCE_COLUMNS)


def _summary_markdown(
    components: pd.DataFrame,
    provenance: pd.DataFrame,
    ingested: dict[str, Any],
    case: str,
) -> str:
    degree = ingested.get("degree")
    signal_calls = query_count(int(degree)) if degree is not None else None
    lines = [
        "# QSVT-Compatible End-to-End Cost Accounting",
        "",
        SAFE_WORDING_HEADER,
        "",
        "## Cost Model",
        "",
        "```text",
        "T_total = T_access + T_prep + d T_U + (d+1) T_phase",
        "          + T_post + T_amp + T_readout",
        "          + T_full_vector_recovery   (excluded)",
        "```",
        "",
        f"Representative case: `{case}`. "
        + (
            f"Degree d = {int(degree)} gives {int(signal_calls)} signal-unitary calls, "
            f"{int(degree) + 1} projector phases, and alternating length "
            f"{2 * int(degree) + 1}."
            if degree is not None
            else "Degree unavailable from existing outputs."
        ),
        "",
        "Representative numeric values are ingested from existing resource outputs; a "
        "blank value means the source output was absent (no value is invented).",
        "",
        "## Components (included vs excluded)",
        "",
        "| Component | Term | Status | In existing table | Dominant risk | Representative |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in components.iterrows():
        rep = row["representative_value"]
        rep_text = "-" if rep == "" else f"{rep} {row['representative_unit']}"
        lines.append(
            f"| {row['component']} | `{row['formula_term']}` | {row['current_status']} | "
            f"{row['included_in_existing_resource_table']} | "
            f"{row['dominant_risk'].split(' - ')[0]} | {rep_text} |"
        )
    lines += [
        "",
        "## Signal-Call and Alternating-Sequence Provenance",
        "",
        "| Source | d | Signal calls | Phase ops | Alternating length | "
        "Reported count | Matches signal calls |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in provenance.iterrows():
        lines.append(
            f"| {row['source']} | {row['degree_d']} | "
            f"{row['computed_signal_unitary_calls']} | "
            f"{row['computed_projector_phase_operations']} | "
            f"{row['computed_alternating_sequence_length']} | "
            f"{row['reported_query_count']} | "
            f"{row['reported_count_matches_signal_calls']} |"
        )
    lines += [
        "",
        "## Manuscript-Safe Wording per Component",
        "",
    ]
    for _, row in components.iterrows():
        lines.append(
            f"- **{row['component']}** ({row['current_status']}): {row['manuscript_safe_wording']}"
        )
    lines += [
        "",
        "## Excluded Costs",
        "",
        "- `full_vector_recovery` is **excluded**: recovering every signed state component "
        "would require a separate selected-observable readout per coordinate and is out of "
        "scope. The included `readout` row is flagged as a dominant-cost risk in the modeled "
        "total under the current observable-extraction model.",
        "",
        "## Boundary",
        "",
        "These figures are a cost-accounting proxy under modeled sparse-access assumptions. "
        "This is **not a speedup claim**; it is simulator/resource-model level only (no "
        "hardware run) on benchmark generated measurements (not field PMU/SCADA data). The "
        "`avoid_wording` column lists phrasing to avoid; it is a do-not-use list, not a set "
        "of supported claims.",
        "",
    ]
    return "\n".join(lines)


def _self_check_safe_fields(components: pd.DataFrame) -> None:
    safe_text = "\n".join(
        [SAFE_WORDING_HEADER, *components["manuscript_safe_wording"].astype(str).tolist()]
    )
    violations = find_forbidden(safe_text)
    if violations:
        raise RuntimeError(f"cost-accounting safe wording contains forbidden phrases: {violations}")


def _row_for_case(frame: pd.DataFrame, case: str, case_col: str) -> pd.Series | None:
    if frame.empty:
        return None
    if case_col in frame.columns:
        matched = frame[frame[case_col].astype(str) == case]
        if not matched.empty:
            return matched.iloc[0]
    return frame.iloc[0]


def _hardware_representative_row(
    frame: pd.DataFrame, *, case: str, degree_hint: int
) -> pd.Series | None:
    if frame.empty:
        return None
    subset = frame
    if "case" in frame.columns:
        by_case = frame[frame["case"].astype(str) == case]
        if not by_case.empty:
            subset = by_case
    if "qsvt_degree" in subset.columns:
        by_degree = subset[subset["qsvt_degree"] == degree_hint]
        if not by_degree.empty:
            subset = by_degree
    if "observable_readout_shots" in subset.columns:
        subset = subset.sort_values("observable_readout_shots", ascending=False)
    return subset.iloc[0]


def _scalar(row: pd.Series, key: str) -> Any:
    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the QSVT cost-accounting table")
    parser.add_argument("--output-dir", default=str(COST_ACCOUNTING_DIR))
    parser.add_argument("--case", default=REPRESENTATIVE_CASE)
    args = parser.parse_args(argv)
    run = build_cost_accounting({"output_dir": args.output_dir, "case": args.case})
    print(f"Cost accounting complete: {run['artifacts']['qsvt_cost_accounting_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
