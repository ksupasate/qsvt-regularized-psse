"""Shared block / functional inventory writers (parameterized by study id)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import CaseDesign, json_safe_floats
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    array_fingerprint,
    atomic_write_csv,
    atomic_write_json,
    now_iso,
)


def write_block_inventory(
    case_name: str, design: CaseDesign, destination: Path, study_id: str
) -> dict[str, Any]:
    binding = design.binding
    small = design.small
    payload = {
        "study_id": study_id,
        "generated_at": now_iso(),
        "claim_boundary": CLAIM_BOUNDARY,
        "case_name": case_name,
        "case_source": "pypower",
        "matrix_seed": binding.seed,
        "block_shape": [small.dimension, small.dimension],
        "block_selection_policy": "largest_row_col_norms",
        "selected_global_rows": list(binding.selected_rows),
        "selected_global_columns": list(binding.selected_columns),
        "block_alpha_4_sigma_min_pos_sq": small.alpha,
        "matrix_fingerprint": array_fingerprint(small.matrix),
        "conditioning": json_safe_floats(design.conditioning),
        "state_dimension": binding.state_dimension,
        "slack_bus": binding.slack_bus,
        "columns": [
            {
                "local_index": rec.local_index,
                "global_state_index": rec.global_state_index,
                "state_type": rec.state_type,
                "bus_id": rec.bus_id,
            }
            for rec in binding.columns
        ],
        "measurement_rows": [
            {"row": int(r), "type": t, "label": lbl}
            for r, t, lbl in zip(
                binding.selected_rows, binding.measurement_types, binding.measurement_labels,
                strict=True,
            )
        ],
        "representable_angle_branches": [list(b) for b in binding.representable_angle_branches],
        "n_network_branches": len(binding.branches),
    }
    atomic_write_json(destination / "block_inventory.json", payload)
    return payload


def write_functional_inventory(
    design: CaseDesign, destination: Path, study_id: str
) -> dict[str, Any]:
    records = design.functional_records
    for rec in records:
        if not rec.unit_norm:
            raise RuntimeError(
                f"functional {rec.functional_id} not unit-norm ({rec.unit_norm_error})"
            )
    frame = pd.DataFrame([rec.to_row() for rec in records])
    atomic_write_csv(destination / "functional_inventory.csv", frame)
    unavailable_rows = [
        {
            "requested_functional_id": u.requested_functional_id,
            "family": u.family,
            "reason_unavailable": u.reason_unavailable,
            "deterministic_selection_policy": u.deterministic_selection_policy,
            "requested_target": u.requested_target,
        }
        for u in design.unavailable
    ]
    unavailable = pd.DataFrame(unavailable_rows) if unavailable_rows else pd.DataFrame(
        columns=[
            "requested_functional_id", "family", "reason_unavailable",
            "deterministic_selection_policy", "requested_target",
        ]
    )
    atomic_write_csv(destination / "unavailable_functionals.csv", unavailable)
    inventory_json = {
        "study_id": study_id,
        "generated_at": now_iso(),
        "functionals": [rec.to_row() | {"vector_list": list(rec.vector)} for rec in records],
        "functional_fingerprints": {
            rec.functional_id: array_fingerprint(np.asarray(rec.vector)) for rec in records
        },
        "family_counts": frame.groupby("family")["functional_id"].count().to_dict()
        if not frame.empty else {},
        "unavailable": [u.requested_functional_id for u in design.unavailable],
    }
    atomic_write_json(destination / "functional_inventory.json", inventory_json)
    return inventory_json
