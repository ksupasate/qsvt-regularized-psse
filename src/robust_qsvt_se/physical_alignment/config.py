"""Configuration loading and frozen-protocol validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/tqe_physical_alignment/campaign.json")
PROTOCOL_ID = "tqe_physical_alignment_and_generalization_v1"
REQUIRED_CASES = ("ieee14", "ieee30", "ieee57")
REQUIRED_RISK_SELECTORS = {
    "noise_propagation_risk_mean_initial",
    "noise_propagation_risk_mean_refined",
    "noise_propagation_risk_worst_initial",
    "noise_propagation_risk_worst_refined",
    "posterior_variance_reference_mean_initial",
    "posterior_variance_reference_mean_refined",
    "posterior_variance_reference_worst_initial",
    "posterior_variance_reference_worst_refined",
}


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def configuration_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_campaign_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("campaign configuration must contain a JSON object")
    if payload.get("configuration_id") != PROTOCOL_ID or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"configuration and protocol IDs must equal {PROTOCOL_ID}")
    if not payload.get("declared_before_evaluation", False):
        raise ValueError("campaign must be declared before evaluation")
    if tuple(payload.get("cases", ())) != REQUIRED_CASES:
        raise ValueError("campaign cases must be IEEE-14, IEEE-30, and IEEE-57 in order")

    structure = payload["structure_design"]
    if int(structure["groups_per_case"]) < 4:
        raise ValueError("at least four independent structures per case are required")
    if int(structure["realizations_per_group"]) < 2:
        raise ValueError("at least two numerical realizations per structure are required")
    if int(structure["training_seed_count_per_instance"]) < 20:
        raise ValueError("at least 20 training residual seeds per instance are required")
    if int(structure["held_out_seed_count_per_instance"]) < 20:
        raise ValueError("at least 20 held-out residual seeds per instance are required")
    if not structure.get("selection_must_be_outcome_independent", False):
        raise ValueError("structure selection must be outcome-independent")

    audit = payload["physical_audit"]
    selectors = list(audit["selectors"])
    if len(selectors) != len(set(selectors)):
        raise ValueError("selector names must be unique")
    missing = REQUIRED_RISK_SELECTORS - set(selectors)
    if missing:
        raise ValueError(f"required risk selectors are missing: {sorted(missing)}")
    if "full_support" not in selectors:
        raise ValueError("full support is a mandatory reference")
    if set(audit["deployable_selectors"]) & set(audit["diagnostic_selectors"]):
        raise ValueError("deployable and diagnostic selector inventories must be disjoint")
    if float(audit["physical_error_floor"]) <= 0 or float(audit["support_error_floor"]) <= 0:
        raise ValueError("normalization floors must be positive")
    if not audit["support_budgets"] or not audit["slot_budgets"]:
        raise ValueError("support and slot budgets must be nonempty")

    stats = payload["statistics"]
    if stats["primary_independent_unit"] != "structural_group_id":
        raise ValueError("the structure must be the primary independent unit")
    if int(stats["bootstrap_replicates"]) < 10_000:
        raise ValueError("at least 10,000 bootstrap replicates are required")

    nonlinear = payload["nonlinear_ac"]
    if len(nonlinear["seeds"]) < 20 or len(set(nonlinear["seeds"])) != len(nonlinear["seeds"]):
        raise ValueError("nonlinear campaign needs at least 20 unique seeds")
    scenario_ids = {row["scenario_id"] for row in nonlinear["scenarios"]}
    required_scenarios = {
        "gaussian_noise_baseline",
        "random_missing_measurement_stress",
        "sparse_signed_bad_data_stress",
    }
    if scenario_ids != required_scenarios:
        raise ValueError("nonlinear scenario inventory must contain the three frozen families")

    payload["configuration_hash"] = configuration_hash(payload)
    payload["configuration_path"] = str(source)
    return payload
