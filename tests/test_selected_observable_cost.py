from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_common import (
    WORKLOAD_BANNED_OVERCLAIMS,
    forbidden_in,
)
from robust_qsvt_se.paper.selected_observable_consolidation import run_consolidation
from robust_qsvt_se.paper.selected_observable_cost import (
    COST_COLUMNS,
    run_selected_observable_cost_accounting,
)

# Exact banned phrases from the task brief (case-insensitive).
TASK_BANNED = (
    "quantum speedup demonstrated",
    "qsvt outperforms ridge",
    "field pmu/scada validation",
    "full ieee-scale hardware execution",
    "full-vector readout solved",
)


@pytest.fixture(scope="module")
def cost_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("cost")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_selected_observable_cost_accounting(
            {"output_dir": str(out), "cases": ["ieee14"], "trials": 20, "command": "test"}
        )


def test_cost_columns_present(cost_run: dict) -> None:
    assert set(COST_COLUMNS).issubset(cost_run["cost"].columns)


def test_cost_terms_have_expected_status_labels(cost_run: dict) -> None:
    cost = cost_run["cost"]
    assert (cost["T_U_status"] == "spectrum_point_action_only").all()
    assert (cost["T_access_status"] == "modeled").all()
    assert (cost["T_prep_status"] == "modeled").all()
    assert (cost["T_amp_status"] == "modeled").all()
    assert (cost["T_post_status"] == "proxy").all()
    assert (cost["T_readout_status"] == "proxy").all()
    assert (cost["classical_sparse_baseline_status"] == "proxy").all()
    # Full-vector recovery is excluded for every row.
    assert bool((~cost["full_vector_recovery_included"]).all())


def test_query_count_is_two_d_plus_one(cost_run: dict) -> None:
    cost = cost_run["cost"]
    for _, row in cost.iterrows():
        if row["degree"] != "" and row["qsvt_query_count"] != "":
            assert int(row["qsvt_query_count"]) == 2 * int(row["degree"]) + 1


def test_repetition_cost_multiplies_shots_postselection_and_unitary_calls(
    cost_run: dict,
) -> None:
    for _, row in cost_run["cost"].iterrows():
        expected_attempts = float(row["shots"]) / float(row["success_probability_proxy"])
        assert float(row["expected_attempts_no_aa"]) == pytest.approx(expected_attempts)
        assert float(row["expected_unitary_queries_no_aa"]) == pytest.approx(
            expected_attempts * int(row["unitary_queries_per_attempt"])
        )


def test_amplitude_amplified_proxy_uses_sqrt_success_scaling(cost_run: dict) -> None:
    for _, row in cost_run["cost"].iterrows():
        attempts = float(row["shots"]) / np.sqrt(float(row["success_probability_proxy"]))
        assert float(row["expected_attempts_with_aa_proxy"]) == pytest.approx(attempts)
        assert float(row["expected_unitary_queries_with_aa_proxy"]) == pytest.approx(
            attempts * int(row["unitary_queries_per_attempt"])
        )


def test_success_probability_proxy_in_unit_interval(cost_run: dict) -> None:
    proxy = cost_run["cost"]["success_probability_proxy"]
    assert ((proxy >= 0.0) & (proxy <= 1.0)).all()


def test_cost_summary_states_boundaries_and_is_safe(cost_run: dict) -> None:
    output_dir = Path(cost_run["output_dir"])
    text = (output_dir / "selected_observable_cost_summary.md").read_text(encoding="utf-8")
    assert forbidden_in(text) == []
    lowered = text.lower()
    assert "selected-observable" in lowered
    assert "not full-vector readout" in lowered
    assert "excluded" in lowered


def test_consolidation_claim_boundary_excludes_task_banned_phrases(
    tmp_path: Path,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_consolidation(
            {
                "output_dir": str(tmp_path),
                "readout_trials": 20,
                "cost_trials": 20,
                "command": "test",
            }
        )
    # Every required final file exists.
    required = [
        "implementation_audit.md",
        "repo_integration_plan.json",
        "sparse_access_summary.csv",
        "sparse_access_validation.csv",
        "selected_observables.csv",
        "readout_shot_sweep.csv",
        "readout_map.csv",
        "degree_aware_alpha_grid.csv",
        "degree_aware_alpha_summary.csv",
        "selected_observable_cost.csv",
        "selected_observable_cost_summary.md",
        "claim_boundary_update.md",
        "paper_ready_tables.md",
        "manifest.json",
    ]
    for name in required:
        assert (tmp_path / name).is_file(), f"missing {name}"

    for name in ("claim_boundary_update.md", "paper_ready_tables.md"):
        text = (tmp_path / name).read_text(encoding="utf-8").lower()
        assert forbidden_in(text) == []
        for phrase in TASK_BANNED:
            assert phrase not in text
        for phrase in WORKLOAD_BANNED_OVERCLAIMS:
            assert phrase not in text

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["changes_estimator_behavior"] is False
    assert manifest["checksums"]
    assert manifest["git_commit_hash"] is None or isinstance(manifest["git_commit_hash"], str)
    assert manifest["aggregated_from_existing"] is True
    assert run["artifacts"]["manifest"].is_file()
