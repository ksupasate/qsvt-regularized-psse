"""Train/test leakage guards and protected-baseline regression tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    compute_output_aware_entry_scores,
    load_study_configuration,
    load_support,
    refine_support_one_swap,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint


def _task(split: str) -> RidgeTask:
    return RidgeTask(
        task_id=f"{split}_seed1_coordinate_e0",
        seed_id=1,
        split=split,
        residual=np.array([1.0, -0.5, 0.2, 0.7]),
        functional_id="coordinate_e0",
        functional=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_predeclared_training_and_heldout_seeds_are_strictly_disjoint():
    config = load_study_configuration()
    training = set(config["training_seed_ids"])
    heldout = set(config["held_out_seed_ids"])
    assert training
    assert heldout
    assert training.isdisjoint(heldout)
    assert config["declared_before_selector_evaluation"] is True


def test_heldout_tasks_are_rejected_by_score_construction():
    matrix = np.eye(4)
    with pytest.raises(ValueError, match="training tasks only"):
        compute_output_aware_entry_scores(
            matrix, [_task("held_out")], alpha=1.0, epsilon=1.0e-15
        )


def test_heldout_tasks_are_rejected_by_refinement():
    matrix = np.ones((4, 4)) + 3.0 * np.eye(4)
    support = np.eye(4, dtype=bool)
    with pytest.raises(ValueError, match="training tasks only"):
        refine_support_one_swap(
            matrix,
            support,
            [_task("held_out")],
            SupportConstraints(4, 1, True),
            alpha=1.0,
            y_floor=1.0e-6,
            objective="mean_normalized_error",
            max_iterations=1,
            improvement_tolerance=1.0e-12,
        )


def test_generated_split_and_support_provenance_exclude_heldout_if_present():
    output = Path("outputs/output_aware_sparse_selection")
    split_path = output / "residual_split.json"
    registry_path = output / "support_registry.csv"
    if not split_path.is_file() or not registry_path.is_file():
        pytest.skip("campaign split/support artifacts not generated yet")
    import pandas as pd

    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert set(split["training_seed_ids"]).isdisjoint(split["held_out_seed_ids"])
    registry = pd.read_csv(registry_path)
    assert not registry["selection_data_split"].astype(str).str.contains("held_out").any()
    metadata = json.loads((output / "entry_score_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selection_data_split"] == "training_only"
    assert all(task_id.startswith("training_") for task_id in metadata["task_ids"])


def test_readonly_evaluation_cannot_modify_support_files_if_present():
    output = Path("outputs/output_aware_sparse_selection")
    registry_path = output / "support_registry.csv"
    if not registry_path.is_file():
        pytest.skip("campaign support registry not generated yet")
    import pandas as pd

    registry = pd.read_csv(registry_path)
    completed = registry[registry["status"] == "completed"]
    paths = [output / path for path in completed["support_file"].head(10)]
    before = {path: _sha256(path) for path in paths}
    for path in paths:
        support = load_support(output, str(path.relative_to(output)))
        assert support.dtype == bool
    after = {path: _sha256(path) for path in paths}
    assert before == after


def test_existing_sparse_baseline_arrays_remain_byte_identical():
    original = np.load("outputs/sparse_error_precision_study/matrix_original.npy")
    sparse = np.load("outputs/sparse_error_precision_study/matrix_sparse_exact.npy")
    quantized = np.load("outputs/sparse_integrated_chain/matrix_quantized.npy")
    assert stable_array_fingerprint(original) == (
        "b158d34b86b778f0c290519ca98985345107012e225798a4cfc7fbf9178df7f9"
    )
    assert stable_array_fingerprint(sparse) == (
        "c6e29a98365f6e79e50bac5551c646e1178a4c898cb7ef47a73294a0d80ea88c"
    )
    assert stable_array_fingerprint(quantized) == (
        "26159050694e76abc32692332daba94e9cd5e22d958a242236b4d57509aeab21"
    )


def test_manuscript_packages_and_prior_outputs_match_campaign_snapshot_if_present():
    snapshot_path = Path(
        "outputs/output_aware_sparse_selection/protected_path_snapshot.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("campaign protected-path snapshot not generated yet")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))["files"]
    changed = [
        path
        for path, expected in snapshot.items()
        if not Path(path).is_file() or _sha256(Path(path)) != expected
    ]
    assert changed == []
