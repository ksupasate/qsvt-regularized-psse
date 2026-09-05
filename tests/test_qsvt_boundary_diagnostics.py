from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.qsvt_boundary_diagnostics import (
    BOUNDARY_TYPES,
    derive_boundary,
    run_boundary_diagnostics,
)
from robust_qsvt_se.paper.tqe_revision_support_common import find_forbidden

_MISSING = {
    "multicase_csv": "does/not/exist_multicase.csv",
    "larger_qsvt_csv": "does/not/exist_larger.csv",
    "gate_coverage_csv": "does/not/exist_gate.csv",
    "precision_sweep_csv": "does/not/exist_precision.csv",
}


def test_derive_boundary_pass_and_miss_are_correct() -> None:
    passing = derive_boundary(
        {"actual_max_error": 6.0e-4, "target_tolerance": 1.0e-2, "degree": 101}
    )
    assert passing[0] == "pass"
    assert passing[1] is True

    degree_limited = derive_boundary(
        {"actual_max_error": 9.0e-2, "target_tolerance": 1.0e-2, "degree": 101}
    )
    assert degree_limited[0] == "degree_limited"
    assert degree_limited[1] is False

    tolerance_missed = derive_boundary(
        {"actual_max_error": 0.5, "target_tolerance": 1.0e-2, "degree": 5}
    )
    assert tolerance_missed[0] == "tolerance_missed"
    assert tolerance_missed[1] is False


def test_derive_boundary_phase_and_skip() -> None:
    phase = derive_boundary(
        {
            "actual_max_error": 1.0e-4,
            "target_tolerance": 1.0e-3,
            "degree": 51,
            "phase_in_scope": True,
            "phase_status": "not_attempted",
        }
    )
    assert phase[0] == "phase_unavailable"
    assert phase[1] is True

    gate_pass = derive_boundary(
        {
            "actual_max_error": None,
            "evaluation_stage": "gate_level",
            "circuit_status_raw": "completed",
        }
    )
    assert gate_pass[0] == "pass"

    skipped = derive_boundary(
        {
            "actual_max_error": None,
            "evaluation_stage": "selected_block_polynomial",
            "run_status_raw": "ValueError: requested block 64x64 exceeds matrix shape",
        }
    )
    assert skipped[0] == "skipped"


def test_all_derived_types_are_in_controlled_vocabulary() -> None:
    samples = [
        {"actual_max_error": 1e-4, "target_tolerance": 1e-2, "degree": 101},
        {"actual_max_error": 1.0, "target_tolerance": 1e-2, "degree": 101},
        {"actual_max_error": 1.0, "target_tolerance": 1e-2, "degree": 5},
        {
            "actual_max_error": 1e-4,
            "target_tolerance": 1e-2,
            "degree": 51,
            "phase_in_scope": True,
            "phase_status": "failed",
        },
        {
            "actual_max_error": None,
            "evaluation_stage": "gate_level",
            "circuit_status_raw": "circuit_object_built",
        },
    ]
    for record in samples:
        assert derive_boundary(record)[0] in BOUNDARY_TYPES


def test_missing_sources_are_reported_not_dropped(tmp_path: Path) -> None:
    run = run_boundary_diagnostics({"output_dir": str(tmp_path), **_MISSING})
    frame = run["frame"]

    missing = frame[frame["boundary_type"] == "output_missing"]
    # One explicit row per missing source, never silently dropped.
    assert len(missing) == 4
    assert bool((missing["pass_tolerance"] == "unknown").all())
    assert set(frame["boundary_type"]) <= BOUNDARY_TYPES


def test_controlled_multicase_classification_and_outputs(tmp_path: Path) -> None:
    multicase = tmp_path / "multicase.csv"
    pd.DataFrame(
        [
            # Below reference epsilon -> pass.
            {
                "case_name": "good",
                "m": 80,
                "n": 26,
                "sigma_min": 80.0,
                "sigma_max": 2700.0,
                "kappa": 33.0,
                "alpha": 0.01,
                "degree": 101,
                "max_pointwise_error": 6.0e-4,
                "status": "ok",
            },
            # Above reference epsilon at high degree -> degree_limited.
            {
                "case_name": "hard",
                "m": 1700,
                "n": 600,
                "sigma_min": 10.0,
                "sigma_max": 220000.0,
                "kappa": 20000.0,
                "alpha": 0.01,
                "degree": 101,
                "max_pointwise_error": 9.3e-2,
                "status": "ok",
            },
        ]
    ).to_csv(multicase, index=False)

    run = run_boundary_diagnostics(
        {
            "output_dir": str(tmp_path),
            "multicase_csv": str(multicase),
            "larger_qsvt_csv": "does/not/exist.csv",
            "gate_coverage_csv": "does/not/exist.csv",
            "precision_sweep_csv": "does/not/exist.csv",
        }
    )
    frame = run["frame"]

    good = frame[frame["case_or_block"] == "good"].iloc[0]
    hard = frame[frame["case_or_block"] == "hard"].iloc[0]
    assert good["boundary_type"] == "pass"
    assert good["pass_tolerance"] is True
    assert hard["boundary_type"] == "degree_limited"
    assert hard["pass_tolerance"] is False

    # No forbidden overclaim in the manuscript-safe fields.
    safe_text = "\n".join(
        frame["safe_interpretation"].astype(str).tolist()
        + frame["likely_cause"].astype(str).tolist()
    )
    assert find_forbidden(safe_text) == []

    for name in (
        "qsvt_boundary_diagnostics.csv",
        "qsvt_boundary_diagnostics.md",
        "qsvt_boundary_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    manifest = json.loads((tmp_path / "qsvt_boundary_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tuned_to_pass"] is False
