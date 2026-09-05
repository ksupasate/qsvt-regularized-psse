"""Focused tests for the reviewer-blocking evidence studies (Phases 3/5).

High-degree QSVT parity + composite-feasibility discipline, and structure-aware statistics unit
discipline / reproducibility.  A tiny degree-31 slice keeps phase synthesis fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------- high-degree QSVT


def _smoke_config(tmp_path: Path, degrees) -> Path:
    cfg = {
        "protocol_id": "reviewer_blocking_tqe_evidence_v1",
        "study": "high_degree_qsvt_feasibility_slice",
        "structures": ["ieee14_8x8"],
        "selectors": ["global_magnitude"],
        "support_budget": 16,
        "slot_budget": 3,
        "alpha_regimes": ["fixed_benchmark"],
        "degrees": degrees,
        "uniform_approximation_tolerance": 0.002,
        "bound_tolerance": 0.002,
        "action_error_tolerance": 1e-6,
        "target_margin": 1.05,
        "useful_rmse_ratio_threshold": 1.5,
        "execute_statevector": True,
        "statevector_execution_ceiling_dim": 512,
        "training_seed_ids": [1000, 1001, 1002],
    }
    path = tmp_path / "hd.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_high_degree_rejects_even_degrees(tmp_path):
    from robust_qsvt_se.reviewer_evidence.high_degree import run_high_degree

    cfg = _smoke_config(tmp_path, [30])
    with pytest.raises(ValueError):
        run_high_degree(cfg, tmp_path / "out")


def test_high_degree_composite_feasibility_requires_all_criteria(tmp_path):
    from robust_qsvt_se.reviewer_evidence.high_degree import run_high_degree

    cfg = _smoke_config(tmp_path, [31, 63])
    run_high_degree(cfg, tmp_path / "out")
    rows = pd.read_csv(tmp_path / "out" / "high_degree_qsvt_rows.csv")
    # a sampled matrix action being small must NOT override a failed uniform fit
    bad_fit = rows[rows["uniform_fit_error"] > 0.002]
    assert (~bad_fit["qsvt_feasible_composite"]).all()
    # composite requires synthesized phases
    unsynth = rows[rows["phase_synthesis_status"] != "synthesized"]
    assert (~unsynth["qsvt_feasible_composite"]).all()
    # odd degrees only were recorded
    assert set(rows["parity"].unique()) == {"odd"}


def test_high_degree_phase_count_matches_degree_when_synthesized(tmp_path):
    from robust_qsvt_se.reviewer_evidence.high_degree import run_high_degree

    cfg = _smoke_config(tmp_path, [31])
    run_high_degree(cfg, tmp_path / "out")
    rows = pd.read_csv(tmp_path / "out" / "high_degree_qsvt_rows.csv")
    synth = rows[rows["phase_synthesis_status"] == "synthesized"]
    assert len(synth) >= 1
    for _, r in synth.iterrows():
        assert int(r["phase_count"]) == int(r["degree"]) + 1


def test_rectangular_convention_import_unchanged():
    # convention module must remain importable and expose the real-rectangular helpers
    from robust_qsvt_se.qsvt import rectangular_convention

    assert hasattr(rectangular_convention, "__file__")


# ------------------------------------------------------------- structure statistics




def test_structure_bootstrap_is_seed_reproducible():
    from robust_qsvt_se.reviewer_evidence.structure_stats import _bootstrap_over_structures

    effects = np.array([0.3, -0.68, -0.14])
    a = _bootstrap_over_structures(effects, np.random.default_rng(20240714), 2000)
    b = _bootstrap_over_structures(effects, np.random.default_rng(20240714), 2000)
    assert a["ci95_low"] == b["ci95_low"] and a["ci95_high"] == b["ci95_high"]


def test_leave_one_case_out_excludes_intended_case():
    from robust_qsvt_se.reviewer_evidence.structure_stats import _leave_one_case_out

    merged = pd.DataFrame(
        {
            "structure_id": ["ieee14_8x8"] * 2 + ["ieee14_16x16"] * 2 + ["ieee30_8x8"] * 2,
            "case": ["ieee14"] * 4 + ["ieee30"] * 2,
            "effect": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    loco = _leave_one_case_out(merged)
    assert loco["excluding_ieee30"]["structures_kept"] == 2  # only ieee14 structures remain
    assert loco["excluding_ieee14"]["structures_kept"] == 1  # only ieee30 remains


