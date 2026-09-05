"""Phase 2: phase-synthesis solver variant comparison and selection."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def test_supported_and_unsupported_variants_recorded(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0))
    rows = psr.compare_phase_solver_variants(tiny_context, alpha=0.1, degree=7, seed=1)
    variants = {r["phase_solver_variant"]: r for r in rows}
    # every supported and unsupported variant appears
    for name, _params in psr.SUPPORTED_SOLVER_VARIANTS:
        assert variants[name]["solver_supported"] is True
    for name, _reason in psr.UNSUPPORTED_SOLVER_VARIANTS:
        assert variants[name]["solver_supported"] is False
        assert variants[name]["phase_synthesis_status"] == "unsupported_solver_variant"
        assert variants[name]["gate_run_status"] == "not_run"


def test_supported_variant_runs_gate_and_records_runtime(
    tiny_context, fake_gate, monkeypatch
) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0))
    rows = psr.compare_phase_solver_variants(tiny_context, alpha=0.1, degree=7, seed=1)
    default = next(r for r in rows if r["phase_solver_variant"] == "default")
    assert default["gate_run_status"] == "computed"
    assert np.isfinite(default["gate_error_vs_ridge"])
    assert np.isfinite(default["runtime_seconds"])
    assert set(psr.SOLVER_VARIANT_COLUMNS) == set(default)


def test_selection_only_from_valid_candidates(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0))
    rows = psr.compare_phase_solver_variants(tiny_context, alpha=0.1, degree=7, seed=1)
    selected = psr.select_best_solver_variant(rows)
    pair = (tiny_context.entry["case"], tiny_context.entry["subproblem_type"])
    if pair in selected:
        chosen = selected[pair]
        assert chosen["solver_supported"] is True
        assert chosen["selected_variant_candidate"] is True
        assert chosen["boundedness_ok"] is True


def test_phase_failure_not_treated_as_readout_failure(tiny_context, fake_gate, monkeypatch) -> None:
    # gate unavailable -> the variant is not a selection candidate, but it is not a readout failure
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0, available=False))
    rows = psr.compare_phase_solver_variants(tiny_context, alpha=0.1, degree=7, seed=1)
    default = next(r for r in rows if r["phase_solver_variant"] == "default")
    assert default["gate_run_status"] == "failed"
    assert default["selected_variant_candidate"] is False
    selected = psr.select_best_solver_variant(rows)
    pair = (tiny_context.entry["case"], tiny_context.entry["subproblem_type"])
    assert pair not in selected


def test_variant_summary_markdown(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0))
    rows = psr.compare_phase_solver_variants(tiny_context, alpha=0.1, degree=7, seed=1)
    md = psr.phase_solver_variant_summary_markdown(rows, psr.select_best_solver_variant(rows))
    assert "Supported" in md and "Unsupported" in md
    assert "outperform" not in md.lower()
