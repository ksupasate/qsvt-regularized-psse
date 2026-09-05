from __future__ import annotations

import pandas as pd

from robust_qsvt_se.paper.qsvt_evidence_tiering import (
    MAIN_TIER,
    NormalizedEvidence,
    classify_tier,
    run_qsvt_evidence_tiering,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in


def _evidence(**overrides) -> NormalizedEvidence:
    base = dict(
        source="src",
        case="ieee14",
        subproblem="4x4",
        alpha=1e-2,
        degree=31,
        phase_attempted=True,
        phase_synthesized=True,
        bounded_ok=True,
        uniform_admissible=True,
        recommended=True,
        limitation="",
    )
    base.update(overrides)
    return NormalizedEvidence(**base)


def test_classify_main_tier():
    assert classify_tier(_evidence()) == MAIN_TIER


def test_classify_phase_unavailable():
    assert classify_tier(_evidence(phase_synthesized=False, limitation="")) == "phase_unavailable"


def test_classify_failed_boundedness():
    assert classify_tier(_evidence(bounded_ok=False)) == "failed_boundedness"
    assert classify_tier(_evidence(limitation="overshoot detected")) == "failed_boundedness"


def test_classify_failed_parity():
    assert classify_tier(_evidence(limitation="parity violation")) == "failed_parity"


def test_classify_degree_limited():
    assert classify_tier(_evidence(limitation="degree-limited reconstruction")) == "degree_limited"


def test_classify_spectrum_point_only_when_admissible_without_phase():
    # Uniform-admissible grid evidence but phase never attempted -> appendix.
    evidence = _evidence(phase_attempted=False, phase_synthesized=False, limitation="")
    assert classify_tier(evidence) == "spectrum_point_only"


def test_classify_failed_tolerance():
    evidence = _evidence(
        phase_synthesized=False,
        phase_attempted=False,
        uniform_admissible=False,
        recommended=False,
        limitation="residual infeasible",
    )
    assert classify_tier(evidence) == "failed_tolerance"


def test_only_main_tier_is_recommended():
    assert classify_tier(_evidence()) == MAIN_TIER
    assert classify_tier(_evidence(recommended=False)) == "phase_synthesized"


def test_run_tiering_routes_pass_to_main_and_degree_limited_to_appendix(tmp_path):
    demo_csv = tmp_path / "demo_summary.csv"
    pd.DataFrame(
        [
            {
                "case": "ieee14",
                "block_shape": "4x4",
                "alpha": 3.6e5,
                "degree": 31,
                "phase_synthesis_status": "completed",
                "boundedness_ok": True,
                "status_label": "pass",
            },
            {
                "case": "ieee14",
                "block_shape": "8x8",
                "alpha": 1.6e3,
                "degree": 31,
                "phase_synthesis_status": "completed",
                "boundedness_ok": True,
                "status_label": "degree_limited",
            },
        ]
    ).to_csv(demo_csv, index=False)

    run = run_qsvt_evidence_tiering(
        {"output_dir": str(tmp_path / "out"), "demo_summary_csv": str(demo_csv), "sources": []}
    )
    main = run["main_paper"]
    appendix = run["appendix"]
    assert (main["subproblem"] == "4x4").any()
    assert (appendix["tier"] == "degree_limited").any()
    assert not (main["subproblem"] == "8x8").any()


def test_tiering_outputs_have_no_forbidden_wording(tmp_path):
    demo_csv = tmp_path / "demo_summary.csv"
    pd.DataFrame(
        [
            {
                "case": "ieee14",
                "block_shape": "4x4",
                "alpha": 3.6e5,
                "degree": 31,
                "phase_synthesis_status": "completed",
                "boundedness_ok": True,
                "status_label": "pass",
            }
        ]
    ).to_csv(demo_csv, index=False)
    out_dir = tmp_path / "out"
    run = run_qsvt_evidence_tiering(
        {"output_dir": str(out_dir), "demo_summary_csv": str(demo_csv), "sources": []}
    )
    readme = (run["output_dir"] / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    for tier in run["status_summary"]["tier"]:
        assert isinstance(tier, str)
