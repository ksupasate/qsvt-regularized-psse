from __future__ import annotations

from robust_qsvt_se.paper.full_vector_readout import build_total_cost_model_rows


def _costs():
    return [
        {
            "case": "ieee14",
            "subproblem_id": "hl_00",
            "subproblem_type": "high_leverage",
            "dimension": 4,
            "success_probability": 0.25,
        }
    ]


def test_total_equals_component_sum_times_overhead() -> None:
    rows = build_total_cost_model_rows(_costs(), target_error=1.0e-2)
    full = next(
        r for r in rows if r["readout_protocol"] == "full_vector_combined_magnitude_sign_norm"
    )
    components = (
        full["basis_sampling_shots"] + full["sign_interference_shots"] + full["norm_recovery_shots"]
    )
    assert full["estimated_total_shots"] == components * full["postselection_overhead"]
    # success_probability 0.25 -> overhead ceil(1/0.25) = 4.
    assert full["postselection_overhead"] == 4


def test_full_vector_protocol_is_at_least_linear_and_not_speedup_preserving() -> None:
    rows = build_total_cost_model_rows(_costs(), target_error=1.0e-2)
    full = next(
        r for r in rows if r["readout_protocol"] == "full_vector_combined_magnitude_sign_norm"
    )
    assert "linear" in full["scaling_in_n"]
    assert full["preserves_possible_speedup"] == "no_if_full_vector_required"


def test_observable_first_is_output_sparse_alternative() -> None:
    rows = build_total_cost_model_rows(_costs(), target_error=1.0e-2)
    observable = next(r for r in rows if r["readout_protocol"] == "observable_first_readout")
    assert observable["basis_sampling_shots"] == 0
    assert observable["preserves_possible_speedup"] == "yes_speedup_preserving_alternative"
    assert "sublinear" in observable["scaling_in_n"]


def test_total_cost_increases_with_dimension() -> None:
    small = build_total_cost_model_rows(
        [
            {
                "case": "c",
                "subproblem_id": "s",
                "subproblem_type": "t",
                "dimension": 4,
                "success_probability": 1.0,
            }
        ],
        target_error=1.0e-2,
    )
    large = build_total_cost_model_rows(
        [
            {
                "case": "c",
                "subproblem_id": "s",
                "subproblem_type": "t",
                "dimension": 64,
                "success_probability": 1.0,
            }
        ],
        target_error=1.0e-2,
    )
    small_total = next(
        r["estimated_total_shots"]
        for r in small
        if r["readout_protocol"] == "full_vector_combined_magnitude_sign_norm"
    )
    large_total = next(
        r["estimated_total_shots"]
        for r in large
        if r["readout_protocol"] == "full_vector_combined_magnitude_sign_norm"
    )
    assert large_total > small_total
