import numpy as np
import pytest

from robust_qsvt_se.paper.tqe_revision_core import (
    estimate_integrated_counts,
    exact_integrated_readout_distribution,
    sample_integrated_readout,
)


@pytest.mark.parametrize(
    ("p_succ", "overlap"),
    [(1.0, 0.2), (1e-8, 0.0), (0.5, 0.0), (0.6, 0.3), (0.6, -0.3), (0.2, 0.0)],
)
def test_complete_joint_readout_equations(p_succ, overlap):
    probabilities = exact_integrated_readout_distribution(
        postselection_probability=p_succ, signed_overlap=overlap
    )
    scale = 10_000_000
    counts = {key: round(value * scale) for key, value in probabilities.items()}
    estimate = estimate_integrated_counts(counts, physical_scale=1.0)
    assert estimate["estimated_postselection_probability"] == pytest.approx(p_succ, abs=1e-6)
    assert estimate["signed_overlap_estimate"] == pytest.approx(overlap, abs=1e-6)


def test_readout_sampling_is_seed_deterministic_and_standard_error_calibrated():
    kwargs = dict(
        postselection_probability=0.4,
        signed_overlap=-0.2,
        physical_scale=2.0,
        shots=50_000,
        seed=9,
    )
    assert sample_integrated_readout(**kwargs) == sample_integrated_readout(**kwargs)
    samples = [sample_integrated_readout(**{**kwargs, "seed": seed}) for seed in range(40)]
    empirical = np.std([row["selected_output_estimate"] for row in samples], ddof=1)
    analytic = np.mean([row["selected_output_standard_error"] for row in samples])
    assert 0.5 < empirical / analytic < 1.5
