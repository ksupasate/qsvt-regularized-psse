"""Phase 3: weighted singular-support polynomial fitting and the fit-mode comparison."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr
from robust_qsvt_se.qsvt.weighted_singular_support import compute_singular_support


def test_support_weights_sum_to_one(tiny_context) -> None:
    support = compute_singular_support(tiny_context.H, tiny_context.r, alpha=1e-2)
    for mode in psr.WEIGHT_MODES:
        weights = psr._support_weights(support, mode)
        assert np.isclose(float(np.sum(weights)), 1.0)
        assert np.all(weights >= 0.0)


def test_zero_residual_component_handled_safely() -> None:
    # residual orthogonal to one singular direction -> a zero weight, still normalized
    H = np.diag([3.0, 2.0, 1.0, 0.5])
    r = np.array([1.0, 1.0, 1.0, 0.0])
    support = compute_singular_support(H, r, alpha=1e-2)
    weights = psr._support_weights(support, "residual_energy_weight")
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert np.all(np.isfinite(weights))


def test_weighted_fit_is_bounded_on_dense_grid(tiny_context) -> None:
    state = psr.qsvt_target_readout(tiny_context.H, tiny_context.r, alpha=1e-2, degree=15)
    sv_A = np.asarray(state.singular_values) / float(state.singular_values[0])
    support = compute_singular_support(tiny_context.H, tiny_context.r, alpha=1e-2)
    weights = psr._support_weights(support, "residual_energy_weight")
    coeffs, scale_C = psr._fit_weighted_odd_polynomial(
        singular_values_A=sv_A,
        support_weight=weights,
        alpha_norm=float(state.alpha_norm),
        domain_min=psr._domain_min(sv_A),
        degree=15,
        fit_mode="weighted_singular_support_fit",
    )
    dense = np.linspace(-1.0, 1.0, 4097)
    assert float(np.max(np.abs(np.polynomial.Polynomial(coeffs)(dense)))) <= 1.0 + 1e-3
    assert scale_C > 0


def test_fit_modes_comparison_columns_and_gate_confirmation(
    tiny_context, fake_realize, monkeypatch
) -> None:
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    detail, comparison = psr.evaluate_fit_modes(
        tiny_context, alpha=1e-2, degree=15, seed=1, gate_confirm_top=2
    )
    assert detail and comparison
    assert all(set(psr.WEIGHTED_FIT_COLUMNS) == set(row) for row in detail)
    assert all(set(psr.FIT_MODE_COMPARISON_COLUMNS) == set(row) for row in comparison)
    # at most gate_confirm_top fits are gate-confirmed
    confirmed = [r for r in comparison if r["gate_run_status"] == "computed"]
    assert 0 < len(confirmed) <= 2
    others = [r for r in comparison if r["boundedness_ok"] and r["gate_run_status"] != "computed"]
    assert all(r["gate_run_status"] == "polynomial_level_only" for r in others)


def test_residual_energy_weight_concentrates_on_high_residual(
    tiny_context, fake_realize, monkeypatch
) -> None:
    # the residual-energy weighting is the lever: the singular direction with the largest
    # residual projection receives the largest fit weight (this is what reduces the update error
    # on ill-conditioned subproblems; the outcome itself is validated by the real artifact).
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    detail, _comparison = psr.evaluate_fit_modes(tiny_context, alpha=1e-2, degree=21, seed=1)
    rows = [
        r
        for r in detail
        if r["fit_mode"] == "weighted_singular_support_fit"
        and r["weight_mode"] == "residual_energy_weight"
    ]
    assert rows
    max_residual_row = max(rows, key=lambda r: abs(r["residual_component"]))
    max_weight_row = max(rows, key=lambda r: r["weight"])
    assert max_residual_row["singular_value_index"] == max_weight_row["singular_value_index"]
    assert np.isclose(sum(r["weight"] for r in rows), 1.0)


def test_unsafe_fit_is_not_eligible(monkeypatch) -> None:
    # force an unbounded polynomial -> not boundedness_ok -> diagnostic-only, never eligible
    import robust_qsvt_se.paper.phase_synthesis_refinement as mod

    rng = np.random.default_rng(0)
    H = np.diag([3.0, 2.0, 1.0, 0.4]) + 0.01 * rng.standard_normal((4, 4))
    r = rng.standard_normal(4)
    ctx = mod._RefinementContext({"case": "c", "subproblem_id": "s", "subproblem_type": "t"}, H, r)

    def unbounded_fit(**kwargs):
        full = np.zeros(int(kwargs["degree"]) + 1)
        full[1] = 5.0  # |p(x)| up to 5 -> unbounded
        return full, 1.0

    monkeypatch.setattr(mod, "_fit_weighted_odd_polynomial", unbounded_fit)
    monkeypatch.setattr(mod, "realize_config_gate", lambda *a, **k: {"available": False})
    _detail, comparison = mod.evaluate_fit_modes(ctx, alpha=1e-2, degree=15, seed=1)
    assert comparison
    assert all(not r["eligible_for_best_config"] for r in comparison)
    assert all(r["status"] == "diagnostic_only_unsafe_fit" for r in comparison)
