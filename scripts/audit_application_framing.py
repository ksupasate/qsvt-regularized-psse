"""Phase 7 application-utility framing audit (classical, exact).

Quantifies the difference between the two RMSE-ratio criteria that share the
preregistered 1.25 threshold, WITHOUT changing any recorded result:

  Criterion A (benchmark-anchored application usefulness, IEEE-14 final
  campaign): RMSE(Ridge at candidate alpha) / RMSE(Ridge at fixed benchmark
  alpha = 1e-4) <= 1.25, evaluated classically against ground truth.

  Criterion B (matched-lambda execution accuracy, generalized IEEE-14
  robustness / IEEE-30 / IEEE-57 rows): RMSE(QSVT at (d, lambda) vs truth) /
  RMSE(Ridge at the same lambda vs truth) <= 1.25.

For every passing generalized IEEE-30/57 row this script computes the
criterion-A context ratio the row would have had, so the manuscript can state
mechanically that those rows are execution-accuracy evidence rather than
benchmark-anchored application evidence. Matched-lambda Ridge RMSEs are
cross-checked against the recorded sweep artifacts to prove the same pipeline
is used (tolerance 1e-9 relative).

Run: .venv/bin/python scripts/audit_application_framing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"
GEN = ROOT / "outputs" / "generalized_rectangular_qsvt"

BENCHMARK_ALPHA_PHYSICAL = 1.0e-4
THRESHOLD = 1.25
APPROVED_TERM = "controlled benchmark useful-overlap criterion"


def ridge_rmse(H: np.ndarray, r: np.ndarray, dx_true: np.ndarray, alpha: float) -> float:
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    filt = sv / (sv**2 + alpha)
    dx = Vh.T @ (filt * (U.T @ r))
    return float(np.linalg.norm(dx - dx_true))


def load_case(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    system, _ = build_engineering_system(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H = np.asarray(system.H_tilde, dtype=float)
    r = np.asarray(system.r_tilde, dtype=float)
    x_true = np.asarray(system.metadata["true_state"], dtype=float)
    lin_state = np.asarray(system.metadata.get("linearization_state", x_true), dtype=float)
    return H, r, x_true - lin_state


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sweeps = {
        "ieee30": pd.read_csv(GEN / "ieee30_statevector_results.csv"),
        "ieee57": pd.read_csv(GEN / "ieee57_escalation_results.csv"),
    }
    rows = []
    for case, sweep in sweeps.items():
        H, r, dx_true = load_case(case)
        beta = float(np.linalg.svd(H, compute_uv=False)[0])
        bench_rmse = ridge_rmse(H, r, dx_true, BENCHMARK_ALPHA_PHYSICAL)
        passing = sweep[sweep["status"] == "STATEVECTOR_PASSED"]
        for _, rec in passing.iterrows():
            alpha = float(rec["alpha"])
            matched = ridge_rmse(H, r, dx_true, alpha)
            recorded = float(rec["rmse_ridge"])
            rel_gap = abs(matched - recorded) / max(recorded, 1e-300)
            if rel_gap > 1e-9:
                raise AssertionError(
                    f"{case}: recomputed matched Ridge RMSE {matched} disagrees with "
                    f"recorded {recorded} (rel gap {rel_gap:.2e}); pipelines differ"
                )
            rows.append(
                {
                    "case": case,
                    "seed": 123,
                    "degree": int(rec["degree"]),
                    "lambda": float(rec["lambda"]),
                    "alpha_physical": alpha,
                    "beta": beta,
                    "criterionB_rmse_ratio_qsvt_vs_matched_ridge": float(rec["rmse_ratio"]),
                    "criterionB_pass": float(rec["rmse_ratio"]) <= THRESHOLD,
                    "matched_ridge_rmse": matched,
                    "benchmark_ridge_rmse_alpha_1e-4": bench_rmse,
                    "criterionA_context_ratio_matched_over_benchmark": matched / bench_rmse,
                    "criterionA_context_pass": (matched / bench_rmse) <= THRESHOLD,
                    "recorded_rmse_ridge_crosscheck_rel_gap": rel_gap,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "application_framing_audit.csv", index=False)

    claim_status = {
        "threshold": THRESHOLD,
        # The first repository commit already contains the IEEE-14 frontier and
        # final result.  It proves immutability thereafter, but not independent
        # pre-result declaration for that original campaign.  The generalized
        # IEEE-30/57 campaign has a separate, verifiable pre-result record.
        "threshold_declared_before_results": None,
        "threshold_provenance": {
            "ieee14_original_campaign": (
                "not independently verifiable: the first commit already contains "
                "the frontier and final IEEE-14 result"
            ),
            "generalized_ieee30_ieee57_campaign": True,
            "not_redefined_after_baseline_commit": True,
        },
        "selection_data_independent_from_evaluation": False,
        "held_out_evaluation_used": False,
        "separate_non_oracle_held_out_side_study_exists": True,
        "ieee30_best_row_selected_from_sweep": True,
        "ieee57_best_row_selected_from_sweep": True,
        "application_rmse_uses_ground_truth": True,
        "operational_selector_demonstrated_for_headline_row": False,
        "approved_term": APPROVED_TERM,
        "prohibited_terms": [
            "operationally useful",
            "deployment-ready",
            "control-center validated",
            "application-optimal",
            "application-useful (when unqualified)",
        ],
        "remaining_limitations": [
            "The IEEE-14 lambda=1e-5 headline point was found with a ground-truth "
            "oracle diagnostic.",
            "The frozen IEEE-14 reproduction reuses the same seed-123 generated benchmark matrix.",
            "No held-out seed or held-out IEEE case is the decision basis for the "
            "headline utility claim.",
            "Separate non-oracle selectors were evaluated on untouched seeds "
            "1000-1029 but do not establish deployment readiness.",
            "IEEE-30 and IEEE-57 displayed passing rows are post-sweep best rows "
            "under the execution-accuracy criterion.",
            "Every passing IEEE-30/57 execution-accuracy row fails the "
            "benchmark-anchored context ratio.",
        ],
    }
    (OUT / "utility_claim_status.json").write_text(
        json.dumps(claim_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Utility-Framing Audit (Phase 7)",
        "",
        f"Approved term: **{APPROVED_TERM}**.",
        "",
        "## Threshold provenance",
        "",
        "- 1.25 appears in artifacts frozen at the first commit `c617774`:",
        "  `outputs/final_useful_overlap_validation/final_scientific_configuration.json`",
        "  (`application_threshold: 1.25`) and the 30-seed degree-31 utility tables",
        "  under `outputs/tqe_implementation_revision/`.",
        "- It was declared before the generalized IEEE-30/57 results in",
        "  `outputs/generalized_rectangular_qsvt/preregistered_criteria.yaml`",
        "  (`written_before_results: true`; sensitivity bands 1.10/1.50/2.00 are",
        "  reported but non-decisional), and was not redefined after those results.",
        "- For the original IEEE-14 campaign, pre-result declaration is",
        "  **independently unverifiable**: the repository's first commit already",
        "  contains both the frontier and the final result. Accordingly the JSON",
        "  field `threshold_declared_before_results` is null, not true.",
        "",
        "## Two criteria share the threshold and must never be conflated",
        "",
        "- Criterion A (IEEE-14 final campaign): candidate-lambda Ridge RMSE over",
        "  fixed benchmark (alpha=1e-4) Ridge RMSE = 0.9654 <= 1.25 (pass;",
        "  `final_application_reproduction.csv`). In the paper this is called the",
        f"  {APPROVED_TERM}.",
        "- Criterion B (generalized IEEE-14 robustness, IEEE-30, IEEE-57 rows):",
        "  QSVT ground-truth RMSE over matched-lambda Ridge ground-truth RMSE.",
        "  This is an EXECUTION-ACCURACY criterion at the tested lambda.",
        "",
        "## Criterion-A context for the passing criterion-B rows",
        "",
        frame.to_string(index=False),
        "",
        "Every passing IEEE-30/57 row FAILS the criterion-A context ratio (the",
        "tested lambdas are far more smoothing than the alpha=1e-4 benchmark on",
        "these cases), so those rows are matched-filter execution-accuracy",
        "evidence, not evidence for the controlled benchmark useful-overlap",
        "criterion. The manuscript must state both definitions and label the",
        "IEEE-30/57 rows accordingly.",
        "",
        "## Selection/evaluation separation",
        "",
        "- The IEEE-14 lambda=1e-5 candidate was located by earlier sweeps using",
        "  simulated ground-truth RMSE as an oracle diagnostic. The frozen run",
        "  reproduces the same seed-123 generated benchmark; selection data and",
        "  headline evaluation are therefore not independent.",
        "- No held-out seed or held-out case is the decision basis for the headline",
        "  criterion. A separate selector side-study uses untouched seeds 1000-1029",
        "  for GCV, L-curve, discrepancy, and held-out-row methods, but that study",
        "  does not turn the oracle-selected headline point into an operational",
        "  policy.",
        "- IEEE-30/57 sweep grids (degree x lambda) were preregistered before",
        "  execution; all failed candidates are retained in the artifacts.",
        "- The displayed IEEE-30 and IEEE-57 best passing rows are selected after",
        "  their sweeps by construction; the per-row pass/fail threshold itself was",
        "  preregistered for that generalized campaign.",
        "",
        "## Wording decision",
        "",
        f"Use `{APPROVED_TERM}`. Do not use `operationally useful`,",
        "`deployment-ready`, `control-center validated`, `application-optimal`, or",
        "an unqualified `application-useful` label.",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "utility_framing_audit.md").write_text(report, "utf-8")
    # Preserve the prior output name as a compatibility alias with identical content.
    (OUT / "application_framing_audit.md").write_text(report, "utf-8")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
