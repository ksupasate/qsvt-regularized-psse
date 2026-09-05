from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.engineering_audit import classify_forbidden_wording
from robust_qsvt_se.qsvt.engineering_extension_report import (
    build_engineering_extension_summary,
)


def test_claim_support_matrix_contains_required_claims_and_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_engineering_extension_summary({"output_dir": str(tmp_path / "claims")})
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "claim_support_matrix.csv")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    claims = set(frame["claim"])

    required_claims = {
        "Dense block-encoding prototype was validated on small normalized matrices.",
        "Exact QSVT-target spectral filtering matches Ridge/Tikhonov under the same alpha.",
        "Selected-alpha bounded polynomial/phase approximations were validated.",
        "QSVT resource estimates support feasibility discussion only.",
        "Shot-level readout analysis quantifies sampling cost for selected observables.",
        "Full-vector readout remains a limitation.",
        "Hardware-aware analysis is simulation/proxy only.",
        "Dense block encoding is not a scalable oracle.",
        "Multi-case resource diagnostics extend beyond IEEE14 where feasible.",
        "The extension does not demonstrate quantum speedup.",
        "The extension does not demonstrate quantum advantage.",
        "The extension does not execute full IEEE-scale QSVT on quantum hardware.",
        "The extension does not use real PMU/SCADA field data.",
        "QSVT does not numerically outperform Ridge/Tikhonov under the same alpha/filter.",
        "Selected-alpha polynomial approximation diagnostics were implemented.",
        "Degree sweep quantifies approximation error versus resource cost.",
        "Adaptive degree selection identifies whether target tolerances can be met.",
        "Optional phase synthesis is performed only if dependencies are available.",
        "Polynomial fallback is not full QSP/QSVT phase synthesis.",
        "Passing/failing 1e-3 tolerance is reported explicitly.",
        "Query count increases with polynomial degree.",
        "Approximation diagnostics support feasibility discussion only.",
        "Phase-response convention diagnostics validate the PennyLane scalar response convention.",
        "Known sanity-polynomial QSP/QSVT responses are checked before Ridge-target validation.",
        (
            "Full phase-level Ridge/Tikhonov target validation remains unresolved "
            "when reported failed."
        ),
        "Phase backend capabilities were audited.",
        "Stable polynomial candidates were tested.",
        "Bounded Ridge/Tikhonov target phase validation passed only if all gates passed.",
        "Sanity-polynomial phase response passed.",
        "Chebyshev-to-monomial conversion instability was measured.",
        "No unstable polynomial was forced into phase synthesis.",
        "No tolerance relaxation was used.",
        "No quantum speedup or hardware execution is claimed.",
        "External QSP/QSVT phase backends were audited.",
        "pyqsp/QSPPACK/PennyLane/local optimization backend availability was tested.",
        "Backend sanity regression was performed.",
        (
            "Target-level bounded Ridge/Tikhonov phase validation passed only if "
            "full-domain error <= 1e-3."
        ),
        "Actual-singular-value-only validation is not full-domain validation.",
        "No unsafe monomial candidate was forced into phase synthesis.",
        "Adaptive multicase degree search quantifies larger-case degree and query requirements.",
        "Some larger IEEE cases require higher degree than IEEE14 under the same tolerance.",
    }
    assert required_claims.issubset(claims)
    assert len(frame) >= 41
    assert {"command", "generated_at", "git_commit", "input_config"}.issubset(manifest)
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "final_engineering_summary.md").is_file()


def test_stable_phase_docs_do_not_make_unsafe_claims() -> None:
    docs = [
        "docs/QSVT_STABLE_PHASE_SYNTHESIS.md",
        "docs/QSVT_EXTERNAL_PHASE_BACKENDS.md",
        "docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md",
        "docs/QSVT_APPROXIMATION_VALIDATION.md",
        "docs/QSVT_ENGINEERING_EXTENSION.md",
    ]
    for path in docs:
        text = Path(path).read_text(encoding="utf-8")
        unsafe = [
            hit
            for hit in classify_forbidden_wording(text)
            if hit["classification"] == "unsafe_context"
        ]
        assert unsafe == []
        assert "hardware validation passed" not in text.lower()
        assert "qsvt-over-ridge superiority" not in text.lower()
