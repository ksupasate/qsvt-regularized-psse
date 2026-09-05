from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.nonlinear_logging_refresh import (  # noqa: E402
    DEFAULT_SEEDS,
    build_nonlinear_logging_refresh,
)

# Broadened default: small + medium + large controlled IEEE cases under three stress
# profiles, with the robust Huber estimator alongside Ridge / TSVD / QSVT-target.
_BROADER_CASES = ("ieee14", "ieee57", "ieee30", "ieee118")
_BROADER_ESTIMATORS = (
    "ridge_tikhonov",
    "truncated_svd",
    "huber_irls",
    "qsvt_target_classical",
)
_BROADER_STRESS = ("clean_or_noise", "missing_20_percent", "bad_data_5_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2/3 hardening: nonlinear AC per-iteration logging refresh "
        "(records per-iteration RMSE and weighted-Jacobian condition number)."
    )
    parser.add_argument("--cases", nargs="+", default=list(_BROADER_CASES))
    parser.add_argument("--estimators", nargs="+", default=list(_BROADER_ESTIMATORS))
    parser.add_argument("--stress-types", nargs="+", default=list(_BROADER_STRESS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--max-total-seconds", type=float, default=900.0)
    parser.add_argument("--output-dir", default="outputs/nonlinear_logging_refresh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_nonlinear_logging_refresh(
        {
            "cases": args.cases,
            "estimators": args.estimators,
            "stress_types": args.stress_types,
            "seeds": args.seeds,
            "alpha": args.alpha,
            "case_source": args.case_source,
            "max_total_seconds": args.max_total_seconds,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote nonlinear logging refresh to {run['output_dir']}")
    print(
        f"rows={run['n_rows']} rmse_logged={run['rmse_logged']} "
        f"kappa_logged={run['kappa_logged']} failures={len(run['failures'])} "
        f"runtime_limited={len(run['runtime_limited'])}"
    )


if __name__ == "__main__":
    main()
