from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.ieee300_runtime_extension import (  # noqa: E402
    DEFAULT_ESTIMATORS,
    DEFAULT_SUBSETS,
    build_ieee300_runtime_extension,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8 (optional): reduced IEEE300 alpha / measurement ablation."
    )
    parser.add_argument("--case", default="ieee300")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--measurement-subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-build-seconds", type=float, default=180.0)
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/ieee300_runtime_extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_ieee300_runtime_extension(
        {
            "case": args.case,
            "alphas": args.alphas,
            "measurement_subsets": args.measurement_subsets,
            "estimators": args.estimators,
            "seeds": args.seeds,
            "max_build_seconds": args.max_build_seconds,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote IEEE300 runtime extension to {run['output_dir']}")
    print(
        f"alpha_rows={run['n_alpha_rows']} qsvt==ridge={run['qsvt_ridge_equivalent']} "
        f"runtime_limited={len(run['runtime_limited_rows'])}"
    )


if __name__ == "__main__":
    main()
