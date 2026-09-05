from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.sparse_oracle_gates import (  # noqa: E402
    run_toy_sparse_oracle_gate_demo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy sparse-oracle gate demo")
    parser.add_argument("--matrix-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/toy_sparse_oracle_gate_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_toy_sparse_oracle_gate_demo(
        {
            "matrix_size": args.matrix_size,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    validation = run["validation"]
    print(f"Toy sparse-oracle gate demo complete: {run['output_dir']}")
    print(
        "validation_passed="
        f"{validation['validation_passed']} "
        f"max_value_error={validation['max_value_probability_error']:.3e}"
    )


if __name__ == "__main__":
    main()
