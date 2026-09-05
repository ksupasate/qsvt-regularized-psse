from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import (  # noqa: E402
    run_sparse_oracle_prototype_strengthening,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strengthen the toy sparse-oracle prototype and tighten its cost model"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="dc_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--matrix-size", type=int, default=4)
    parser.add_argument("--slots-per-row", type=int, default=2)
    parser.add_argument("--value-precision-bits", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--basis-gates", nargs="+", default=["u3", "cx"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_sparse_oracle_prototype_strengthening",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_sparse_oracle_prototype_strengthening(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "matrix_size": args.matrix_size,
            "slots_per_row": args.slots_per_row,
            "value_precision_bits": args.value_precision_bits,
            "basis_gates": args.basis_gates,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    validation = run["validation"]
    print(f"Sparse-oracle prototype strengthening complete: {run['output_dir']}")
    print(
        "overall_validation_passed="
        f"{validation['overall_validation_passed']} "
        f"precisions={[row['value_precision_bits'] for row in run['sweep_rows']]}"
    )


if __name__ == "__main__":
    main()
