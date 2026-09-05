from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.residual_feasible_config_search import (  # noqa: E402
    run_qsvt_residual_feasible_config_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search residual-feasible QSVT configs on policy-selected subproblems"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    )
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument(
        "--target-designs",
        nargs="+",
        default=[
            "current_global",
            "support_scaled",
            "margin_1p05",
            "margin_1p10",
            "degree_adaptive",
        ],
    )
    parser.add_argument(
        "--scale-protocols",
        nargs="+",
        default=[
            "known_C",
            "success_amplitude_proxy",
            "amplitude_estimation_proxy",
            "best_scalar_diagnostic",
        ],
    )
    parser.add_argument("--condition-threshold", type=float, default=1.0e8)
    parser.add_argument("--policy-alpha", type=float, default=1.0e-4)
    parser.add_argument("--max-selected", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument("--gain-cap-factor", type=float, default=0.5)
    parser.add_argument("--eps-rel", type=float, default=1.0e-2)
    parser.add_argument("--amplitude-max-queries", type=int, default=1000)
    parser.add_argument(
        "--diagnostic-cases",
        nargs="+",
        default=["ieee30", "ieee57"],
    )
    parser.add_argument(
        "--no-matrix-free-diagnostics",
        dest="include_matrix_free_diagnostics",
        action="store_false",
    )
    parser.set_defaults(include_matrix_free_diagnostics=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_residual_feasible_config_search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_qsvt_residual_feasible_config_search(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "target_designs": args.target_designs,
            "scale_protocols": args.scale_protocols,
            "condition_threshold": args.condition_threshold,
            "policy_alpha": args.policy_alpha,
            "max_selected": args.max_selected,
            "grid_size": args.grid_size,
            "gain_cap_factor": args.gain_cap_factor,
            "eps_rel": args.eps_rel,
            "amplitude_max_queries": args.amplitude_max_queries,
            "diagnostic_cases": args.diagnostic_cases,
            "include_matrix_free_diagnostics": args.include_matrix_free_diagnostics,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    rows = run["rows"]
    feasible = sum(1 for row in rows if row.get("residual_feasible"))
    recommended = sum(1 for row in rows if row.get("gate_validation_recommended"))
    print(f"QSVT residual-feasible config search complete: {run['output_dir']}")
    print(
        f"selected_subproblems={run['selected_subproblem_count']} rows={len(rows)} "
        f"feasible={feasible} gate_recommended={recommended}"
    )


if __name__ == "__main__":
    main()
