from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import current_command, git_commit, utc_timestamp
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import run_full_matrix_qsvt_demo
from robust_qsvt_se.qsvt.partial_observable_readout import (
    basis_probability,
    estimate_bernoulli_probability,
    estimate_overlap_from_hadamard_proxy,
    linear_functional_overlap,
    normalize_state,
    subset_probability,
)
from robust_qsvt_se.qsvt.power_observables import (
    branch_angle_difference_observable,
    component_observable,
    subset_energy_observable,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

READOUT_SCOPE_STATEMENT = (
    "This is a small partial-observable QSVT readout simulation. It does not "
    "demonstrate scalable block encoding, full IEEE-scale hardware execution, "
    "quantum speedup, or QSVT numerical superiority over Ridge/Tikhonov."
)
FULL_VECTOR_DIAGNOSTIC_CAVEAT = (
    "Full-vector diagnostics are simulator validation checks, not readout claims."
)
DEFAULT_SHOTS = [100, 1000, 10000]


def run_partial_observable_readout_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rng = np.random.default_rng(int(resolved["seed"]))

    qsvt_output_dir = _load_or_run_qsvt_output(resolved, output_dir)
    qsvt_data = _load_qsvt_output(qsvt_output_dir)

    B = np.asarray(qsvt_data["metadata"]["original_matrix_B"], dtype=np.float64)
    residual = np.asarray(qsvt_data["metadata"]["residual_vector_normalized"], dtype=np.float64)
    ridge_update = _ridge_update_from_transpose_operator(
        B,
        residual,
        alpha=float(resolved["alpha"]),
    )
    qsvt_update = qsvt_data["state_comparison"]["qsvt_state_update"].to_numpy(dtype=np.float64)

    ridge_state, ridge_norm = normalize_state(ridge_update)
    qsvt_state, qsvt_norm = normalize_state(qsvt_update)
    observables = _build_observables(
        resolved,
        dimension=qsvt_state.size,
        state_labels=list(qsvt_data["metadata"].get("selected_state_labels", [])),
    )
    observable_frame = _observable_summary_frame(
        observables=observables,
        ridge_state=ridge_state,
        qsvt_state=qsvt_state,
    )
    shot_frame = _shot_summary_frame(
        observables=observables,
        qsvt_state=qsvt_state,
        ridge_state=ridge_state,
        shots=list(resolved["shots"]),
        seed=int(resolved["seed"]),
        rng=rng,
    )
    diagnostics = _state_vector_diagnostics(
        ridge_update=ridge_update,
        qsvt_update=qsvt_update,
        ridge_state=ridge_state,
        qsvt_state=qsvt_state,
        ridge_norm=ridge_norm,
        qsvt_norm=qsvt_norm,
        qsvt_data=qsvt_data,
    )
    artifacts = _write_readout_artifacts(
        output_dir=output_dir,
        resolved=resolved,
        qsvt_output_dir=qsvt_output_dir,
        qsvt_data=qsvt_data,
        observable_frame=observable_frame,
        shot_frame=shot_frame,
        diagnostics=diagnostics,
        observables=observables,
    )
    return {
        "output_dir": output_dir,
        "qsvt_output_dir": qsvt_output_dir,
        "observable_summary": observable_frame,
        "shot_summary": shot_frame,
        "state_vector_diagnostics": diagnostics,
        "artifacts": artifacts,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "case": "ieee14",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "max_synthesis_degree": 35,
        "seed": 123,
        "shots": DEFAULT_SHOTS,
        "observable_mode": "default",
        "weak_area_indices": None,
        "component_indices": None,
        "difference_pairs": None,
        "reuse_existing_qsvt_output": None,
        "output_dir": "outputs/qsvt_partial_observable_readout_demo",
    }
    if config:
        resolved.update(config)
    if "case" in resolved and "case_name" not in (config or {}):
        resolved["case_name"] = resolved["case"]
    size = int(resolved["submatrix_size"])
    if size <= 0:
        raise ValueError("submatrix_size must be positive")
    if not _is_power_of_two(size):
        raise ValueError("submatrix_size must be a power of two for explicit QSVT readout")
    degree = int(resolved["degree"])
    if degree <= 0 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    shots = [int(value) for value in resolved["shots"]]
    if not shots or any(value <= 0 for value in shots):
        raise ValueError("shots must contain positive integers")
    resolved["submatrix_size"] = size
    resolved["alpha"] = float(resolved["alpha"])
    resolved["degree"] = degree
    resolved["max_synthesis_degree"] = int(resolved["max_synthesis_degree"])
    resolved["shots"] = shots
    resolved["component_indices"] = _optional_int_list(resolved.get("component_indices"))
    resolved["weak_area_indices"] = _optional_int_list(resolved.get("weak_area_indices"))
    resolved["difference_pairs"] = _optional_pairs(resolved.get("difference_pairs"))
    return resolved


def _load_or_run_qsvt_output(resolved: dict[str, Any], output_dir: Path) -> Path:
    reuse_path = resolved.get("reuse_existing_qsvt_output")
    if reuse_path:
        qsvt_output_dir = Path(str(reuse_path))
        _require_qsvt_output_files(qsvt_output_dir)
        return qsvt_output_dir

    qsvt_output_dir = output_dir / "matrix_level_qsvt_demo"
    run_full_matrix_qsvt_demo(
        {
            "case": resolved["case"],
            "case_name": resolved["case_name"],
            "case_source": resolved["case_source"],
            "matrix_source": resolved["matrix_source"],
            "submatrix_size": resolved["submatrix_size"],
            "alpha": resolved["alpha"],
            "degree": resolved["degree"],
            "max_synthesis_degree": resolved["max_synthesis_degree"],
            "output_dir": str(qsvt_output_dir),
            "seed": resolved["seed"],
        }
    )
    _require_qsvt_output_files(qsvt_output_dir)
    return qsvt_output_dir


def _require_qsvt_output_files(qsvt_output_dir: Path) -> None:
    required = {
        "matrix_metadata.json",
        "block_encoding_report.json",
        "qsvt_matrix_level_comparison.csv",
        "qsvt_state_solution_comparison.csv",
        "phase_angles.csv",
        "resolved_config.json",
    }
    missing = sorted(name for name in required if not (qsvt_output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"QSVT output directory {qsvt_output_dir} is missing required files: {missing}"
        )


def _load_qsvt_output(qsvt_output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((qsvt_output_dir / "matrix_metadata.json").read_text(encoding="utf-8"))
    block_report = json.loads(
        (qsvt_output_dir / "block_encoding_report.json").read_text(encoding="utf-8")
    )
    resolved_config = json.loads(
        (qsvt_output_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    matrix_comparison = pd.read_csv(qsvt_output_dir / "qsvt_matrix_level_comparison.csv")
    state_comparison = pd.read_csv(qsvt_output_dir / "qsvt_state_solution_comparison.csv")
    phase_angles = pd.read_csv(qsvt_output_dir / "phase_angles.csv")
    phase_count = len(phase_angles)
    return {
        "output_dir": qsvt_output_dir,
        "metadata": metadata,
        "block_report": block_report,
        "resolved_config": resolved_config,
        "matrix_comparison": matrix_comparison,
        "state_comparison": state_comparison,
        "phase_count": phase_count,
        "synthesized_degree": max(phase_count - 1, 0),
        "qsvt_vs_polynomial_svd_max_error": float(
            matrix_comparison["abs_error_vs_polynomial_svd"].max()
        ),
        "qsvt_vs_ridge_svd_max_error": float(matrix_comparison["abs_error_vs_ridge_svd"].max()),
        "state_error_vs_ridge_max_abs": float(state_comparison["abs_error_vs_ridge"].max()),
    }


def _ridge_update_from_transpose_operator(
    B: np.ndarray,
    residual: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    U, singular_values, Vh = np.linalg.svd(np.asarray(B, dtype=np.float64), full_matrices=False)
    return U @ (ridge_filter(singular_values, alpha=float(alpha)) * (Vh @ residual))


def _build_observables(
    resolved: dict[str, Any],
    *,
    dimension: int,
    state_labels: list[str],
) -> list[dict[str, Any]]:
    component_indices = resolved["component_indices"]
    if component_indices is None:
        component_indices = [0] if dimension == 1 else [0, 1]
    subset_indices = resolved["weak_area_indices"]
    if subset_indices is None:
        subset_indices = list(range(min(2, dimension)))
    difference_pairs = resolved["difference_pairs"]
    if difference_pairs is None:
        difference_pairs = [(0, 1)] if dimension >= 2 else []

    observables: list[dict[str, Any]] = []
    for index in component_indices:
        _validate_index(index, dimension)
        observable = component_observable(index, name=f"component_{index}_probability")
        observable["state_label"] = _label_for_index(state_labels, index)
        observables.append(observable)
    valid_subset = [_validate_index(index, dimension) for index in subset_indices]
    if valid_subset:
        subset_name = (
            "first_two_state_energy" if valid_subset == [0, 1] else "selected_subset_energy"
        )
        observable = subset_energy_observable(valid_subset, name=subset_name)
        observable["state_labels"] = [
            _label_for_index(state_labels, index) for index in valid_subset
        ]
        observables.append(observable)
    for first, second in difference_pairs:
        _validate_index(first, dimension)
        _validate_index(second, dimension)
        observable = branch_angle_difference_observable(
            first,
            second,
            dimension,
            name=f"difference_{first}_{second}_overlap",
        )
        observable["state_labels"] = [
            _label_for_index(state_labels, first),
            _label_for_index(state_labels, second),
        ]
        observables.append(observable)
    return observables


def _observable_summary_frame(
    *,
    observables: list[dict[str, Any]],
    ridge_state: np.ndarray,
    qsvt_state: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for observable in observables:
        ridge_value = _exact_observable_value(ridge_state, observable)
        qsvt_value = _exact_observable_value(qsvt_state, observable)
        absolute_error = abs(qsvt_value - ridge_value)
        rows.append(
            {
                "observable_name": observable["observable_name"],
                "observable_type": observable["observable_type"],
                "indices": json.dumps(observable.get("indices", [])),
                "ridge_exact_normalized": ridge_value,
                "qsvt_exact_normalized": qsvt_value,
                "absolute_error": absolute_error,
                "relative_error": _relative_error(absolute_error, ridge_value),
                "qsvt_minus_ridge": qsvt_value - ridge_value,
                "notes": observable.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _shot_summary_frame(
    *,
    observables: list[dict[str, Any]],
    qsvt_state: np.ndarray,
    ridge_state: np.ndarray,
    shots: list[int],
    seed: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for observable in observables:
        true_qsvt_value = _exact_observable_value(qsvt_state, observable)
        ridge_value = _exact_observable_value(ridge_state, observable)
        for shot_count in shots:
            if observable["observable_type"] in {"basis_probability", "subset_probability"}:
                shot_estimate, standard_error = estimate_bernoulli_probability(
                    true_qsvt_value,
                    shot_count,
                    rng,
                )
            else:
                overlap = _observable_overlap(qsvt_state, observable)
                shot_estimate, standard_error = estimate_overlap_from_hadamard_proxy(
                    overlap,
                    shot_count,
                    rng,
                    component=str(observable.get("component", "real")),
                )
            rows.append(
                {
                    "observable_name": observable["observable_name"],
                    "observable_type": observable["observable_type"],
                    "shots": int(shot_count),
                    "seed": int(seed),
                    "true_qsvt_value": true_qsvt_value,
                    "shot_estimate": shot_estimate,
                    "standard_error": standard_error,
                    "absolute_sampling_error": abs(shot_estimate - true_qsvt_value),
                    "ridge_reference_value": ridge_value,
                    "absolute_error_vs_ridge": abs(shot_estimate - ridge_value),
                    "notes": _shot_notes(observable),
                }
            )
    return pd.DataFrame(rows)


def _exact_observable_value(state: np.ndarray, observable: dict[str, Any]) -> float:
    observable_type = str(observable["observable_type"])
    if observable_type == "basis_probability":
        return basis_probability(state, int(observable["indices"][0]))
    if observable_type == "subset_probability":
        return subset_probability(state, list(observable["indices"]))
    if observable_type == "linear_overlap_real":
        overlap = _observable_overlap(state, observable)
        return float(np.real(overlap))
    raise ValueError(f"unsupported observable type: {observable_type}")


def _observable_overlap(state: np.ndarray, observable: dict[str, Any]) -> complex:
    coeffs = np.asarray(observable.get("coefficients"), dtype=np.complex128)
    return linear_functional_overlap(state, coeffs)


def _state_vector_diagnostics(
    *,
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
    ridge_state: np.ndarray,
    qsvt_state: np.ndarray,
    ridge_norm: float,
    qsvt_norm: float,
    qsvt_data: dict[str, Any],
) -> dict[str, Any]:
    normalized_l2_error = float(np.linalg.norm(qsvt_state - ridge_state))
    full_vector_l2_error = float(np.linalg.norm(qsvt_update - ridge_update))
    return {
        "ridge_update_norm": float(ridge_norm),
        "qsvt_state_norm_before_normalization": float(qsvt_norm),
        "qsvt_state_norm_after_normalization": float(np.linalg.norm(qsvt_state)),
        "normalized_state_l2_error_vs_ridge": normalized_l2_error,
        "full_vector_l2_error_vs_ridge": full_vector_l2_error,
        "max_abs_component_error": float(np.max(np.abs(qsvt_update - ridge_update))),
        "matrix_level_qsvt_vs_polynomial_svd_max_error": qsvt_data[
            "qsvt_vs_polynomial_svd_max_error"
        ],
        "matrix_level_qsvt_vs_ridge_svd_max_error": qsvt_data["qsvt_vs_ridge_svd_max_error"],
        "state_update_max_abs_error_vs_ridge_svd": qsvt_data["state_error_vs_ridge_max_abs"],
        "full_vector_diagnostic_caveat": FULL_VECTOR_DIAGNOSTIC_CAVEAT,
        "norm_readout_caveat": (
            "The normalized state direction is sampled here; the update norm is included "
            "only because this is a classical simulator comparison."
        ),
    }


def _write_readout_artifacts(
    *,
    output_dir: Path,
    resolved: dict[str, Any],
    qsvt_output_dir: Path,
    qsvt_data: dict[str, Any],
    observable_frame: pd.DataFrame,
    shot_frame: pd.DataFrame,
    diagnostics: dict[str, Any],
    observables: list[dict[str, Any]],
) -> dict[str, Path]:
    observable_path = output_dir / "observable_summary.csv"
    shot_path = output_dir / "shot_readout_summary.csv"
    diagnostics_path = output_dir / "state_vector_diagnostics.json"
    summary_path = output_dir / "qsvt_readout_summary.md"
    manifest_path = output_dir / "manifest.json"

    observable_frame.to_csv(observable_path, index=False)
    shot_frame.to_csv(shot_path, index=False)
    write_json(diagnostics_path, diagnostics)
    summary_path.write_text(
        _summary_markdown(
            resolved=resolved,
            qsvt_data=qsvt_data,
            diagnostics=diagnostics,
            observable_frame=observable_frame,
            shot_frame=shot_frame,
            observables=observables,
        ),
        encoding="utf-8",
    )
    artifacts = {
        "observable_summary": observable_path,
        "shot_readout_summary": shot_path,
        "state_vector_diagnostics": diagnostics_path,
        "qsvt_readout_summary": summary_path,
        "matrix_level_qsvt_output_dir": qsvt_output_dir,
    }
    manifest = _manifest(
        resolved=resolved,
        qsvt_data=qsvt_data,
        artifacts=artifacts,
        diagnostics=diagnostics,
    )
    write_json(manifest_path, manifest)
    artifacts["manifest"] = manifest_path
    return artifacts


def _manifest(
    *,
    resolved: dict[str, Any],
    qsvt_data: dict[str, Any],
    artifacts: dict[str, Path],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    metadata = qsvt_data["metadata"]
    block_report = qsvt_data["block_report"]
    return {
        "generated_at": utc_timestamp(),
        "command": current_command(),
        "command_line_arguments": _json_safe(resolved),
        "git_commit": git_commit(),
        "case_name": resolved["case_name"],
        "matrix_source": metadata["matrix_source"],
        "submatrix_size": int(resolved["submatrix_size"]),
        "alpha": float(resolved["alpha"]),
        "requested_degree": int(resolved["degree"]),
        "synthesized_degree": int(qsvt_data["synthesized_degree"]),
        "beta": float(metadata["normalization_factor_beta"]),
        "number_of_phases": int(qsvt_data["phase_count"]),
        "block_encoding_unitarity_error": float(block_report["unitarity_error"]),
        "top_left_block_reconstruction_error": float(block_report["top_left_block_error"]),
        "random_seed": int(resolved["seed"]),
        "shot_counts": [int(value) for value in resolved["shots"]],
        "files_generated": {key: str(path) for key, path in artifacts.items()},
        "state_vector_diagnostics": diagnostics,
        "known_limitations": [
            READOUT_SCOPE_STATEMENT,
            FULL_VECTOR_DIAGNOSTIC_CAVEAT,
            "The overlap readout uses a Hadamard-test-style sampling proxy in a simulator.",
            "The update norm is not inferred from computational-basis samples.",
        ],
    }


def _summary_markdown(
    *,
    resolved: dict[str, Any],
    qsvt_data: dict[str, Any],
    diagnostics: dict[str, Any],
    observable_frame: pd.DataFrame,
    shot_frame: pd.DataFrame,
    observables: list[dict[str, Any]],
) -> str:
    metadata = qsvt_data["metadata"]
    block_report = qsvt_data["block_report"]
    max_exact_error = float(observable_frame["absolute_error"].max())
    max_sampling_error = float(shot_frame["absolute_sampling_error"].max())
    observable_names = ", ".join(str(row["observable_name"]) for row in observables)
    return "\n".join(
        [
            "# Partial-Observable QSVT Readout Demo",
            "",
            "## Purpose",
            (
                "This experiment evaluates selected normalized-state observables from the "
                "small explicit block-encoded QSVT state-estimation update."
            ),
            "",
            "## Matrix and QSVT Setup",
            f"- Matrix source: {metadata['matrix_source']} ({metadata['matrix_orientation']}).",
            f"- Matrix size: {metadata['matrix_size'][0]} x {metadata['matrix_size'][1]}.",
            f"- Beta: {metadata['normalization_factor_beta']:.17g}.",
            f"- Alpha: {resolved['alpha']:.17g}.",
            (
                f"- QSP/QSVT degree: {qsvt_data['synthesized_degree']} "
                f"(requested {resolved['degree']}); phases: {qsvt_data['phase_count']}."
            ),
            f"- Block-encoding unitarity error: {block_report['unitarity_error']:.17g}.",
            "",
            "## Observables Evaluated",
            observable_names,
            "",
            "## Exact Observable Comparison",
            (
                "Exact normalized-state observables are compared between the QSVT simulator "
                "state and the Ridge/Tikhonov SVD reference."
            ),
            f"- Maximum exact observable absolute error vs Ridge: {max_exact_error:.6g}.",
            (
                "- Normalized state L2 diagnostic error vs Ridge: "
                f"{diagnostics['normalized_state_l2_error_vs_ridge']:.6g}."
            ),
            "",
            "## Shot-Based Readout Behavior",
            (
                f"Shot counts: {', '.join(str(value) for value in resolved['shots'])}. "
                f"Maximum absolute sampling error vs exact QSVT value: {max_sampling_error:.6g}."
            ),
            "",
            "## Limitations",
            f"- {READOUT_SCOPE_STATEMENT}",
            f"- {FULL_VECTOR_DIAGNOSTIC_CAVEAT}",
            "- The update norm is not estimated from the simulated readout samples.",
            "- The linear-overlap row assumes the simulator's fixed real phase convention.",
            "",
            "## Safe Manuscript Wording",
            (
                "We include a small partial-observable readout simulation for the explicit "
                "block-encoded QSVT state-estimation update. The study evaluates selected "
                "normalized component probabilities, subset energies, and a signed "
                "difference-overlap proxy against the Ridge/Tikhonov SVD reference, with "
                "simple shot-based sampling models for the selected observables only."
            ),
            "",
        ]
    )


def _shot_notes(observable: dict[str, Any]) -> str:
    if observable["observable_type"] in {"basis_probability", "subset_probability"}:
        return "Bernoulli computational-basis probability sampling proxy"
    return "Hadamard-test-style real-overlap sampling proxy"


def _relative_error(absolute_error: float, reference: float) -> float:
    if abs(reference) <= 1.0e-15:
        return 0.0 if absolute_error <= 1.0e-15 else float("inf")
    return float(absolute_error / abs(reference))


def _label_for_index(state_labels: list[str], index: int) -> str:
    if 0 <= int(index) < len(state_labels):
        return state_labels[int(index)]
    return f"state_{int(index)}"


def _validate_index(index: int, dimension: int) -> int:
    selected = int(index)
    if selected < 0 or selected >= int(dimension):
        raise IndexError("observable index out of range for selected QSVT state")
    return selected


def _optional_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def _optional_pairs(values: Any) -> list[tuple[int, int]] | None:
    if values is None:
        return None
    pairs = []
    for value in values:
        if isinstance(value, str):
            left, right = value.split(":", maxsplit=1)
            pairs.append((int(left), int(right)))
        else:
            left, right = value
            pairs.append((int(left), int(right)))
    return pairs


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run partial-observable QSVT readout demo")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_partial_observable_readout_demo")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--observable-mode", default="default")
    parser.add_argument("--weak-area-indices", nargs="*", type=int, default=None)
    parser.add_argument("--component-indices", nargs="*", type=int, default=None)
    parser.add_argument("--difference-pairs", nargs="*", default=None)
    parser.add_argument("--reuse-existing-qsvt-output", default=None)
    args = parser.parse_args(argv)
    run = run_partial_observable_readout_demo(
        {
            "case": args.case,
            "case_name": args.case,
            "case_source": args.case_source,
            "matrix_source": args.matrix_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "observable_mode": args.observable_mode,
            "weak_area_indices": args.weak_area_indices,
            "component_indices": args.component_indices,
            "difference_pairs": args.difference_pairs,
            "reuse_existing_qsvt_output": args.reuse_existing_qsvt_output,
        }
    )
    observable_frame = run["observable_summary"]
    shot_frame = run["shot_summary"]
    print(f"Partial-observable QSVT readout demo complete: {run['output_dir']}")
    print(
        "observables="
        f"{len(observable_frame)} max_exact_error="
        f"{observable_frame['absolute_error'].max():.3e} max_sampling_error="
        f"{shot_frame['absolute_sampling_error'].max():.3e}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
