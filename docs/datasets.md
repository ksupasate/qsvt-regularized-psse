# Dataset Sources

This project distinguishes three data categories.

## Synthetic Generated Data

Synthetic configs use `system.mode: synthetic_linearized`. They construct a
weighted matrix with controlled singular values from a deterministic random
seed. No power-system case is loaded.

Examples:

```bash
python -m robust_qsvt_se.cli.run_experiment --config configs/ieee14_spectral_smoke.yaml
python -m robust_qsvt_se.cli.run_experiment --config configs/ieee14_synthetic_sweeps.yaml
```

## Built-In IEEE14 Fixture Data

The built-in fixture path uses case data stored in:

```text
src/robust_qsvt_se/data/cases.py
```

These fixtures support the original small DC, AC-linearized, iterative AC,
bad-data, and robust-baseline development runs. They are useful for fast
regression testing, but they are not external benchmark datasets.

## Real External IEEE Benchmark Cases

Real-case configs use `system.case_source: pypower`. PYPOWER ships
MATPOWER-compatible case fixtures as Python package data. No network download is
performed at experiment runtime; reproducibility comes from the pinned Python
environment and the resolved config saved in each output directory.

Runtime dependency:

```bash
pip install -e ".[dev]"
```

The package dependency is declared in `pyproject.toml`:

```text
pypower>=5.1.19
```

Supported real external cases:

| Config case name | PYPOWER module |
|---|---|
| `ieee14` | `pypower.case14.case14` |
| `ieee30` | `pypower.case30.case30` |
| `ieee57` | `pypower.case57.case57` |
| `ieee118` | `pypower.case118.case118` |
| `ieee300` | `pypower.case300.case300` |

The loader is implemented in:

```text
src/robust_qsvt_se/data/real_cases.py
```

The real-case benchmark configs are:

```text
configs/real_ieee14.yaml
configs/real_ieee30.yaml
configs/real_ieee57.yaml
configs/real_ieee118.yaml
configs/real_ieee300.yaml
```

Run commands:

```bash
python -m robust_qsvt_se.experiments.run_benchmark --config configs/real_ieee14.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/real_ieee30.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/real_ieee57.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/real_ieee118.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/real_ieee300.yaml
```

Outputs are stored separately from synthetic and built-in fixture outputs:

```text
outputs/real_ieee14/
outputs/real_ieee30/
outputs/real_ieee57/
outputs/real_ieee118/
outputs/real_ieee300/
```

The paper-level seed-expanded configs use the same command paths and write to:

```text
outputs/real_ieee14_seed10/
outputs/real_ieee30_seed10/
outputs/real_ieee57_seed10/
outputs/real_ieee118_seed10/
outputs/real_ieee300_seed10/
```

Nonlinear AC Gauss-Newton benchmark configs cover PYPOWER IEEE14, IEEE30,
IEEE57, IEEE118, and IEEE300 with 10 seeds:

```bash
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee14_seed10.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee30_seed10.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee57_seed10.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee118_seed10.yaml
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee300_seed10.yaml
```

Those outputs are stored under `outputs/nonlinear_ac_ieee*_seed10/`. The
completed IEEE300 nonlinear output is
`outputs/nonlinear_ac_ieee300_seed10/`, with `checkpoint_state.json` reporting
`status: complete`, 90 completed trials, and 0 failed trials.

An earlier full IEEE300 attempt was stopped when it was projected to take many
hours. The resumed full seed10 run later completed successfully. The
reduced-runtime config remains available as historical/diagnostic output only:

```bash
python -m robust_qsvt_se.experiments.run_benchmark --config configs/nonlinear_ac_ieee300_reduced_runtime.yaml
```

Its output is stored under:

```text
outputs/nonlinear_ac_ieee300_reduced_runtime/
```

The reduced IEEE300 output is useful for implementation validation and scaling
diagnostics, but it is not the final paper-level IEEE300 nonlinear evidence.

Long IEEE300 nonlinear runs now support checkpoint/resume. A full run writes
trial-level records to:

```text
outputs/nonlinear_ac_ieee300_seed10/trial_results.jsonl
outputs/nonlinear_ac_ieee300_seed10/checkpoint_state.json
outputs/nonlinear_ac_ieee300_seed10/progress.log
```

Use `--resume` when running or auditing the full config manually. See
`docs/ieee300_run_guide.md` for `tmux`, `nohup`, and CPU-thread commands.

Each real-case output records `dataset_source: pypower`,
`dataset_source_detail`, `external_case: true`, the case name, seed, scenario
parameters, estimator names, RMSE, residuals, condition number, runtime, failure
status, and QSVT resource proxy rows when enabled.

## Modeling Limitations

The real-case experiments are still research benchmarks, not production grid
state estimation. They use PYPOWER case fixtures to build AC-linearized
measurement systems and controlled perturbation scenarios. They do not download
private PMU/SCADA data, do not implement bad-data detection, and do not execute
hardware QSVT. The QSVT method remains a classical spectral-filter simulator
with transparent resource-proxy diagnostics for large IEEE cases. The
PennyLane/Qiskit QSVT circuit demos use weighted Jacobians derived from these
PYPOWER benchmark cases. Full matrix circuit construction is attempted for
small feasible cases such as IEEE14/30; IEEE118/300 are documented as
resource-estimate plus deterministic submatrix demonstrations, not field
PMU/SCADA data and not full hardware-scale QSVT execution.

## Audit Note

The clean-room audit confirms that the final raw benchmark evidence uses
PYPOWER/MATPOWER-compatible IEEE cases with synthetic PMU/SCADA-like
perturbations. It does not use real utility PMU/SCADA field data.

Final seed-expanded outputs remain in:

```text
outputs/real_ieee*_seed10/
outputs/nonlinear_ac_ieee*_seed10/
```

Earlier non-seed10 real-case outputs were archived under:

```text
outputs/historical/legacy_real_cases/
```
