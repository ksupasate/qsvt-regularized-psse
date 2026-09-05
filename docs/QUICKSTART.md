# Quick Start

## Full artifact environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate qsvt-se
python scripts/run_smoke_test.py
python scripts/validate_reproduction.py
```

The environment contains the classical numerical stack, test tools, PennyLane,
Qiskit/Aer, and phase-synthesis dependencies. No GPU is required.

## Pip-only alternative

Use CPython 3.12 for the closest match to the recorded environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/run_smoke_test.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

`requirements.txt` pins the direct publication dependencies. The conda file
also pins the Python interpreter. Platform-specific BLAS/LAPACK and transitive
wheel builds may differ, so compare numerical outputs with declared tolerances,
not byte-for-byte floating-point equality.

## What the smoke command does

The default smoke suite runs five small configurations:

| Name | Model | Config |
|---|---|---|
| `synthetic` | Controlled weighted matrix | `configs/ieee14_spectral_smoke.yaml` |
| `dc_linearized` | Built-in IEEE14 DC model | `configs/ieee14_dc_smoke.yaml` |
| `ac_linearized` | Built-in IEEE14 AC Jacobian | `configs/ieee14_ac_smoke.yaml` |
| `iterative_ac` | Built-in IEEE14 iterative AC update | `configs/ieee14_ac_iterative_smoke.yaml` |
| `robust_bad_data` | Generated bad data and robust estimators | `configs/ieee14_robust_bad_data_smoke.yaml` |

Expected runtime is seconds to a few minutes on a laptop-class CPU. The command
prints measured per-case and total runtime, so the recorded value is auditable
on each machine.

Expected output root:

```text
outputs/examples/smoke_test/
```

Expected files within each applicable run include:

- `config_resolved.yaml`: complete resolved configuration and runtime metadata;
- `metrics.csv` or sweep `aggregate_metrics.csv` / `summary_metrics.csv`;
- `estimator_results.json` or `trial_results.json`;
- `singular_values.csv`;
- `qsvt_resource_estimates.csv` when enabled;
- `run.log`.

The wrapper returns a nonzero exit status if any smoke case fails. For a single
minimal check:

```bash
python scripts/run_smoke_test.py --minimal
```

## Run the minimal example

```bash
python scripts/run_experiment.py \
  --config examples/minimal_run/config.yaml \
  --output-dir outputs/examples/minimal_run
python scripts/validate_outputs.py
```

The general validator reads the publication experiment manifest; it is not
limited to the minimal example.

## Preserve frozen evidence

Use `scripts/run_experiment.py` for researcher reruns. It redirects output to
`outputs/generated/` unless an explicit destination is supplied. Do not invoke
a package module directly with a canonical config unless you intend to write to
that config's declared evidence directory.

The artifact validator performs derived table/figure regeneration in a
temporary directory. It does not modify canonical experiment outputs.

## Next steps

- Experiment families and commands: [README](../README.md)
- Environment and determinism: [Reproducibility](REPRODUCIBILITY.md)
- Equations and estimator roles: [Experiment Model](EXPERIMENT_MODEL.md)
- Generated data semantics: [Measurement Model](MEASUREMENT_MODEL.md)
- Interpretation limits: [Claim Scope](CLAIM_SCOPE.md)
- Common failures: [Troubleshooting](TROUBLESHOOTING.md)
