# Reproducibility Protocol

## Artifact identity

- Released artifact version: `1.0.0` (`VERSION`, `pyproject.toml`,
  `CITATION.cff`)
- Python distribution: `qsvt-regularized-psse`
- Python import package: `robust_qsvt_se`, whose `__version__` is `0.1.0` — the
  version that generated the frozen evidence, deliberately not raised so the
  `package_version` recorded in shipped manifests stays truthful
- Initial preparation commit: `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`
- Primary machine-readable registry:
  `outputs/reproducibility_audit/experiment_manifest.json`

The initial preparation occurred in a research worktree with uncommitted
changes. The public tree was exported by explicit include/exclude rules rather
than by copying that worktree wholesale; `release_audit/` records the exact
rules, the per-file digests, and the verification results.

## Software

The verified publication environment at preparation time used macOS ARM64 and
CPython 3.12.11. Direct versions are pinned in `requirements.txt`:

| Role | Packages |
|---|---|
| Numerical | NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.3 |
| Network cases | PYPOWER 5.1.19 |
| Configuration/reporting | PyYAML 6.0.3, matplotlib 3.10.9 |
| Tests | pytest 9.0.3, Ruff 0.15.12 |
| Quantum simulation | PennyLane 0.45.0, Qiskit 2.4.1, Qiskit Aer 0.17.2 |
| Phase/polynomial support | pyqsp 0.2.0, mpmath 1.3.0, SymPy 1.14.0 |

Classical linear algebra uses NumPy SVD, least squares, and dense solves. The
least-absolute-value baseline uses `scipy.optimize.linprog(method="highs")`.
Huber IRLS is implemented in the repository and uses repeated NumPy least
squares. QSVT circuit and phase paths use the explicitly recorded PennyLane,
Qiskit/Aer, and pyqsp versions.

The audited SciPy 1.17.1 wheel exposes bundled HiGHS 1.12.0. HiGHS is not a
separate direct package in this environment; its version follows the installed
SciPy build.

The OS-provided BLAS/LAPACK and SciPy HiGHS build may differ by platform. Small
floating-point and runtime differences are therefore expected. Use test and
config tolerances; do not require byte-identical CSVs or equal wall-clock times.

## Environment creation

Preferred full environment:

```bash
conda env create -f environment.yml
conda activate qsvt-se
```

Pip alternative:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The editable install is intentional for a research artifact: commands execute
the checked-out source while exact direct third-party versions come from
`requirements.txt`.

## Hardware requirements

- Smoke suite: laptop-class 64-bit CPU, no GPU, modest memory.
- IEEE14/30/57 classical sweeps: ordinary workstation/laptop; runtime grows
  with case, seed count, and robust estimators.
- IEEE118/300 and nonlinear sweeps: workstation-class CPU and persistent shell
  recommended. IEEE300 supports checkpoint/resume and can be long-running.
- Small QSVT simulations: CPU execution; memory and circuit compilation grow
  rapidly with selected block size and degree.
- Full IEEE-scale quantum hardware is neither required nor executed.

No fixed cross-platform runtime claim is made. Every standard result records
`runtime_seconds`, and the smoke wrapper prints an end-to-end total.

## Randomness

### Main experiments

`robust_qsvt_se.utils.seed.make_rng(seed)` constructs
`numpy.random.default_rng(seed)`. A standard single run uses the integer
top-level `seed`. Sweep trials replace that value with the explicit integers in
each `sweeps[*].seeds` array.

The paper-scale AC-linearized and nonlinear YAML files use the explicit ten-seed
set:

```text
101, 202, 303, 404, 505, 606, 707, 808, 909, 1001
```

The seed controls generated state perturbations, noise, missing-row sampling,
and eligible bad-data sampling along the code path. Resolved configs and
trial-level records retain the seed.

### QSVT and circuit studies

QSVT configs record their own seed, phase method, degree, domain, grid, and
cache directory. Public wrappers redirect phase-cache writes beneath
`outputs/generated/`; they never update the canonical phase cache. Given the
same dependency versions, coefficients, convention, solver settings, and cache
entry, phase-response values are deterministic within numerical tolerance.

Deterministic block/support paths use declared row/column order, stable ranking,
or explicit lexicographic tie-breaking. Random-support studies record a base
seed and seed formula in their campaign JSON rather than relying on global
NumPy state.

### Bootstrap and sampling seeds

There is no hidden global bootstrap seed. Each statistical campaign stores its
seed in its own config or manifest. Examples include:

- `configs/reviewer_blocking_tqe_evidence/structure_stats.json`:
  `bootstrap_seed=20240714`, 10,000 resamples;
- `configs/tqe_physical_alignment/campaign.json`:
  ordinary bootstrap seed `20260715`, case-stratified seed `20260716`, 10,000
  replicates;
- `configs/output_aware_generalization.json`:
  bootstrap seed `8675309`, 10,000 samples, plus an explicit random-objective
  base seed and formula.

Finite-shot studies likewise retain seed IDs and shot counts in their config or
manifest. Review the relevant campaign config before interpreting an interval;
residual seeds, structures, and shot repetitions are not interchangeable
independent units.

## Data generation

### Network inputs

PYPOWER package fixtures provide IEEE14, IEEE30, IEEE57, IEEE118, and IEEE300
network arrays. They supply buses, branches, generators, topology, parameters,
and operating-point values. No network download occurs during a normal run.

The built-in project fixture provides a small IEEE14-style regression system.
Synthetic experiments instead generate weighted matrices by a controlled SVD
construction.

### Generated measurements

Measurement functions and inclusion rules construct DC or AC rows from the
network model. Configured standard deviations define an implicit diagonal
covariance, `R_ii = sigma_i^2`; rows are divided by `sigma_i` to obtain the
weighted system.

Nonlinear runs generate `z = h(x_true) + e + b`, apply missing-row selection,
and refresh residuals and Jacobians at each iteration. Single-step runs perturb
an already weighted residual. See `docs/MEASUREMENT_MODEL.md` for the exact
distinction.

Noise, missing rows, bad data, and weak-area multipliers are controlled inputs,
not field-calibrated statistics.

## Reproduction levels

### Level 1: smoke

```bash
python scripts/run_smoke_test.py
```

This exercises the main model-building and estimator paths without reproducing
paper-scale sweeps.

### Level 2: registered outputs and regeneration

```bash
python scripts/validate_reproduction.py
```

The validator checks:

- Python and pinned direct packages;
- metadata and writable audit output;
- manifest completeness and config availability;
- registered output directories and required CSV schemas;
- temporary regeneration of derived tables and figures.

Use `--core-only` only when intentionally reviewing classical paths without the
optional quantum stack. Use `--skip-regeneration` only for a file/schema
preflight; it yields a warning in the report.

### Level 3: selected full experiment

```bash
python scripts/run_experiment.py --config configs/real_ieee14.yaml
```

The public wrapper writes to `outputs/generated/real_ieee14_seed10/`. Repeat
for the config of interest from the experiment manifest.

### Level 4: complete test and style gates

```bash
pytest
ruff check .
```

The public suite spans numerical behavior, optional backends, evidence
integrity, and configuration checks and can take much longer than the smoke
suite. Its exact scope and outcome are recorded in
`release_audit/public_test_scope_report.md`.

## Output provenance and schemas

Every standard run writes its resolved config. Single runs use `metrics.csv`;
sweeps use `aggregate_metrics.csv` and `summary_metrics.csv`. Required columns
are enforced by `scripts/validate_outputs.py`. QSVT response, circuit-scaling,
and resource outputs have schema-specific checks.

The experiment manifest records model type, measurement types, dimensions,
estimators, metrics, config, and output location. Its dimensions describe the
configured system before random missing-row removal unless explicitly noted.

## Protected evidence

Do not use public wrappers to modify:

- canonical experiment outputs;
- solver or estimator definitions;
- QSVT phase data or phase caches;
- residual banks or support/configuration registries;
- frozen scientific evidence or provenance snapshots.

The wrappers isolate runs under `outputs/examples/` and `outputs/generated/`.
The table/figure regeneration check uses a temporary system directory.

## Large output distribution

Raw campaign outputs can be hundreds of megabytes and are intentionally excluded
from ordinary Git objects. The required external DOI-backed archive is pending;
see `docs/data_access.md`. A validator failure that says a raw output folder is
missing means the corresponding archive must be restored or the registered
experiment must be rerun; it is not permission to fabricate a placeholder.
