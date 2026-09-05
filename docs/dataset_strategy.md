# Dataset And Benchmark Strategy

This project separates algorithmic benchmarks from field-data validation.

## 1. Synthetic Generated Benchmarks

`system.mode: synthetic_linearized` creates weighted linear systems with
controlled singular-value decay, noise, missing rows, and bad-data perturbations.
These runs are useful for isolating numerical behavior, but they are not power
grid case studies.

## 2. Built-In Project Fixture Data

The built-in IEEE14-style fixture in `src/robust_qsvt_se/data/cases.py` supports
fast DC, AC-linearized, nonlinear iterative AC, bad-data, and robust-baseline
regression tests. It is bundled project fixture data and should not be described
as an external dataset.

## 3. PYPOWER IEEE/MATPOWER-Compatible Benchmark Cases

The primary benchmark uses:

> standard IEEE/MATPOWER-compatible benchmark systems loaded through PYPOWER

The current loader uses `pypower>=5.1.19` and supports IEEE14, IEEE30, IEEE57,
IEEE118, and IEEE300 through PYPOWER's package fixtures. No network download is
performed during experiment runs; reproducibility comes from the Python package
version, the resolved YAML config, and the saved output artifacts.

Use this wording in public descriptions:

> We evaluate on standard IEEE/MATPOWER-compatible benchmark systems loaded
> through PYPOWER, with synthetic PMU/SCADA-like measurement noise, missing
> measurements, bad-data stress, and weak-observability perturbations.

Do not use this wording:

> real PMU/SCADA field data

PYPOWER/MATPOWER cases are appropriate for an algorithmic spectral-filter and
state-estimation benchmark, but they are not direct utility PMU/SCADA
measurement archives.

## 4. pandapower Option

pandapower may be useful later if the project needs easier nonlinear AC power
flow tooling, additional public networks, or a second independent case source.
It is not required for the current benchmark because PYPOWER already
provides the requested IEEE14/30/57/118/300 MATPOWER-compatible cases and keeps
the dependency set smaller.

## 5. Texas A&M Synthetic Grid Cases

Texas A&M synthetic grids could add larger realistic synthetic transmission
systems. They should be described as synthetic grid cases, not field PMU/SCADA
data. Add them only if a future study needs scaling evidence beyond IEEE300.

## 6. PMU Event Data

Public PMU event datasets, such as event libraries, are better treated as future
external validation material. They often lack the full network topology,
measurement placement, and ground-truth state-estimation labels required by this
benchmark. They should not be used as the main benchmark unless those missing
inputs are resolved explicitly.

## Recommendation

PYPOWER/MATPOWER IEEE cases are the primary algorithmic benchmark. Public
interpretation should clearly separate:

- synthetic numerical stress tests;
- bundled project fixture tests;
- PYPOWER/MATPOWER IEEE benchmark case experiments;
- small QSP/QSVT proof-of-concept circuits;
- future field-data or larger synthetic-grid validation.

The current nonlinear AC seed10 experiments cover IEEE14, IEEE30, IEEE57,
IEEE118, and IEEE300. IEEE300 is long-running on a laptop-scale machine, so the
run uses checkpoint/resume artifacts (`trial_results.jsonl`,
`checkpoint_state.json`, and `progress.log`) to make completion auditable.

The small QSP/QSVT backend demos are separate from these case benchmarks. They
show bounded polynomial approximation, real phase synthesis, PennyLane/Qiskit
matrix QSVT correctness circuits, and an explicit small block-encoding QSVT
prototype on weighted Jacobians derived from the PYPOWER IEEE benchmark
pipeline. Full IEEE14/30 matrix construction is attempted where feasible;
IEEE118/300 are represented by full-matrix resource estimates and deterministic
submatrix circuit demonstrations. They do not implement production-scale
hardware-native QSVT for IEEE118 or IEEE300.

The QSVT validation report is a small research-matrix quantum-circuit proof of
concept. It is connected to the state-estimation problem through weighted
Jacobians, but it remains separate from the large IEEE classical spectral-filter
benchmark. Documentation should distinguish the
dense-unitary correctness path from the explicit block-encoding prototype. The
Qiskit dense-unitary path imports a complete QSVT unitary and may transpile that
unitary to report gate-cost artifacts, but this is not a scalable
hardware-native block-encoding oracle implementation. The explicit prototype
uses a dense canonical block-encoding primitive for a small matrix, followed by
a structured sequence of block-encoding calls and projector phase rotations;
this is still not a hardware-native oracle decomposition for full IEEE systems.

For larger IEEE cases, the defensible strategy is to report classical
spectral-filter benchmark results plus QSVT resource estimates. Full circuit
simulation of IEEE118/300 weighted Jacobians is not expected to be laptop
feasible.

The next dataset extension, if needed, should be a larger synthetic transmission
case family with complete topology and state-estimation inputs. Public PMU event
data should remain future work unless it can be tied to a full network model and
ground-truth labels.

The completed IEEE300 nonlinear seed10 run is the full configured benchmark
output. The reduced-runtime IEEE300 output is historical/diagnostic evidence
and must not be substituted for the complete run.
