# Results Index

This index distinguishes compact release evidence, generated reruns, and raw
data requiring external archival. It does not designate manuscript files or
submission packages; those are outside the repository scope.

## Registered experiment inventory

The primary machine-readable index is:

```text
outputs/reproducibility_audit/experiment_manifest.json
```

Each registered record includes its config path, model type, measurement types,
state and row dimensions, estimators, metrics, output path, seed/RNG information,
and source commit.

## Standard benchmark outputs

| Family | Established output pattern |
|---|---|
| Synthetic conditioning | `outputs/ieee14_spectral_*` |
| DC-linearized IEEE14 | `outputs/ieee14_dc_*` |
| AC-linearized PYPOWER | `outputs/real_ieee*_seed10/` |
| Nonlinear AC PYPOWER | `outputs/nonlinear_ac_ieee*_seed10/` |
| Controlled bad data/robustness | `outputs/ieee14_*bad_data*` |

These directories may be local-only when they are large or regenerable. Use the
registered config and `scripts/run_experiment.py` to create a new isolated copy
under `outputs/generated/`.

## QSVT validation outputs

| Evidence | Established output | Shipped in Git? |
|---|---|:---:|
| Selected-submatrix boundary | `outputs/phase2_qsvt_boundary/` | yes |
| Access/resource traceability | `outputs/phase3_resource_reproducibility/` | yes |
| Integrated small-block readout | `outputs/phase8_integrated_readout/` | yes |
| Block/full-system bridge characterization | `outputs/phase8_bridge_characterization/` | yes |
| Scalar target/polynomial/phase response | `outputs/qsvt_phase_validation_paper/` | no — regenerate |
| Small selected-block circuit scaling | `outputs/qsvt_circuit_scaling/` | no — regenerate |
| Matrix resource proxies | `outputs/qsvt_resource_full_ieee/` | no — regenerate |

Scalar, polynomial, circuit, postselection, and finite-shot errors are distinct.

## Which output paths are in a Git checkout

The repository versions exactly the evidence its own checks read, so a bare
clone can run `pytest` and `scripts/validate_reproduction.py` without
downloading anything. Forty-one `outputs/` roots are included.

Evidence roots asserted by the test suite (25):

```text
classical_selected_observable_baseline   phase8_bridge_characterization
cross_case_larger_block_validation       phase8_integrated_readout
final_contribution_evidence              phase9_integrated_8x8_readout
final_useful_overlap_validation          phase10_full_rectangular_selected_output_qsvt
full_rectangular_breakthrough            phase10_sparse_wrapper_8x8_complete
generalized_rectangular_qsvt             qsvt_phase_cache
generic_sparse_qsvt_compiler             rectangular_convention_fix
nonlinear_closed_loop_qsvt               reproducibility_audit
output_aware_generalization              sparse_chain_reconciliation
output_aware_structural_generalization   sparse_error_precision_study
phase2_qsvt_boundary                     sparse_integrated_chain
phase3_resource_reproducibility          tqe_blocking_revision
                                         tqe_implementation_revision
```

Registered benchmark and resource outputs the reproduction validator and the
cost-accounting checks read (16):

```text
real_ieee14_seed10        nonlinear_ac_ieee14_seed10     qsvt_phase_validation_paper
real_ieee30_seed10        nonlinear_ac_ieee30_seed10     qsvt_circuit_scaling
real_ieee57_seed10        nonlinear_ac_ieee57_seed10     qsvt_resource_full_ieee
real_ieee118_seed10       nonlinear_ac_ieee118_seed10    qsvt_oracle_model_resources
real_ieee300_seed10       nonlinear_ac_ieee300_seed10    qsvt_hardware_ieee14_2x2
                                                         qsvt_hardware_ieee14_4x4
```

plus `outputs/examples/smoke_test/` and the `outputs/examples/README.md` and
`outputs/generated/README.md` policy files. QSP phase-angle caches
(`**/phase_cache/`) inside these roots are versioned too: they are
expensive-to-recompute scientific data, not a disposable tool cache.

**Any other `outputs/...` path named in this repository's documentation is an
output *destination*, not a shipped file.** Those paths name where a run writes
its results and where the recorded evidence lived in the development workspace.
Recreate any of them with the registered configuration and
`scripts/run_experiment.py` (or the matching `scripts/run_*.py` entry point),
which write under `outputs/generated/`.

The six raw tables that remain too large for ordinary Git are inventoried in
[`data_manifest.json`](data_manifest.json) with the reserved archive DOI. They
are **not** required to run the test suite.
No single output directory should be interpreted as full IEEE-scale hardware
execution or quantum-speedup evidence.

## Fresh reruns

```text
outputs/examples/     lightweight examples
outputs/generated/    explicit researcher reruns
```

Fresh runs are not authoritative until their resolved configuration, seed,
environment, source commit, and output checksums are recorded.

## High-volume data

Raw files exceeding Git hosting limits are listed with sizes and SHA-256 hashes
in `docs/data_access.md`. Their reserved external DOI is `10.5281/zenodo.22326883`;
the Zenodo record is still a draft and is not yet publicly available. The
absence of those raw tables from a Git checkout must be distinguished from the
absence of configs or code needed to regenerate them.
