# Reproduction Validation Report

Generated: 2026-09-04T02:20:55.110453+00:00

Overall status: **PASS**

- Branch: `research/generalized-rectangular-qsvt`
- Commit: `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`
- Working tree: dirty
- Runtime: 18.100 seconds
- Checks: 143 passed, 1 warning, 0 failed

## Checks

| Status | Category | Target | Detail |
|---|---|---|---|
| PASS | artifact metadata | README.md | present |
| PASS | artifact metadata | LICENSE | present |
| PASS | artifact metadata | CITATION.cff | present |
| PASS | artifact metadata | VERSION | present |
| PASS | artifact metadata | environment.yml | present |
| PASS | artifact metadata | requirements.txt | present |
| PASS | environment | audit output directory | available and writable |
| PASS | environment | Python | 3.12.11 via python; requires >=3.11 |
| WARN | environment | conda command | not on PATH; an already-created venv remains usable |
| PASS | package | matplotlib | installed=3.10.9; expected=3.10.9 |
| PASS | package | numpy | installed=2.4.4; expected=2.4.4 |
| PASS | package | pandas | installed=3.0.3; expected=3.0.3 |
| PASS | package | PYPOWER | installed=5.1.19; expected=5.1.19 |
| PASS | package | PyYAML | installed=6.0.3; expected=6.0.3 |
| PASS | package | scipy | installed=1.17.1; expected=1.17.1 |
| PASS | package | pytest | installed=9.0.3; expected=9.0.3 |
| PASS | package | ruff | installed=0.15.12; expected=0.15.12 |
| PASS | package | mpmath | installed=1.3.0; expected=1.3.0 |
| PASS | package | pennylane | installed=0.45.0; expected=0.45.0 |
| PASS | package | pyqsp | installed=0.2.0; expected=0.2.0 |
| PASS | package | qiskit | installed=2.4.1; expected=2.4.1 |
| PASS | package | qiskit-aer | installed=0.17.2; expected=0.17.2 |
| PASS | package | sympy | installed=1.14.0; expected=1.14.0 |
| PASS | manifest | outputs/reproducibility_audit/experiment_manifest.json | loaded 18 experiment records |
| PASS | manifest | synthetic_conditioning_smoke | required fields present |
| PASS | manifest provenance | synthetic_conditioning_smoke | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/ieee14_spectral_smoke.yaml | configuration is available |
| PASS | output folder | outputs/examples/smoke_test/synthetic | output directory is available |
| PASS | output provenance | outputs/examples/smoke_test/synthetic | resolved configuration is present |
| PASS | output schema | outputs/examples/smoke_test/synthetic/metrics.csv | 31 columns; required schema present |
| PASS | manifest | ieee14_dc_linearized_smoke | required fields present |
| PASS | manifest provenance | ieee14_dc_linearized_smoke | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/ieee14_dc_smoke.yaml | configuration is available |
| PASS | output folder | outputs/examples/smoke_test/dc_linearized | output directory is available |
| PASS | output provenance | outputs/examples/smoke_test/dc_linearized | resolved configuration is present |
| PASS | output schema | outputs/examples/smoke_test/dc_linearized/metrics.csv | 31 columns; required schema present |
| PASS | manifest | ieee14_ac_linearized_smoke | required fields present |
| PASS | manifest provenance | ieee14_ac_linearized_smoke | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/ieee14_ac_smoke.yaml | configuration is available |
| PASS | output folder | outputs/examples/smoke_test/ac_linearized | output directory is available |
| PASS | output provenance | outputs/examples/smoke_test/ac_linearized | resolved configuration is present |
| PASS | output schema | outputs/examples/smoke_test/ac_linearized/metrics.csv | 31 columns; required schema present |
| PASS | manifest | ieee14_iterative_ac_smoke | required fields present |
| PASS | manifest provenance | ieee14_iterative_ac_smoke | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/ieee14_ac_iterative_smoke.yaml | configuration is available |
| PASS | output folder | outputs/examples/smoke_test/iterative_ac | output directory is available |
| PASS | output provenance | outputs/examples/smoke_test/iterative_ac | resolved configuration is present |
| PASS | output schema | outputs/examples/smoke_test/iterative_ac/metrics.csv | 32 columns; required schema present |
| PASS | manifest | ieee14_robust_bad_data_smoke | required fields present |
| PASS | manifest provenance | ieee14_robust_bad_data_smoke | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/ieee14_robust_bad_data_smoke.yaml | configuration is available |
| PASS | output folder | outputs/examples/smoke_test/robust_bad_data | output directory is available |
| PASS | output provenance | outputs/examples/smoke_test/robust_bad_data | resolved configuration is present |
| PASS | output schema | outputs/examples/smoke_test/robust_bad_data/metrics.csv | 31 columns; required schema present |
| PASS | manifest | ieee14_ac_linearized_seed10 | required fields present |
| PASS | manifest provenance | ieee14_ac_linearized_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/real_ieee14.yaml | configuration is available |
| PASS | output folder | outputs/real_ieee14_seed10 | output directory is available |
| PASS | output provenance | outputs/real_ieee14_seed10 | resolved configuration is present |
| PASS | output schema | outputs/real_ieee14_seed10/aggregate_metrics.csv | 34 columns; required schema present |
| PASS | output schema | outputs/real_ieee14_seed10/summary_metrics.csv | 65 columns; required schema present |
| PASS | manifest | ieee30_ac_linearized_seed10 | required fields present |
| PASS | manifest provenance | ieee30_ac_linearized_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/real_ieee30.yaml | configuration is available |
| PASS | output folder | outputs/real_ieee30_seed10 | output directory is available |
| PASS | output provenance | outputs/real_ieee30_seed10 | resolved configuration is present |
| PASS | output schema | outputs/real_ieee30_seed10/aggregate_metrics.csv | 34 columns; required schema present |
| PASS | output schema | outputs/real_ieee30_seed10/summary_metrics.csv | 65 columns; required schema present |
| PASS | manifest | ieee57_ac_linearized_seed10 | required fields present |
| PASS | manifest provenance | ieee57_ac_linearized_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/real_ieee57.yaml | configuration is available |
| PASS | output folder | outputs/real_ieee57_seed10 | output directory is available |
| PASS | output provenance | outputs/real_ieee57_seed10 | resolved configuration is present |
| PASS | output schema | outputs/real_ieee57_seed10/aggregate_metrics.csv | 34 columns; required schema present |
| PASS | output schema | outputs/real_ieee57_seed10/summary_metrics.csv | 65 columns; required schema present |
| PASS | manifest | ieee118_ac_linearized_seed10 | required fields present |
| PASS | manifest provenance | ieee118_ac_linearized_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/real_ieee118.yaml | configuration is available |
| PASS | output folder | outputs/real_ieee118_seed10 | output directory is available |
| PASS | output provenance | outputs/real_ieee118_seed10 | resolved configuration is present |
| PASS | output schema | outputs/real_ieee118_seed10/aggregate_metrics.csv | 34 columns; required schema present |
| PASS | output schema | outputs/real_ieee118_seed10/summary_metrics.csv | 65 columns; required schema present |
| PASS | manifest | ieee300_ac_linearized_seed10 | required fields present |
| PASS | manifest provenance | ieee300_ac_linearized_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/real_ieee300.yaml | configuration is available |
| PASS | output folder | outputs/real_ieee300_seed10 | output directory is available |
| PASS | output provenance | outputs/real_ieee300_seed10 | resolved configuration is present |
| PASS | output schema | outputs/real_ieee300_seed10/aggregate_metrics.csv | 34 columns; required schema present |
| PASS | output schema | outputs/real_ieee300_seed10/summary_metrics.csv | 65 columns; required schema present |
| PASS | manifest | ieee14_nonlinear_ac_seed10 | required fields present |
| PASS | manifest provenance | ieee14_nonlinear_ac_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/nonlinear_ac_ieee14_seed10.yaml | configuration is available |
| PASS | output folder | outputs/nonlinear_ac_ieee14_seed10 | output directory is available |
| PASS | output provenance | outputs/nonlinear_ac_ieee14_seed10 | resolved configuration is present |
| PASS | output schema | outputs/nonlinear_ac_ieee14_seed10/aggregate_metrics.csv | 35 columns; required schema present |
| PASS | output schema | outputs/nonlinear_ac_ieee14_seed10/summary_metrics.csv | 73 columns; required schema present |
| PASS | manifest | ieee30_nonlinear_ac_seed10 | required fields present |
| PASS | manifest provenance | ieee30_nonlinear_ac_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/nonlinear_ac_ieee30_seed10.yaml | configuration is available |
| PASS | output folder | outputs/nonlinear_ac_ieee30_seed10 | output directory is available |
| PASS | output provenance | outputs/nonlinear_ac_ieee30_seed10 | resolved configuration is present |
| PASS | output schema | outputs/nonlinear_ac_ieee30_seed10/aggregate_metrics.csv | 35 columns; required schema present |
| PASS | output schema | outputs/nonlinear_ac_ieee30_seed10/summary_metrics.csv | 73 columns; required schema present |
| PASS | manifest | ieee57_nonlinear_ac_seed10 | required fields present |
| PASS | manifest provenance | ieee57_nonlinear_ac_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/nonlinear_ac_ieee57_seed10.yaml | configuration is available |
| PASS | output folder | outputs/nonlinear_ac_ieee57_seed10 | output directory is available |
| PASS | output provenance | outputs/nonlinear_ac_ieee57_seed10 | resolved configuration is present |
| PASS | output schema | outputs/nonlinear_ac_ieee57_seed10/aggregate_metrics.csv | 35 columns; required schema present |
| PASS | output schema | outputs/nonlinear_ac_ieee57_seed10/summary_metrics.csv | 73 columns; required schema present |
| PASS | manifest | ieee118_nonlinear_ac_seed10 | required fields present |
| PASS | manifest provenance | ieee118_nonlinear_ac_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/nonlinear_ac_ieee118_seed10.yaml | configuration is available |
| PASS | output folder | outputs/nonlinear_ac_ieee118_seed10 | output directory is available |
| PASS | output provenance | outputs/nonlinear_ac_ieee118_seed10 | resolved configuration is present |
| PASS | output schema | outputs/nonlinear_ac_ieee118_seed10/aggregate_metrics.csv | 35 columns; required schema present |
| PASS | output schema | outputs/nonlinear_ac_ieee118_seed10/summary_metrics.csv | 73 columns; required schema present |
| PASS | manifest | ieee300_nonlinear_ac_seed10 | required fields present |
| PASS | manifest provenance | ieee300_nonlinear_ac_seed10 | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/nonlinear_ac_ieee300_seed10.yaml | configuration is available |
| PASS | output folder | outputs/nonlinear_ac_ieee300_seed10 | output directory is available |
| PASS | output provenance | outputs/nonlinear_ac_ieee300_seed10 | resolved configuration is present |
| PASS | output schema | outputs/nonlinear_ac_ieee300_seed10/aggregate_metrics.csv | 35 columns; required schema present |
| PASS | output schema | outputs/nonlinear_ac_ieee300_seed10/summary_metrics.csv | 73 columns; required schema present |
| PASS | manifest | qsvt_phase_response_validation | required fields present |
| PASS | manifest provenance | qsvt_phase_response_validation | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/qsvt_phase_validation_paper.yaml | configuration is available |
| PASS | output folder | outputs/qsvt_phase_validation_paper | output directory is available |
| PASS | output provenance | outputs/qsvt_phase_validation_paper | resolved configuration is present |
| PASS | output schema | outputs/qsvt_phase_validation_paper/qsp_validation_grid.csv | 9 columns; required schema present |
| PASS | manifest | qsvt_selected_block_circuit_scaling | required fields present |
| PASS | manifest provenance | qsvt_selected_block_circuit_scaling | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/qsvt_circuit_scaling.yaml | configuration is available |
| PASS | output folder | outputs/qsvt_circuit_scaling | output directory is available |
| PASS | output provenance | outputs/qsvt_circuit_scaling | resolved configuration is present |
| PASS | output schema | outputs/qsvt_circuit_scaling/circuit_scaling_results.csv | 26 columns; required schema present |
| PASS | manifest | qsvt_full_ieee_resource_estimation | required fields present |
| PASS | manifest provenance | qsvt_full_ieee_resource_estimation | seed, RNG, config path, and source commit are recorded |
| PASS | config | configs/qsvt_resource_full_ieee.yaml | configuration is available |
| PASS | output folder | outputs/qsvt_resource_full_ieee | output directory is available |
| PASS | output provenance | outputs/qsvt_resource_full_ieee | resolved configuration is present |
| PASS | output schema | outputs/qsvt_resource_full_ieee/qsvt_resource_estimates.csv | 30 columns; required schema present |
| PASS | regeneration | derived tables | temporary rebuild produced 7 table bundles |
| PASS | regeneration | derived figures | temporary rebuild produced 5 figure bundles |

## Interpretation

A PASS confirms that the declared environment, registered configs, existing output schemas, and isolated table/figure rebuild were available in this working copy. It does not independently re-run every long IEEE sweep, prove field-data validity, execute full-scale quantum hardware, or establish speedup.
