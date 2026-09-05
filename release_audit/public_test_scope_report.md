# Public Test Scope Report

Prepared: 2026-09-05

This report states exactly which tests the public repository carries, what
was removed at the public boundary, and why each removal is not a loss of
scientific coverage.

## Scope change

| | Development tree | Public repository |
|---|---:|---:|
| `tests/*.py` files | 430 | 396 |
| Whole files removed | — | 34 |
| Files with individual nodes removed | — | 43 |
| Individual nodes removed from surviving files | — | 63 |

## Removal principle

A test was removed only when it exercised a surface the public repository
does not ship — manuscript text, submission packaging, reviewer-audit
reports, or a frozen evidence artifact distributed externally. Every entry
below carries the same confirmation, recorded per node at pruning time:

> No unique solver, estimator, measurement-generation, or experiment
> execution path is covered only by the removed node.

**No assertion was weakened, no tolerance widened, and no test was skipped
or xfailed to obtain a passing run.**

## Removal categories

| Category | Nodes | Meaning |
|---|---:|---|
| `nonpublic-frozen-evidence` | 26 | asserts a frozen evidence/reporting artifact outside the approved public output roots |
| `submission-audit-only` | 19 | reads a retired submission-stage audit artifact and executes no solver or experiment |
| `manuscript-only` | 14 | checks manuscript text, tables, or manuscript-linked hashes |
| `private-provenance-snapshot` | 3 | requires a path-bearing or manuscript-bearing protected snapshot |
| `external-data-dependent` | 3 | requires a raw dataset represented by data_manifest.json rather than shipped in Git |

## Whole test files removed (34)

Each targets manuscript, submission-package, or reviewer-delivery code that
is itself excluded from the public tree.

- `tests/test_canonical_paper_numbers.py`
- `tests/test_claim_boundary_writer.py`
- `tests/test_claim_lint.py`
- `tests/test_cross_case_solver_prototype_package.py`
- `tests/test_evidence_freeze.py`
- `tests/test_final_artifact_validator.py`
- `tests/test_final_configuration_freeze.py`
- `tests/test_final_figure_rendering.py`
- `tests/test_final_gap_resolution_package.py`
- `tests/test_final_implementation_hardening_package.py`
- `tests/test_final_implementation_verification_package.py`
- `tests/test_final_optional_evidence_package.py`
- `tests/test_final_readout_hardening_package.py`
- `tests/test_final_table_selection.py`
- `tests/test_final_writing_readiness_package.py`
- `tests/test_full_vector_readout_claim_boundary.py`
- `tests/test_gap_closed_final_package.py`
- `tests/test_latex_asset_export.py`
- `tests/test_main_paper_tables.py`
- `tests/test_paper_measurement_inventory.py`
- `tests/test_paper_readiness_outputs.py`
- `tests/test_paper_ready_qsvt_tables.py`
- `tests/test_paper_ready_results.py`
- `tests/test_qsvt_artifact_freeze.py`
- `tests/test_qsvt_updated_paper_ready_tables.py`
- `tests/test_quantum_contribution_audit.py`
- `tests/test_reporting_exports.py`
- `tests/test_reproducibility_package_audit.py`
- `tests/test_selected_solver_prototype_package.py`
- `tests/test_test_quality_appendix.py`
- `tests/test_test_quality_audit.py`
- `tests/test_test_quality_inventory_consistency.py`
- `tests/test_tqe_main_paper_results.py`
- `tests/test_tqe_reviewer_revision_audit.py`

## Individual nodes removed from surviving files (63 nodes in 43 files)

| Test file | Node | Category | Surface it required |
|---|---|---|---|
| `tests/test_alpha_selection_classification.py` | `test_alpha_selection_labels_feasibility_sweeps` | submission-audit-only | `outputs/final_falsification_and_submission/alpha_selection_audit.csv` |
| `tests/test_classical_baseline_reproduction.py` | `test_classical_baselines_include_fast_dense_ridge` | submission-audit-only | `outputs/final_falsification_and_submission/classical_baseline_reproduction.csv` |
| `tests/test_complex_scope_boundary.py` | `test_complex_support_is_not_implied` | submission-audit-only | `outputs/final_falsification_and_submission/complex_boundary_audit.csv` |
| `tests/test_even_degree_scope_boundary.py` | `test_even_degrees_are_rejected` | submission-audit-only | `outputs/final_falsification_and_submission/even_degree_boundary_audit.csv` |
| `tests/test_final_claim_support_matrix.py` | `test_high_risk_claims_have_artifacts_and_boundaries` | submission-audit-only | `outputs/final_falsification_and_submission/final_claim_support_matrix.csv` |
| `tests/test_final_error_budget_traceability.py` | `test_error_budget_has_required_separated_terms` | submission-audit-only | `outputs/final_falsification_and_submission/final_error_budget_audit.csv` |
| `tests/test_final_evidence_configuration_registry.py` | `test_configuration_families_remain_separate` | nonpublic-frozen-evidence | `outputs/final_contribution_evidence/canonical_configuration_registry.csv` |
| `tests/test_final_evidence_configuration_registry.py` | `test_configuration_registry_is_unique_typed_and_resolved` | nonpublic-frozen-evidence | `outputs/final_contribution_evidence/canonical_configuration_registry.csv` |
| `tests/test_final_evidence_protected_sources.py` | `test_all_protected_sources_are_byte_identical` | private-provenance-snapshot | `outputs/final_contribution_evidence/protected_source_snapshot.json` |
| `tests/test_final_evidence_protected_sources.py` | `test_prior_manifests_and_checksum_registries_match_snapshot` | private-provenance-snapshot | `outputs/final_contribution_evidence/protected_source_snapshot.json` |
| `tests/test_final_evidence_result_registry.py` | `test_canonical_result_ids_sources_and_configurations_resolve` | nonpublic-frozen-evidence | `outputs/final_contribution_evidence/canonical_configuration_registry.csv` |
| `tests/test_generalization_certificates.py` | `test_certificate_holds_for_every_generalization_result` | external-data-dependent | `outputs/output_aware_generalization/certificate_results.csv` |
| `tests/test_generalization_frozen_configuration.py` | `test_frozen_configuration_matches_completed_development_study` | nonpublic-frozen-evidence | `outputs/output_aware_sparse_selection/study_configuration.json` |
| `tests/test_generalization_instance_registry.py` | `test_exclusions_retain_reasons_and_protected_snapshot_is_unchanged` | private-provenance-snapshot | `outputs/output_aware_generalization/protected_path_snapshot.json` |
| `tests/test_generalization_instance_registry.py` | `test_instance_fingerprints_and_metadata_are_stable` | nonpublic-frozen-evidence | `outputs/output_aware_generalization/instances/ieee14_eval_seed_14104_block_8x8.json` |
| `tests/test_generalization_selector_evaluation.py` | `test_campaign_supports_share_constraints_and_refined_supports_are_feasible` | nonpublic-frozen-evidence | `outputs/output_aware_generalization/supports/ieee14_eval_seed_14101_block_8x8/ieee14_eval_seed_14101_block_8x8__sensitivity_initial_mean_k8_s2_239e7a9180fc.json` |
| `tests/test_generic_sparse_qsvt_artifacts.py` | `test_manifest_and_checksums_cover_and_validate_new_evidence` | nonpublic-frozen-evidence | `outputs/generic_sparse_qsvt_compiler/artifact_manifest.json` |
| `tests/test_generic_sparse_qsvt_artifacts.py` | `test_protected_roots_and_manuscript_match_pre_edit_hashes` | manuscript-only | `outputs/generic_sparse_qsvt_compiler/protected_hash_audit.json` |
| `tests/test_generic_sparse_qsvt_artifacts.py` | `test_required_artifacts_exist_and_are_nonempty` | nonpublic-frozen-evidence | `removed artifact surface` |
| `tests/test_heldout_matrix_reproduction.py` | `test_heldout_matrix_rows_reproduced` | submission-audit-only | `outputs/final_falsification_and_submission/heldout_falsification_results.csv` |
| `tests/test_heldout_seed_separation.py` | `test_heldout_seeds_do_not_overlap_development_seeds` | submission-audit-only | `outputs/final_falsification_and_submission/heldout_seed_overlap_audit.csv` |
| `tests/test_ieee14_robustness_reproduction.py` | `test_ieee14_robustness_40_rows_pass` | submission-audit-only | `outputs/final_falsification_and_submission/ieee14_robustness_reproduction.csv` |
| `tests/test_independent_convention_reproduction.py` | `test_convention_rule_is_not_degree255_specific` | submission-audit-only | `outputs/final_falsification_and_submission/independent_convention_reproduction.csv` |
| `tests/test_independent_convention_reproduction.py` | `test_independent_convention_reproduces_all_odd_degrees` | submission-audit-only | `outputs/final_falsification_and_submission/independent_convention_reproduction.csv` |
| `tests/test_mlae_evidence_boundary.py` | `test_mlae_controlled_execution_not_ieee_execution` | submission-audit-only | `outputs/final_falsification_and_submission/mlae_execution_audit.csv` |
| `tests/test_multi_ieee_application_reproduction.py` | `test_multi_ieee_useful_overlap_rows_present` | submission-audit-only | `outputs/final_falsification_and_submission/multi_ieee_application_reproduction.csv` |
| `tests/test_multi_ieee_quantum_reproduction.py` | `test_quantum_reproduction_keeps_case_degree_pairs` | submission-audit-only | `outputs/final_falsification_and_submission/multi_ieee_quantum_reproduction.csv` |
| `tests/test_phase3_resource_reproducibility.py` | `test_generated_phase3_artifacts_exist_and_ledger_contract` | manuscript-only | `removed artifact surface` |
| `tests/test_phase3_resource_reproducibility.py` | `test_manuscript_tables_have_expected_labels` | manuscript-only | `removed artifact surface` |
| `tests/test_phase3_resource_reproducibility.py` | `test_traceability_manifest_paths_exist_and_checksums_match` | manuscript-only | `removed artifact surface` |
| `tests/test_qsvt_resource_accounting.py` | `test_generated_degree255_table_labels_depth_and_signal_calls` | manuscript-only | `removed artifact surface` |
| `tests/test_qsvt_resource_accounting.py` | `test_manuscript_states_l_alt_is_not_a_query_count` | manuscript-only | `removed artifact surface` |
| `tests/test_readout_registry_consistency.py` | `test_sampled_rows_have_traceable_counts_and_artifacts` | nonpublic-frozen-evidence | `outputs/phase9_integrated_8x8_readout/integrated_readout_summary.csv` |
| `tests/test_readout_two_view_summary.py` | `test_canonical_numbers_record_codesign` | nonpublic-frozen-evidence | `removed artifact surface` |
| `tests/test_readout_two_view_summary.py` | `test_end_to_end_sweep_then_summary` | nonpublic-frozen-evidence | `removed artifact surface` |
| `tests/test_resource_ledger_traceability.py` | `test_resource_categories_separated` | submission-audit-only | `outputs/final_falsification_and_submission/resource_ledger_audit.csv` |
| `tests/test_resource_ledger_traceability.py` | `test_resource_query_counts_match_degree` | submission-audit-only | `outputs/final_falsification_and_submission/resource_query_count_checks.csv` |
| `tests/test_reviewer_evidence_studies.py` | `test_floor_sensitivity_does_not_mutate_primary_floor` | nonpublic-frozen-evidence | `outputs/reviewer_blocking_tqe_evidence/physical_selected_output_rows.csv` |
| `tests/test_reviewer_evidence_studies.py` | `test_structure_stats_unit_is_structure_not_row` | nonpublic-frozen-evidence | `outputs/reviewer_blocking_tqe_evidence/physical_selected_output_rows.csv` |
| `tests/test_selected_observable_revision.py` | `test_manuscript_removes_unsupported_perturbation_bound` | manuscript-only | `removed artifact surface` |
| `tests/test_shot_confidence_intervals.py` | `test_highest_shot_confidence_interval_precision` | submission-audit-only | `outputs/final_falsification_and_submission/shot_confidence_interval_audit.csv` |
| `tests/test_shot_estimator_independent.py` | `test_shot_estimator_seed_rows_and_derivation` | submission-audit-only | `outputs/final_falsification_and_submission/shot_estimator_independent_validation.csv` |
| `tests/test_sparse_chain_reconciliation.py` | `test_manuscript_matches_verdict_and_resource_convention` | manuscript-only | `removed artifact surface` |
| `tests/test_sparse_error_decomposition.py` | `test_stage_baseline_reproduces_frozen_chain` | nonpublic-frozen-evidence | `outputs/sparse_integrated_chain/finite_shot_summary.csv` |
| `tests/test_structural_candidate_registry.py` | `test_included_candidates_are_active_fingerprinted_8x8_blocks` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/candidates/ieee14/ieee14_angle_dominant_v03.json` |
| `tests/test_structural_certificates.py` | `test_structural_certificates_hold_and_do_not_use_actual_error` | external-data-dependent | `outputs/output_aware_structural_generalization/certificate_results.csv` |
| `tests/test_structural_data_isolation.py` | `test_support_construction_uses_no_heldout_data_and_group_test_is_grouped` | external-data-dependent | `outputs/output_aware_structural_generalization/heldout_results.csv` |
| `tests/test_structural_frozen_method.py` | `test_frozen_method_reconciles_with_previous_benchmark` | nonpublic-frozen-evidence | `outputs/output_aware_generalization/study_configuration.json` |
| `tests/test_structural_pareto.py` | `test_all_four_resource_matched_pareto_frontiers_are_valid` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/pareto_candidates_error_gates.csv` |
| `tests/test_structural_selector_evaluation.py` | `test_primary_completed_supports_satisfy_cardinality_degree_and_coverage` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/instances/ieee14_structural_group_01_realization_02_seed_14202_8x8.json` |
| `tests/test_structured_access_boundary.py` | `test_structured_access_quantum_oracle_modeled_only` | submission-audit-only | `outputs/final_falsification_and_submission/structured_access_boundary_audit.csv` |
| `tests/test_three_view_readout_summary.py` | `test_matched_view_preserves_high_error` | nonpublic-frozen-evidence | `removed artifact surface` |
| `tests/test_three_view_readout_summary.py` | `test_three_views_present_per_subproblem` | nonpublic-frozen-evidence | `removed artifact surface` |
| `tests/test_tqe_closed_loop_audit.py` | `test_18_manuscript_avoids_statistical_indistinguishability` | manuscript-only | `removed artifact surface` |
| `tests/test_tqe_closed_loop_audit.py` | `test_31_degree31_operator_63_is_logical_not_primitive` | manuscript-only | `removed artifact surface` |
| `tests/test_tqe_closed_loop_audit.py` | `test_35_manuscript_plateau_and_finite_shot_wording_matches_evidence` | manuscript-only | `removed artifact surface` |
| `tests/test_tqe_closed_loop_audit.py` | `test_36_all_declared_closed_loop_assets_exist_and_are_generated` | manuscript-only | `removed artifact surface` |
| `tests/test_tqe_evidence_registry.py` | `test_artifact_paths_exist` | nonpublic-frozen-evidence | `outputs/phase10_full_rectangular_selected_output_qsvt/full_rectangular_qsvt_vs_ridge.csv` |
| `tests/test_tqe_physical_alignment_protocol.py` | `test_frozen_registry_has_twelve_outcome_independent_unique_structures` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/residual_splits/ieee14_structural_group_02_realization_02_seed_14202_8x8.json` |
| `tests/test_tqe_physical_alignment_protocol.py` | `test_physical_functionals_are_unit_norm_and_unavailable_ones_remain_unsubstituted` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/residual_splits/ieee14_structural_group_02_realization_02_seed_14202_8x8.json` |
| `tests/test_tqe_physical_alignment_protocol.py` | `test_small_physical_campaign_is_reproducible_except_runtime` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/residual_splits/ieee14_structural_group_02_realization_02_seed_14202_8x8.json` |
| `tests/test_tqe_physical_alignment_protocol.py` | `test_small_physical_campaign_separates_support_and_truth_metrics` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/residual_splits/ieee14_structural_group_02_realization_02_seed_14202_8x8.json` |
| `tests/test_tqe_physical_alignment_protocol.py` | `test_truth_reference_is_independently_x_true_minus_x0` | nonpublic-frozen-evidence | `outputs/output_aware_structural_generalization/residual_splits/ieee14_structural_group_02_realization_02_seed_14202_8x8.json` |

## Scientific assertion that was preserved, not deleted

One removed reviewer-audit test carried a genuine scientific claim. It was
**extracted into a standalone public test** rather than dropped:

```
tests/test_classical_selected_observable_baseline.py
  ::test_classical_baseline_has_repeated_ieee300_timings
```

It asserts the repeated IEEE300 selected-observable classical timing
evidence: the `sparse_factorized` and `adjoint_functional` methods are both
present for `ieee300`, `timing_repeats == 30`, and the runtime-quartile,
preprocessing-median, query-median, and solver-type columns exist.

## Manuscript-dependent node in the boundary test

`tests/test_ieee_qsvt_pipeline_boundary.py` lost exactly one node,
`test_manuscript_claim_boundary_audit_still_passes`, together with its
`quantum_contribution_audit` import. That node read `manuscript/main.tex`
and self-skipped when absent, so it could never do useful work in a public
checkout. The file's remaining eight tests are intact, including
`test_matched_alpha_qsvt_target_ridge_equivalence`, the matched-alpha
Ridge/QSVT-target equivalence check.

## Verification that the public suite works in a public checkout

The exact-tree run is not by itself proof that an external user can run the
suite: the staging directory held evidence that `.gitignore` would have
excluded from the commit. This was checked directly, by materializing only the
files Git would track into a separate directory and running the suite there.

| Clean-clone run | Tracked files | Failed | Errors | Passed | Skipped |
|---|---:|---:|---:|---:|---:|
| Before the fix | 1,238 | 181 | 97 | 1,389 | 35 |
| After widening the evidence whitelist | 1,556 | **0** | **0** | **1,670** | 32 |

The 278 initial failures spanned 77 test files and had a single cause: tests
assert against `outputs/` evidence that the original whitelist excluded, so the
files existed on the staging disk but would never have reached a clone. Two
smaller instances of the same cause were found and fixed the same way — the
reproduction validator failed 19 checks, and
`tests/test_qsvt_cost_accounting.py` failed on an empty
`signal_unitary_calls` field.

The fix was to narrow the `outputs/**` exclusion so the repository versions
exactly the evidence its own checks read — 41 roots, plus the QSP phase-angle
caches inside them and the five registered smoke-test snapshots. **No test was
weakened, skipped, or deleted to reach this result**, and no scientific file was
modified: the same assertions now run against the same bytes, in a checkout that
actually contains them.

## What the public suite still covers

Estimators and Ridge/Tikhonov behavior; the exact classical QSVT target and
its equivalence to Ridge at matched alpha; QSP/QSVT phase synthesis and
phase-response validation; block encoding, state preparation, and readout;
measurement generation for DC, AC, and nonlinear AC models; IEEE/PYPOWER
case construction; bad-data and robustness behavior; sweep and aggregation
logic; configuration validity; output schemas; and numerical regression
against the shipped compact evidence roots.
