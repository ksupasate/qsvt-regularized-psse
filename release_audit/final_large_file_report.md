# Final Large-File Audit

Prepared: 2026-09-06 (regenerated in the permanent repository)

## Scope

The exact set of files staged for the initial commit, enumerated with
`git ls-files` in the permanent repository.

## Repository size

- Staged files: **1562**
- Total staged bytes: **119,959,047 (114.4 MiB)**
- Largest staged file: **20,331,973 bytes (19.39 MiB)** — `outputs/final_contribution_evidence/artifact_dependency_graph.json`

## Thresholds

| Threshold | Files | Requirement | Met |
|---|---:|---|:---:|
| > 100 MB | 0 | **0 required** (GitHub hard limit) | yes |
| > 50 MB | 0 | GitHub warns above this | yes |
| > 25 MB | 0 | informational | yes |
| > 20 MB | 0 | informational | yes |

**No staged file exceeds 20 MB, 25 MB, 50 MB, or 100 MB.** Git LFS is not
required. The tree is text- and CSV-dominated, so the packed size on GitHub
will be well below the working-tree figure above.

## 25 largest staged files

| # | Bytes | Size | Path |
|---:|---:|---:|---|
| 1 | 20,331,973 | 19.39 MiB | `outputs/final_contribution_evidence/artifact_dependency_graph.json` |
| 2 | 18,245,776 | 17.40 MiB | `outputs/output_aware_structural_generalization/support_registry.csv` |
| 3 | 10,301,547 | 9.82 MiB | `outputs/output_aware_structural_generalization/heldout_instance_summary.csv` |
| 4 | 9,441,472 | 9.00 MiB | `outputs/output_aware_generalization/support_registry.csv` |
| 5 | 7,466,999 | 7.12 MiB | `outputs/output_aware_structural_generalization/pareto_candidates_error_nnz.csv` |
| 6 | 7,466,428 | 7.12 MiB | `outputs/output_aware_structural_generalization/pareto_candidates_error_slots.csv` |
| 7 | 6,162,417 | 5.88 MiB | `outputs/output_aware_generalization/heldout_instance_summary.csv` |
| 8 | 3,756,574 | 3.58 MiB | `outputs/output_aware_generalization/pareto_candidates_error_gates.csv` |
| 9 | 3,484,598 | 3.32 MiB | `outputs/output_aware_generalization/pareto_candidates_error_nnz.csv` |
| 10 | 2,848,145 | 2.72 MiB | `outputs/output_aware_structural_generalization/resource_registry.csv` |
| 11 | 1,951,298 | 1.86 MiB | `outputs/output_aware_structural_generalization/pareto_frontier_error_slots.csv` |
| 12 | 1,736,427 | 1.66 MiB | `outputs/output_aware_structural_generalization/pareto_frontier_error_nnz.csv` |
| 13 | 1,494,795 | 1.43 MiB | `outputs/output_aware_generalization/resource_registry.csv` |
| 14 | 1,157,002 | 1.10 MiB | `outputs/final_contribution_evidence/canonical_limitation_registry.csv` |
| 15 | 1,096,724 | 1.05 MiB | `outputs/final_contribution_evidence/canonical_result_registry.csv` |
| 16 | 991,138 | 967.9 KiB | `outputs/final_contribution_evidence/near_zero_output_audit.csv` |
| 17 | 896,566 | 875.6 KiB | `outputs/output_aware_structural_generalization/structural_case_stratified_bootstrap.csv` |
| 18 | 861,679 | 841.5 KiB | `outputs/output_aware_structural_generalization/structural_group_bootstrap.csv` |
| 19 | 656,817 | 641.4 KiB | `release_audit/public_scientific_hash_verification.json` |
| 20 | 580,060 | 566.5 KiB | `outputs/qsvt_phase_validation_paper/approximation_error.csv` |
| 21 | 451,923 | 441.3 KiB | `outputs/output_aware_generalization/generalization_bootstrap.csv` |
| 22 | 441,515 | 431.2 KiB | `release_audit/public_export_mapping.csv` |
| 23 | 386,856 | 377.8 KiB | `outputs/output_aware_structural_generalization/residual_registry.csv` |
| 24 | 329,457 | 321.7 KiB | `outputs/nonlinear_closed_loop_qsvt/extended_horizon/iteration_ledgers/extended_iterations.csv` |
| 25 | 268,016 | 261.7 KiB | `outputs/output_aware_structural_generalization/support_stability.csv` |

## Archives

A recursive search of the whole permanent repository for `*.zip`, `*.tar`,
`*.tar.gz`, `*.tgz`, `*.7z`, and `*.rar` returns **zero** matches.

## The historical ~528 MB submission blob

**Absent**, on four independent grounds:

1. No archive of any kind exists anywhere in the repository.
2. The permanent repository was created by `rsync --exclude='.git'`, so no
   history object was copied from any source.
3. `git init` was run in a directory verified to contain no `.git`; the
   reflog is empty and there are zero commits.
4. The blob exists only in the development repository's history, where it
   was located and identified exactly:

   ```
   503.83 MiB (528.3 MB)  submission_package_tqe_final/tqe_reproducibility_supplement.zip
   ```

The configured remote is `git@github.com:ksupasate/qsvt-regularized-psse.git`
only. The development repository is not a remote, and no fetch, pull, or
merge from it has occurred, so the blob cannot enter this history later.
