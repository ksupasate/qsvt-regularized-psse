# Data Access and Large-Artifact Policy

## Public Git policy

Source code, fixed configurations, compact machine-readable evidence, and
public-safe audit records belong in Git. Raw campaign tables or archives larger
than 100 MB do not. They should be deposited in Zenodo, Figshare, or another
durable research-data repository and linked by DOI from this document and the
release record.

Reserved Zenodo DOI for this reproducibility release: `10.5281/zenodo.22326883`.

The Zenodo record is currently an **unpublished draft**. The DOI is reserved and
is recorded in `CITATION.cff` and `data_manifest.json`, but it does not yet
resolve to a downloadable deposit, and the files below are therefore not yet
publicly archived. Do not substitute any other identifier for the reserved DOI.
The machine-readable inventory is [`data_manifest.json`](../data_manifest.json).

## Local large-file inventory

The following protected scientific outputs were present in the development
workspace during the 2026-09-04 release audit. They were not modified or deleted
by the repository-hygiene pass.

| Bytes | SHA-256 | Local artifact | Public distribution |
|---:|---|---|---|
| 946,761,562 | `7d56db12a8e5029198ddea11d2feb9cd48a7d5b9677779deb9b2ea47bea7ea93` | `outputs/output_aware_structural_generalization/heldout_results.csv` | External DOI archive |
| 582,168,411 | `d20f3203cd9667a5758e156ddfffeb87a7bea614c607171b8fe47097f96d8d39` | `outputs/output_aware_generalization/heldout_results.csv` | External DOI archive |
| 557,112,925 | `76376a8e8ff61795e6aee25a799d7756328b5991ddb14174d87c40f2a8f1dfec` | `outputs/output_aware_structural_generalization/certificate_results.csv` | External DOI archive |
| 552,347,335 | `848e1bd7c38558d44ba796582fcb78ab533f90e3877c58687e0db54607f8726a` | `outputs/tqe_physical_alignment_and_generalization/physical_audit/raw_physical_rows.csv` | External DOI archive |
| 302,075,046 | `1d5eb4f79ae07a5d6f77aa232a373891bfa0d3a817d63242afc6d69fc993b071` | `outputs/output_aware_generalization/certificate_results.csv` | External DOI archive |
| 273,021,505 | `86b5ebe71c7d34e72bd3ba859845d26cafe83fcece4c3acb80e5836ccc17009b` | `outputs/tqe_measurement_row_sparsification/raw_evaluation_rows.csv` | External DOI archive |

The checksums identify the audited source artifacts; they do not assert that an
external deposit already exists. `data_manifest.json` records the reserved DOI
`10.5281/zenodo.22326883` together with
`"archive_status": "reserved_doi_zenodo_draft_not_published"` and a per-dataset
`"doi_status": "reserved_not_yet_published"`; `retrieval_location` stays `null`
until the record is published. When the deposit is published, record the
concrete retrieval URL, version, license, file-level checksums, and relationship
to the Git release, and clear the pending-publication status fields.

## Retrieval and integrity workflow

Until the reserved record `10.5281/zenodo.22326883` is published, full
evidence-level reproduction is not self-contained. After publication, download each file using the location in
`data_manifest.json`, retain the expected archive filename, and verify its
SHA-256 digest before using it. A digest mismatch means the download must not be
treated as the audited evidence artifact.

## Regeneration provenance

The corresponding configurations and entry points are:

| Evidence family | Configuration | Entry point |
|---|---|---|
| Output-aware generalization | `configs/output_aware_generalization.json` | `scripts/run_output_aware_generalization.py` |
| Output-aware structural generalization | `configs/output_aware_structural_generalization.json` | `scripts/run_output_aware_structural_generalization.py` |
| Physical alignment/generalization | `configs/tqe_physical_alignment/campaign.json` | `scripts/run_tqe_physical_alignment.py` |
| Measurement-row sparsification | `configs/tqe_measurement_row_sparsification.yaml` | `scripts/run_tqe_measurement_row_sparsification.py` |

These are full scientific campaigns and are not invoked by
`scripts/reproduce_all.sh`. Consult each manifest/configuration for seeds,
selection rules, and output schema before launching a campaign.

## Deposit checklist

Before the reserved draft becomes a published, citable record:

1. Freeze the exact files and compute SHA-256 checksums.
2. Include configs, run manifests, environment metadata, and licenses.
3. State whether the deposit is raw evidence, a derived table, or both.
4. Verify that no credentials or machine-local absolute paths are present.
5. Record the Git commit and artifact version represented by the deposit.
6. Publish the reserved record `10.5281/zenodo.22326883` and then set
   `retrieval_location` and the publication status in `data_manifest.json`,
   `RELEASE.md`, and this document.
