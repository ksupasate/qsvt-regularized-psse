# DOI Metadata Alignment

Prepared: 2026-09-05

## Reserved identifier

```
10.5281/zenodo.22326883
```

This DOI was reserved by the author against a Zenodo **draft**. The record has
**not** been published. Nothing in this session published it, and no other
identifier was invented or substituted.

Status wording used throughout the repository:

> Reserved Zenodo DOI for this reproducibility release: `10.5281/zenodo.22326883`

The phrases "publicly archived on Zenodo", "available for download", and any
equivalent claim of an existing public deposit are deliberately **absent**.

## Every location containing the DOI

| # | File | Field / context | Value written | Kind |
|---:|---|---|---|---|
| 1 | `CITATION.cff` | `doi:` | `10.5281/zenodo.22326883` | machine-readable |
| 2 | `data_manifest.json` | `reserved_doi` (top level) | `10.5281/zenodo.22326883` | machine-readable |
| 3 | `data_manifest.json` | `datasets[0].doi` — `output_aware_structural_generalization_heldout_results` | `10.5281/zenodo.22326883` | machine-readable |
| 4 | `data_manifest.json` | `datasets[1].doi` — `output_aware_generalization_heldout_results` | `10.5281/zenodo.22326883` | machine-readable |
| 5 | `data_manifest.json` | `datasets[2].doi` — `output_aware_structural_generalization_certificate_results` | `10.5281/zenodo.22326883` | machine-readable |
| 6 | `data_manifest.json` | `datasets[3].doi` — `tqe_physical_alignment_raw_rows` | `10.5281/zenodo.22326883` | machine-readable |
| 7 | `data_manifest.json` | `datasets[4].doi` — `output_aware_generalization_certificate_results` | `10.5281/zenodo.22326883` | machine-readable |
| 8 | `data_manifest.json` | `datasets[5].doi` — `tqe_measurement_row_sparsification_raw_rows` | `10.5281/zenodo.22326883` | machine-readable |
| 9 | `README.md` | "Expected Outputs" — external-data pointer | reserved-DOI prose | prose |
| 10 | `README.md` | "Limitations" — raw-dataset bullet | reserved-DOI prose | prose |
| 11 | `README.md` | "Citation" | reserved-DOI prose | prose |
| 12 | `RELEASE.md` | header block | `Reserved Zenodo DOI for this reproducibility release:` | prose |
| 13 | `RELEASE.md` | opening scope paragraph | reserved-not-published statement | prose |
| 14 | `RELEASE.md` | "External data" | reserved-DOI prose | prose |
| 15 | `RELEASE.md` | "Known limitations" | reserved-but-unpublished | prose |
| 16 | `CHANGELOG.md` | `[Unreleased] → Added` | reserved-DOI entry | prose |
| 17 | `CHANGELOG.md` | `[Unreleased] → Known limitations` | reserved-but-unpublished | prose |
| 18 | `MANIFEST.md` | "Generated and external data" | reserved-DOI prose | prose |
| 19 | `RESULTS_INDEX.md` | "High-volume data" | reserved-DOI prose | prose |
| 20 | `docs/data_access.md` | "Public Git policy" | reserved-DOI + draft status | prose |
| 21 | `docs/data_access.md` | large-file inventory note | reserved DOI + `archive_status` explanation | prose |
| 22 | `docs/data_access.md` | "Retrieval and integrity workflow" | reserved-DOI prose | prose |
| 23 | `docs/data_access.md` | "Deposit checklist" item 6 | publish-the-reserved-record instruction | prose |
| 24 | `release_audit/metadata_completion_checklist.md` | DOI row + preamble + completion rule | marked **RESOLVED** | prose |

## Status fields added alongside the DOI

`data_manifest.json`:

```json
"archive_status": "reserved_doi_zenodo_draft_not_published",
"reserved_doi": "10.5281/zenodo.22326883",
"doi_status": "Reserved Zenodo DOI for this reproducibility release; the Zenodo record is still a draft and has not been published, so no file is downloadable yet."
```

and, per dataset, `"doi_status": "reserved_not_yet_published"`.

`retrieval_location` stays `null` for all six datasets: no download URL exists
while the record is a draft. Producing configurations, byte sizes, and SHA-256
digests were **not** altered.

## Placeholder status (updated 2026-09-06)

Every placeholder recorded here has since been resolved from author-supplied
values. Nothing was invented.

| File | Field | Final value |
|---|---|---|
| `CITATION.cff` | `authors` | Vorathammathorn, Supasate; Phassadawongse, Dhana; Turner, Stephen John |
| `CITATION.cff` | `authors[0].orcid` | `https://orcid.org/0009-0009-2751-1023` (only ORCID supplied) |
| `CITATION.cff` | `version` | `1.0.0` |
| `CITATION.cff` | `repository-code`, `url` | `https://github.com/ksupasate/qsvt-regularized-psse` |
| `RELEASE.md` | `Release version` | `1.0.0` |

Still deliberately absent: ORCIDs for Phassadawongse and Turner (not supplied),
`date-released` (the release has not happened), and any affiliation. No ORCID,
author name, GitHub owner, or release date was invented.

## Parse verification after the edits

- `CITATION.cff` parses as YAML; `doi` reads back as `10.5281/zenodo.22326883`.
- `data_manifest.json` parses as JSON; all six dataset DOIs read back correctly.
