# Generated reruns

This directory receives explicit researcher reruns. `scripts/run_experiment.py`
redirects here (under `outputs/generated/<run-id>/`) unless `--output-dir` is
supplied, and `scripts/reproduce_all.sh` writes its validation report here.

Nothing in this directory is canonical evidence. Files here are produced by the
current checkout on the current machine and are not tracked by Git.

A rerun becomes citable only once its resolved configuration, seed, environment,
source commit, and output checksums are recorded alongside it. Do not overwrite
a frozen evidence root under `outputs/` to store a rerun; see
[`../../RESULTS_INDEX.md`](../../RESULTS_INDEX.md) for which roots are versioned.
