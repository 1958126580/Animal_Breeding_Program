# ADR 0003: All-or-nothing outputs, run manifests and explicit statuses

Status: accepted (2026-09-25)

## Context

The spec forbids partial results that look complete, requires traceability
from any report number back to inputs, model, code and logs, and requires
cancellation, resume and disk-space checks.

## Decision

* A run writes into `<out>.partial-<run_id>/` and is renamed atomically to
  `<out>/` only after every step succeeds. On error the folder becomes
  `<out>.failed-<run_id>/`; on cancellation `<out>.cancelled-<run_id>/`. Both
  keep `run.log` and `manifest.json` with the error code.
* Every file is written to a temporary name and moved with `os.replace`.
* `manifest.json` records the spec (raw hash and effective values including
  defaults), input hashes, sample-mapping hash, code version, commit and
  uncommitted-patch hash, environment (OS, Python, NumPy, SciPy, BLAS, native
  kernel), backend, diagnostics, output hashes, wall time, peak memory and
  status (`passed`, `failed`, `cancelled`).
* `report.md` is rendered only from `results.json` and `manifest.json`.
* REML writes a checkpoint next to the output folder after each iteration.
  `--resume` accepts it only if the fingerprint (spec and input hashes)
  matches.
* Free disk space is checked before staging and before publishing.

## Consequences

* A folder with the requested name is always a complete, successful run.
* Failed runs remain inspectable without being mistaken for results.
