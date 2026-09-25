# Contracts

| File | What it defines | Kept in sync by |
|---|---|---|
| `analysis_spec.schema.json` | the TOML analysis spec (keys, types, allowed values, defaults) | generated from `abp.core.spec.SCHEMA` (`python -m abp.core.spec`); `tests/test_examples.py::test_published_json_schema_is_current` |
| `data_dictionary.json` | input tables (pedigree, phenotypes, marker map, genotypes, frequencies) and output files | reviewed with each change to `abp.io`, `abp.qc`, `abp.workflows` |
| `run_manifest.schema.json` | the run record written to `manifest.json` | `tests/test_manifest_contract.py` |

Cross-field rules (e.g. "variances must list exactly the random terms plus
residual") are enforced by `abp.core.spec.validate_spec_dict` and documented
in `docs/user_manual.md`, section 5.
