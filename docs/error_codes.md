# Error codes and exit statuses

Generated from `abp.errors` (`abp errors` prints the same table). Codes are stable and never re-used.

| Code | Name | Exit status | Meaning | Remedy |
|---|---|---:|---|---|
| ABP-E100 | INPUT_NOT_FOUND | 3 | An input file does not exist or cannot be read. | Check the path in the analysis spec (relative paths are resolved against the spec file's folder). |
| ABP-E101 | INPUT_ENCODING | 3 | An input file is not valid UTF-8 text. | Re-save the file as UTF-8 (UTF-8 with BOM is accepted). |
| ABP-E102 | SCHEMA_MISSING_COLUMN | 3 | A column required by the data contract is missing. | Add the column or map it to an existing header in the spec. |
| ABP-E103 | SCHEMA_TYPE | 3 | A value cannot be parsed as the type declared in the data contract. | Correct the cell or declare its token as a missing-value code. |
| ABP-E104 | SPEC_INVALID | 3 | The analysis spec is malformed, has unknown keys, or has illegal values. | Fix the reported key; see docs/user_manual.md section 'Analysis spec'. |
| ABP-E105 | ID_INVALID | 3 | An identifier is empty or has leading/trailing whitespace. | Remove the whitespace; identifiers are compared as exact strings. |
| ABP-E106 | DUPLICATE_KEY | 3 | A primary key (record or marker ID) appears more than once. | Make the key unique or remove the duplicated row. |
| ABP-E107 | EMPTY_INPUT | 3 | An input table has a header but no data rows. | Provide at least one data row. |
| ABP-E200 | PEDIGREE_CYCLE | 4 | The pedigree contains a cycle (an animal is its own ancestor). | Correct the parent IDs of the listed animals. |
| ABP-E201 | PEDIGREE_SELF_PARENT | 4 | An animal is recorded as its own sire or dam. | Correct the parent IDs of the listed animals. |
| ABP-E202 | PEDIGREE_SEX_CONFLICT | 4 | An animal is used both as sire and dam, or its declared sex contradicts its parental role. | Correct the parent IDs or the sex column for the listed animals. |
| ABP-E203 | PEDIGREE_BIRTH_ORDER | 4 | A parent is born on or after the birth date of its offspring. | Correct the birth dates or the parent IDs of the listed animals. |
| ABP-E204 | PEDIGREE_CONFLICTING_DUPLICATE | 4 | An animal appears in several pedigree rows with different parents or sex. | Keep exactly one row per animal. |
| ABP-E210 | PHENOTYPE_UNKNOWN_ANIMAL | 4 | A phenotype record refers to an animal that is not in the pedigree. | Add the animal to the pedigree, fix the ID, or set qc.unknown_animals = "add_as_founder" if this is intended. |
| ABP-E211 | PHENOTYPE_OUT_OF_RANGE | 4 | A phenotype lies outside the valid range declared for the trait. | Correct the value, or set qc.out_of_range = "quarantine" to exclude such records (they are then listed in the QC report). |
| ABP-E212 | NO_USABLE_RECORDS | 4 | No phenotype record remains for the analysis after QC. | Check trait columns, missing-value codes and QC settings. |
| ABP-E220 | GENOTYPE_DOSAGE_RANGE | 4 | A genotype dosage is outside [0, ploidy]. | Check the allele coding; dosages count copies of the counted allele. |
| ABP-E221 | GENOTYPE_ZERO_SCALING | 4 | The genomic relationship scaling 2*sum(p*(1-p)) is zero (no polymorphic markers). | Provide polymorphic markers or review the marker filters. |
| ABP-E222 | GENOTYPE_ID_CONFLICT | 4 | Genotype and marker-map identifiers do not match. | Make genotype columns and marker map rows refer to the same markers. |
| ABP-E223 | GENOTYPE_ALLELE_MISMATCH | 4 | Counted allele / assembly information is missing or inconsistent. | Declare the counted allele and assembly for every marker. |
| ABP-E300 | MODEL_NOT_IDENTIFIABLE | 5 | The model or a requested quantity is not identifiable from the data. | Simplify the model or add connecting information; see the diagnostic. |
| ABP-E301 | COVARIANCE_NOT_PD | 5 | A covariance matrix or variance is not positive (semi)definite as required. | Provide valid (co)variances; correlations must lie in (-1, 1). |
| ABP-E302 | RELATIONSHIP_SINGULAR | 5 | A relationship matrix that must be inverted is singular. | Choose an explicit genomic.singular_policy (blend or ridge) and record why. |
| ABP-E303 | UNSUPPORTED_COMBINATION | 5 | The requested combination of model features is not supported. | See the method registry for supported combinations. |
| ABP-E400 | SOLVER_NOT_CONVERGED | 6 | The iterative solver did not reach the requested tolerance. | Increase solver.max_iter, use solver.method = "dense" or "sparse_direct", or check the model for near-singularity. |
| ABP-E401 | BACKWARD_ERROR_TOO_LARGE | 6 | The solution does not satisfy the mixed-model equations to the required accuracy. | The system is ill-conditioned; check the model and scaling. |
| ABP-E402 | RELIABILITY_OUT_OF_RANGE | 6 | A computed reliability lies clearly outside [0, 1]. | This signals a scale or model inconsistency; results are withheld. |
| ABP-E403 | REML_NOT_CONVERGED | 6 | REML did not converge within the iteration budget. | Increase reml.max_iter, change start values, or simplify the model. |
| ABP-E404 | FACTORIZATION_FAILED | 6 | A matrix that should be positive definite could not be factorized. | Check variance components (must be > 0) and fixed-effect dependencies. |
| ABP-E500 | RESOURCE_MEMORY | 7 | The requested computation exceeds the configured memory budget. | Raise resources.max_memory_gb or choose a sparse/iterative solver. |
| ABP-E501 | RESOURCE_DISK | 7 | Not enough free disk space to write the outputs safely. | Free disk space or choose another output folder. |
| ABP-E502 | BACKEND_UNAVAILABLE | 7 | The requested compute backend is not available. | Use backend.device = "cpu" or set backend.on_unavailable = "fallback_cpu". |
| ABP-E503 | OUTPUT_EXISTS | 2 | The output folder already contains a completed run. | Choose a new folder or pass --force to replace it atomically. |
| ABP-E504 | CHECKPOINT_MISMATCH | 2 | A checkpoint exists but belongs to different inputs or a different spec. | Remove the checkpoint or restore the original inputs. |
| ABP-E600 | CANCELLED | 130 | The run was cancelled by the user; no result was published. | Re-run, optionally with --resume if a checkpoint was written. |
| ABP-E900 | INTERNAL | 1 | An unexpected internal error occurred. | Report the log file (run.log) together with the run manifest. |

Exit statuses: 0 success, 1 internal error, 2 usage/output-folder problem, 3 input/contract error, 4 blocking QC finding, 5 model/identifiability problem, 6 numerical failure, 7 resource/platform limit, 130 cancelled.
