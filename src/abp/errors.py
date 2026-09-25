"""Error codes, exit statuses and the single exception type raised by ABP.

Every failure that ABP detects deliberately is raised as :class:`ABPError`
carrying a stable, documented code (``ABP-Exxx``).  The command-line
interface maps the code to a process exit status so that scripts, schedulers
and the Windows launcher can react without parsing text.

Design rules
------------
* Codes are never re-used or renumbered; retired codes stay in the table.
* ``details`` holds machine-readable context (file, row, column, animal ID,
  numeric values).  Messages must say *what* is wrong, *where*, and *how to
  fix it*, without leaking unrelated internals.
* Scientific failures (non-convergence, illegal reliabilities, singular
  covariance) are errors, never warnings: a partial or doubtful result must
  not be published as if it were complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Process exit statuses (kept small and stable for shell scripting).
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_QC_BLOCKED = 4
EXIT_MODEL = 5
EXIT_NUMERICAL = 6
EXIT_RESOURCE = 7
EXIT_CANCELLED = 130


@dataclass(frozen=True)
class ErrorSpec:
    """Static description of one error code."""

    code: str
    name: str
    exit_status: int
    summary: str
    remedy: str


_SPECS: tuple[ErrorSpec, ...] = (
    # -- E1xx: input files, encodings and schema/spec validation -------------
    ErrorSpec("ABP-E100", "INPUT_NOT_FOUND", EXIT_INPUT,
              "An input file does not exist or cannot be read.",
              "Check the path in the analysis spec (relative paths are resolved "
              "against the spec file's folder)."),
    ErrorSpec("ABP-E101", "INPUT_ENCODING", EXIT_INPUT,
              "An input file is not valid UTF-8 text.",
              "Re-save the file as UTF-8 (UTF-8 with BOM is accepted)."),
    ErrorSpec("ABP-E102", "SCHEMA_MISSING_COLUMN", EXIT_INPUT,
              "A column required by the data contract is missing.",
              "Add the column or map it to an existing header in the spec."),
    ErrorSpec("ABP-E103", "SCHEMA_TYPE", EXIT_INPUT,
              "A value cannot be parsed as the type declared in the data contract.",
              "Correct the cell or declare its token as a missing-value code."),
    ErrorSpec("ABP-E104", "SPEC_INVALID", EXIT_INPUT,
              "The analysis spec is malformed, has unknown keys, or has illegal values.",
              "Fix the reported key; see docs/user_manual.md section 'Analysis spec'."),
    ErrorSpec("ABP-E105", "ID_INVALID", EXIT_INPUT,
              "An identifier is empty or has leading/trailing whitespace.",
              "Remove the whitespace; identifiers are compared as exact strings."),
    ErrorSpec("ABP-E106", "DUPLICATE_KEY", EXIT_INPUT,
              "A primary key (record or marker ID) appears more than once.",
              "Make the key unique or remove the duplicated row."),
    ErrorSpec("ABP-E107", "EMPTY_INPUT", EXIT_INPUT,
              "An input table has a header but no data rows.",
              "Provide at least one data row."),
    # -- E2xx: blocking quality-control findings -------------------------------
    ErrorSpec("ABP-E200", "PEDIGREE_CYCLE", EXIT_QC_BLOCKED,
              "The pedigree contains a cycle (an animal is its own ancestor).",
              "Correct the parent IDs of the listed animals."),
    ErrorSpec("ABP-E201", "PEDIGREE_SELF_PARENT", EXIT_QC_BLOCKED,
              "An animal is recorded as its own sire or dam.",
              "Correct the parent IDs of the listed animals."),
    ErrorSpec("ABP-E202", "PEDIGREE_SEX_CONFLICT", EXIT_QC_BLOCKED,
              "An animal is used both as sire and dam, or its declared sex contradicts its parental role.",
              "Correct the parent IDs or the sex column for the listed animals."),
    ErrorSpec("ABP-E203", "PEDIGREE_BIRTH_ORDER", EXIT_QC_BLOCKED,
              "A parent is born on or after the birth date of its offspring.",
              "Correct the birth dates or the parent IDs of the listed animals."),
    ErrorSpec("ABP-E204", "PEDIGREE_CONFLICTING_DUPLICATE", EXIT_QC_BLOCKED,
              "An animal appears in several pedigree rows with different parents or sex.",
              "Keep exactly one row per animal."),
    ErrorSpec("ABP-E210", "PHENOTYPE_UNKNOWN_ANIMAL", EXIT_QC_BLOCKED,
              "A phenotype record refers to an animal that is not in the pedigree.",
              "Add the animal to the pedigree, fix the ID, or set "
              "qc.unknown_animals = \"add_as_founder\" if this is intended."),
    ErrorSpec("ABP-E211", "PHENOTYPE_OUT_OF_RANGE", EXIT_QC_BLOCKED,
              "A phenotype lies outside the valid range declared for the trait.",
              "Correct the value, or set qc.out_of_range = \"quarantine\" to exclude "
              "such records (they are then listed in the QC report)."),
    ErrorSpec("ABP-E212", "NO_USABLE_RECORDS", EXIT_QC_BLOCKED,
              "No phenotype record remains for the analysis after QC.",
              "Check trait columns, missing-value codes and QC settings."),
    ErrorSpec("ABP-E220", "GENOTYPE_DOSAGE_RANGE", EXIT_QC_BLOCKED,
              "A genotype dosage is outside [0, ploidy].",
              "Check the allele coding; dosages count copies of the counted allele."),
    ErrorSpec("ABP-E221", "GENOTYPE_ZERO_SCALING", EXIT_QC_BLOCKED,
              "The genomic relationship scaling 2*sum(p*(1-p)) is zero (no polymorphic markers).",
              "Provide polymorphic markers or review the marker filters."),
    ErrorSpec("ABP-E222", "GENOTYPE_ID_CONFLICT", EXIT_QC_BLOCKED,
              "Genotype and marker-map identifiers do not match.",
              "Make genotype columns and marker map rows refer to the same markers."),
    ErrorSpec("ABP-E223", "GENOTYPE_ALLELE_MISMATCH", EXIT_QC_BLOCKED,
              "Counted allele / assembly information is missing or inconsistent.",
              "Declare the counted allele and assembly for every marker."),
    # -- E3xx: model definition and identifiability ---------------------------
    ErrorSpec("ABP-E300", "MODEL_NOT_IDENTIFIABLE", EXIT_MODEL,
              "The model or a requested quantity is not identifiable from the data.",
              "Simplify the model or add connecting information; see the diagnostic."),
    ErrorSpec("ABP-E301", "COVARIANCE_NOT_PD", EXIT_MODEL,
              "A covariance matrix or variance is not positive (semi)definite as required.",
              "Provide valid (co)variances; correlations must lie in (-1, 1)."),
    ErrorSpec("ABP-E302", "RELATIONSHIP_SINGULAR", EXIT_MODEL,
              "A relationship matrix that must be inverted is singular.",
              "Choose an explicit genomic.singular_policy (blend or ridge) and record why."),
    ErrorSpec("ABP-E303", "UNSUPPORTED_COMBINATION", EXIT_MODEL,
              "The requested combination of model features is not supported.",
              "See the method registry for supported combinations."),
    # -- E4xx: numerical failures ---------------------------------------------
    ErrorSpec("ABP-E400", "SOLVER_NOT_CONVERGED", EXIT_NUMERICAL,
              "The iterative solver did not reach the requested tolerance.",
              "Increase solver.max_iter, use solver.method = \"dense\" or "
              "\"sparse_direct\", or check the model for near-singularity."),
    ErrorSpec("ABP-E401", "BACKWARD_ERROR_TOO_LARGE", EXIT_NUMERICAL,
              "The solution does not satisfy the mixed-model equations to the required accuracy.",
              "The system is ill-conditioned; check the model and scaling."),
    ErrorSpec("ABP-E402", "RELIABILITY_OUT_OF_RANGE", EXIT_NUMERICAL,
              "A computed reliability lies clearly outside [0, 1].",
              "This signals a scale or model inconsistency; results are withheld."),
    ErrorSpec("ABP-E403", "REML_NOT_CONVERGED", EXIT_NUMERICAL,
              "REML did not converge within the iteration budget.",
              "Increase reml.max_iter, change start values, or simplify the model."),
    ErrorSpec("ABP-E405", "MCMC_NOT_CONVERGED", EXIT_NUMERICAL,
              "MCMC diagnostics (R-hat, bulk/tail ESS) did not pass within the iteration budget.",
              "Increase bayes.max_iterations or thin, simplify the model, or review the "
              "diagnostics in the failed run folder; results are withheld."),
    ErrorSpec("ABP-E404", "FACTORIZATION_FAILED", EXIT_NUMERICAL,
              "A matrix that should be positive definite could not be factorized.",
              "Check variance components (must be > 0) and fixed-effect dependencies."),
    # -- E5xx: resources and platform -----------------------------------------
    ErrorSpec("ABP-E500", "RESOURCE_MEMORY", EXIT_RESOURCE,
              "The requested computation exceeds the configured memory budget.",
              "Raise resources.max_memory_gb or choose a sparse/iterative solver."),
    ErrorSpec("ABP-E501", "RESOURCE_DISK", EXIT_RESOURCE,
              "Not enough free disk space to write the outputs safely.",
              "Free disk space or choose another output folder."),
    ErrorSpec("ABP-E502", "BACKEND_UNAVAILABLE", EXIT_RESOURCE,
              "The requested compute backend is not available.",
              "Use backend.device = \"cpu\" or set backend.on_unavailable = \"fallback_cpu\"."),
    ErrorSpec("ABP-E503", "OUTPUT_EXISTS", EXIT_USAGE,
              "The output folder already contains a completed run.",
              "Choose a new folder or pass --force to replace it atomically."),
    ErrorSpec("ABP-E504", "CHECKPOINT_MISMATCH", EXIT_USAGE,
              "A checkpoint exists but belongs to different inputs or a different spec.",
              "Remove the checkpoint or restore the original inputs."),
    # -- E6xx / E9xx: lifecycle and internal -----------------------------------
    ErrorSpec("ABP-E600", "CANCELLED", EXIT_CANCELLED,
              "The run was cancelled by the user; no result was published.",
              "Re-run, optionally with --resume if a checkpoint was written."),
    ErrorSpec("ABP-E900", "INTERNAL", EXIT_INTERNAL,
              "An unexpected internal error occurred.",
              "Report the log file (run.log) together with the run manifest."),
)

ERROR_SPECS: dict[str, ErrorSpec] = {s.code: s for s in _SPECS}
_BY_NAME: dict[str, ErrorSpec] = {s.name: s for s in _SPECS}


class ABPError(Exception):
    """The one exception type ABP raises for detected, documented failures.

    Parameters
    ----------
    name:
        Symbolic error name (e.g. ``"PEDIGREE_CYCLE"``) or code (``"ABP-E200"``).
    message:
        Human-readable explanation, specific to this occurrence.
    **details:
        Machine-readable context written to the run manifest and log.
    """

    def __init__(self, name: str, message: str, **details: Any) -> None:
        spec = _BY_NAME.get(name) or ERROR_SPECS.get(name)
        if spec is None:  # programming error: unknown code
            raise KeyError(f"unknown ABP error name or code: {name!r}")
        self.spec = spec
        self.message = message
        self.details = details
        super().__init__(f"[{spec.code} {spec.name}] {message}")

    @property
    def code(self) -> str:
        return self.spec.code

    @property
    def exit_status(self) -> int:
        return self.spec.exit_status

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for manifests and JSON logs."""
        return {
            "code": self.spec.code,
            "name": self.spec.name,
            "message": self.message,
            "remedy": self.spec.remedy,
            "details": _jsonable(self.details),
        }


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars / tuples / sets into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def markdown_table() -> str:
    """The error-code reference (``docs/error_codes.md``), generated from the registry."""
    lines = ["# Error codes and exit statuses", "",
             "Generated from `abp.errors` (`python -m abp.errors docs/error_codes.md`; "
             "`abp errors` prints the same table). Codes are stable and never re-used.", "",
             "| Code | Name | Exit status | Meaning | Remedy |", "|---|---|---:|---|---|"]
    for s in sorted(_SPECS, key=lambda e: e.code):
        lines.append(f"| {s.code} | {s.name} | {s.exit_status} | {s.summary} | {s.remedy} |")
    lines += ["", "Exit statuses: 0 success, 1 internal error, 2 usage/output-folder problem, "
                  "3 input/contract error, 4 blocking QC finding, 5 model/identifiability "
                  "problem, 6 numerical failure, 7 resource/platform limit, 130 cancelled.", ""]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - documentation generator
    import sys
    from pathlib import Path
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(markdown_table(), encoding="utf-8")
        print(f"wrote {sys.argv[1]}")
    else:
        print(markdown_table())
