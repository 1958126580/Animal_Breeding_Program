# ABP User Manual

Version 0.2.0 · 2026-09-25

This manual is for breeders, geneticists and evaluation-centre staff who run
ABP. It covers installation, data preparation, every analysis-spec key, the
analyses themselves, the output files and troubleshooting. The mathematics
is in [`methods.md`](methods.md) and the Python library in [`api.md`](api.md).

Contents

1. [Concepts you need before you start](#1-concepts-you-need-before-you-start)
2. [Installation](#2-installation)
3. [Quick start: your first evaluation](#3-quick-start-your-first-evaluation)
4. [Preparing your data](#4-preparing-your-data)
5. [The analysis spec (reference)](#5-the-analysis-spec-reference)
6. [Quality control](#6-quality-control)
7. [Analyses](#7-analyses)
8. [Solvers, memory and run time](#8-solvers-memory-and-run-time)
9. [Output files](#9-output-files)
10. [Reproducibility, traceability and recovery](#10-reproducibility-traceability-and-recovery)
11. [Command reference](#11-command-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations and good practice](#13-limitations-and-good-practice)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. Concepts you need before you start

### 1.1 What ABP estimates

ABP estimates **additive breeding values (EBVs)**: the part of an
animal's genetic merit that is passed on to its offspring. On average an
offspring receives half of each parent's breeding value. An EBV is:

* **not** a prediction of the animal's own next phenotype. That would also
  include fixed and environmental effects.
* **not** a total genetic value. That would also include dominance and
  epistasis, which ABP does not model.
* always expressed **relative to a genetic base**. For a pedigree model the
  base is the founders: animals whose parents are unknown are treated as
  unrelated, non-inbred and with mean EBV 0. When unknown parents are
  assigned to *genetic groups* (§7.9), the base is formed by the ungrouped
  unknown parents and each group has its own estimated level.

Every analysis spec must state this explicitly (`analysis.task`,
`analysis.genetic_base`, `analysis.target_population`,
`analysis.information_cutoff`). The values are copied into the report and
the run manifest, so a result can never be separated from its definition.

### 1.2 Reliability, PEV, SEP and accuracy

For each animal ABP reports:

| Quantity | Meaning |
|---|---|
| `pev` | prediction error variance, `Var(EBV - true BV)`, in squared trait units |
| `sep` | standard error of prediction, `sqrt(pev)`, in trait units |
| `reliability` | `1 - pev / (genetic variance of that animal)`, between 0 and 1 |
| `accuracy` | `sqrt(reliability)`: the model-based correlation between EBV and true BV |

These are **model-based** quantities. They are correct if the model and the
variance components are correct. They do not account for errors in the
variance components themselves. They are not a validated measure of
prediction accuracy on your population. A reliability of 0.35 means that
roughly a third of the genetic variation among animals like this one is
captured by the EBV. Such animals can re-rank considerably as information
accumulates. Forward-in-time validation (§7.11) checks the predictions
against later data.

Two analyses report something different and say so in every output:

* **Bayesian marker models** (§7.10) report the posterior mean
  (`gebv_posterior_mean`) and the posterior standard deviation
  (`gebv_posterior_sd`) instead of PEV and reliability.
* With **fixed** genetic groups (§7.9) the prior variance of an EBV is not
  defined, so reliabilities are left empty; PEV and SEP are reported.

### 1.3 Synthetic data

ABP ships a **synthetic** sheep flock for learning and testing
(`examples/sheep_data`). Every number in it is simulated, and every result
from it carries a "SYNTHETIC DATA" banner. The economic weights in the
examples are placeholders, not industry values.

---

## 2. Installation

### 2.1 Requirements

* Python **3.11 or newer** (64-bit) on Linux or Windows.
* NumPy and SciPy, which pip installs automatically.
* Optional: a C++20 compiler (GCC ≥ 10, Clang ≥ 12, or Visual Studio 2019
  16.11+/2022) to build the fast inbreeding kernel. Without one ABP still
  works, but inbreeding on large, deep pedigrees is about 10× slower (see
  [benchmarks](benchmarks.md)).
* Memory: small analyses need well under 1 GB. The dense solver needs about
  `24 × N²` bytes for `N` equations when PEV or REML is requested (see §8).

### 2.2 Linux

```bash
git clone https://github.com/1958126580/Animal_Breeding_Program.git
cd Animal_Breeding_Program
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[test]"
abp selftest
```

### 2.3 Windows

1. Install Python 3.11+ from python.org and tick "Add python.exe to PATH".
2. Optional, for the fast kernel: install "Build Tools for Visual Studio"
   with the "Desktop development with C++" workload.
3. In PowerShell:

```powershell
git clone https://github.com/1958126580/Animal_Breeding_Program.git
cd Animal_Breeding_Program
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
abp selftest
```

Use the launcher `packaging\windows\abp.ps1` (PowerShell) or
`packaging\windows\abp.cmd` (Command Prompt) for day-to-day runs. The
launcher:

* switches the console to UTF-8, so Chinese characters and spaces in folder
  names, file names and animal IDs work;
* streams ABP's messages to the screen and also saves them to
  `%LOCALAPPDATA%\ABP\logs\abp-<time>.log`;
* returns ABP's exit status unchanged, so batch files and schedulers can
  react to it (§11.2).

```powershell
.\packaging\windows\abp.ps1 run "D:\育种 数据\analysis.toml" --out "D:\育种 数据\结果 2025"
```

Set `ABP_PYTHON` to use a specific interpreter, for example the one in a
virtual environment. The Linux equivalent is `packaging/linux/abp.sh`, which
logs to `~/.local/state/abp/logs`.

### 2.4 Reproducible or offline installation

`requirements-lock.txt` pins every dependency with SHA-256 hashes.

```bash
# on a machine with internet access
python -m pip download --require-hashes -r requirements-lock.txt -d wheels
# copy the repository and the wheels folder to the offline machine, then
python -m pip install --no-index --find-links wheels --require-hashes -r requirements-lock.txt
python -m pip install --no-index --no-deps -e .
abp selftest
```

### 2.5 Verifying an installation

`abp selftest` runs the installed build on analytical cases whose answers
were derived by hand or come from an independent implementation: minor
allele frequency, a selection index, inbreeding and the inverse relationship
matrix, BLUP and PEV, a REML derivative, the single-step identity, the
textbook example of Mrode (2005), PLINK byte decoding, the inverse
relationship matrix with genetic groups, a closed-form optimal-contribution
problem, MCMC diagnostics against ArviZ reference values, and (if compiled)
the C++ Bayesian sweep against the Python reference. It must end with
`RESULT: PASS`. It also reports whether the native C++ kernel is in use.

```
ABP 0.2.0 self-test (native kernel: True)
  [PASS] T01 MAF
  [PASS] T02 index b = [3/7, 2/7], reliability 12/35
  [PASS] T03 F5 = 0.25, A row 5, exact A-inverse - kernel native_cpp_meuwissen_luo
  [PASS] T04 intercept 3, EBV -/+0.5, PEV 0.75, reliability 0.25
  [PASS] T05 REML genetic score = -0.01463020355
  [PASS] T06 H-inverse = A-inverse when G* = A22
  [PASS] Mrode Ex. 3.1 sex and animal solutions (+-6e-4)
  [PASS] T07 PLINK bytes 78 00 2F 01 -> A1 dosages
  [PASS] T08 group A*-inverse and gene fractions (Quaas 1988)
  [PASS] T09 OCS closed form a = (1 + sqrt(16C - 2))/4
  [PASS] T10 R-hat, bulk and tail ESS = reference values
  [PASS] T11 native Bayesian sweep = Python reference
RESULT: PASS
```

To force the pure-Python kernels (for troubleshooting), set the environment
variable `ABP_DISABLE_NATIVE=1`.

### 2.6 Upgrading and uninstalling

Upgrade with `git pull` followed by `python -m pip install -e .`, then run
`abp selftest` again. Uninstall with `python -m pip uninstall abp-breeding`.
Output folders are plain files and are never modified by an upgrade. The
manifest of every run records the ABP version and commit that produced it.

---

## 3. Quick start: your first evaluation

### 3.1 Run the textbook example

```bash
abp run examples/01_textbook_mrode_3_1/analysis.toml --out runs/ex01
```

```
... INFO pedigree QC passed: 8 animals, max generation 2, mean F 0.0000 (native_cpp_meuwissen_luo kernel)
... INFO phenotype QC passed: 5 records used, 0 excluded (listed in qc_excluded_records.csv)
... INFO wwg: solved 10 equations with dense (requested explicitly in spec); relative residual 1.88e-16; 0.00 s
status: passed
outputs: .../runs/ex01
report: .../runs/ex01/report.md
```

The folder `runs/ex01` now holds `report.md`, `ebv_wwg.csv`,
`fixed_effects_wwg.csv`, three QC files, `results.json`, `manifest.json` and
`run.log`. The EBVs reproduce the textbook solutions (for example animal 8:
0.183, animal 7: −0.249).

### 3.2 A realistic run: REML and BLUP on a flock

```bash
abp run examples/02_sheep_wwt_reml/analysis.toml --out runs/ex02
```

The report begins with what was estimated, then shows the variance
components and the best animals:

```
Variance components (reml): animal = 3.5523, residual = 12.3597; heritability h2 = 0.223 (approx. SE 0.047).

REML: status converged, 7 iterations, log-likelihood -3524.164931. ...

| Rank | Animal | Sex | EBV | Reliability | SEP | Own records |
|---:|---|---|---:|---:|---:|---:|
| 1 | M2401121 | M | 5.203 | 0.481 | 1.358 | 1 |
| 2 | W2501845 | F | 4.926 | 0.367 | 1.639 | 1 |
```

Three weaning weights in the synthetic data were typed with a misplaced
decimal point (for example 277.6 kg). The spec declares the valid range
5–80 kg and chooses `qc.out_of_range = "quarantine"`. Those three records
are therefore excluded and listed in `qc_excluded_records.csv`:

```
reason,line,record_id,animal,trait,value,column
PHE-RANGE,462,line462,W2200461,wwt,277.6,
```

Remove the `[qc]` section and the same run stops with exit status 4 and an
error naming each offending line. ABP never drops data without telling you.

### 3.3 Where to go next

| You want to… | Read |
|---|---|
| prepare your own files | §4 |
| write your own analysis spec | §5 and the examples in `examples/` |
| use genotypes (CSV or PLINK files) | §4.3, §7.4–7.5 |
| fit BayesA/B/C/Cπ/R or Bayesian ridge regression | §7.10 |
| handle purchased animals or imports without ancestry | §7.9 (genetic groups) |
| combine traits into an economic index | §7.6–7.7 |
| check predictions forward in time | §7.11 |
| choose parents and plan matings | §7.12 |
| understand an error message | §12 and [`error_codes.md`](error_codes.md) |

---

## 4. Preparing your data

All inputs are **UTF-8 delimited text files** with a header line. Excel's
"CSV UTF-8" format is accepted, including its byte-order mark. Other
encodings (for example GBK, or Windows-1252 as used by "CSV (Comma
delimited)" on some systems) are rejected with `ABP-E101`, because guessing
an encoding can corrupt IDs. Identifiers are compared as exact strings.
`A12` and `a12` are different animals, and IDs with leading or trailing
spaces are rejected (`ABP-E105`). The complete machine-readable contract is
`contracts/data_dictionary.json`.

### 4.1 Pedigree

```
id,sire,dam,sex,birth_date
R0001,0,0,M,2018-04-10
E0001,0,0,F,2018-04-01
W2100001,R0001,E0001,F,2021-04-12
```

| Column | Required | Content |
|---|---|---|
| `id` | yes | animal identifier |
| `sire`, `dam` | yes | parent IDs, or an unknown-parent code (default `0`, empty, `NA`, `.`) |
| `sex` | no | `M`, `F` or `U` (lower case accepted) |
| `birth_date` | no | `YYYY`, `YYYY-MM` or `YYYY-MM-DD` |

Column names can differ. Map them in `[data.pedigree_columns]`. Rows may be
in any order: ABP sorts parents before offspring. Parents that have no row of
their own are added as founders and listed in the QC report.

**Genetic-group codes.** If some unknown parents belong to identifiable
populations (for example rams bought from a particular breeder in a
particular period), write a group code instead of `0` in the parent field
and declare the code prefix in `[upg]` (§5.13, §7.9):

```
id,sire,dam,sex,birth_year
P16A1,UPG:A_16_19,UPG:A_16_19,M,2014
E11001,0,0,F,2011
```

A code starting with the prefix is **not** an animal: it names a group of
unknown parents. Animal IDs must not start with the prefix (`PED-UPG-ID`).
Without an `[upg]` section such codes would be read as ordinary parent IDs.

### 4.2 Phenotypes

One row per **record**: one animal measured on one occasion. A row can hold
several traits. The fixed and random classification variables sit in the
same row.

```
id,flock,year,sex,birth_type,dam_age,cg,wwt,fat,fec,nlb1,lambing1_year
W2100001,F1,2021,F,1,2-5,F1-2021-F,37.17,3.38,5.441,2,F1-2022
W2100002,F2,2021,F,2,2-5,F2-2021-F,31.85,NA,NA,2,F2-2022
```

* A missing-value code (default empty, `NA`, `.`) means **not measured**.
  A real zero must be written as `0`.
* Repeated records of an animal (for example one litter size per lambing)
  are allowed only if the model contains a permanent-environment term on the
  animal ID (§7.2). Otherwise ABP stops, because treating repeated records as
  independent would overstate the information.
* A record counts as used for a trait when it has a value for that trait.
  Every classification variable the model applies to that trait must then be
  present.
* Forward validation (§7.11) needs the date of each record: map a date
  column (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`) in
  `[data.phenotype_columns] date = "..."`.

### 4.3 Genotypes and marker map

Marker map (`markers.csv`):

```
marker_id,chrom,pos,ref,alt,counted_allele,assembly
snp00001,1,1234567,A,G,G,SYNTHETIC-OVINE-ASSEMBLY-0
```

`counted_allele` must be `ref` or `alt`. The genotype file holds the number
of copies of this allele. The whole file must use one `assembly`. Markers
whose two alleles are A/T or C/G are flagged for review, because their strand
cannot be checked from the allele names.

Genotypes (`genotypes.csv`, ABP dosage-matrix format):

```
id,snp00001,snp00002,snp00003
M2400901,0,1,2
W2400902,1,NA,2
```

* The first column is `id`. The remaining header must contain exactly the
  markers of the map, in any order.
* Values are 0, 1 or 2. Decimals between 0 and 2 are accepted as imputed
  dosages. Missing-value codes are allowed.

**PLINK 1 binary files.** Instead of `genotypes` + `marker_map`, give the
prefix of a PLINK 1 fileset (`.bed`, `.bim`, `.fam`) and the assembly of its
positions:

```toml
[data]
plink = "geno/flock2025"                 # reads flock2025.bed/.bim/.fam
genotype_assembly = "Oar_rambouillet_v1.0"
```

* Only the standard SNP-major `.bed` layout is accepted (the obsolete
  individual-major layout is refused).
* The **counted allele is A1** of the `.bim` file, exactly the allele that
  `plink --recode A` counts. A `.bim` file does not say which allele is the
  reference and carries no assembly, so ABP records ref/alt as unknown and
  requires `genotype_assembly`; it never guesses either.
* The individual ID (`IID`) of the `.fam` file is the animal ID; family IDs
  are ignored. Duplicated IIDs or variant IDs, and variants whose two alleles
  are identical, stop the run; strand-ambiguous A/T and C/G variants are
  flagged for review.
* VCF and BGEN files are not read directly. Convert them to PLINK 1 binary
  or to the dosage format above with a trusted tool, keeping the counted
  allele explicit.

### 4.4 Allele frequencies (optional)

When `genomic.frequency_source = "file"`, supply a CSV with columns
`marker_id,frequency`, giving the frequency of the counted allele in your
chosen reference population. Frequencies define the genomic base, so record
where they came from.

---

## 5. The analysis spec (reference)

An analysis is described by one **TOML** file (`analysis.toml`). ABP validates
it completely before touching any data:

* **unknown keys are errors** (`ABP-E104`), so a typo such as `methd` cannot
  silently fall back to a default;
* missing required keys, wrong types and values outside the allowed list are
  errors;
* all defaults are filled in and written to `manifest.json` → `spec.effective`.

Relative paths are resolved against the folder of the spec file. The
machine-readable schema is `contracts/analysis_spec.schema.json`.

### 5.1 Top level

| Key | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | must be `"1"` |

### 5.2 `[project]`

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | string | yes | | short name shown in the report |
| `species` | string | yes | | free text (sheep, cattle, pig, chicken, …) |
| `synthetic_data` | bool | yes | | declare `true` for simulated data; adds a banner to the report |
| `description` | string | no | `""` | |

### 5.3 `[analysis]`

| Key | Type | Required | Notes |
|---|---|---|---|
| `task` | string | yes | `additive_ebv` (implemented); `phenotype_prediction`, `total_genetic_value` and `mating_utility` are reserved and rejected in this version |
| `target_population` | string | yes | the population the results apply to |
| `information_cutoff` | date | yes | `YYYY-MM-DD`: the latest date of information used |
| `genetic_base` | string | yes | how the base is defined (for example "founders with unknown parents") |
| `applicability` | string | no | limits of use, shown in the report |

### 5.4 `[data]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `pedigree` | path | none | required for `pedigree` and `single_step` relationships |
| `phenotypes` | path | **required** | |
| `genotypes`, `marker_map` | path | none | required for `genomic` and `single_step` (or use `plink`) |
| `plink` | path prefix | none | PLINK 1 fileset `<prefix>.bed/.bim/.fam` (§4.3); counted allele = A1 |
| `genotype_assembly` | string | none | required with `plink` (a `.bim` file has no assembly) |
| `allele_frequencies` | path | none | required when `genomic.frequency_source = "file"` |
| `delimiter` | 1 character | `","` | use `"\t"` for tab-separated files |
| `missing_values` | list | `["", "NA", "."]` | codes meaning "not measured" |
| `unknown_parent_values` | list | `["0", "", "NA", "."]` | codes meaning "parent unknown" |

`[data.pedigree_columns]`: `id` (default `"id"`), `sire` (`"sire"`),
`dam` (`"dam"`), `sex` (none), `birth_date` (none).
`[data.phenotype_columns]`: `id` (default `"id"`), `record_id` (none; if
given, values must be unique), `date` (none; record dates, required by
`[validation]`).

### 5.5 `[[traits]]` (one block per trait)

| Key | Type | Required | Notes |
|---|---|---|---|
| `name` | identifier | yes | letters, digits, `_`; starts with a letter |
| `column` | string | no | column in the phenotype file (default: `name`) |
| `unit` | string | yes | shown with every EBV (for example `kg`) |
| `min`, `max` | number | no | inclusive valid range; see `qc.out_of_range` |
| `description` | string | no | |

### 5.6 `[model]`

| Key | Type | Default | Notes |
|---|---|---|---|
| `traits` | list | **required** | one trait = single-trait model; several = multi-trait model |
| `intercept` | bool | `true` | overall mean |
| `fixed` | array of tables | `[]` | `{ column = "...", type = "factor" or "covariate", traits = [...] }`; `traits` restricts a term to some traits (default: all) |
| `random` | array of tables | **required** | exactly one `kind = "additive"` term with `relationship = "pedigree" \| "genomic" \| "single_step"`; optional `kind = "iid"` terms with `column` (default: the animal id column) |

Example:

```toml
[model]
traits = ["wwt"]
fixed = [
  { column = "cg", type = "factor" },
  { column = "birth_type", type = "factor" },
]
random = [
  { name = "animal", kind = "additive", relationship = "pedigree" },
  { name = "pe", kind = "iid" },           # permanent environment of the animal
]
```

Covariates enter linearly and unchanged. Centre them in the data file if
you want the intercept to refer to the mean covariate.

### 5.7 `[variances]` and `[reml]`

| Key | Notes |
|---|---|
| `variances.mode` | `"known"`, `"reml"` or `"bayes"` (Bayesian marker models: variances are sampled, see §5.14) |
| `variances.values` | required for `known`: a variance for every random term and `residual`. Multi-trait models need square covariance matrices ordered like `model.traits`. |

```toml
[variances]
mode = "known"
values = { animal = 20.0, residual = 40.0 }
```

`[reml]` (used with `mode = "reml"`; single-trait models only):

| Key | Default | Notes |
|---|---|---|
| `algorithm` | `"ai"` | average information with EM fallback; `"em"` = EM only (slow near zero variances) |
| `max_iter` | 200 | exceeding it is an error (`ABP-E403`); no result is issued |
| `tol` | 1e-8 | both the relative parameter change and the Newton decrement must fall below it |
| `start` | data-based | `{ animal = ..., residual = ... }`; default: the OLS residual variance split 1/3 genetic, 1/6 per other term |

### 5.8 `[solver]`

| Key | Default | Notes |
|---|---|---|
| `method` | `"auto"` | `auto`, `dense`, `sparse_direct` or `pcg` (§8) |
| `tol` | 1e-10 | PCG convergence: true relative residual |
| `max_iter` | 10000 | PCG iteration limit |
| `pev` | `"exact"` | `"none"` skips PEV and reliabilities (needed for very large models) |

### 5.9 `[qc]`

| Key | Default | Alternatives |
|---|---|---|
| `unknown_animals` | `"error"` | `"add_as_founder"`: animals with records but no pedigree row become founders (listed) |
| `out_of_range` | `"error"` | `"quarantine"`: out-of-range values are excluded and listed |
| `missing_fixed` | `"error"` | `"exclude"`: used records lacking a classification value are excluded and listed |
| `outlier_sd` | 4.0 | values beyond this many SD are flagged for review. They are never removed by this rule. |
| `ungenotyped_records` | `"error"` | GBLUP only. `"exclude"` analyses records of genotyped animals only (listed). |

### 5.10 `[genomic]`

| Key | Default | Notes |
|---|---|---|
| `ploidy` | 2 | only diploid is supported |
| `frequency_source` | `"genotyped_all"` | `genotyped_all`: all QC-passed genotyped animals, including unphenotyped candidates. This is recorded as *transductive* use of candidate genotypes; candidate phenotypes are never used. `training_genotyped`: genotyped animals with records. `file`: see §4.4. `fixed_0.5`: all frequencies set to 0.5. |
| `min_call_rate_marker` | 0.90 | markers below it are excluded (listed) |
| `min_call_rate_animal` | 0.90 | animals below it are excluded (listed) |
| `min_maf` | 0.0 | a value > 0 excludes markers with MAF below it; MAF < 0.01 is always flagged for review |
| `singular_policy` | `"error"` | `blend`: G* = (1−α)G + αA22 (needs a pedigree); `ridge`: G* = G + λI |
| `blend_alpha` | 0.05 | α for `blend` |
| `ridge` | 0.01 | λ for `ridge` |
| `tuning` | `"none"` | `match_a22`: rescale G so its mean diagonal and off-diagonal equal those of A22 |

### 5.11 `[index]` (optional)

| Key | Required | Notes |
|---|---|---|
| `weights` | yes | `{ trait = weight, ... }` per unit of the trait; traits left out get weight 0 |
| `weight_units` | yes | for example `"EUR per trait unit"` |
| `synthetic_weights` | yes | declare `true` for placeholder weights |

### 5.12 `[output]`, `[resources]`, `[backend]`

| Key | Default | Notes |
|---|---|---|
| `output.top_n` | 20 | rows in the report's top-animal tables |
| `resources.max_memory_gb` | 4.0 | dense solvers that would exceed it are refused (`ABP-E500`) |
| `resources.min_free_disk_mb` | 200 | checked before and after the run (`ABP-E501`) |
| `backend.device` | `"cpu"` | `"cuda"` is accepted as a request, but ABP has no CUDA code yet |
| `backend.on_unavailable` | `"error"` | `"fallback_cpu"`: run on the CPU and record the fallback |

### 5.13 `[upg]` (optional): unknown-parent groups

| Key | Default | Notes |
|---|---|---|
| `prefix` | **required** | parent codes starting with it name groups, for example `"UPG:"` → `UPG:A_16_19` |
| `effect` | `"random"` | `"random"` (recommended) or `"fixed"` (must be estimable, §7.9) |
| `variance_ratio` | none | σ²g/σ²a, required for random groups, not allowed for fixed groups |

```toml
[upg]
prefix = "UPG:"
effect = "random"
variance_ratio = 1.0
```

### 5.14 `[bayes]`: Bayesian marker models

Used with `variances.mode = "bayes"` (and only then).

| Key | Default | Notes |
|---|---|---|
| `method` | **required** | `BRR`, `BayesA`, `BayesB`, `BayesC`, `BayesCpi`, `BayesR` (§7.10) |
| `pi0` | 0.95 | prior probability that a marker effect is **exactly zero** (fixed for BayesB/BayesC; start value for BayesCpi) |
| `chains` | 4 | at least 2; 4 or more recommended |
| `iterations` | 6000 | per chain, including burn-in |
| `burn_in` | 1000 | must be smaller than `iterations` |
| `thin` | 5 | keep every `thin`-th draw after burn-in |
| `seed` | 20260925 | master seed; each chain gets an independent stream |
| `prior_r2` | 0.5 | share of the phenotypic variance expected to be genetic; sets the default prior scales |
| `nu`, `nu_e` | 5.0, 5.0 | degrees of freedom of the scaled inverse-χ² priors (> 2) |
| `rhat_max` | 1.01 | convergence: split R-hat must be below it |
| `ess_min` | 400 | convergence: bulk and tail ESS must reach it |
| `max_iterations` | 30000 | the chains are doubled in length until the criteria pass or this budget is used |

### 5.15 `[validation]`: forward-in-time (LR) validation

| Key | Default | Notes |
|---|---|---|
| `method` | **required** | `"lr"` |
| `cutoff` | **required** | records dated after this day are hidden from the partial evaluation; must be earlier than `analysis.information_cutoff` |
| `focal` | `"new_records_only"` | `new_records_only`: animals whose only records are after the cutoff; `born_after_cutoff`: animals born after it (needs `birth_date`) |
| `bootstrap_replicates` | 1000 | 0 disables the bootstrap |
| `bootstrap_cluster` | `"sire"` | resample sire families (`"none"`: single animals) |
| `seed` | 20260925 | bootstrap seed |

### 5.16 Cross-field rules

ABP also checks, before running:

* every `model.traits` entry is declared in `[[traits]]`;
* exactly one additive genetic term. Random-term names are unique and never `residual`;
* `pedigree`/`single_step` need a pedigree; `genomic`/`single_step` need
  genotypes and a map, or a PLINK fileset (not both). `plink` needs
  `genotype_assembly`;
* known variances list exactly the random terms plus `residual`. Single-trait
  models take numbers; multi-trait models take t×t matrices;
* REML is single-trait only, and `reml.start` (if given) lists every component;
* multi-trait models contain only the additive term;
* `variances.mode = "bayes"` and `[bayes]` go together; Bayesian marker
  models are single-trait, have exactly one additive term with
  `relationship = "genomic"`, and cannot be combined with `[validation]`;
* `[validation]` needs `data.phenotype_columns.date` and a single-trait
  model;
* `[upg]` needs `relationship = "pedigree"` and a single-trait model; fixed
  groups need `variances.mode = "known"` and cannot be combined with
  `[validation]`; the prefix must differ from every unknown-parent code;
* `index.weights` refer to model traits.

---

## 6. Quality control

QC runs automatically inside `abp run`. It can also be run on its own
without fitting a model: `abp validate analysis.toml`. Findings have one of
four severities:

| Severity | Effect |
|---|---|
| `info` | descriptive (for example parents added as founders) |
| `review` | shown for you to check; data are **not** changed |
| `quarantine` | records excluded by a rule *you* enabled; every one is listed in `qc_excluded_records.csv` (reversible) |
| `error` | the run stops (exit status 4) and nothing is published |

All problems of a kind are collected before stopping, so one run shows every
offending line.

### 6.1 Pedigree rules

| Check | Severity | Rule |
|---|---|---|
| PED-ID | error | empty IDs or IDs with surrounding spaces |
| PED-SEX-CODE, PED-DATE | error | sex not M/F/U; malformed birth date |
| PED-DUP-CONFLICT | error | the same animal twice with different parents, sex or birth date |
| PED-DUP-IDENTICAL | review | exact duplicate rows (kept once) |
| PED-SELF-PARENT | error | animal is its own parent |
| PED-SAME-SIRE-DAM | error | one animal recorded as both parents |
| PED-ROLE-CONFLICT | error | an animal used both as a sire and as a dam |
| PED-SEX-CONFLICT | error | declared sex contradicts the parental role |
| PED-BIRTH-ORDER | error | a parent born on or after its offspring |
| PED-CYCLE | error | an animal is its own ancestor |
| PED-FOUNDER-ADDED | info | parents without their own row |
| PED-RECORD-ANIMAL-ADDED | info | animals with records added as founders (`qc.unknown_animals`) |
| PED-SINGLE-PARENT | info | animals with one known parent |
| PED-UPG-ID | error | an animal ID starts with the group prefix (`[upg]`) |
| PED-UPG | info | groups found, with how often each replaces a sire or a dam |

### 6.2 Phenotype rules

| Check | Severity | Rule |
|---|---|---|
| PHE-ID, PHE-RECORD-DUP | error | invalid IDs; duplicate `record_id` |
| PHE-UNKNOWN-ANIMAL | error or info | record animal not in the pedigree (`qc.unknown_animals`) |
| PHE-RANGE | error or quarantine | outside `[min, max]` (`qc.out_of_range`) |
| PHE-MISSING-CLASS | error or quarantine | used record without a needed classification value (`qc.missing_fixed`) |
| PHE-REPEATED | error | repeated records without a permanent-environment term |
| PHE-OUTLIER | review | beyond `qc.outlier_sd` standard deviations (kept) |
| PHE-SINGLETON-CLASS | review | a fixed-effect level with one record |
| PHE-UNGENOTYPED | quarantine | GBLUP with `qc.ungenotyped_records = "exclude"` |

### 6.3 Genotype rules

| Check | Severity | Rule |
|---|---|---|
| GEN-MAP | error | map/header mismatch, `counted_allele` not ref/alt, more than one assembly |
| GEN-ID | error | duplicated animal in the genotype file |
| GEN-DOSAGE-RANGE | error | dosage outside [0, 2] or not a number |
| GEN-AMBIGUOUS | review | A/T or C/G markers |
| GEN-CALLRATE-ANIMAL / -MARKER | quarantine | below the configured call rate |
| GEN-MONOMORPHIC | quarantine | no variation in the frequency sample |
| GEN-MAF / GEN-LOW-MAF | quarantine / review | below `min_maf` / below 0.01 |
| GEN-MENDEL | review / error | genotyped parent–offspring pairs with opposing homozygotes above 1% (review) or above 5% (error: pedigree error or sample swap) |
| GEN-PLINK | info | PLINK input: counted allele = A1; reference/alternative unknown; declared assembly |

---

## 7. Analyses

### 7.1 Single-trait animal model (pedigree BLUP)

Model: `y = Xb + Za + e`, with `a ~ N(0, A σ²a)` and `e ~ N(0, I σ²e)`.
Use it for one trait with one record per animal. Examples 01 and 02.

Fixed effects must be connected well enough to compare animals across
groups. ABP makes the fixed-effect design identifiable by setting redundant
levels to zero (§3 of `methods.md`) and reports how many it constrained.
**Individual fixed-effect solutions depend on this choice and should not be
interpreted on their own.** EBVs do not depend on it.

### 7.2 Repeatability model

Add an iid term on the animal ID for the permanent environment:

```toml
random = [
  { name = "animal", kind = "additive", relationship = "pedigree" },
  { name = "pe", kind = "iid" },
]
```

Example 03 analyses litter size at every lambing of every ewe. The
permanent-environment solutions are written to `random_pe_<trait>.csv`.
Litter size is a count (1, 2, 3). Analysing it with a linear model is an
approximation that is common in practice, but it is still an approximation
(threshold models are on the roadmap).

### 7.3 Estimating variance components (REML)

Set `variances.mode = "reml"`. ABP maximises the restricted likelihood with
average-information steps. It falls back to EM steps whenever an AI step
would not improve the likelihood, and reports:

* `status`: `converged`, or `converged_boundary` when a variance is
  estimated as exactly zero. That term is then removed from the model for
  BLUP. A zero is accepted only if the likelihood slope at zero points
  outward, i.e. zero is a genuine maximum.
* the variances, their approximate standard errors (from the inverse
  average-information matrix; withheld at a boundary), heritability and its
  approximate standard error, and the iteration history (`results.json`).

REML that does not converge within `reml.max_iter` is an **error**. No EBVs
are published from unconverged variances. If the genetic variance itself is
estimated as zero, ABP refuses to rank animals (`ABP-E300`).

REML uses the dense solver. Its memory need is about 24 × N² bytes for N
equations (about 1 GB for 6,500 equations; see §8). Checkpoints are written
after every iteration next to the output folder. If a run is interrupted,
re-run the same command with `--resume` to continue. ABP refuses a checkpoint
that belongs to different inputs or a different spec.

### 7.4 GBLUP

```toml
random = [{ name = "animal", kind = "additive", relationship = "genomic" }]

[genomic]
frequency_source = "genotyped_all"
singular_policy = "blend"      # or "ridge"; the default "error" stops on a singular G
blend_alpha = 0.05

[qc]
ungenotyped_records = "exclude"
```

The genomic relationship matrix is `G = WW'/(2Σp(1−p))`, with `W` the
centred dosages (§6 of `methods.md`). Points to keep in mind:

* **Frequencies define the base.** The source and its transductive or
  inductive character are written to the manifest.
* **G is often singular** (more animals than independent markers, clones,
  duplicated samples). ABP never repairs it silently. Choose `blend` or
  `ridge` explicitly; the smallest eigenvalue before and after is recorded.
* Records of non-genotyped animals cannot be used by GBLUP. ABP stops unless
  you exclude them explicitly (they are then listed) or switch to
  single-step.

Example 04 uses it for faecal egg count.

### 7.5 Single-step GBLUP (ssGBLUP)

```toml
random = [{ name = "animal", kind = "additive", relationship = "single_step" }]

[genomic]
tuning = "match_a22"
singular_policy = "blend"
blend_alpha = 0.05
```

All animals in the pedigree are evaluated together, and genotyped animals
contribute through `H⁻¹ = A⁻¹ + [0 0; 0 G*⁻¹ − A22⁻¹]`. `A22⁻¹` is computed
from `A22` itself (it is **not** the corresponding block of `A⁻¹`). The
manifest records the tuning constants and the mean diagonal and off-diagonal
elements of `A22` and `G*`, so the compatibility of the two bases can be
checked. Every genotyped animal must be in the pedigree. Example 05.

> **Caution (validation finding F6).** In ABP's 50-replicate calibration
> study, single step with `match_a22` tuning and a 5% blend under-predicted
> the selection candidates by 0.66 kg on average, and its PEV understated
> the real prediction error by about 28% (95% intervals covered 91.7% of
> true values). Pedigree BLUP in the same study was calibrated. The cause
> is a mismatch between the genomic and pedigree bases, not a numerical
> error. Until base alignment (metafounders) is implemented, compare
> single-step and pedigree EBV levels for your population, validate
> forward in time (§7.11), and do not rely on single-step reliabilities
> alone.

### 7.6 Multi-trait BLUP

Give several traits and covariance matrices:

```toml
[model]
traits = ["wwt", "fat"]
fixed = [
  { column = "cg", type = "factor" },
  { column = "lambing1_year", type = "factor", traits = ["nlb1"] },
]
random = [{ name = "animal", kind = "additive", relationship = "pedigree" }]

[variances]
mode = "known"
values.animal   = [[4.0, 0.3], [0.3, 0.25]]
values.residual = [[12.25, 0.735], [0.735, 0.49]]
```

* Rows and columns follow the order of `model.traits`.
* Missing trait values are handled through the likelihood of the observed
  values; nothing is imputed.
* Enter a residual covariance of **0** for traits that cannot share a
  residual. An example is a lamb's weaning weight and the same animal's
  first litter size two years later (a structural zero).
* ABP checks that both matrices are symmetric positive definite.
* Multi-trait models support the additive term only, and variances must be
  known: multi-trait REML is not implemented yet.

Outputs: `ebv_multitrait.csv` with EBV, reliability and SEP per trait.

### 7.7 Economic index on EBVs

Add an `[index]` section to a multi-trait spec (example 06):

```toml
[index]
weights = { wwt = 2.0, fat = -3.0, fec = -4.0, nlb1 = 25.0 }
weight_units = "synthetic currency units (SCU) per trait unit"
synthetic_weights = true
```

With EBVs from the joint multi-trait model, `I = Σ aⱼ·EBVⱼ` is the BLUP of
the aggregate genotype `H = Σ aⱼ·gⱼ`. Its reliability,
`1 − a'·PEV·a / (a'·G0·a · Aᵢᵢ)`, uses each animal's full PEV block. The
ranking is written to `index.csv` and shown in the report. Weights are in
money (or another unit) **per unit of the trait**. If a trait's unit changes
(kg → g), its weight must change inversely.

### 7.8 Stand-alone selection index (`abp index`)

For classical index calculations from phenotypic information sources:

```bash
abp index examples/07_selection_index/index.toml
```

The file supplies `P = Var(x)`, `C = Cov(x, g)`, `G = Var(g)`, the weights,
and optionally `restrict` (traits whose expected change must be zero) and
`proportion_selected`. The output gives the index coefficients, `Var(I)`,
`Var(H)`, reliability and accuracy (`r_IH = sqrt(Var(I)/Var(H))`), the
selection intensity, and the expected response per trait. Duplicated
information (a singular `P`) is reported with the offending combination.
Infeasible restrictions are reported and never relaxed.

### 7.9 Unknown-parent groups

**When you need them.** An animal with an unknown parent is normally
treated as a son or daughter of the genetic base (mean 0). That is wrong
when the missing parents come from populations with a different genetic
level: rams bought from a breeder who has been selecting for years, imported
semen, or founders entering in different decades. Their EBVs, and those of
all their descendants, are then biased towards the base. Genetic groups give
each such population its own estimated level.

**How to declare them.** Put group codes in the pedigree (§4.1) and add

```toml
[upg]
prefix = "UPG:"
effect = "random"
variance_ratio = 1.0
```

Define groups by what describes the missing parent: origin, period (for
example the purchase or birth year of its offspring, in blocks of several
years) and, if relevant, sex. Each group needs enough descendants with
records to be estimated. Never define groups from the performance of the
animals themselves.

**What is computed.** Each animal's breeding value becomes
`u*ᵢ = uᵢ + Qᵢ·g`, where `Qᵢ` is the expected fraction of its genes from each
group (a parent that is a group contributes ½ to that group; a known parent
passes on half of its own fractions) and `g` are the group effects. ABP uses
the QP transformation of Quaas (1988): the group equations are added to
`A⁻¹` by Henderson's rules with groups acting as parents, so no extra
matrices are formed (§12 of `methods.md`).

* The EBV file contains `u*` (group contributions included) for animals
  only. Group solutions, their PEV/SEP, the number of animals with a
  non-zero gene fraction and the summed gene fraction are written to
  `upg_solutions_<trait>.csv`.
* **Random groups** (`effect = "random"`, recommended): `g ~ N(0, σ²g·I)`
  with the declared ratio `σ²g/σ²a`. Reliabilities use
  `Var(u*ᵢ) = σ²a·(Aᵢᵢ + ratio·QᵢQᵢ')`. REML is available; it estimates σ²a
  and σ²e with the ratio held at its declared value. Random groups are
  shrunk towards 0: a small ratio shrinks strongly, a large ratio approaches
  fixed groups (see the study below).
* **Fixed groups** (`effect = "fixed"`): no shrinkage, but the group effects
  must be estimable together with the fixed effects. ABP checks the rank of
  `[X, ZQ]` before solving and stops with `ABP-E300` if, for example, the
  model has an intercept and *every* recorded lineage ends in a group (the
  gene fractions then sum to one and duplicate the intercept), or a group
  has no recorded descendants. It never picks an arbitrary constraint. The
  prior variance of `u*` is not defined for fixed groups, so reliabilities
  are left empty; PEV includes the estimation error of the groups. REML is
  refused (the covariance matrix is improper): declare the variances.

**Example 11** simulates a closed ewe flock that buys all its rams from two
breeders without recorded ancestry: breeder A, whose rams improve by
0.6 kg/year, and breeder B, whose rams are 1 kg below the base. Groups are
`UPG:A_16_19`, `UPG:A_20_24` and `UPG:B`; the local base flock has ordinary
unknown parents (`0`), which keeps fixed groups estimable.

```bash
python examples/11_sheep_upg/make_data.py        # regenerates the data (seeded)
abp run examples/11_sheep_upg/analysis.toml --out runs/ex11        # random groups, REML
abp run examples/11_sheep_upg/analysis_fixed.toml --out runs/ex11f # fixed groups
python examples/11_sheep_upg/compare.py          # all three models against the truth
```

`compare.py` also runs the same data *without* groups (codes replaced by
`0`). For this one synthetic replicate:

| model | corr(EBV, TBV), lambs | bias of purchased rams (kg) | bias of 2024 lambs (kg) |
|---|---:|---:|---:|
| random groups (ratio 1.0) | 0.767 | −0.71 | −1.79 |
| fixed groups | 0.778 | +0.38 | −0.72 |
| no groups | 0.717 | −1.91 | −3.00 |

Over 20 replicates (`benchmarks/upg_study.py`, evidence in
`docs/validation/upg_study*.json`) groups removed 0.74 ± 0.10 kg (random)
and 1.11 ± 0.14 kg (fixed) of the absolute bias of purchased rams and raised
the EBV–TBV correlation by 0.04. Some bias remains even with fixed groups,
because breeder A's level also rises *within* each 4–5-year period, which a
period group cannot represent; in a control scenario without that
within-period trend, fixed groups were unbiased within Monte-Carlo error.
These numbers describe one synthetic design, not your population.

Limitations: pedigree relationship and single-trait models only; groups are
unrelated and non-inbred (no metafounders); the variance ratio is declared,
not estimated.

### 7.10 Bayesian marker models

```toml
random = [{ name = "animal", kind = "additive", relationship = "genomic" }]

[variances]
mode = "bayes"

[bayes]
method = "BayesC"
pi0 = 0.95          # probability that a marker effect is exactly zero
chains = 4
iterations = 4000
burn_in = 1000
thin = 3
seed = 20260925
prior_r2 = 0.3
max_iterations = 16000
```

The model is `y = Xb + Wβ + e`, with `W` the centred dosages (the same
coding as G, §7.4) and one of these priors on each marker effect:

| Method | Prior on a marker effect βⱼ |
|---|---|
| `BRR` | normal, one common variance (Bayesian ridge regression; with the variances fixed, its posterior mean equals SNP-BLUP/GBLUP) |
| `BayesA` | normal with a marker-specific variance (heavy-tailed) |
| `BayesB` | exactly 0 with probability `pi0`, otherwise normal with a marker-specific variance |
| `BayesC` | exactly 0 with probability `pi0`, otherwise normal with a common variance |
| `BayesCpi` | as BayesC, with `pi0` estimated (uniform prior) |
| `BayesR` | 0, or normal with variance 10⁻⁴, 10⁻³ or 10⁻² × σ²b; mixture weights estimated |

`pi0` is always the probability of a **zero** effect (some software uses π
for the opposite). Prior scales are derived from `prior_r2`: the
phenotypic variance after the fixed effects is split into an expected
genetic and residual part, and each variance prior is centred on the value
this implies. Every hyper-parameter and its derivation is written to
`mcmc_diagnostics_<trait>.json`.

**Sampling and convergence.** ABP runs `chains` independent Gibbs chains
(each with its own random stream derived from `seed`). After `iterations`
it computes, for every monitored quantity — residual variance, genetic
variance, heritability, marker-variance parameters, `pi0` where it is
estimated — **and for every GEBV**, the rank-normalised split R-hat and the
bulk and tail effective sample sizes (Vehtari et al. 2021). If any value
misses `rhat_max` or `ess_min`, the chains are doubled in length, up to
`max_iterations`. If the criteria are still not met, **no EBVs are
published**: the run stops with `ABP-E405` (exit status 6) and the
diagnostics stay in the failed run folder. (If storing all GEBV draws
would exceed 5×10⁷ numbers, GEBV-level diagnostics are skipped and the
report says so.)

**Example 10** (BayesC, `pi0 = 0.95`, log faecal egg count of 476 genotyped
animals, 2,000 markers) converged after one extension to 8,000 iterations:
posterior mean h² 0.31 (SD 0.12), R-hat ≤ 1.005 for every scalar and ≤ 1.003
for every GEBV, smallest bulk ESS 813. The same data with `BayesCpi` did
**not** pass the diagnostics within 16,000 iterations (`pi0` mixes slowly),
and ABP withheld the results — this is the intended behaviour.

**Checking the model against the data.** At every saved draw ABP also
simulates a replicate data set from the model and compares four features
with the observed records: SD, skewness, minimum and maximum. The
posterior predictive p-value `P(T(y_rep) ≥ T(y))` of each is reported; a
value below 0.01 or above 0.99 means that the model does not reproduce that
feature (for example skewed data analysed with a normal residual) and is
logged as a warning. Convergence diagnostics and predictive checks answer
different questions — one does not replace the other.

**Evidence that the sampler is right.** Besides the unit tests against
exact results, every method passed a prior simulation-based calibration
study (`benchmarks/sbc_bayes.py`, §3 of the validation report): parameters
drawn from the prior, data simulated, the sampler run, and the ranks of the
true values among the posterior draws tested for uniformity.

Outputs: `ebv_<trait>.csv` (`gebv_posterior_mean`, `gebv_posterior_sd`
for every QC-passed genotyped animal, with or without records),
`marker_effects_<trait>.csv` (posterior mean effect and inclusion
probability per marker), `fixed_effects_<trait>.csv`,
`mcmc_diagnostics_<trait>.json` (priors, seeds, diagnostics, predictive
checks) and `mcmc_trace_<trait>.csv` (every saved draw of every monitored
quantity, per chain, for trace plots). Records of non-genotyped animals follow
`qc.ungenotyped_records` as for GBLUP. Only one record per animal is
allowed.

Run time grows with records × markers × iterations × chains. The optional
C++ kernel does the marker sweep; it matches the Python reference to
10⁻¹² and is reported in the output (`kernel`).

### 7.11 Forward-in-time validation (LR method)

Add a `[validation]` section and a record-date column to a single-trait
pedigree, genomic or single-step spec (example 08):

```toml
[data.phenotype_columns]
date = "year"          # weaning weights are dated by their year

[validation]
method = "lr"
cutoff = "2024-12-31"
focal = "new_records_only"
bootstrap_replicates = 1000
```

ABP then runs two extra evaluations with the **same** variance components:
a *partial* one in which all records dated after `cutoff` are hidden (their
values are never read), and a *whole* one with all records. With
`variances.mode = "reml"` the variances are estimated on the partial data
only, so no hidden record can influence the partial evaluation. On the
focal animals (by default: animals whose only records are after the cutoff)
it reports (Legarra and Reverter 2018):

| Statistic | Definition | Expected if the model is right |
|---|---|---|
| bias `Δp` | mean(partial EBV) − mean(whole EBV) | 0 |
| dispersion `b_w\|p` | slope of whole on partial EBVs | 1 (below 1: partial EBVs over-dispersed) |
| `ρ_wp` | correlation of partial and whole EBVs | acc(partial)/acc(whole), below 1 |

Each statistic gets a bootstrap standard error and 95% interval, resampling
sire families (seed recorded). The published main evaluation is not changed.

Example 08 (weaning weight, cutoff end of 2024, 403 focal lambs):
`Δp = 0.057 kg` (95% interval −0.25 to 0.21), `b_w|p = 0.99` (0.90 to
1.24), `ρ_wp = 0.67` (0.58 to 0.78): no sign of bias or over-dispersion at
this sample size.

Read the results with care: `ρ_wp` near 1 only shows that the two
evaluations agree, not that either is accurate; the statistics refer to
this cutoff and these animals. The LR "population accuracy" estimator is
deliberately not computed in this version (its published formula has an
erratum that could not be verified). Outputs: `lr_validation.json`
(statistics, bootstrap, assumptions) and `lr_focal_<trait>.csv` (partial and
whole EBVs and reliabilities of every focal animal).

### 7.12 Optimal contributions and mating plans (`abp mate`)

`abp mate` answers two questions for the next breeding season: how much
should each candidate contribute, and who should be mated with whom? The
result is a **proposal** for the breeder; ABP never executes matings.

```bash
abp mate examples/09_sheep_mating/mating.toml --out runs/mating
```

The mating spec:

```toml
schema_version = "1"

[project]
name = "sheep_mating_2026"
species = "sheep"
synthetic_data = true

[mating]
candidates = "candidates.csv"   # id, sex (M/F), merit, capacity [, carrier]
pedigree = "../sheep_data/data/pedigree.csv"
merit_units = "SCU (synthetic index units)"
n_matings = 150
delta_f = 0.01                  # or max_coancestry = ... (exactly one)
max_pair_relationship = 0.25    # optional: no pair with A_sd above it
carrier_column = "carrier"      # optional: carrier probability of a recessive
max_affected_risk = 0.01        # optional: needs carrier_column
```

| Key | Notes |
|---|---|
| `candidates` | CSV with `id`, `sex` (`M`/`F`), `merit` (EBV or index), `capacity` (maximum number of matings); column names can be changed with `id_column`, `sex_column`, `merit_column`, `capacity_column` |
| `pedigree` | defines the relationships `A` among candidates; `group_prefix` marks unknown-parent group codes (§4.1) |
| `n_matings` | number of matings to plan |
| `delta_f` / `max_coancestry` | coancestry ceiling: `C_max = C_t + ΔF·(1 − C_t)` with `C_t` the mean coancestry of the candidates, or an explicit value |
| `max_pair_relationship` | forbids pairs with `A_sd` above it (0.25 forbids half sibs and closer) |
| `carrier_column`, `max_affected_risk` | forbids pairs whose risk of an affected lamb for a declared autosomal recessive, `p_s·p_d/4`, exceeds the limit |
| `forbidden_pairs` | optional CSV with columns `sire,dam` |
| `delimiter` | default `","` |

**Step 1, contributions** (Meuwissen 1997): maximise the expected merit
`c'g` of the next generation subject to each sex contributing ½, each
candidate between 0 and `capacity/(2·n_matings)`, and the group coancestry
`c'Ac/2 ≤ C_max`. ABP solves this exactly and prints a KKT certificate
(stationarity, feasibility and multipliers), so the optimality can be
checked. If the ceiling is below the lowest achievable coancestry, the run
stops and reports that minimum.

**Step 2, whole matings.** Contributions are converted to whole numbers of
matings and, if rounding pushes the coancestry above the ceiling, repaired
by a local search. The ceiling is a hard constraint for the integer plan
too; the loss of merit against the continuous optimum is reported.

**Step 3, pairs.** Sires and dams are paired so that each parent gets its
number of matings, no pair is repeated, no forbidden pair is used, and the
total inbreeding of the progeny (`A_sd/2`) is minimal. The progeny's
expected merit is the same for every such plan. ABP solves this as a
transportation problem, checks that the plan is integral and satisfies
every constraint, and, if no plan exists, names the parents that cannot be
matched instead of relaxing a constraint.

Example 09 (150 matings among the 2025 candidates of example 06): the
ceiling `C_max = 0.0506` was binding (the maximum-merit selection would
have had coancestry 0.120); the plan uses 20 sires and 150 ewes, its merit
is 0.0019 SCU below the continuous optimum, 331 pairs were forbidden by the
relationship limit and 258 by the recessive risk, and the mean planned
progeny inbreeding is 0.00003.

Outputs: `contributions.csv` (optimal and realised contribution and number
of matings per candidate), `mating_plan.csv` (pairs with `A_sd`, progeny
inbreeding, expected progeny merit and recessive risk), `mating_summary.json`,
`mating_report.md` and `manifest.json`. Coancestry refers to the pedigree
base; candidates' merits are taken as given (their errors are not
propagated).

---

## 8. Solvers, memory and run time

`solver.method = "auto"` chooses as follows. The choice and the reason are
written to the report and the manifest.

| Situation | Solver | Why |
|---|---|---|
| ≤ 12,000 equations and within the memory budget | dense Cholesky (LAPACK) | fastest; exact PEV |
| larger, `pev = "exact"`, ≤ 30,000 equations | sparse direct (SuperLU) | exact PEV by selected solves |
| larger, `pev = "none"` | Jacobi-preconditioned conjugate gradients | fast; falls back to sparse direct if it does not converge |
| larger than 30,000 equations with `pev = "exact"` | refused (`ABP-E303`) | set `pev = "none"`; approximate reliabilities are on the roadmap |

After every solve ABP recomputes the relative residual `‖C·s − r‖/‖r‖` on the
original equations and refuses solutions above 1e-8 (direct) or `solver.tol`
(PCG).

Memory for the dense path is about 16 × N² bytes (solutions) or
24 × N² bytes (with PEV or REML). Measured times on a 4-core Linux machine
([benchmarks](benchmarks.md)):

| Task | Size | Time |
|---|---|---|
| inbreeding, C++ kernel | 100,000 animals, 10 generations | 0.8 s |
| BLUP, PCG, no PEV | 100,500 equations | 0.26 s |
| BLUP, dense, with PEV | 5,500 equations | 4.8 s |
| REML (AI), dense | 3,050 equations, 6 iterations | 3.2 s |
| four-trait BLUP with PEV blocks (example 06) | 8,506 equations | about 8 s end to end |
| G matrix | 2,000 animals × 50,000 markers | 1.3 s |
| BayesC, 4 chains × 8,000 iterations (example 10) | 247 records × 2,000 markers | about 25 s |
| one Gibbs marker sweep, C++ kernel | 1,000 records × 10,000 markers | 0.012 s |
| A⁻¹ with 50 genetic groups | 100,000 animals | 0.05 s (after inbreeding, 0.7 s) |
| optimal contributions + mating LP | 1,500 candidates, 97,200 candidate pairs | 1.0 s + 12.6 s |

---

## 9. Output files

| File | Content |
|---|---|
| `report.md` | breeder-facing summary: what was estimated, variance components, top animals, index, QC, numerical verification, limitations, provenance |
| `ebv_<trait>.csv` | `animal, sire, dam, sex, generation, inbreeding, n_records, ebv, pev, sep, reliability, accuracy` (single-trait) |
| `ebv_multitrait.csv` | pedigree columns, then `ebv_<t>, reliability_<t>, sep_<t>, n_records_<t>` per trait |
| `fixed_effects_<trait>.csv` | `term, level, solution, status` (`estimated_under_constraints` or `constrained_to_zero`) |
| `random_<term>_<trait>.csv` | solutions and PEV of iid terms (for example permanent environment) |
| `index.csv` | `rank, animal, sex, index, reliability` |
| `upg_solutions_<trait>.csv` | genetic groups: `group, effect, solution, pev, sep, reliability, n_animals_with_contribution, sum_gene_fraction` (§7.9) |
| `lr_validation.json`, `lr_focal_<trait>.csv` | LR validation statistics, bootstrap and assumptions; partial and whole EBVs of the focal animals (§7.11) |
| `ebv_<trait>.csv` (Bayesian) | `animal, sire, dam, sex, n_records, gebv_posterior_mean, gebv_posterior_sd` (§7.10) |
| `marker_effects_<trait>.csv` | `marker_id, counted_allele, frequency, effect_posterior_mean, inclusion_probability` |
| `mcmc_diagnostics_<trait>.json`, `mcmc_trace_<trait>.csv` | priors, chain seeds, R-hat/ESS/MCSE, predictive checks; per-draw traces |
| `contributions.csv`, `mating_plan.csv`, `mating_summary.json`, `mating_report.md` | outputs of `abp mate` (§7.12), with their own `manifest.json` |
| `qc_pedigree.json`, `qc_phenotypes.json`, `qc_genotypes.json` | QC statistics and findings |
| `qc_excluded_records.csv` | every record excluded by an approved rule, with the reason |
| `results.json` | every number shown in the report (machine-readable) |
| `manifest.json` | the run record (§10) |
| `run.log` | time-stamped log |

Numbers are written with full double precision. An empty cell means "not
computed" (for example PEV when `solver.pev = "none"`). The EBV file lists
all animals of the relationship structure, with parents before offspring.

---

## 10. Reproducibility, traceability and recovery

### 10.1 The run manifest

`manifest.json` answers "what exactly produced this number?":

* the spec file's SHA-256 and the complete effective spec, defaults included;
* the path, SHA-256 and row count of every input file, and a hash of the
  ordered animal list used by all matrices;
* the estimand, genetic base, information cut-off and synthetic-data flag;
* the ABP version, git commit and a hash of any uncommitted source changes;
* OS, Python, NumPy, SciPy, BLAS and whether the native kernel ran;
* the relationship construction (for genomic analyses: frequency source,
  transductive use, G policy and compatibility statistics);
* solver, reason for its choice, relative residual, iterations, time and the
  REML history;
* the SHA-256 of every output file, wall time, peak memory (Linux), exit
  status and, for failures, the error code and details.

### 10.2 All-or-nothing outputs

A run writes into `<out>.partial-<run id>` and is renamed to `<out>` only
after everything has succeeded. A folder with the name you requested
therefore always contains a complete, successful run. After an error the
folder becomes `<out>.failed-<run id>`, and after Ctrl+C
`<out>.cancelled-<run id>`. Both keep `run.log` and `manifest.json` for
diagnosis. An existing output folder is replaced only with `--force`, and
then atomically.

### 10.3 Determinism

Given the same inputs, spec and software environment, ABP writes
byte-identical EBV files; the test suite checks this. BLUP, REML, indices,
genetic groups, contributions and mating plans use no random numbers.
Three steps do, always from a declared seed that is written to the
manifest: the Bayesian samplers (`bayes.seed`; each chain gets its own
stream), the LR bootstrap (`validation.seed`) and the data generators
(`abp simulate-sheep --seed …`, `examples/11_sheep_upg/make_data.py`).
The same seed reproduces the same draws with the same kernel; the C++ and
Python samplers agree to about 10⁻¹² per sweep, so long chains run with
different kernels may diverge slightly, as any floating-point Markov chain
does.

### 10.4 Interruptions and resume

Ctrl+C, or a termination signal from a scheduler, cancels the run cleanly
(exit status 130). REML checkpoints allow `abp run … --resume`.

---

## 11. Command reference

### 11.1 Commands

| Command | Purpose |
|---|---|
| `abp run SPEC --out DIR [--force] [--resume] [--quiet]` | full evaluation |
| `abp validate SPEC` | spec validation and all QC, no model fitting; prints a JSON summary |
| `abp pedigree PED.csv --out DIR [--sex COL] [--birth-date COL] [--id/--sire/--dam COL] [--delimiter C] [--group-prefix P]` | pedigree QC, `inbreeding.csv` and `ainv_triplets.csv` (1-based lower triangle); with `--group-prefix` group codes are unknown parents for A and are listed in the QC report |
| `abp index INDEX.toml` | stand-alone Smith-Hazel or restricted index (JSON to stdout) |
| `abp mate MATING.toml --out DIR [--force]` | optimal contributions and a mating plan proposal (§7.12) |
| `abp simulate-sheep --out DIR [--seed N] [--force]` | write the synthetic sheep data set |
| `abp selftest` | installation check |
| `abp errors` | list error codes and exit statuses |
| `abp --version` | version |

`python -m abp …` is equivalent to `abp …`.

### 11.2 Exit statuses

| Status | Meaning |
|---|---|
| 0 | success |
| 1 | internal error (please report `run.log` and `manifest.json`) |
| 2 | usage problem, or the output folder already exists |
| 3 | input or spec contract violated |
| 4 | blocking QC finding |
| 5 | model not identifiable or not supported |
| 6 | numerical failure (non-convergence, reliability out of range, …) |
| 7 | resource or platform limit (memory, disk, backend) |
| 130 | cancelled |

---

## 12. Troubleshooting

Every error prints a code, what happened, where, and a remedy, for example:

```
error ABP-E202 (PEDIGREE_SEX_CONFLICT): pedigree QC found 2 blocking problem(s): ...
remedy: Correct the parent IDs or the sex column for the listed animals.
  - PED-ROLE-CONFLICT: animal used both as sire and as dam [1] {"animal": "5"}
  - PED-SEX-CONFLICT: declared sex contradicts parental role [1] {"animal": "5", "declared_sex": "F", "role": "sire"}
```

| Symptom | Likely cause and fix |
|---|---|
| `ABP-E101 INPUT_ENCODING` | Save the file as "CSV UTF-8" |
| `ABP-E103 SCHEMA_TYPE` with a line number | Text in a numeric column, a ragged row, or an undeclared missing-value code. Fix the cell or add the code to `data.missing_values`. |
| `ABP-E104 SPEC_INVALID ... unknown key` | Misspelt key. The message lists the allowed keys. |
| `ABP-E210` for GBLUP | Records of non-genotyped animals. Use `single_step`, or set `qc.ungenotyped_records = "exclude"`. |
| `ABP-E302 RELATIONSHIP_SINGULAR` | G is singular. Choose `genomic.singular_policy = "blend"` or `"ridge"`. |
| `ABP-E303` "repeated records" | Add `{ name = "pe", kind = "iid" }` to `model.random`. |
| `ABP-E303` "exact PEV … exceeds" | Set `solver.pev = "none"` for very large models. |
| `ABP-E403 REML_NOT_CONVERGED` | Raise `reml.max_iter`, give better `reml.start` values, or simplify the model (see the history in `manifest.json`). |
| `ABP-E405 MCMC_NOT_CONVERGED` | The chains did not meet R-hat/ESS within `bayes.max_iterations`. Raise it, increase `thin`, fix `pi0` instead of estimating it (BayesCπ mixes slowly), or check the model. Diagnostics are in the failed run folder. |
| `ABP-E300` "fixed unknown-parent group effects are confounded" | Every recorded lineage ends in a group, or a group has no recorded descendants. Use `upg.effect = "random"`, merge groups, or leave the base population's unknown parents ungrouped. |
| `ABP-E202` for parents such as `UPG:…` | Group codes without an `[upg]` section are read as animals. Declare `[upg] prefix` (or `group_prefix` in a mating spec). |
| `ABP-E223` with PLINK input | Declare `data.genotype_assembly`, or fix variants with identical alleles in the `.bim` file. |
| `ABP-E500 RESOURCE_MEMORY` | Raise `resources.max_memory_gb` if the machine has the memory, or use a sparse/iterative solver. |
| `ABP-E503 OUTPUT_EXISTS` | Choose a new `--out` folder or add `--force`. |
| Garbled Chinese text in the console on Windows | Use `packaging\windows\abp.ps1`, which sets UTF-8. The files ABP writes are always UTF-8. |
| `abp selftest` reports native kernel False | No C++20 compiler at install time. Results are identical, only slower. |

The complete list is in [`error_codes.md`](error_codes.md).

---

## 13. Limitations and good practice

What ABP does **not** do yet: metafounders, maternal and social effects,
random regression and test-day models, threshold and survival models,
genotype × environment models, dominance and epistasis, APY and other
approximations for very large genomic data, multi-trait REML, multi-trait
or single-step Bayesian models, Bayesian LASSO/horseshoe priors, genomic
OCS, VCF/BGEN/PLINK 2 readers, and GPU computation.

What has **not** been verified: external validity on real data, comparisons
with BLUPF90, MiXBLUP, ASReml, DMU, JWAS or BGLR, CUDA, and Windows beyond
the CI jobs. The [validation report](validation_report.md) gives details.

Good practice:

1. Run `abp validate` on new data and read every `review` finding before
   fitting models.
2. Keep the `analysis.toml`, the input files and the whole output folder
   together. The manifest hashes let anyone check later that nothing changed.
3. Estimate variance components on your own population (REML) rather than
   borrowing them. Record where known variances came from in a comment, as
   the examples do.
4. Judge candidates on both EBV and reliability. Treat low-reliability
   animals with caution.
5. Validate predictions forward in time (§7.11) before relying on them for
   selection decisions, and repeat the validation when the population or
   the model changes.
6. Use genetic groups when unknown parents come from populations of
   different genetic level (§7.9); record how the groups were defined.
7. Never use Bayesian results whose diagnostics failed; ABP withholds them.
   Read the predictive checks as well.
8. ABP produces **recommendations** for breeders to review, including the
   mating plans of `abp mate`. It never triggers matings or other
   real-world actions.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| A | numerator (pedigree) relationship matrix; `Aᵢᵢ = 1 + Fᵢ` |
| A22 | pedigree relationships among genotyped animals |
| BLUP | best linear unbiased prediction |
| coancestry | probability that two alleles drawn from two animals are identical by descent; `A_ij/2`; the group coancestry of parents with contributions `c` is `c'Ac/2` |
| ESS | effective sample size of MCMC draws (bulk: centre of the distribution; tail: 5% and 95% quantiles) |
| genetic group (UPG) | a population of unknown parents with its own genetic level |
| LR method | forward validation comparing partial and whole evaluations (Legarra and Reverter 2018) |
| MCSE | Monte-Carlo standard error of a posterior summary |
| OCS | optimal contribution selection: maximising genetic merit under a coancestry ceiling |
| π₀ (`pi0`) | prior probability that a marker effect is exactly zero |
| R-hat | convergence diagnostic comparing chains; values near 1 indicate agreement |
| SBC | simulation-based calibration of a sampler (ranks of true values must be uniform) |
| contemporary group | animals managed and recorded together (for example flock × year × sex) |
| EBV / GEBV | (genomic) estimated breeding value |
| F | inbreeding coefficient |
| G | genomic relationship matrix; G* after tuning, blending or ridge |
| H | single-step relationship matrix combining A and G* |
| MAF | minor allele frequency, `min(p, 1 − p)` |
| PEV / SEP | prediction error variance / its square root |
| REML | restricted maximum likelihood |
| reliability | `1 − PEV / genetic variance`; squared model-based accuracy |
| transductive | using candidates' genotypes (never their phenotypes) when building the model |
| u* | breeding value including genetic-group contributions, `u + Qg` |

---

## 15. References

* Aguilar I, Misztal I, Johnson DL, Legarra A, Tsuruta S, Lawlor TJ (2010) J Dairy Sci 93:743–752.
* Christensen OF, Lund MS (2010) Genet Sel Evol 42:2.
* Christensen OF, Madsen P, Nielsen B, Ostersen T, Su G (2012) Genet Sel Evol 44:37.
* Colleau JJ (2002) Genet Sel Evol 34:409–421.
* Gilmour AR, Thompson R, Cullis BR (1995) Biometrics 51:1440–1450.
* Hazel LN (1943) Genetics 28:476–490.
* Henderson CR (1975) Biometrics 31:423–447; (1976) Biometrics 32:69–83.
* Kempthorne O, Nordskog AW (1959) Biometrics 15:10–19.
* Meuwissen THE, Luo Z (1992) Genet Sel Evol 24:305–313.
* Mrode RA (2005) Linear Models for the Prediction of Animal Breeding Values, 2nd ed. CABI.
* Erbe M, Hayes BJ, Matukumalli LK, et al. (2012) J Dairy Sci 95:4114–4129.
* Habier D, Fernando RL, Kizilkaya K, Garrick DJ (2011) BMC Bioinformatics 12:186.
* Legarra A, Reverter A (2018) Genet Sel Evol 50:53.
* Meuwissen THE (1997) J Anim Sci 75:934–940.
* Meuwissen THE, Hayes BJ, Goddard ME (2001) Genetics 157:1819–1829.
* Quaas RL (1976) Biometrics 32:949–953.
* Quaas RL (1988) J Dairy Sci 71:1338–1345.
* Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A (2018) arXiv:1804.06788.
* VanRaden PM (2008) J Dairy Sci 91:4414–4423.
* Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021) Bayesian Analysis 16:667–718.
* Westell RA, Quaas RL, Van Vleck LD (1988) J Dairy Sci 71:1310–1318.
