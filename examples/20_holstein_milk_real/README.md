# Example 20: real Holstein data (pedigreemm `milk`)

**Real data, not stored in this repository.** Fetch and convert them first:

```bash
pip install rdata                                   # MIT licence; used only by fetch_data.py
python examples/20_holstein_milk_real/fetch_data.py
```

## Source and licence

* `milk` and `pedCowsR` from the R package **pedigreemm 0.3-5** (Bates, Vazquez, Perez
  Rodriguez; licence GPL (>= 2)), downloaded from the read-only GitHub mirror of CRAN
  (`raw.githubusercontent.com/cran/pedigreemm/0.3-5/data/`); SHA-256 checked by the script.
* The package documents `milk` as 3,397 lactation records (parities 1-5) of 1,359
  Holstein cows, daughters of 38 sires, in 57 herds, downloaded from the USDA (AIPL) web
  site; reference: Vazquez AI, Bates DM, Rosa GJM, Gianola D, Weigel KA (2010) J Anim Sci
  88:497-504.
* Because the project licence is not chosen yet and the data are distributed under the
  package's GPL, the data are downloaded on demand and not committed.

## Conversion (`fetch_data.py`)

* `data/pedigree.csv` (`id, sire, dam`; ids prefixed `C`, unknown parent `0`),
  `data/lactations.csv` (`record_id, id, lact, herd, sire_code, dim, milk, fat, prot, scs,
  first_lact`), `data/lactations_first.csv` (first lactations only).
* **Units**: yields are in **pounds** (mean milk 25,632; the documentation's 11,636 kg
  equals 25,653 lb).
* **Sire repair**: `pedCowsR` lacks the sire of 607 cows whose record carries a sire code
  (`milk$sire`). Every sire code that occurs with a cow of known pedigree sire maps to
  exactly one pedigree sire (checked by the script), so the missing sire is filled from
  that map. `data/pedigree_as_published.csv` keeps the original pedigree.

## Analyses

* `analysis_scs_repeatability.toml`: somatic cell score, all lactations, herd and
  lactation fixed, additive + permanent environment, REML (about 30 s; 196 s before the round-15 sparse-trace rule, methods §35).
* `analysis_first_lactation_multitrait.toml`: first-lactation milk, fat and protein,
  herd fixed, multi-trait REML (about 30 s; 1,314 cows have a first-lactation record).
* The repeatability model for the yields puts the additive variance at zero and ABP
  refuses a ranking (`ABP-E300`); an independent V-form REML agrees (validation report
  §7.19, `benchmarks/real_milk_validation.py`).

Results are for method validation only (gate G5), not a breeding evaluation.
