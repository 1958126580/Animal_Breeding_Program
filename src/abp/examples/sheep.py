"""Synthetic sheep flock generator (clearly labelled SYNTHETIC data).

Purpose
-------
Provide a reproducible, realistic-looking data set for the examples, the user
manual and simulation-based checks (acceptance gate G4), with the true
breeding values kept in a separate ``truth/`` folder that analyses never read.
Nothing here represents a real breed, flock or industry parameter.

Independence
------------
The generator does **not** use ABP's pedigree, relationship or solver code.
Genetic values arise from explicit gene dropping: founders receive random
haplotypes; each gamete is produced by meiosis with crossovers (Haldane map,
Poisson number of crossovers per chromosome); true breeding values (TBV) are
sums of additive QTL effects.  Relationships are therefore whatever the
simulated genomes imply, which lets pedigree- and marker-based analyses be
checked against the same truth.

Population and traits (all values SYNTHETIC)
--------------------------------------------
* Two flocks, founders born 2018-2019, lambing seasons 2021-2025, overlapping
  generations; rams are selected on their own weaning weight.
* Lamb traits: weaning weight ``wwt`` (kg, growth), C-site fat depth ``fat``
  (mm, carcass), log faecal egg count ``fec`` (ln(epg + 1), health).
* Ewe trait: number of lambs born ``nlb`` (reproduction), generated on a
  liability scale with a permanent-environment effect and thresholds, so the
  recorded counts 1/2/3 are *not* Gaussian - analysing them with a linear model
  is an approximation (as in common practice) and is labelled as such.
* Genotypes: 2,000 SNPs (dosage of the counted ALT allele, ~0.5% missing) on
  genotyped rams and a sample of recent lambs.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TRAITS = ("wwt", "fat", "fec", "nlb")


@dataclass
class SheepSimConfig:
    seed: int = 20260925
    n_chrom: int = 10
    chrom_len_morgan: float = 1.0
    snp_per_chrom: int = 200
    qtl_per_chrom: int = 50
    n_founder_rams: int = 12
    n_founder_ewes: int = 200
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    rams_per_year: int = 12
    ewe_replacement_rate: float = 0.35
    ewe_max_age: int = 6
    ewe_survival: float = 0.85
    # additive genetic covariance at the founder base (wwt, fat, fec, nlb-liability)
    genetic_sd: tuple[float, ...] = (2.0, 0.5, 0.55, 0.30)
    genetic_corr: tuple[tuple[float, ...], ...] = (
        (1.0, 0.30, 0.00, 0.10),
        (0.30, 1.0, 0.00, 0.10),
        (0.00, 0.00, 1.0, -0.10),
        (0.10, 0.10, -0.10, 1.0))
    residual_sd: tuple[float, float, float] = (3.5, 0.7, 0.95)   # wwt, fat, fec
    residual_corr: tuple[tuple[float, ...], ...] = (
        (1.0, 0.30, 0.10), (0.30, 1.0, 0.10), (0.10, 0.10, 1.0))
    nlb_pe_sd: float = 0.30
    nlb_residual_sd: float = 0.90
    nlb_thresholds: tuple[float, float] = (-0.10, 1.45)
    p_fat_recorded: float = 0.6
    p_fec_recorded: float = 0.5
    p_genotyped_recent_lamb: float = 0.5
    genotype_missing_rate: float = 0.005
    n_wwt_typos: int = 3
    assembly: str = "SYNTHETIC-OVINE-ASSEMBLY-0"
    extra: dict = field(default_factory=dict)


class _Genome:
    """Founder haplotypes, meiosis and locus bookkeeping."""

    def __init__(self, cfg: SheepSimConfig, rng: np.random.Generator):
        self.cfg = cfg
        per = cfg.snp_per_chrom + cfg.qtl_per_chrom
        self.n_loci = cfg.n_chrom * per
        self.chrom = np.repeat(np.arange(cfg.n_chrom), per)
        pos = [np.sort(rng.uniform(0, cfg.chrom_len_morgan, per)) for _ in range(cfg.n_chrom)]
        self.pos = np.concatenate(pos)
        kind = []
        for _ in range(cfg.n_chrom):
            k = np.array(["snp"] * cfg.snp_per_chrom + ["qtl"] * cfg.qtl_per_chrom)
            rng.shuffle(k)
            kind.append(k)
        self.kind = np.concatenate(kind)
        self.freq = np.where(self.kind == "snp", rng.uniform(0.05, 0.95, self.n_loci),
                             rng.uniform(0.02, 0.98, self.n_loci))
        self.starts = np.searchsorted(self.chrom, np.arange(cfg.n_chrom))
        self.ends = np.append(self.starts[1:], self.n_loci)

    def founder(self, rng) -> np.ndarray:
        return (rng.random((2, self.n_loci)) < self.freq).astype(np.int8)

    def gamete(self, genome: np.ndarray, rng) -> np.ndarray:
        out = np.empty(self.n_loci, dtype=np.int8)
        for c in range(self.cfg.n_chrom):
            a, b = self.starts[c], self.ends[c]
            n_co = rng.poisson(self.cfg.chrom_len_morgan)
            strand = np.zeros(b - a, dtype=np.int8)
            if n_co:
                cuts = np.sort(rng.uniform(0, self.cfg.chrom_len_morgan, n_co))
                strand = (np.searchsorted(cuts, self.pos[a:b]) % 2).astype(np.int8)
            start = rng.integers(2)
            strand = strand ^ start
            out[a:b] = np.where(strand == 0, genome[0, a:b], genome[1, a:b])
        return out


def simulate(cfg: SheepSimConfig) -> dict:
    """Run the simulation; returns in-memory tables (see :func:`write_sheep_example`)."""
    rng = np.random.default_rng(cfg.seed)
    g = _Genome(cfg, rng)
    qtl = np.flatnonzero(g.kind == "qtl")
    snp = np.flatnonzero(g.kind == "snp")
    Gcorr = np.array(cfg.genetic_corr)
    Gsd = np.array(cfg.genetic_sd)
    G0 = Gcorr * np.outer(Gsd, Gsd)
    raw = rng.standard_normal((qtl.size, 4)) @ np.linalg.cholesky(Gcorr).T

    animals: list[dict] = []
    genomes: list[np.ndarray] = []

    def add(aid, sire, dam, sex, born, flock, genome):
        animals.append({"id": aid, "sire": sire, "dam": dam, "sex": sex, "born": born,
                        "flock": flock})
        genomes.append(genome)
        return len(animals) - 1

    for k in range(cfg.n_founder_rams):
        add(f"R{k + 1:04d}", "0", "0", "M", f"{2018 + k % 2}-04-{10 + k % 18:02d}",
            f"F{k % 2 + 1}", g.founder(rng))
    for k in range(cfg.n_founder_ewes):
        add(f"E{k + 1:04d}", "0", "0", "F", f"{2018 + k % 2}-04-{1 + k % 28:02d}",
            f"F{k % 2 + 1}", g.founder(rng))

    # Scale QTL effects so that the founder TBV covariance equals G0 exactly.
    dos_f = np.array([gn[0, qtl] + gn[1, qtl] for gn in genomes], dtype=float)
    tbv_f = dos_f @ raw
    S = np.cov(tbv_f.T)
    T = np.linalg.solve(np.linalg.cholesky(S).T, np.linalg.cholesky(G0).T)
    effects = raw @ T
    base_mean = (dos_f @ effects).mean(axis=0)

    def tbv(idx):
        gn = genomes[idx]
        return (gn[0, qtl] + gn[1, qtl]) @ effects - base_mean

    Rsd = np.array(cfg.residual_sd)
    Rc = np.linalg.cholesky(np.array(cfg.residual_corr) * np.outer(Rsd, Rsd))
    cg_eff = {}
    lamb_rows, lambing_rows = [], []
    ewe_pe: dict[int, float] = {}
    alive_ewes = {i for i, a in enumerate(animals) if a["sex"] == "F"}
    rams_prev = [i for i, a in enumerate(animals) if a["sex"] == "M"]
    lamb_no = 0
    male_lambs_by_year: dict[int, list[tuple[int, float]]] = {}
    for year in cfg.years:
        # rams: founders in the first season, then the best young males on own wwt
        if year == cfg.years[0]:
            rams = rams_prev
        else:
            cand = male_lambs_by_year.get(year - 1, [])
            cand = sorted(cand, key=lambda t: -t[1])
            rams = [i for i, _ in cand[:cfg.rams_per_year]]
        ewes = sorted(e for e in alive_ewes
                      if year - 1 >= int(animals[e]["born"][:4])
                      and year - int(animals[e]["born"][:4]) <= cfg.ewe_max_age)
        for e in ewes:
            flock = animals[e]["flock"]
            flock_rams = [r for r in rams if animals[r]["flock"] == flock] or rams
            ram = flock_rams[rng.integers(len(flock_rams))]
            if e not in ewe_pe:
                ewe_pe[e] = rng.normal(0, cfg.nlb_pe_sd)
            parity = sum(1 for r in lambing_rows if r["id"] == animals[e]["id"]) + 1
            liab = tbv(e)[3] + ewe_pe[e] + rng.normal(0, cfg.nlb_residual_sd) \
                + (-0.25 if parity == 1 else 0.0)
            n_lambs = 1 + int(liab > cfg.nlb_thresholds[0]) + int(liab > cfg.nlb_thresholds[1])
            dam_age = year - int(animals[e]["born"][:4])
            lambing_rows.append({"record_id": f"L{year}-{animals[e]['id']}",
                                 "id": animals[e]["id"], "year": year, "parity": min(parity, 4),
                                 "flock": flock, "nlb": n_lambs})
            day = 1 + int(rng.integers(0, 28))
            for _ in range(n_lambs):
                lamb_no += 1
                sex = "M" if rng.random() < 0.5 else "F"
                aid = f"{'M' if sex == 'M' else 'W'}{year % 100:02d}{lamb_no:05d}"
                genome = np.vstack([g.gamete(genomes[ram], rng), g.gamete(genomes[e], rng)])
                i = add(aid, animals[ram]["id"], animals[e]["id"], sex, f"{year}-04-{day:02d}",
                        flock, genome)
                u = tbv(i)
                cg = f"{flock}-{year}-{sex}"
                if cg not in cg_eff:
                    cg_eff[cg] = rng.normal(0, [2.0, 0.3, 0.4])
                eps = Rc @ rng.standard_normal(3)
                bt = min(n_lambs, 3)
                wwt = 30 + cg_eff[cg][0] + (2.0 if sex == "M" else 0) - 3.0 * (bt - 1) \
                    - (1.5 if dam_age == 1 else 0) + u[0] + eps[0]
                fat = 3.0 + cg_eff[cg][1] - (0.3 if sex == "M" else 0) - 0.2 * (bt - 1) + u[1] + eps[1]
                fec = 6.0 + cg_eff[cg][2] + (0.2 if sex == "M" else 0) + u[2] + eps[2]
                lamb_rows.append({"id": aid, "flock": flock, "year": year, "sex": sex,
                                  "birth_type": str(bt),
                                  "dam_age": "1" if dam_age == 1 else ("2-5" if dam_age <= 5 else "6+"),
                                  "cg": cg, "wwt": round(float(wwt), 2),
                                  "fat": round(float(fat), 2) if rng.random() < cfg.p_fat_recorded else None,
                                  "fec": round(float(fec), 3) if rng.random() < cfg.p_fec_recorded else None})
                if sex == "M":
                    male_lambs_by_year.setdefault(year, []).append((i, float(wwt)))
                elif rng.random() < cfg.ewe_replacement_rate:
                    alive_ewes.add(i)  # retained as a replacement ewe
        alive_ewes = {e for e in alive_ewes if rng.random() < cfg.ewe_survival
                      or int(animals[e]["born"][:4]) >= year}

    # data-entry errors (to be caught by QC): weight typed with a misplaced decimal
    typo_idx = rng.choice(len(lamb_rows), cfg.n_wwt_typos, replace=False)
    for k in typo_idx:
        lamb_rows[k]["wwt"] = round(lamb_rows[k]["wwt"] * 10, 1)

    # first-parity NLB on the ewe's own row for the multi-trait example
    first = {}
    for r in lambing_rows:
        if r["parity"] == 1 and r["id"] not in first:
            first[r["id"]] = r
    for r in lamb_rows:
        f = first.get(r["id"])
        r["nlb1"] = f["nlb"] if f else None
        r["lambing1_year"] = f"{f['flock']}-{f['year']}" if f else None

    # genotyped animals: rams used as sires + a sample of the last two lamb crops
    sires_used = {a["sire"] for a in animals if a["sire"] != "0"}
    recent = [a["id"] for a in animals
              if a["born"][:4] in {str(cfg.years[-1]), str(cfg.years[-2])}]
    geno_ids = sorted(sires_used) + sorted(a for a in recent if rng.random() < cfg.p_genotyped_recent_lamb
                                           and a not in sires_used)
    index = {a["id"]: i for i, a in enumerate(animals)}
    dos = np.array([genomes[index[a]][0, snp] + genomes[index[a]][1, snp] for a in geno_ids],
                   dtype=float)
    miss = rng.random(dos.shape) < cfg.genotype_missing_rate
    nonamb = [("A", "C"), ("A", "G"), ("C", "T"), ("G", "T")]
    markers = []
    for j, loc in enumerate(snp):
        ref, alt = nonamb[rng.integers(4)]
        markers.append({"marker_id": f"snp{j + 1:05d}", "chrom": int(g.chrom[loc]) + 1,
                        "pos": int(g.pos[loc] * 1e8) + 1, "ref": ref, "alt": alt,
                        "counted_allele": alt, "assembly": cfg.assembly})
    tbv_rows = [{"id": a["id"], **{f"tbv_{t}": float(v) for t, v in zip(TRAITS, tbv(i))}}
                for i, a in enumerate(animals)]
    return {"animals": animals, "lambs": lamb_rows, "lambing": lambing_rows,
            "geno_ids": geno_ids, "dosage": dos, "missing": miss, "markers": markers,
            "tbv": tbv_rows, "G0": G0, "typos": [lamb_rows[k]["id"] for k in typo_idx]}


def _write(path: Path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["NA" if v is None else v for v in r])


def write_sheep_example(out: Path, seed: int = 20260925, force: bool = False,
                        cfg: SheepSimConfig | None = None) -> Path:
    """Write the synthetic data set to ``out`` (``data/`` and ``truth/``)."""
    cfg = cfg or SheepSimConfig(seed=seed)
    cfg.seed = seed
    out = Path(out)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists (use force=True)")
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "truth").mkdir(parents=True)
    sim = simulate(cfg)
    _write(out / "data" / "pedigree.csv", ["id", "sire", "dam", "sex", "birth_date"],
           [[a["id"], a["sire"], a["dam"], a["sex"], a["born"]] for a in sim["animals"]])
    cols = ["id", "flock", "year", "sex", "birth_type", "dam_age", "cg", "wwt", "fat", "fec",
            "nlb1", "lambing1_year"]
    _write(out / "data" / "lambs.csv", cols, [[r[c] for c in cols] for r in sim["lambs"]])
    cols = ["record_id", "id", "year", "parity", "flock", "nlb"]
    _write(out / "data" / "lambing.csv", cols, [[r[c] for c in cols] for r in sim["lambing"]])
    mk = sim["markers"]
    _write(out / "data" / "markers.csv", ["marker_id", "chrom", "pos", "ref", "alt",
                                          "counted_allele", "assembly"],
           [[m[c] for c in ("marker_id", "chrom", "pos", "ref", "alt", "counted_allele",
                            "assembly")] for m in mk])
    dos, miss = sim["dosage"], sim["missing"]
    rows = []
    for i, a in enumerate(sim["geno_ids"]):
        rows.append([a] + ["NA" if miss[i, j] else str(int(dos[i, j])) for j in range(dos.shape[1])])
    _write(out / "data" / "genotypes.csv", ["id"] + [m["marker_id"] for m in mk], rows)
    _write(out / "truth" / "tbv.csv", ["id"] + [f"tbv_{t}" for t in TRAITS],
           [[r["id"]] + [r[f"tbv_{t}"] for t in TRAITS] for r in sim["tbv"]])
    params = asdict(cfg)
    params["founder_genetic_covariance"] = sim["G0"].tolist()
    params["residual_covariance_lamb_traits"] = (np.array(cfg.residual_corr) * np.outer(
        cfg.residual_sd, cfg.residual_sd)).tolist()
    params["wwt_typo_animals"] = sim["typos"]
    params["note"] = ("SYNTHETIC parameters for software demonstration only. nlb genetic "
                      "parameters are on the liability scale; recorded nlb counts are "
                      "thresholded and therefore not Gaussian.")
    (out / "truth" / "simulation_parameters.json").write_text(json.dumps(params, indent=2),
                                                              encoding="utf-8")
    (out / "README.md").write_text(
        "# Synthetic sheep flock (SYNTHETIC DATA)\n\n"
        f"Generated by `abp simulate-sheep --seed {seed}` (ABP). `data/` holds the analysis "
        "inputs; `truth/` holds the simulated true breeding values and parameters, which "
        "analyses must never read (they are used only for simulation-based checks).\n",
        encoding="utf-8")
    return out
