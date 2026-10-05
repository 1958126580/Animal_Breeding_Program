# Fits the example-20 models with the R package pedigreemm (Vazquez et al. 2010) for the
# round-16 comparison with ABP (benchmarks/real_milk_pedigreemm_comparison.py).
# Usage: Rscript pedigreemm_fit.R <data dir> <out.json>
# Needs lme4 and pedigreemm (built from the CRAN sources; see the comparison script).
suppressPackageStartupMessages({library(pedigreemm); library(lme4)})
args <- commandArgs(trailingOnly = TRUE)
dd <- args[1]; out <- args[2]
ped <- read.csv(file.path(dd, "pedigree.csv"), colClasses = "character")
rec <- read.csv(file.path(dd, "lactations.csv"), colClasses = c(id = "character",
                herd = "character", lact = "character"))
# pedigreemm needs parents before offspring: ABP's pedigree.csv is already ordered so
ped$sire[ped$sire == "0"] <- NA
ped$dam[ped$dam == "0"] <- NA
P <- pedigree(sire = ped$sire, dam = ped$dam, label = ped$id)
rec$herd <- factor(rec$herd); rec$lact <- factor(rec$lact)
rec$id <- factor(rec$id); rec$pe <- rec$id
first <- droplevels(rec[rec$lact == "1", ])
fit1 <- function(f, d) {
  t0 <- proc.time()[["elapsed"]]
  # one record per cow in the first-lactation model: the animal effect is identified by
  # the pedigree, so lme4's "levels < observations" checks are switched off
  ctl <- lmerControl(check.nobs.vs.nlev = "ignore", check.nobs.vs.rankZ = "ignore",
                     check.nobs.vs.nRE = "ignore")
  m <- pedigreemm(f, pedigree = list(id = P), data = d, REML = TRUE, control = ctl)
  vc <- as.data.frame(VarCorr(m))
  v <- setNames(vc$vcov, ifelse(vc$grp == "Residual", "residual", vc$grp))
  u <- ranef(m)$id[, 1]                    # BLUP of the additive effect (original scale)
  list(variances = as.list(v), reml_criterion = REMLcrit(m),
       n = nrow(d), p = ncol(model.matrix(m)), rank_x = qr(model.matrix(m))$rank,
       seconds = proc.time()[["elapsed"]] - t0,
       ebv = setNames(as.list(u), rownames(ranef(m)$id)))
}
res <- list(
  versions = list(R = R.version.string, lme4 = as.character(packageVersion("lme4")),
                  pedigreemm = as.character(packageVersion("pedigreemm")),
                  Matrix = as.character(packageVersion("Matrix"))),
  milk_first_lactation = fit1(milk ~ herd + (1 | id), first),
  scs_repeatability = fit1(scs ~ herd + lact + (1 | id) + (1 | pe), rec),
  milk_repeatability = fit1(milk ~ herd + lact + (1 | id) + (1 | pe), rec))
# minimal JSON writer (no jsonlite dependency)
j <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x))) paste0("{", paste0('"', names(x), '": ', sapply(x, j), collapse = ", "), "}")
    else paste0("[", paste(sapply(x, j), collapse = ", "), "]")
  } else if (is.character(x)) paste0('"', x, '"') else format(x, digits = 17)
}
writeLines(j(res), out)
