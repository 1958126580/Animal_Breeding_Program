# GBLUP of the example-21 mice with the R packages rrBLUP (Endelman 2011) and sommer
# (Covarrubias-Pazaran 2016) for the round-16 comparison with ABP
# (benchmarks/real_mice_software_comparison.py).
# Usage: Rscript mice_gblup_fit.R <data dir> <out dir>
# Writes <out dir>/G_rrBLUP.bin (n x n doubles, column-major), <out dir>/r_results.json.
suppressPackageStartupMessages({library(rrBLUP); library(sommer)})
args <- commandArgs(trailingOnly = TRUE)
dd <- args[1]; od <- args[2]
ph <- read.csv(file.path(dd, "phenotypes.csv"), colClasses = "character")
geno <- read.csv(file.path(dd, "genotypes.csv"), check.names = FALSE)
stopifnot(identical(as.character(geno$id), ph$id))
M <- as.matrix(geno[, -1]) - 1                     # rrBLUP coding -1/0/1
t0 <- proc.time()[["elapsed"]]
G <- A.mat(M, min.MAF = 0, return.imputed = FALSE)  # VanRaden (2008) method 1
tG <- proc.time()[["elapsed"]] - t0
n <- nrow(G)
writeBin(as.vector(G), file.path(od, "G_rrBLUP.bin"))
K <- G + 0.01 * diag(n)
dimnames(K) <- list(ph$id, ph$id)
ph$sex <- factor(ph$sex); ph$year_season <- factor(ph$year_season)
ph$cage <- factor(ph$cage); ph$id <- factor(ph$id, levels = ph$id)
X <- model.matrix(~ sex + year_season, ph)
res <- list(versions = list(R = R.version.string,
                            rrBLUP = as.character(packageVersion("rrBLUP")),
                            sommer = as.character(packageVersion("sommer"))),
            seconds_A_mat = tG)
for (tr in c("bw", "body_length")) {
  y <- as.numeric(ph[[tr]])
  t0 <- proc.time()[["elapsed"]]
  ms <- mixed.solve(y, K = K, X = X, method = "REML")
  t1 <- proc.time()[["elapsed"]] - t0
  d <- data.frame(y = y, sex = ph$sex, year_season = ph$year_season, cage = ph$cage,
                  id = ph$id)
  t0 <- proc.time()[["elapsed"]]
  m <- mmer(y ~ sex + year_season, random = ~ vsr(id, Gu = K) + cage,
            rcov = ~ units, data = d, verbose = FALSE, tolParConvLL = 1e-6)
  t2 <- proc.time()[["elapsed"]] - t0
  s <- summary(m)$varcomp
  u <- m$U[["u:id"]][["y"]]
  res[[tr]] <- list(
    rrBLUP_animal_only = list(variances = list(animal = ms$Vu, residual = ms$Ve),
                              loglik = ms$LL, seconds = t1,
                              ebv = as.list(setNames(ms$u, ph$id))),
    sommer_animal_cage = list(variances = list(animal = s[1, "VarComp"],
                                               cage = s[2, "VarComp"],
                                               residual = s[3, "VarComp"]),
                              varcomp_rows = rownames(s),
                              loglik = as.numeric(m$monitor[1, ncol(m$monitor)]),
                              converged = m$convergence, seconds = t2,
                              ebv = as.list(u[ph$id])))
}
j <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x))) paste0("{", paste0('"', names(x), '": ', sapply(x, j), collapse = ", "), "}")
    else paste0("[", paste(sapply(x, j), collapse = ", "), "]")
  } else if (is.character(x)) {
    if (length(x) == 1) paste0('"', x, '"') else paste0("[", paste0('"', x, '"', collapse = ", "), "]")
  } else if (is.logical(x)) tolower(as.character(x)) else format(x, digits = 17)
}
writeLines(j(res), file.path(od, "r_results.json"))
