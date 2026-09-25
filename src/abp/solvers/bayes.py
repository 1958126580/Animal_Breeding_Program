"""Bayesian marker regression by Gibbs sampling (module M08).

Model (method registry ids ``bayes.*``)
---------------------------------------
``y = X b + W beta + e`` with ``W = M - 2p`` (centred dosages of the counted
allele, missing dosages set to 0 at the reference frequency), a flat prior
on ``b``, ``e ~ N(0, sigma_e^2 I)``, ``sigma_e^2 ~ Inv-scaled-chi^2(nu_e, S_e^2)``
and the marker priors below.  ``pi_0`` is always the prior probability that
a marker effect is **exactly zero**.

==========  =================================================================
method      prior on marker effect ``beta_j``
==========  =================================================================
BRR         N(0, sigma_b^2);  sigma_b^2 ~ Inv-scaled-chi^2(nu, S^2)
BayesA      N(0, v_j);        v_j ~ Inv-scaled-chi^2(nu, S^2), marker specific
BayesB      0 w.p. pi_0, else N(0, v_j); v_j ~ Inv-scaled-chi^2(nu, S^2);
            pi_0 fixed (Gibbs variant with v_j drawn from its prior when
            beta_j = 0, as in Habier et al. 2011)
BayesC      0 w.p. pi_0, else N(0, sigma_b^2) (common variance); pi_0 fixed
BayesCpi    as BayesC with pi_0 ~ Beta(1, 1)
BayesR      0 w.p. pi_0, else N(0, gamma_k sigma_b^2), k = 1..K-1, with
            gamma = (0, 1e-4, 1e-3, 1e-2) scaled so that the default
            sigma_b^2 prior matches the expected genetic variance (see
            ``default_priors``); pi ~ Dirichlet(1, ..., 1).  The mixture
            follows Erbe et al. (2012); the scale parameterization on centred
            (not standardized) genotypes is ABP's and is stated here.
==========  =================================================================

Default hyper-parameters (all recorded in the output): the residual
variance ``V_y`` of ``y`` after the fixed effects is split by ``prior_r2``
(default 0.5) into an expected genetic variance ``V_g = r2 V_y`` and residual
variance ``(1 - r2) V_y``; the expected marker variance is
``V_g / (sum_j 2 p_j (1 - p_j) * E[inclusion] * E[gamma])`` and every scale
``S^2`` is set so that the prior *mean* ``nu S^2 / (nu - 2)`` equals the
target (``nu > 2``).

Sampler: single-site Gibbs with residual updating.  One iteration samples
``b`` jointly, sweeps the markers in order (inclusion indicator with
log-sum-exp probabilities, then the effect), then updates the marker
variance(s), mixture weights and ``sigma_e^2`` from their full conditionals.
All random numbers of a sweep are drawn beforehand from a per-chain NumPy
PCG64 stream (``SeedSequence(seed).spawn(chains)``), so a compiled sweep can
be checked bit-for-bit against the Python reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import ABPError
from . import mcmc_diagnostics as dg

METHODS = ("BRR", "BayesA", "BayesB", "BayesC", "BayesCpi", "BayesR")
BAYESR_GAMMA = (0.0, 1e-4, 1e-3, 1e-2)
#: largest number of stored GEBV draws (animals x draws x chains) for per-GEBV diagnostics
GEBV_STORE_LIMIT = 5e7


@dataclass
class BayesPriors:
    nu: float
    s2: float                 # scale of the marker variance prior
    nu_e: float
    s2_e: float
    pi0: float                # zero-effect probability (fixed or start value)
    gamma: tuple[float, ...]  # BayesR component variance ratios (gamma[0] = 0)
    dirichlet: tuple[float, ...]
    derivation: dict = field(default_factory=dict)


def default_priors(method: str, vy: float, sum2pq: float, prior_r2: float = 0.5,
                   pi0: float = 0.95, nu: float = 5.0, nu_e: float = 5.0) -> BayesPriors:
    """Hyper-parameters from the phenotypic variance (see module docstring)."""
    if method not in METHODS:
        raise ABPError("SPEC_INVALID", f"unknown Bayesian method {method!r}")
    if not (0 < prior_r2 < 1) or nu <= 2 or nu_e <= 2 or not (0 <= pi0 < 1):
        raise ABPError("SPEC_INVALID", "need 0 < prior_r2 < 1, nu > 2, nu_e > 2, 0 <= pi0 < 1")
    vg, ve = prior_r2 * vy, (1 - prior_r2) * vy
    gamma = BAYESR_GAMMA if method == "BayesR" else (0.0, 1.0)
    K = len(gamma)
    dirichlet = tuple([1.0] * K)
    if method == "BayesR":
        pibar = np.array(dirichlet) / sum(dirichlet)
        expected_scale = float(np.dot(pibar, gamma))
        pi0_start = float(pibar[0])
    elif method in ("BRR", "BayesA"):
        expected_scale, pi0_start = 1.0, 0.0
    else:
        expected_scale, pi0_start = 1.0 - pi0, pi0
    v_target = vg / (sum2pq * expected_scale)
    return BayesPriors(nu, v_target * (nu - 2) / nu, nu_e, ve * (nu_e - 2) / nu_e, pi0_start,
                       gamma, dirichlet,
                       {"V_y (after fixed effects)": vy, "prior_r2": prior_r2,
                        "expected_genetic_variance": vg, "expected_residual_variance": ve,
                        "sum_2pq": sum2pq, "marker_variance_prior_mean": v_target,
                        "rule": "S^2 = target * (nu - 2) / nu (prior mean = target)"})


def _scaled_inv_chi2(rng, nu: float, s2_num: float) -> float:
    """Draw ``s2_num / chi2(nu)`` where ``s2_num = nu * S^2`` (+ data terms)."""
    return float(s2_num / rng.chisquare(nu))


def sweep_python(Wf: np.ndarray, wtw: np.ndarray, e: np.ndarray, beta: np.ndarray,
                 delta: np.ndarray, var_j: np.ndarray, log_pi: np.ndarray, comp_var: np.ndarray,
                 sigma_e2: float, method: int, z: np.ndarray, u: np.ndarray) -> None:
    """One in-place sweep over all markers (reference implementation).

    ``method`` codes: 0 = single normal (BRR, BayesA; slab variance ``var_j``),
    1 = zero + normal with ``var_j`` (BayesB, BayesC, BayesCpi),
    2 = zero + normals with variances ``comp_var[1:]`` (BayesR).
    ``log_pi`` are the log mixture weights (index 0 = zero component).
    """
    m = beta.size
    for j in range(m):
        wj = Wf[:, j]
        bj = beta[j]
        rhs = (wj @ e + wtw[j] * bj) / sigma_e2
        if method == 0:
            lhs = wtw[j] / sigma_e2 + 1.0 / var_j[j]
            new = rhs / lhs + z[j] / math.sqrt(lhs)
            delta[j] = 1
        elif method == 1:
            v = var_j[j]
            lhs = wtw[j] / sigma_e2 + 1.0 / v
            logd1 = -0.5 * (math.log(lhs) + math.log(v)) + 0.5 * rhs * rhs / lhs + log_pi[1]
            d = log_pi[0] - logd1
            p1 = 1.0 / (1.0 + math.exp(d)) if d < 700 else 0.0
            if u[j] < p1:
                delta[j] = 1
                new = rhs / lhs + z[j] / math.sqrt(lhs)
            else:
                delta[j] = 0
                new = 0.0
        else:
            K = comp_var.size
            logs = np.empty(K)
            logs[0] = log_pi[0]
            lhs_k = np.empty(K)
            for k in range(1, K):
                v = comp_var[k]
                lhs_k[k] = wtw[j] / sigma_e2 + 1.0 / v
                logs[k] = -0.5 * (math.log(lhs_k[k]) + math.log(v)) + 0.5 * rhs * rhs / lhs_k[k] \
                    + log_pi[k]
            mx = logs.max()
            p = np.exp(logs - mx)
            p /= p.sum()
            k = int(np.searchsorted(np.cumsum(p), u[j], side="right"))
            k = min(k, K - 1)
            delta[j] = k
            new = 0.0 if k == 0 else rhs / lhs_k[k] + z[j] / math.sqrt(lhs_k[k])
        if new != bj:
            e -= wj * (new - bj)
            beta[j] = new


def _sweep(*args) -> None:
    from ..core.pedigree import native_kernel_available
    if native_kernel_available():
        from .. import _native  # type: ignore[attr-defined]
        if hasattr(_native, "bayes_sweep"):
            Wf, wtw, e, beta, delta, var_j, log_pi, comp_var, sigma_e2, method, z, u = args
            # Wf is Fortran-ordered, so Wf.T is a zero-copy C-contiguous (m x n) view
            _native.bayes_sweep(Wf.T, wtw, e, beta, delta, np.ascontiguousarray(var_j), log_pi,
                                comp_var, float(sigma_e2), int(method), z, u)
            return
    sweep_python(*args)


@dataclass
class BayesConfig:
    method: str = "BayesC"
    chains: int = 4
    iterations: int = 6000
    burn_in: int = 1000
    thin: int = 5
    seed: int = 20260925
    prior_r2: float = 0.5
    pi0: float = 0.95
    nu: float = 5.0
    nu_e: float = 5.0
    rhat_max: float = 1.01
    ess_min: float = 400.0
    max_iterations: int = 30000
    fix_variances: dict | None = None      # testing only: {"sigma_b2":..., "sigma_e2":...}
    priors: "BayesPriors | None" = None    # explicit hyper-parameters (default: from prior_r2)
    return_draws: bool = False             # keep GEBV and marker-effect draws (SBC, API)


@dataclass
class BayesResult:
    method: str
    gebv_mean: np.ndarray            # all genotyped animals (W_all rows)
    gebv_sd: np.ndarray
    beta_mean: np.ndarray
    inclusion_prob: np.ndarray       # P(beta_j != 0 | y)
    fixed_mean: np.ndarray
    summaries: dict                  # per monitored scalar
    gebv_diagnostics: dict           # worst R-hat / smallest ESS over GEBVs
    converged: bool
    iterations: int
    draws_per_chain: int
    priors: BayesPriors
    seeds: list[int]
    kernel: str
    wall_seconds: float
    traces: dict = field(default_factory=dict)   # scalar -> (chains, draws)
    ppc: dict = field(default_factory=dict)      # posterior predictive checks
    gebv_draws: np.ndarray | None = None         # (chains, draws, animals) if requested
    beta_draws: np.ndarray | None = None         # (chains, draws, markers) if requested


#: statistics of the posterior predictive check, T(y) for records y
PPC_STATISTICS = ("sd", "skewness", "min", "max")


def _ppc_stats(v: np.ndarray) -> np.ndarray:
    c = v - v.mean()
    sd = float(np.sqrt(np.mean(c * c)))
    skew = float(np.mean(c**3) / sd**3) if sd > 0 else 0.0
    return np.array([sd, skew, float(v.min()), float(v.max())])


def _check_priors(method: str, pri: BayesPriors) -> None:
    ok = pri.nu > 2 and pri.nu_e > 2 and pri.s2 > 0 and pri.s2_e > 0 and 0 <= pri.pi0 < 1
    if method == "BayesR":
        ok = ok and len(pri.gamma) == len(pri.dirichlet) >= 2 and pri.gamma[0] == 0.0 \
            and all(g > 0 for g in pri.gamma[1:]) and all(a > 0 for a in pri.dirichlet)
    else:
        ok = ok and tuple(pri.gamma) == (0.0, 1.0)
    if not ok:
        raise ABPError("SPEC_INVALID", f"explicit priors are not valid for {method}: need "
                       "nu, nu_e > 2, positive scales, 0 <= pi0 < 1 and matching mixture "
                       "components (gamma = (0, 1) except for BayesR)")


class _Chain:
    def __init__(self, cfg: BayesConfig, pri: BayesPriors, X, Wf, wtw, y, rng, method_code):
        self.rng = rng
        m, n = Wf.shape[1], y.size
        self.m = m
        self.method = cfg.method
        self.code = method_code
        self.beta = np.zeros(m)
        self.delta = np.ones(m, dtype=np.int64) if method_code == 0 else np.zeros(m, dtype=np.int64)
        # dispersed starting values drawn from the priors
        self.sigma_e2 = _scaled_inv_chi2(rng, pri.nu_e, pri.nu_e * pri.s2_e)
        self.sigma_b2 = _scaled_inv_chi2(rng, pri.nu, pri.nu * pri.s2)
        self.var_j = np.full(m, self.sigma_b2)
        if cfg.method in ("BayesA", "BayesB"):
            self.var_j = pri.nu * pri.s2 / rng.chisquare(pri.nu, size=m)
        K = len(pri.gamma)
        if cfg.method == "BayesR":
            self.pi = np.array(pri.dirichlet) / sum(pri.dirichlet)
        elif method_code == 1:
            self.pi = np.array([pri.pi0, 1.0 - pri.pi0])
        else:
            self.pi = np.array([0.0, 1.0])
        self.K = K
        if cfg.fix_variances:
            self.sigma_e2 = cfg.fix_variances["sigma_e2"]
            self.sigma_b2 = cfg.fix_variances["sigma_b2"]
            self.var_j[:] = self.sigma_b2
        XtX = X.T @ X
        self.L = np.linalg.cholesky(XtX)
        self.X = X
        self.b = np.linalg.solve(XtX, X.T @ y)
        self.e = y - X @ self.b
        self.cfg, self.pri, self.Wf, self.wtw, self.n = cfg, pri, Wf, wtw, n

    def iterate(self):
        rng, cfg, pri = self.rng, self.cfg, self.pri
        # fixed effects jointly
        r = self.e + self.X @ self.b
        mean = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.X.T @ r))
        noise = np.linalg.solve(self.L.T, rng.standard_normal(self.b.size))
        self.b = mean + math.sqrt(self.sigma_e2) * noise
        self.e = r - self.X @ self.b
        # marker sweep with pre-drawn random numbers
        z = rng.standard_normal(self.m)
        u = rng.random(self.m)
        with np.errstate(divide="ignore"):
            log_pi = np.log(np.maximum(self.pi, 0.0))
        comp_var = np.array(pri.gamma) * self.sigma_b2
        if cfg.method in ("BRR", "BayesC", "BayesCpi"):
            self.var_j[:] = self.sigma_b2
        _sweep(self.Wf, self.wtw, self.e, self.beta, self.delta, self.var_j, log_pi, comp_var,
               self.sigma_e2, self.code, z, u)
        nz = self.delta > 0
        if not cfg.fix_variances:
            if cfg.method in ("BRR", "BayesC", "BayesCpi"):
                self.sigma_b2 = _scaled_inv_chi2(rng, pri.nu + nz.sum(),
                                                 pri.nu * pri.s2 + float(self.beta[nz] @ self.beta[nz]))
            elif cfg.method in ("BayesA", "BayesB"):
                chi_post = rng.chisquare(pri.nu + 1.0, size=self.m)
                chi_prior = rng.chisquare(pri.nu, size=self.m)
                post = (pri.nu * pri.s2 + self.beta**2) / chi_post
                prior = pri.nu * pri.s2 / chi_prior
                self.var_j = np.where(nz, post, prior)
            elif cfg.method == "BayesR":
                g = np.array(pri.gamma)
                ss = float(np.sum(self.beta[nz] ** 2 / g[self.delta[nz]]))
                self.sigma_b2 = _scaled_inv_chi2(rng, pri.nu + nz.sum(), pri.nu * pri.s2 + ss)
            self.sigma_e2 = _scaled_inv_chi2(rng, pri.nu_e + self.n,
                                             pri.nu_e * pri.s2_e + float(self.e @ self.e))
        if cfg.method == "BayesCpi":
            m1 = int(nz.sum())
            p0 = rng.beta(self.m - m1 + 1.0, m1 + 1.0)
            self.pi = np.array([p0, 1.0 - p0])
        elif cfg.method == "BayesR":
            counts = np.bincount(self.delta, minlength=self.K)
            self.pi = rng.dirichlet(np.array(pri.dirichlet) + counts)


def run_bayes(y: np.ndarray, X: np.ndarray, W_train: np.ndarray, W_all: np.ndarray,
              cfg: BayesConfig, sum2pq: float) -> BayesResult:
    """Run ``cfg.chains`` chains, extend until diagnostics pass or the budget ends."""
    import time
    t0 = time.perf_counter()
    if cfg.method not in METHODS:
        raise ABPError("SPEC_INVALID", f"bayes.method must be one of {METHODS}")
    if cfg.chains < 2:
        raise ABPError("SPEC_INVALID", "at least 2 chains are required for R-hat (4 recommended)")
    if not (0 <= cfg.burn_in < cfg.iterations) or cfg.thin < 1:
        raise ABPError("SPEC_INVALID", "need 0 <= burn_in < iterations and thin >= 1")
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Wf = np.asfortranarray(W_train, dtype=np.float64)
    wtw = np.einsum("ij,ij->j", Wf, Wf)
    if np.any(wtw <= 0):
        raise ABPError("GENOTYPE_ZERO_SCALING", "a marker has no variation among the records")
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    vy = float(np.var(y - X @ beta_ols, ddof=X.shape[1]))
    if cfg.priors is not None:
        _check_priors(cfg.method, cfg.priors)
        pri = cfg.priors
    else:
        pri = default_priors(cfg.method, vy, sum2pq, cfg.prior_r2, cfg.pi0, cfg.nu, cfg.nu_e)
    code = 0 if cfg.method in ("BRR", "BayesA") else (2 if cfg.method == "BayesR" else 1)
    ss = np.random.SeedSequence(cfg.seed)
    child = ss.spawn(cfg.chains)
    chains = [_Chain(cfg, pri, X, Wf, wtw, y, np.random.default_rng(c), code) for c in child]
    # posterior predictive replicates use their own streams, so the chains are unaffected
    ppc_rng = [np.random.default_rng(c) for c in ss.spawn(cfg.chains)]
    t_obs = _ppc_stats(y)
    ppc_ge = np.zeros(len(PPC_STATISTICS))
    from ..core.pedigree import native_kernel_available
    kernel = "python_reference"
    if native_kernel_available():
        from .. import _native  # type: ignore[attr-defined]
        if hasattr(_native, "bayes_sweep"):
            kernel = "native_cpp"
    names = ["sigma_e2", "genetic_variance", "h2", "n_nonzero"] + (
        ["sigma_b2"] if cfg.method in ("BRR", "BayesC", "BayesCpi", "BayesR") else
        ["mean_marker_variance"]) + (["pi0"] if cfg.method in ("BayesCpi", "BayesR") else [])
    n_all = W_all.shape[0]
    store = {k: [[] for _ in chains] for k in names}
    gebv_draws = [[] for _ in chains]
    beta_draws = [[] for _ in chains]
    keep_gebv = n_all * (cfg.max_iterations // cfg.thin) * cfg.chains <= GEBV_STORE_LIMIT
    sum_b = np.zeros(X.shape[1])
    sum_beta = np.zeros(W_all.shape[1])
    incl = np.zeros(W_all.shape[1])
    g_sum = np.zeros(n_all)
    g_sq = np.zeros(n_all)
    n_saved = 0
    it = 0
    target = cfg.iterations
    converged = False
    while True:
        while it < target:
            it += 1
            for c, ch in enumerate(chains):
                ch.iterate()
                if it > cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                    g_all = W_all @ ch.beta
                    g_tr = Wf @ ch.beta
                    vg = float(np.var(g_tr))
                    vals = {"sigma_e2": ch.sigma_e2, "genetic_variance": vg,
                            "h2": vg / (vg + ch.sigma_e2),
                            "n_nonzero": float((ch.delta > 0).sum())}
                    if "sigma_b2" in names:
                        vals["sigma_b2"] = ch.sigma_b2
                    else:
                        vals["mean_marker_variance"] = float(ch.var_j.mean())
                    if "pi0" in names:
                        vals["pi0"] = float(ch.pi[0])
                    for k in names:
                        store[k][c].append(vals[k])
                    if keep_gebv:
                        gebv_draws[c].append(g_all)
                    if cfg.return_draws:
                        beta_draws[c].append(ch.beta.copy())
                    y_rep = ch.X @ ch.b + g_tr + math.sqrt(ch.sigma_e2) * \
                        ppc_rng[c].standard_normal(g_tr.size)
                    ppc_ge += _ppc_stats(y_rep) >= t_obs
                    sum_b += ch.b
                    sum_beta += ch.beta
                    incl += ch.delta > 0
                    g_sum += g_all
                    g_sq += g_all * g_all
                    if c == 0:
                        n_saved += 1
        summaries = {k: dg.summarize(np.array(store[k])) for k in names
                     if np.ptp(np.array(store[k])) > 0}
        gdiag = {} if keep_gebv else {
            "not_computed": "per-GEBV diagnostics skipped: storing the GEBV draws would exceed "
                            f"{GEBV_STORE_LIMIT:.0e} values; only the scalar quantities were "
                            "diagnosed", "n_animals": n_all}
        if keep_gebv and n_saved >= 4:
            G = np.array(gebv_draws)          # chains x draws x animals
            rh = [dg.rhat(G[:, :, i]) for i in range(n_all)]
            eb = [dg.bulk_ess(G[:, :, i]) for i in range(n_all)]
            gdiag = {"max_rhat": float(np.nanmax(rh)), "min_ess_bulk": float(np.nanmin(eb)),
                     "n_animals": n_all}
        ok = all(dg.passes(s, cfg.rhat_max, cfg.ess_min) for k, s in summaries.items()
                 if k != "n_nonzero") and ("max_rhat" not in gdiag or (
                     gdiag["max_rhat"] < cfg.rhat_max and gdiag["min_ess_bulk"] >= cfg.ess_min))
        if ok or cfg.fix_variances:
            converged = bool(ok)
            break
        if target >= cfg.max_iterations:
            break
        target = min(cfg.max_iterations, target * 2)
    total = n_saved * len(chains)
    mean_g = g_sum / total
    sd_g = np.sqrt(np.maximum(g_sq / total - mean_g**2, 0.0) * total / max(total - 1, 1))
    ppc = {"statistics": {k: {"observed": float(t_obs[i]), "p_value": float(ppc_ge[i] / total)}
                          for i, k in enumerate(PPC_STATISTICS)},
           "definition": "p = P(T(y_rep) >= T(y) | y) over the saved draws, y_rep = Xb + "
                         "W beta + e_rep with e_rep ~ N(0, sigma_e^2 I); values near 0 or 1 "
                         "indicate a mismatch between model and data",
           "n_replicates": int(total)}
    return BayesResult(cfg.method, mean_g, sd_g, sum_beta / total, incl / total, sum_b / total,
                       summaries, gdiag, converged, it, n_saved, pri,
                       [int(c.generate_state(1)[0]) for c in child], kernel,
                       time.perf_counter() - t0,
                       traces={k: np.array(store[k]) for k in names}, ppc=ppc,
                       gebv_draws=np.array(gebv_draws) if (cfg.return_draws and keep_gebv) else None,
                       beta_draws=np.array(beta_draws) if cfg.return_draws else None)
