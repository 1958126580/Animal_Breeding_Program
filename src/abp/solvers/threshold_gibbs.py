"""Gibbs sampler for the threshold (ordered probit) animal model (method id ``threshold.gibbs``).

Model: as :mod:`abp.solvers.threshold` - liability ``l = X b + sum_k Z_k u_k + e``,
``e ~ N(0, I)``, ``u_k ~ N(0, sigma_k^2 K_k)``, ordered thresholds ``tau`` with
``tau_1 = 0`` when the model has an intercept - but the liability variances
``sigma_k^2`` are unknown and sampled together with everything else
(Sorensen, Andersen, Gianola & Korsgaard 1995, Genet Sel Evol 27:229).

Priors: flat on ``b`` and on the free thresholds (subject to ordering); for
each variance a scaled inverse chi-square with ``nu`` degrees of freedom and
scale ``s2``, by default ``nu = -2, s2 = 0``, i.e. **uniform on (0, inf)**
(proper posterior when ``q_k > 2``).

One iteration (the joint posterior is invariant under each step):

1. thresholds and liabilities jointly (Cowles 1996, Stat Comput 6:101): a
   Metropolis-Hastings proposal ``tau*`` (sequential truncated normals that keep
   the order) is accepted with the ratio of the ordinal likelihoods
   ``prod_i P(y_i | eta_i, tau*) / P(y_i | eta_i, tau)`` (liabilities integrated
   out) times the proposal correction; then every liability is drawn from
   ``N(eta_i, 1)`` truncated to ``(tau_{y_i - 1}, tau_{y_i}]``.  The proposal SD
   is tuned during burn-in only (acceptance 0.2-0.5) and frozen afterwards.
2. location effects ``theta = (b, u)`` as one block from their exact
   conditional ``N(C^{-1} W'l, C^{-1})``, ``C = W'W + blockdiag(0, K_k^{-1}/sigma_k^2)``,
   by perturbation (Papandreou & Yuille 2010; Garcia-Cortes & Sorensen 1996):
   ``theta = C^{-1} (W'(l + z_1) + [0; F_k z_2 / sigma_k])`` with ``F_k F_k' = K_k^{-1}``
   and ``z ~ N(0, I)``; the covariance is ``C^{-1}(W'W + K^{-1}/sigma^2)C^{-1} = C^{-1}``.
   ``C`` keeps its sparsity pattern, so the sparse LDL' ordering and symbolic
   factor are computed once and only the numeric factor is redone.
2b. parameter expansion (Liu & Sabatti 2000, Biometrika 87:353; generalised
   Gibbs): for each term the scale ``c > 0`` of the move
   ``(u_k, sigma_k^2) -> (c u_k, c^2 sigma_k^2)`` is drawn from
   ``pi(c u_k, c^2 sigma_k^2, ...) |J| / c``, ``|J| = c^{q_k + 2}``, i.e. from
   ``exp(-||r - c v||^2 / 2) c (c^2 sigma_k^2)-prior`` with ``v = Z_k u_k`` and
   ``r`` the liabilities minus all other effects - a univariate density drawn
   exactly by slice sampling (Neal 2003).  Without this move ``sigma_k^2`` and
   ``u_k`` are strongly dependent when animals have few records, and the chains
   mix very slowly.
3. ``sigma_k^2 | u_k ~ (u_k' K_k^{-1} u_k + nu s2) / chi^2(q_k + nu)``.

Chains, diagnostics and the stopping rule are those of the marker models
(:mod:`abp.solvers.bayes`): at least two chains from independent streams,
rank-normalized split R-hat, bulk/tail ESS and MCSE for every scalar and
every breeding value; the chains are doubled in length until the criteria
pass or ``max_iterations`` is used; otherwise the result is withheld
(``ABP-E405``).  Breeding values are posterior means and PEV is the posterior
variance (it includes the uncertainty of the variances).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import ndtr, ndtri

from ..errors import ABPError
from . import mcmc_diagnostics as dg
from .blup import RandomTerm, TermResult
from .threshold import _log_p, threshold_blup

EBV_STORE_LIMIT = 5e7          #: largest number of stored EBV draws (chains x draws x animals)


@dataclass
class ThresholdGibbsConfig:
    chains: int = 4
    iterations: int = 6000
    burn_in: int = 1000
    thin: int = 5
    seed: int = 20260925
    rhat_max: float = 1.01
    ess_min: float = 400.0
    max_iterations: int = 30000
    nu: float = -2.0              #: prior degrees of freedom (-2 with s2 = 0: uniform)
    s2: float = 0.0               #: prior scale
    start: dict | None = None     #: starting variances (default 0.2 per term)
    fix_variances: bool = False   #: keep the variances at ``start`` (known variances)


@dataclass
class ThresholdGibbsResult:
    terms: dict                   # name -> TermResult (posterior means, posterior variances)
    fixed_mean: np.ndarray
    thresholds_mean: np.ndarray   # tau_1 .. tau_{K-1}
    categories: np.ndarray
    variances: dict               # name -> {"mean","median","sd","q025","q975"} incl. h2
    summaries: dict               # diagnostics of every scalar
    ebv_diagnostics: dict
    converged: bool
    iterations: int
    draws_per_chain: int
    seeds: list
    acceptance: float | None      # threshold MH acceptance after burn-in
    proposal_sd: float | None
    wall_seconds: float
    n_equations: int
    traces: dict = field(default_factory=dict)


def rtruncnorm(rng, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``N(mean, 1)`` truncated to ``(lo, hi]`` by inversion, numerically stable in
    either tail (sampling on the side of the smaller tail probabilities)."""
    a = lo - mean
    b = hi - mean
    out = np.empty_like(mean, dtype=np.float64)
    upper = a > 0                      # whole interval in the upper tail: use survival side
    u = rng.random(mean.shape)
    if np.any(upper):
        pa, pb = ndtr(-a[upper]), ndtr(-b[upper])          # pa > pb
        out[upper] = -ndtri(pb + u[upper] * (pa - pb))
    low = ~upper
    if np.any(low):
        pa, pb = ndtr(a[low]), ndtr(b[low])
        out[low] = ndtri(pa + u[low] * (pb - pa))
    out = np.clip(out, a, b)           # guards the last ulp at the bounds
    return mean + out


def precision_root(k_inv):
    """``z -> F z`` with ``F F' = K^{-1}`` (diagonal, sparse via LDL', or dense Cholesky)."""
    if sp.issparse(k_inv):
        K = sp.csr_matrix(k_inv)
        off = K - sp.diags(K.diagonal())
        if off.count_nonzero() == 0:
            s = np.sqrt(K.diagonal())
            return lambda z: s * z
        from .cholesky import SparseLDL
        f = SparseLDL(K)
        sd = np.sqrt(f.d)
        return lambda z: f.l_times(sd * z)[f.pos]
    L = np.linalg.cholesky(np.asarray(k_inv, dtype=np.float64))
    return lambda z: L @ z


def _align(M: sp.spmatrix, pat: sp.csr_matrix) -> np.ndarray:
    """Values of ``M`` on the pattern of ``pat`` (``M``'s pattern must be contained in it)."""
    M = sp.csr_matrix(M)
    M.eliminate_zeros()                # stored zeros (e.g. exact cancellation in A^-1) add nothing
    M = M.tocoo()
    n = pat.shape[1]
    pc = pat.tocoo()
    key_p = pc.row.astype(np.int64) * n + pc.col
    order = np.argsort(key_p)
    pos = np.searchsorted(key_p[order], M.row.astype(np.int64) * n + M.col)
    if np.any(pos >= key_p.size) or np.any(key_p[order][np.minimum(pos, key_p.size - 1)]
                                           != M.row.astype(np.int64) * n + M.col):
        raise ValueError("pattern mismatch")
    out = np.zeros(pat.nnz)
    np.add.at(out, order[pos], M.data)
    return out


class _Problem:
    """Everything that is shared by the chains."""

    def __init__(self, y, X, terms: list[RandomTerm], intercept: bool,
                 sample_variances: bool = True):
        y = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)) or not np.all(y == np.round(y)):
            raise ABPError("SCHEMA_TYPE", "a threshold trait needs integer category codes")
        self.cats = np.unique(y)
        self.K = self.cats.size
        if self.K < 2:
            raise ABPError("MODEL_NOT_IDENTIFIABLE", "all records are in one category")
        self.yk = np.searchsorted(self.cats, y)
        self.n = y.size
        self.terms = terms
        X = sp.csr_matrix(X)
        self.p = X.shape[1]
        self.W = sp.hstack([X] + [t.Z for t in terms], format="csr")
        self.Wt = self.W.T.tocsr()
        self.offs = {}
        pos = self.p
        for t in terms:
            if sample_variances and t.q <= 2:
                raise ABPError("MODEL_NOT_IDENTIFIABLE", f"term {t.name!r} has {t.q} levels; "
                               "the flat variance prior needs more than 2")
            self.offs[t.name] = (pos, pos + t.q)
            pos += t.q
        self.n_eq = pos
        self.first_free = 1 if intercept else 0
        self.free = np.arange(1 + self.first_free, self.K)      # indices into tau (0..K)
        WtW = (self.Wt @ self.W).tocsr()
        emb = []
        for t in terms:
            a0, _ = self.offs[t.name]
            Kc = sp.coo_matrix(t.k_inv)
            emb.append(sp.csr_matrix((Kc.data, (Kc.row + a0, Kc.col + a0)),
                                     shape=(self.n_eq, self.n_eq)))
        pat = abs(WtW)
        for E in emb:
            pat = pat + abs(E)
        pat = pat.tocsr()
        pat.sum_duplicates()
        pat.sort_indices()
        pat.data[:] = 1.0
        self.pat = pat
        self.wtw_vals = _align(WtW, pat)
        self.k_vals = [_align(E, pat) for E in emb]
        self.k_inv = [sp.csr_matrix(t.k_inv) if sp.issparse(t.k_inv) else np.asarray(t.k_inv)
                      for t in terms]
        self.roots = [precision_root(t.k_inv) for t in terms]

    def coefficient(self, variances: np.ndarray) -> sp.csr_matrix:
        data = self.wtw_vals.copy()
        for kv, v in zip(self.k_vals, variances):
            data += kv / v
        return sp.csr_matrix((data, self.pat.indices, self.pat.indptr), shape=self.pat.shape)


def _slice_sample(logf, x0: float, w: float, rng, max_steps: int = 50) -> float:
    """One update of a univariate slice sampler with stepping out and shrinkage
    (Neal 2003, Ann Stat 31:705); leaves the density ``exp(logf)`` invariant."""
    y = logf(x0) + math.log(max(rng.random(), 1e-300))
    lo = x0 - w * rng.random()
    hi = lo + w
    j = int(rng.random() * max_steps)
    k = max_steps - 1 - j
    while j > 0 and logf(lo) > y:
        lo -= w
        j -= 1
    while k > 0 and logf(hi) > y:
        hi += w
        k -= 1
    while True:
        x = lo + rng.random() * (hi - lo)
        if logf(x) > y:
            return x
        if x < x0:
            lo = x
        else:
            hi = x


def draw_location(P: "_Problem", fac, liab: np.ndarray, var: np.ndarray, rng) -> np.ndarray:
    """One exact draw of ``theta ~ N(C^{-1} W'l, C^{-1})`` by perturbation (``fac``
    factorises ``C`` at ``var``; ``liab`` are the liabilities ``l``)."""
    rhs = P.Wt @ (liab + rng.standard_normal(P.n))
    for t, root, v in zip(P.terms, P.roots, var):
        a0, b0 = P.offs[t.name]
        rhs[a0:b0] += root(rng.standard_normal(t.q)) / math.sqrt(v)
    return fac.solve(rhs)


class _Chain:
    def __init__(self, prob: _Problem, cfg: ThresholdGibbsConfig, rng, start_var: np.ndarray,
                 theta0: np.ndarray, tau0: np.ndarray):
        from .cholesky import SparseLDL
        self.P, self.cfg, self.rng = prob, cfg, rng
        self.var = start_var.copy()
        self.theta = theta0.copy()
        self.tau = np.concatenate([[-np.inf], tau0, [np.inf]])
        self.fac = SparseLDL(prob.coefficient(self.var))
        self.sd = 0.1                       # threshold proposal SD (tuned during burn-in)
        self.n_prop = 0
        self.n_acc = 0
        self.l = self._draw_liabilities(prob.W @ self.theta)

    def _draw_liabilities(self, eta):
        P = self.P
        return rtruncnorm(self.rng, eta, self.tau[P.yk], self.tau[P.yk + 1])

    def _update_thresholds(self, eta, tuning: bool, it: int):
        P = self.P
        if P.free.size == 0:
            return
        old = self.tau
        new = old.copy()
        s = self.sd
        for j in P.free:                    # sequential truncated-normal proposal (Cowles 1996)
            lo, hi = new[j - 1], old[j + 1]
            a, b = (lo - old[j]) / s, (hi - old[j]) / s
            pa, pb = ndtr(a), ndtr(b)
            new[j] = old[j] + s * ndtri(pa + self.rng.random() * (pb - pa))
        # proposal correction log q(tau | tau*) - log q(tau* | tau) (Cowles 1996, eq. 3);
        # the normal densities cancel, the truncation constants do not
        log_q = 0.0
        for j in P.free:
            log_q += math.log(max(ndtr((old[j + 1] - old[j]) / s)
                                  - ndtr((new[j - 1] - old[j]) / s), 1e-300))
            log_q -= math.log(max(ndtr((new[j + 1] - new[j]) / s)
                                  - ndtr((old[j - 1] - new[j]) / s), 1e-300))
        lp_new, _ = _log_p(new[P.yk] - eta, new[P.yk + 1] - eta)
        lp_old, _ = _log_p(old[P.yk] - eta, old[P.yk + 1] - eta)
        log_r = float(lp_new.sum() - lp_old.sum()) + log_q
        self.n_prop += 1
        if math.log(max(self.rng.random(), 1e-300)) < log_r:
            self.tau = new
            self.n_acc += 1
        if tuning and it % 50 == 0 and self.n_prop:
            rate = self.n_acc / self.n_prop
            if rate < 0.2:
                self.sd *= 0.7
            elif rate > 0.5:
                self.sd *= 1.4
            self.n_prop = self.n_acc = 0

    def _scale_moves(self):
        """Parameter-expanded move per random term (see module notes, step 2b)."""
        P, rng, cfg = self.P, self.rng, self.cfg
        eta = P.W @ self.theta
        for k, t in enumerate(P.terms):
            a0, b0 = P.offs[t.name]
            v = t.Z @ self.theta[a0:b0]
            vv = float(v @ v)
            if vv <= 0.0:
                continue
            r = self.l - (eta - v)
            vr = float(v @ r)
            s2v = self.var[k]

            def logf(x):                      # x = log c
                c = math.exp(x)
                out = -0.5 * (c * c * vv - 2.0 * c * vr) + (2.0 - (cfg.nu + 2.0)) * x
                if cfg.s2 > 0:
                    out -= cfg.nu * cfg.s2 / (2.0 * c * c * s2v)
                return out
            x = _slice_sample(logf, 0.0, 0.5, rng)
            c = math.exp(x)
            self.theta[a0:b0] *= c
            self.var[k] *= c * c
            eta = eta + (c - 1.0) * v

    def iterate(self, tuning: bool, it: int):
        P, rng = self.P, self.rng
        eta = P.W @ self.theta
        self._update_thresholds(eta, tuning, it)
        self.l = self._draw_liabilities(eta)
        self.theta = draw_location(P, self.fac, self.l, self.var, rng)
        if not self.cfg.fix_variances:
            self._scale_moves()
            for k, (t, Ki) in enumerate(zip(P.terms, P.k_inv)):
                a0, b0 = P.offs[t.name]
                u = self.theta[a0:b0]
                ss = float(u @ (Ki @ u)) + self.cfg.nu * self.cfg.s2
                self.var[k] = ss / rng.chisquare(t.q + self.cfg.nu)
            self.fac.refactor(P.coefficient(self.var))


def threshold_gibbs(y, X, terms: list[RandomTerm], intercept: bool,
                    cfg: ThresholdGibbsConfig, genetic_term: str | None = None
                    ) -> ThresholdGibbsResult:
    """Run the sampler (see module notes)."""
    t0 = time.perf_counter()
    if cfg.chains < 2:
        raise ABPError("SPEC_INVALID", "at least 2 chains are required for R-hat (4 recommended)")
    if not (0 <= cfg.burn_in < cfg.iterations) or cfg.thin < 1:
        raise ABPError("SPEC_INVALID", "need 0 <= burn_in < iterations and thin >= 1")
    prob = _Problem(y, X, terms, intercept, sample_variances=not cfg.fix_variances)
    names = [t.name for t in terms]
    gen = genetic_term or next((t.name for t in terms if t.genetic), names[0])
    start = np.array([float((cfg.start or {}).get(nm, 0.2)) for nm in names])
    if np.any(start <= 0):
        raise ABPError("COVARIANCE_NOT_PD", "starting variances must be > 0")
    mode = threshold_blup(y, X, terms, {**dict(zip(names, start)), "residual": 1.0}, intercept,
                          compute_pev=False)
    theta0 = np.concatenate([mode.fixed_solution] + [mode.terms[nm].solution for nm in names])
    ss = np.random.SeedSequence(cfg.seed)
    child = ss.spawn(cfg.chains)
    chains = []
    for c in child:
        rng = np.random.default_rng(c)
        v0 = start if cfg.fix_variances else start * np.exp(rng.uniform(-0.7, 0.7, start.size))
        chains.append(_Chain(prob, cfg, rng, v0, theta0, mode.thresholds))
    free_names = [f"tau_{j}" for j in prob.free]
    scal = [f"var_{nm}" for nm in names] + ["h2_liability"] + free_names
    store = {k: [[] for _ in chains] for k in scal}
    g0, g1 = prob.offs[gen]
    qg = g1 - g0
    keep_ebv = qg * (cfg.max_iterations // cfg.thin) * cfg.chains <= EBV_STORE_LIMIT
    ebv_draws = [[] for _ in chains]
    s_theta = np.zeros(prob.n_eq)
    q_theta = np.zeros(prob.n_eq)
    s_tau = np.zeros(prob.K - 1)
    n_saved = 0
    it = 0
    target = cfg.iterations
    while True:
        while it < target:
            it += 1
            tuning = it <= cfg.burn_in
            for c, ch in enumerate(chains):
                if it == cfg.burn_in + 1:
                    ch.n_prop = ch.n_acc = 0
                ch.iterate(tuning, it)
                if it > cfg.burn_in and (it - cfg.burn_in) % cfg.thin == 0:
                    vals = {f"var_{nm}": float(v) for nm, v in zip(names, ch.var)}
                    vals["h2_liability"] = float(ch.var[names.index(gen)] / (ch.var.sum() + 1.0))
                    for j, nm in zip(prob.free, free_names):
                        vals[nm] = float(ch.tau[j])
                    for k in scal:
                        store[k][c].append(vals[k])
                    if keep_ebv:
                        ebv_draws[c].append(ch.theta[g0:g1].astype(np.float32))
                    s_theta += ch.theta
                    q_theta += ch.theta * ch.theta
                    s_tau += ch.tau[1:prob.K]
                    if c == 0:
                        n_saved += 1
        summaries = {k: dg.summarize(np.array(store[k])) for k in scal
                     if np.ptp(np.array(store[k])) > 0}
        ediag = {"not_computed": "per-EBV diagnostics skipped: storing the EBV draws would "
                                 f"exceed {EBV_STORE_LIMIT:.0e} values; only the scalar "
                                 "quantities were diagnosed", "n_animals": int(qg)} \
            if not keep_ebv else {}
        if keep_ebv and n_saved >= 4:
            E = np.array(ebv_draws, dtype=np.float64)          # chains x draws x animals
            rh = np.array([dg.rhat(E[:, :, i]) for i in range(qg)])
            eb = np.array([dg.bulk_ess(E[:, :, i]) for i in range(qg)])
            ediag = {"max_rhat": float(np.nanmax(rh)), "min_ess_bulk": float(np.nanmin(eb)),
                     "n_animals": int(qg)}
        ok = all(dg.passes(s, cfg.rhat_max, cfg.ess_min) for s in summaries.values()) and (
            "max_rhat" not in ediag or (ediag["max_rhat"] < cfg.rhat_max
                                        and ediag["min_ess_bulk"] >= cfg.ess_min))
        if ok or target >= cfg.max_iterations:
            break
        target = min(cfg.max_iterations, target * 2)
    acc_prop = sum(ch.n_prop for ch in chains)
    acc_n = sum(ch.n_acc for ch in chains)
    total = n_saved * len(chains)
    mean = s_theta / total
    var = np.maximum(q_theta / total - mean ** 2, 0.0) * total / max(total - 1, 1)
    vsum = {}
    for nm in names + ["h2_liability"]:
        key = f"var_{nm}" if nm != "h2_liability" else nm
        v = np.array(store[key]).ravel()
        vsum[nm] = {"mean": float(v.mean()), "median": float(np.median(v)),
                    "sd": float(v.std(ddof=1)), "q025": float(np.quantile(v, 0.025)),
                    "q975": float(np.quantile(v, 0.975))}
    out = {}
    for t in terms:
        a0, b0 = prob.offs[t.name]
        sol, pev = mean[a0:b0].copy(), var[a0:b0].copy()
        rel = None
        n_cl = 0
        if t.genetic and t.k_diag is not None:
            rr = 1.0 - pev / (vsum[t.name]["mean"] * np.asarray(t.k_diag, dtype=np.float64))
            rel = np.clip(rr, 0.0, 1.0)
            n_cl = int(np.sum(rel != rr))
        out[t.name] = TermResult(t.name, t.labels, sol, pev, rel, n_cl)
    return ThresholdGibbsResult(
        out, mean[:prob.p].copy(), s_tau / total, prob.cats, vsum, summaries, ediag, bool(ok),
        it, n_saved, [int(c.generate_state(1)[0]) for c in child],
        (acc_n / acc_prop) if acc_prop else None,
        float(np.mean([ch.sd for ch in chains])) if prob.free.size else None,
        time.perf_counter() - t0, prob.n_eq, traces={k: np.array(store[k]) for k in scal})
