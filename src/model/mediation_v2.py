"""KineBalance v2: joint spike-and-slab two-outcome mediation.

Replaces the v1 vanilla-horseshoe two-stage sampler (``mediation.py``) with a
JOINT spike-and-slab model that puts a selection indicator on BOTH paths and ties
them with a *subsetting* prior, so the mediation composite null (alpha*beta = 0)
is represented explicitly.

Model (per phylogenetic balance ``k``, outcome ``o``):

    Stage 1 (mediator, shared across outcomes):
        B_k = Zu1 @ xi1_k + alpha_k * X + eta_k,     eta_k ~ N(0, sM2_k)
        gamma_k ~ Bernoulli(pi_a)                    # exposure regulates balance k
        alpha_k | gamma_k=1 ~ N(0, v_alpha)          # fixed Gaussian slab
        alpha_k | gamma_k=0 = 0                       # spike

    Stage 2 (outcome o):
        Y_o = Zu2 @ xi2_o + sum_k beta_ko * B_k + eps_o,   eps_o ~ N(0, so2_o)
        delta_ko ~ Bernoulli(pi_b_o * gamma_k)       # SUBSETTING: needs gamma_k=1
        beta_ko | delta_ko=1 ~ N(0, v_beta)          # fixed Gaussian slab
        beta_ko | delta_ko=0 = 0

    Natural indirect effect:  NIE_ko = alpha_k * beta_ko.

The slab is a FIXED Gaussian, v_alpha = v_beta = slab_scale**2 (weakly informative
in standardized units). We deliberately do NOT use an adaptive (regularized /
Finnish) horseshoe slab: the horseshoe local-scale full conditional is not
conjugate under the bounded slab, so a horseshoe Gibbs step would not target the
stated model. With a fixed slab every full conditional is closed-form and the
production sampler targets EXACTLY the model the exact enumeration validates
(src/model/validation.py). Because the slab is bounded by construction, the v1
adaptive-jitter / pinv / variance-clipping patches are unnecessary (a fixed tiny
jitter suffices). The price is one user-set scale; its influence on selection is
quantified by a slab-sensitivity sweep.

Everything is conjugate -> pure-NumPy Gibbs. Posterior inclusion probabilities
(PIPs) and the composite-null specificity decomposition live in ``specificity_v2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np

from src.model.mediation import _inv_gamma, _build_unpenalized, _batch_dummies

__all__ = ["SpikeSlabSamples", "SpikeSlabMediation"]

_JITTER = 1e-10


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


def _ridge_block_draw(rng, Z, ZtZ, r, sigma2, prior_prec):
    """Draw a Gaussian regression block ~ N(P^-1 Z^T r / sigma2, P^-1),
    P = Z^T Z / sigma2 + prior_prec*I (Cholesky, fixed jitter)."""
    p = Z.shape[1]
    if p == 0:
        return np.zeros(0)
    P = ZtZ / sigma2 + (prior_prec + _JITTER) * np.eye(p)
    b = Z.T @ r / sigma2
    L = np.linalg.cholesky(P)
    w = np.linalg.solve(L, b)
    m = np.linalg.solve(L.T, w)
    z = rng.standard_normal(p)
    u = np.linalg.solve(L.T, z)
    return m + u


def pi_beta_posterior(prior, n_d: int, n_g: int, K: int, subsetting: bool):
    """Beta full-conditional parameters of an endpoint-path inclusion rate pi_beta.

    Under the coupled (subsetting) prior delta_k ~ Bern(pi_beta) only for balances with
    gamma_k = 1, so the Bernoulli trials are the n_g exposure-regulated balances. Under
    the decoupled prior delta_k ~ Bern(pi_beta) for every balance, so there are K trials.
    """
    trials = max(n_g, n_d) if subsetting else K
    return float(prior[0]) + n_d, float(prior[1]) + max(trials - n_d, 0)


@dataclass
class SpikeSlabSamples:
    """Posterior draws from :class:`SpikeSlabMediation`.

    Attributes
    ----------
    mediator : dict
        ``alpha`` (S, K) exposure->balance coefficients (0 when not selected),
        ``gamma`` (S, K) exposure-regulation indicators.
    outcomes : dict
        Per outcome name: ``beta`` (S, K), ``delta`` (S, K) indicators,
        ``nie`` (S, K) natural indirect effects in ORIGINAL outcome-SD units,
        ``sigma`` (S,).
    meta : dict
        Run metadata (names, dims, y_scale, n_iter/burn/thin, seed, ...).
    """

    mediator: Dict[str, np.ndarray]
    outcomes: Dict[str, Dict[str, np.ndarray]]
    meta: Dict[str, object] = field(default_factory=dict)


class SpikeSlabMediation:
    """Joint spike-and-slab two-outcome compositional mediation (Gibbs).

    Parameters
    ----------
    n_iter, burn, thin : int
        Total sweeps, burn-in, thinning (the binary indicator chains mix slower
        than continuous coefficients, so defaults are larger than v1).
    pi_a, pi_b : tuple(float, float)
        Beta hyperprior (a, b) on the inclusion rates pi_alpha / pi_beta
        (default Beta(1, 9): prior mean inclusion 0.1).
    slab_scale : float
        Fixed Gaussian-slab scale; the slab variance is ``v = slab_scale**2``
        (standardized units, so 2.0 -> v=4 allows large-but-finite effects). This
        is the one user-set knob of the prior.
    seed : int
        RNG seed.
    """

    def __init__(self, n_iter: int = 4000, burn: int = 2000, thin: int = 2,
                 pi_a=(1.0, 9.0), pi_b=(1.0, 9.0), slab_scale: float = 2.0,
                 seed: int = 0, fixed: Optional[dict] = None,
                 force_include: bool = False, warm_start: bool = False,
                 mrf: Optional[dict] = None, subsetting: bool = True,
                 adaptive_slab: bool = False, slab_ig=(2.0, 2.0),
                 standardize: bool = True):
        if burn >= n_iter:
            raise ValueError("burn must be < n_iter")
        self.n_iter = int(n_iter)
        self.burn = int(burn)
        self.thin = max(1, int(thin))
        self.pi_a = (float(pi_a[0]), float(pi_a[1]))
        self.pi_b = (float(pi_b[0]), float(pi_b[1]))
        self.c2 = float(slab_scale) ** 2
        self.seed = int(seed)
        # --- Ablation switches (defaults give the production model) ---
        # subsetting=False decouples the indicators: delta_{k,o} ~ Bern(pi_b_o) is
        #   drawn independently of gamma_k (no delta=>gamma constraint, no subsetting
        #   -prior mass term), so a balance may "mediate" an outcome even if the
        #   exposure never regulates it. Isolates the subsetting prior.
        # adaptive_slab=True swaps the fixed Gaussian slab (v=slab_scale**2) for an
        #   inverse-gamma hierarchical slab refreshed each sweep from the included
        #   coefficients (an "adaptive" slab, in the spirit of the v1 horseshoe).
        #   Isolates the fixed-slab choice; slab_ig=(a0,b0) is its InvGamma prior.
        self.subsetting = bool(subsetting)
        # standardize=False (validation only) fits B and Y on their own scale, so the
        # prior applies to the generating scale and a simulation-based calibration can
        # draw data from exactly the fitted model.
        self.standardize = bool(standardize)
        self.adaptive_slab = bool(adaptive_slab)
        self.slab_a0 = float(slab_ig[0])
        self.slab_b0 = float(slab_ig[1])
        # Optional, off by default: an Ising / Markov-random-field prior over the
        # balance graph that couples the exposure-regulation indicators of correlated
        # balances. dict with keys
        #   'W'   : (K, K) symmetric nonneg adjacency (0 diagonal), and
        #   'lambda': coupling strength >= 0.
        # Prior  P(gamma) ~ exp( lambda * sum_{k,l} W_kl (2 gamma_k-1)(2 gamma_l-1) ),
        # so the gamma_k log-odds gains  2*lambda*sum_l W_kl (2 gamma_l - 1), pooling
        # inclusion evidence across correlated balances. None -> identical to the
        # independent-Bernoulli sampler.
        self.mrf = mrf
        # Validation-only fixed-scale mode: dict with keys sM2, so2, va, vb,
        # pi_a, pi_b -> a fixed-variance spike-and-slab whose posterior can be
        # enumerated exactly (see src/model/validation.py). When set, the
        # hyperparameter / sigma2 / inclusion-rate updates are skipped.
        self.fixed = fixed
        # Validation-only: force all gamma=delta=1 -> a pure two-stage
        # fixed-Gaussian-slab regression (no selection), used to cross-check the
        # continuous coefficient posterior.
        self.force_include = bool(force_include)
        # Optional empirical-Bayes screen (NOT a pure initialization): warm-start the
        # indicators from a marginal-correlation screen AND hold the inclusion rate
        # fixed at its prior mean (skip the Beta update). In the K ~ n regime the
        # all-zero indicator state is a sticky basin for any spike-and-slab Gibbs
        # sampler; the screen seeds plausible inclusions so the chain explores it.
        # The continuous-coefficient target is unchanged. OFF by default
        # (kept only for the high-dim mixing unit test).
        self.warm_start = bool(warm_start)
        self.samples: Optional[SpikeSlabSamples] = None

    # ------------------------------------------------------------------ #
    def fit(self, X, B, Y, C=None, batch=None) -> SpikeSlabSamples:
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float).reshape(-1)
        B = np.asarray(B, dtype=float)
        n, K = B.shape
        names = list(Y)

        # Standardize balances (col) and each outcome; NIE rescaled by y_scale
        # (the balance scale cancels in the alpha*beta product).
        if self.standardize:
            Bc = B - B.mean(0)
            B_sd = Bc.std(0); B_sd = np.where(B_sd < 1e-12, 1.0, B_sd)
            Bs = Bc / B_sd
        else:
            Bs = B.copy()
        Ys, y_scale = {}, {}
        for o in names:
            y = np.asarray(Y[o], dtype=float).reshape(-1)
            if self.standardize:
                s = y.std(); s = s if s > 1e-12 else 1.0
                Ys[o] = (y - y.mean()) / s
            else:
                s = 1.0
                Ys[o] = y.copy()
            y_scale[o] = s

        C = None if C is None else np.asarray(C, dtype=float)
        if C is not None and C.ndim == 1:
            C = C.reshape(n, 1)
        bd = _batch_dummies(batch, n)

        # Stage-1 nuisance design [1, C, batch] (X is the SELECTED predictor here);
        # Stage-2 nuisance design [1, X, C, batch] (cX = unpenalized direct effect).
        # Stage-1 nuisance design = [1, C, batch] (X is the selected predictor).
        cols1 = [np.ones((n, 1))]
        if C is not None and C.shape[1] > 0:
            cols1.append(C)
        if bd.shape[1] > 0:
            cols1.append(bd)
        Zu1 = np.concatenate(cols1, axis=1)
        Zu2 = _build_unpenalized(n, X, C, bd)            # [1, X, C, batch]
        Zu1tZu1 = Zu1.T @ Zu1
        Zu2tZu2 = Zu2.T @ Zu2
        xtx = float(X @ X)
        ridge_prec = 1.0 / (10.0 ** 2)                   # wide ridge on nuisance

        # ---- state ----
        xi1 = np.zeros((K, Zu1.shape[1]))
        alpha = np.zeros(K)
        gamma = np.ones(K, dtype=int)
        sM2 = np.ones(K)
        pi_alpha = 0.1

        beta = {o: np.zeros(K) for o in names}
        delta = {o: np.zeros(K, dtype=int) for o in names}
        xi2 = {o: np.zeros(Zu2.shape[1]) for o in names}
        so2 = {o: 1.0 for o in names}
        pi_beta = {o: 0.1 for o in names}

        # warm-start indicators from a marginal screen (init only; see __init__).
        if self.warm_start and self.fixed is None and not self.force_include:
            thr = 2.5 / np.sqrt(n)                        # ~ marginal significance
            def _abscorr(a, M):
                a = a - a.mean()
                den = (a @ a) ** 0.5 * np.sqrt((M ** 2).sum(0))
                den = np.where(den < 1e-12, 1.0, den)
                return np.abs((a @ M) / den)
            cx = _abscorr(X, Bs)
            gamma = (cx > thr).astype(int)
            for o in names:
                yo = Ys[o]
                cy = _abscorr(yo, Bs)
                dk = ((cy > thr) & (gamma == 1)).astype(int)
                delta[o] = dk
                beta[o] = np.where(dk == 1, np.sign(Bs.T @ (yo - yo.mean())) *
                                   np.minimum(cy, 0.9), 0.0)   # rough slab start

        # fixed-scale (validation) mode: pin variances / slab / inclusion rates
        fx = self.fixed
        va_fixed = vb_fixed = None
        if fx is not None:
            sM2[:] = float(fx["sM2"])
            pi_alpha = float(fx.get("pi_a", 0.1))
            for o in names:
                so2[o] = float(fx["so2"]); pi_beta[o] = float(fx.get("pi_b", 0.1))
            va_fixed = float(fx["va"]); vb_fixed = float(fx["vb"])

        # Fixed Gaussian-slab variances. We use a fixed-scale spike-and-slab
        # (alpha|gamma=1 ~ N(0, v_alpha), beta|delta=1 ~ N(0, v_beta)) rather than an
        # adaptive regularized horseshoe, because the horseshoe local-scale full
        # conditional is not conjugate under the bounded (Finnish) slab: a fixed slab
        # makes the production sampler target EXACTLY the model the exact enumeration
        # validates (src/model/validation.py). v = slab_scale**2 (weakly informative
        # in standardized units).
        v_alpha = va_fixed if va_fixed is not None else self.c2
        v_beta = vb_fixed if vb_fixed is not None else self.c2
        v_beta_map = {o: v_beta for o in names}   # per-outcome slab (diverges only if adaptive)
        _adapt = self.adaptive_slab and fx is None

        # Optional MRF/Ising coupling over the balance graph (experimental).
        mrf_W = None
        mrf_lam = 0.0
        if self.mrf is not None:
            mrf_W = np.asarray(self.mrf["W"], dtype=float).copy()
            if mrf_W.shape != (K, K):
                raise ValueError(f"mrf['W'] must be ({K}, {K}); got {mrf_W.shape}")
            np.fill_diagonal(mrf_W, 0.0)
            mrf_lam = float(self.mrf.get("lambda", 0.0))

        keep = np.arange(self.burn, self.n_iter, self.thin)
        S = keep.size
        a_draws = np.zeros((S, K)); g_draws = np.zeros((S, K))
        b_draws = {o: np.zeros((S, K)) for o in names}
        d_draws = {o: np.zeros((S, K)) for o in names}
        nie_draws = {o: np.zeros((S, K)) for o in names}
        sig_draws = {o: np.zeros(S) for o in names}
        s_idx = 0

        for it in range(self.n_iter):
            # ===================== STAGE 1 (per balance k) =====================
            for k in range(K):
                bk = Bs[:, k]
                # nuisance xi1_k given current alpha_k
                r_nuis = bk - alpha[k] * X
                xi1[k] = _ridge_block_draw(rng, Zu1, Zu1tZu1, r_nuis, sM2[k], ridge_prec)
                r1 = bk - Zu1 @ xi1[k]                    # partial residual for alpha_k
                s2_slab = v_alpha                         # fixed Gaussian slab
                V = 1.0 / (xtx / sM2[k] + 1.0 / s2_slab)
                m = V * (float(X @ r1) / sM2[k])
                # gamma_k is sampled only when all delta_{k,o}=0 (else forced=1
                # below). Under the subsetting prior delta_{k,o} ~ Bern(pi_b_o * gamma_k),
                # turning gamma_k on while every delta stays 0 must pay the prior mass
                # prod_o (1 - pi_b_o) -- the "regulated but mediates nothing" penalty.
                logodds = (np.log(pi_alpha / (1.0 - pi_alpha))
                           + 0.5 * np.log(V / s2_slab) + 0.5 * m * m / V)
                if self.subsetting:
                    # subsetting-prior mass: an exposure-regulated balance mediating
                    # NO outcome must pay prod_o (1 - pi_b_o).
                    logodds += sum(np.log1p(-pi_beta[o]) for o in names)
                if mrf_W is not None:            # Ising neighbour coupling (experimental)
                    logodds += 2.0 * mrf_lam * float(mrf_W[k] @ (2.0 * gamma - 1.0))
                # subsetting: if any outcome currently selects k, gamma_k must be 1
                forced = self.force_include or (self.subsetting
                                                and any(delta[o][k] == 1 for o in names))
                p_g = 1.0 if forced else _sigmoid(logodds)
                gamma[k] = 1 if (forced or rng.random() < p_g) else 0
                alpha[k] = (m + np.sqrt(V) * rng.standard_normal()) if gamma[k] == 1 else 0.0
                if fx is None:
                    # mediator residual variance (conjugate inverse-gamma)
                    rss = float(np.sum((bk - Zu1 @ xi1[k] - alpha[k] * X) ** 2))
                    sM2[k] = _inv_gamma(rng, 0.5 + n / 2.0, 0.5 + rss / 2.0)

            n_g = int(gamma.sum())
            if fx is None and not self.warm_start:
                pi_alpha = rng.beta(self.pi_a[0] + n_g, self.pi_a[1] + (K - n_g))
                pi_alpha = min(max(pi_alpha, 1e-4), 1 - 1e-4)
            # warm_start holds pi fixed at its prior mean (empirical-Bayes screen).
            if _adapt:   # adaptive slab: refresh v_alpha from the included alphas
                inc = alpha[gamma == 1]
                v_alpha = _inv_gamma(rng, self.slab_a0 + 0.5 * inc.size,
                                     self.slab_b0 + 0.5 * float(inc @ inc))
                v_alpha = float(min(max(v_alpha, 1e-3), 1e4))

            # ===================== STAGE 2 (per outcome) =======================
            for o in names:
                y = Ys[o]
                # nuisance block xi2_o given current beta_o
                r_nuis = y - Bs @ beta[o]
                xi2[o] = _ridge_block_draw(rng, Zu2, Zu2tZu2, r_nuis, so2[o], ridge_prec)
                base = y - Zu2 @ xi2[o]                    # remove unpenalized part
                fitted = Bs @ beta[o]
                s2_slab = v_beta_map[o]                        # (adaptive slab -> per outcome)
                for k in range(K):
                    rk = base - (fitted - beta[o][k] * Bs[:, k])   # partial resid for beta_k
                    bk_col = Bs[:, k]
                    bkk = float(bk_col @ bk_col)
                    V = 1.0 / (bkk / so2[o] + 1.0 / s2_slab)
                    m = V * (float(bk_col @ rk) / so2[o])
                    logodds = (np.log(pi_beta[o] / (1.0 - pi_beta[o]))
                               + 0.5 * np.log(V / s2_slab) + 0.5 * m * m / V)
                    if mrf_W is not None:        # Ising coupling on the delta (outcome) path
                        logodds += 2.0 * mrf_lam * float(mrf_W[k] @ (2.0 * delta[o] - 1.0))
                    # subsetting: delta_{k,o} can fire only if gamma_k=1; without it
                    # the two paths are independent (delta ablation).
                    if (not self.subsetting) or gamma[k] == 1:
                        p_d = _sigmoid(logodds)
                        delta[o][k] = 1 if (self.force_include or rng.random() < p_d) else 0
                        new_bk = (m + np.sqrt(V) * rng.standard_normal()) if delta[o][k] == 1 else 0.0
                    else:
                        delta[o][k] = 0
                        new_bk = 0.0
                    fitted = fitted - beta[o][k] * bk_col + new_bk * bk_col
                    beta[o][k] = new_bk
                if _adapt:   # adaptive slab: refresh v_beta_o from the included betas
                    incb = beta[o][delta[o] == 1]
                    v_beta_map[o] = float(min(max(_inv_gamma(
                        rng, self.slab_a0 + 0.5 * incb.size,
                        self.slab_b0 + 0.5 * float(incb @ incb)), 1e-3), 1e4))
                if fx is None:
                    rss2 = float(np.sum((y - Zu2 @ xi2[o] - Bs @ beta[o]) ** 2))
                    so2[o] = _inv_gamma(rng, 0.5 + n / 2.0, 0.5 + rss2 / 2.0)
                    n_d = int(delta[o].sum())
                    if not self.warm_start:
                        a_post, b_post = pi_beta_posterior(self.pi_b, n_d, n_g, K, self.subsetting)
                        pi_beta[o] = rng.beta(a_post, b_post)
                        pi_beta[o] = min(max(pi_beta[o], 1e-4), 1 - 1e-4)

            # ===================== record =====================
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                a_draws[s_idx] = alpha
                g_draws[s_idx] = gamma
                for o in names:
                    b_draws[o][s_idx] = beta[o]
                    d_draws[o][s_idx] = delta[o]
                    nie_draws[o][s_idx] = alpha * beta[o] * y_scale[o]
                    sig_draws[o][s_idx] = so2[o] ** 0.5 * y_scale[o]
                s_idx += 1

        self.samples = SpikeSlabSamples(
            mediator={"alpha": a_draws[:s_idx], "gamma": g_draws[:s_idx]},
            outcomes={o: {"beta": b_draws[o][:s_idx], "delta": d_draws[o][:s_idx],
                          "nie": nie_draws[o][:s_idx], "sigma": sig_draws[o][:s_idx]}
                      for o in names},
            meta={"names": names, "n": n, "K": K, "y_scale": y_scale,
                  "n_iter": self.n_iter, "burn": self.burn, "thin": self.thin,
                  "n_samples": int(s_idx), "slab_c2": self.c2, "seed": self.seed},
        )
        return self.samples
