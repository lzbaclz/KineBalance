"""Multi-outcome residual-covariance mediation (SUR): an optional joint-likelihood variant.

The conference model regresses each outcome on the balances with INDEPENDENT
Gaussian residuals. Mechanistically distinct metabolites are, however, correlated
(deoxycholate and propionate co-vary), and ignoring that throws away cross-outcome
information. This module upgrades the two-stage joint spike-and-slab to a
Seemingly-Unrelated-Regression (SUR) likelihood with a full residual covariance:

    Y_i = mu_i + e_i,   e_i ~ N_R(0, Sigma_Y),   Sigma_Y ~ InvWishart(nu0, Psi0),

keeping stage 1 (gamma_k, alpha_k) and the spike-and-slab on (delta_{k,r}, beta_{k,r})
unchanged, under either prior coupling (``subsetting=True``: delta => gamma, the
coupled prior; ``False``: the decoupled prior of the conference sampler). The only change is that each outcome's stage-2
update conditions on the OTHER outcomes' residuals: by the Gaussian conditional,

    y_{i,r} | y_{i,-r} ~ N( mu_{i,r} + Sigma_{r,-r} Sigma_{-r,-r}^{-1} e_{i,-r},
                            Sigma_{rr} - Sigma_{r,-r} Sigma_{-r,-r}^{-1} Sigma_{-r,r} ),

so the existing spike-and-slab beta/delta update runs on the adjusted response
``ytil_r = y_r - shift`` with the conditional variance ``s_cond`` -- borrowing
strength across correlated outcomes. Sigma_Y is drawn from its inverse-Wishart full
conditional. With ``sigma_diag=True`` Sigma is held diagonal with the conference
model's InvGamma(1/2, 1/2) residual prior on each entry, so the sampler targets exactly
the independent-residual model (the unit tests compare the two samplers this way).

It reuses the v2 Gibbs helpers and produces a ``SpikeSlabSamples`` compatible with
``specificity_v2`` (so the same posterior-class decision rule applies), plus the
posterior residual correlation in ``meta``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import invwishart

from src.model.mediation import _inv_gamma, _build_unpenalized, _batch_dummies
from src.model.mediation_v2 import (SpikeSlabSamples, _ridge_block_draw, _sigmoid,
                                     _JITTER, pi_beta_posterior)

__all__ = ["SpikeSlabMediationCov"]


class SpikeSlabMediationCov:
    """Two-stage joint spike-and-slab with a full residual covariance across outcomes.

    Parameters mirror :class:`SpikeSlabMediation` (production path only: no fixed /
    warm-start / force-include / MRF modes). ``iw_nu0`` / ``iw_psi0_scale`` set the
    inverse-Wishart prior on the (standardized-outcome) residual covariance.
    ``subsetting`` selects the coupled (True) or decoupled (False) prior, as in
    :class:`SpikeSlabMediation`. ``sigma_diag=True`` (validation) replaces the
    inverse-Wishart step by independent InvGamma(1/2, 1/2) draws of the diagonal.
    """

    def __init__(self, n_iter: int = 4000, burn: int = 2000, thin: int = 2,
                 pi_a=(1.0, 9.0), pi_b=(1.0, 9.0), slab_scale: float = 2.0,
                 iw_nu0: Optional[float] = None, iw_psi0_scale: float = 1.0,
                 seed: int = 0, subsetting: bool = True, sigma_diag: bool = False):
        if burn >= n_iter:
            raise ValueError("burn must be < n_iter")
        self.n_iter, self.burn, self.thin = int(n_iter), int(burn), max(1, int(thin))
        self.pi_a = (float(pi_a[0]), float(pi_a[1]))
        self.pi_b = (float(pi_b[0]), float(pi_b[1]))
        self.c2 = float(slab_scale) ** 2
        self.iw_nu0 = iw_nu0
        self.iw_psi0_scale = float(iw_psi0_scale)
        self.seed = int(seed)
        self.subsetting = bool(subsetting)
        self.sigma_diag = bool(sigma_diag)
        self.samples: Optional[SpikeSlabSamples] = None

    def fit(self, X, B, Y, C=None, batch=None) -> SpikeSlabSamples:
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float).reshape(-1)
        B = np.asarray(B, float)
        n, K = B.shape
        names: List = list(Y)
        R = len(names)

        Bc = B - B.mean(0)
        B_sd = Bc.std(0); B_sd = np.where(B_sd < 1e-12, 1.0, B_sd)
        Bs = Bc / B_sd
        Ys = np.zeros((n, R)); y_scale = np.zeros(R)
        for r, o in enumerate(names):
            y = np.asarray(Y[o], float).reshape(-1)
            s = y.std(); s = s if s > 1e-12 else 1.0
            Ys[:, r] = (y - y.mean()) / s; y_scale[r] = s

        C = None if C is None else np.asarray(C, float)
        if C is not None and C.ndim == 1:
            C = C.reshape(n, 1)
        bd = _batch_dummies(batch, n)
        cols1 = [np.ones((n, 1))]
        if C is not None and C.shape[1] > 0:
            cols1.append(C)
        if bd.shape[1] > 0:
            cols1.append(bd)
        Zu1 = np.concatenate(cols1, axis=1)
        Zu2 = _build_unpenalized(n, X, C, bd)
        Zu1tZu1 = Zu1.T @ Zu1; Zu2tZu2 = Zu2.T @ Zu2
        xtx = float(X @ X); ridge_prec = 1.0 / (10.0 ** 2)
        v_alpha = v_beta = self.c2

        # ---- state ----
        xi1 = np.zeros((K, Zu1.shape[1])); alpha = np.zeros(K)
        gamma = np.ones(K, dtype=int); sM2 = np.ones(K); pi_alpha = 0.1
        beta = np.zeros((R, K)); delta = np.zeros((R, K), dtype=int)
        xi2 = np.zeros((R, Zu2.shape[1])); pi_beta = np.full(R, 0.1)
        Sigma = np.eye(R)                                    # residual covariance
        nu0 = self.iw_nu0 if self.iw_nu0 is not None else (R + 2.0)
        Psi0 = self.iw_psi0_scale * np.eye(R)

        keep = np.arange(self.burn, self.n_iter, self.thin); S = keep.size
        a_draws = np.zeros((S, K)); g_draws = np.zeros((S, K))
        b_draws = {o: np.zeros((S, K)) for o in names}
        d_draws = {o: np.zeros((S, K)) for o in names}
        nie_draws = {o: np.zeros((S, K)) for o in names}
        sig_draws = {o: np.zeros(S) for o in names}
        corr_draws = np.zeros(S); s_idx = 0

        def _resid():
            E = np.empty((n, R))
            for r in range(R):
                E[:, r] = Ys[:, r] - Zu2 @ xi2[r] - Bs @ beta[r]
            return E

        for it in range(self.n_iter):
            # ===================== STAGE 1 (per balance k) =====================
            # exposure-regulation path is outcome-agnostic; identical to the v2 sampler.
            for k in range(K):
                bk = Bs[:, k]
                r_nuis = bk - alpha[k] * X
                xi1[k] = _ridge_block_draw(rng, Zu1, Zu1tZu1, r_nuis, sM2[k], ridge_prec)
                r1 = bk - Zu1 @ xi1[k]
                V = 1.0 / (xtx / sM2[k] + 1.0 / v_alpha)
                m = V * (float(X @ r1) / sM2[k])
                logodds = (np.log(pi_alpha / (1.0 - pi_alpha))
                           + 0.5 * np.log(V / v_alpha) + 0.5 * m * m / V)
                if self.subsetting:
                    # coupled prior: a regulated balance mediating no outcome pays
                    # prod_r (1 - pi_b_r), and any selected delta forces gamma = 1
                    logodds += float(np.sum(np.log1p(-pi_beta)))
                forced = self.subsetting and any(delta[r][k] == 1 for r in range(R))
                gamma[k] = 1 if (forced or rng.random() < _sigmoid(logodds)) else 0
                alpha[k] = (m + np.sqrt(V) * rng.standard_normal()) if gamma[k] == 1 else 0.0
                rss = float(np.sum((bk - Zu1 @ xi1[k] - alpha[k] * X) ** 2))
                sM2[k] = _inv_gamma(rng, 0.5 + n / 2.0, 0.5 + rss / 2.0)
            n_g = int(gamma.sum())
            pi_alpha = float(np.clip(rng.beta(self.pi_a[0] + n_g, self.pi_a[1] + (K - n_g)),
                                     1e-4, 1 - 1e-4))

            # ===================== STAGE 2 (per outcome, SUR-conditioned) ============
            E = _resid()
            for r in range(R):
                others = [o for o in range(R) if o != r]
                if others:
                    Soo = Sigma[np.ix_(others, others)]
                    Sro = Sigma[r, others]
                    Soo_inv = np.linalg.inv(Soo + _JITTER * np.eye(len(others)))
                    b_cond = Sro @ Soo_inv                         # (R-1,)
                    shift = E[:, others] @ b_cond                  # (n,)
                    s_cond = float(Sigma[r, r] - Sro @ Soo_inv @ Sro)
                else:
                    shift = np.zeros(n); s_cond = float(Sigma[r, r])
                s_cond = max(s_cond, 1e-8)
                ytil = Ys[:, r] - shift                            # conditional response

                r_nuis = ytil - Bs @ beta[r]
                xi2[r] = _ridge_block_draw(rng, Zu2, Zu2tZu2, r_nuis, s_cond, ridge_prec)
                base = ytil - Zu2 @ xi2[r]
                fitted = Bs @ beta[r]
                for k in range(K):
                    rk = base - (fitted - beta[r][k] * Bs[:, k])
                    bk_col = Bs[:, k]; bkk = float(bk_col @ bk_col)
                    V = 1.0 / (bkk / s_cond + 1.0 / v_beta)
                    m = V * (float(bk_col @ rk) / s_cond)
                    logodds = (np.log(pi_beta[r] / (1.0 - pi_beta[r]))
                               + 0.5 * np.log(V / v_beta) + 0.5 * m * m / V)
                    if (not self.subsetting) or gamma[k] == 1:
                        delta[r][k] = 1 if rng.random() < _sigmoid(logodds) else 0
                        new_bk = (m + np.sqrt(V) * rng.standard_normal()) if delta[r][k] == 1 else 0.0
                    else:
                        delta[r][k] = 0; new_bk = 0.0
                    fitted = fitted - beta[r][k] * bk_col + new_bk * bk_col
                    beta[r][k] = new_bk
                E[:, r] = Ys[:, r] - Zu2 @ xi2[r] - Bs @ beta[r]   # refresh this outcome's resid
                n_d = int(delta[r].sum())
                a_post, b_post = pi_beta_posterior(self.pi_b, n_d, n_g, K, self.subsetting)
                pi_beta[r] = float(np.clip(rng.beta(a_post, b_post), 1e-4, 1 - 1e-4))

            # ===================== residual covariance (inverse-Wishart) ===========
            if self.sigma_diag:     # independent residuals: InvGamma(1/2, 1/2) per outcome
                Sigma = np.diag([_inv_gamma(rng, 0.5 + n / 2.0, 0.5 + float(E[:, r] @ E[:, r]) / 2.0)
                                 for r in range(R)])
            else:
                Sigma = invwishart.rvs(df=nu0 + n, scale=Psi0 + E.T @ E, random_state=rng)
                Sigma = np.atleast_2d(Sigma)

            # ===================== record =====================
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                a_draws[s_idx] = alpha; g_draws[s_idx] = gamma
                d = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
                for r, o in enumerate(names):
                    b_draws[o][s_idx] = beta[r]; d_draws[o][s_idx] = delta[r]
                    nie_draws[o][s_idx] = alpha * beta[r] * y_scale[r]
                    sig_draws[o][s_idx] = np.sqrt(Sigma[r, r]) * y_scale[r]
                corr_draws[s_idx] = (Sigma[0, 1] / (d[0] * d[1])) if R > 1 else 0.0
                s_idx += 1

        self.samples = SpikeSlabSamples(
            mediator={"alpha": a_draws[:s_idx], "gamma": g_draws[:s_idx]},
            outcomes={o: {"beta": b_draws[o][:s_idx], "delta": d_draws[o][:s_idx],
                          "nie": nie_draws[o][:s_idx], "sigma": sig_draws[o][:s_idx]}
                      for o in names},
            meta={"names": names, "n": n, "K": K, "R": R, "y_scale": dict(zip(names, y_scale)),
                  "n_iter": self.n_iter, "burn": self.burn, "thin": self.thin,
                  "n_samples": int(s_idx), "slab_c2": self.c2, "seed": self.seed,
                  "subsetting": self.subsetting, "sigma_diag": self.sigma_diag,
                  "resid_corr_mean": float(corr_draws[:s_idx].mean()) if s_idx else 0.0,
                  "resid_corr_draws": corr_draws[:s_idx]},
        )
        return self.samples
