"""Exact-enumeration validation of the spike-and-slab Gibbs sampler.

For a *fixed-scale* spike-and-slab (known residual variances and slab variances),
the posterior over the discrete selection states is a finite sum: with the
subsetting prior each balance is in one of three states
    0: (gamma=0, delta=0)   1: (gamma=1, delta=0)   2: (gamma=1, delta=1)
so there are 3**K configurations (one outcome). Each configuration's marginal
likelihood is a closed-form zero-mean multivariate-normal density (the regression
coefficients and the wide-ridge nuisance are integrated out analytically), so the
posterior inclusion probabilities can be computed EXACTLY.

Running ``mediation_v2.SpikeSlabMediation(fixed=...)`` on the same data and
checking its Monte-Carlo PIPs against this enumeration is a gold-standard
correctness check of the discrete-selection machinery (the novel, error-prone
part) -- stronger and more self-contained than a Stan port, which cannot sample
the discrete indicators directly. (A Stan/NUTS continuous cross-check on the
coefficient posteriors is provided separately when cmdstanpy is available.)
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np

__all__ = ["enumerate_posterior", "enumerate_posterior_2outcome"]


def _mvn_logpdf_zero_mean(y: np.ndarray, cov: np.ndarray) -> float:
    """log N(y; 0, cov) via Cholesky."""
    n = y.size
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    alpha = np.linalg.solve(L, y)
    quad = float(alpha @ alpha)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)


def enumerate_posterior(X: np.ndarray, B: np.ndarray, y: np.ndarray,
                        va: float, vb: float, sM2: float, so2: float,
                        pi_a: float, pi_b: float, ridge_var: float = 100.0
                        ) -> Dict[str, np.ndarray]:
    """Exact posterior of the fixed-scale spike-and-slab (one outcome).

    Parameters
    ----------
    X : (n,) exposure. B : (n, K) balances (already standardized). y : (n,) outcome.
    va, vb : slab variances for alpha (X->B) and beta (B->Y).
    sM2, so2 : residual variances of the mediator and outcome stages.
    pi_a, pi_b : prior inclusion probabilities for gamma and delta.
    ridge_var : variance of the wide-ridge nuisance prior (intercept; [1,X] in stage 2).

    Returns
    -------
    dict with ``pip`` (K,) = Pr(gamma_k=1 AND delta_k=1 | data), ``pip_gamma`` (K,),
    and ``logZ`` (the log marginal likelihood).
    """
    X = np.asarray(X, float).reshape(-1)
    B = np.asarray(B, float)
    y = np.asarray(y, float).reshape(-1)
    n, K = B.shape
    ones = np.ones(n)

    # Stage-1 marginal log-likelihood of B_k for gamma_k in {0,1} (independent k).
    Snuis1 = ridge_var * np.outer(ones, ones)            # intercept nuisance
    XXt = np.outer(X, X)
    logML1 = np.zeros((K, 2))
    for k in range(K):
        base = sM2 * np.eye(n) + Snuis1
        logML1[k, 0] = _mvn_logpdf_zero_mean(B[:, k], base)
        logML1[k, 1] = _mvn_logpdf_zero_mean(B[:, k], base + va * XXt)

    # Stage-2 base covariance (intercept + X nuisance); add vb*B_k B_k^T per delta=1.
    Zu2 = np.column_stack([ones, X])
    Snuis2 = ridge_var * (Zu2 @ Zu2.T)
    base2 = so2 * np.eye(n) + Snuis2

    la, l1a = np.log(pi_a), np.log1p(-pi_a)
    lb, l1b = np.log(pi_b), np.log1p(-pi_b)

    log_post: List[float] = []
    configs: List[tuple] = []
    for state in itertools.product((0, 1, 2), repeat=K):   # per-balance state
        gamma = np.array([1 if s >= 1 else 0 for s in state])
        delta = np.array([1 if s == 2 else 0 for s in state])
        lp = 0.0
        for k in range(K):
            lp += la if gamma[k] else l1a
            if gamma[k]:
                lp += lb if delta[k] else l1b
            lp += logML1[k, gamma[k]]
        cov2 = base2.copy()
        for k in np.nonzero(delta)[0]:
            cov2 = cov2 + vb * np.outer(B[:, k], B[:, k])
        lp += _mvn_logpdf_zero_mean(y, cov2)
        log_post.append(lp)
        configs.append(state)

    log_post = np.asarray(log_post)
    M = log_post.max()
    w = np.exp(log_post - M)
    Z = w.sum()
    post = w / Z
    pip = np.zeros(K); pip_g = np.zeros(K)
    for cfg, p in zip(configs, post):
        for k, s in enumerate(cfg):
            if s >= 1:
                pip_g[k] += p
            if s == 2:
                pip[k] += p
    return {"pip": pip, "pip_gamma": pip_g, "logZ": float(M + np.log(Z))}


def enumerate_posterior_2outcome(X, B, yF, yD, va, vb, sM2, soF2, soD2,
                                 pi_a, pi_bF, pi_bD, ridge_var=100.0, subsetting=True):
    """Exact two-outcome fixed-scale posterior by enumerating every configuration.

    With the subsetting prior (default) the admissible per-balance states are
        0: gamma=0
        1: gamma=1, deltaF=0, deltaD=0
        2: gamma=1, deltaF=1, deltaD=0
        3: gamma=1, deltaF=0, deltaD=1
        4: gamma=1, deltaF=1, deltaD=1
    (5**K configurations). With ``subsetting=False`` (the decoupled variant) all
    8 combinations of (gamma, deltaF, deltaD) are admissible and delta's prior no
    longer depends on gamma (8**K configurations). Each configuration's marginal
    likelihood is the product of the stage-1 term (per balance) and two stage-2
    Gaussian MVN terms (outcomes F and D), all closed form. Returns exact PIP_F,
    PIP_D (= Pr(gamma & delta_o)) and PIP_gamma.
    """
    X = np.asarray(X, float).reshape(-1); B = np.asarray(B, float)
    yF = np.asarray(yF, float).reshape(-1); yD = np.asarray(yD, float).reshape(-1)
    n, K = B.shape; ones = np.ones(n)
    Snuis1 = ridge_var * np.outer(ones, ones); XXt = np.outer(X, X)
    logML1 = np.zeros((K, 2))
    for k in range(K):
        base = sM2 * np.eye(n) + Snuis1
        logML1[k, 0] = _mvn_logpdf_zero_mean(B[:, k], base)
        logML1[k, 1] = _mvn_logpdf_zero_mean(B[:, k], base + va * XXt)
    Zu2 = np.column_stack([ones, X]); Snuis2 = ridge_var * (Zu2 @ Zu2.T)
    baseF = soF2 * np.eye(n) + Snuis2; baseD = soD2 * np.eye(n) + Snuis2
    la, l1a = np.log(pi_a), np.log1p(-pi_a)
    lbF, l1bF = np.log(pi_bF), np.log1p(-pi_bF)
    lbD, l1bD = np.log(pi_bD), np.log1p(-pi_bD)
    # state -> (gamma, dF, dD)
    if subsetting:
        ST = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 0, 1), 4: (1, 1, 1)}
    else:
        ST = dict(enumerate(itertools.product((0, 1), repeat=3)))
    log_post, configs = [], []
    for state in itertools.product(range(len(ST)), repeat=K):
        g = np.array([ST[s][0] for s in state]); dF = np.array([ST[s][1] for s in state])
        dD = np.array([ST[s][2] for s in state])
        lp = 0.0
        for k in range(K):
            lp += la if g[k] else l1a
            if g[k] or not subsetting:
                lp += (lbF if dF[k] else l1bF) + (lbD if dD[k] else l1bD)
            lp += logML1[k, g[k]]
        covF = baseF.copy()
        for k in np.nonzero(dF)[0]:
            covF = covF + vb * np.outer(B[:, k], B[:, k])
        covD = baseD.copy()
        for k in np.nonzero(dD)[0]:
            covD = covD + vb * np.outer(B[:, k], B[:, k])
        lp += _mvn_logpdf_zero_mean(yF, covF) + _mvn_logpdf_zero_mean(yD, covD)
        log_post.append(lp); configs.append(state)
    log_post = np.asarray(log_post); M = log_post.max()
    post = np.exp(log_post - M); post /= post.sum()
    pipF = np.zeros(K); pipD = np.zeros(K); pg = np.zeros(K)
    for cfg, p in zip(configs, post):
        for k, s in enumerate(cfg):
            gg, ff, dd = ST[s]
            if gg: pg[k] += p
            if gg and ff: pipF[k] += p       # mediation = both paths active
            if gg and dd: pipD[k] += p
    return {"pip_F": pipF, "pip_D": pipD, "pip_gamma": pg}
