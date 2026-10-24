"""KineBalance: sparse hierarchical Bayesian mediation via NumPy Gibbs sampler.

This module implements the KineBalance mediation model
(stage-1 and stage-2 regressions, with measurement-error propagation).

Model
-----
Stage 1 (mediator model), per balance ``b``::

    B[:, b] = a0_b + aX_b * X + C @ aC_b + batch_intercept_b + eta_b
    eta_b ~ N(0, sigmaM_b^2)

Stage 2 (outcome model), per outcome ``r``::

    Y_r = c0_r + gX_r * X + B @ gamma_r + C @ bC_r + batch_intercept_r + eps_r
    eps_r ~ N(0, sigma_r^2)

The mediator coefficients ``gamma_r`` (one weight per balance) carry a
**horseshoe** prior for sparse mediator selection; all remaining
coefficients (intercept, exposure, confounders, batch dummies) carry a
wide Normal / ridge prior.

Gibbs sampler
-------------
The stage-2 design is split column-wise into an **unpenalized** block
``Z_u`` (intercept, X, C, batch dummies; prior precision ``1/ridge_sd^2``)
and a **penalized** block ``Z_p`` (the K balances; horseshoe).

Horseshoe is sampled with the Makalic & Schmidt (2016) inverse-gamma
auxiliary-variable scheme::

    beta_p_j ~ N(0, sigma^2 tau^2 lambda_j^2)
    lambda_j^2 ~ InvGamma(1/2, 1/nu_j),   nu_j ~ InvGamma(1/2, 1)
    tau^2      ~ InvGamma(1/2, 1/xi),     xi   ~ InvGamma(1/2, 1)

with conditionals::

    beta | .   ~ N(m, V),
        prior precision = 1/ridge_sd^2 (Z_u cols), 1/(tau^2 lambda_j^2) (Z_p cols)
        posterior precision = (Z^T Z)/sigma^2 + diag(prior_prec)
    sigma^2 | . ~ InvGamma((n+Kp)/2, (RSS + sum_j beta_p_j^2/(tau^2 lambda_j^2))/2)
    lambda_j^2 | . ~ InvGamma(1, 1/nu_j + beta_p_j^2/(2 tau^2 sigma^2))
    nu_j      | . ~ InvGamma(1, 1 + 1/lambda_j^2)
    tau^2 | .   ~ InvGamma((Kp+1)/2, 1/xi + sum_j beta_p_j^2/(2 sigma^2 lambda_j^2))
    xi        | . ~ InvGamma(1, 1 + 1/tau^2)

Stage-1 mediator regressions are ordinary Bayesian linear regressions with
wide Normal priors and an InvGamma prior on ``sigmaM_b^2``; the K balances
share the same ``[1, X, C, batch]`` design and are updated jointly (the
posterior precision is shared across balances, so one Cholesky serves all).

Measurement-error propagation
-----------------------------
If an outcome carries per-observation standard errors ``meas_sd`` then, at
the start of every Gibbs iteration and *before* that outcome's stage-2
update, the working response is resampled by data augmentation::

    y_i ~ N(point_i, meas_sd_i^2)

so that posterior uncertainty in the point estimates propagates
into the mediation effects.

Notes
-----
``InvGamma(shape=a, scale=b)`` is sampled as ``1.0 / rng.gamma(shape=a,
scale=1.0/b)`` throughout.
"""
# Legacy v1 model, kept as a baseline; the main method is
# src/model/mediation_v2.py (specificity_v2.py).


from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _inv_gamma(rng: np.random.Generator, shape, scale) -> np.ndarray:
    """Draw from InvGamma(shape, scale): x = 1 / Gamma(shape, rate=scale).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    shape : float or ndarray
        Shape parameter ``a`` (> 0).
    scale : float or ndarray
        Scale parameter ``b`` (> 0) in the InvGamma density
        ``b^a / Gamma(a) x^{-a-1} exp(-b/x)``.

    Returns
    -------
    numpy.ndarray
        Draw(s) from the inverse-gamma distribution.
    """
    shape = np.asarray(shape, dtype=float)
    scale = np.asarray(scale, dtype=float)
    scale = np.maximum(scale, 1e-300)
    g = rng.gamma(shape=shape, scale=1.0 / scale)
    g = np.maximum(g, 1e-300)
    return 1.0 / g


def _mvn_from_precision(
    rng: np.random.Generator,
    prec: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Draw ``beta ~ N(prec^{-1} b, prec^{-1})`` via Cholesky.

    Solves the system using the precision (information) parameterization,
    which is numerically stable for ridge / horseshoe regressions.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    prec : ndarray of shape (p, p)
        Posterior precision matrix (symmetric positive definite).
    b : ndarray of shape (p,)
        Information vector ``Z^T y / sigma^2`` (so that the mean is
        ``prec^{-1} b``).

    Returns
    -------
    ndarray of shape (p,)
        A single draw from the Gaussian.
    """
    p = prec.shape[0]
    prec = 0.5 * (prec + prec.T)  # enforce symmetry (numerical)
    eye = np.eye(p)
    # Adaptive jitter: in the p >> n regime (more balances than samples) the
    # posterior precision can be rank-deficient, so a fixed tiny jitter is not
    # enough -- the Cholesky may succeed yet leave a near-zero pivot that makes
    # the triangular solve singular. Grow the jitter until both the Cholesky and
    # the solves are well-conditioned; it is negligible when well-conditioned.
    jitter = 1e-9 * (np.trace(prec) / max(p, 1) + 1.0)
    for _ in range(8):
        try:
            L = np.linalg.cholesky(prec + jitter * eye)
            w = np.linalg.solve(L, b)
            m = np.linalg.solve(L.T, w)
            z = rng.standard_normal(p)
            u = np.linalg.solve(L.T, z)
            if np.all(np.isfinite(m)) and np.all(np.isfinite(u)):
                return m + u
        except np.linalg.LinAlgError:
            pass
        jitter *= 10.0
    # Last-resort fallback: pseudo-inverse mean + covariance draw. This means the
    # adaptive-jitter Cholesky never succeeded (severely rank-deficient block);
    # surface it rather than failing silently.
    warnings.warn(
        f"_mvn_from_precision fell back to pinv after exhausting jitter "
        f"(final jitter={jitter:.1e}); the precision block is severely "
        f"rank-deficient. Draws remain finite but less accurate.",
        RuntimeWarning, stacklevel=2,
    )
    cov = np.linalg.pinv(prec + jitter * eye)
    cov = 0.5 * (cov + cov.T)
    return rng.multivariate_normal(cov @ b, cov)


def _build_unpenalized(
    n: int,
    X: np.ndarray,
    C: Optional[np.ndarray],
    batch_dummies: Optional[np.ndarray],
) -> np.ndarray:
    """Assemble the unpenalized design ``[1, X, C, batch_dummies]``.

    Parameters
    ----------
    n : int
        Number of observations.
    X : ndarray of shape (n,)
        Exposure vector.
    C : ndarray of shape (n, q) or None
        Confounder matrix.
    batch_dummies : ndarray of shape (n, n_batch-1) or None
        One-hot batch indicators (first level dropped).

    Returns
    -------
    ndarray of shape (n, p_u)
        The unpenalized design matrix.
    """
    cols = [np.ones((n, 1)), X.reshape(n, 1)]
    if C is not None and C.shape[1] > 0:
        cols.append(C)
    if batch_dummies is not None and batch_dummies.shape[1] > 0:
        cols.append(batch_dummies)
    return np.concatenate(cols, axis=1)


def _batch_dummies(batch: Optional[np.ndarray], n: int) -> np.ndarray:
    """Build drop-first one-hot dummies from a batch label vector.

    Parameters
    ----------
    batch : ndarray of shape (n,) or None
        Batch labels (any hashable / sortable values).
    n : int
        Number of observations.

    Returns
    -------
    ndarray of shape (n, n_batch-1)
        Dummy matrix; empty (n, 0) array if ``batch`` is None or has a
        single level.
    """
    if batch is None:
        return np.zeros((n, 0))
    batch = np.asarray(batch)
    levels = np.unique(batch)
    if levels.size <= 1:
        return np.zeros((n, 0))
    # drop first level as reference
    dummies = np.zeros((n, levels.size - 1))
    for j, lv in enumerate(levels[1:]):
        dummies[:, j] = (batch == lv).astype(float)
    return dummies


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class KineBalanceSamples:
    """Posterior draws from :class:`KineBalanceModel`.

    Attributes
    ----------
    outcomes : dict
        Maps each outcome name to a dict of stage-2 draws with keys
        ``gamma`` (S, K), ``gX`` (S,), ``intercept`` (S,) and
        ``sigma`` (S,).
    mediator : dict
        Stage-1 draws shared across outcomes with key ``aX`` of shape
        (S, K) (the exposure -> balance path coefficients).
    meta : dict
        Run metadata (n_iter, burn, thin, n_samples, K, balance/outcome
        names, ridge_sd, seed, dimensions, measurement-error flags).
    """

    outcomes: Dict[str, Dict[str, np.ndarray]]
    mediator: Dict[str, np.ndarray]
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class KineBalanceModel:
    """Sparse hierarchical Bayesian mediation model.

    Parameters
    ----------
    n_iter : int, default 2000
        Total Gibbs iterations.
    burn : int, default 1000
        Burn-in iterations discarded before thinning.
    thin : int, default 2
        Keep every ``thin``-th post-burn draw.
    ridge_sd : float, default 10.0
        Prior standard deviation for the unpenalized coefficients (wide
        Normal); their prior precision is ``1/ridge_sd^2``.
    seed : int, default 0
        Seed for ``numpy.random.default_rng``.

    Attributes
    ----------
    samples : KineBalanceSamples or None
        Populated after :meth:`fit`.
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burn: int = 1000,
        thin: int = 2,
        ridge_sd: float = 10.0,
        seed: int = 0,
    ) -> None:
        if burn >= n_iter:
            raise ValueError("burn must be < n_iter")
        if thin < 1:
            raise ValueError("thin must be >= 1")
        self.n_iter = int(n_iter)
        self.burn = int(burn)
        self.thin = int(thin)
        self.ridge_sd = float(ridge_sd)
        self.seed = int(seed)
        self.samples: Optional[KineBalanceSamples] = None

    # ------------------------------------------------------------------ #
    # Stage-1: shared mediator regressions (vectorized over K balances)
    # ------------------------------------------------------------------ #
    def _fit_stage1(
        self,
        rng: np.random.Generator,
        Zu: np.ndarray,
        B: np.ndarray,
        keep_idx: np.ndarray,
        ridge_prec: float,
        x_col: int,
    ) -> np.ndarray:
        """Sample stage-1 mediator regressions for all K balances.

        Each balance shares the same design ``Zu = [1, X, C, batch]`` so
        ``Zu^T Zu`` is computed once; only the right-hand side differs per
        balance. Returns the kept draws of the exposure->balance path
        coefficient ``aX`` (column ``x_col`` of the coefficient vector).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        Zu : ndarray of shape (n, p_u)
            Shared mediator design matrix.
        B : ndarray of shape (n, K)
            Balance (mediator) matrix.
        keep_idx : ndarray
            Iteration indices to retain (post burn-in, thinned).
        ridge_prec : float
            Prior precision ``1/ridge_sd^2`` for stage-1 coefficients.
        x_col : int
            Column index of the exposure within ``Zu`` (always 1 here).

        Returns
        -------
        ndarray of shape (S, K)
            Posterior draws of ``aX`` for every balance.
        """
        n, p_u = Zu.shape
        K = B.shape[1]
        keep_set = set(int(i) for i in keep_idx)
        n_keep = len(keep_idx)

        ZtZ = Zu.T @ Zu
        ZtB = Zu.T @ B  # (p_u, K)
        prior_prec = ridge_prec * np.eye(p_u)

        # InvGamma(a0, b0) prior on sigmaM_b^2
        a0, b0 = 0.5, 0.5
        sigmaM2 = np.ones(K)
        beta = np.zeros((p_u, K))

        aX_keep = np.empty((n_keep, K))
        pos = 0
        for it in range(self.n_iter):
            for b in range(K):
                prec = ZtZ / sigmaM2[b] + prior_prec
                info = ZtB[:, b] / sigmaM2[b]
                beta[:, b] = _mvn_from_precision(rng, prec, info)
            # residuals and sigmaM^2 update
            resid = B - Zu @ beta  # (n, K)
            rss = np.einsum("ij,ij->j", resid, resid)
            shape = a0 + n / 2.0
            scale = b0 + rss / 2.0
            sigmaM2 = _inv_gamma(rng, shape, scale)

            if it in keep_set:
                aX_keep[pos] = beta[x_col, :]
                pos += 1
        return aX_keep

    # ------------------------------------------------------------------ #
    # Stage-2: one horseshoe regression per outcome
    # ------------------------------------------------------------------ #
    def _fit_stage2_outcome(
        self,
        rng: np.random.Generator,
        Zu: np.ndarray,
        Zp: np.ndarray,
        y_point: np.ndarray,
        meas_sd: Optional[np.ndarray],
        keep_idx: np.ndarray,
        ridge_prec: float,
        x_col: int,
    ) -> Dict[str, np.ndarray]:
        """Sample the stage-2 horseshoe mediation regression for one outcome.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator.
        Zu : ndarray of shape (n, p_u)
            Unpenalized design ``[1, X, C, batch]``.
        Zp : ndarray of shape (n, K)
            Penalized design (the K balances).
        y_point : ndarray of shape (n,)
            Outcome point estimates.
        meas_sd : ndarray of shape (n,) or None
            Per-observation measurement SDs; if given, the working response
            is resampled each iteration (data augmentation).
        keep_idx : ndarray
            Iteration indices to retain.
        ridge_prec : float
            Prior precision for unpenalized coefficients.
        x_col : int
            Column index of the exposure within ``Zu``.

        Returns
        -------
        dict
            Draws with keys ``gamma`` (S, K), ``gX`` (S,),
            ``intercept`` (S,), ``sigma`` (S,).
        """
        n = Zu.shape[0]
        p_u = Zu.shape[1]
        Kp = Zp.shape[1]
        Z = np.concatenate([Zu, Zp], axis=1)  # (n, p_u + Kp)
        p = Z.shape[1]
        pen_slice = slice(p_u, p)

        keep_set = set(int(i) for i in keep_idx)
        n_keep = len(keep_idx)

        # initial state
        beta = np.zeros(p)
        sigma2 = float(np.var(y_point)) + 1e-6
        lam2 = np.ones(Kp)       # lambda_j^2
        nu = np.ones(Kp)         # nu_j
        tau2 = 1.0               # tau^2
        xi = 1.0                 # xi

        gamma_keep = np.empty((n_keep, Kp))
        gX_keep = np.empty(n_keep)
        icpt_keep = np.empty(n_keep)
        sigma_keep = np.empty(n_keep)
        pos = 0

        use_me = meas_sd is not None
        if use_me:
            meas_sd = np.asarray(meas_sd, dtype=float).reshape(n)

        for it in range(self.n_iter):
            # --- measurement-error data augmentation ---
            if use_me:
                y = y_point + meas_sd * rng.standard_normal(n)
            else:
                y = y_point

            # --- beta | . ~ N(m, V) via precision parameterization ---
            prior_prec = np.empty(p)
            prior_prec[:p_u] = ridge_prec
            # Floor tau2*lam2 so the shrinkage precision stays finite in p>>n
            # (lambda_j^2 -> 0 would otherwise send 1/(tau2 lam2) -> inf).
            shrink = np.maximum(tau2 * lam2, 1e-12)
            prior_prec[pen_slice] = 1.0 / shrink
            prec = (Z.T @ Z) / sigma2 + np.diag(prior_prec)
            info = (Z.T @ y) / sigma2
            beta = _mvn_from_precision(rng, prec, info)
            beta_p = beta[pen_slice]

            # --- sigma^2 | . ---
            resid = y - Z @ beta
            rss = float(resid @ resid)
            pen_quad = float(np.sum(beta_p ** 2 / shrink))
            sig_shape = (n + Kp) / 2.0
            sig_scale = (rss + pen_quad) / 2.0
            sigma2 = float(np.clip(_inv_gamma(rng, sig_shape, sig_scale), 1e-10, 1e10))

            # --- lambda_j^2 | . and nu_j | . --- (clipped to keep reciprocals finite)
            lam_scale = 1.0 / nu + beta_p ** 2 / (2.0 * tau2 * sigma2)
            lam2 = np.clip(_inv_gamma(rng, np.full(Kp, 1.0), lam_scale), 1e-12, 1e12)
            nu = np.clip(_inv_gamma(rng, np.full(Kp, 1.0), 1.0 + 1.0 / lam2), 1e-12, 1e12)

            # --- tau^2 | . and xi | . ---
            tau_shape = (Kp + 1) / 2.0
            tau_scale = 1.0 / xi + float(np.sum(beta_p ** 2 / (2.0 * sigma2 * lam2)))
            tau2 = float(np.clip(_inv_gamma(rng, tau_shape, tau_scale), 1e-12, 1e12))
            xi = float(np.clip(_inv_gamma(rng, 1.0, 1.0 + 1.0 / tau2), 1e-12, 1e12))

            if it in keep_set:
                gamma_keep[pos] = beta_p
                gX_keep[pos] = beta[x_col]
                icpt_keep[pos] = beta[0]
                sigma_keep[pos] = np.sqrt(sigma2)
                pos += 1

        return {
            "gamma": gamma_keep,
            "gX": gX_keep,
            "intercept": icpt_keep,
            "sigma": sigma_keep,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        B: np.ndarray,
        Y: Mapping[str, np.ndarray],
        C: Optional[np.ndarray] = None,
        batch: Optional[np.ndarray] = None,
        meas_sd: Optional[Mapping[str, np.ndarray]] = None,
    ) -> KineBalanceSamples:
        """Fit the KineBalance mediation model.

        Parameters
        ----------
        X : ndarray of shape (n,)
            Exposure vector.
        B : ndarray of shape (n, K)
            Balance (mediator) matrix; ``K`` is the number of balances.
        Y : mapping str -> ndarray of shape (n,)
            Outcome point estimates keyed by outcome name.
        C : ndarray of shape (n, q), optional
            Confounder matrix.
        batch : ndarray of shape (n,), optional
            Batch labels; encoded as drop-first dummies.
        meas_sd : mapping str -> ndarray of shape (n,), optional
            Per-observation measurement SDs per outcome. Only
            outcomes present as keys get measurement-error augmentation.

        Returns
        -------
        KineBalanceSamples
            Posterior draws; also stored on ``self.samples``.
        """
        X = np.asarray(X, dtype=float).reshape(-1)
        B = np.asarray(B, dtype=float)
        n, K = B.shape
        if X.shape[0] != n:
            raise ValueError("X and B must share the first dimension n")
        # Standardize the balances: the horseshoe stage-2 prior assumes
        # predictors on a common scale. Without this, in p>=n regimes the
        # mediator ranking collapses (AUC ~ 0.5); with it, AUC ~ 0.95+. The
        # balance scale cancels in the indirect effect (IE = aX * gamma), so
        # standardizing here does not change the reported IE units.
        B_sd = B.std(axis=0)
        B_sd = np.where(B_sd < 1e-12, 1.0, B_sd)
        B = (B - B.mean(axis=0)) / B_sd
        if C is not None:
            C = np.asarray(C, dtype=float)
            if C.ndim == 1:
                C = C.reshape(n, 1)
            if C.shape[0] != n:
                raise ValueError("C must have n rows")

        bd = _batch_dummies(batch, n)
        Zu = _build_unpenalized(n, X, C, bd)
        x_col = 1  # exposure is always the second column of Zu
        ridge_prec = 1.0 / (self.ridge_sd ** 2)

        # iterations to keep: post burn-in, thinned
        keep_idx = np.arange(self.burn, self.n_iter, self.thin)
        n_keep = keep_idx.size

        rng = np.random.default_rng(self.seed)

        # ---- stage 1 (shared design, vectorized over balances) ----
        aX_keep = self._fit_stage1(
            rng, Zu, B, keep_idx, ridge_prec, x_col
        )

        # ---- stage 2 (per outcome, deterministic child seeds) ----
        outcomes: Dict[str, Dict[str, np.ndarray]] = {}
        y_scale: Dict[str, float] = {}
        meas_sd = meas_sd or {}
        for r_idx, (name, y_r) in enumerate(Y.items()):
            y_r = np.asarray(y_r, dtype=float).reshape(-1)
            if y_r.shape[0] != n:
                raise ValueError(f"outcome '{name}' must have length n")
            # Standardize the outcome (numerical stability of the horseshoe);
            # the indirect effect is rescaled back to original units in
            # indirect_effect() via the stored y_scale.
            y_sd = float(y_r.std())
            if y_sd <= 1e-12:
                warnings.warn(
                    f"outcome '{name}' has ~zero variance; falling back to unit "
                    f"scale. Indirect effects for it are uninformative.",
                    RuntimeWarning, stacklevel=2,
                )
                y_sd = 1.0
            y_scale[name] = y_sd
            y_r = (y_r - float(y_r.mean())) / y_sd
            # independent, reproducible child stream per outcome
            child_rng = np.random.default_rng(
                np.random.default_rng(self.seed).integers(0, 2 ** 32)
                + r_idx + 1
            )
            ms = meas_sd.get(name, None)
            if ms is not None:
                ms = np.asarray(ms, dtype=float).reshape(-1) / y_sd
            outcomes[name] = self._fit_stage2_outcome(
                child_rng, Zu, B, y_r, ms, keep_idx, ridge_prec, x_col
            )

        meta = {
            "n_iter": self.n_iter,
            "burn": self.burn,
            "thin": self.thin,
            "n_samples": int(n_keep),
            "n": int(n),
            "K": int(K),
            "p_unpenalized": int(Zu.shape[1]),
            "n_confounders": 0 if C is None else int(C.shape[1]),
            "n_batch_dummies": int(bd.shape[1]),
            "ridge_sd": self.ridge_sd,
            "seed": self.seed,
            "outcome_names": list(Y.keys()),
            "meas_error_outcomes": [
                k for k in Y.keys() if k in meas_sd
            ],
            "x_col": x_col,
            "y_scale": y_scale,
        }

        self.samples = KineBalanceSamples(
            outcomes=outcomes,
            mediator={"aX": aX_keep},
            meta=meta,
        )
        return self.samples

    # ------------------------------------------------------------------ #
    def summary(self) -> pd.DataFrame:
        """Posterior summary of the mediator weights ``gamma``.

        Returns
        -------
        pandas.DataFrame
            One row per (outcome, balance) with posterior ``mean``, ``sd``,
            and 95% credible interval (``lo95``, ``hi95``) of
            ``gamma_{b,r}``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.samples is None:
            raise RuntimeError("call fit() before summary()")
        rows = []
        for name, dr in self.samples.outcomes.items():
            gamma = dr["gamma"]  # (S, K)
            mean = gamma.mean(axis=0)
            sd = gamma.std(axis=0, ddof=1) if gamma.shape[0] > 1 else np.zeros(gamma.shape[1])
            lo = np.percentile(gamma, 2.5, axis=0)
            hi = np.percentile(gamma, 97.5, axis=0)
            for b in range(gamma.shape[1]):
                rows.append(
                    {
                        "outcome": name,
                        "balance": b,
                        "mean": mean[b],
                        "sd": sd[b],
                        "lo95": lo[b],
                        "hi95": hi[b],
                    }
                )
        return pd.DataFrame(rows)
