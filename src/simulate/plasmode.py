"""Semi-synthetic (plasmode) and misspecification generators.

These break the *self-matched-generator circularity* of the pure simulation,
where data are drawn from exactly the structural model KineBalance
assumes, so strong recovery is partly tautological. Two probes:

* **plasmode** -- inject a KNOWN sparse indirect effect on top of a REAL
  microbiome balance covariance from a public IBD cohort (Franzosa et al. 2019), so the mediator structure
  (correlation, sparsity, heavy tails) is empirical rather than Gaussian. If
  recovery survives, performance is not an artefact of the generator.
* **misspecification** -- generate outcomes that VIOLATE the fit model's
  assumptions and measure how recovery/coverage degrade vs a well-specified
  control:
    - ``heavy_tail``  : Student-$t_3$ outcome noise (the model assumes Gaussian);
    - ``nonlinear``   : a quadratic mediator->outcome term (model assumes linear);
    - ``confounded``  : an UNMEASURED common cause of mediators and outcome
                        (violates sequential ignorability -- the key causal
                        assumption); coverage should drop and IE bias.

Every generator returns the SAME dict schema as
``src.simulate.generator.simulate_dataset`` (keys ``X, C, batch, B, Y,
true_mediators, true_gamma, true_alphaX, true_IE, counts``). The true
natural indirect effect is the structural ``gamma * alphaX * delta_x`` in every
mode, so IE-RMSE / coverage / AUC remain well defined even under
misspecification.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.simulate.generator import LEVELS, SimConfig, _build_mediator_sets

MODES = ("wellspec", "plasmode", "heavy_tail", "nonlinear", "confounded")


def _exposure(n: int, rng: np.random.Generator):
    """Centred exposure on a 3-level design, and the low-to-high contrast."""
    sizes = np.asarray(LEVELS["exposure_levels"], dtype=float)
    idx = rng.integers(0, sizes.size, size=n)
    X = np.log(sizes[idx]) - float(np.log(sizes).mean())
    x_lo, x_hi = LEVELS["ie_contrast"]
    return X, float(np.log(x_hi) - np.log(x_lo))


def _coeffs(cfg: SimConfig, rng: np.random.Generator):
    """Active mediator sets (partial F/D overlap) and structural coefficients."""
    med = _build_mediator_sets(cfg, rng)
    union = np.unique(np.concatenate([idx for idx in med.values()]))
    alphaX = np.zeros(cfg.n_balances)
    alphaX[union] = rng.uniform(0.6, 1.4, union.size) * rng.choice([-1.0, 1.0], union.size)
    names = ["F"] if cfg.outcomes == "F" else ["F", "D_Mw", "D_Mn"]
    gamma: Dict[str, np.ndarray] = {}
    for nm in names:
        a = med[nm]
        g = np.zeros(cfg.n_balances)
        g[a] = rng.uniform(0.5, 1.5, a.size) * rng.choice([-1.0, 1.0], a.size)
        gamma[nm] = g
    return med, alphaX, gamma, names


def simulate_step3(
    mode: str,
    n: int,
    n_balances: int,
    sparsity: float,
    seed: int,
    outcomes: str = "FD",
    real_B: Optional[np.ndarray] = None,
    nonlin: float = 0.8,
    conf: float = 1.2,
) -> Dict[str, object]:
    """Generate one Step-3 dataset in the ``simulate_dataset`` dict schema.

    Parameters
    ----------
    mode : str
        One of :data:`MODES`.
    real_B : ndarray, optional
        Real balance matrix (rows = samples) used for ``mode="plasmode"``; rows
        are bootstrap-resampled to ``n`` and the first ``n_balances`` columns are
        used, centred and rescaled to the synthetic residual SD ($0.5$).
    nonlin, conf : float
        Strength of the quadratic link (``nonlinear``) and of the unmeasured
        confounder (``confounded``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    rng = np.random.default_rng(seed)
    cfg = SimConfig(n=n, n_balances=n_balances, sparsity=sparsity,
                    outcomes=outcomes, batch_strength="none", missing="none", seed=seed)

    X, delta_x = _exposure(n, rng)
    C = rng.standard_normal((n, cfg.p_covariates))
    batch = np.zeros(n, dtype=int)
    med, alphaX, gamma, names = _coeffs(cfg, rng)

    # --- Balance residual structure: real (plasmode) vs Gaussian (others). ---
    if mode == "plasmode":
        if real_B is None or real_B.shape[1] < n_balances:
            raise ValueError("plasmode needs real_B with >= n_balances columns")
        ridx = rng.integers(0, real_B.shape[0], size=n)
        Bres = np.asarray(real_B[ridx, :n_balances], dtype=float).copy()
        Bres -= Bres.mean(axis=0)
        Bres *= 0.5 / (Bres.std() + 1e-12)   # match synthetic eps_b SD
    else:
        Bres = rng.normal(0.0, 0.5, size=(n, n_balances))

    # --- Unmeasured confounder (only used, and never revealed, when confounded).
    U = rng.standard_normal(n)
    conf_w = np.zeros(n_balances)
    if mode == "confounded":
        conf_w[np.unique(np.concatenate([med[k] for k in names]))] = 1.0

    B = np.outer(X, alphaX) + Bres
    if mode == "confounded":
        B = B + conf * np.outer(U, conf_w)

    # --- Outcomes; structural true IE is gamma*alphaX*delta in every mode. ---
    Y: Dict[str, np.ndarray] = {}
    true_gamma: Dict[str, np.ndarray] = {}
    true_med: Dict[str, np.ndarray] = {}
    true_IE: Dict[str, np.ndarray] = {}
    for nm in names:
        g = gamma[nm]
        lin = B @ g
        y = 0.1 * X + lin
        if mode == "nonlinear":
            ls = (lin - lin.mean()) / (lin.std() + 1e-12)
            y = y + nonlin * (ls ** 2)
        if mode == "confounded":
            y = y + conf * U
        if mode == "heavy_tail":
            t = rng.standard_t(3, size=n)
            y = y + t / (t.std() + 1e-12)
        else:
            y = y + rng.normal(0.0, 1.0, size=n)
        Y[nm] = y
        true_gamma[nm] = g
        true_med[nm] = med[nm]
        true_IE[nm] = g * alphaX * delta_x

    return {
        "X": X, "C": C, "batch": batch, "B": B, "Y": Y,
        "true_mediators": true_med, "true_gamma": true_gamma,
        "true_alphaX": alphaX, "true_IE": true_IE,
        "counts": np.zeros((n, n_balances + 1), dtype=np.int64),  # unused by scorer
    }


def simulate_specificity_design(
    n: int,
    n_balances: int,
    seed: int,
    real_B: np.ndarray,
    n_joint: int = 2,
    n_f: int = 2,
    n_d: int = 2,
    effect_scale: float = 1.0,
    resid_rho: float = 0.0,
    direct: float = 0.1,
    n_yonly_f: int = 0,
    n_yonly_d: int = 0,
) -> Dict[str, object]:
    """Two-outcome balance-level plasmode with an EXPLICIT specificity design.

    Same construction as ``simulate_step3(mode="plasmode", outcomes="FD")`` (balance
    residuals from a public IBD cohort (Franzosa et al. 2019), bootstrap-resampled and rescaled to SD 0.5; the
    3-level centred exposure; exposure->balance paths ``U(0.6, 1.4)``;
    balance->outcome paths ``U(0.5, 1.5)``; unit outcome noise), but the class
    sizes are set directly so a caller can vary them:

    * ``n_joint`` / ``n_f`` / ``n_d`` -- balances mediating both outcomes, only
      ``F``, only ``D`` (the rest are ``none``);
    * ``effect_scale`` -- multiplies every balance->outcome coefficient (signal
      strength of the mediated path);
    * ``resid_rho`` -- correlation of the two outcome residuals
      ``(eps_F, eps_D)`` given exposure and balances (0 = the independent-residual
      working model is exactly right).
    * ``n_yonly_f`` / ``n_yonly_d`` -- OUTCOME-ONLY null balances: no exposure path
      (``a_k = 0``) but a nonzero balance->outcome path to ``F`` / ``D``. These are
      composite nulls of the other kind (``a_k beta_kr = 0`` because ``a_k = 0``);
      their draws happen after all others, so ``0`` reproduces the default design
      bit for bit.

    Returns the ``simulate_step3`` schema with outcomes ``"F"`` and ``"D_Mw"`` plus
    ``"labels"`` (per-balance truth in ``{"joint","feeding","depoly","none"}``).
    """
    if not -1.0 < resid_rho < 1.0:
        raise ValueError("resid_rho must lie in (-1, 1)")
    m = int(n_joint) + int(n_f) + int(n_d)
    if m < 1 or m > n_balances:
        raise ValueError("need 1 <= n_joint + n_f + n_d <= n_balances")
    if real_B is None or real_B.shape[1] < n_balances:
        raise ValueError("real_B must have >= n_balances columns")
    rng = np.random.default_rng(seed)
    X, delta_x = _exposure(n, rng)

    ridx = rng.integers(0, real_B.shape[0], size=n)
    Bres = np.asarray(real_B[ridx, :n_balances], dtype=float).copy()
    Bres -= Bres.mean(axis=0)
    Bres *= 0.5 / (Bres.std() + 1e-12)

    med = rng.choice(n_balances, size=m, replace=False)
    joint = np.sort(med[:n_joint])
    fonly = np.sort(med[n_joint:n_joint + n_f])
    donly = np.sort(med[n_joint + n_f:])
    alphaX = np.zeros(n_balances)
    alphaX[med] = rng.uniform(0.6, 1.4, m) * rng.choice([-1.0, 1.0], m)
    B = np.outer(X, alphaX) + Bres

    F_set = np.sort(np.concatenate([joint, fonly]))
    D_set = np.sort(np.concatenate([joint, donly]))
    coef: Dict[str, np.ndarray] = {}
    for nm, act in (("F", F_set), ("D_Mw", D_set)):
        g = np.zeros(n_balances)
        g[act] = effect_scale * rng.uniform(0.5, 1.5, act.size) * rng.choice([-1.0, 1.0], act.size)
        coef[nm] = g
    eF = rng.normal(0.0, 1.0, size=n)
    eD0 = rng.normal(0.0, 1.0, size=n)
    eD = resid_rho * eF + np.sqrt(1.0 - resid_rho ** 2) * eD0
    Y = {"F": direct * X + B @ coef["F"] + eF,
         "D_Mw": direct * X + B @ coef["D_Mw"] + eD}

    yonly = np.array([], dtype=int)
    if n_yonly_f + n_yonly_d > 0:
        free = np.setdiff1d(np.arange(n_balances), med)
        yonly = rng.choice(free, size=n_yonly_f + n_yonly_d, replace=False)
        for j, nm in zip(yonly, ["F"] * n_yonly_f + ["D_Mw"] * n_yonly_d):
            c = effect_scale * rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
            Y[nm] = Y[nm] + c * B[:, j]          # alphaX[j] == 0: outcome-only
    labels = np.array(["none"] * n_balances, dtype=object)
    labels[joint] = "joint"
    labels[fonly] = "feeding"
    labels[donly] = "depoly"
    return {
        "outcome_only": np.sort(yonly),
        "X": X, "B": B, "Y": Y, "labels": labels,
        "true_mediators": {"F": F_set, "D_Mw": D_set},
        "true_gamma": coef, "true_alphaX": alphaX,
        "true_IE": {k: v * alphaX * delta_x for k, v in coef.items()},
    }
