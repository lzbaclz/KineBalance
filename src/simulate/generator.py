"""Synthetic compositional-mediation data generator.

This module is the data-generating mechanism for high-dimensional balance
mediation, paired with the PhILR-style balance machinery in
``src.preprocess.balances``.

The generative model is:

    Exposure model:
        X is a centred exposure on a 3-level design.

    Mediator (balance) model:
        B_j = alphaX_j * X + (C @ alphaC_j) + b_batch[g(i)] + eps_b      (1)
        for j = 1, ..., n_balances, where B are latent ILR balances.

    Outcome model:
        Y_o = beta0_o + tauX_o * X + sum_j gamma_{o,j} B_j
              + (C @ betaC_o) + eps_y                                     (2)
        for each outcome o in {F} or {F, D_Mw, D_Mn}.

    Compositional observation model:
        p_i = ilr_inv(B_i)  (closed to the simplex),
        counts_i ~ Multinomial(depth_i, p_i) with zero-inflation dropout.  (3)

    Indirect effect (natural indirect effect):
        IE_{o,j} = gamma_{o,j} * alphaX_j * (X_hi - X_lo),                 (4)
        evaluated for the contrast between the low and high exposure levels.

Partial overlap between the F-specific and D-specific active mediator sets is
imposed. All randomness flows through ``np.random.default_rng`` and
the generator is deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["LEVELS", "SimConfig", "simulate_dataset"]


# ---------------------------------------------------------------------------
# Factor levels.
# ---------------------------------------------------------------------------
LEVELS: Dict[str, object] = {
    "n": (24, 48, 96),
    "n_balances": (50, 150),
    "sparsity": (0.03, 0.1),
    "zero_inflation": ("low", "med", "high"),
    "phylo_cluster": ("scattered", "clustered"),
    "batch_strength": ("none", "low", "med", "high"),
    "outcomes": ("F", "FD"),
    "missing": ("none", "mcar", "mar"),
    # Exposure design: a centred exposure on a 3-level design.
    "exposure_levels": (200.0, 400.0, 600.0),
    # Low-to-high contrast used for the indirect-effect ground truth (Eq. 4).
    "ie_contrast": (200.0, 600.0),
}

# Mapping from the zero-inflation level to (median sequencing depth, dropout
# probability). Lower depth + higher dropout -> sparser counts.
_ZERO_INFLATION_MAP: Dict[str, Tuple[int, float]] = {
    "low": (50_000, 0.05),
    "med": (10_000, 0.20),
    "high": (2_000, 0.45),
}

# Mapping from batch_strength to the standard deviation of the batch random
# intercept added to the balances (Eq. 1, term b_batch).
_BATCH_STRENGTH_MAP: Dict[str, float] = {
    "none": 0.0,
    "low": 0.15,
    "med": 0.40,
    "high": 0.80,
}


@dataclass
class SimConfig:
    """Configuration for a single synthetic dataset.

    Parameters
    ----------
    n : int
        Number of samples.
    n_balances : int
        Number of latent ILR balances (so ``n_taxa = n_balances + 1``).
    sparsity : float
        Fraction of balances that are true mediators for a given outcome.
    zero_inflation : str
        One of ``{"low", "med", "high"}`` controlling depth and dropout.
    phylo_cluster : str
        ``"scattered"`` spreads active mediators across the balance index;
        ``"clustered"`` places them in a contiguous block (phylogenetic clade).
    batch_strength : str
        One of ``{"none", "low", "med", "high"}`` controlling batch variance.
    outcomes : str
        ``"F"`` for a single outcome, ``"FD"`` for F plus D_Mw and D_Mn.
    missing : str
        ``"none"``, ``"mcar"`` or ``"mar"`` missingness mechanism.
    seed : int
        Master seed for the deterministic RNG.
    """

    n: int = 48
    n_balances: int = 50
    sparsity: float = 0.1
    zero_inflation: str = "med"
    phylo_cluster: str = "scattered"
    batch_strength: str = "med"
    outcomes: str = "FD"
    missing: str = "mcar"
    seed: int = 0
    p_covariates: int = 3
    n_batches: int = 4
    field_meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.zero_inflation not in _ZERO_INFLATION_MAP:
            raise ValueError(
                f"zero_inflation must be one of {tuple(_ZERO_INFLATION_MAP)}; "
                f"got {self.zero_inflation!r}"
            )
        if self.batch_strength not in _BATCH_STRENGTH_MAP:
            raise ValueError(
                f"batch_strength must be one of {tuple(_BATCH_STRENGTH_MAP)}; "
                f"got {self.batch_strength!r}"
            )
        if self.phylo_cluster not in ("scattered", "clustered"):
            raise ValueError(
                f"phylo_cluster must be 'scattered' or 'clustered'; "
                f"got {self.phylo_cluster!r}"
            )
        if self.outcomes not in ("F", "FD"):
            raise ValueError(
                f"outcomes must be 'F' or 'FD'; got {self.outcomes!r}"
            )
        if self.missing not in ("none", "mcar", "mar"):
            raise ValueError(
                f"missing must be 'none', 'mcar' or 'mar'; got {self.missing!r}"
            )
        if not 0.0 < self.sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1]; got {self.sparsity}")
        if self.n_balances < 2:
            raise ValueError("n_balances must be >= 2.")


# ---------------------------------------------------------------------------
# ILR helpers (Helmert / Gram-Schmidt contrast basis).
# ---------------------------------------------------------------------------
def _ilr_basis(n_taxa: int) -> np.ndarray:
    """Return an orthonormal ILR contrast basis ``Psi`` of shape ``(D-1, D)``.

    Uses the Helmert sub-matrix construction, which yields a valid SBP-free
    orthonormal basis for the isometric log-ratio transform.

    Parameters
    ----------
    n_taxa : int
        Number of taxa ``D`` (so the basis has ``D - 1`` rows).

    Returns
    -------
    numpy.ndarray
        Orthonormal basis of shape ``(n_taxa - 1, n_taxa)`` with rows summing to
        zero and unit norm.
    """
    d = n_taxa
    psi = np.zeros((d - 1, d), dtype=np.float64)
    for i in range(1, d):
        scale = np.sqrt(i / (i + 1.0))
        psi[i - 1, :i] = 1.0 / i
        psi[i - 1, i] = -1.0
        psi[i - 1, : i + 1] *= scale
    # Normalize defensively so rows are exactly unit norm.
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    return psi


def _ilr_inv(balances: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Map ILR balances back to compositions on the simplex (Eq. 3).

    Parameters
    ----------
    balances : numpy.ndarray
        Array of shape ``(n, D-1)``.
    psi : numpy.ndarray
        Orthonormal ILR basis of shape ``(D-1, D)``.

    Returns
    -------
    numpy.ndarray
        Compositions of shape ``(n, D)`` closed to sum one.
    """
    clr = balances @ psi  # (n, D) in clr space.
    clr -= clr.max(axis=1, keepdims=True)  # numerical stabilization.
    comp = np.exp(clr)
    comp /= comp.sum(axis=1, keepdims=True)
    return comp


def _active_indices(
    n_balances: int, k: int, phylo_cluster: str, rng: np.random.Generator
) -> np.ndarray:
    """Select ``k`` active balance indices given the clustering scheme.

    Parameters
    ----------
    n_balances : int
        Total number of balances.
    k : int
        Number of active mediators to select.
    phylo_cluster : str
        ``"scattered"`` or ``"clustered"``.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Sorted integer index array of length ``k``.
    """
    k = int(min(max(k, 1), n_balances))
    if phylo_cluster == "clustered":
        start = int(rng.integers(0, n_balances - k + 1))
        idx = np.arange(start, start + k)
    else:
        idx = rng.choice(n_balances, size=k, replace=False)
    return np.sort(idx)


def _build_mediator_sets(
    cfg: SimConfig, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Build per-outcome active mediator sets with partial overlap.

    The F-specific and D-specific sets share roughly half of their members and
    differ in the remainder.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    dict
        Mapping from outcome name to the sorted active-index array.
    """
    k = max(1, int(round(cfg.sparsity * cfg.n_balances)))
    base = _active_indices(cfg.n_balances, k, cfg.phylo_cluster, rng)
    sets: Dict[str, np.ndarray] = {"F": base}

    if cfg.outcomes == "FD":
        # D outcomes share ~half of F's mediators, plus fresh ones (partial
        # overlap). D_Mw and D_Mn share their own structure but each
        # perturbs one element so they are not identical.
        n_shared = max(1, k // 2)
        shared = rng.choice(base, size=min(n_shared, base.size), replace=False)
        remaining = np.setdiff1d(np.arange(cfg.n_balances), base)
        n_fresh = k - shared.size
        if n_fresh > 0 and remaining.size > 0:
            fresh = rng.choice(
                remaining, size=min(n_fresh, remaining.size), replace=False
            )
        else:
            fresh = np.array([], dtype=int)
        d_core = np.sort(np.concatenate([shared, fresh]))
        sets["D_Mw"] = d_core
        # Perturb D_Mn by swapping one mediator so the two D outcomes differ.
        pool = np.setdiff1d(np.arange(cfg.n_balances), d_core)
        if pool.size > 0 and d_core.size > 0:
            swap_out = int(rng.choice(d_core))
            swap_in = int(rng.choice(pool))
            d_mn = np.sort(
                np.concatenate(
                    [np.setdiff1d(d_core, [swap_out]), np.array([swap_in])]
                )
            )
        else:
            d_mn = d_core.copy()
        sets["D_Mn"] = d_mn
    return sets


def _missing_mask(
    cfg: SimConfig,
    X: np.ndarray,
    C: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a sample-level missingness mask.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    X : numpy.ndarray
        Exposure of shape ``(n,)`` (used for MAR).
    C : numpy.ndarray
        Covariates of shape ``(n, p_c)`` (first column used for MAR).
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(n,)`` where ``True`` marks an observed sample.
    """
    n = cfg.n
    if cfg.missing == "none":
        return np.ones(n, dtype=bool)
    if cfg.missing == "mcar":
        return rng.random(n) >= 0.10
    # MAR: missingness probability depends on the first covariate.
    z = (C[:, 0] - C[:, 0].mean()) / (C[:, 0].std() + 1e-8)
    p_miss = 1.0 / (1.0 + np.exp(-(z - 0.5)))  # logistic in covariate.
    p_miss = 0.05 + 0.30 * p_miss  # keep in a reasonable band.
    return rng.random(n) >= p_miss


def simulate_dataset(
    n: int = 48,
    n_balances: int = 50,
    sparsity: float = 0.1,
    zero_inflation: str = "med",
    phylo_cluster: str = "scattered",
    batch_strength: str = "med",
    outcomes: str = "FD",
    missing: str = "mcar",
    seed: int = 0,
) -> Dict[str, object]:
    """Simulate one compositional mediation dataset with ground truth.

    Implements the structural model of Equations (1)-(4): an exposure ``X`` drawn
    from a 3-level design drives latent ILR balances ``B`` through
    ``true_alphaX``; a sparse subset of balances mediate each outcome through
    ``true_gamma``; compositional counts are generated by inverse-ILR plus a
    zero-inflated multinomial; and natural indirect effects for the
    low-to-high exposure contrast are returned as ground truth.

    Parameters
    ----------
    n : int, default=48
        Number of samples (levels: 24, 48, 96).
    n_balances : int, default=50
        Number of latent ILR balances (levels: 50, 150).
    sparsity : float, default=0.1
        Fraction of balances acting as mediators (levels: 0.03, 0.1).
    zero_inflation : str, default="med"
        One of ``{"low", "med", "high"}`` controlling depth and dropout.
    phylo_cluster : str, default="scattered"
        ``"scattered"`` or ``"clustered"`` mediator placement.
    batch_strength : str, default="med"
        One of ``{"none", "low", "med", "high"}`` batch-effect magnitude.
    outcomes : str, default="FD"
        ``"F"`` for one outcome or ``"FD"`` for F plus D_Mw and D_Mn.
    missing : str, default="mcar"
        ``"none"``, ``"mcar"`` or ``"mar"`` missingness mechanism.
    seed : int, default=0
        Master seed; the function is deterministic given this seed.

    Returns
    -------
    dict
        Dictionary with keys:

        ``X`` : ndarray (n,)
            Centred exposure on a 3-level design.
        ``C`` : ndarray (n, p_c)
            Covariates.
        ``batch`` : ndarray (n,)
            Integer batch ids.
        ``counts`` : ndarray (n, n_taxa)
            Zero-inflated compositional counts.
        ``B`` : ndarray (n, n_balances)
            True latent ILR balances.
        ``Y`` : dict
            Outcomes; always contains ``"F"`` and, when ``outcomes == "FD"``,
            ``"D_Mw"`` and ``"D_Mn"``.
        ``true_mediators`` : dict
            Outcome -> active balance index array.
        ``true_gamma`` : dict
            Outcome -> length-``n_balances`` coefficient vector (Eq. 2).
        ``true_alphaX`` : ndarray (n_balances,)
            Exposure -> balance coefficients (Eq. 1).
        ``true_IE`` : dict
            Outcome -> length-``n_balances`` natural indirect effect for the
            low-to-high exposure contrast (Eq. 4).
        ``survival`` : ndarray (n,)
            Latent survival-time-like response derived from the active balances.
        ``mask`` : ndarray (n,) of bool
            ``True`` marks an observed sample (consistent with ``missing``).
        ``config`` : SimConfig
            The resolved configuration.
        ``ilr_basis`` : ndarray (n_balances, n_taxa)
            Orthonormal ILR basis used to build the compositions.
    """
    cfg = SimConfig(
        n=n,
        n_balances=n_balances,
        sparsity=sparsity,
        zero_inflation=zero_inflation,
        phylo_cluster=phylo_cluster,
        batch_strength=batch_strength,
        outcomes=outcomes,
        missing=missing,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    n_taxa = cfg.n_balances + 1
    p_c = cfg.p_covariates

    # --- Exposure: centred exposure on a 3-level design.
    sizes = np.asarray(LEVELS["exposure_levels"], dtype=np.float64)
    level_idx = rng.integers(0, sizes.size, size=cfg.n)
    log_level = np.log(sizes[level_idx])
    log_center = float(np.log(sizes).mean())
    X = log_level - log_center

    # Low-to-high contrast on the centred log scale (Eq. 4).
    x_lo, x_hi = LEVELS["ie_contrast"]
    delta_x = float(np.log(x_hi) - np.log(x_lo))

    # --- Covariates: standard-normal confounders.
    C = rng.standard_normal((cfg.n, p_c))

    # --- Batch design (Eq. 1, term b_batch).
    batch = rng.integers(0, cfg.n_batches, size=cfg.n)
    batch_sd = _BATCH_STRENGTH_MAP[cfg.batch_strength]
    batch_intercepts = rng.normal(0.0, 1.0, size=(cfg.n_batches, cfg.n_balances))
    batch_intercepts *= batch_sd

    # --- Mediator (balance) model coefficients (Eq. 1).
    med_sets = _build_mediator_sets(cfg, rng)
    union_active = np.unique(
        np.concatenate([idx for idx in med_sets.values()])
    )

    # alphaX is nonzero on the union of active mediators so that every active
    # mediator is genuinely on the X -> B -> Y path.
    true_alphaX = np.zeros(cfg.n_balances, dtype=np.float64)
    alpha_mag = rng.uniform(0.6, 1.4, size=union_active.size)
    alpha_sign = rng.choice([-1.0, 1.0], size=union_active.size)
    true_alphaX[union_active] = alpha_mag * alpha_sign

    # Covariate -> balance loadings (confounding of mediators).
    alphaC = rng.normal(0.0, 0.3, size=(p_c, cfg.n_balances))

    # --- Latent balances B (Eq. 1).
    eps_b = rng.normal(0.0, 0.5, size=(cfg.n, cfg.n_balances))
    B = (
        np.outer(X, true_alphaX)
        + C @ alphaC
        + batch_intercepts[batch]
        + eps_b
    )

    # --- Outcome model (Eq. 2) and indirect effects (Eq. 4).
    Y: Dict[str, np.ndarray] = {}
    true_gamma: Dict[str, np.ndarray] = {}
    true_IE: Dict[str, np.ndarray] = {}
    true_mediators: Dict[str, np.ndarray] = {}

    outcome_names = ["F"] if cfg.outcomes == "F" else ["F", "D_Mw", "D_Mn"]
    # Direct (unmediated) effect of X on each outcome.
    tauX = {name: float(rng.uniform(-0.3, 0.3)) for name in outcome_names}
    betaC = {name: rng.normal(0.0, 0.2, size=p_c) for name in outcome_names}
    beta0 = {name: float(rng.normal(0.0, 0.1)) for name in outcome_names}

    for name in outcome_names:
        active = med_sets[name]
        gamma = np.zeros(cfg.n_balances, dtype=np.float64)
        g_mag = rng.uniform(0.5, 1.5, size=active.size)
        g_sign = rng.choice([-1.0, 1.0], size=active.size)
        gamma[active] = g_mag * g_sign

        eps_y = rng.normal(0.0, 1.0, size=cfg.n)
        y = (
            beta0[name]
            + tauX[name] * X
            + B @ gamma
            + C @ betaC[name]
            + eps_y
        )
        Y[name] = y
        true_gamma[name] = gamma
        true_mediators[name] = active
        # Natural indirect effect for the low-to-high exposure contrast (Eq. 4).
        true_IE[name] = gamma * true_alphaX * delta_x

    # --- Survival-like response derived from the union of F mediators.
    # A log-normal survival time whose hazard increases with the F balance score.
    f_score = B @ true_gamma["F"]
    f_score_std = (f_score - f_score.mean()) / (f_score.std() + 1e-8)
    eps_s = rng.normal(0.0, 0.4, size=cfg.n)
    survival = np.exp(2.0 - 0.5 * f_score_std + eps_s)

    # --- Compositional counts via inverse-ILR + zero-inflated multinomial (Eq. 3).
    psi = _ilr_basis(n_taxa)
    comp = _ilr_inv(B, psi)  # (n, n_taxa) on the simplex.

    depth_med, dropout_p = _ZERO_INFLATION_MAP[cfg.zero_inflation]
    # Sample-specific depth: log-normal around the median depth.
    depths = rng.lognormal(
        mean=np.log(depth_med), sigma=0.35, size=cfg.n
    ).astype(np.int64)
    depths = np.clip(depths, 100, None)

    # Zero-inflation: Bernoulli dropout mask zeroes out taxa before drawing.
    dropout = rng.random((cfg.n, n_taxa)) < dropout_p
    p_eff = comp.copy()
    p_eff[dropout] = 0.0
    row_sums = p_eff.sum(axis=1, keepdims=True)
    # Guard against fully-dropped rows by reinstating the original composition.
    degenerate = (row_sums <= 0).ravel()
    if degenerate.any():
        p_eff[degenerate] = comp[degenerate]
        row_sums[degenerate] = comp[degenerate].sum(axis=1, keepdims=True)
    p_eff /= row_sums

    counts = np.zeros((cfg.n, n_taxa), dtype=np.int64)
    for i in range(cfg.n):
        counts[i] = rng.multinomial(int(depths[i]), p_eff[i])

    # --- Missingness mask.
    mask = _missing_mask(cfg, X, C, rng)

    return {
        "X": X,
        "C": C,
        "batch": batch,
        "counts": counts,
        "B": B,
        "Y": Y,
        "true_mediators": true_mediators,
        "true_gamma": true_gamma,
        "true_alphaX": true_alphaX,
        "true_IE": true_IE,
        "survival": survival,
        "mask": mask,
        "config": cfg,
        "ilr_basis": psi,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check.
    import os
    import sys

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    data = simulate_dataset(seed=0)
    cfg = data["config"]
    n_taxa = cfg.n_balances + 1
    assert data["X"].shape == (cfg.n,)
    assert data["counts"].shape == (cfg.n, n_taxa)
    assert data["B"].shape == (cfg.n, cfg.n_balances)
    assert set(data["Y"]) == {"F", "D_Mw", "D_Mn"}
    assert data["true_alphaX"].shape == (cfg.n_balances,)
    # F and D mediator sets only partially overlap.
    overlap = np.intersect1d(
        data["true_mediators"]["F"], data["true_mediators"]["D_Mw"]
    )
    assert 0 < overlap.size < data["true_mediators"]["F"].size
    print("generator smoke check OK")
