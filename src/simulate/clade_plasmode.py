"""Clade-level count plasmode for the Ward-ILR positive control.

The real-data analysis builds its balances from a DATA-DERIVED tree (Ward clustering
of the taxa), so its calibration must be checked under that exact pipeline. The
existing negative control injects a single-TAXON signal and shows that projecting it
onto a Ward basis spreads it across coordinates and breaks FDR control. This module
provides the matched POSITIVE control: the injected signal is CLADE-structured, i.e.
it moves one sister clade against the other, which is exactly the hypothesis a
balance encodes.

Construction (per seed):

1. Resample samples from a public IBD cohort (Franzosa et al. 2019; top-40 genera)
   with replacement; keep their sequencing depths.
2. Draw ``n_joint + n_f + n_d`` target nodes from the REFERENCE Ward SBP built on
   the real counts (clade size between ``min_size`` and ``max_size`` taxa).
3. For each target node ``b`` with unit ILR loading ``v_b`` (CLR space) add
   ``a_b * X * v_b`` to the CLR composition of the exposed samples. Because the ILR
   basis is orthonormal this changes ONLY coordinate ``b`` of the reference basis;
   ``a_b = c * sqrt(r s / (r + s))`` makes the log-ratio of the two sister-clade
   geometric means shift by exactly ``c`` (a clade-level log-fold change).
4. Draw multinomial counts from the perturbed compositions (count noise, real
   depths); generate both outcomes from the standardized LATENT target balances.

The simulated counts are meant to be analysed with a Ward tree rebuilt on them
(zero replacement, then ILR balances).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.preprocess.balances import SBPNode


def ilr_loading(node: SBPNode, n_parts: int) -> np.ndarray:
    """Unit-norm CLR-space contrast vector of an SBP node (balance = clr @ v)."""
    left = np.fromiter(node.left, dtype=np.intp)
    right = np.fromiter(node.right, dtype=np.intp)
    r, s = left.size, right.size
    v = np.zeros(n_parts)
    v[left] = np.sqrt(s / (r * (r + s)))
    v[right] = -np.sqrt(r / (s * (r + s)))
    return v


def sbp_loadings(sbp: Sequence[SBPNode], n_parts: int) -> np.ndarray:
    """(n_nodes, n_parts) matrix of ILR loadings, rows aligned with ``sbp``."""
    return np.vstack([ilr_loading(nd, n_parts) for nd in sbp])


def node_size(node: SBPNode) -> int:
    return len(node.left) + len(node.right)


def simulate_clade_counts(real_counts: np.ndarray, n: int, seed: int,
                          ref_sbp: Sequence[SBPNode],
                          n_joint: int = 2, n_f: int = 2, n_d: int = 2,
                          clade_lfc: float = 1.0, beta_eff: float = 1.3,
                          direct: float = 0.2, noise: float = 1.0,
                          min_size: int = 3, max_size: int = 20,
                          depth_floor: int = 500) -> Dict[str, object]:
    """Two-outcome count plasmode with CLADE-level mediators (see module docstring).

    Effect sizes mirror :func:`src.simulate.count_plasmode.simulate_two_outcome_counts`
    (log-fold change 1.0 on the mediator, outcome coefficient 1.3 per standardized
    latent mediator, direct effect 0.2, unit noise, binary exposure).
    """
    rng = np.random.default_rng(seed)
    real_counts = np.asarray(real_counts, float)
    N0, P = real_counts.shape
    L = sbp_loadings(ref_sbp, P)
    cand = [j for j, nd in enumerate(ref_sbp) if min_size <= node_size(nd) <= max_size]
    m = int(n_joint) + int(n_f) + int(n_d)
    if len(cand) < m:
        raise ValueError(f"only {len(cand)} candidate clades for {m} targets")
    targets = rng.choice(cand, size=m, replace=False)
    classes = (["joint"] * n_joint) + (["feeding"] * n_f) + (["depoly"] * n_d)

    idx = rng.integers(0, N0, size=n)
    base = real_counts[idx] + 0.5
    depth = np.clip(real_counts[idx].sum(axis=1).astype(int), depth_floor, None)
    logP = np.log(base / base.sum(axis=1, keepdims=True))
    clr = logP - logP.mean(axis=1, keepdims=True)
    X = rng.integers(0, 2, size=n).astype(float)

    signs = rng.choice([-1.0, 1.0], size=m)
    for t, sg in zip(targets, signs):
        nd = ref_sbp[int(t)]
        r, s = len(nd.left), len(nd.right)
        a = clade_lfc * np.sqrt(r * s / (r + s))
        clr = clr + np.outer(X, sg * a * L[int(t)])

    Pp = np.exp(clr - clr.max(axis=1, keepdims=True))
    Pp /= Pp.sum(axis=1, keepdims=True)
    counts = np.empty((n, P), dtype=np.int64)
    for i in range(n):
        counts[i] = rng.multinomial(int(depth[i]), Pp[i])

    z = clr @ L[targets].T                              # latent target balances (n, m)
    z_sd = z.std(axis=0) + 1e-12
    z = (z - z.mean(axis=0)) / z_sd
    yF = direct * X + rng.normal(0.0, noise, n)
    yD = direct * X + rng.normal(0.0, noise, n)
    # Linear effects in CLR space (for coordinate-level truth in ANY analysis basis W:
    # a = W @ exposure_clr_effect, beta_r = W @ outcome_clr_coef[r]). The RNG draws
    # below happen in the same order as before, so simulated data are unchanged.
    exposure_clr = np.zeros(P)
    for t, sg in zip(targets, signs):
        nd = ref_sbp[int(t)]
        r_, s_ = len(nd.left), len(nd.right)
        exposure_clr += sg * clade_lfc * np.sqrt(r_ * s_ / (r_ + s_)) * L[int(t)]
    coefF, coefD = np.zeros(P), np.zeros(P)
    for j, c in enumerate(classes):
        if c in ("joint", "feeding"):
            sgn = rng.choice([-1.0, 1.0])
            yF = yF + beta_eff * sgn * z[:, j]
            coefF += beta_eff * sgn / z_sd[j] * L[int(targets[j])]
        if c in ("joint", "depoly"):
            sgn = rng.choice([-1.0, 1.0])
            yD = yD + beta_eff * sgn * z[:, j]
            coefD += beta_eff * sgn / z_sd[j] * L[int(targets[j])]
    return {"counts": counts, "X": X, "Y": {"F": yF, "D": yD},
            "targets": np.asarray(targets, int), "target_class": np.asarray(classes, object),
            "target_loadings": L[targets], "target_sizes": np.array(
                [node_size(ref_sbp[int(t)]) for t in targets]),
            "exposure_clr_effect": exposure_clr,
            "outcome_clr_coef": {"F": coefF, "D": coefD}}


def match_balances(fit_loadings: np.ndarray, target_loadings: np.ndarray,
                   threshold: float = 0.9) -> Dict[str, np.ndarray]:
    """Match fitted balances to injected clades by |cosine| of their CLR loadings.

    Rows of both inputs are unit-norm contrasts; with an orthonormal fitted basis at
    most one fitted balance can exceed a threshold > 1/sqrt(2) for a given target, so
    the match is unique. Returns, per fitted balance, the best target index and its
    |cosine|, and a boolean ``matched`` mask.
    """
    C = np.abs(fit_loadings @ target_loadings.T)          # (K_fit, m)
    best = C.argmax(axis=1)
    cos = C[np.arange(C.shape[0]), best]
    return {"best": best, "cos": cos, "matched": cos >= threshold, "C": C}
