"""Count-space plasmode for a FAIR SOTA mediation benchmark.

Most competitors (MarZIC, CRAmed, ccmm, SparseMCMM) operate on taxon COUNTS, not
on phylogenetic balances. To compare them head-to-head with KineBalance on a
COMMON ground truth, we resample genus counts from a public IBD cohort (Franzosa et al. 2019), preserving the real
compositional covariance, a self-contained stand-in for SparseDOSSA2, which would
not build here), then inject a known exposure -> mediator-taxon -> outcome path on
a chosen set of taxa. The true mediators are exactly those taxa, so every method's
discoveries can be scored (FDR / power / F1) at the taxon level.

KineBalance is run on the CLR coordinates of the same counts so it, too, selects
taxa -- apples-to-apples with the taxon-based competitors.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def clr(counts: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    """Centered log-ratio of a count matrix (rows = samples)."""
    c = np.asarray(counts, float) + pseudo
    logc = np.log(c)
    return logc - logc.mean(axis=1, keepdims=True)


def simulate_count_plasmode(real_counts: np.ndarray, n: int, m_med: int, seed: int,
                            alpha_eff: float = 1.0, beta_eff: float = 1.2,
                            direct: float = 0.2, noise: float = 1.0,
                            min_prev: float = 0.3,
                            outcome_model: str = "clr") -> Dict[str, np.ndarray]:
    """Generate one count-space plasmode dataset with known mediator taxa.

    Parameters
    ----------
    real_counts : (N0, K) real genus count matrix to resample from.
    n : number of samples to draw. m_med : number of true mediator taxa.
    alpha_eff : exposure -> mediator log-proportion shift. beta_eff : mediator
    -> outcome coefficient. direct : direct exposure -> outcome effect.
    outcome_model : "clr" (outcome linear in the mediator's log-ratio; favoured by
    log-contrast methods) or "ra" (outcome linear in standardized relative
    abundance; the marginal model MarZIC assumes). Running both is a DGP-fairness
    sensitivity check so no method is favoured by construction.

    Returns dict: counts (n,K int), X (n,), Y (n,), clr (n,K), true_mediators (K,)
    0/1, med_idx (m_med,).
    """
    rng = np.random.default_rng(seed)
    real_counts = np.asarray(real_counts, float)
    N0, K = real_counts.shape
    if m_med >= K:
        raise ValueError("m_med must be < number of taxa")

    # bootstrap rows -> realistic compositional structure; relative abundances
    idx = rng.integers(0, N0, size=n)
    C = real_counts[idx] + 0.5
    depth = C.sum(axis=1)
    depth_draw = np.clip(depth.astype(int), 500, None)        # sequencing depth
    P = C / C.sum(axis=1, keepdims=True)

    X = rng.integers(0, 2, size=n).astype(float)              # binary exposure

    prev = (real_counts > 0).mean(axis=0)
    cand = np.where(prev > min_prev)[0]
    if cand.size < m_med:
        cand = np.argsort(prev)[::-1][:max(m_med * 3, m_med)]
    med = rng.choice(cand, size=m_med, replace=False)
    a_signs = rng.choice([-1.0, 1.0], size=m_med)

    # exposure -> mediator taxa (shift their log-proportion), then renormalize
    logP = np.log(P)
    for j, s in zip(med, a_signs):
        logP[:, j] += alpha_eff * s * X
    Pp = np.exp(logP)
    Pp /= Pp.sum(axis=1, keepdims=True)

    counts = np.empty((n, K), dtype=int)
    for i in range(n):
        counts[i] = rng.multinomial(depth_draw[i], Pp[i])

    M_clr = clr(counts)                                       # OBSERVED; what methods regress on
    # Generate the outcome from the LATENT composition (pre-count-noise) so NO method
    # sees its exact regression predictor -- each estimates the mediator from the
    # observed counts through its own (noisy) transform. Predictors are standardized
    # so the two outcome models carry comparable signal strength.
    latent_logP = np.log(Pp)
    latent_clr = latent_logP - latent_logP.mean(axis=1, keepdims=True)
    b_signs = rng.choice([-1.0, 1.0], size=m_med)
    Y = direct * X + rng.normal(0.0, noise, size=n)
    col_sd = []
    for j, b in zip(med, b_signs):
        col = Pp[:, j] if outcome_model == "ra" else latent_clr[:, j]   # ra: MarZIC link
        col_sd.append(col.std() + 1e-12)
        col = (col - col.mean()) / col_sd[-1]
        Y = Y + beta_eff * b * col

    true = np.zeros(K, dtype=int)
    true[med] = 1
    return {"counts": counts, "X": X, "Y": Y, "clr": M_clr,
            "true_mediators": true, "med_idx": np.sort(med),
            # effect bookkeeping (unsorted mediator order) for coordinate-level truth
            "med_order": np.asarray(med, int), "a_signs": a_signs, "b_signs": b_signs,
            "col_sd": np.asarray(col_sd, float), "alpha_eff": float(alpha_eff),
            "beta_eff": float(beta_eff), "outcome_model": outcome_model}


def simulate_two_outcome_counts(real_counts: np.ndarray, n: int, seed: int,
                                m_joint: int = 2, m_fonly: int = 2, m_donly: int = 2,
                                alpha_eff: float = 1.0, beta_eff: float = 1.3,
                                direct: float = 0.2, noise: float = 1.0,
                                min_prev: float = 0.3) -> Dict[str, np.ndarray]:
    """Two-outcome count plasmode with KNOWN specificity structure (taxon level).

    Like :func:`simulate_count_plasmode` but assigns mediator taxa into three
    groups -- ``joint`` (affect both outcomes F and D), ``F``-only, ``D``-only --
    so the four-way specificity partition has a ground truth. Counts are drawn
    from the (exposure-perturbed) latent composition; both outcomes are generated
    from the LATENT log-ratios (pre-count-noise), so no method sees its exact
    predictor. Returns counts, X, the two outcomes, the per-taxon specificity
    label, and the mediator index groups.
    """
    rng = np.random.default_rng(seed)
    real_counts = np.asarray(real_counts, float)
    N0, K = real_counts.shape
    m_med = m_joint + m_fonly + m_donly
    if m_med >= K:
        raise ValueError("too many mediators for K taxa")

    idx = rng.integers(0, N0, size=n)
    C = real_counts[idx] + 0.5
    depth = np.clip(C.sum(axis=1).astype(int), 500, None)
    P = C / C.sum(axis=1, keepdims=True)
    X = rng.integers(0, 2, size=n).astype(float)

    prev = (real_counts > 0).mean(axis=0)
    cand = np.where(prev > min_prev)[0]
    if cand.size < m_med:
        cand = np.argsort(prev)[::-1][:max(m_med * 3, m_med)]
    med = rng.choice(cand, size=m_med, replace=False)
    joint, fonly, donly = med[:m_joint], med[m_joint:m_joint + m_fonly], med[m_joint + m_fonly:]
    a_signs = {int(j): float(s) for j, s in zip(med, rng.choice([-1.0, 1.0], size=m_med))}

    logP = np.log(P)
    for j in med:                                            # exposure -> all mediator taxa
        logP[:, j] += alpha_eff * a_signs[j] * X
    Pp = np.exp(logP); Pp /= Pp.sum(axis=1, keepdims=True)
    counts = np.empty((n, K), dtype=int)
    for i in range(n):
        counts[i] = rng.multinomial(depth[i], Pp[i])

    latent = np.log(Pp); latent = latent - latent.mean(axis=1, keepdims=True)

    def _signal(taxa):
        s = np.zeros(n)
        for j in taxa:
            col = latent[:, j]; col = (col - col.mean()) / (col.std() + 1e-12)
            s = s + beta_eff * rng.choice([-1.0, 1.0]) * col
        return s

    Y_F = direct * X + _signal(list(joint) + list(fonly)) + rng.normal(0, noise, n)
    Y_D = direct * X + _signal(list(joint) + list(donly)) + rng.normal(0, noise, n)

    label = np.array(["none"] * K, dtype=object)
    for j in joint: label[j] = "joint"
    for j in fonly: label[j] = "F"
    for j in donly: label[j] = "D"
    return {"counts": counts, "X": X, "Y": {"F": Y_F, "D": Y_D},
            "taxon_label": label, "med_idx": np.sort(med),
            "joint": np.sort(joint), "fonly": np.sort(fonly), "donly": np.sort(donly)}


def isolating_balance(sbp, K_taxa: int) -> np.ndarray:
    """For each taxon, the SBP balance index that most isolates it (largest |ILR
    loading|), purely from SBP geometry -- a non-circular taxon->balance map."""
    nb = len(sbp)
    best = np.full(K_taxa, -1, dtype=int)
    best_w = np.full(K_taxa, -1.0)
    for b, node in enumerate(sbp):
        r, s = len(node.left), len(node.right)
        if r == 0 or s == 0:
            continue
        wL = np.sqrt(s / (r * (r + s)))                     # |loading| for left members
        wR = np.sqrt(r / (s * (r + s)))                     # |loading| for right members
        for j in node.left:
            if wL > best_w[j]: best_w[j] = wL; best[j] = b
        for j in node.right:
            if wR > best_w[j]: best_w[j] = wR; best[j] = b
    return best
