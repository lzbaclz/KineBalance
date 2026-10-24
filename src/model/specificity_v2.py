"""Composite-null specificity decomposition from spike-and-slab draws (v2).

Replaces the v1 hard-threshold / product-lfsr scoring with posterior inclusion
probabilities (PIPs) and a *composite-null-aware* Bayesian FDR. For balance ``k``
and outcome ``o``:

    PIP_{k,o} = Pr(gamma_k=1 AND delta_{k,o}=1 | data)        # active mediator
    pnull_{k,o} = 1 - PIP_{k,o}                               # composite null H0: alpha*beta=0

Because ``pnull`` is the posterior probability of the *composite* null (it
absorbs gamma=0, delta=0, and both), the Bayesian FDR built from it (Newton 2004;
Stephens 2017) controls the posterior expected proportion of false discoveries
(declaring a mediator when either path is absent) at level ``alpha`` -- no
separate HDMT/DACT three-component mixture estimation is needed.

The four-way specificity labels reuse :data:`CLASS_LABELS`
(``joint`` / ``feeding`` = F-specific / ``depoly`` = D-specific / ``none``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.model.specificity import CLASS_LABELS

__all__ = [
    "posterior_inclusion_probs",
    "pip_gamma",
    "four_way_specificity",
    "bayes_fdr_qvalues_from_pip",
    "classify_balances_v2",
    "classify_balances_posterior",
    "select_mediators_fdr",
    "nie_summary",
]


def posterior_inclusion_probs(samples) -> Dict[str, np.ndarray]:
    """PIP_{k,o} = Pr(active mediator of o) per outcome, shape (K,) each."""
    g = samples.mediator["gamma"]
    return {o: (g * dr["delta"]).mean(axis=0) for o, dr in samples.outcomes.items()}


def pip_gamma(samples) -> np.ndarray:
    """Pr(gamma_k = 1 | data) -- exposure regulates balance k."""
    return samples.mediator["gamma"].mean(axis=0)


def _two_outcomes(samples, outcome_F, outcome_D):
    names = samples.meta["names"]
    F = outcome_F if outcome_F is not None else names[0]
    D = outcome_D if outcome_D is not None else (names[1] if len(names) > 1 else names[0])
    return F, D


def four_way_specificity(samples, outcome_F: Optional[str] = None,
                         outcome_D: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Per-balance posterior probabilities of the four specificity classes.

    Monte-Carlo over the joint posterior of (gamma_k, delta_{k,F}, delta_{k,D}).
    The four arrays sum to 1 by construction (no overlapping events).
    """
    F, D = _two_outcomes(samples, outcome_F, outcome_D)
    g = samples.mediator["gamma"]
    aF = g * samples.outcomes[F]["delta"]      # active on F (subsetting already in delta)
    aD = g * samples.outcomes[D]["delta"]
    P_joint = (aF * aD).mean(0)
    P_feed = (aF * (1 - aD)).mean(0)
    P_dep = ((1 - aF) * aD).mean(0)
    P_none = 1.0 - P_joint - P_feed - P_dep
    return {"joint": P_joint, "feeding": P_feed, "depoly": P_dep, "none": P_none}


def bayes_fdr_qvalues_from_pip(pip: np.ndarray) -> np.ndarray:
    """Bayesian FDR q-values from PIPs (Newton 2004 / Stephens direct FDR).

    ``pnull = 1 - pip`` is the posterior null probability. Sort ascending by
    ``pnull`` (= descending PIP); the q-value at rank ``j`` is the cumulative
    mean of the sorted ``pnull`` (the posterior expected FDR if the top ``j`` are
    declared). Monotone-guarded and returned in the input order.
    """
    pnull = 1.0 - np.asarray(pip, dtype=float).reshape(-1)
    M = pnull.size
    if M == 0:
        return pnull.copy()
    order = np.argsort(pnull, kind="mergesort")
    s = pnull[order]
    q_sorted = np.maximum.accumulate(np.cumsum(s) / np.arange(1, M + 1))
    q = np.empty(M, dtype=float)
    q[order] = q_sorted
    return np.clip(q, 0.0, 1.0)


def classify_balances_v2(samples, alpha: float = 0.05,
                         outcome_F: Optional[str] = None,
                         outcome_D: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Four-way specificity labels under composite-null FDR control at ``alpha``.

    A balance is active on an outcome iff its Bayesian-FDR q-value (computed over
    the joint F+D grid) is <= alpha. Returns labels plus the PIPs and q-values.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    F, D = _two_outcomes(samples, outcome_F, outcome_D)
    K = samples.meta["K"]
    pip = posterior_inclusion_probs(samples)
    pipF, pipD = pip[F], pip[D]
    q = bayes_fdr_qvalues_from_pip(np.concatenate([pipF, pipD]))  # joint grid
    qF, qD = q[:K], q[K:]
    aF, aD = qF <= alpha, qD <= alpha
    labels = np.empty(K, dtype=object)
    labels[aF & aD] = "joint"
    labels[aF & ~aD] = "feeding"
    labels[~aF & aD] = "depoly"
    labels[~aF & ~aD] = "none"
    return {"labels": labels, "pip_F": pipF, "pip_D": pipD,
            "q_F": qF, "q_D": qD, "active_F": aF, "active_D": aD}


def classify_balances_posterior(samples, alpha: float = 0.10,
                                outcome_F: Optional[str] = None,
                                outcome_D: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Two-stage specificity rule: cell-FDR discovery gate + posterior-class label.

    This is the recommended rule. It separates the two decisions that
    :func:`classify_balances_v2` conflates by crossing two marginal thresholds:

    Stage 1 (discovery, error-controlled): a balance is *active* iff it passes the
    cell-level composite-null Bayesian FDR on at least one outcome (its pooled
    F+D-grid q-value is <= alpha for F or D).

    Stage 2 (label, posterior): among active balances, the four-way label is the
    argmax of the GENUINE joint posterior class probabilities
    (:func:`four_way_specificity`: joint / F-specific=feeding / D-specific=depoly),
    NOT a crossing of two marginal thresholds. We also return the winning-class
    posterior ``class_post`` and the ``margin`` over the runner-up, so a "specific"
    call is reported WITH its confidence rather than implied to be posterior proof of
    absence on the other outcome. Because the label is an argmax of a posterior (not
    a threshold crossing) it does not flip with alpha; alpha only sets the discovery
    set.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    F, D = _two_outcomes(samples, outcome_F, outcome_D)
    K = int(samples.meta["K"])
    pip = posterior_inclusion_probs(samples)
    pipF, pipD = pip[F], pip[D]
    q = bayes_fdr_qvalues_from_pip(np.concatenate([pipF, pipD]))
    qF, qD = q[:K], q[K:]
    active = (qF <= alpha) | (qD <= alpha)                 # discovery gate (>= 1 outcome)
    fw = four_way_specificity(samples, outcome_F=F, outcome_D=D)
    classes = ["joint", "feeding", "depoly"]               # candidate active classes
    P = np.vstack([fw[c] for c in classes])                # (3, K)
    top = P.argmax(axis=0)
    srt = np.sort(P, axis=0)                               # ascending: top=[-1], 2nd=[-2]
    margin = srt[-1] - srt[-2]
    labels = np.array(["none"] * K, dtype=object)
    class_post = np.zeros(K)
    for k in range(K):
        if active[k]:
            labels[k] = classes[int(top[k])]
            class_post[k] = float(P[int(top[k]), k])
    return {"labels": labels, "class_post": class_post, "margin": margin,
            "P_joint": fw["joint"], "P_feeding": fw["feeding"], "P_depoly": fw["depoly"],
            "pip_F": pipF, "pip_D": pipD, "q_F": qF, "q_D": qD, "active": active,
            "active_F": qF <= alpha, "active_D": qD <= alpha}


def select_mediators_fdr(samples, outcome: str, alpha: float = 0.10) -> Dict[str, np.ndarray]:
    """Single-outcome mediator selection at target Bayesian FDR ``alpha``.

    A feature is declared a mediator iff its composite-null Bayesian-FDR q-value
    (from its PIP = Pr(gamma & delta)) is <= alpha. Used for the SOTA benchmark,
    where KineBalance selects taxa (CLR coordinates) head-to-head with taxon-based
    competitors. Returns ``selected`` (bool, K), ``pip`` (K), ``qval`` (K).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    pip = posterior_inclusion_probs(samples)[outcome]
    q = bayes_fdr_qvalues_from_pip(pip)
    return {"selected": q <= alpha, "pip": pip, "qval": q}


def nie_summary(samples, outcome: str) -> pd.DataFrame:
    """Per-balance natural-indirect-effect posterior summary (original units).

    Columns: balance, mean, lo95, hi95, and lfsr (local false sign rate, the
    secondary direction-confidence report).
    """
    nie = samples.outcomes[outcome]["nie"]  # (S, K), original outcome-SD units
    mean = nie.mean(0)
    lo = np.percentile(nie, 2.5, axis=0)
    hi = np.percentile(nie, 97.5, axis=0)
    # Local false sign rate with spike-aware tie handling: the point mass at
    # NIE=0 (inactive draws) is split half to each sign, so lfsr stays in [0, 0.5]
    # (0.5 = sign undetermined / inactive, 0 = sign certain). Interpreted for
    # FDR-declared-active balances.
    p_neg = (nie < 0).mean(0)
    p_pos = (nie > 0).mean(0)
    p_zero = (nie == 0).mean(0)
    lfsr = np.minimum(p_neg, p_pos) + 0.5 * p_zero
    return pd.DataFrame({
        "balance": np.arange(nie.shape[1]),
        "mean": mean, "lo95": lo, "hi95": hi, "lfsr": lfsr,
    })
