"""Label-level (class-specific) Bayesian FDR for the four-way specificity labels.

Cell-level FDR is controlled at discovery, and the four-way
label is then assigned by the posterior-class argmax. That argmax is a point decision with a
confidence margin but no error guarantee on the *label*. This module adds a
class-specific Bayesian FDR on the label itself.

Setup. From the joint posterior we have, per balance k, the class posteriors
    P_c(k) = Pr(L_k = c | data),  c in {joint, F-specific, D-specific}
(``four_way_specificity``; these sum with P_none to 1). For a declared class-c set
S_c, the posterior expected proportion of balances whose true label is not c is

    cFDR_c(S_c) = (1/|S_c|) * sum_{k in S_c} (1 - P_c(k)),                     (1)

the exact Bayesian FDR (Newton 2004; Stephens 2017) for the class-c null
{L_k != c}. Because (1) is a posterior expectation of the realized false-class
proportion, controlling it at alpha controls the posterior expected class-FDR:

    Proposition. Let 1-P_c be the per-item class-c null probabilities. Sorting them
    ascending and declaring the largest prefix whose cumulative mean stays <= alpha
    yields a set S_c with E[ FDP_c(S_c) | data ] <= alpha, where
    FDP_c = (# declared c but true label != c) / max(1, |declared c|).
    (Direct Bayesian FDR; the cumulative mean of the sorted null probabilities is
    exactly the posterior expected FDP of the corresponding prefix.)

    Lemma (argmax restriction is conservative). We do NOT declare the whole prefix
    S_c; we declare only D_c = {k : argmax_c' P_c'(k) = c and q_c(k) <= alpha} <= S_c.
    Each k in S_c \\ D_c has argmax != c, so P_c(k) <= P_c'(k) for some c' != c, hence
    P_c(k) <= 1/2 and (for alpha <= 1/2, which covers all FDR targets) 1-P_c(k) >= 1/2 >= alpha. Removing items whose null prob is
    >= alpha from a set whose mean is <= alpha keeps the mean <= alpha:
    sum_{D_c}(1-P_c) = sum_{S_c}(1-P_c) - sum_{S_c\\D_c}(1-P_c) <= alpha|S_c| -
    alpha|S_c\\D_c| = alpha|D_c|. So E[FDP_c(D_c) | data] <= alpha as well -- the
    argmax-restricted labels inherit the guarantee, and the restriction is
    (weakly) conservative.

A balance is assigned its argmax class only if that class's q-value <= alpha;
otherwise it is left ``uncertain`` (declared active by the cell-FDR discovery, but
not confidently classifiable). This is strictly stronger than the argmax label: it
drops low-margin calls into ``uncertain`` rather than
committing to a class the posterior does not support.

``classfdr_labels`` is the class-FDR decision rule; ``class_fdr_qvalues``
exposes the per-class q-values; ``sbc_check`` is a simulation-based calibration
harness that verifies the empirical class-FDR tracks the target.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.model.specificity_v2 import four_way_specificity, posterior_inclusion_probs
from src.model.specificity_v2 import bayes_fdr_qvalues_from_pip

_CLASSES = ("joint", "feeding", "depoly")   # feeding=F-specific, depoly=D-specific


def class_fdr_qvalues(class_post: np.ndarray) -> np.ndarray:
    """Bayesian-FDR q-values for one class from its posterior probabilities P_c(k).

    Reuses the direct-FDR machinery on pnull = 1 - P_c (sort ascending by pnull,
    cumulative-mean, monotone-guarded, returned in input order).
    """
    return bayes_fdr_qvalues_from_pip(np.asarray(class_post, float))


def classfdr_labels(samples, alpha: float = 0.10,
                    outcome_F: Optional[str] = None,
                    outcome_D: Optional[str] = None,
                    require_discovery: bool = True,
                    margin_gate: float = 0.0) -> Dict[str, np.ndarray]:
    """Label-level (class-FDR-controlled) four-way assignment.

    For each class c, compute q_c via :func:`class_fdr_qvalues` on P_c. A balance is
    labelled c iff c is its argmax class AND q_c[k] <= alpha (and, if
    ``require_discovery``, it also passes the cell-level discovery gate on at least
    one outcome). Balances that are active but whose best class fails class-FDR are
    labelled ``uncertain``; the rest are ``none``.

    ``margin_gate`` (abstention widening) additionally requires the winning class's
    posterior margin over the runner-up class to be >= margin_gate; borderline calls
    (small margin) are demoted to ``uncertain``. This trades power for frequentist
    calibration at the detection boundary, where the fixed-slab posterior class-probs
    are overconfident (boundary SBC). margin_gate=0 recovers the plain rule.

    Returns labels, the per-class q-values, the argmax class posterior, and the
    discovery mask. ``alpha`` must be <= 1/2 (the regime the argmax-restriction
    Lemma covers, which includes every FDR target of interest).
    """
    if not (0.0 < alpha <= 0.5):
        # The argmax-restriction Lemma needs alpha <= 1/2 (a removed item has
        # 1-P_c >= 1/2, which must be >= alpha for the FDP bound to carry to D_c).
        raise ValueError(
            f"alpha must lie in (0, 0.5]; the argmax-restriction guarantee "
            f"requires alpha <= 1/2; got {alpha}")
    fw = four_way_specificity(samples, outcome_F=outcome_F, outcome_D=outcome_D)
    K = int(samples.meta["K"])
    P = np.vstack([fw[c] for c in _CLASSES])                 # (3, K)
    q = np.vstack([class_fdr_qvalues(P[i]) for i in range(3)])  # (3, K) class q-values
    top = P.argmax(axis=0)
    class_post = P[top, np.arange(K)]
    # posterior margin of the winning class over the runner-up class (over the 3 classes)
    Psort = np.sort(P, axis=0)                               # ascending, (3, K)
    class_margin = Psort[-1] - Psort[-2]                     # top - second, (K,)

    # cell-level discovery gate if requested
    if require_discovery:
        names = samples.meta["names"]
        F = outcome_F or names[0]
        D = outcome_D or (names[1] if len(names) > 1 else names[0])
        pip = posterior_inclusion_probs(samples)
        qcell = bayes_fdr_qvalues_from_pip(np.concatenate([pip[F], pip[D]]))
        active = (qcell[:K] <= alpha) | (qcell[K:] <= alpha)
    else:
        active = np.ones(K, bool)

    labels = np.array(["none"] * K, dtype=object)
    class_q = np.full(K, np.nan)
    for k in range(K):
        if not active[k]:
            continue
        c = int(top[k])
        class_q[k] = float(q[c, k])
        labels[k] = _CLASSES[c] if (q[c, k] <= alpha and class_margin[k] >= margin_gate) else "uncertain"
    return {"labels": labels, "class_q": class_q, "class_post": class_post,
            "class_margin": class_margin,
            "q_by_class": {c: q[i] for i, c in enumerate(_CLASSES)}, "active": active}


def sbc_check(simulate_fn, fit_fn, n_rep: int = 200, alpha: float = 0.10,
              seed: int = 0) -> Dict[str, float]:
    """Simulation-based calibration of the class-FDR rule.

    ``simulate_fn(rng) -> (data, true_labels)`` and ``fit_fn(data) -> samples``.
    Over ``n_rep`` replicates, declare labels with :func:`classfdr_labels` and pool
    the realized class false-discovery proportion across all class declarations.
    Returns the pooled empirical class-FDR (should be <= alpha up to Monte-Carlo
    error) and the mean number of confident declarations.
    """
    rng = np.random.default_rng(seed)
    fp = tot = 0
    n_decl = []
    for _ in range(n_rep):
        data, true_labels = simulate_fn(rng)
        s = fit_fn(data)
        res = classfdr_labels(s, alpha=alpha, require_discovery=False)
        decl = [k for k in range(len(res["labels"]))
                if str(res["labels"][k]) in _CLASSES]
        n_decl.append(len(decl))
        for k in decl:
            tot += 1
            if str(res["labels"][k]) != str(true_labels[k]):
                fp += 1
    return {"empirical_class_fdr": fp / max(tot, 1), "target_alpha": alpha,
            "mean_declarations": float(np.mean(n_decl)), "n_rep": n_rep}
