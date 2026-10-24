"""Posterior specificity scoring and 4-class balance assignment.

Implements the specificity decomposition.  For
each candidate balance ``k`` we have posterior **draws** of two indirect
effects (IE):

* ``IE_F`` -- the indirect effect on endpoint ``F``;
* ``IE_D`` -- the indirect effect on endpoint ``D``.

Both arrays have shape ``(S, K)`` where ``S`` is the number of posterior draws
and ``K`` the number of balances.  Given practical-significance thresholds
``tau_F`` and ``tau_D`` (a region of practical equivalence),
four mutually exclusive posterior events are defined per draw ``b``::

    A_F = { |IE_F| >  tau_F }            (endpoint F active)
    A_D = { |IE_D| >  tau_D }            (endpoint D active)

    specificity (D-specific):             A_D and not A_F
    feeding      (F-specific):            A_F and not A_D
    joint        (both endpoints active): A_F and A_D
    none         (neither endpoint active): not A_F and not A_D

The corresponding **posterior probabilities** are the Monte-Carlo averages of
these indicator events over the ``S`` draws::

    S_b^{spec}    = (1/S) * sum_b 1{ |IE_D,b| >  tau_D  and  |IE_F,b| <= tau_F }
    S_b^{feed}    = (1/S) * sum_b 1{ |IE_F,b| >  tau_F  and  |IE_D,b| <= tau_D }
    S_b^{joint}   = (1/S) * sum_b 1{ |IE_F,b| >  tau_F  and  |IE_D,b| >  tau_D }
    S_b^{none}    = (1/S) * sum_b 1{ |IE_F,b| <= tau_F  and  |IE_D,b| <= tau_D }

The four probabilities sum to one for every balance.  A balance is assigned the
label whose posterior probability is *dominant* and exceeds the decision
threshold ``decision``; otherwise it is labelled ``"none"``
(dominant-probability assignment rule).
"""
# Legacy v1 model, kept as a baseline; the main method is
# src/model/mediation_v2.py (specificity_v2.py).


from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "specificity_score",
    "feeding_score",
    "joint_score",
    "none_score",
    "posterior_class_probabilities",
    "classify_balances",
    "local_false_sign_rate",
    "bayes_fdr_qvalues",
    "classify_balances_fdr",
    "CLASS_LABELS",
]

#: Canonical label ordering for the four specificity classes.
#: ``"depoly"`` is the D-specific class returned by
#: :func:`specificity_score`.
CLASS_LABELS: Tuple[str, str, str, str] = ("joint", "feeding", "depoly", "none")


def _validate_draws(
    ie_F_draws: np.ndarray, ie_D_draws: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce and validate a pair of posterior indirect-effect draw arrays.

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior draws of the ``F`` and ``D`` indirect
        effects, each of shape ``(S, K)`` with ``S`` draws and ``K`` balances.

    Returns
    -------
    tuple of numpy.ndarray
        The two arrays as contiguous float64 2-D arrays.

    Raises
    ------
    ValueError
        If the arrays are not 2-D or their shapes disagree.
    """
    F = np.asarray(ie_F_draws, dtype=np.float64)
    D = np.asarray(ie_D_draws, dtype=np.float64)
    if F.ndim != 2 or D.ndim != 2:
        raise ValueError(
            f"draws must be 2-D (S, K); got F.ndim={F.ndim}, D.ndim={D.ndim}"
        )
    if F.shape != D.shape:
        raise ValueError(
            f"ie_F_draws and ie_D_draws must share shape; "
            f"got {F.shape} vs {D.shape}"
        )
    return F, D


def specificity_score(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
) -> np.ndarray:
    """Posterior probability of *D-specific* mediation per balance.

    Implements ``S_b^{spec}``: endpoint ``D`` is active
    (``|IE_D| > tau_D``) while endpoint ``F`` is practically null
    (``|IE_F| <= tau_F``).

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds for the two
        endpoints.

    Returns
    -------
    numpy.ndarray
        Posterior probabilities of shape ``(K,)`` in ``[0, 1]``.
    """
    F, D = _validate_draws(ie_F_draws, ie_D_draws)
    event = (np.abs(D) > tau_D) & (np.abs(F) <= tau_F)
    return event.mean(axis=0)


def feeding_score(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
) -> np.ndarray:
    """Posterior probability of *F-specific* mediation.

    Implements ``S_b^{feed}``: endpoint ``F`` is active
    (``|IE_F| > tau_F``) while endpoint ``D`` is practically null
    (``|IE_D| <= tau_D``).

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds.

    Returns
    -------
    numpy.ndarray
        Posterior probabilities of shape ``(K,)`` in ``[0, 1]``.
    """
    F, D = _validate_draws(ie_F_draws, ie_D_draws)
    event = (np.abs(F) > tau_F) & (np.abs(D) <= tau_D)
    return event.mean(axis=0)


def joint_score(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
) -> np.ndarray:
    """Posterior probability of *joint* mediation (both endpoints active).

    Implements ``S_b^{joint}``: ``|IE_F| > tau_F`` and
    ``|IE_D| > tau_D`` simultaneously.

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds.

    Returns
    -------
    numpy.ndarray
        Posterior probabilities of shape ``(K,)`` in ``[0, 1]``.
    """
    F, D = _validate_draws(ie_F_draws, ie_D_draws)
    event = (np.abs(F) > tau_F) & (np.abs(D) > tau_D)
    return event.mean(axis=0)


def none_score(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
) -> np.ndarray:
    """Posterior probability of *no* mediation (neither endpoint active).

    Implements ``S_b^{none}``: ``|IE_F| <= tau_F`` and
    ``|IE_D| <= tau_D``.  Together with
    :func:`specificity_score`, :func:`feeding_score` and :func:`joint_score`
    the four probabilities partition the unit interval and sum to one per
    balance.

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds.

    Returns
    -------
    numpy.ndarray
        Posterior probabilities of shape ``(K,)`` in ``[0, 1]``.
    """
    F, D = _validate_draws(ie_F_draws, ie_D_draws)
    event = (np.abs(F) <= tau_F) & (np.abs(D) <= tau_D)
    return event.mean(axis=0)


def posterior_class_probabilities(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
) -> Dict[str, np.ndarray]:
    """Compute all four posterior specificity probabilities at once.

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping with keys ``"joint"``, ``"feeding"``, ``"depoly"`` and
        ``"none"`` (the order of :data:`CLASS_LABELS`), each value an array of
        shape ``(K,)``.  By construction the four arrays sum to one elementwise.

    Notes
    -----
    Implements the full partition.  The ``"depoly"`` entry is
    the D-specific score of :func:`specificity_score`.
    """
    if not (tau_F >= 0.0 and tau_D >= 0.0):
        raise ValueError(
            f"thresholds must be non-negative; got tau_F={tau_F}, tau_D={tau_D}"
        )
    return {
        "joint": joint_score(ie_F_draws, ie_D_draws, tau_F, tau_D),
        "feeding": feeding_score(ie_F_draws, ie_D_draws, tau_F, tau_D),
        "depoly": specificity_score(ie_F_draws, ie_D_draws, tau_F, tau_D),
        "none": none_score(ie_F_draws, ie_D_draws, tau_F, tau_D),
    }


def classify_balances(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    tau_F: float,
    tau_D: float,
    decision: float = 0.5,
) -> np.ndarray:
    """Assign each balance a specificity label via the dominant-probability rule.

    For every balance ``k`` the four posterior probabilities
    are computed (see :func:`posterior_class_probabilities`).  The label is the
    class with the largest probability *provided* that probability exceeds the
    ``decision`` threshold; otherwise the balance is labelled ``"none"``
    (dominant-probability assignment rule).

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    tau_F, tau_D : float
        Non-negative practical-significance thresholds.
    decision : float, optional
        Minimum dominant posterior probability required to commit to a non-null
        label, by default ``0.5``.  Must lie in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(K,)`` and dtype ``object`` (string labels) drawn from
        ``{"joint", "feeding", "depoly", "none"}``.

    Notes
    -----
    Ties between equally-dominant classes are resolved by the canonical order
    of :data:`CLASS_LABELS` (``argmax`` returns the first maximum), making the
    function deterministic.  Because ``"none"`` is itself one of the four
    classes, a balance whose dominant class *is* ``"none"`` is labelled
    ``"none"`` regardless of the ``decision`` threshold.
    """
    if not (0.0 <= decision <= 1.0):
        raise ValueError(f"decision must lie in [0, 1]; got {decision}")

    probs = posterior_class_probabilities(ie_F_draws, ie_D_draws, tau_F, tau_D)
    # Stack into (K, 4) following the canonical CLASS_LABELS ordering.
    stacked = np.column_stack([probs[label] for label in CLASS_LABELS])
    K = stacked.shape[0]

    dominant_idx = np.argmax(stacked, axis=1)
    dominant_val = stacked[np.arange(K), dominant_idx]

    labels = np.array(CLASS_LABELS, dtype=object)[dominant_idx]
    # Fall back to "none" when the dominant probability is not decisive.
    labels[dominant_val <= decision] = "none"
    return labels


# --------------------------------------------------------------------------- #
# Threshold-free decision rule: local false sign rate (lfsr) + Bayesian FDR
# (Stephens 2017, "False discovery rates: a new deal", Biostatistics 18(2):275)
# --------------------------------------------------------------------------- #
def local_false_sign_rate(ie_draws: np.ndarray) -> np.ndarray:
    """Local false sign rate (lfsr) per balance from posterior draws.

    ``lfsr_k = min( P(IE_k >= 0), P(IE_k <= 0) )`` estimated as the Monte-Carlo
    average over the ``S`` posterior draws.  It is the posterior probability of
    getting the **sign** of the effect wrong if one commits to its posterior
    sign.  Unlike a test of the exact null ``IE_k = 0`` -- which has probability
    zero under the continuous horseshoe prior and is therefore undefined as a
    discovery criterion -- the lfsr is well posed and requires **no practical-
    significance threshold** (Stephens 2017).

    Parameters
    ----------
    ie_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.

    Returns
    -------
    numpy.ndarray
        Per-balance lfsr of shape ``(K,)`` in ``[0, 0.5]`` (0.5 = sign entirely
        undetermined; 0 = sign certain).
    """
    X = np.asarray(ie_draws, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"ie_draws must be 2-D (S, K); got ndim={X.ndim}")
    p_nonneg = np.mean(X >= 0.0, axis=0)
    p_nonpos = np.mean(X <= 0.0, axis=0)
    return np.minimum(p_nonneg, p_nonpos)


def bayes_fdr_qvalues(lfsr: np.ndarray) -> np.ndarray:
    """Bayesian FDR q-values from per-balance local false sign rates.

    Balances are ranked by increasing ``lfsr``; the q-value of the ``k``-th most
    significant balance is the cumulative **mean** of the sorted lfsr up to
    ``k`` -- the expected proportion of sign errors (false discoveries) incurred
    by selecting the top ``k``.  Selecting every balance with ``q <= alpha``
    therefore controls the Bayesian FDR at ``alpha`` (Stephens 2017).  Because
    the input is sorted ascending, the cumulative mean is non-decreasing; a
    monotone guard is applied for numerical safety.

    Note: this is the Bayesian false-*sign*-rate q-value of Stephens (2017),
    computed from posterior lfsr -- NOT the Benjamini--Hochberg frequentist FDR
    on p-values. There are no p-values here.

    Parameters
    ----------
    lfsr : numpy.ndarray
        Per-balance local false sign rates of shape ``(K,)``.

    Returns
    -------
    numpy.ndarray
        q-values of shape ``(K,)`` in ``[0, 0.5]``, aligned to the input order.
    """
    s = np.asarray(lfsr, dtype=np.float64).reshape(-1)
    K = s.size
    if K == 0:
        return s.copy()
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    q_sorted = np.cumsum(sorted_s) / np.arange(1, K + 1)
    q_sorted = np.maximum.accumulate(q_sorted)  # numerical-safety monotone guard
    q = np.empty(K, dtype=np.float64)
    q[order] = q_sorted
    return q


def classify_balances_fdr(
    ie_F_draws: np.ndarray,
    ie_D_draws: np.ndarray,
    alpha: float = 0.05,
    control: str = "lfsr",
) -> Dict[str, np.ndarray]:
    """Two-outcome specificity partition via lfsr / Bayesian FDR (threshold-free).

    Replaces the arbitrary practical-significance thresholds ``(tau_F, tau_D)``
    and the ``0.5`` dominant-probability cutoff of :func:`classify_balances`
    with a principled rule based on the local false sign rate (Stephens 2017).
    A balance is declared **active** on an outcome when the sign of its indirect
    effect there is resolved with controlled error:

    * ``control="lfsr"`` : active ``<=>`` ``lfsr <= alpha``  (per-balance sign-error rate)
    * ``control="fdr"``  : active ``<=>`` ``q    <= alpha``  (Bayesian FDR across balances)

    The four labels follow from the activity pattern on the two outcomes
    (``F`` = first outcome, label ``"feeding"``; ``D`` = second outcome, label ``"depoly"``)::

        joint   = active on BOTH F and D
        feeding = active on F only
        depoly  = active on D only
        none    = active on neither

    Notes
    -----
    The lfsr is a **sign-error** rate, not a test of the exact null.  Report
    ``alpha`` and ``control`` alongside the partition; results are sensitive to
    ``alpha`` (inspect at least 0.05 and 0.10).

    Parameters
    ----------
    ie_F_draws, ie_D_draws : numpy.ndarray
        Posterior indirect-effect draws of shape ``(S, K)``.
    alpha : float, optional
        Target sign-error / FDR level in ``(0, 0.5]``, by default ``0.05``.
    control : {"lfsr", "fdr"}, optional
        Per-balance lfsr control (default) or across-balance Bayesian FDR.

    Returns
    -------
    dict of str to numpy.ndarray
        ``"labels"`` -- ``(K,)`` object array from :data:`CLASS_LABELS`;
        ``"lfsr_F"``, ``"lfsr_D"`` -- per-outcome lfsr;
        ``"q_F"``, ``"q_D"`` -- per-outcome Bayesian-FDR q-values;
        ``"active_F"``, ``"active_D"`` -- boolean activity masks.
    """
    F, D = _validate_draws(ie_F_draws, ie_D_draws)
    if not (0.0 < alpha <= 0.5):
        raise ValueError(f"alpha must lie in (0, 0.5]; got {alpha}")
    if control not in ("lfsr", "fdr"):
        raise ValueError(f"control must be 'lfsr' or 'fdr'; got {control!r}")

    lfsr_F = local_false_sign_rate(F)
    lfsr_D = local_false_sign_rate(D)
    q_F = bayes_fdr_qvalues(lfsr_F)
    q_D = bayes_fdr_qvalues(lfsr_D)

    crit_F = lfsr_F if control == "lfsr" else q_F
    crit_D = lfsr_D if control == "lfsr" else q_D
    active_F = crit_F <= alpha
    active_D = crit_D <= alpha

    K = F.shape[1]
    labels = np.empty(K, dtype=object)
    labels[active_F & active_D] = "joint"
    labels[active_F & ~active_D] = "feeding"
    labels[~active_F & active_D] = "depoly"
    labels[~active_F & ~active_D] = "none"

    return {
        "labels": labels,
        "lfsr_F": lfsr_F, "lfsr_D": lfsr_D,
        "q_F": q_F, "q_D": q_D,
        "active_F": active_F, "active_D": active_D,
    }
