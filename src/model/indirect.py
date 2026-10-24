"""Posterior indirect (mediation) effects from KineBalance draws.

This module implements the natural indirect effect.
For exposure levels ``x_a`` (reference) and ``x_b`` (comparison), the
posterior indirect effect transmitted through balance ``b`` for outcome
``r`` is, draw by draw::

    IE_{b,r}^{(s)} = aX_b^{(s)} * (x_b - x_a) * gamma_{b,r}^{(s)}

where ``aX_b`` is the stage-1 exposure -> balance path and
``gamma_{b,r}`` is the stage-2 balance -> outcome path.
Pairing the *same* posterior draw ``s`` of both paths preserves the full
joint posterior uncertainty of the product-of-coefficients indirect effect.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model.mediation import KineBalanceSamples  # noqa: E402


def indirect_effect(
    samples: KineBalanceSamples,
    x_a: float,
    x_b: float,
) -> Dict[str, np.ndarray]:
    """Posterior indirect-effect draws per balance and outcome.

    Parameters
    ----------
    samples : KineBalanceSamples
        Fitted posterior draws from :class:`~src.model.mediation.KineBalanceModel`.
    x_a : float
        Reference exposure level.
    x_b : float
        Comparison exposure level. The contrast is ``(x_b - x_a)``.

    Returns
    -------
    dict
        Maps each outcome name to an array of shape ``(S, K)`` of indirect
        effect draws ``IE_{b,r}^{(s)} = aX_b^{(s)} (x_b - x_a) gamma_{b,r}^{(s)}``.

    Raises
    ------
    ValueError
        If the stored mediator and outcome draws have mismatched shapes.
    """
    contrast = float(x_b) - float(x_a)
    aX = samples.mediator["aX"]  # (S, K)
    S, K = aX.shape
    # The model standardizes the outcome internally; rescale the indirect effect
    # back to original outcome units via the stored per-outcome SD (the balance
    # scale cancels, so only y_scale is needed). Defaults to 1.0 for samples that
    # predate internal standardization.
    y_scale = samples.meta.get("y_scale", {}) if samples.meta else {}
    out: Dict[str, np.ndarray] = {}
    for name, dr in samples.outcomes.items():
        gamma = dr["gamma"]  # (S, K)
        if gamma.shape != (S, K):
            raise ValueError(
                f"outcome '{name}' gamma shape {gamma.shape} != aX shape {(S, K)}"
            )
        if name not in y_scale:
            warnings.warn(
                f"no y_scale for outcome '{name}'; indirect effects left on the "
                f"standardized (unit-SD) scale. Re-fit with a current model to "
                f"recover original-unit IEs.",
                RuntimeWarning, stacklevel=2,
            )
        out[name] = aX * contrast * gamma * float(y_scale.get(name, 1.0))
    return out


def summarize_ie(ie_draws: np.ndarray) -> pd.DataFrame:
    """Summarize per-balance posterior indirect effects.

    Parameters
    ----------
    ie_draws : ndarray of shape (S, K)
        Posterior indirect-effect draws for a single outcome (one column
        per balance), as returned per outcome by :func:`indirect_effect`.

    Returns
    -------
    pandas.DataFrame
        One row per balance with columns ``balance``, ``mean``, ``lo95``,
        ``hi95``, ``prob_positive`` (posterior ``P(IE > 0)``) and
        ``direction_stability`` (``max(P(IE>0), P(IE<0))``, a value in
        ``[0.5, 1]`` measuring how consistently the effect keeps one sign).
    """
    ie = np.asarray(ie_draws, dtype=float)
    if ie.ndim != 2:
        raise ValueError("ie_draws must be a 2-D (S, K) array")
    S, K = ie.shape
    mean = ie.mean(axis=0)
    lo = np.percentile(ie, 2.5, axis=0)
    hi = np.percentile(ie, 97.5, axis=0)
    p_pos = (ie > 0).mean(axis=0)
    p_neg = (ie < 0).mean(axis=0)
    stability = np.maximum(p_pos, p_neg)
    return pd.DataFrame(
        {
            "balance": np.arange(K),
            "mean": mean,
            "lo95": lo,
            "hi95": hi,
            "prob_positive": p_pos,
            "direction_stability": stability,
        }
    )
