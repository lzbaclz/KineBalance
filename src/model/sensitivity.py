"""Unmeasured-confounding sensitivity analysis for indirect effects (E-values).

For an observational exposure, the natural indirect effect is only identified
under sequential ignorability (no unmeasured mediator--outcome confounding),
which is untestable. The **E-value** (VanderWeele & Ding, 2017) quantifies how
strong such confounding would have to be to explain the estimate away: an
unmeasured confounder associated with both the mediator and the outcome by a
risk ratio of at least the E-value (and not already controlled) could nullify
the result; weaker confounding could not.

Our indirect effects are on the outcome-SD scale, so we use the standard
approximate conversion of a standardized mean difference ``d`` to a risk ratio
``RR = exp(0.91 * d)`` (VanderWeele, 2017, following Chinn 2000) before applying
the E-value formula. This is an approximation and targets the mediator->outcome
path; report it as such.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["smd_to_rr", "e_value", "indirect_effect_evalue"]

_SMD_TO_LOGRR = 0.91  # VanderWeele (2017): RR ~= exp(0.91 * standardized effect)


def smd_to_rr(d: float) -> float:
    """Approximate risk ratio for a standardized mean difference ``d``."""
    return float(np.exp(_SMD_TO_LOGRR * abs(d)))


def e_value(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele & Ding, 2017).

    ``E = RR + sqrt(RR * (RR - 1))`` after orienting ``RR >= 1``. ``RR = 1``
    (null) gives ``E = 1`` (no confounding needed).
    """
    rr = float(rr)
    if rr <= 0:
        return float("nan")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def indirect_effect_evalue(estimate: float, lo: float, hi: float) -> Dict[str, float]:
    """E-value for an indirect effect estimate and its CI (outcome-SD units).

    Returns the E-value for the point estimate and for the confidence limit
    closest to the null (the standard reported pair). If the CI already crosses
    zero, the CI E-value is ``1.0`` (no confounding is needed to include the
    null).

    Parameters
    ----------
    estimate, lo, hi : float
        Indirect-effect mean and 95% interval bounds, in outcome-SD units.

    Returns
    -------
    dict
        ``rr_point``, ``evalue_point``, ``evalue_ci`` (and the bound used).
    """
    rr_point = smd_to_rr(estimate)
    ev_point = e_value(rr_point)
    crosses_zero = (lo <= 0.0 <= hi)
    if crosses_zero:
        ev_ci = 1.0
        bound = 0.0
    else:
        bound = lo if abs(lo) < abs(hi) else hi  # CI limit nearest the null
        ev_ci = e_value(smd_to_rr(bound))
    return {
        "rr_point": rr_point,
        "evalue_point": ev_point,
        "evalue_ci": ev_ci,
        "ci_bound_used": float(bound),
    }
