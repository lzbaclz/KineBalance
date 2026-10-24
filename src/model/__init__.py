"""Modeling sub-package.

Main method (v2): a JOINT spike-and-slab two-outcome compositional
mediation model with a fixed Gaussian slab and a subsetting prior tying the two
selection paths, fit by pure-NumPy Gibbs (``mediation_v2``), with a composite-null
Bayesian-FDR specificity decomposition (``specificity_v2``) and exact-enumeration
validation (``validation``). The v1 two-stage horseshoe model (``mediation`` /
``specificity`` / ``indirect``) is retained only as a legacy baseline.
"""

__all__ = [
    "mediation_v2",
    "specificity_v2",
    "validation",
    "diagnostics",
    "sensitivity",
    # legacy v1 (baseline only):
    "mediation",
    "indirect",
    "specificity",
]
