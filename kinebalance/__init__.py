"""KineBalance: calibrated outcome-specific screening for multi-endpoint compositional mediation.

High-level API (the full analysis in two calls)::

    import kinebalance as kb
    B, info = kb.balances(counts, basis="gtdb")          # counts: samples x genera (DataFrame)
    res = kb.screen(X, B, {"DCA": y1, "propionate": y2}, C=age, prior="decoupled")
    res.table                                             # one row per coordinate

``balances`` keeps genera present in at least 20% of samples and the 40 most abundant,
then builds log-ratio coordinates: ILR balances on the GTDB reference phylogeny
(``"gtdb"``; needs ``bash scripts/get_gtdb.sh`` or ``KINEBALANCE_GTDB_DIR``), on the GTDB
rank hierarchy (``"taxonomy-ladder"`` / ``"taxonomy-balanced"``), on a Ward tree estimated
from the data (``"ward"``), or CLR coordinates (``"clr"``).
``screen`` runs several chains of the spike-and-slab sampler, applies the
Discover-then-Label rule (posterior expected FDR over the feature-by-endpoint grid, then
the four-way label with its margin) and reports per-coordinate convergence diagnostics.
The command-line entry point ``kinebalance`` wraps both (``kinebalance --help``).
"""
from kinebalance.api import ScreenResult, balances, screen

__all__ = ["balances", "screen", "ScreenResult"]
__version__ = "0.2.0"
