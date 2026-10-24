"""High-level API: coordinates from counts, and the Discover-then-Label screen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASES = ("gtdb", "taxonomy-ladder", "taxonomy-balanced", "ward", "clr")
CLASS_NAMES = {"joint": "joint", "feeding": "{F}-specific", "depoly": "{D}-specific", "none": "none"}


def balances(counts: pd.DataFrame, basis: str = "ward", prevalence_min: float = 0.20,
             top_n: int = 40, pseudo_counts: Optional[float] = None,
             seed: int = 0) -> Tuple[pd.DataFrame, dict]:
    """Log-ratio coordinates of a samples-by-taxa count table.

    Taxa present in at least ``prevalence_min`` of the samples are kept, then the
    ``top_n`` with the largest mean relative abundance. Zeros are replaced
    multiplicatively (``0.5 / p`` of the closed composition) unless ``pseudo_counts`` is
    given. For a tree basis the taxon names must be GTDB lineage strings
    (``d__...;p__...;...;g__X``) when ``basis`` is ``"gtdb"`` or ``"taxonomy-*"``.

    Returns the coordinates (DataFrame, samples x K) and a dict with the kept taxa and,
    for ILR bases, the sequential binary partition (numerator / denominator taxa of
    every balance).
    """
    from src.preprocess.balances import filter_balances, philr_balances, taxonomy_tree, zero_replace
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}; got {basis!r}")
    counts = pd.DataFrame(counts).astype(float)
    if (counts.to_numpy() < 0).any():
        raise ValueError("counts must be non-negative")
    prev = (counts > 0).mean(axis=0)
    kept = counts.loc[:, prev >= prevalence_min]
    rel = kept.div(kept.sum(axis=1).clip(lower=1e-12), axis=0)
    if kept.shape[1] > top_n:
        order = np.sort(np.argsort(rel.mean(axis=0).to_numpy())[::-1][:top_n])
        kept = kept.iloc[:, order]
    taxa = [str(c) for c in kept.columns]
    M = kept.to_numpy()
    if basis == "clr":
        comp = zero_replace(M, pseudo_counts=pseudo_counts)
        lc = np.log(comp)
        B = lc - lc.mean(axis=1, keepdims=True)
        return (pd.DataFrame(B, index=kept.index, columns=[f"clr:{t.split(';')[-1]}" for t in taxa]),
                {"basis": basis, "taxa": taxa})
    tree = None
    if basis == "gtdb":
        from src.preprocess.phylogeny import gtdb_genus_tree
        tree = gtdb_genus_tree(taxa)
    elif basis.startswith("taxonomy"):
        tree = taxonomy_tree(taxa, resolve=basis.split("-", 1)[1])
    B, sbp, _ = philr_balances(M, tree=tree, taxa_ids=taxa, seed=seed, pseudo_counts=pseudo_counts)
    B, keep = filter_balances(B, var_threshold=1e-8)
    nodes = [sbp[int(i)] for i in keep]
    short = [t.split(";")[-1] for t in taxa]
    names = [f"b{j}" for j in range(len(nodes))]
    parts = pd.DataFrame({"coordinate": names,
                          "numerator": [";".join(short[i] for i in nd.left) for nd in nodes],
                          "denominator": [";".join(short[i] for i in nd.right) for nd in nodes],
                          "node": [nd.label for nd in nodes]})
    return pd.DataFrame(B, index=kept.index, columns=names), {"basis": basis, "taxa": taxa,
                                                              "partition": parts, "tree": tree}


@dataclass
class ScreenResult:
    """Output of :func:`screen`: one row per coordinate plus run metadata."""

    table: pd.DataFrame
    meta: Dict[str, object] = field(default_factory=dict)

    def calls(self) -> pd.DataFrame:
        """Coordinates passing the discovery gate, with their labels."""
        return self.table[self.table["label"] != "none"]

    def summary(self) -> str:
        c = self.calls()
        lines = [f"KineBalance ({self.meta['prior']} prior, {self.meta['chains']} chains x "
                 f"{self.meta['n_iter']} sweeps, alpha={self.meta['alpha']}): "
                 f"{len(c)} of {len(self.table)} coordinates called"]
        for _, r in c.iterrows():
            lines.append(f"  {r['coordinate']}: {r['label']} (class posterior {r['class_post']:.2f}, "
                         f"margin {r['margin']:.2f}, ESS {r['ess_min']:.0f}, R-hat {r['rhat_max']:.3f})")
        return "\n".join(lines)


def screen(X, B, Y: Mapping[str, Sequence[float]], C=None, prior: str = "decoupled",
           chains: int = 4, n_iter: int = 24000, burn: int = 4000, alpha: float = 0.10,
           seed: int = 100, slab_scale: float = 2.0) -> ScreenResult:
    """Fit KineBalance and apply the Discover-then-Label rule.

    ``X``: exposure (n,); ``B``: log-ratio coordinates (n, K) or DataFrame; ``Y``: one or
    two endpoints (name -> (n,)); ``C``: pre-exposure covariates (n, q) or None.
    ``prior``: ``"coupled"`` (an endpoint path needs the exposure path; more power, assumes
    every endpoint-acting coordinate is exposure-regulated) or ``"decoupled"``. Chains use
    seeds ``seed, seed+1, ...`` and are pooled. Discovery bounds the posterior expected
    FDR at ``alpha`` over the coordinate-by-endpoint cells; with two endpoints each
    discovery gets the four-way label (joint / endpoint-specific) with its posterior
    margin, which carries no separate error guarantee.
    """
    from src.model.diagnostics import ess_per_chain, gelman_rubin_rhat
    from src.model.mediation_v2 import SpikeSlabMediation, SpikeSlabSamples
    from src.model.specificity_v2 import classify_balances_posterior, select_mediators_fdr
    if prior not in ("coupled", "decoupled"):
        raise ValueError("prior must be 'coupled' or 'decoupled'")
    names = list(Y)
    if len(names) not in (1, 2):
        raise ValueError("screen supports one or two endpoints")
    cols = list(B.columns) if isinstance(B, pd.DataFrame) else [f"b{j}" for j in range(np.shape(B)[1])]
    Xa, Ba = np.asarray(X, float).reshape(-1), np.asarray(B, float)
    Ya = {o: np.asarray(Y[o], float).reshape(-1) for o in names}
    Ca = None if C is None else np.asarray(C, float)
    for nm, v in [("X", Xa), ("B", Ba)] + [(o, Ya[o]) for o in names] + ([("C", Ca)] if Ca is not None else []):
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{nm} contains missing or non-finite values")
    gm, dl = [], {o: [] for o in names}
    for c in range(chains):
        s = SpikeSlabMediation(n_iter=n_iter, burn=burn, thin=2, seed=seed + c,
                               slab_scale=slab_scale, subsetting=prior == "coupled").fit(Xa, Ba, Ya, C=Ca)
        g = s.mediator["gamma"].astype(np.int8)
        gm.append(g)
        for o in names:
            dl[o].append(g * s.outcomes[o]["delta"].astype(np.int8))
    S = min(x.shape[0] for x in gm)
    G = np.stack([x[:S] for x in gm])                                 # (chains, S, K)
    D = {o: np.stack([x[:S] for x in dl[o]]) for o in names}
    K = Ba.shape[1]
    pooled = SpikeSlabSamples(mediator={"gamma": G.reshape(-1, K)},
                              outcomes={o: {"delta": D[o].reshape(-1, K)} for o in names},
                              meta={"names": names, "K": K})

    def _diag(series):
        ess, rh = np.full(K, np.nan), np.full(K, np.nan)
        for k in range(K):
            e, r = [], []
            for x in series:
                xk = x[:, :, k].astype(float)
                if xk.std() < 1e-12:
                    continue
                e.append(float(ess_per_chain(xk[:, :, None])[:, 0].sum()))
                if chains > 1:
                    r.append(float(gelman_rubin_rhat(xk[:, :, None])[0]))
            ess[k] = min(e) if e else np.nan
            rh[k] = max(r) if r else np.nan
        return ess, rh

    tab = pd.DataFrame({"coordinate": cols, "pip_gamma": G.mean(axis=(0, 1))})
    for o in names:
        tab[f"pip_{o}"] = D[o].mean(axis=(0, 1))
    if len(names) == 2:
        F, Dn = names
        res = classify_balances_posterior(pooled, alpha=alpha, outcome_F=F, outcome_D=Dn)
        lab = {k: v.format(F=F, D=Dn) for k, v in CLASS_NAMES.items()}
        tab[f"q_{F}"], tab[f"q_{Dn}"] = res["q_F"], res["q_D"]
        tab["P_joint"], tab[f"P_{F}_only"], tab[f"P_{Dn}_only"] = res["P_joint"], res["P_feeding"], res["P_depoly"]
        tab["label"] = [lab[str(x)] for x in res["labels"]]
        tab["class_post"], tab["margin"] = res["class_post"], res["margin"]
        ind = {"joint": D[F] * D[Dn], "feeding": D[F] * (1 - D[Dn]), "depoly": (1 - D[F]) * D[Dn]}
        win = np.argmax(np.vstack([res["P_joint"], res["P_feeding"], res["P_depoly"]]), axis=0)
        wser = np.stack([[ind[("joint", "feeding", "depoly")[w]][c, :, k] for k, w in enumerate(win)]
                         for c in range(chains)])                     # (chains, K, S)
        series = [D[F], D[Dn], np.transpose(wser, (0, 2, 1))]
    else:
        o = names[0]
        r = select_mediators_fdr(pooled, o, alpha=alpha)
        tab[f"q_{o}"] = r["qval"]
        tab["label"] = np.where(r["selected"], "mediator", "none")
        tab["class_post"], tab["margin"] = tab[f"pip_{o}"], np.nan
        series = [D[o]]
    tab["ess_min"], tab["rhat_max"] = _diag(series)
    meta = {"prior": prior, "chains": chains, "n_iter": n_iter, "burn": burn, "alpha": alpha,
            "endpoints": names, "draws_per_chain": int(S), "seed": seed}
    return ScreenResult(table=tab, meta=meta)
