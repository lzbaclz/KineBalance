"""Command-line entry point: ``kinebalance --counts C.tsv --samples S.tsv ...``.

Reads a samples-by-taxa count table and a samples table (exposure, endpoints,
covariates), builds log-ratio coordinates, runs the screen under one or both priors and
writes one CSV per prior (plus the balance partition for ILR bases).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def main(argv=None) -> int:
    from kinebalance.api import BASES, balances, screen
    ap = argparse.ArgumentParser(prog="kinebalance", description=__doc__.splitlines()[0])
    ap.add_argument("--counts", required=True, help="samples x taxa counts (first column: sample id)")
    ap.add_argument("--samples", required=True, help="samples table (first column: sample id)")
    ap.add_argument("--exposure", required=True, help="exposure column (numeric, e.g. 0/1)")
    ap.add_argument("--endpoints", required=True, nargs="+", help="one or two endpoint columns")
    ap.add_argument("--covariates", nargs="*", default=[], help="pre-exposure covariate columns")
    ap.add_argument("--log1p", action="store_true", help="transform the endpoints by log(1 + x)")
    ap.add_argument("--basis", choices=BASES, default="ward")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--prevalence", type=float, default=0.20)
    ap.add_argument("--prior", choices=("coupled", "decoupled", "both"), default="both")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=24000)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>_<prior>.csv)")
    a = ap.parse_args(argv)
    counts, samples = _read(a.counts), _read(a.samples)
    ids = [s for s in counts.index if s in samples.index]
    need = [a.exposure] + a.endpoints + a.covariates
    ok = samples.loc[ids, need].notna().all(axis=1)
    ids = [s for s, k in zip(ids, ok) if k]
    if len(ids) < 10:
        sys.exit(f"only {len(ids)} samples with complete exposure/endpoints/covariates")
    B, info = balances(counts.loc[ids], basis=a.basis, prevalence_min=a.prevalence, top_n=a.top_n)
    X = pd.to_numeric(samples.loc[ids, a.exposure]).to_numpy(float)
    Y = {e: pd.to_numeric(samples.loc[ids, e]).to_numpy(float) for e in a.endpoints}
    if a.log1p:
        Y = {e: np.log1p(v) for e, v in Y.items()}
    C = None
    if a.covariates:
        C = pd.get_dummies(samples.loc[ids, a.covariates], drop_first=True).astype(float).to_numpy()
        sd = C.std(axis=0)
        C = (C - C.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    priors = ("decoupled", "coupled") if a.prior == "both" else (a.prior,)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if "partition" in info:
        info["partition"].to_csv(f"{a.out}_partition.csv", index=False)
    for p in priors:
        res = screen(X, B, Y, C=C, prior=p, chains=a.chains, n_iter=a.n_iter, burn=a.burn,
                     alpha=a.alpha)
        res.table.to_csv(f"{a.out}_{p}.csv", index=False)
        print(res.summary())
    print(f"[kinebalance] n={len(ids)}, K={B.shape[1]} ({a.basis}); wrote {a.out}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
