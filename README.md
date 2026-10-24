# KineBalance

KineBalance is a Bayesian screening method for compositional microbiome mediation with
multiple endpoints. Given an exposure, a microbial count table and one or two endpoints, it
finds the microbial features (CLR coordinates or ILR balances) that carry the exposure's
association with each endpoint, with Bayesian FDR control, and labels each discovery as
endpoint-specific or shared.

## Installation

Requires Python >= 3.10.

```bash
git clone https://github.com/lzbaclz/KineBalance.git
cd KineBalance
pip install -e .
bash scripts/get_gtdb.sh    # optional, only needed for basis="gtdb"
```

## Quick start

```python
import kinebalance as kb

# counts: samples x taxa count DataFrame; X: exposure; y_dca, y_prop: endpoints; age: covariate
B, info = kb.balances(counts, basis="ward")   # or "clr", "gtdb", "taxonomy-ladder", "taxonomy-balanced"
res = kb.screen(X, B, {"DCA": y_dca, "propionate": y_prop}, C=age, prior="decoupled", alpha=0.10)
print(res.summary())
```

Command line (`kinebalance --help` lists all options):

```bash
kinebalance --counts counts.tsv --samples samples.csv --exposure ibd \
  --endpoints dca propionate --covariates age --basis ward --prior both --out results/kb
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Citation

If you use KineBalance, please cite our IEEE BIBM 2026 paper (see `CITATION.cff`).

## License

MIT
