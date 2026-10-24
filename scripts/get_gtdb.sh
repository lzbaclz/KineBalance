#!/usr/bin/env bash
# Download the GTDB R07-RS207 bacterial reference tree and its taxonomy (the release the
# curated cohorts were classified against) into data/gtdb/, and verify their SHA-256.
# Used by src/preprocess/phylogeny.py for the reference-phylogeny (PhILR) balances.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/gtdb"
BASE="https://data.gtdb.ecogenomic.org/releases/release207/207.0"
mkdir -p "$DEST"
fetch() {   # file sha256
  local f=$1 sum=$2
  if [ ! -s "$DEST/$f" ]; then
    echo "[get_gtdb] $f"
    curl -fsSL --max-time 900 "$BASE/$f" -o "$DEST/$f"
  else
    echo "[get_gtdb] have $f (skip)"
  fi
  echo "$sum  $DEST/$f" | shasum -a 256 -c -
}
fetch bac120_r207.tree          767241cc7f52ea3b487c0b755f007e57c7bb438c81f9a39aab796024496648b8
fetch bac120_taxonomy_r207.tsv  08df1d14239cd4b5bc04e9af522e814a5f0065d91dd0b03d01aa79e1e85324a5
echo "[get_gtdb] wrote $DEST"
