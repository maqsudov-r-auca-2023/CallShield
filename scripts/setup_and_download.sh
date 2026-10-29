#!/usr/bin/env bash
# Step 1: create the Python environment, install packages, download datasets,
# and write a data inventory to reports/data_inventory.txt.
#   cd ~/Documents/CallShield && bash scripts/setup_and_download.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw reports logs

PY=""
for c in python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Need Python >= 3.10:  brew install python@3.11"; exit 1; fi
echo "Using $($PY --version) at $(command -v $PY)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch
pip install -q -r requirements.txt
python -c "import lightgbm" 2>/dev/null || { echo "LightGBM needs libomp: brew install libomp"; exit 1; }
echo "[ok] packages installed"

# --- FraudZen synthetic CDRs (INRIA, open access) - per-file download with retry/resume ---
rm -f data/raw/fraudzen.zip
python scripts/download_fraudzen.py || echo "[warn] FraudZen download incomplete - rerun this script to resume"

# --- transcript corpora (Hugging Face) + inventory ---
PYTHONPATH=src python - <<'EOF' > reports/data_inventory.txt 2>&1
import os, glob, pandas as pd
from datasets import load_dataset
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

for repo in ["BothBosu/scam-dialogue", "BothBosu/single-agent-scam-conversations", "BothBosu/youtube-scam-conversations"]:
    ds = load_dataset(repo)
    print("=" * 80, "\n", repo, ds)
    df = ds[list(ds)[0]].to_pandas()
    print(df.dtypes)
    for c in df.columns:
        if df[c].nunique() < 30: print(c, df[c].value_counts().to_dict())
    print(df.head(2).to_string()[:1500])

print("=" * 80, "\nFRAUDZEN FILES")
files = [f for f in glob.glob("data/raw/fraudzen/**/*", recursive=True) if os.path.isfile(f)]
for f in sorted(files)[:200]:
    print(f"{os.path.getsize(f)/1e6:9.2f} MB  {f}")
print("total files:", len(files))
seen = set()
for f in sorted(files):
    ext = os.path.splitext(f)[1].lower()
    if ext in (".csv", ".txt", ".tsv", ".json", ".md", ".parquet") and ext + os.path.basename(os.path.dirname(f)) not in seen:
        seen.add(ext + os.path.basename(os.path.dirname(f)))
        print("-" * 80, "\n", f)
        try:
            if ext == ".parquet":
                print(pd.read_parquet(f).head(5).to_string())
            else:
                with open(f, errors="replace") as fh: print("".join(fh.readlines()[:12])[:3000])
        except Exception as e: print("ERR", e)
    if len(seen) > 25: break
EOF
echo "[ok] inventory written to reports/data_inventory.txt"
echo "DONE - tell Claude step 1 is finished."
