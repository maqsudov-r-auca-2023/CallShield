"""Turn FraudZen CDR traces into a per-call training table for Stage 1.

FraudZen (INRIA, CC BY 4.0) simulates one day of a mobile network per scenario,
with SIM-box fraudsters using different evasion strategies. For every outgoing
local call (CDR type MOC) we replay the network in time order and compute the
caller SIM's features from its *previous* activity, then label the call as fraud
if the caller SIM is in fraudulentUsers.txt.

    python -m callshield.cdr.data
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from callshield.cdr.features import FEATURE_NAMES, Event, RollingHistory, features_from_history
from callshield.common import PROCESSED, RAW

CDR_ROOT = RAW / "fraudzen" / "Fraudzen-dataset" / "Fraudzen-dataset" / "FraudZen_Cdrs"

# Strategies whose fraud calls actually appear in the operator CDRs. In the
# 'traffic' and 'all_advanced' families the SIM-box rejects or cancels almost
# all bypass calls (0-6 fraud calls per scenario), so there is nothing to
# detect per call - they are reported separately, not trained on.
STRATEGIES = {"all_naive": "naive", "advanced_mobility": "mobility", "advanced_social": "social"}
SCEN_RE = re.compile(r"_(\d+)%_(\d+)sims$")
KIND = {"MOC": "out", "MO-SMS": "sms", "DATA": "data"}
LEGIT_WEIGHT = {"train": 10 / 6, "val": 10 / 2, "test": 10 / 2}


def _ts(date: str, time: str) -> float:
    return datetime.strptime(f"{date} {time}", "%d-%m-%Y %H:%M:%S").timestamp()


def replay_scenario(folder: Path) -> pd.DataFrame:
    fraud = set(folder.joinpath("fraudulentUsers.txt").read_text().split())
    cols = ["date", "time", "type", "src", "src_imei", "src_cell0", "src_cell", "dst",
            "dst_imei", "dst_cell0", "dst_cell", "duration", "size"]
    cdr = pd.read_csv(folder / "Op_1_CDRTrace.txt", sep=r"\s+", skiprows=1, names=cols,
                      dtype=str, engine="c")
    cdr = cdr[cdr["type"].isin(KIND)]
    cdr["ts"] = [_ts(d, t) for d, t in zip(cdr["date"], cdr["time"])]
    cdr["duration"] = pd.to_numeric(cdr["duration"], errors="coerce").fillna(0.0)
    cdr = cdr.sort_values("ts", kind="stable")

    hist, rows = RollingHistory(), []
    for r in cdr.itertuples(index=False):
        kind = KIND[r.type]
        if kind == "out":
            # score the call with what was known when it was placed ...
            f = features_from_history(hist.get(r.src, r.ts), r.ts)
            f.update(ts=r.ts, caller=r.src, callee=r.dst, label=int(r.src in fraud))
            rows.append(f)
            # ... then record it for both parties
            hist.add(r.src, Event(r.ts, "out", r.dst, r.duration, r.src_cell))
            hist.add(r.dst, Event(r.ts, "in", r.src, r.duration, r.dst_cell))
        else:
            hist.add(r.src, Event(r.ts, kind, r.dst, 0.0, r.src_cell))
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    frames = []
    for fam, strategy in STRATEGIES.items():
        for folder in sorted((CDR_ROOT / fam).iterdir()):
            m = SCEN_RE.search(folder.name)
            if not m:
                continue
            df = replay_scenario(folder)
            df["scenario"], df["strategy"] = folder.name, strategy
            df["fraud_pct"], df["n_sims"] = int(m.group(1)), int(m.group(2))
            print(f"{folder.name:28s} calls={len(df):6d} fraud={df.label.sum():5d}", flush=True)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Fraud SIMs: split by SCENARIO (each scenario is a separate simulated day).
    scen_split = out["n_sims"].map(lambda s: "test" if s == 200 else "val" if s == 150 else "train")
    # Legitimate subscribers: FraudZen reuses the SAME legitimate population in every
    # scenario, so a scenario split alone would put identical legit calls in train and
    # test. We therefore also split legit callers by phone number (60/20/20): test
    # users are never seen in training. Weights (1 / bucket share) restore the true
    # class balance when computing metrics.
    bucket = out["caller"].map(lambda c: int(hashlib.md5(c.encode()).hexdigest(), 16) % 10)
    user_split = bucket.map(lambda b: "train" if b < 6 else "val" if b < 8 else "test")
    keep = (out["label"] == 1) | (user_split == scen_split)
    out = out[keep].assign(split=scen_split[keep])
    out["weight"] = np.where(out["label"] == 1, 1.0, out["split"].map(LEGIT_WEIGHT))
    out = out.reset_index(drop=True)
    out.to_parquet(PROCESSED / "cdr_calls.parquet", index=False)
    return out


def load() -> pd.DataFrame:
    p = PROCESSED / "cdr_calls.parquet"
    return pd.read_parquet(p) if p.exists() else build()


if __name__ == "__main__":
    d = build()
    print(d.groupby(["split", "strategy"])["label"].agg(["count", "sum", "mean"]))
    w = d.groupby("split").apply(lambda g: np.average(g.label, weights=g.weight))
    print("weighted fraud rate per split:", w.round(4).to_dict())
    print(d[FEATURE_NAMES + ["label"]].groupby("label").mean().T)
