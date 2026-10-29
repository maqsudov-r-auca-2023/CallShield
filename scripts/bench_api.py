"""Benchmark the running API end-to-end over HTTP (localhost).

Starts uvicorn, seeds SIM history by replaying real FraudZen traffic through the
API, then measures:
  - latency: sequential requests, p50 / p95 / p99 (includes HTTP + JSON + Redis/memory + ONNX)
  - throughput: concurrent requests/s with N workers

    python scripts/bench_api.py --workers 4            # in-memory history
    REDIS_URL=redis://localhost:6379/0 python scripts/bench_api.py --workers 4
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from callshield.cdr.data import CDR_ROOT, KIND, _ts  # noqa: E402
from callshield.common import PROCESSED, hardware_info, save_results  # noqa: E402


def load_splits_safe(n=300, words=120):
    """Transcript chunks (first 120 words of test conversations), as the live service would see them."""
    df = pd.read_parquet(PROCESSED / "text_singleagent.parquet")
    df = df[df.split == "test"].head(n)
    return [" ".join(t.split()[:words]) for t in df.text]

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--port", type=int, default=8077)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--seed-events", type=int, default=20000)
args = ap.parse_args()
URL = f"http://127.0.0.1:{args.port}"

env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CALLSHIELD_ROOT": str(ROOT), "ORT_THREADS": "1"}
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "callshield.serve.app:app", "--port", str(args.port),
                           "--workers", str(args.workers), "--log-level", "warning"], env=env)


def pct(xs):
    return {"p50_ms": round(float(np.percentile(xs, 50)), 2), "p95_ms": round(float(np.percentile(xs, 95)), 2),
            "p99_ms": round(float(np.percentile(xs, 99)), 2)}


async def throughput(path, payloads, conc):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(base_url=URL, timeout=30, limits=httpx.Limits(max_connections=conc)) as c:
        async def one(p):
            async with sem:
                r = await c.post(path, json=p)
                r.raise_for_status()
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in payloads))
        return round(len(payloads) / (time.perf_counter() - t0), 1)


try:
    with httpx.Client(base_url=URL, timeout=30) as c:
        for _ in range(100):
            try:
                health = c.get("/health").json()
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        print("health:", health)

        # seed history with real traffic (first events of a test scenario)
        folder = CDR_ROOT / "advanced_social" / "social_12%_200sims"
        cols = ["date", "time", "type", "src", "a", "b", "src_cell", "dst", "c", "d", "dst_cell", "duration", "size"]
        cdr = pd.read_csv(folder / "Op_1_CDRTrace.txt", sep=r"\s+", skiprows=1, names=cols, dtype=str)
        cdr = cdr[cdr.type.isin(KIND)]
        calls = []
        for r in cdr.head(args.seed_events).itertuples(index=False):
            ts = _ts(r.date, r.time)
            if KIND[r.type] == "out":
                body = {"caller": r.src, "callee": r.dst, "ts": ts, "duration": float(r.duration),
                        "cell": r.src_cell, "callee_cell": r.dst_cell}
                c.post("/v1/calls/complete", json=body)
                calls.append({"caller": r.src, "callee": r.dst, "ts": ts + 1})
            else:
                c.post("/v1/events", json={"sim": r.src, "kind": KIND[r.type], "ts": ts, "cell": r.src_cell})
        rng = np.random.default_rng(0)
        payloads = [calls[i] for i in rng.integers(0, len(calls), args.n)]

        for p in payloads[:200]:  # warm-up
            c.post("/v1/calls/score", json=p)
        lat = []
        for p in payloads:
            t0 = time.perf_counter()
            c.post("/v1/calls/score", json=p).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        res = {"stage1_latency": pct(lat)}
        res["stage1_rps"] = asyncio.run(throughput("/v1/calls/score", payloads, args.concurrency))
        print("stage1:", res)

        if health.get("stage2"):
            texts = load_splits_safe()
            chunks = [{"text": t} for t in texts]
            for p in chunks[:20]:
                c.post("/v1/transcript/score", json=p)
            lat = []
            for p in chunks:
                t0 = time.perf_counter()
                c.post("/v1/transcript/score", json=p).raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)
            res["stage2_latency"] = pct(lat)
            res["stage2_rps"] = asyncio.run(throughput("/v1/transcript/score", chunks * 3, min(args.concurrency, 16)))
            res["stage2_chunk_words"] = 120
            print("stage2:", {k: v for k, v in res.items() if k.startswith("stage2")})

    res.update(workers=args.workers, concurrency=args.concurrency, history_backend="redis" if os.environ.get("REDIS_URL") else "memory",
               hardware=hardware_info())
    save_results(f"api_bench_{res['history_backend']}", res)
finally:
    server.terminate()
    server.wait()
