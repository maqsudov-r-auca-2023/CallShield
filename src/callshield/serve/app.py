"""CallShield scoring API.

POST /v1/calls/score       Stage 1 - score a call from the caller SIM's recent behaviour
POST /v1/calls/complete    record a finished call (updates caller AND callee history)
POST /v1/events            record SMS / data-session activity of a SIM
POST /v1/transcript/score  Stage 2 - score a transcript chunk while the call is live
GET  /health

SIM history lives in Redis (one sorted set per SIM, scored by timestamp, trimmed to
24 h) so every API worker sees the same state. Without REDIS_URL it falls back to
in-process memory (tests / local dev). Feature code is shared with training
(callshield.cdr.features) so online and offline features are identical.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from callshield.cdr.features import WINDOW_S, Event, features_from_history

SERVING = Path(os.environ.get("CALLSHIELD_ROOT", Path(__file__).resolve().parents[3])) / "models" / "serving"


class HistoryStore:
    def __init__(self, url: str | None):
        self.r = None
        if url:
            import redis

            self.r = redis.Redis.from_url(url, decode_responses=True)
        self.mem: dict[str, list[Event]] = {}

    def get(self, sim: str, now: float) -> list[Event]:
        if self.r is None:
            return [e for e in self.mem.get(sim, []) if e.ts >= now - WINDOW_S]
        raw = self.r.zrangebyscore(f"h:{sim}", now - WINDOW_S, "+inf")
        return [Event(**json.loads(x)) for x in raw]

    def add(self, sim: str, ev: Event) -> None:
        if self.r is None:
            self.mem.setdefault(sim, []).append(ev)
            return
        key = f"h:{sim}"
        member = json.dumps({"ts": ev.ts, "kind": ev.kind, "peer": ev.peer, "duration": ev.duration,
                             "cell": ev.cell, "n": time.perf_counter_ns()})  # n keeps members unique
        p = self.r.pipeline()
        p.zadd(key, {member: ev.ts})
        p.zremrangebyscore(key, "-inf", ev.ts - WINDOW_S)  # keep only the 24 h window
        p.expire(key, WINDOW_S * 2)
        p.execute()


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


meta_cfg = json.loads((SERVING / "stage1.json").read_text())
FEATS = meta_cfg["features"]
meta_sess = _session(SERVING / "stage1_lgbm.onnx")

text_sess = tok = text_cfg = None
if (SERVING / "stage2_distilbert.int8.onnx").exists():
    from tokenizers import Tokenizer

    text_cfg = json.loads((SERVING / "stage2.json").read_text())
    text_sess = _session(SERVING / "stage2_distilbert.int8.onnx")
    tok = Tokenizer.from_file(str(SERVING / "tokenizer.json"))
    tok.enable_truncation(text_cfg["max_len"])
    tok.no_padding()

store = HistoryStore(os.environ.get("REDIS_URL"))
app = FastAPI(title="CallShield", version="1.0.0")


class CallIn(BaseModel):
    caller: str
    callee: str
    ts: float = Field(default_factory=time.time)
    cell: str = ""


class CallDone(CallIn):
    duration: float
    callee_cell: str = ""


class SimEvent(BaseModel):
    sim: str
    kind: Literal["sms", "data"]
    ts: float = Field(default_factory=time.time)
    cell: str = ""


class TranscriptIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "redis": store.r is not None, "stage2": text_sess is not None}


@app.post("/v1/calls/score")
def score_call(c: CallIn):
    t0 = time.perf_counter()
    feats = features_from_history(store.get(c.caller, c.ts), c.ts)
    x = np.array([[feats[f] for f in FEATS]], dtype=np.float32)
    p = float(meta_sess.run(None, {"features": x})[1][0, 1])  # [labels, probabilities]
    return {
        "fraud_probability": round(p, 4),
        "action": "warn" if p >= meta_cfg["threshold"] else "allow",
        "features": feats,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


@app.post("/v1/calls/complete")
def complete_call(c: CallDone):
    store.add(c.caller, Event(c.ts, "out", c.callee, c.duration, c.cell))
    store.add(c.callee, Event(c.ts, "in", c.caller, c.duration, c.callee_cell))
    return {"stored": True}


@app.post("/v1/events")
def add_event(e: SimEvent):
    store.add(e.sim, Event(e.ts, e.kind, "", 0.0, e.cell))
    return {"stored": True}


@app.post("/v1/transcript/score")
def score_transcript(t: TranscriptIn):
    if text_sess is None:
        raise HTTPException(503, "stage 2 model not deployed")
    t0 = time.perf_counter()
    enc = tok.encode(t.text)
    feed = {"input_ids": np.array([enc.ids], dtype=np.int64), "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    logits = text_sess.run(None, feed)[0][0]
    e = np.exp(logits - logits.max())
    p = float(e[1] / e.sum())
    return {
        "scam_probability": round(p, 4),
        "action": "warn" if p >= text_cfg["threshold"] else "allow",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }
