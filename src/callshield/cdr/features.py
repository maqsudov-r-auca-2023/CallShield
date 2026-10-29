"""Caller-behaviour features for Stage 1 (call metadata, scored when a call is placed).

ONE function - `features_from_history` - computes the features from a SIM's recent
activity. It is used both offline (replaying the CDR log in time order to build
the training set) and online (the API reads the SIM's recent events from Redis).
Same code in both places = no train/serve skew.

Only events that *finished before* the current call are used - no look-ahead leakage.

Why these features (SIM-box fraud): a SIM-box is a rack of SIM cards that
re-originates international calls as cheap local calls. Its SIMs place many short
calls to many different numbers, rarely receive calls, rarely use SMS or mobile
data, and often never move between cell towers - unlike a real subscriber.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

WINDOW_S = 24 * 3600  # history kept per SIM
SHORT_CALL_S = 30
KINDS = ("out", "in", "sms", "data")


@dataclass(slots=True)
class Event:
    ts: float  # unix seconds, event start
    kind: str  # out | in | sms | data
    peer: str = ""  # other party (callee for 'out', caller for 'in')
    duration: float = 0.0  # seconds
    cell: str = ""  # serving cell id


FEATURE_NAMES = [
    "out_calls_1h", "out_calls_24h", "distinct_callees_24h", "distinct_ratio_24h",
    "mean_duration_24h", "std_duration_24h", "short_call_share_24h",
    "mean_gap_s_24h", "min_gap_s_24h",
    "in_calls_24h", "sms_24h", "data_sessions_24h", "distinct_cells_24h",
    "hour_of_day", "secs_since_last_event",
]


def features_from_history(history: list[Event], now: float) -> dict:
    """Features for a call placed at `now`, from the SIM's events in the previous 24 h."""
    h = [e for e in history if now - WINDOW_S <= e.ts and e.ts + e.duration <= now]
    out = [e for e in h if e.kind == "out"]
    n = len(out)
    f = {k: 0.0 for k in FEATURE_NAMES}
    f["hour_of_day"] = (now % 86400) / 3600.0
    f["secs_since_last_event"] = float(now - max(e.ts for e in h)) if h else float(WINDOW_S)
    f["in_calls_24h"] = sum(e.kind == "in" for e in h)
    f["sms_24h"] = sum(e.kind == "sms" for e in h)
    f["data_sessions_24h"] = sum(e.kind == "data" for e in h)
    f["distinct_cells_24h"] = len({e.cell for e in h if e.cell})  # SIM-boxes often never move
    f["mean_gap_s_24h"] = f["min_gap_s_24h"] = float(WINDOW_S)
    if n:
        ts = np.sort(np.fromiter((e.ts for e in out), float, n))
        dur = np.fromiter((e.duration for e in out), float, n)
        distinct = len({e.peer for e in out})
        f.update(
            out_calls_1h=float((ts >= now - 3600).sum()),
            out_calls_24h=float(n),
            distinct_callees_24h=float(distinct),
            distinct_ratio_24h=distinct / n,  # SIM-boxes rarely call the same number twice
            mean_duration_24h=float(dur.mean()),
            std_duration_24h=float(dur.std()),
            short_call_share_24h=float((dur < SHORT_CALL_S).mean()),
        )
        if n > 1:
            gaps = np.diff(ts)
            f["mean_gap_s_24h"], f["min_gap_s_24h"] = float(gaps.mean()), float(gaps.min())
    return {k: float(v) for k, v in f.items()}


class RollingHistory:
    """In-memory per-SIM 24 h history, used for the offline replay."""

    def __init__(self):
        self._h: dict[str, deque[Event]] = {}

    def get(self, sim: str, now: float) -> list[Event]:
        q = self._h.get(sim)
        if not q:
            return []
        while q and q[0].ts < now - WINDOW_S:
            q.popleft()
        return list(q)

    def add(self, sim: str, ev: Event) -> None:
        self._h.setdefault(sim, deque()).append(ev)
