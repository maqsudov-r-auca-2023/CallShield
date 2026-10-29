"""Evaluate the DEPLOYED Stage 2 model (INT8 ONNX) the way it runs on a live call:
the transcript is scored in overlapping windows as the call progresses, and the
call is flagged as soon as any window crosses the threshold.

Reports, per test set:
  - call-level precision / recall / F1 and false-positive rate on legitimate calls
  - how far into the call the first warning comes (words spoken so far)
compared with scoring only the first window (what a single forward pass sees).

    python -m callshield.text.eval_windows --model-dir models/distilbert_mix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from callshield.common import MODELS, binary_metrics, save_results
from callshield.text.data import ALL_SOURCES, load_splits, strip_speaker_tags
from callshield.text.export_onnx import onnx_scores, session


def windows(text: str, size: int, stride: int) -> list[tuple[int, str]]:
    """(words spoken by the end of the window, window text)."""
    w = text.split()
    starts = range(0, max(len(w) - size, 0) + 1, stride)
    out = [(min(s + size, len(w)), " ".join(w[s:s + size])) for s in starts]
    if out[-1][0] < len(w):  # tail
        out.append((len(w), " ".join(w[-size:])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / "distilbert_mix"))
    ap.add_argument("--size", type=int, default=120, help="window length in words")
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    md = Path(args.model_dir)
    max_len = json.loads((md / "callshield.json").read_text())["max_len"]
    tok, sess = AutoTokenizer.from_pretrained(md), session(md / "model.int8.onnx")

    results = {"window_words": args.size, "stride_words": args.stride, "threshold": args.threshold, "model": md.name}
    for name in ALL_SOURCES:
        te = load_splits(name)
        te = te[te.split == "test"]
        texts, y = [strip_speaker_tags(t) for t in te.text], te.label.to_numpy()
        first, best, words_to_flag = [], [], []
        for t in texts:
            ws = windows(t, args.size, args.stride)
            s = onnx_scores(sess, tok, [w for _, w in ws], max_len)
            first.append(s[0])
            best.append(s.max())
            hit = np.flatnonzero(s >= args.threshold)
            words_to_flag.append(ws[hit[0]][0] if len(hit) else None)
        first, best = np.array(first), np.array(best)
        r = {}
        for mode, sc in (("first_window_only", first), ("sliding_window_max", best)):
            m = binary_metrics(y, sc, args.threshold)
            neg = y == 0
            m["false_positive_rate"] = round(float((sc[neg] >= args.threshold).mean()), 4) if neg.any() else None
            r[mode] = m
        flagged_scams = [w for w, lab in zip(words_to_flag, y) if lab == 1 and w is not None]
        r["median_words_until_warning_on_scams"] = float(np.median(flagged_scams)) if flagged_scams else None
        r["median_transcript_words"] = float(np.median([len(t.split()) for t in texts]))
        results[f"test={name}"] = r
        print(name, {k: (v.get("recall"), v.get("f1"), v.get("false_positive_rate")) if isinstance(v, dict) else v for k, v in r.items()})
    save_results("stage2_windows", results)


if __name__ == "__main__":
    main()
