"""Export the fine-tuned DistilBERT to ONNX, quantise it to INT8, and measure the trade-off.

Dynamic quantisation: weights of Linear/MatMul layers are converted to INT8
offline; activations are quantised on the fly at runtime, so no calibration
data is needed. Gains: ~4x smaller weights and faster CPU matmuls, at a small
accuracy cost - which we measure on the same test set.

    python -m callshield.text.export_onnx --model-dir models/distilbert_mix
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callshield.common import MODELS, binary_metrics, hardware_info, save_results
from callshield.text.data import ALL_SOURCES, load_splits, strip_speaker_tags


class _LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model_dir: Path, onnx_path: Path, max_len: int):
    # eager attention exports cleanly to ONNX (fused SDPA kernels may not)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    dummy = tok(["hello this is your bank calling about your account"] * 2, return_tensors="pt",
                padding="max_length", max_length=64, truncation=True)
    args = (dummy["input_ids"], dummy["attention_mask"])
    common = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17)
    try:  # classic TorchScript exporter
        torch.onnx.export(_LogitsOnly(model), args, onnx_path, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch"}}, **common)
    except Exception as e:  # newer PyTorch: dynamo-based exporter
        print("TorchScript export unavailable, using dynamo exporter:", type(e).__name__)
        batch, seq = torch.export.Dim("batch", min=1, max=256), torch.export.Dim("seq", min=2, max=max_len)
        torch.onnx.export(_LogitsOnly(model), args, onnx_path, dynamo=True, external_data=False,
                          dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}), **common)


def session(path: Path, threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_scores(sess, tok, texts, max_len, bs=32) -> np.ndarray:
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="np")
        logits = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        out.append(e[:, 1] / e.sum(1))
    return np.concatenate(out)


def latency_ms(fn, texts, warmup=10, n=200) -> dict:
    for t in texts[:warmup]:
        fn(t)
    ts = []
    for t in (texts * (n // max(len(texts), 1) + 1))[:n]:
        t0 = time.perf_counter()
        fn(t)
        ts.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS / "distilbert_mix"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk-words", type=int, default=120, help="transcript chunk size for the latency test")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    max_len = json.loads((model_dir / "callshield.json").read_text())["max_len"]
    fp32, int8 = model_dir / "model.onnx", model_dir / "model.int8.onnx"
    export(model_dir, fp32, max_len)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    tok = AutoTokenizer.from_pretrained(model_dir)
    torch.set_num_threads(args.threads)
    pt = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sessions = {"onnx_fp32": session(fp32, args.threads), "onnx_int8": session(int8, args.threads)}

    def pt_one(t):
        with torch.no_grad():
            pt(**tok([t], truncation=True, max_length=max_len, return_tensors="pt"))

    def ort_one(sess):
        def f(t):
            enc = tok([t], truncation=True, max_length=max_len, return_tensors="np")
            sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})
        return f

    results = {"sizes_mb": {
        "pytorch_fp32": round(sum(p.stat().st_size for p in model_dir.glob("*.safetensors")) / 1e6, 1),
        "onnx_fp32": round(fp32.stat().st_size / 1e6, 1),
        "onnx_int8": round(int8.stat().st_size / 1e6, 1),
    }}

    # accuracy: same test sets for every variant
    acc = {}
    for name in ALL_SOURCES:
        te = load_splits(name)
        te = te[te.split == "test"]
        texts = te.text.map(strip_speaker_tags).tolist()
        for variant, sess in sessions.items():
            acc[f"{variant}|test={name}"] = binary_metrics(te.label, onnx_scores(sess, tok, texts, max_len))
    results["accuracy"] = acc

    # latency: single transcript chunk (what the live service scores every few seconds)
    te = load_splits("singleagent")
    chunks = [" ".join(strip_speaker_tags(t).split()[: args.chunk_words]) for t in te[te.split == "test"].text.head(100)]
    results["latency_single_chunk"] = {
        "pytorch_fp32": latency_ms(pt_one, chunks),
        "onnx_fp32": latency_ms(ort_one(sessions["onnx_fp32"]), chunks),
        "onnx_int8": latency_ms(ort_one(sessions["onnx_int8"]), chunks),
    }
    results["chunk_words"] = args.chunk_words
    results["threads"] = args.threads
    results["hardware"] = hardware_info()
    save_results("stage2_onnx", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
