"""Stage 2: fine-tune DistilBERT to classify call transcripts as scam / legitimate.

A plain PyTorch training loop (no Trainer magic) so every step is explainable:
tokenise -> batches -> forward -> cross-entropy -> AdamW step -> linear LR decay.

    python -m callshield.text.finetune --train-on scamdialogue --epochs 3
    python -m callshield.text.finetune --train-on mix
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from callshield.common import mlflow_or_stub, MODELS, binary_metrics, hardware_info, mlflow_run, save_results, seed_everything
from callshield.text.data import ALL_SOURCES, CORPORA, drop_overlap, load_mix, load_splits, strip_speaker_tags


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(df: pd.DataFrame, tok, max_len: int, bs: int, shuffle: bool):
    idx = np.arange(len(df))
    loader = DataLoader(idx, batch_size=bs, shuffle=shuffle)
    texts, labels = df.text.tolist(), df.label.to_numpy()
    for b in loader:
        b = b.numpy()
        enc = tok([texts[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels[b], dtype=torch.long)
        yield enc


@torch.no_grad()
def predict(model, tok, df, device, max_len, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for enc in batches(df, tok, max_len, bs, shuffle=False):
        enc.pop("labels")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.append(torch.softmax(logits, -1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", default="mix", choices=[*CORPORA, "mix"])
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    seed_everything()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device()
    print("device:", device, hardware_info())

    data = {n: load_splits(n) for n in ALL_SOURCES}
    data["mix"] = load_mix()
    for d in data.values():
        d["text"] = d["text"].map(strip_speaker_tags)
    src = data[args.train_on]
    train, val = src[src.split == "train"], src[src.split == "val"]
    if args.max_train and len(train) > args.max_train:
        train = train.groupby("label", group_keys=False).apply(
            lambda g: g.sample(int(round(args.max_train * len(g) / len(train))), random_state=42)
        )
    print(f"train={len(train)} val={len(val)} pos_rate={train.label.mean():.3f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    # Small LR + warmup: large updates would overwrite pre-trained knowledge
    # (catastrophic forgetting); warmup avoids unstable first steps.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(train) / args.bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    out_dir = MODELS / f"distilbert_{args.train_on}"
    best_f1, history, t0 = -1.0, [], time.time()
    with mlflow_run("stage2-text", f"distilbert-{args.train_on}"):
        mlflow = mlflow_or_stub()

        mlflow.log_params(vars(args))
        step = 0
        for epoch in range(args.epochs):
            model.train()
            for enc in batches(train, tok, args.max_len, args.bs, shuffle=True):
                loss = model(**{k: v.to(device) for k, v in enc.items()}).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    rate = step * args.bs / (time.time() - t0)
                    print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({rate:.1f} samples/s)", flush=True)
                    mlflow.log_metric("train_loss", loss.item(), step=step)
            vm = binary_metrics(val.label, predict(model, tok, val, device, args.max_len))
            history.append({"epoch": epoch, **vm})
            mlflow.log_metric("val_f1", vm["f1"], step=epoch)
            print(f"epoch {epoch} val {vm}", flush=True)
            if vm["f1"] > best_f1:  # keep the best checkpoint by validation F1
                best_f1 = vm["f1"]
                model.save_pretrained(out_dir), tok.save_pretrained(out_dir)

        train_minutes = (time.time() - t0) / 60
        model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
        results = {}
        for name in ALL_SOURCES:
            te = drop_overlap(data[name][data[name].split == "test"], train)
            m = binary_metrics(te.label, predict(model, tok, te, device, args.max_len))
            results[f"test={name}"] = m
            mlflow.log_metric(f"test_{name}_f1", m["f1"])
            print(f"test on {name}: {m}")

    save_results(
        f"stage2_distilbert_{args.train_on}",
        {"args": vars(args), "hardware": hardware_info(), "train_rows": len(train),
         "train_minutes": round(train_minutes, 1), "history": history, "results": results},
    )
    (out_dir / "callshield.json").write_text(json.dumps({"max_len": args.max_len, "train_on": args.train_on}))


if __name__ == "__main__":
    main()
