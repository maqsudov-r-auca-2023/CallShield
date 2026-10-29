"""Load and normalise the scam-call transcript corpora.

Every corpus is converted to one schema:  text | label (0/1) | split | source | lang
Splits are made once, deterministically, and saved to data/processed so every
model (baseline, DistilBERT, ONNX) is evaluated on exactly the same rows.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from callshield.common import PROCESSED, SEED

CORPORA = {
    # 1.6k balanced, LLM-generated English dialogues (8 scam / non-scam call types)
    "scamdialogue": "BothBosu/scam-dialogue",
    # LLM-generated dialogues from a different generation setup (single agent, varied receiver personas)
    "singleagent": "BothBosu/single-agent-scam-conversations",
}
# transcripts of REAL scam calls (scam-baiting recordings) - used only as an
# out-of-distribution test set: does a model trained on synthetic text catch real scams?
REAL_WORLD = {"youtube": "BothBosu/youtube-scam-conversations"}
ALL_SOURCES = {**CORPORA, **REAL_WORLD}

POSITIVE_STRINGS = {"1", "fraud", "scam", "spam", "true", "yes", "fraudulent", "suspicious"}


def _to_label(v) -> int:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return int(str(v).strip().lower() in POSITIVE_STRINGS)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {candidates} in columns {list(df.columns)}")


def normalise_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def text_key(t: str) -> str:
    """Hash of lower-cased, punctuation-free text - used to find near-duplicates."""
    return hashlib.md5(re.sub(r"[^a-z0-9]+", "", t.lower()).encode()).hexdigest()


def _from_hf(repo: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(repo)
    frames = []
    for split_name, split in ds.items():
        df = split.to_pandas()
        df["hf_split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_raw(name: str) -> pd.DataFrame:
    raw = _from_hf(ALL_SOURCES[name])
    text_col = _pick(raw, ["dialogue", "conversation", "text", "transcript", "Dialogue", "content"])
    try:
        labels = raw[_pick(raw, ["label", "labels", "is_fraud", "is_scam"])].map(_to_label)
    except KeyError:
        if name not in REAL_WORLD:
            raise
        print(f"[{name}] no label column -> every transcript is a real scam call (label=1)")
        labels = 1
    out = pd.DataFrame(
        {
            "text": raw[text_col].map(normalise_text),
            "label": labels,
            "hf_split": raw["hf_split"],
            "source": name,
            "lang": raw["language"].astype(str) if "language" in raw.columns else "en",
        }
    )
    if "type" in raw.columns:
        out["subtype"] = raw["type"].astype(str)
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def build_splits(name: str) -> pd.DataFrame:
    """Deduplicate, then make a stratified 80/10/10 train/val/test split.

    Exact/near duplicates are removed *before* splitting; otherwise the same
    conversation can sit in train and test and inflate the score.
    """
    df = load_raw(name)
    n0 = len(df)
    df["key"] = df["text"].map(text_key)
    # a text that appears with both labels is ambiguous -> drop it entirely
    conflicted = df.groupby("key")["label"].nunique()
    df = df[~df["key"].isin(conflicted[conflicted > 1].index)]
    df = df.drop_duplicates("key").reset_index(drop=True)
    print(f"[{name}] {n0} rows -> {len(df)} after dedup / conflict removal")

    if name in REAL_WORLD:  # evaluation-only corpus
        df = df.assign(split="test")
        df.to_parquet(PROCESSED / f"text_{name}.parquet", index=False)
        return df
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    out = pd.concat([train, val, test], ignore_index=True)
    out.to_parquet(PROCESSED / f"text_{name}.parquet", index=False)
    return out


def load_splits(name: str) -> pd.DataFrame:
    path = PROCESSED / f"text_{name}.parquet"
    return pd.read_parquet(path) if path.exists() else build_splits(name)


def drop_overlap(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Remove test conversations that also occur in the training corpus (cross-corpus leakage)."""
    return test[~test["key"].isin(set(train["key"]))]


def load_mix() -> pd.DataFrame:
    """Union of both corpora, keeping each corpus's own split assignment."""
    return pd.concat([load_splits(n) for n in CORPORA], ignore_index=True)


def strip_speaker_tags(t: str) -> str:
    """Remove role tags such as 'Suspect:' / 'Innocent:' / 'Caller:'.

    Some synthetic corpora name the speaker 'Suspect' only in scam calls, which
    would let a model 'detect fraud' by reading the tag. We strip them so the
    model has to learn from what is actually said.
    """
    return re.sub(r"\b(suspect|innocent|caller|receiver|agent|customer|speaker ?\d?)\s*:\s*", " ", t, flags=re.I)


if __name__ == "__main__":
    for n in ALL_SOURCES:
        d = build_splits(n)
        print(d.groupby(["split"])["label"].agg(["count", "mean"]))
        print(d["lang"].value_counts().head(10))
