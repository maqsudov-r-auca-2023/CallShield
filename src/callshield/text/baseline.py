"""Stage 2 baseline: TF-IDF + logistic regression on call transcripts.

A transformer only earns its cost if it beats this. Also evaluates the
cross-dataset setting (train on one corpus, test on the other).

    python -m callshield.text.baseline
"""
from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from callshield.common import mlflow_or_stub, MODELS, binary_metrics, mlflow_run, save_results, seed_everything
from callshield.text.data import ALL_SOURCES, CORPORA, drop_overlap, load_mix, load_splits, strip_speaker_tags


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    )


def main():
    seed_everything()
    data = {n: load_splits(n) for n in ALL_SOURCES}
    data["mix"] = load_mix()
    for d in data.values():
        d["text"] = d["text"].map(strip_speaker_tags)

    results = {}
    with mlflow_run("stage2-text", "tfidf-logreg"):
        mlflow = mlflow_or_stub()

        for train_name in [*CORPORA, "mix"]:
            tr = data[train_name][data[train_name].split == "train"]
            model = make_model().fit(tr.text, tr.label)
            joblib.dump(model, MODELS / f"tfidf_{train_name}.joblib")
            for test_name in ALL_SOURCES:
                te = drop_overlap(data[test_name][data[test_name].split == "test"], tr)
                m = binary_metrics(te.label, model.predict_proba(te.text)[:, 1])
                key = f"train={train_name}|test={test_name}"
                results[key] = m
                mlflow.log_metric(f"{train_name}_on_{test_name}_f1", m["f1"])
                print(f"{key:45s} n={m['n']:5d} F1={m['f1']:.3f} recall={m['recall']:.3f} PR-AUC={m.get('pr_auc', float('nan')):.3f}")
    save_results("stage2_baseline", {"model": "tfidf+logreg", "results": results})


if __name__ == "__main__":
    main()
