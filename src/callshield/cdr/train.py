"""Stage 1: train and evaluate the call-metadata fraud model.

Models (same scenario-level split for all):
  1. rule baseline  - classic SIM-box heuristic an operator might hand-write
  2. logistic regression on the same features
  3. LightGBM (gradient-boosted trees)
Plus a leave-one-strategy-out test: train on two fraud strategies, test on the
third, to see whether the model generalises to fraud behaviour it never saw.

    python -m callshield.cdr.train
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from callshield.cdr.data import load
from callshield.cdr.features import FEATURE_NAMES
from callshield.common import mlflow_or_stub, MODELS, binary_metrics, mlflow_run, save_results, seed_everything

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, random_state=42,
)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Classic operator rule for SIM-box detection: call volume of SIMs that use no
    mobile data and send no SMS. Its cut-off is tuned on validation like the models'."""
    silent = (df.data_sessions_24h == 0) & (df.sms_24h == 0)
    return np.where(silent, df.out_calls_24h + df.out_calls_1h / 100, 0.0)


def best_f1_threshold(y, p, w=None) -> float:
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def recall_at_fpr(y, p, w=None, fpr_target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    return float(tpr[fpr <= fpr_target].max())


def fit_lgb(train: pd.DataFrame, val: pd.DataFrame, feats: list[str] = FEATURE_NAMES) -> lgb.LGBMClassifier:
    # scale_pos_weight up-weights the rare fraud class so the trees don't just predict "legit"
    pos_w = train.weight[train.label == 0].sum() / max(train.weight[train.label == 1].sum(), 1)
    m = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=pos_w)
    m.fit(train[feats], train.label, sample_weight=train.weight,
          eval_set=[(val[feats], val.label)], eval_sample_weight=[val.weight],
          eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def evaluate(name, val: pd.DataFrame, p_val, test: pd.DataFrame, p_test) -> dict:
    """Threshold chosen on VALIDATION (max F1), then applied unchanged to TEST."""
    thr = best_f1_threshold(val.label, p_val, val.weight) if len(np.unique(p_val)) > 2 else 0.5
    m = binary_metrics(test.label, p_test, threshold=thr, sample_weight=test.weight)
    m["recall_at_1pct_fpr"] = round(recall_at_fpr(test.label, p_test, test.weight), 4)
    print(f"{name:10s} PR-AUC={m.get('pr_auc')} ROC-AUC={m.get('roc_auc')} F1={m['f1']} "
          f"P={m['precision']} R={m['recall']} R@1%FPR={m['recall_at_1pct_fpr']}")
    return m


def main():
    seed_everything()
    df = load()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)}  val={len(va)}  test={len(te)} "
          f"(weighted test fraud rate {np.average(te.label, weights=te.weight):.2%})")

    results = {"data": {
        "calls_total": int(len(df)), "fraud_calls": int(df.label.sum()),
        "scenarios": int(df.scenario.nunique()) if "scenario" in df else None,
        "split_rows": {s: int((df.split == s).sum()) for s in ("train", "val", "test")},
        "test_fraud_rate_weighted": round(float(np.average(te.label, weights=te.weight)), 4),
    }}

    with mlflow_run("stage1-cdr", "rule-logreg-lightgbm"):
        mlflow = mlflow_or_stub()

        results["rule"] = evaluate("rule", va, rule_score(va), te, rule_score(te))

        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(tr[FEATURE_NAMES], tr.label)
        results["logreg"] = evaluate("logreg", va, lr.predict_proba(va[FEATURE_NAMES])[:, 1],
                                     te, lr.predict_proba(te[FEATURE_NAMES])[:, 1])

        model = fit_lgb(tr, va)
        p_va, p_te = model.predict_proba(va[FEATURE_NAMES])[:, 1], model.predict_proba(te[FEATURE_NAMES])[:, 1]
        results["lightgbm"] = evaluate("lightgbm", va, p_va, te, p_te)
        results["lightgbm"]["best_iteration"] = int(model.best_iteration_ or LGB_PARAMS["n_estimators"])
        results["lightgbm_per_strategy"] = {
            s: binary_metrics(g.label, p_te[(te.strategy == s).to_numpy()], results["lightgbm"]["threshold"], g.weight)
            for s, g in te.groupby("strategy")
        }
        imp = model.booster_.feature_importance("gain")
        results["feature_importance_gain"] = dict(sorted(
            {f: round(float(v / imp.sum()), 4) for f, v in zip(FEATURE_NAMES, imp)}.items(), key=lambda kv: -kv[1]))
        for k in ("rule", "logreg", "lightgbm"):
            for mk in ("pr_auc", "roc_auc", "f1"):
                if mk in results[k]:
                    mlflow.log_metric(f"{k}_{mk}", results[k][mk])
        mlflow.log_params({k: v for k, v in LGB_PARAMS.items()})

    # ablation: how much does the model lean on time-of-day? (in FraudZen, bypass
    # traffic follows the international calling pattern, which may be a simulator artefact)
    no_hour = [f for f in FEATURE_NAMES if f != "hour_of_day"]
    m_nohour = fit_lgb(tr, va, no_hour)
    results["ablation_no_hour_of_day"] = evaluate(
        "no-hour", va, m_nohour.predict_proba(va[no_hour])[:, 1], te, m_nohour.predict_proba(te[no_hour])[:, 1])

    # leave-one-strategy-out generalisation
    loso = {}
    for held in sorted(df.strategy.unique()):
        trn, tst = df[(df.strategy != held) & (df.split != "test")], df[(df.strategy == held) & (df.split == "test")]
        fit_part, val_part = trn[trn.split == "train"], trn[trn.split == "val"]
        m = fit_lgb(fit_part, val_part)
        loso[held] = evaluate(f"LOSO-{held}", val_part, m.predict_proba(val_part[FEATURE_NAMES])[:, 1],
                              tst, m.predict_proba(tst[FEATURE_NAMES])[:, 1])
    results["leave_one_strategy_out"] = loso

    MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS / "stage1_lgbm.txt"))
    (MODELS / "stage1_meta.json").write_text(json.dumps(
        {"features": FEATURE_NAMES, "threshold": results["lightgbm"]["threshold"]}))
    save_results("stage1_cdr", results)
    return model, results


if __name__ == "__main__":
    main()
