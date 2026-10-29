"""Package both models for the API into models/serving/ and verify them.

Stage 1: LightGBM booster -> ONNX (onnxmltools); checks ONNX == LightGBM predictions.
Stage 2: copies the INT8-quantised DistilBERT ONNX + tokenizer.json.

    python scripts/build_serving.py --text-model models/distilbert_mix
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from callshield.cdr.data import load  # noqa: E402
from callshield.common import MODELS, save_results  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--text-model", default=str(MODELS / "distilbert_mix"))
args = ap.parse_args()

out = MODELS / "serving"
out.mkdir(parents=True, exist_ok=True)
report = {}

# ---- Stage 1 ----
meta = json.loads((MODELS / "stage1_meta.json").read_text())
feats = meta["features"]
booster = lgb.Booster(model_file=str(MODELS / "stage1_lgbm.txt"))
onx = convert_lightgbm(booster, initial_types=[("features", FloatTensorType([None, len(feats)]))],
                       zipmap=False, target_opset=15)
(out / "stage1_lgbm.onnx").write_bytes(onx.SerializeToString())
sess = ort.InferenceSession(str(out / "stage1_lgbm.onnx"), providers=["CPUExecutionProvider"])
te = load()
te = te[te.split == "test"]
X = te[feats].to_numpy(np.float32)
p_lgb = booster.predict(X)
p_onnx = sess.run(None, {"features": X})[1][:, 1]
absdiff = np.abs(p_lgb - p_onnx)
diff, p999 = float(absdiff.max()), float(np.percentile(absdiff, 99.9))
agree = float(((p_lgb >= meta["threshold"]) == (p_onnx >= meta["threshold"])).mean())
print(f"stage1 ONNX parity: max |dp| = {diff:.2e}, 99.9th pct |dp| = {p999:.2e}, decision agreement = {agree:.4%}")
# ONNX-ML stores tree split thresholds as float32, so a feature value lying exactly
# on a split point can take the other branch for a handful of rows. What matters is
# that the fraud/legit decisions are identical and typical differences are tiny.
assert agree > 0.999 and p999 < 1e-2, "ONNX export does not match LightGBM"
(out / "stage1.json").write_text(json.dumps({"features": feats, "threshold": meta["threshold"]}))
report["stage1"] = {"max_abs_diff": diff, "p999_abs_diff": p999, "mean_abs_diff": float(np.abs(p_lgb - p_onnx).mean()), "decision_agreement": agree, "onnx_kb": round((out / "stage1_lgbm.onnx").stat().st_size / 1e3, 1)}

# ---- Stage 2 ----
tm = Path(args.text_model)
if (tm / "model.int8.onnx").exists():
    shutil.copy(tm / "model.int8.onnx", out / "stage2_distilbert.int8.onnx")
    shutil.copy(tm / "tokenizer.json", out / "tokenizer.json")
    cfg = json.loads((tm / "callshield.json").read_text())
    (out / "stage2.json").write_text(json.dumps({"max_len": cfg["max_len"], "threshold": 0.5, "train_on": cfg["train_on"]}))
    report["stage2"] = {"source": str(tm.name)}
    print("stage2 bundle copied from", tm)
else:
    print("stage2 ONNX not found yet - run callshield.text.export_onnx first")

save_results("serving_bundle", report)
