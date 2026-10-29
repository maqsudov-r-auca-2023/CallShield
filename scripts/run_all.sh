#!/usr/bin/env bash
# Run the whole CallShield pipeline end to end. Each step is skipped if its
# output already exists, so after a failure you can simply run it again.
#   bash scripts/run_all.sh            (FORCE=1 to redo everything)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
mkdir -p logs reports

step() {  # step <name> <output-file> <command...>
  local name=$1 out=$2; shift 2
  if [ -z "${FORCE:-}" ] && [ -e "$out" ]; then echo "== [skip] $name"; return; fi
  echo "== [run]  $name   ($(date +%H:%M:%S))"
  "$@" 2>&1 | tee "logs/$name.log"
}

step 01_unit_tests        logs/.unit_ok        bash -c 'python -m pytest -q tests/test_features.py && touch logs/.unit_ok'
step 02_cdr_dataset       data/processed/cdr_calls.parquet   python -m callshield.cdr.data
step 03_stage1_train      reports/stage1_cdr.json            python -m callshield.cdr.train
step 04_text_dataset      data/processed/text_youtube.parquet python -m callshield.text.data
step 05_text_baseline     reports/stage2_baseline.json       python -m callshield.text.baseline
step 06_distilbert_scamdialogue reports/stage2_distilbert_scamdialogue.json python -m callshield.text.finetune --train-on scamdialogue --epochs 3
step 07_distilbert_mix    reports/stage2_distilbert_mix.json python -m callshield.text.finetune --train-on mix --epochs 3
step 08_onnx_quantize     reports/stage2_onnx.json           python -m callshield.text.export_onnx --model-dir models/distilbert_mix
step 08b_window_eval      reports/stage2_windows.json        python -m callshield.text.eval_windows --model-dir models/distilbert_mix
step 09_serving_bundle    reports/serving_bundle.json        python scripts/build_serving.py --text-model models/distilbert_mix
step 10_all_tests         logs/.all_ok         bash -c 'python -m pytest -q tests && touch logs/.all_ok'
step 11_api_bench_memory  reports/api_bench_memory.json      python scripts/bench_api.py --workers 4

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose up -d redis
  step 12_api_bench_redis reports/api_bench_redis.json env REDIS_URL=redis://localhost:6379/0 python scripts/bench_api.py --workers 4
  step 13_docker_build    logs/.docker_ok bash -c 'docker compose build api && touch logs/.docker_ok'
  docker compose down
else
  echo "== [skip] Docker not running - Redis benchmark and image build skipped"
fi

python scripts/summarize.py
echo "ALL DONE - results in reports/RESULTS.md"
