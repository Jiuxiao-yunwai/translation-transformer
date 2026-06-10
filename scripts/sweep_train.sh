#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${SWEEP_EPOCHS:-50}"
PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
MIN_DELTA="${MIN_DELTA:-0.01}"
BLEU_EVERY="${BLEU_EVERY:-1}"

TRAIN_PATH="${TRAIN_PATH:-data/cmn-eng-simple/training.txt}"
VALID_PATH="${VALID_PATH:-data/cmn-eng-simple/validation.txt}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"

COMMON_ARGS=(
  --train_path "$TRAIN_PATH"
  --valid_path "$VALID_PATH"
  --test_path "$TEST_PATH"
  --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS"
  --scheduler warmup_cosine
  --warmup_steps 800
  --eta_min 1e-5
  --label_smoothing 0.03
  --decode greedy
  --test_decode beam
  --beam_size 5
  --length_penalty 0.8
  --early_stopping_patience "$PATIENCE"
  --min_delta "$MIN_DELTA"
  --bleu_every "$BLEU_EVERY"
  --device "$DEVICE"
)

run_exp() {
  local tag="$1"
  shift
  local run_name="${STAMP}_${tag}"
  local log_dir="outputs/${run_name}"
  mkdir -p "$log_dir"

  echo
  echo "========== ${run_name} =========="
  echo "log: ${log_dir}/train.log"
  echo

  "$PYTHON" src/train.py \
    "${COMMON_ARGS[@]}" \
    --run_name "$run_name" \
    "$@" | tee "${log_dir}/train.log"
}

# Fast, practical grid. Add or remove lines here as needed.
run_exp "dm384_bs128_lr3e4_do01" \
  --batch_size 128 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 3e-4

run_exp "dm384_bs96_lr2e4_do01" \
  --batch_size 96 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 2e-4

run_exp "dm384_bs128_lr3e4_do015" \
  --batch_size 128 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.15 \
  --lr 3e-4

run_exp "dm512_bs64_lr2e4_do015" \
  --batch_size 64 \
  --d_model 512 \
  --nhead 8 \
  --num_layers 3 \
  --dim_feedforward 1536 \
  --dropout 0.15 \
  --lr 2e-4

echo
echo "Sweep finished. Results are under outputs/${STAMP}_*/result.json"
