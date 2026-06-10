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
PATIENCE="${EARLY_STOPPING_PATIENCE:-50}"
MIN_DELTA="${MIN_DELTA:-0.01}"
BLEU_EVERY="${BLEU_EVERY:-1}"

TRAIN_ORIG="${TRAIN_ORIG:-data/cmn-eng-simple/training.txt}"
VALID_ORIG="${VALID_ORIG:-data/cmn-eng-simple/validation.txt}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"
FULL_TRAIN_PATH="${FULL_TRAIN_PATH:-data/train_plus_valid.txt}"

mkdir -p "$(dirname "$FULL_TRAIN_PATH")"
cat "$TRAIN_ORIG" "$VALID_ORIG" > "$FULL_TRAIN_PATH"

echo "full train data: $FULL_TRAIN_PATH"
echo "source train   : $TRAIN_ORIG"
echo "source valid   : $VALID_ORIG"
echo "test data      : $TEST_PATH"
echo "stamp          : $STAMP"

COMMON_ARGS=(
  --train_path "$FULL_TRAIN_PATH"
  --valid_path "$VALID_ORIG"
  --test_path "$TEST_PATH"
  --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS"
  --scheduler warmup_cosine
  --warmup_steps 400
  --eta_min 1e-5
  --label_smoothing 0.03
  --decode greedy
  --test_decode beam
  --final_ckpt last
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
  local run_name="${STAMP}_full_${tag}"
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

# Balanced baseline: same model family as the current best, smaller batch than 256.
run_exp "dm384_h6_l3_bs128_ff1024_lr3e4_do01" \
  --batch_size 128 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 3e-4

# Lower LR, often improves phrase-level BLEU stability.
run_exp "dm384_h6_l3_bs96_ff1024_lr2e4_do01" \
  --batch_size 96 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 2e-4

# More heads with same d_model.
run_exp "dm384_h8_l3_bs128_ff1024_lr3e4_do01" \
  --batch_size 128 \
  --d_model 384 \
  --nhead 8 \
  --num_layers 3 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 3e-4

# Deeper encoder-decoder stack.
run_exp "dm384_h6_l4_bs96_ff1024_lr2e4_do01" \
  --batch_size 96 \
  --d_model 384 \
  --nhead 6 \
  --num_layers 4 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --lr 2e-4

# Larger model, conservative batch.
run_exp "dm512_h8_l3_bs64_ff1536_lr2e4_do015" \
  --batch_size 64 \
  --d_model 512 \
  --nhead 8 \
  --num_layers 3 \
  --dim_feedforward 1536 \
  --dropout 0.15 \
  --lr 2e-4

# Larger and deeper, most expensive config in this sweep.
run_exp "dm512_h8_l4_bs48_ff1536_lr2e4_do015" \
  --batch_size 48 \
  --d_model 512 \
  --nhead 8 \
  --num_layers 4 \
  --dim_feedforward 1536 \
  --dropout 0.15 \
  --lr 2e-4

echo
echo "Full-data sweep finished."
echo "Results are under outputs/${STAMP}_full_*/result.json"
