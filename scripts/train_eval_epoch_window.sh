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

TRAIN_ORIG="${TRAIN_ORIG:-data/cmn-eng-simple/training.txt}"
VALID_ORIG="${VALID_ORIG:-data/cmn-eng-simple/validation.txt}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"
FULL_TRAIN_PATH="${FULL_TRAIN_PATH:-data/train_plus_valid.txt}"

EPOCHS="${EPOCHS:-60}"
SAVE_EPOCH_FROM="${SAVE_EPOCH_FROM:-40}"
SAVE_EPOCH_TO="${SAVE_EPOCH_TO:-60}"
SAVE_EPOCH_EVERY="${SAVE_EPOCH_EVERY:-1}"

BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-400}"
D_MODEL="${D_MODEL:-512}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1536}"
DROPOUT="${DROPOUT:-0.12}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.03}"
BEAM_SIZE="${BEAM_SIZE:-5}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.8}"
BLEU_EVERY="${BLEU_EVERY:-999}"

mkdir -p "$(dirname "$FULL_TRAIN_PATH")"
cat "$TRAIN_ORIG" "$VALID_ORIG" > "$FULL_TRAIN_PATH"

RUN_NAME="${RUN_NAME:-${STAMP}_epoch_window_do${DROPOUT}_ls${LABEL_SMOOTHING}_ep${EPOCHS}}"
LOG_DIR="outputs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "========== ${RUN_NAME} =========="
echo "saving epoch checkpoints: ${SAVE_EPOCH_FROM}-${SAVE_EPOCH_TO}, every ${SAVE_EPOCH_EVERY}"
echo "log: ${LOG_DIR}/train.log"

"$PYTHON" src/train.py \
  --train_path "$FULL_TRAIN_PATH" \
  --valid_path "$VALID_ORIG" \
  --test_path "$TEST_PATH" \
  --checkpoint_dir checkpoints \
  --output_dir outputs \
  --run_name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay 0.0 \
  --scheduler warmup_cosine \
  --warmup_steps "$WARMUP_STEPS" \
  --eta_min 1e-5 \
  --d_model "$D_MODEL" \
  --nhead "$NHEAD" \
  --num_layers "$NUM_LAYERS" \
  --dim_feedforward "$DIM_FEEDFORWARD" \
  --dropout "$DROPOUT" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --grad_clip 1.0 \
  --max_src_len 80 \
  --max_tgt_len 80 \
  --max_len 80 \
  --num_workers "$NUM_WORKERS" \
  --min_freq 1 \
  --decode greedy \
  --test_decode beam \
  --beam_size "$BEAM_SIZE" \
  --length_penalty "$LENGTH_PENALTY" \
  --min_decode_len 2 \
  --early_stopping_patience 0 \
  --min_delta 0.01 \
  --final_ckpt last \
  --bleu_every "$BLEU_EVERY" \
  --save_epoch_from "$SAVE_EPOCH_FROM" \
  --save_epoch_to "$SAVE_EPOCH_TO" \
  --save_epoch_every "$SAVE_EPOCH_EVERY" \
  --eval_saved_epochs \
  --device "$DEVICE" | tee "${LOG_DIR}/train.log"

echo
echo "Epoch-window summary: ${LOG_DIR}/epoch_evals/summary.json"
