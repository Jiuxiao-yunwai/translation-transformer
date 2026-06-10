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
STAGE="${STAGE:-regularization}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${SWEEP_EPOCHS:-50}"
BLEU_EVERY="${BLEU_EVERY:-999}"

TRAIN_ORIG="${TRAIN_ORIG:-data/cmn-eng-simple/training.txt}"
VALID_ORIG="${VALID_ORIG:-data/cmn-eng-simple/validation.txt}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"
FULL_TRAIN_PATH="${FULL_TRAIN_PATH:-data/train_plus_valid.txt}"

mkdir -p "$(dirname "$FULL_TRAIN_PATH")"
cat "$TRAIN_ORIG" "$VALID_ORIG" > "$FULL_TRAIN_PATH"

warmup_for_batch() {
  case "$1" in
    48) echo 400 ;;
    64) echo "${WARMUP_64:-400}" ;;
    96) echo "${WARMUP_96:-300}" ;;
    128) echo "${WARMUP_128:-300}" ;;
    *) echo 400 ;;
  esac
}

COMMON_ARGS=(
  --train_path "$FULL_TRAIN_PATH"
  --valid_path "$VALID_ORIG"
  --test_path "$TEST_PATH"
  --checkpoint_dir checkpoints
  --output_dir outputs
  --weight_decay 0.0
  --scheduler warmup_cosine
  --eta_min 1e-5
  --grad_clip 1.0
  --max_src_len 80
  --max_tgt_len 80
  --max_len 80
  --num_workers "$NUM_WORKERS"
  --min_freq 1
  --decode greedy
  --test_decode beam
  --beam_size "${BEAM_SIZE:-5}"
  --length_penalty "${LENGTH_PENALTY:-0.8}"
  --min_decode_len 2
  --early_stopping_patience 0
  --min_delta 0.01
  --final_ckpt last
  --bleu_every "$BLEU_EVERY"
  --device "$DEVICE"
)

run_exp() {
  local tag="$1"
  local epochs="$2"
  local batch_size="$3"
  local label_smoothing="$4"
  shift 4

  local warmup_steps
  warmup_steps="$(warmup_for_batch "$batch_size")"
  local run_name="${STAMP}_${STAGE}_${tag}_ep${epochs}"
  local log_dir="outputs/${run_name}"
  mkdir -p "$log_dir"

  echo
  echo "========== ${run_name} =========="
  echo "log: ${log_dir}/train.log"
  echo "warmup_steps=${warmup_steps}"
  echo

  "$PYTHON" src/train.py \
    "${COMMON_ARGS[@]}" \
    --run_name "$run_name" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --warmup_steps "$warmup_steps" \
    --label_smoothing "$label_smoothing" \
    "$@" | tee "${log_dir}/train.log"
}

run_regularization_stage() {
  # Stage 2: fixed current best model; only search dropout and label_smoothing.
  run_exp "A_do012_ls003" "$EPOCHS" 48 0.03 \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout 0.12

  run_exp "B_do018_ls003" "$EPOCHS" 48 0.03 \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout 0.18

  run_exp "C_do015_ls002" "$EPOCHS" 48 0.02 \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout 0.15

  run_exp "D_do015_ls004" "$EPOCHS" 48 0.04 \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout 0.15

  run_exp "E_do015_ls003_baseline" "$EPOCHS" 48 0.03 \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout 0.15
}

run_lr_stage() {
  # Stage 3: set best dropout/label_smoothing via env after stage 2 if needed.
  local dropout="${DROPOUT:-0.15}"
  local label_smoothing="${LABEL_SMOOTHING:-0.03}"

  for lr in 1.5e-4 2e-4 2.5e-4 3e-4; do
    local lr_tag="${lr/e-/e}"
    lr_tag="${lr_tag/./p}"
    run_exp "lr${lr_tag}_do${dropout}_ls${label_smoothing}" "$EPOCHS" 48 "$label_smoothing" \
      --lr "$lr" --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout "$dropout"
  done
}

run_layer_stage() {
  # Stage 4: keep both candidates; they have been close historically.
  local dropout="${DROPOUT:-0.15}"
  local label_smoothing="${LABEL_SMOOTHING:-0.03}"

  run_exp "candidate1_l4_bs48" "$EPOCHS" 48 "$label_smoothing" \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 4 --dim_feedforward 1536 --dropout "$dropout"

  run_exp "candidate2_l3_bs64" "$EPOCHS" 64 "$label_smoothing" \
    --lr 2e-4 --d_model 512 --nhead 8 --num_layers 3 --dim_feedforward 1536 --dropout "$dropout"
}

run_epoch_stage() {
  # Optional fixed-epoch comparison for final training without early stopping.
  local dropout="${DROPOUT:-0.15}"
  local label_smoothing="${LABEL_SMOOTHING:-0.03}"
  local lr="${LR:-2e-4}"
  local layers="${NUM_LAYERS:-4}"
  local batch_size="${BATCH_SIZE:-48}"

  for epochs in 43 47 50 55; do
    run_exp "epochs${epochs}_l${layers}_bs${batch_size}" "$epochs" "$batch_size" "$label_smoothing" \
      --lr "$lr" --d_model 512 --nhead 8 --num_layers "$layers" --dim_feedforward 1536 --dropout "$dropout"
  done
}

case "$STAGE" in
  regularization)
    run_regularization_stage
    ;;
  lr)
    run_lr_stage
    ;;
  layers)
    run_layer_stage
    ;;
  epochs)
    run_epoch_stage
    ;;
  all)
    run_regularization_stage
    STAGE="lr" run_lr_stage
    STAGE="layers" run_layer_stage
    STAGE="epochs" run_epoch_stage
    ;;
  *)
    echo "Unknown STAGE=${STAGE}"
    echo "Use one of: regularization, lr, layers, epochs, all"
    exit 1
    ;;
esac

echo
echo "Sweep stage '${STAGE}' finished."
echo "Results are under outputs/${STAMP}_*/result.json"
