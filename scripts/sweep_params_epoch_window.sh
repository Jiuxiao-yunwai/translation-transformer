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
STAGE="${STAGE:-param_epoch_window}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PROGRESS_FLAG="${PROGRESS_FLAG:---progress}"
SKIP_DONE="${SKIP_DONE:-1}"

TRAIN_ORIG="${TRAIN_ORIG:-data/cmn-eng-simple/training.txt}"
VALID_ORIG="${VALID_ORIG:-data/cmn-eng-simple/validation.txt}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"
FULL_TRAIN_PATH="${FULL_TRAIN_PATH:-data/train_plus_valid.txt}"

EPOCHS="${EPOCHS:-60}"
SAVE_EPOCH_FROM="${SAVE_EPOCH_FROM:-40}"
SAVE_EPOCH_TO="${SAVE_EPOCH_TO:-60}"
SAVE_EPOCH_EVERY="${SAVE_EPOCH_EVERY:-1}"
BLEU_EVERY="${BLEU_EVERY:-999}"

BATCH_SIZE_VALUES="${BATCH_SIZE_VALUES:-48}"
LR_VALUES="${LR_VALUES:-2e-4}"
DROPOUT_VALUES="${DROPOUT_VALUES:-0.10 0.11 0.12 0.13}"
LABEL_SMOOTHING_VALUES="${LABEL_SMOOTHING_VALUES:-0.025 0.03 0.035}"
NUM_LAYERS_VALUES="${NUM_LAYERS_VALUES:-4}"

D_MODEL="${D_MODEL:-512}"
NHEAD="${NHEAD:-8}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1536}"
BEAM_SIZE="${BEAM_SIZE:-5}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.8}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$FULL_TRAIN_PATH")"
cat "$TRAIN_ORIG" "$VALID_ORIG" > "$FULL_TRAIN_PATH"

warmup_for_batch() {
  case "$1" in
    48) echo "${WARMUP_48:-400}" ;;
    64) echo "${WARMUP_64:-400}" ;;
    96) echo "${WARMUP_96:-300}" ;;
    128) echo "${WARMUP_128:-300}" ;;
    *) echo "${WARMUP_STEPS:-400}" ;;
  esac
}

tag_float() {
  local value="$1"
  value="${value/e-/e}"
  value="${value/./p}"
  echo "$value"
}

run_one() {
  local batch_size="$1"
  local lr="$2"
  local dropout="$3"
  local label_smoothing="$4"
  local layers="$5"

  local warmup_steps
  warmup_steps="$(warmup_for_batch "$batch_size")"

  local lr_tag dropout_tag ls_tag
  lr_tag="$(tag_float "$lr")"
  dropout_tag="$(tag_float "$dropout")"
  ls_tag="$(tag_float "$label_smoothing")"

  local run_name="${STAMP}_${STAGE}_l${layers}_bs${batch_size}_lr${lr_tag}_do${dropout_tag}_ls${ls_tag}_ep${EPOCHS}"
  local log_dir="outputs/${run_name}"
  local summary_path="${log_dir}/epoch_evals/summary.json"

  if [[ "$SKIP_DONE" == "1" && -f "$summary_path" ]]; then
    echo
    echo "========== skip ${run_name} =========="
    echo "found ${summary_path}"
    return
  fi

  mkdir -p "$log_dir"
  echo
  echo "========== ${run_name} =========="
  echo "params: layers=${layers} batch_size=${batch_size} lr=${lr} dropout=${dropout} label_smoothing=${label_smoothing}"
  echo "epoch checkpoints: ${SAVE_EPOCH_FROM}-${SAVE_EPOCH_TO}, every ${SAVE_EPOCH_EVERY}"
  echo "log: ${log_dir}/train.log"
  echo

  "$PYTHON" src/train.py \
    --train_path "$FULL_TRAIN_PATH" \
    --valid_path "$VALID_ORIG" \
    --test_path "$TEST_PATH" \
    --checkpoint_dir checkpoints \
    --output_dir outputs \
    --run_name "$run_name" \
    --epochs "$EPOCHS" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --weight_decay 0.0 \
    --scheduler warmup_cosine \
    --warmup_steps "$warmup_steps" \
    --eta_min 1e-5 \
    --d_model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num_layers "$layers" \
    --dim_feedforward "$DIM_FEEDFORWARD" \
    --dropout "$dropout" \
    --label_smoothing "$label_smoothing" \
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
    --seed "$SEED" \
    --device "$DEVICE" \
    "$PROGRESS_FLAG" | tee "${log_dir}/train.log"
}

for layers in $NUM_LAYERS_VALUES; do
  for batch_size in $BATCH_SIZE_VALUES; do
    for lr in $LR_VALUES; do
      for dropout in $DROPOUT_VALUES; do
        for label_smoothing in $LABEL_SMOOTHING_VALUES; do
          run_one "$batch_size" "$lr" "$dropout" "$label_smoothing" "$layers"
        done
      done
    done
  done
done

AGG_PATH="outputs/${STAMP}_${STAGE}_summary.json"
"$PYTHON" - "$STAMP" "$STAGE" "$AGG_PATH" <<'PY'
import json
import sys
from pathlib import Path

stamp, stage, agg_path = sys.argv[1:4]
rows = []
for summary_path in sorted(Path("outputs").glob(f"{stamp}_{stage}_*/epoch_evals/summary.json")):
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    best = summary.get("best") or {}
    if not best:
        continue
    rows.append(
        {
            "run_name": summary.get("run_name"),
            "best_epoch": best.get("epoch"),
            "best_checkpoint": best.get("checkpoint"),
            "test_bleu4": best.get("test_bleu4", best.get("test_bleu")),
            "test_bleu1": best.get("test_bleu1"),
            "test_bleu2": best.get("test_bleu2"),
            "test_bleu3": best.get("test_bleu3"),
            "test_token_accuracy": best.get("test_token_accuracy"),
            "test_loss": best.get("test_loss"),
            "summary_path": str(summary_path),
        }
    )

rows.sort(key=lambda item: item.get("test_bleu4", -1), reverse=True)
payload = {"stamp": stamp, "stage": stage, "count": len(rows), "best": rows[0] if rows else None, "results": rows}
Path(agg_path).parent.mkdir(parents=True, exist_ok=True)
with Path(agg_path).open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"wrote {agg_path}")
if rows:
    print("top results:")
    for row in rows[:10]:
        print(
            f"{row['test_bleu4']:.2f}\t"
            f"epoch={row['best_epoch']}\t"
            f"acc={row['test_token_accuracy']:.4f}\t"
            f"{row['run_name']}"
        )
PY

echo
echo "Sweep finished."
echo "Aggregate summary: ${AGG_PATH}"
