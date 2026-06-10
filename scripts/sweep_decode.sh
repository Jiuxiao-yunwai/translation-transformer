#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python" ]]; then
  PYTHON="${PYTHON:-/home/jiuxiao-u/miniconda3/envs/translation-transformer/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/sweep_decode.sh checkpoints/<run_name>/last.pt"
  echo "   or: bash scripts/sweep_decode.sh checkpoints/<run_name>/best.pt"
  exit 1
fi

CKPT="$1"
STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
TEST_PATH="${TEST_PATH:-data/cmn-eng-simple/testing.txt}"
BATCH_SIZE="${BATCH_SIZE:-128}"

LP_VALUES=(0.6 0.7 0.75 0.8 0.85 0.9 1.0)
BEAM_VALUES=(3 4 5 8)

run_eval() {
  local beam_size="$1"
  local length_penalty="$2"
  local phase="$3"
  local run_name="${STAMP}_${phase}_beam${beam_size}_lp${length_penalty}"

  echo
  echo "========== ${run_name} =========="

  "$PYTHON" src/evaluate.py \
    --ckpt "$CKPT" \
    --test_path "$TEST_PATH" \
    --run_name "$run_name" \
    --batch_size "$BATCH_SIZE" \
    --decode beam \
    --beam_size "$beam_size" \
    --length_penalty "$length_penalty" \
    --device "$DEVICE" \
    # --no-progress
}

echo "Phase 1: fixed beam_size=5, sweeping length_penalty"
for lp in "${LP_VALUES[@]}"; do
  run_eval 5 "$lp" "decode_lp"
done

BEST_LP="$(
  "$PYTHON" - <<PY
import json
from pathlib import Path

rows = []
for p in Path("outputs").glob("${STAMP}_decode_lp_beam5_lp*/result.json"):
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append((float(data.get("test_bleu4", data.get("test_bleu", -1))), data.get("length_penalty"), str(p)))
rows.sort(reverse=True)
if not rows:
    raise SystemExit("no decode lp results found")
print(rows[0][1])
PY
)"

echo
echo "Best length_penalty from phase 1: ${BEST_LP}"
echo "Phase 2: fixed length_penalty=${BEST_LP}, sweeping beam_size"

for beam in "${BEAM_VALUES[@]}"; do
  run_eval "$beam" "$BEST_LP" "decode_beam"
done

echo
echo "Decode sweep finished."
echo "Phase 1 results: outputs/${STAMP}_decode_lp_*/result.json"
echo "Phase 2 results: outputs/${STAMP}_decode_beam_*/result.json"
