#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PYTHON="${HOME}/miniconda3/envs/translation-transformer/bin/python"

if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON="${PYTHON:-python}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON}" src/train.py \
  --data-dir data/cmn-eng-simple \
  --epochs 120 \
  --batch-size 128 \
  --d-model 256 \
  --num-layers 3 \
  --num-heads 4 \
  --d-ff 512 \
  --dropout 0.1 \
  --label-smoothing 0.03 \
  --scheduler warmup_cosine \
  --warmup-steps 500 \
  --lr 3e-4 \
  --eta-min 1e-5 \
  --decode greedy \
  "$@"
