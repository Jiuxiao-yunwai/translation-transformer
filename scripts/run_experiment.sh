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

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_experiment.sh [experiment options] [train options]

Experiment options:
  --run-id VALUE                Name of the run directory under runs/.
  --eval-split VALUE            Split to evaluate: training, validation, testing. Default: testing.
  --eval-decode VALUE           Decode mode for evaluation: greedy or beam. Default: greedy.
  --eval-beam-size VALUE        Beam size when --eval-decode beam. Default: 4.
  --eval-length-penalty VALUE   Beam length penalty. Default: 0.6.
  --eval-show VALUE             Number of translation examples to print. Default: 10.
  -h, --help                    Show this help message.

All other options are passed through to scripts/train.sh.
Both --name value and --name=value forms are supported for experiment options.

Examples:
  ./scripts/run_experiment.sh --epochs 30 --batch-size 128
  ./scripts/run_experiment.sh --eval-decode beam --eval-beam-size 3 --epochs 120
  ./scripts/run_experiment.sh --run-id small_model_01 --label-smoothing 0.03
EOF
}

RUN_ID="${RUN_ID:-}"
EVAL_SPLIT="${EVAL_SPLIT:-testing}"
EVAL_DECODE="${EVAL_DECODE:-greedy}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-4}"
EVAL_LENGTH_PENALTY="${EVAL_LENGTH_PENALTY:-0.6}"
EVAL_SHOW="${EVAL_SHOW:-10}"
TRAIN_ARGS=()

require_value() {
  local option="$1"
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    echo "Missing value for ${option}" >&2
    exit 2
  fi
}

clean_log_stream() {
  "${PYTHON}" -c '
import re
import sys

ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

for chunk in sys.stdin:
    for line in chunk.replace("\r", "\n").splitlines():
        line = ansi_escape.sub("", line).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if "%|" in stripped or "it/s" in stripped:
            continue
        print(line, flush=True)
'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="$2"
      shift 2
      ;;
    --run-id=*)
      RUN_ID="${1#*=}"
      shift
      ;;
    --eval-split)
      require_value "$1" "${2:-}"
      EVAL_SPLIT="$2"
      shift 2
      ;;
    --eval-split=*)
      EVAL_SPLIT="${1#*=}"
      shift
      ;;
    --eval-decode)
      require_value "$1" "${2:-}"
      EVAL_DECODE="$2"
      shift 2
      ;;
    --eval-decode=*)
      EVAL_DECODE="${1#*=}"
      shift
      ;;
    --eval-beam-size)
      require_value "$1" "${2:-}"
      EVAL_BEAM_SIZE="$2"
      shift 2
      ;;
    --eval-beam-size=*)
      EVAL_BEAM_SIZE="${1#*=}"
      shift
      ;;
    --eval-length-penalty)
      require_value "$1" "${2:-}"
      EVAL_LENGTH_PENALTY="$2"
      shift 2
      ;;
    --eval-length-penalty=*)
      EVAL_LENGTH_PENALTY="${1#*=}"
      shift
      ;;
    --eval-show)
      require_value "$1" "${2:-}"
      EVAL_SHOW="$2"
      shift 2
      ;;
    --eval-show=*)
      EVAL_SHOW="${1#*=}"
      shift
      ;;
    --)
      shift
      TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_ID}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TRAIN_LOG="${RUN_DIR}/train.log"
EVAL_LOG="${RUN_DIR}/eval_${EVAL_SPLIT}_${EVAL_DECODE}.log"
ENV_LOG="${RUN_DIR}/environment.txt"
REPORT_FILE="${RUN_DIR}/report.md"

mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

{
  echo "python=$("${PYTHON}" --version)"
  "${PYTHON}" - <<'PY'
import platform
import torch

print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
} > "${ENV_LOG}"

TRAIN_CMD=(
  ./scripts/train.sh
  "${TRAIN_ARGS[@]}"
  --save-dir "${CHECKPOINT_DIR}"
)

printf 'Run directory: %s\n' "${RUN_DIR}"
printf 'Training command: '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}" 2>&1 | tee >(clean_log_stream > "${TRAIN_LOG}")

CHECKPOINT="${CHECKPOINT_DIR}/best.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="${CHECKPOINT_DIR}/last.pt"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint was produced; cannot run evaluation." >&2
  exit 1
fi

EVAL_CMD=(
  ./scripts/evaluate.sh
  --checkpoint "${CHECKPOINT}"
  --split "${EVAL_SPLIT}"
  --decode "${EVAL_DECODE}"
  --beam-size "${EVAL_BEAM_SIZE}"
  --length-penalty "${EVAL_LENGTH_PENALTY}"
  --show "${EVAL_SHOW}"
)

printf 'Evaluation command: '
printf '%q ' "${EVAL_CMD[@]}"
printf '\n'

"${EVAL_CMD[@]}" 2>&1 | tee >(clean_log_stream > "${EVAL_LOG}")

FINAL_EPOCH_LINE="$("${PYTHON}" - "${TRAIN_LOG}" <<'PY'
import sys

path = sys.argv[1]
matches = []
with open(path, encoding="utf-8", errors="replace") as f:
    for chunk in f:
        for line in chunk.replace("\r", "\n").splitlines():
            line = line.strip()
            if line.startswith("epoch "):
                matches.append(line)
print(matches[-1] if matches else "")
PY
)"
TEST_BLEU_LINE="$("${PYTHON}" - "${EVAL_LOG}" "${EVAL_SPLIT}" <<'PY'
import sys

path, split = sys.argv[1], sys.argv[2]
needle = f"{split} BLEU:"
matches = []
with open(path, encoding="utf-8", errors="replace") as f:
    for chunk in f:
        for line in chunk.replace("\r", "\n").splitlines():
            line = line.strip()
            if needle in line:
                matches.append(line)
print(matches[-1] if matches else "")
PY
)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"
GIT_STATUS="$(git status --short 2>/dev/null || true)"

CHECKPOINT_SUMMARY="$("${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print(f"checkpoint={path}")
print(f"epoch={ckpt.get('epoch')}")
print(f"val_loss={ckpt.get('val_loss')}")
print(f"val_bleu={ckpt.get('bleu')}")
train_args = ckpt.get("train_args", {})
for key in [
    "epochs",
    "batch_size",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "dropout",
    "label_smoothing",
    "scheduler",
    "warmup_steps",
    "lr",
    "eta_min",
    "decode",
]:
    if key in train_args:
        print(f"{key}={train_args[key]}")
PY
)"

cat > "${REPORT_FILE}" <<EOF
# Transformer Translation Run ${RUN_ID}

## Summary

- Run directory: \`${RUN_DIR}\`
- Git commit: \`${GIT_COMMIT}\`
- Final training line: \`${FINAL_EPOCH_LINE}\`
- Evaluation result: \`${TEST_BLEU_LINE}\`

## Environment

\`\`\`text
$(cat "${ENV_LOG}")
\`\`\`

## Commands

Training:

\`\`\`bash
$(printf '%q ' "${TRAIN_CMD[@]}")
\`\`\`

Evaluation:

\`\`\`bash
$(printf '%q ' "${EVAL_CMD[@]}")
\`\`\`

## Checkpoint

\`\`\`text
${CHECKPOINT_SUMMARY}
\`\`\`

## Git Status

\`\`\`text
${GIT_STATUS}
\`\`\`

## Logs

- Training log: \`${TRAIN_LOG}\`
- Evaluation log: \`${EVAL_LOG}\`
- Environment: \`${ENV_LOG}\`
EOF

printf 'Report written to: %s\n' "${REPORT_FILE}"
