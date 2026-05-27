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

exec "${PYTHON}" src/translate.py "$@"
