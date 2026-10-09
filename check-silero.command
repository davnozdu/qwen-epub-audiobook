#!/bin/zsh
set -e
set -o pipefail
TASK_PROJECT="${0:A:h}"
TASK_RUNTIME="$TASK_PROJECT"
if [[ -x "${TASK_PROJECT:h}/.venv-stress/bin/python" ]]; then
  TASK_RUNTIME="${TASK_PROJECT:h}"
fi
TASK_MANIFEST="${1:-$TASK_RUNTIME/book-test-rab-roles/source/manifest.json}"
TASK_OUT="${2:-$TASK_RUNTIME/book-test-rab-roles/hybrid-check}"
mkdir -p "$TASK_OUT"
"$TASK_RUNTIME/.venv-stress/bin/python" -u "$TASK_PROJECT/prepare_silero.py" \
  --manifest "$TASK_MANIFEST" --strategy hybrid --out "$TASK_OUT/manifest.json" 2>&1 | tee "$TASK_OUT/run.log"
echo "Проверка без аудио завершена: $TASK_OUT/book-stressed.txt"
echo "Сравнение: $TASK_OUT/silero-comparison.json"
