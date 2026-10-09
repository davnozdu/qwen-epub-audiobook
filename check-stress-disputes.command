#!/bin/zsh
set -e
set -o pipefail
TASK_PROJECT="${0:A:h}"
TASK_RUNTIME="$TASK_PROJECT"
if [[ -x "${TASK_PROJECT:h}/.venv-stress/bin/python" ]]; then
  TASK_RUNTIME="${TASK_PROJECT:h}"
fi
TASK_OUT="$TASK_RUNTIME/book-test-rab-roles/hybrid-llm-check"
mkdir -p "$TASK_OUT"
"$TASK_RUNTIME/.venv-stress/bin/python" -u "$TASK_PROJECT/prepare_silero.py" \
  --manifest "$TASK_RUNTIME/book-test-rab-roles/source/manifest.json" \
  --strategy hybrid --llm-disputes ask --out "$TASK_OUT/manifest.json" 2>&1 | tee "$TASK_OUT/run.log"
echo "Готово без аудио: $TASK_OUT/book-stressed.txt"
