#!/bin/zsh
set -e
set -o pipefail
TASK_ROOT="${0:A:h}"
cd "$TASK_ROOT"
TASK_PROJECT="${1:-$TASK_ROOT/work}"
mkdir -p "$TASK_PROJECT/audio" "$TASK_PROJECT/result"
if ! (
  "$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/synthesize_qwen_mlx.py" \
    --manifest "$TASK_PROJECT/source/manifest.json" \
    --model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base-8bit" \
    --design-model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" \
    --ephemeral-voices --mode clone --whole-book --out "$TASK_PROJECT/audio" || exit 1
  "$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/audiobook_epub.py" package \
    --manifest "$TASK_PROJECT/source/manifest.json" --audio-dir "$TASK_PROJECT/audio" --out "$TASK_PROJECT/result"
) 2>&1 | tee "$TASK_PROJECT/run.log"; then
  echo "Ошибка. Журнал: $TASK_PROJECT/run.log"
  read "?Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
open "$TASK_PROJECT/result"
echo "Готово: M4B + EPUB, модели освобождены, временные образцы удалены."
read "?Нажмите Enter, чтобы закрыть окно."
