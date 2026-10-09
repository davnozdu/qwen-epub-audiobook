#!/bin/zsh
set -e
set -o pipefail
TASK_ROOT="${0:A:h}"
TASK_RUNTIME="$TASK_ROOT"
TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
if [[ -x "${TASK_ROOT:h}/.venv-mlx/bin/python" ]]; then
  TASK_RUNTIME="${TASK_ROOT:h}"
  TASK_PYTHON="$TASK_RUNTIME/.venv-mlx/bin/python"
fi
cd "$TASK_ROOT"
TASK_PROJECT="${1:-$TASK_ROOT/work}"
mkdir -p "$TASK_PROJECT/audio" "$TASK_PROJECT/result"
if ! (
  "$TASK_PYTHON" -u "$TASK_ROOT/setup_stress_qwen.py" verify || exit 1
  TASK_MODEL="$TASK_RUNTIME/models/Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source"
  "$TASK_RUNTIME/.venv-stress/bin/python" -u "$TASK_ROOT/prepare_silero.py" \
    --llm-disputes ask --manifest "$TASK_PROJECT/source/manifest.json" || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/synthesize_qwen_mlx.py" \
    --manifest "$TASK_PROJECT/source/manifest.json" \
    --model "$TASK_MODEL" \
    --design-model "$TASK_RUNTIME/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" \
    --ephemeral-voices --mode clone --rab-generation --temperature 0.9 \
    --stress-marks keep --whole-book --out "$TASK_PROJECT/audio" || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/normalize_book_audio.py" \
    --audio-dir "$TASK_PROJECT/audio" --out "$TASK_PROJECT/audio-normalized" || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/audiobook_epub.py" package \
    --manifest "$TASK_PROJECT/source/manifest.json" --audio-dir "$TASK_PROJECT/audio-normalized" --out "$TASK_PROJECT/result"
) 2>&1 | tee "$TASK_PROJECT/run.log"; then
  echo "Ошибка. Журнал: $TASK_PROJECT/run.log"
  read "?Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
open "$TASK_PROJECT/result"
echo "Готово: M4B + EPUB, модели освобождены, временные образцы удалены."
read "?Нажмите Enter, чтобы закрыть окно."
