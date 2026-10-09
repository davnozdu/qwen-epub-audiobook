#!/bin/zsh
set -e
set -o pipefail
TASK_PROJECT="${0:A:h}"
TASK_ROOT="$TASK_PROJECT"
TASK_PYTHON="$TASK_PROJECT/.venv/bin/python"
if [[ -x "${TASK_PROJECT:h}/.venv-mlx/bin/python" ]]; then
  TASK_ROOT="${TASK_PROJECT:h}"
  TASK_PYTHON="$TASK_ROOT/.venv-mlx/bin/python"
fi
cd "$TASK_ROOT"
TASK_OUT="$TASK_ROOT/book-test-rab-roles"
if [[ "${1:-}" == "--full" ]]; then
  TASK_OUT="$TASK_ROOT/book-full-rab-roles"
elif [[ -n "${1:-}" ]]; then
  echo "Использование: $0 [--full]"
  exit 1
fi
mkdir -p "$TASK_OUT/audio" "$TASK_OUT/result"
echo "Ru-Stress-CF / rab: готовые роли → предложения → словари + контроль LLM без размышлений → VoiceDesign → Base → M4B + EPUB"
if ! (
  "$TASK_PYTHON" -u "$TASK_PROJECT/setup_stress_qwen.py" verify || exit 1
  if [[ "${1:-}" == "--full" ]]; then
    "$TASK_PYTHON" "$TASK_PROJECT/audiobook_epub.py" prepare \
      --manifest "$TASK_ROOT/mytts-books/out/skuki-qwen-fast/manifest.json" --highlight sentence --out "$TASK_OUT/source" || exit 1
  else
    "$TASK_PYTHON" "$TASK_PROJECT/audiobook_epub.py" prepare \
      --manifest "$TASK_ROOT/mytts-books/out/skuki-qwen-fast/manifest.json" --highlight sentence \
      --seconds 300 --timing "$TASK_ROOT/audio-test-fresh/timeline.json" \
      --start-paragraph p00058 --out "$TASK_OUT/source" || exit 1
  fi
  "$TASK_PYTHON" -u "$TASK_PROJECT/prepare_speech.py" run \
    --manifest "$TASK_OUT/source/manifest.json" || exit 1
  "$TASK_PYTHON" -u "$TASK_PROJECT/synthesize_qwen_mlx.py" \
    --manifest "$TASK_OUT/source/manifest.json" \
    --model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source" \
    --design-model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" \
    --ephemeral-voices --mode clone --rab-generation --temperature 0.9 \
    --stress-marks keep --whole-book --out "$TASK_OUT/audio" || exit 1
  "$TASK_PYTHON" -u "$TASK_PROJECT/audiobook_epub.py" package \
    --manifest "$TASK_OUT/source/manifest.json" --audio-dir "$TASK_OUT/audio" --out "$TASK_OUT/result"
) 2>&1 | tee "$TASK_OUT/run.log"; then
  echo "Ошибка. Журнал: $TASK_OUT/run.log"
  read "?Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
open "$TASK_OUT/result"
echo "Готово: book.m4b и book-read-along.epub. Модели выгружены, временные голоса удалены."
read "?Нажмите Enter, чтобы закрыть окно."
