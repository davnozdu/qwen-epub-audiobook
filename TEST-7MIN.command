#!/bin/zsh
set -e
set -o pipefail
TASK_ROOT="${0:A:h}"
TASK_ROLES="${1:-$TASK_ROOT/demo/roles/manifest.json}"
TASK_TIMING="${2:-$TASK_ROOT/demo/timeline.json}"
if [[ ! -f "$TASK_ROLES" || ! -f "$TASK_TIMING" ]]; then
  echo "Нужны прежние роли и измерения: $0 /путь/manifest.json /путь/timeline.json"
  echo "Текст книги не публикуется на GitHub; в локальном комплекте он находится в demo."
  read "?Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
TASK_PYTHON="$TASK_ROOT/.venv-mlx/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  TASK_PYTHON="/opt/homebrew/opt/python@3.11/bin/python3.11"
fi
TASK_OUT="$TASK_ROOT/results/test-7min-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TASK_OUT"
echo "Около 7 минут: прежний фрагмент + начало; роли, ударения, паузы, музыка 5%"
if ! (
  "$TASK_PYTHON" -u "$TASK_ROOT/setup_audiobook.py" --root "$TASK_ROOT" || exit 1
  TASK_PYTHON="$TASK_ROOT/.venv-mlx/bin/python"
  "$TASK_PYTHON" "$TASK_ROOT/audiobook_epub.py" prepare \
    --manifest "$TASK_ROLES" --timing "$TASK_TIMING" --seconds 420 \
    --start-paragraph p00049 --end-paragraph p00097 --highlight sentence \
    --out "$TASK_OUT/source" || exit 1
  "$TASK_ROOT/.venv-stress/bin/python" -u "$TASK_ROOT/prepare_silero.py" \
    --manifest "$TASK_OUT/source/manifest.json" --dictionary-dir "$TASK_ROOT/data/stress" --llm-disputes ask || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/synthesize_qwen_mlx.py" \
    --manifest "$TASK_OUT/source/manifest.json" \
    --model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source" \
    --design-model "$TASK_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" \
    --ephemeral-voices --mode clone --rab-generation --temperature 0.9 \
    --stress-marks keep --whole-book --out "$TASK_OUT/audio" || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/normalize_book_audio.py" \
    --audio-dir "$TASK_OUT/audio" --out "$TASK_OUT/audio-normalized" || exit 1
  "$TASK_PYTHON" -u "$TASK_ROOT/mix_book_music.py" \
    --audio-dir "$TASK_OUT/audio-normalized" --out "$TASK_OUT/audio-music" --volume-percent 5 || exit 1
  "$TASK_PYTHON" "$TASK_ROOT/audiobook_epub.py" package \
    --manifest "$TASK_OUT/source/manifest.json" --audio-dir "$TASK_OUT/audio-music" --out "$TASK_OUT/result" || exit 1
) 2>&1 | tee "$TASK_OUT/run.log"; then
  echo "Ошибка. Журнал: $TASK_OUT/run.log"
  read "?Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
open "$TASK_OUT/result"
open -a "QuickTime Player" "$TASK_OUT/audio-music/sample.mp3" || true
echo "Готово: $TASK_OUT/result — M4B и EPUB с аудио."
read "?Нажмите Enter, чтобы закрыть окно."
