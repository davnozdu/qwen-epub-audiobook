#!/bin/zsh
set -e
set -o pipefail
TASK_ROOT="${0:A:h}"
cd "$TASK_ROOT"
TASK_PYTHON="$TASK_ROOT/.venv-mlx/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  TASK_PYTHON="/opt/homebrew/opt/python@3.11/bin/python3.11"
  if [[ ! -x "$TASK_PYTHON" ]]; then
    TASK_PYTHON="$(command -v python3.11 || true)"
  fi
  if [[ -z "$TASK_PYTHON" ]]; then
    echo "Нужен Python 3.11: brew install python@3.11"
    read "?Нажмите Enter, чтобы закрыть окно."
    exit 1
  fi
fi
if ! "$TASK_PYTHON" -u "$TASK_ROOT/setup_audiobook.py" --root "$TASK_ROOT"; then
  read "?Ошибка проверки/скачивания. Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
if ! "$TASK_ROOT/.venv-mlx/bin/python" -u "$TASK_ROOT/build_audiobook.py" "$@"; then
  read "?Ошибка сборки. Нажмите Enter, чтобы закрыть окно."
  exit 1
fi
read "?Нажмите Enter, чтобы закрыть окно."
