#!/bin/sh
# run-book.command — обёртка для macOS и Linux. Вся работа в run-book.py (он же для Windows).
#
#   ./run-book.command kniga.epub              # книга в этой папке
#   ./run-book.command ../kniga.epub           # или любым относительным путём
#   ./run-book.command https://site.ru/b.epub  # или ссылкой
#
# Переменные: GPU, SESSION, FRAGMENT, HIGHLIGHT, KEEP_SESSION, DRIVE, DRIVE_DIR, DRIVE_WAIT
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
JOB="$HERE/run-book.py"
[ -f "$JOB" ] || { echo "рядом с этим файлом нет run-book.py"; exit 1; }

# python3 есть почти везде; на старых системах может быть только python
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$JOB" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$JOB" "$@"
else
    echo "нужен python 3: https://www.python.org/downloads/"
    exit 1
fi
