#!/bin/bash
# run-book.command — озвучить книгу через Colab CLI. Запускать из этой папки:
#
#   ./run-book.command kniga.epub              # книга в этой папке
#   ./run-book.command ../kniga.epub           # или любым относительным путём
#   ./run-book.command https://site.ru/b.epub  # или ссылкой
#
# Жёстких путей внутри нет: папка определяется по расположению самого файла, книга — аргументом,
# всё остальное живёт на арендованной машине Colab (/content).
#
# Что делает:
#   1. освобождает прошлые сессии CLI (иначе Colab отвечает «Allocation refused»);
#   2. поднимает сессию с GPU, кладёт туда книгу, запускает задание qwen_tts_book.py;
#   3. забирает готовый zip в ~/Downloads и освобождает машину.
#
# Переменные окружения:
#   GPU=L4              ускоритель (по умолчанию T4)
#   SESSION=book        имя сессии
#   FRAGMENT=300        озвучить только 300 секунд вместо всей книги
#   HIGHLIGHT=fragment  подсветка по абзацам: сегментов меньше, синтез быстрее
#   KEEP_SESSION=1      не гасить машину в конце: следующая книга пойдёт без установки
#                       пакетов и без повторной загрузки весов (потом: colab stop -s book)
set -euo pipefail

BOOK="${1:-}"
if [ -z "$BOOK" ] || [ "$BOOK" = "-h" ] || [ "$BOOK" = "--help" ]; then
    echo "укажите книгу: $0 kniga.epub"
    echo "переменные: GPU, SESSION, FRAGMENT, HIGHLIGHT, KEEP_SESSION"
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="$HERE/qwen_tts_book.py"
[ -f "$JOB" ] || { echo "рядом с этим файлом нет qwen_tts_book.py"; exit 1; }
command -v colab >/dev/null || { echo "нет Colab CLI: pip install google-colab-cli"; exit 1; }

# ссылку скачиваем к себе и дальше работаем как с обычным файлом — один путь для всех случаев
if [ "${BOOK#http}" != "$BOOK" ]; then
    FETCHED="$HOME/Downloads/$(basename "${BOOK%%\?*}")"
    echo "== качаю книгу к себе: $FETCHED =="
    curl -L --fail --silent --show-error -o "$FETCHED" "$BOOK" || { echo "не скачалось: $BOOK"; exit 1; }
    BOOK="$FETCHED"
fi
BOOK="$(cd "$(dirname "$BOOK")" && pwd)/$(basename "$BOOK")"
[ -f "$BOOK" ] || { echo "файла нет: $BOOK"; exit 1; }

GPU="${GPU:-T4}"
SESSION="${SESSION:-book}"
STEM="$(basename "${BOOK%.*}")"
[ -n "$STEM" ] || STEM="book"
REMOTE_BOOK="/content/${STEM}.epub"
REMOTE_ZIP="/content/out/${STEM}.zip"
LOCAL_ZIP="$HOME/Downloads/${STEM}.zip"

echo "== прошлые сессии =="
SESSION_LINES="$(colab sessions 2>/dev/null || true)"
printf '%s\n' "$SESSION_LINES"
# формат строки CLI: [имя] endpoint | Hardware: X | Shape: Y | Variant: Z; у чужих сессий имя «?»
printf '%s\n' "$SESSION_LINES" | python3 -c "
import re, sys
for line in sys.stdin:
    match = re.match(r'\[([^\]]*)\]\s+\S+\s+\|', line.strip())
    if match:
        print(match.group(1))
" | while read -r name; do
    if [ "$name" = "?" ]; then
        echo "  ВНИМАНИЕ: в списке есть сессия «?» — чужая для CLI (обычно это открытая вкладка Colab)."
        echo "  Погасить её из терминала нельзя, квоту она держит: colab.research.google.com →"
        echo "  Manage sessions → Terminate, либо закройте вкладку через Runtime → Disconnect."
        continue
    fi
    [ -n "$name" ] || continue
    echo "  останавливаю: $name"
    colab stop -s "$name" >/dev/null 2>&1 || true
done || true

fail() {
    echo
    echo "== не получилось, освобождаю машину =="
    if [ "${KEEP_SESSION:-0}" != "1" ]; then
        colab stop -s "$SESSION" >/dev/null 2>&1 || true
    fi
    exit 1
}

echo
echo "== поднимаю сессию $SESSION ($GPU) =="
colab new -s "$SESSION" --gpu "$GPU" || fail

echo
echo "== книга на машину: $STEM =="
colab upload -s "$SESSION" "$BOOK" "$REMOTE_BOOK" || fail

# Параметры задания уходят на машину вместе с кодом: colab exec аргументы не передаёт,
# поэтому склеиваем маленький стартовый файл (sys.argv + исходник задания).
TMP_BASE="${TMPDIR:-/tmp}"; TMP_BASE="${TMP_BASE%/}"
GENERATED="$(mktemp "$TMP_BASE/qwen-book-XXXXXX.py")"
trap 'rm -f "$GENERATED"' EXIT
python3 - "$GENERATED" "$REMOTE_BOOK" "${FRAGMENT:-}" "${HIGHLIGHT:-}" "$JOB" <<'PY'
import json, pathlib, sys

generated, book, fragment, highlight, job = sys.argv[1:6]
args = [book]
if fragment:
    args += ["--seconds", fragment]
if highlight:
    args += ["--highlight", highlight]
head = ("import sys\n"
        f"sys.argv = {json.dumps(['qwen_tts_book.py', *args])}\n"
        "__name__ = \"__main__\"\n")
pathlib.Path(generated).write_text(head + pathlib.Path(job).read_text(encoding="utf-8"),
                                   encoding="utf-8")
print("задание с параметрами:", " ".join(args), flush=True)
PY

echo
echo "== задание =="
colab exec -s "$SESSION" -f "$GENERATED" || fail

echo
echo "== забираю архив =="
colab download -s "$SESSION" "$REMOTE_ZIP" "$LOCAL_ZIP" || fail
shasum -a 256 "$LOCAL_ZIP" || true
echo "готово: $LOCAL_ZIP"

if [ "${KEEP_SESSION:-0}" = "1" ]; then
    echo "машина оставлена (KEEP_SESSION=1): следующая книга пойдёт быстрее."
    echo "освободить потом: colab stop -s $SESSION"
else
    echo "== освобождаю машину =="
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
fi
