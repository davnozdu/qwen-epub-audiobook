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
#   DRIVE=1             подключить Google Drive и положить book.m4b и read-along epub в
#                       MyDrive/Audiobooks-cli/<имя книги> (по умолчанию 1; 0 — не подключать)
#   DRIVE_DIR=...       папка внутри MyDrive (по умолчанию Audiobooks-cli)
#   DRIVE_WAIT=120      сколько секунд ждать подтверждения Drive, потом продолжить без него
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
[ -f "$BOOK" ] || { echo "файла нет: $BOOK"; exit 1; }
BOOK="$(cd "$(dirname "$BOOK")" && pwd)/$(basename "$BOOK")"

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
STATE="$HOME/.config/colab-cli/sessions.json"

# Имена своих сессий: формат строки CLI — [имя] endpoint | Hardware: X | Shape: Y | Variant: Z
printf '%s\n' "$SESSION_LINES" | python3 -c "
import re, sys
for line in sys.stdin:
    match = re.match(r'\[([^\]]*)\]\s+\S+\s+\|', line.strip())
    if match and match.group(1) != '?':
        print(match.group(1))
" | while read -r name; do
    [ -n "$name" ] || continue
    echo "  останавливаю: $name"
    colab stop -s "$name" >/dev/null 2>&1 || true
done || true

# Чужие сессии (в списке «?») CLI по имени не находит — он ищет только в своём файле состояния.
# Поэтому добавляем запись с известным endpoint: при остановке CLI сам подтянет к ней токен
# с сервера и освободит назначение. Именно такая сессия обычно и держит квоту.
i=0
printf '%s\n' "$SESSION_LINES" | python3 -c "
import re, sys
for line in sys.stdin:
    match = re.match(r'\[\?\]\s+(\S+)\s+\|', line.strip())
    if match:
        print(match.group(1))
" | while read -r endpoint; do
    i=$((i + 1))
    name="orphan-${i}"
    echo "  чужая сессия ${endpoint} → беру как ${name} и останавливаю"
    python3 - "$STATE" "$name" "$endpoint" <<'PY'
import json, pathlib, sys

state, name, endpoint = sys.argv[1:4]
path = pathlib.Path(state)
data = json.loads(path.read_text()) if path.exists() else {}
data[name] = {"name": name, "token": "", "url": "", "endpoint": endpoint,
              "variant": "GPU", "accelerator": "T4", "machine_shape": "STANDARD"}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
    colab stop -s "$name" >/dev/null 2>&1 || true
done || true

if colab sessions 2>/dev/null | grep -q '^\[?\]'; then
    echo "  ВНИМАНИЕ: сессия «?» всё ещё висит. Значит освободить её из терминала не вышло —"
    echo "  завершите вручную: colab.research.google.com → Manage sessions → Terminate."
fi

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

DRIVE="${DRIVE:-1}"
DRIVE_DIR="${DRIVE_DIR:-Audiobooks-cli}"
DRIVE_WAIT="${DRIVE_WAIT:-120}"      # сколько секунд ждать подтверждения Drive, потом едем без него
if [ "$DRIVE" = "1" ]; then
    echo
    echo "== подключаю Google Drive (если появится ссылка — разрешить доступ и нажать Enter) =="
    echo "   ждём ${DRIVE_WAIT} с: не успели — задание поедет без Drive, архив заберём локально"
    colab drivemount -s "$SESSION" &
    MOUNT_PID=$!
    WAITED=0
    while kill -0 "$MOUNT_PID" 2>/dev/null && [ "$WAITED" -lt "$DRIVE_WAIT" ]; do
        sleep 2
        WAITED=$((WAITED + 2))
    done
    MOUNT_STATE="cli-ok"
    if kill -0 "$MOUNT_PID" 2>/dev/null; then
        echo "  за ${DRIVE_WAIT} с подтверждения не было — снимаю попытку, Drive пропускаем."
        pkill -P "$MOUNT_PID" 2>/dev/null || true
        kill "$MOUNT_PID" 2>/dev/null || true
        wait "$MOUNT_PID" 2>/dev/null || true
        MOUNT_STATE="timeout"
    elif ! wait "$MOUNT_PID"; then
        MOUNT_STATE="cli-failed"
    fi

    # Enter без подтверждения доступа: CLI может завершиться «успешно», а монтирования нет —
    # поэтому смотрим на машину, а не на код возврата.
    if [ "$MOUNT_STATE" = "cli-ok" ] \
        && echo 'import os; print("DRIVE_OK", os.path.isdir("/content/drive/MyDrive"))' \
           | colab exec -s "$SESSION" 2>/dev/null | grep -q "DRIVE_OK True"; then
        echo "  Drive подключён: book.m4b и read-along epub поедут в MyDrive/$DRIVE_DIR/<имя книги>"
    else
        [ "$MOUNT_STATE" = "timeout" ] || echo "  Drive не подключился (или доступ не подтверждён)."
        echo "  едем без Drive: архив всё равно заберём в ~/Downloads."
        DRIVE=0
    fi
fi

echo
echo "== книга на машину: $STEM =="
colab upload -s "$SESSION" "$BOOK" "$REMOTE_BOOK" || fail

# Параметры задания уходят на машину вместе с кодом: colab exec аргументы не передаёт,
# поэтому склеиваем маленький стартовый файл (sys.argv + исходник задания).
TMP_BASE="${TMPDIR:-/tmp}"; TMP_BASE="${TMP_BASE%/}"
GENERATED="$(mktemp "$TMP_BASE/qwen-book-XXXXXX.py")"
trap 'rm -f "$GENERATED"' EXIT
python3 - "$GENERATED" "$REMOTE_BOOK" "${FRAGMENT:-}" "${HIGHLIGHT:-}" "$DRIVE" "$DRIVE_DIR" "$JOB" <<'PY'
import json, pathlib, sys

generated, book, fragment, highlight, drive, drive_dir, job = sys.argv[1:8]
args = [book]
if fragment:
    args += ["--seconds", fragment]
if highlight:
    args += ["--highlight", highlight]
if drive == "1":
    args += ["--drive-dir", drive_dir]
else:
    args += ["--no-drive"]
head = ("import sys\n"
        f"sys.argv = {json.dumps(['qwen_tts_book.py', *args])}\n"
        "__name__ = \"__main__\"\n")
pathlib.Path(generated).write_text(head + pathlib.Path(job).read_text(encoding="utf-8"),
                                   encoding="utf-8")
print("задание с параметрами:", " ".join(args), flush=True)
PY

echo
echo "== задание =="
# --timeout у colab exec по умолчанию 30 секунд: на книгу этого мало, ждём до суток.
set +e
colab exec -s "$SESSION" -f "$GENERATED" --timeout 86400
EXEC_CODE=$?
set -e
[ "$EXEC_CODE" = "0" ] || echo "== CLI вернул код $EXEC_CODE — это может быть таймаут ожидания кернела, а не ошибка задания"

echo
echo "== забираю архив =="
# Машину при неудаче НЕ гасим: готовый архив может уже лежать на ней, а после stop он пропадёт.
if colab download -s "$SESSION" "$REMOTE_ZIP" "$LOCAL_ZIP"; then
    shasum -a 256 "$LOCAL_ZIP" || true
    echo "готово: $LOCAL_ZIP"
else
    echo "архива на машине нет (или он ещё собирается). Машину не гашу, чтобы не потерять работу."
    echo "дальше вручную:"
    echo "  colab download -s $SESSION $REMOTE_ZIP ~/Downloads/$(basename "$LOCAL_ZIP")"
    echo "  colab stop -s $SESSION"
    exit 1
fi

if [ "${KEEP_SESSION:-0}" = "1" ]; then
    echo "машина оставлена (KEEP_SESSION=1): следующая книга пойдёт быстрее."
    echo "освободить потом: colab stop -s $SESSION"
else
    echo "== освобождаю машину =="
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
fi
