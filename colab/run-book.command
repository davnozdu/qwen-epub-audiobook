#!/bin/bash
# run-book.command — озвучить книгу через Colab CLI в одну команду.
#
#   ./run-book.command ~/Downloads/kniga.epub
#
# Что делает:
#   1. останавливает ВСЕ прошлые сессии (иначе Colab отвечает «Allocation refused …
#      too many active sessions» — старая арендованная машина держит квоту);
#   2. поднимает свежую T4 и запускает задание qwen_tts_book.py с этой книгой;
#   3. забирает готовый zip на локальный диск и освобождает машину.
#
# Настройки через переменные окружения:
#   GPU=L4                 ускоритель (по умолчанию T4)
#   SESSION=book           имя сессии
#   FRAGMENT=300           озвучить только 300 с вместо всей книги
#   HIGHLIGHT=fragment     подсветка по абзацам: сегментов меньше, синтез быстрее
#   KEEP_SESSION=1         не гасить машину в конце (когда идут книги одна за другой:
#                          веса и кэш остаются, но квота занята — потом colab stop -s book)
set -euo pipefail

BOOK="${1:-}"
[ -n "$BOOK" ] || { echo "укажите epub: $0 ~/Downloads/kniga.epub"; exit 1; }
[ -f "$BOOK" ] || { echo "файла нет: $BOOK"; exit 1; }
command -v colab >/dev/null || { echo "нет Colab CLI: pip install google-colab-cli"; exit 1; }

GPU="${GPU:-T4}"
SESSION="${SESSION:-book}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STEM="$(basename "${BOOK%.*}")"
REMOTE="/content/out/${STEM}.zip"
LOCAL="$HOME/Downloads/${STEM}.zip"
STATE="$HOME/.config/colab-cli/sessions.json"

echo "== прошлые сессии =="
colab sessions || true
python3 -c "
import json, pathlib
path = pathlib.Path.home() / '.config/colab-cli/sessions.json'
names = list(json.loads(path.read_text()).keys()) if path.exists() else []
print('\n'.join(names))
" | while read -r name; do
    [ -n "$name" ] || continue
    echo "  останавливаю: $name"
    colab stop -s "$name" || true
done

echo
echo "== запускаю ${STEM} на $GPU (сессия $SESSION) =="
RUN=(colab run --gpu "$GPU" --keep -s "$SESSION" --timeout 3600
     "$HERE/qwen_tts_book.py" "$BOOK")
[ -n "${FRAGMENT:-}" ] && RUN+=(--seconds "$FRAGMENT")
[ -n "${HIGHLIGHT:-}" ] && RUN+=(--highlight "$HIGHLIGHT")
echo "+ ${RUN[*]}"
if ! "${RUN[@]}"; then
    echo
    echo "== запуск не удался, освобождаю машину =="
    colab stop -s "$SESSION" || true
    exit 1
fi

echo
echo "== забираю архив =="
colab download -s "$SESSION" "$REMOTE" "$LOCAL"
shasum -a 256 "$LOCAL" || true
echo "готово: $LOCAL"

if [ "${KEEP_SESSION:-0}" = "1" ]; then
    echo "машина оставлена (KEEP_SESSION=1): веса и кэш на месте, освободить — colab stop -s $SESSION"
else
    echo "== освобождаю машину =="
    colab stop -s "$SESSION" || true
fi
