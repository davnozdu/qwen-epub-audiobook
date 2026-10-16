# Озвучка книги через Colab CLI

Папка самодостаточна: её можно скопировать целиком куда угодно (например в `~/Downloads/QwenTTS-cli`)
и запускать из её корня, ничего дописывать в пути не нужно.

Вся логика — в одном переносимом файле `run-book.py`, а рядом тонкие обёртки под систему:

| система | команда |
|---|---|
| macOS, Linux | `./run-book.command kniga.epub` (то же: `python3 run-book.py kniga.epub`) |
| Windows (cmd) | `run-book.cmd kniga.epub` (то же: `py -3 run-book.py kniga.epub`) |
| Windows (PowerShell) | `py run-book.py kniga.epub` |

Нужен установленный и настроенный Colab CLI (`pip install google-colab-cli` или
`uv tool install google-colab-cli`) и python 3. Книга задаётся как угодно:

    ./run-book.command kniga.epub              # книга в этой папке
    ./run-book.command ../kniga.epub           # или относительным путём
    ./run-book.command https://site.ru/b.epub  # или ссылкой (скачается)
    run-book.cmd C:\Books\kniga.epub           # Windows, абсолютный путь
    run-book.cmd https://site.ru/kniga.epub    # Windows, ссылка

Что произойдёт: обёртка освободит прошлые сессии CLI, поднимет машину с GPU, подключит Google Drive,
положит туда книгу, прогонит весь конвейер (манифест → ударения → голос → синтез → громкость →
музыка → упаковка), заберёт готовый архив в папку загрузок (`~/Downloads`, на Windows
`%USERPROFILE%\Downloads\<книга>.zip`) и освободит машину.

В архиве ровно два файла: `book.m4b` (аудиокнига) и `book-read-along.epub` (текст с подсветкой).

## Настройки (переменными перед командой)

| переменная | смысл |
|---|---|
| `GPU=L4` | другой ускоритель (по умолчанию T4) |
| `SESSION=book` | имя сессии |
| `FRAGMENT=300` | озвучить только 300 секунд вместо всей книги |
| `HIGHLIGHT=fragment` | подсветка по абзацам: сегментов меньше, синтез быстрее |
| `KEEP_SESSION=1` | не гасить машину в конце: следующая книга пойдёт без установки пакетов и загрузки весов |
| `DRIVE=1` | подключить Google Drive и положить `book.m4b` и read-along epub в `MyDrive/Audiobooks-colab/<имя книги>` (по умолчанию 1; `DRIVE=0` — не подключать) |
| `DRIVE_DIR=...` | другая папка внутри `MyDrive` (по умолчанию `Audiobooks-colab`) |
| `DRIVE_WAIT=120` | сколько секунд ждать подтверждения Drive, потом продолжить без него |

Примеры:

    FRAGMENT=300 ./run-book.command kniga.epub          # быстрая проверка на пяти минутах
    HIGHLIGHT=fragment ./run-book.command kniga.epub    # экономия ~25 % времени на книге
    KEEP_SESSION=1 ./run-book.command kniga1.epub && KEEP_SESSION=1 ./run-book.command kniga2.epub
    DRIVE=0 ./run-book.command kniga.epub               # только локальное скачивание, без Drive

На Windows то же самое, только переменные ставятся через `set` (cmd) или `$env:` (PowerShell):

    set FRAGMENT=300 & run-book.cmd C:\Books\kniga.epub
    set GPU=L4 & run-book.cmd kniga.epub
    $env:DRIVE=0; py run-book.py kniga.epub

## Google Drive

Отдельная авторизация не нужна: обёртка один раз за сессию делает `colab drivemount -s book` —
это та же учётная запись, под которой работает машина. Если Google просит подтверждение, CLI
печатает ссылку: открыть её, разрешить доступ и нажать Enter в терминале.

Обёртка ждёт подтверждения `DRIVE_WAIT` секунд (по умолчанию 120). Дальше события такие:

| что произошло | куда попадёт результат |
|---|---|
| Drive подключён | `MyDrive/Audiobooks-colab/<имя книги>/` — `book.m4b` и read-along epub **под именем книги**, и zip в `~/Downloads` |
| нажали Enter без подтверждения доступа | Drive пропускаем, zip в `~/Downloads` (книга не теряется) |
| за 120 секунд никто не ответил | попытку снимаем, Drive пропускаем, zip в `~/Downloads` |
| `colab drivemount` упал | Drive пропускаем, zip в `~/Downloads` |

Имя книги берётся из epub: файл `idiot-part1.epub` ляжет как
`Audiobooks-colab/idiot-part1/idiot-part1.m4b` и `Audiobooks-colab/idiot-part1/idiot-part1-read-along.epub`.

Подключить вручную (например, к уже запущенной машине) и положить файлы на Drive без обёртки:

    colab drivemount -s book
    colab exec -s book -f qwen_tts_book.py      # задание само скопирует m4b и epub

`DRIVE=0` (или `--no-drive` в задании) отключает Drive совсем.

## Файлы

| файл | что это |
|---|---|
| `run-book.py` | вся обёртка: сессии, машина, Drive, upload, запуск, download, stop (macOS, Linux, Windows) |
| `run-book.command` | запуск в один клик на macOS/Linux (передаёт работу `run-book.py`) |
| `run-book.cmd` | запуск в один клик на Windows (то же) |
| `qwen_tts_book.py` | само задание (книга аргументом; без аргумента берёт единственный epub в `/content`) |
| `qwen_tts_book_cli.ipynb` | то же задание блокнотом: параметры в первой ячейке, для `colab exec -f` |
| `notebooks/` | браузерные блокноты: полная книга, смоук RTF, проверка всего, план Б на torch |

## Системы

| | что нужно | чем запускать |
|---|---|---|
| macOS | python 3, Colab CLI | `./run-book.command kniga.epub` |
| Linux | python 3, Colab CLI | `./run-book.command kniga.epub` или `python3 run-book.py kniga.epub` |
| Windows 10/11 | python 3 (python.org), Colab CLI | `run-book.cmd kniga.epub` или `py -3 run-book.py kniga.epub` |

Работает одинаково: путь к CLI ищется в `PATH`, потом в `~/.local/bin` и в папках `uv tool install`
(на Windows — `%USERPROFILE%\.local\bin`, `%APPDATA%\uv\tools\...`). Хеш архива считается через
`hashlib`, зависшее подключение Drive глушится через `kill` (POSIX) или `taskkill /T` (Windows).

Локальный синтез на своём компьютере (`--local`, MLX/Metal) возможен только на macOS —
он не входит ни в `run-book.py`, ни в обёртки поверх Colab. Windows и Linux работают как
пульт для Colab: машину поднимает Colab, там всё и считается.

## Если не поднимается машина

`Allocation refused (precondition failed)` — квоту держит другая сессия. Обёртка гасит те, что
знает CLI. Сессия с именем `?` в `colab sessions` — чужая для CLI (обычно открытая вкладка Colab):
её можно завершить только в браузере, `colab.research.google.com` → Manage sessions → Terminate.
Если сессий нет, а отказ повторяется — это лимит на ускоритель: подождите или возьмите `GPU=L4`.

## Вручную, без обёртки

    colab new -s book --gpu T4
    colab drivemount -s book                       # необязательно: m4b и epub лягут в MyDrive/Audiobooks-colab
    colab upload -s book kniga.epub /content/kniga.epub
    colab exec -s book -f qwen_tts_book.py
    colab download -s book /content/out/kniga.zip ~/Downloads/
    colab stop -s book
