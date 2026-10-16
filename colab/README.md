# Озвучка книги через Colab CLI

Всё в этой папке, запускается из её корня, ничего дописывать в пути не нужно:

    ./run-book.command kniga.epub              # книга в этой папке
    ./run-book.command ../kniga.epub           # или относительным путём
    ./run-book.command https://site.ru/b.epub  # или ссылкой (скачается)

Что произойдёт: обёртка освободит прошлые сессии CLI, поднимет машину с GPU, положит туда книгу,
прогонит весь конвейер (манифест → ударения → голос → синтез → громкость → музыка → упаковка),
заберёт готовый архив в `~/Downloads/<книга>.zip` и освободит машину.

В архиве ровно два файла: `book.m4b` (аудиокнига) и `book-read-along.epub` (текст с подсветкой).

## Настройки (переменными перед командой)

| переменная | смысл |
|---|---|
| `GPU=L4` | другой ускоритель (по умолчанию T4) |
| `SESSION=book` | имя сессии |
| `FRAGMENT=300` | озвучить только 300 секунд вместо всей книги |
| `HIGHLIGHT=fragment` | подсветка по абзацам: сегментов меньше, синтез быстрее |
| `KEEP_SESSION=1` | не гасить машину в конце: следующая книга пойдёт без установки пакетов и загрузки весов |

Примеры:

    FRAGMENT=300 ./run-book.command kniga.epub          # быстрая проверка на пяти минутах
    HIGHLIGHT=fragment ./run-book.command kniga.epub    # экономия ~25 % времени на книге
    KEEP_SESSION=1 ./run-book.command kniga1.epub && KEEP_SESSION=1 ./run-book.command kniga2.epub

## Файлы

| файл | что это |
|---|---|
| `run-book.command` | обёртка: сессии, машина, upload, запуск, download, stop |
| `qwen_tts_book.py` | само задание (книга аргументом; без аргумента берёт единственный epub в `/content`) |
| `qwen_tts_book_cli.ipynb` | то же задание блокнотом: параметры в первой ячейке, для `colab exec -f` |
| `notebooks/` | браузерные блокноты: полная книга, смоук RTF, проверка всего, план Б на torch |

## Если не поднимается машина

`Allocation refused (precondition failed)` — квоту держит другая сессия. Обёртка гасит те, что
знает CLI. Сессия с именем `?` в `colab sessions` — чужая для CLI (обычно открытая вкладка Colab):
её можно завершить только в браузере, `colab.research.google.com` → Manage sessions → Terminate.
Если сессий нет, а отказ повторяется — это лимит на ускоритель: подождите или возьмите `GPU=L4`.

## Вручную, без обёртки

    colab new -s book --gpu T4
    colab upload -s book kniga.epub /content/kniga.epub
    colab exec -s book -f qwen_tts_book.py
    colab download -s book /content/out/kniga.zip ~/Downloads/
    colab stop -s book
