# Qwen EPUB Audiobook

Подготовка русских EPUB по ролям, локальная озвучка Qwen3-TTS 1.7B на Apple Silicon,
отдельная аудиокнига M4B и EPUB 3 со встроенным аудио и подсветкой текущего предложения.

В репозитории только исходники и тесты. Книги, аудио, веса моделей, кэши,
окружения Python и API-ключи не публикуются. Проект вырос из работы над
[mytts-books](https://github.com/davnozdu/mytts-books).

## Как работает

1. `prepare_qwen.py`: LLM определяет говорящих, пол, возраст и индивидуальные
   описания голосов. Нарезка сохраняет текст, роли и привязки к исходному EPUB.
2. `audiobook_epub.py prepare`: создаёт отдельный EPUB и его манифест. Режим
   `--highlight sentence` дополнительно делит текст на предложения через Razdel.
3. `synthesize_qwen_mlx.py`: VoiceDesign создаёт по одному образцу каждой роли,
   выгружается и освобождает кэш, затем Base озвучивает весь текст по образцам.
4. `audiobook_epub.py package`: EbookLib собирает EPUB Media Overlays, FFmpeg — M4B
   с главами. SMIL связывает текст с реально измеренными интервалами аудио.

Пол и голос — разные задачи: корректная метка `female` не гарантирует звучание.
Образцы и результат нужно прослушать. Для каждой роли свои описание и seed;
постоянство обеспечивается одним аудиообразцом на роль внутри запуска.

## Установка на macOS / M2 Pro

Проверено на M2 Pro, 16 ГБ памяти, Python 3.11 arm64. Требуется доступ к Metal.

```sh
brew install python@3.11 ffmpeg
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Скачайте два закреплённых 8-битных checkpoint (всего примерно 6,2 ГБ):

```sh
.venv/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --revision f90d617701d9f7f4ca499291e0b57f2b3c2fd2ee \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
.venv/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --revision e7dd0585652209fa0d7783659aad4e8a324de11c \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

Скрипты используют `/opt/homebrew/bin/ffmpeg` и `ffprobe` (Homebrew arm64).
Лимит MLX — 8 ГиБ, кэша — 512 МиБ. Настройки памяти GPU macOS не изменяются.
Во время синтеза сетевые загрузки моделей отключены.

## Подготовка своей книги

```sh
.venv/bin/python prepare_qwen.py run book.epub -o work/prepared \
  --model deepseek-v4.1-flash --no-think
```

Ключ Ollama вводится скрыто при запросе либо передаётся через `OLLAMA_API_KEY`.
Не сохраняйте ключ в репозитории. Подготовка отправляет текст книги в облачную
LLM — используйте этот режим только для книг, которые разрешено туда передавать.
Модель и доступность облачного сервиса определяются вашей учётной записью.
Роли можно исправить через `--cast` и `--roles-overrides`.
Есть файловый возобновляемый режим `advance` и проверка `check`.

## Пять минут как самостоятельная мини-книга

```sh
.venv/bin/python audiobook_epub.py prepare \
  --manifest work/prepared/manifest.json --seconds 300 \
  --start-paragraph p00001 --highlight sentence --out work/source
./run.command
```

Без предыдущего таймлайна длительность оценивается по 14 символам/с. При наличии
измерений можно добавить `--timing previous/audio/timeline.json`. Вырезаются только
полные абзацы. Новое аудио может оказаться длиннее или короче пяти минут.
Мини-книга озвучивается целиком (`--whole-book`), а не обрывается по таймеру.

## Вся книга

```sh
.venv/bin/python audiobook_epub.py prepare \
  --manifest work/prepared/manifest.json --highlight sentence --out work/source
./run.command
```

Можно использовать отдельный каталог проекта: `./run.command another-project`,
если в нём есть `source/manifest.json`. Одна команда запускает синтез, затем
упаковку; при ошибке синтеза упаковка не выполняется.

Результаты в `work/result/`:

- `book.m4b` — аудиокнига AAC 128 кбит/с с метаданными и главами;
- `book-read-along.epub` — текст, встроенное аудио и подсветка;
- `package-report.json` — результаты проверки.

В `work/audio/` сохраняются WAV фрагментов, полный WAV/MP3, `timeline.json`,
`benchmark.json`. `work/run.log` — общий журнал. Повторный запуск обновляет результаты.
Для прослушивания EPUB с подсветкой подходит [Thorium Reader](https://www.thoriumreader.com/).

## Подсветка и память

`--highlight sentence` — режим по умолчанию. Каждое предложение озвучивается
отдельно, подсвечивается по реальным границам WAV. Если предложение пересекает
границу роли или прежнего лимита длины TTS, оно остаётся разделённым на части.
`--highlight fragment` сохраняет прежние более крупные фрагменты.
Подсветка отдельных слов не реализована: для неё нужно отдельное выравнивание.

Launcher использует временные образцы (`--ephemeral-voices`): они создаются заново
и удаляются после успеха, ошибки или обычного прерывания. Старые библиотеки не
импортируются. VoiceDesign и Base не загружаются одновременно; при выходе модели
освобождаются и MLX-кэш очищается. Завершение процесса возвращает оставшиеся
ресурсы ОС. SIGKILL/отключение питания могут оставить временные файлы.
Для постоянной библиотеки при прямом вызове генератора уберите `--ephemeral-voices`.

## Проверки и ограничения

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Проверяются сохранность текста/пробелов/ролей, границы предложений, таймлайн,
встроенное аудио, SMIL, CSS подсветки и реальная сборка тестового M4B через FFmpeg.
MLX-тесты используют имитацию GPU; качество голоса требует отдельного аудиопрогона.
Это не заменяет EPUBCheck и проверку в конкретной читалке.

Исходный EPUB не меняется. Создаётся редакция с простым оформлением, без исходных
обложек, шрифтов и внутритекстовой разметки. Поддерживаются текстовые листовые блоки;
неподдерживаемая структура останавливает подготовку, чтобы не потерять текст.
Ударения и отдельный нормализатор не добавлены. EPUB Media Overlays должны
поддерживаться читалкой; обычная читалка покажет только текст.
Аудио полной книги собирается потоково с диска, без удержания всей записи в RAM.

Старый EPUB без подключённого CSS можно исправить без повторной озвучки:

```sh
.venv/bin/python audiobook_epub.py repair-highlight \
  --epub old.epub --out highlighted.epub
```

Основа: [EbookLib](https://github.com/aerkalov/ebooklib),
[MLX-Audio](https://github.com/Blaizzy/mlx-audio),
[Qwen Design then Clone](https://github.com/QwenLM/Qwen3-TTS#voice-design-then-clone),
[Razdel](https://github.com/natasha/razdel), [FFmpeg](https://github.com/FFmpeg/FFmpeg).
См. [THIRD_PARTY.md](THIRD_PARTY.md) перед распространением производных проектов.
