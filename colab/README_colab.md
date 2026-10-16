# Смоук-тест Qwen3-TTS в Colab (T4) — что открывать и как читать результат

## Файлы

| файл | что это |
|---|---|
| `cli/run-book.command` | **одна команда для Colab CLI**: `./run-book.command kniga.epub` — поднимает машину, заливает книгу, гонит конвейер, кладёт zip в ~/Downloads, освобождает машину |
| `cli/qwen_tts_book.py` | задание, которое запускает обёртка (книга аргументом или единственный epub в /content) |
| `cli/qwen_tts_book_cli.ipynb` | то же задание блокнотом: параметры в первой ячейке, для `colab exec -f` |
| `qwen_tts_colab_full_book.ipynb` | **вся книга в браузере**: EPUB → манифест → ударения (словарь Supertonic + Silero) → синтез MLX/CUDA → музыка → **`book.m4b` + `book-read-along.epub`** |
| `qwen_tts_colab_check_all.ipynb` | один Run all: окружение, установка, правки репозитория с проверкой, RTF |
| `qwen_tts_colab_smoke_mlx.ipynb` | только замер RTF на MLX |
| `qwen_tts_colab_smoke_torch.ipynb` | план Б: torch/CUDA через официальный `qwen-tts` |
| `build_single_voice_manifest.py` | EPUB → манифест с одним чтецом (без облачной LLM), проверки — штатные из репозитория |
| `../resources/voices/narrator.wav` + `.json` | образец голоса 9,4 с (24 кГц, моно) и его точная расшифровка — ноутбук качает его сам по raw-ссылке |
| `../resources/Lamplight_and_Paper.mp3` | фоновая музыка, уже в репозитории (проверяется по sha256 в `mix_book_music.py`) |

## Как запустить

### Из терминала, через Colab CLI (без браузера)

Папка `colab/cli/` — самодостаточный комплект: обёртка `run-book.command`, задание
`qwen_tts_book.py` и блокнот `qwen_tts_book_cli.ipynb`. Скопируйте её куда угодно и запускайте
из её корня — путей внутри нет, книга задаётся аргументом:

```bash
pip install google-colab-cli                 # один раз
cd colab/cli
./run-book.command kniga.epub
```

Обёртка сама освобождает прошлые сессии CLI, поднимает машину с GPU, кладёт туда книгу,
запускает конвейер, забирает `/content/out/<книга>.zip` в `~/Downloads` и гасит машину.
Переменные: `GPU=L4`, `SESSION=book`, `FRAGMENT=300`, `HIGHLIGHT=fragment`, `KEEP_SESSION=1`.

Если хочется вручную, шаг за шагом:

```bash
colab new -s book --gpu T4
colab upload -s book ~/Downloads/kniga.epub /content/kniga.epub
colab exec -s book -f colab/cli/qwen_tts_book.py --timeout 86400
colab download -s book /content/out/kniga.zip ~/Downloads/
colab stop -s book
```

Или одной строкой, книга сразу за именем файла (скриптовая форма; для книги по ссылке):

```bash
colab run --gpu T4 --keep -s book --timeout 3600 colab/cli/qwen_tts_book.py "https://site.ru/kniga.epub"
colab download -s book /content/out/kniga.zip ~/Downloads/
colab stop -s book
```

Во всех случаях на выходе `/content/out/<имя книги>.zip` c `book.m4b` и `book-read-along.epub`
внутри; путь, размер и sha256 печатаются в конце. Книга длиннее 7 часов режется на части заранее
(в обработку идёт первая, остальные уезжают на Drive в папку `Audiobooks-shrink` и сохраняются
локально — их обрабатывают отдельными запусками; порог меняется через `SHRINK_HOURS`, включая 0).
Обёртка подключает Google Drive и кладёт
`book.m4b` и read-along epub — под именем книги — в `MyDrive/Audiobooks-colab/<имя книги>/`
(той же учётной записью, без отдельной авторизации: один раз за сессию `colab drivemount`).
Ждёт подтверждения 120 с (`DRIVE_WAIT`): не дождалась, нажали Enter без доступа или монтирование
упало — Drive пропускается, zip всё равно скачивается в `~/Downloads`. `DRIVE=0` отключает Drive,
`DRIVE_DIR=...` меняет папку. Каталог проекта у каждой книги свой, поэтому кэши сегментов
разных книг не перемешиваются.

### В браузере

1. Загрузите ноутбук в Colab (File → Upload notebook) и включите **Runtime → Change runtime type → T4 GPU**.
2. `qwen_tts_colab_smoke_mlx.ipynb` → Run all → в диалоге выберите `~/Downloads/colab-ref/narrator.wav`.
3. В конце получите `metrics.json`, а в выводе — строку `RTF` и вердикт.

## Как читать вердикт

RTF = секунды синтеза на секунду звука. Часы синтеза книги = часы звука × RTF.

| RTF | вердикт |
|---|---|
| < 0,3 | отлично, полный пайплайн в Colab имеет смысл |
| 0,3–0,7 | рабочо: книга 10 ч ≈ 3–7 ч, в сессию укладывается |
| 0,7–1,0 | на грани: делить книгу на несколько запусков (resume есть) |
| > 1,0 | T4 не годится: A100/L4 (Colab Pro) или план Б |

Это единственный неизвестный параметр переноса. Всё остальное уже проверено:

- `mlx[cuda]==0.32.3` на Linux тянет `mlx-cuda-12==0.32.3` (`manylinux_2_35_x86_64`) + nvidia-колёса;
  под Python 3.12 Colab есть готовое колесо `mlx-0.32.3-cp312-cp312-manylinux_2_35_x86_64.whl` — сборки из исходников нет;
- синтез-код бэкенд-агностичен: `load_model(..., allow_patterns=MODEL_FILES)` → `load_audio(path, sample_rate)` →
  `model.generate(text, lang_code="russian", ref_audio, ref_text, temperature)`;
- модели 4,6 ГБ fp16 + CUDA-рантайм помещаются в ~15 ГБ T4;
- ffmpeg 4.4.2 в Colab уже в `/usr/bin/ffmpeg`.

## Пять правок для полного пайплайна (когда смоук-тест пройдёт)

1. `requirements.txt` / `requirements-mlx.txt`: `mlx-metal==0.32.3` → `mlx[cuda]==0.32.3`
   (иначе `pip install -r requirements.txt` на Linux падает сразу).
2. Пять хардкодов `/opt/homebrew/bin/ffmpeg|ffprobe` → `/usr/bin`:
   `audiobook_epub.py:25-26`, `normalize_book_audio.py:12`, `mix_book_music.py:11`, `synthesize_qwen_mlx.py:435`
   → `sed -i 's#/opt/homebrew/bin/#/usr/bin/#g' <файлы>`.
3. `setup_audiobook.py` не запускать: это установщик для macOS (venv, brew, guard на Apple Silicon).
   Модели забирает HF-кэш по `design_model_files.json` / `stress_model_files.json`.
4. `prepare_qwen.py:485`: `OLLAMA_API_KEY` подавать через env, а не `getpass` — в Colab нет stdin, ноутбук зависнет.
5. `START.command` / `TEST-7MIN.command` не использовать: они зовут `/opt/homebrew/opt/python@3.11`.
   Скрипты запускать напрямую: `prepare_silero.py` → `prepare_qwen.py` → `synthesize_qwen_mlx.py` →
   `normalize_book_audio.py` → `mix_book_music.py`.

Выход держать на Google Drive: 12-часовой сессии на книгу не хватит, а `progress.json` + sha256 сегментов
позволяют продолжить синтез в следующем запуске.

## Если синтез падает на CUDA: «Cache thrashing»

На машине с GPU генератор останавливается посреди работы с текстом

    Cache thrashing is happening, please set the environment variable MLX_CUDA_CONV_CACHE_SIZE
    to a larger value than 128 to fix degraded performance.

Важно: у этих кэшей **единица измерения — записи, а не мегабайты**, и по умолчанию они крошечные
(`docs ml-explore/mlx`, `environment_variables.rst`):

| переменная | по умолчанию | у нас |
|---|---|---|
| `MLX_CUDA_CONV_CACHE_SIZE` (свёртки) | 128 | 12288 |
| `MLX_CUDA_GRAPH_CACHE_SIZE` (графы) | 400 | 8192 |
| `MLX_CUDA_SDPA_CACHE_SIZE` (внимание) | 256 | 4096 |
| `MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE` | 64 | 1024 |
| `MLX_CUDA_FFT_CACHE_SIZE` | 128 | 1024 |

Ячейка «Окружение» и `colab_run_all.py` ставят эти потолки до старта любого синтеза, а ячейка синтеза
передаёт их дочернему процессу явно. Потолок — это лимит на число записей, а не резервация памяти,
поэтому модели (~4,6 ГБ) и кэшам на ~15 ГБ T4 места хватает.

Если «Cache thrashing» всё же случится, синтез повторяется автоматически: сначала с удвоенными
потолками, затем с учетверёнными, а последней ступенью выключается сама проверка
(`MLX_ENABLE_CACHE_THRASHING_CHECK=0`) — лучше закончить книгу с просадкой скорости, чем потерять
прогон. Повтор продолжает с места остановки: уже озвученные сегменты берутся из кэша, GPU впустую
не тратится.

Имя переменной и потолок, которого не хватило, разбираются прямо из сообщения MLX, поэтому в логе
видно, на чём именно упало и с чего продолжаем.

Замер на T4 по реальному прогону фрагмента: загрузка модели ≈ 19,5 с, дальше RTF ≈ 0,9–1,2 по сегментам,
то есть примерно реалтайм. 10-часовая книга ≈ 10 часов GPU: на бесплатном тарифе это несколько сессий,
выход обязательно на Google Drive — синтез продолжится по кэшу сегментов с места остановки.

## Порядок стадий, который нельзя менять

1. `prepare` идёт **до** библиотеки голосов: профиль роли (`library_profile`) привязан к
   `book.input_sha256`, а `prepare` перезаписывает это поле хешем собранного `book-text.epub`.
   Библиотека, собранная раньше, даёт `ValueError: Описание роли narrator изменилось` уже на синтезе.
2. `synthesize` всегда получает `--whole-book`: границу фрагмента задаёт `prepare --seconds`. Без этого
   флага синтез останавливается по своей оценке длительности и упаковка падает с «Не озвучена вся книга».
3. Повторный запуск не пересобирает подготовку, если книга та же (`origin_book.input_sha256`) и длина
   фрагмента не менялась: иначе меняется хеш `book-text.epub`, а он входит в подпись каждого сегмента —
   кэш синтеза обнуляется. Пересобрать принудительно: `REBUILD = True` в ячейке настроек.

## Скорость: почему короткий тест обманчив

RTF зависит от длины сегмента — у каждого вызова синтеза есть постоянные накладные:

| длина сегмента | RTF |
|---|---|
| до 2 с | ≈ 1,4 |
| 2–6 с | ≈ 1,1 |
| больше 6 с | ≈ 0,9 |

Плюс разовая плата: загрузка модели (≈ 20 с) и прогрев первого вызова (на CUDA ≈ 70 с). На двухминутной
книге всё это не размазывается, поэтому цифра там пессимистичная. Ориентиры для книги: 5 минут звука ≈
5–6 минут GPU, 2 часа звука ≈ 2 ч 20 мин, 10 часов ≈ 11–12 ч. Если нужен быстрее и не жалко подсветки —
`prepare --highlight fragment`: сегментов в разы меньше, значит меньше накладных (подсветка станет
поабзацной).

## Несколько книг в одной сессии

Заново ставить зависимости и качать веса не нужно: пропустите ячейки установки, клона и весов и
начинайте с загрузки EPUB. Либо очередью, одной командой на книгу:

    !cd /content/qwen-epub-audiobook && python colab/colab_run_all.py "/content/book2.epub" \
        --out /content/audiobook2 --model-dir /content/models/Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source \
        --skip-install --skip-gpu-check

Между книгами перезагружается только модель (≈ 20 с). Каждой книге — свой `--out`: id сегментов
одинаковые (seg000001, seg000002…), в общем каталоге кэши двух книг перемешаются.
