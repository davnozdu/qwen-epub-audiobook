# Silero v5 как второй движок синтеза

Отдельный скрипт `synthesize_silero_v5.py` делает ту же работу, что `synthesize_qwen_mlx.py`,
но на модели Silero V5. Остальной конвейер не меняется: выход скрипта принимают
`normalize_book_audio.py` → `mix_book_music.py` → `audiobook_epub.py package`
(то есть на выходе те же `book.m4b` и `book-read-along.epub`).

## Что такое Silero V5

Из `models.yml` и README проекта snakers4/silero-models:

| параметр | значение |
|---|---|
| файл модели | `v5_ru.pt`, 145 МБ (лежит одним файлом, `torch.package`) |
| голоса | `aidar`, `baya`, `kseniya`, `eugene`, `xenia` |
| частоты | 8000, 24000, 48000 Гц |
| SSML | да: `<speak>`, `<break time="1500ms"/>`, `<prosody rate/pitch>`, `<p>`, `<s>` |
| ударения | своя нотация: `+` **перед** гласной (`позв+ольте`); наш `U+0301` после гласной модель игнорирует |
| предел на вызов | ~60 с звука (проверено: 60 с проходит, 70 с — отказ «text is probably too long») |
| лицензия | основная лицензия репозитория — **CC-NC-BY (без коммерческого использования)**; исключение — модели `*_cis_base*`, они под MIT |

Про лицензию стоит помнить: `v5_ru.pt` — некоммерческая. Если аудиокниги пойдут в коммерческое
использование, берите `v5_cis_base.pt` (MIT): 92 МБ, тот же интерфейс, и русских голосов там
больше — `ru_igor`, `ru_roman`, `ru_bogdan`, `ru_alexandr`, `ru_eduard`, `ru_zinaida`, `ru_marat`
и другие (всего около 20 `ru_*`), плюс языки СНГ. Переключение одним флагом:

    --model-url https://models.silero.ai/models/tts/ru/v5_cis_base.pt --speaker ru_igor

## Что измерено на этом маке (M2 Pro, CPU, 4 потока)

Прогон книги «Идиот», фрагмент 2 минуты, 40 сегментов:

| | Silero V5 (`v5_ru`, aidar) | Silero V5 (`v5_cis_base`, ru_igor) | Qwen3-TTS 1.7B (для сравнения) |
|---|---|---|---|
| размер модели | 145 МБ | 92 МБ | 4,6 ГБ |
| загрузка модели | 1,25 с | 1,2 с | 14 с (MLX/Metal), 19,5 с (T4) |
| синтез | 137 с звука за **1,8 с** | 149 с звука за 3,8 с | 120–224 с |
| RTF | **0,016** | 0,030 | ≈ 1,0 |
| скорость | **63× реального времени** | 34× | ~1× |
| где работает | CPU (GPU не нужен) | CPU | Metal / CUDA |
| 10-часовая книга | ≈ 10 минут | ≈ 18 минут | ≈ 10 часов |

По скорости разница в десятки раз, и GPU для Silero вообще не требуется.

## Почему всё-таки не замена

Качество и выразительность у Silero V5 ниже: это лёгкая модель для быстрой речи, а не 1.7B
трансформер. Интонация, паузы внутри фраз и «живость» у Qwen3-TTS заметно богаче, а у Силеры
ударения в манифесте она использует, но расставляет их по-своему, если включить `put_accent=True`
(поэтому скрипт по умолчанию ставит `put_accent=False` и берёт ударения из словаря пайплайна).
Слушать и решать — за вами: тестовый фрагмент на 2 минуты обоими движками лежит рядом.

## Как запускать

```bash
# 1. подготовка — как обычно
python colab/build_single_voice_manifest.py kniga.epub -o project          # или prepare_qwen.py run
python audiobook_epub.py prepare --manifest project/manifest.json --out project/source \
    --highlight sentence --start-paragraph p00001 --seconds 300
python prepare_silero.py --manifest project/source/manifest.json \
    --dictionary-dir data/stress --llm-disputes off

# 2. синтез Silero V5 (модель скачается сама в models/silero_v5_ru.pt)
python synthesize_silero_v5.py --manifest project/source/manifest.json --out project/audio \
    --speaker aidar

# 3. дальше без изменений
python normalize_book_audio.py --audio-dir project/audio --out project/audio-normalized
python mix_book_music.py --audio-dir project/audio-normalized --out project/audio-music --volume-percent 5
python audiobook_epub.py package --manifest project/source/manifest.json \
    --audio-dir project/audio-music --out project/result
```

Полезные ключи:

| ключ | смысл |
|---|---|
| `--speaker aidar` | голос по умолчанию (`aidar`, `baya`, `kseniya`, `eugene`, `xenia`) |
| `--voice-map narrator=eugene,dialog=aidar` | разные голоса на роли |
| `--sample-rate 24000` | 8/24/48 кГц (24000 совпадает с остальным конвейером) |
| `--device cpu` | можно попробовать `cuda` (Colab) или `mps` (мак) |
| `--threads 4` | потоки CPU |
| `--stress-marks strip` | озвучить без ударений (по умолчанию переводит наши в `+`) |
| `--model-accents` | разрешить модели ставить ударения самой |
| `--no-reuse` | не переиспользовать уже готовые сегменты |
| `--model-url` | например `https://models.silero.ai/models/tts/ru/v5_cis_base.pt` (MIT) |

Повторный запуск продолжает с места остановки: сегменты с совпадающей подписью берутся из
`audio/segments/*.wav`, в отчёте видно `cached_segments`.

## Проверено

Два полных прогона на фрагменте «Идиот», 2 минуты:

* `v5_ru` / голос `aidar` — синтез 1,8 с (RTF 0,016), нормализация (-24,5 LUFS → -22,25),
  музыка 5 %, `book.m4b` 136,9 с / AAC 24 кГц + `book-read-along.epub`;
* `v5_cis_base` / голос `ru_igor` — синтез 3,8 с (RTF 0,030, 34× реального времени),
  нормализация (-24,1 LUFS → -23,55), тот же результат.

Отчёты синтеза — `audio/benchmark.json` (там же тайминги, подписи сегментов и устройство).
Оба архива для прослушивания: `~/Downloads/idiot-2min-silero.zip` (v5_ru) и
`~/Downloads/idiot-2min-silero-cis.zip` (v5_cis_base, MIT) — рядом лежит `idiot-2min.zip`
от Qwen3-TTS, чтобы сравнить звук в одном месте.

Что не проверено: как это звучит (нужны уши) и поведение на 48 кГц в реальной книге.
