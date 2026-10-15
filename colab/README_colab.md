# Смоук-тест Qwen3-TTS в Colab (T4) — что открывать и как читать результат

## Файлы

| файл | что это |
|---|---|
| `qwen_tts_colab_full_book.ipynb` | **вся книга**: EPUB → манифест → ударения (словарь Supertonic + Silero) → синтез MLX/CUDA → музыка → **`book.m4b` + `book-read-along.epub`** |
| `qwen_tts_colab_check_all.ipynb` | один Run all: окружение, установка, правки репозитория с проверкой, RTF |
| `qwen_tts_colab_smoke_mlx.ipynb` | только замер RTF на MLX |
| `qwen_tts_colab_smoke_torch.ipynb` | план Б: torch/CUDA через официальный `qwen-tts` |
| `build_single_voice_manifest.py` | EPUB → манифест с одним чтецом (без облачной LLM), проверки — штатные из репозитория |
| `../resources/voices/narrator.wav` + `.json` | образец голоса 9,4 с (24 кГц, моно) и его точная расшифровка — ноутбук качает его сам по raw-ссылке |
| `../resources/Lamplight_and_Paper.mp3` | фоновая музыка, уже в репозитории (проверяется по sha256 в `mix_book_music.py`) |

## Как запустить

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
