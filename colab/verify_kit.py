#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка всей colab-обвязки на СВЕЖЕМ клоне репозитория.

Запуск:

    python colab/verify_kit.py --book "~/Downloads/skuki.epub" \
        --model-dir "$HOME/Library/Application Support/Audiobook Studio/Engine/models/Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source"

Без --book проверяются только статические вещи и запуск скриптов; с --book и --model-dir добавляется
сквозной прогон до стадии ударений (и те же стадии внутри ноутбука).

Что именно ловит (класс ошибок, которые уже случались):

  1. импорт модулей репозитория из colab/ — sys.path[0] у скрипта это его собственная папка;
  2. запуск по абсолютному пути из чужого рабочего каталога и с посторонним PYTHONPATH / PYTHONSAFEPATH;
  3. забытый PYTHONPATH у дочерних процессов;
  4. жёсткие пути /content/... там, где ожидается рабочий каталог (и наоборот — загрузка файла «не туда»);
  5. assert по repr-строке устройства вместо проверки смысла;
  6. ячейки ноутбука, которые не компилируются (переносы строк в магиях);
  7. отсутствие обязательных элементов (образец голоса, музыка, словарь, поиск файла в cwd).

Код возврата 1, если хоть одна проверка не прошла.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import types

PY = sys.executable
PROBLEMS: list[str] = []
REPO_URL = "https://github.com/davnozdu/qwen-epub-audiobook.git"


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


def check(condition: bool, message: str) -> None:
    print(("  ✔ " if condition else "  ✘ ") + message)
    if not condition:
        PROBLEMS.append(message)


def sh(cmd, cwd, env=None, timeout=900):
    environment = dict(os.environ if env is None else env)
    environment.pop("PYTHONPATH", None)
    done = subprocess.run([str(part) for part in cmd], cwd=str(cwd), text=True,
                          capture_output=True, timeout=timeout, env=environment)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def compile_notebook(path: pathlib.Path) -> list[str]:
    """Компилирует кодовые ячейки, выкидывая магии и их продолжения."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, cell in enumerate(cells):
        keep, skip = [], False
        for line in "".join(cell["source"]).splitlines():
            if skip or line.lstrip().startswith(("%", "!")):
                skip = line.rstrip().endswith("\\")
                continue
            keep.append(line)
        try:
            compile("\n".join(keep), f"{path.name}:{index}", "exec")
        except SyntaxError as exc:
            bad.append(f"ячейка {index}: {exc}")
    return bad


NOTEBOOK_BUILTINS = {"__name__", "get_ipython", "display", "In", "Out", "_ih", "_oh", "_dh"}


def _code_without_magics(cell: dict) -> str:
    keep, skip = [], False
    for line in "".join(cell["source"]).splitlines():
        if skip or line.lstrip().startswith(("%", "!")):
            skip = line.rstrip().endswith("\\")
            continue
        keep.append(line)
    return "\n".join(keep)


def _bind_names(tree: ast.AST, defined: set[str]) -> None:
    """Добавляет в defined всё, что ячейка связывает: импорты, присваивания, аргументы функций."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                defined.add(node.name)
            node_args = node.args
            for arg in list(node_args.posonlyargs) + list(node_args.args) + list(node_args.kwonlyargs):
                defined.add(arg.arg)
            if node_args.vararg:
                defined.add(node_args.vararg.arg)
            if node_args.kwarg:
                defined.add(node_args.kwarg.arg)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                defined.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.For):
            defined.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    defined.update(n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.comprehension):
            defined.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))


def undefined_names(path: pathlib.Path) -> list[str]:
    """Имена, которые ячейка использует, но ни она, ни предыдущие не определяют.

    Ловит забытый `import json` посреди ноутбука: в Colab это NameError, а локальный прогон ячеек
    с подложенным пространством имён его не видит. Порядок важен: ячейки идут сверху вниз.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    defined = set(dir(builtins)) | NOTEBOOK_BUILTINS
    problems = []
    for index, cell in enumerate(c for c in notebook["cells"] if c["cell_type"] == "code"):
        source = _code_without_magics(cell)
        tree = ast.parse(source or "pass")
        _bind_names(tree, defined)
        missing = sorted({node.id for node in ast.walk(tree)
                          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined})
        if missing:
            problems.append(f"ячейка {index}: нет импорта для {', '.join(missing)}")
    return problems


def unguarded_versions(path: pathlib.Path) -> list[str]:
    """Вызовы версии пакета вне try/except.

    Такой вызов в конце синтеза убивает уже готовое аудио: на Linux нет mlx-metal,
    и `importlib.metadata.version("mlx-metal")` поднимает PackageNotFoundError после
    всей работы. Версии — это строчка в отчёте, падать из-за неё нельзя.
    """
    source = path.read_text(encoding="utf-8")
    if "metadata" not in source:
        return []
    tree = ast.parse(source)
    imports_alias = "from importlib.metadata import version" in source
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                guarded.add(id(child))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in guarded:
            continue
        name = ast.unparse(node.func)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "version" and "metadata" in name) \
                or (imports_alias and name == "version"):
            bad.append(f"строка {node.lineno}: {name}(…)")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка colab-обвязки на свежем клоне репозитория")
    parser.add_argument("--book", help="epub для сквозного прогона (без него — только статические проверки)")
    parser.add_argument("--model-dir", help="готовая папка с весами модели синтеза")
    parser.add_argument("--clone-dir", default="/tmp/verify-clone", help="куда клонировать репозиторий")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--no-clone", action="store_true", help="использовать уже склонированный --clone-dir")
    args = parser.parse_args()

    clone = pathlib.Path(args.clone_dir).expanduser().resolve()

    section("0. Свежий клон опубликованного репозитория")
    if args.no_clone:
        check(clone.joinpath("colab/colab_run_all.py").is_file(), f"в {clone} есть colab/colab_run_all.py")
    else:
        shutil.rmtree(clone, ignore_errors=True)
        code, out = sh(["git", "clone", "-q", args.repo_url, str(clone)], pathlib.Path("/tmp"))
        check(code == 0, f"клон: {out.strip()[:140] or 'готово'}")
    if (clone / ".git").is_dir():
        print("  версия:", sh(["git", "log", "--oneline", "-1"], clone)[1].strip())
    for rel in ("colab/colab_run_all.py", "colab/build_single_voice_manifest.py",
                "colab/qwen_tts_colab_full_book.ipynb", "resources/voices/narrator.wav",
                "resources/voices/narrator.json", "resources/Lamplight_and_Paper.mp3",
                "resources/stress_dictionary.json"):
        check((clone / rel).is_file(), f"в репозитории есть {rel}")

    section("1. Каждый скрипт запускается отдельным процессом")
    for name in ("audiobook_epub.py", "normalize_book_audio.py", "mix_book_music.py", "synthesize_qwen_mlx.py",
                 "prepare_qwen.py", "prepare_silero.py", "prepare_speech.py"):
        code_root, out_root = sh([PY, name, "--help"], clone)
        check(code_root == 0 and "Traceback" not in out_root, f"{name}: из корня репозитория")
        code_far, out_far = sh([PY, clone / name, "--help"], pathlib.Path("/tmp"))
        check(code_far == 0 and "Traceback" not in out_far, f"{name}: по абсолютному пути из /tmp")
    for name in ("build_single_voice_manifest.py", "colab_run_all.py"):
        path = clone / "colab" / name
        for cwd, label in ((clone, "из корня репозитория"), (pathlib.Path("/tmp"), "из /tmp")):
            arg = f"colab/{name}" if cwd == clone else path
            code_n, out_n = sh([PY, arg, "--help"], cwd)
            check(code_n == 0 and "Traceback" not in out_n, f"colab/{name}: {label}")
        code_s, _ = sh([PY, path, "--help"], clone, env=dict(os.environ, PYTHONPATH=str(clone), PYTHONSAFEPATH="1"))
        check(code_s == 0, f"colab/{name}: со строгим PYTHONSAFEPATH")
        code_e, _ = sh([PY, path, "--help"], clone,
                       env=dict(os.environ, PYTHONPATH=str(pathlib.Path("/nonexistent"))))
        check(code_e == 0, f"colab/{name}: с посторонним PYTHONPATH")

    section("2. Статические проверки исходников")
    for path in sorted(clone.glob("*.py")) + sorted(clone.glob("colab/**/*.py")):
        code_c, _ = sh([PY, "-m", "py_compile", path], pathlib.Path("/tmp"))
        check(code_c == 0, f"компилируется: {path.relative_to(clone)}")
        loose = unguarded_versions(path)
        check(not loose, f"{path.name}: версии пакетов читаются через try/except"
              + (" / " + "; ".join(loose) if loose else ""))
    harness = {"verify_kit.py"}          # сам чекер содержит эти строки как текст проверок
    for path in sorted(clone.glob("colab/**/*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(clone)
        if path.name in harness:
            print(f"  · {rel}: правила проверяются, но сам чекер из них исключён")
            continue
        if path.name == "build_single_voice_manifest.py":
            check('(HERE.parent / "prepare_qwen.py").is_file()' in text,
                  f"{rel}: сам находит корень репозитория")
        bad = [l.strip() for l in text.splitlines()
               if "assert" in l and '== "' in l and ("default_device" in l or "gpu" in l.lower())]
        check(not bad, f"{rel}: нет assert по repr-строке устройства")
        check("google.colab" not in text, f"{rel}: не зависит от Colab API")
        for line in text.splitlines():
            if "subprocess.Popen(" in line or "subprocess.run([" in line:
                window = text[text.index(line):text.index(line) + 400]
                if any(k in window for k in ("nvidia-smi", "ffmpeg", "py_compile", '"git"', "git ")):
                    continue
                check("env=" in window, f"{rel}: PYTHONPATH у дочернего процесса — {line.strip()[:50]}")

    all_stages = (clone / "colab/colab_run_all.py").read_text(encoding="utf-8")
    match = re.search(r"STAGES = \[(.*?)\]", all_stages, re.S)
    stages = re.findall(r'"([a-z]+)"', match.group(1)) if match else []
    for earlier, later in (("manifest", "prepare"), ("prepare", "voices"), ("stress", "voices"), ("voices", "synth")):
        if earlier in stages and later in stages:
            check(stages.index(earlier) < stages.index(later),
                  f"colab_run_all.py: стадия {earlier} идёт до {later}")
        else:
            check(False, f"colab_run_all.py: в STAGES нет {earlier} или {later}")
    check('append("--whole-book")' in all_stages,
          "colab_run_all.py: синтез всегда получает --whole-book (фрагмент задаёт prepare)")
    check('(Path(source) / "manifest.json")' in all_stages,
          "colab_run_all.py: библиотека голосов берёт идентичность из подготовленного манифеста")
    check('MLX_CUDA_CONV_CACHE_SIZE' in all_stages,
          "colab_run_all.py: кэш свёрток MLX/CUDA увеличен до запуска синтеза")

    section("3. Ячейки ноутбуков")
    for path in sorted(clone.glob("colab/**/*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
        bad = compile_notebook(path)
        check(not bad, f"{path.name}: {cells} кодовых ячеек компилируются" + (" / " + "; ".join(bad) if bad else ""))
        missing_names = undefined_names(path)
        check(not missing_names, f"{path.name}: каждая ячейка импортирует то, что использует"
              + (" / " + "; ".join(missing_names) if missing_names else ""))
        if path.name == "qwen_tts_colab_full_book.ipynb":
            # именно исходный текст ячеек, а не json.dumps: в дампе кавычки экранированы
            text = "\n".join("".join(c["source"]) for c in notebook["cells"])
            for marker, label in (("Path.cwd() / name", "загруженный файл ищется в cwd и в /content"),
                                  ("PYTHONPATH", "PYTHONPATH у дочерних процессов"),
                                  ("snapshot_download", "веса через snapshot_download"),
                                  ("HF_HUB_DISABLE_XET", "xet отключён"),
                                  ("book-read-along.epub", "упаковка M4B + EPUB"),
                                  ("prepare_silero.py", "ударения: словарь + Silero"),
                                  ("mix_book_music.py", "музыка из репозитория"),
                                  ('BASE.append("--whole-book")', "синтез озвучивает весь подготовленный фрагмент"),
                                  ("REBUILD = False", "повторный запуск не пересобирает подготовку и кэш синтеза"),
                                  ("MLX_CUDA_CONV_CACHE_SIZE", "кэш свёрток MLX/CUDA увеличен (иначе Cache thrashing)"),
                                  ('SOURCE / "manifest.json"', "библиотека голосов берёт идентичность из подготовленного манифеста")):
                check(marker in text, f"полная книга: {label}")

    if not args.book:
        section("4. Сквозной прогон пропущен (нет --book)")
    else:
        section("4. Сквозной прогон colab_run_all.py до стадии «voices» (всё, что до синтеза)")
        # Прогон патчит файлы репозитория (mlx-metal → mlx[cuda], ffmpeg → /usr/bin), поэтому работаем
        # на копии: аудируемое дерево портить нельзя — при --no-clone это ваша рабочая папка.
        worktree = pathlib.Path(tempfile.mkdtemp(prefix="verify-repo-")) / "repo"
        shutil.copytree(clone, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        print("  копия для прогона:", worktree)
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="verify-run-"))
        command = [PY, "colab/colab_run_all.py", pathlib.Path(args.book).expanduser(),
                   "--out", out_dir, "--test-seconds", "300", "--upto", "voices",
                   "--skip-install", "--skip-gpu-check"]
        if args.model_dir:
            command += ["--model-dir", pathlib.Path(args.model_dir).expanduser()]
        code, out = sh(command, worktree, timeout=1800)
        print("\n".join(out.splitlines()[-14:]))
        check(code == 0, "прогон завершился без ошибок")
        check("Traceback" not in out, "в логе нет трейсбеков")
        for required in ("project/manifest.json", "project/source.json",
                         "project/voice-library/narrator.json", "project/source/manifest.json",
                         "project/source/book-text.epub"):
            check((out_dir / required).is_file(), f"создан {required}")
        prepared = json.loads((out_dir / "project/source/manifest.json").read_text(encoding="utf-8"))
        marked = sum(1 for s in prepared["segments"] if "\u0301" in s.get("tts_text", ""))
        check(marked > 0, f"ударения расставлены: {marked} сегментов из {len(prepared['segments'])}")

        section("5. Ячейки ноутбука по порядку (без GPU, с подстановкой путей)")
        work = pathlib.Path(tempfile.mkdtemp(prefix="verify-nb-"))
        os.chdir(work)
        shutil.copyfile(pathlib.Path(args.book).expanduser(), work / pathlib.Path(args.book).name)

        fake_colab = types.ModuleType("google.colab")
        uploaded_name = pathlib.Path(args.book).name
        fake_colab.files = types.SimpleNamespace(upload=lambda: {uploaded_name: b""},
                                                 download=lambda path: print(f"  (отдал бы файл {path})"))
        fake_colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
        google = types.ModuleType("google")
        google.colab = fake_colab
        sys.modules["google"], sys.modules["google.colab"] = google, fake_colab

        notebook = json.loads((worktree / "colab/qwen_tts_colab_full_book.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        labels = {0: "окружение", 4: "проверка правок репозитория", 5: "каталоги проекта", 6: "загрузка EPUB",
                  8: "манифест из EPUB", 9: "prepare: нарезка и book-text.epub", 10: "ударения",
                  11: "URL образца голоса", 12: "библиотека голосов под подготовленный манифест"}
        skip = {1: "установка пакетов", 2: "проверка MLX", 3: "клонирование (shell)", 7: "скачивание весов",
                13: "план синтеза", 14: "синтез", 15: "громкость", 16: "музыка", 17: "упаковка", 18: "скачивание"}
        # Пространство имён пустое: только встроенные. Всё, что ячейка использует, она обязана
        # импортировать сама — ровно так в Colab теряется `import json` и падает в конце шага.
        globals_dict = {"__name__": "__main__"}
        for index, cell in enumerate(cells):
            if index in skip:
                print(f"  · ячейка «{skip[index]}» — пропущена (нужен Colab/GPU)")
                continue
            source = (cell.replace('ROOT = pathlib.Path("/content/qwen-epub-audiobook")',
                                   f'ROOT = pathlib.Path("{worktree}")')
                          .replace('REPO = "/content/qwen-epub-audiobook"', f'REPO = "{worktree}"')
                          .replace('OUT_ROOT = Path("/content/audiobook")', f'OUT_ROOT = Path("{work}/audiobook")')
                          .replace('"python"', f'"{PY}"'))
            try:
                exec(source, globals_dict)
                print(f"  ✔ ячейка «{labels.get(index, index)}» выполнилась")
            except SystemExit as exc:
                (print(f"  ✔ ячейка «окружение» штатно остановилась без GPU: {exc}") if index == 0
                 else check(False, f"ячейка «{labels.get(index, index)}»: SystemExit {exc}"))
            except Exception as exc:
                if index == 0 and "GPU не найден" in str(exc):
                    print(f"  ✔ ячейка «окружение» штатно требует GPU: {str(exc)[:60]}…")
                else:
                    check(False, f"ячейка «{labels.get(index, index)}»: {type(exc).__name__}: {exc}")
        check((work / "audiobook/project/source/book-text.epub").is_file(), "ноутбук дошёл до book-text.epub")

        # Переменная, из-за которой синтез падал на CUDA, должна доехать до дочернего процесса.
        code_env, out_env = sh([PY, "-c", "import os;print(os.environ.get('MLX_CUDA_CONV_CACHE_SIZE', ''))"],
                               worktree)
        value = out_env.strip()
        check(code_env == 0 and value.isdigit() and int(value) >= 4096,
              f"дочерний процесс синтеза видит MLX_CUDA_CONV_CACHE_SIZE={value or '—'} (нужно ≥ 4096)")

    section("ИТОГ")
    if PROBLEMS:
        print(f"НАЙДЕНО ПРОБЛЕМ: {len(PROBLEMS)}")
        for item in PROBLEMS:
            print("  -", item)
        return 1
    print("все проверки пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
