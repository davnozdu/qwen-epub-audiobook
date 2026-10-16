#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run-book.py — озвучить книгу через Colab CLI. Один код для macOS, Linux и Windows.

    python run-book.py kniga.epub              # книга в любой папке или ссылкой
    ./run-book.command kniga.epub              # macOS/Linux (тонкая обёртка над этим файлом)
    run-book.cmd kniga.epub                    # Windows (то же самое)

Что делает:
  1. освобождает прошлые сессии CLI (иначе Colab отвечает «Allocation refused»);
  2. поднимает сессию с GPU, подключает Google Drive, кладёт туда книгу,
     запускает задание qwen_tts_book.py;
  3. забирает zip в папку «Загрузки» и освобождает машину.

Ничего макового внутри нет: путь к CLI находится через PATH, хеш считается hashlib,
ожидание Drive — через таймаут процесса, зависший процесс глушится своим способом
на POSIX (kill) и на Windows (taskkill /T).

Переменные окружения (то же можно задать ключами):
  GPU=L4              ускоритель (по умолчанию T4)
  SESSION=book        имя сессии
  FRAGMENT=300        озвучить только 300 секунд вместо всей книги
  HIGHLIGHT=fragment  подсветка по абзацам: сегментов меньше, синтез быстрее
  KEEP_SESSION=1      не гасить машину в конце (потом: colab stop -s book)
  DRIVE=1             положить book.m4b и read-along epub в MyDrive/<DRIVE_DIR>/<имя книги>
  DRIVE_DIR=...       папка внутри MyDrive (по умолчанию Audiobooks-colab)
  DRIVE_WAIT=120      сколько секунд ждать подтверждения Drive, потом продолжить без него
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

IS_WINDOWS = os.name == "nt"
DEFAULT_LOCAL_BIN = pathlib.Path.home() / ".local" / "bin"


def setup_stdout() -> None:
    """Русский текст в консоли Windows: без этого часть систем печатает крякозябры."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


# ────────────────────────────── где лежит CLI ──────────────────────────────
def find_colab() -> str:
    found = shutil.which("colab")
    if found:
        return found
    suffix = ".exe" if IS_WINDOWS else ""
    guesses = [DEFAULT_LOCAL_BIN / f"colab{suffix}",
               pathlib.Path.home() / ".local/share/uv/tools/google-colab-cli/bin" / f"colab{suffix}",
               pathlib.Path(os.environ.get("APPDATA", "")) / "uv/tools/google-colab-cli/Scripts" / f"colab{suffix}",
               pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "uv/tools/google-colab-cli/Scripts" / f"colab{suffix}",
               pathlib.Path(os.environ.get("USERPROFILE", "")) / ".local/bin" / f"colab{suffix}"]
    for guess in guesses:
        if guess.is_file():
            return str(guess)
    raise SystemExit("нет Colab CLI в PATH — поставьте его: pip install google-colab-cli\n"
                     "  (или uv tool install google-colab-cli)")


# ────────────────────────────── запуск процессов ──────────────────────────────
def run(cmd, timeout=None) -> subprocess.CompletedProcess:
    """Команда «насквозь»: вывод печатается, стандартный ввод доступен (нужно для Drive)."""
    return subprocess.run([str(part) for part in cmd], text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", env=dict(os.environ))


def run_quiet(cmd) -> subprocess.CompletedProcess:
    """То же, но вывод собирается: нужно для разбора списка сессий."""
    try:
        return subprocess.run([str(part) for part in cmd], text=True, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=120, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", "timeout")


def kill_command(pid: int, nt: bool | None = None) -> list[str]:
    """Чем глушить процесс: на Windows — вместе с детьми, иначе — сигналом."""
    windows = IS_WINDOWS if nt is None else nt
    return ["taskkill", "/F", "/T", "/PID", str(pid)] if windows else ["kill", str(pid)]


def child_pids(pid: int) -> list[int]:
    """Прямые потомки процесса — их тоже надо погасить (ps есть и на macOS, и на Linux)."""
    done = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=dict(os.environ))
    return [int(value) for value in (done.stdout or "").split() if value.isdigit()]


def kill_tree(process) -> None:
    """Гасим процесс и его детей. Свою группу не трогаем: мы в ней сами и терминал у нас общий."""
    try:
        if IS_WINDOWS:
            subprocess.run(kill_command(process.pid, nt=True), capture_output=True, env=dict(os.environ))
        else:
            for child in child_pids(process.pid):
                try:
                    os.kill(child, 9)
                except OSError:
                    pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait(timeout=30)
    except Exception:
        pass


LIVE_CHILDREN = []            # дети, которых могли не дождаться: при выходе гасим


def _reap_children() -> None:
    for child in list(LIVE_CHILDREN):
        if child.poll() is None:
            kill_tree(child)


atexit.register(_reap_children)


def supports_hyperlinks() -> bool:
    """Умеет ли терминал ссылки OSC 8 (в Apple Terminal их нет — там Cmd+клик по обычной ссылке)."""
    program = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")
    if program in ("iTerm.app", "WezTerm", "vscode", "Hyper", "WarpTerminal", "ghostty", "kitty"):
        return True
    return any(name in term for name in ("kitty", "wezterm", "ghostty", "alacritty"))


def is_drive_auth(url: str) -> bool:
    """Это та самая ссылка авторизации Google Drive, которую выдаёт CLI."""
    return "accounts.google.com" in url and "/o/oauth2/" in url


def clickable(url: str) -> str:
    """Ссылка авторизации: подсвечена в любом терминале, а где есть гиперссылки — ещё и кликабельна.

    В логи (вывод не в терминал) ничего не подмешиваем: подсветка нужна человеку, а не автоматизации.
    """
    if not sys.stdout.isatty():
        return url
    text = f"\x1b[1;4;36m{url}\x1b[0m"          # жирный подчёркнутый голубой
    if supports_hyperlinks():
        return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"
    return text


def open_browser(url: str) -> None:
    """Открываем ссылку в браузере пользователя (OPEN_CMD заменяет открывалку — удобно для тестов)."""
    override = os.environ.get("OPEN_CMD")
    try:
        if override:
            subprocess.run([override, url], env=dict(os.environ))
        elif IS_WINDOWS:
            subprocess.run(["cmd", "/c", "start", "", url], env=dict(os.environ))
        elif sys.platform == "darwin":
            subprocess.run(["open", url], env=dict(os.environ))
        else:
            subprocess.run(["xdg-open", url], env=dict(os.environ))
    except OSError as exc:
        print(f"  (браузер открыть не вышло: {exc})", flush=True)


def stream_step(cmd, timeout: float, open_links: bool = False):
    """Запуск шага с живым выводом и потолком по времени.

    Вывод печатаем сразу, кусками: CLI пишет приглашение «Press Enter…» без перевода строки,
    и построчное чтение его съедало.

    Украшаем вывод только там, где терминал умеет гиперссылки (iTerm2, WezTerm, VS Code, Warp,
    Ghostty, kitty). В Apple Terminal гиперссылок нет — там отдаём поток как есть, без подсветки:
    длинную ссылку всё равно открывают ⌘-кликом. Если явно попросили --open, ссылку находим,
    но по-прежнему ничего не раскрашиваем.
    """
    import select

    decorate = sys.stdout.isatty()        # в терминале подсвечиваем, в логах — ничего лишнего
    collect = decorate or open_links
    process = subprocess.Popen([str(part) for part in cmd], env=dict(os.environ),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    LIVE_CHILDREN.append(process)
    tail, opened, state, links = "", set(), "ok", []

    def emit(chunk, final=False):
        """Печатаем поток; ссылки подсвечиваем на месте — токен может прийти по частям."""
        nonlocal tail
        if not collect:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            return
        tail += chunk
        pieces = re.split(r"(\s+)", tail)
        if not final and not tail[-1:].isspace():
            tail = pieces.pop() if pieces else ""
        else:
            tail = ""
        for piece in pieces:
            if piece.startswith(("http://", "https://")) and is_drive_auth(piece):
                sys.stdout.write(clickable(piece))     # подсвечиваем только ссылку авторизации Drive
                if piece not in links:
                    links.append(piece)
            else:
                sys.stdout.write(piece)                # прочие ссылки не трогаем
        sys.stdout.flush()

    deadline = time.monotonic() + max(1.0, timeout)
    try:
        while True:
            if process.poll() is not None:
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if not ready:
                    break
            left = deadline - time.monotonic()
            if left <= 0:
                state = "timeout"
                break
            ready, _, _ = select.select([process.stdout], [], [], min(1.0, left))
            if not ready:
                continue
            chunk = os.read(process.stdout.fileno(), 4096).decode("utf-8", "replace")
            if not chunk:
                break
            emit(chunk)
        emit("", final=True)                       # хвост без пробела всё равно печатаем
        if open_links:                           # ссылку показываем один раз — в строке самого CLI
            for url in links or []:
                if url not in opened:
                    opened.add(url)
                    print(f"\n   открываю браузер: {url}", flush=True)
                    open_browser(url)
        if state == "timeout":
            kill_tree(process)
        else:
            process.wait()
            state = "ok" if process.returncode == 0 else "failed"
    except KeyboardInterrupt:
        kill_tree(process)
        raise
    finally:
        if process in LIVE_CHILDREN:
            LIVE_CHILDREN.remove(process)
    return state


def run_with_timeout(cmd, timeout: float) -> str:
    """Запуск с потолком по времени. Возвращает: 'ok', 'failed' или 'timeout'.

    Вывод НЕ перехватываем и НЕ уводим процесс в новую сессию: CLI печатает приглашение без
    перевода строки и читает ответ «Press Enter...» прямо из /dev/tty. В новой сессии
    управляющего терминала нет, /dev/tty не открывается — и ожидание виснет навсегда.
    """
    process = subprocess.Popen([str(part) for part in cmd], env=dict(os.environ))
    LIVE_CHILDREN.append(process)
    try:
        process.wait(timeout=max(1.0, timeout))
        return "ok" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        kill_tree(process)
        return "timeout"
    except KeyboardInterrupt:
        kill_tree(process)                      # иначе ожидание монтирования останется сиротой
        raise
    finally:
        if process in LIVE_CHILDREN:
            LIVE_CHILDREN.remove(process)


# ────────────────────────────── сессии ──────────────────────────────
def cli_state_path() -> pathlib.Path:
    """Где CLI держит свои сессии: на всех системах это ~/.config/colab-cli/sessions.json."""
    return pathlib.Path.home() / ".config" / "colab-cli" / "sessions.json"


def list_sessions(colab: str) -> str:
    done = run_quiet([colab, "sessions"])
    return done.stdout or ""


def stop_stale_sessions(colab: str) -> str:
    """Гасит всё, что держит квоту: свои сессии и «чужие» (в списке помечены «?»)."""
    print("== прошлые сессии ==")
    text = list_sessions(colab)
    print(text.rstrip() or "  (пусто)")

    for name in re.findall(r"^\[([^\]]*)\]\s+\S+\s+\|", text, re.M):
        if name and name != "?":
            print(f"  останавливаю: {name}")
            run_quiet([colab, "stop", "-s", name])

    # Чужие сессии CLI по имени не находит — он ищет только в своём файле состояния.
    # Дописываем запись с известным endpoint: при остановке CLI сам подтянет к ней токен.
    state = cli_state_path()
    for index, endpoint in enumerate(re.findall(r"^\[\?\]\s+(\S+)\s+\|", text, re.M), 1):
        name = f"orphan-{index}"
        print(f"  чужая сессия {endpoint} → беру как {name} и останавливаю")
        try:
            data = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
            data[name] = {"name": name, "token": "", "url": "", "endpoint": endpoint,
                          "variant": "GPU", "accelerator": "T4", "machine_shape": "STANDARD"}
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            print(f"  (не смог поправить {state}: {exc})")
            continue
        run_quiet([colab, "stop", "-s", name])

    return list_sessions(colab)


def drive_is_mounted(colab: str, session: str) -> bool:
    """Спрашиваем саму машину: Enter без подтверждения доступа тоже даёт код 0 у CLI."""
    done = subprocess.run([colab, "exec", "-s", session],
                          input='import os; print("DRIVE_OK", os.path.isdir("/content/drive/MyDrive"))\n',
                          text=True, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=180, env=dict(os.environ))
    return "DRIVE_OK True" in (done.stdout or "")


def mount_drive(colab: str, session: str, wait_seconds: int, open_links: bool = True) -> bool:
    """Подключает Drive. Возвращает True только если монтирование действительно есть.

    wait_seconds — это потолок ожидания, а не обязательная пауза: как только пользователь нажал
    Enter после авторизации, CLI дорабатывает монтирование и обёртка идёт дальше сразу.
    """
    print("\n== подключаю Google Drive ==")
    print("   если Google попросит подтверждение, покажу ссылку подсвеченной — откройте её и разрешите")
    print(f"   доступ, затем нажмите Enter в этом окне; не дождались за {wait_seconds} с — задание")
    print("   поедет без Drive, архив заберём локально")
    print("   если CLI напишет «Press Enter after you have granted access» — вернитесь сюда и нажмите Enter")
    state = stream_step([colab, "drivemount", "-s", session], timeout=wait_seconds,
                        open_links=open_links)
    if state == "timeout":
        print(f"  за {wait_seconds} с подтверждения не было — снимаю попытку, Drive пропускаем.")
        print("  едем без Drive: архив всё равно заберём в загрузки.")
        return False
    if state == "failed":
        print("  Drive не подключился (или доступ не подтверждён).")
        print("  едем без Drive: архив всё равно заберём в загрузки.")
        return False

    # Только что смонтированный Drive виден не мгновенно, поэтому спрашиваем машину несколько раз.
    for attempt in range(6):
        if drive_is_mounted(colab, session):
            return True
        time.sleep(2)
    print("  монтирования не видно (или доступ не подтверждён).")
    print("  едем без Drive: архив всё равно заберём в загрузки.")
    return False


# ────────────────────────────── файлы и пути ──────────────────────────────
def keep_awake():
    """macOS: caffeinate, чтобы мак не заснул посреди работы (сон рвёт сессию Colab).

    Процесс живёт ровно столько, сколько живём мы: caffeinate -w <наш pid>.
    В конце убиваем сами — atexit срабатывает на любом выходе, включая ранние ошибки.
    """
    if sys.platform != "darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        return None
    process = subprocess.Popen([caffeinate, "-i", "-s", "-w", str(os.getpid())],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=dict(os.environ))
    print(f"мак не заснёт во время работы: caffeinate -i -s -w {os.getpid()} (pid {process.pid})")
    atexit.register(lambda: process.terminate())
    return process


def downloads_dir() -> pathlib.Path:
    """Папка загрузок на любой системе (на Windows — профиль пользователя)."""
    candidates = [pathlib.Path.home() / "Downloads",
                  pathlib.Path(os.environ.get("USERPROFILE", "")) / "Downloads"]
    for candidate in candidates:
        if str(candidate.parent) and candidate.parent.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    fallback = pathlib.Path.cwd()
    return fallback


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_book(url: str, target_dir: pathlib.Path) -> pathlib.Path:
    target = target_dir / pathlib.Path(url.split("?")[0]).name
    print(f"== качаю книгу к себе: {target} ==")
    urllib.request.urlretrieve(url, target)
    return target


def fetch_plan(colab: str, session: str, stem: str, target_dir: pathlib.Path) -> dict | None:
    """Забираем план нарезки, если задание резало книгу. Нет плана — книга влезла целиком."""
    remote = f"/content/out/{stem}-parts/plan.json"
    local = target_dir / f"{stem}-plan.json"
    if run_quiet([colab, "download", "-s", session, remote, str(local)]).returncode != 0:
        return None
    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fetch_parts(colab: str, session: str, plan: dict | None, target_dir: pathlib.Path) -> list:
    """Части книги сохраняем и локально: первые обрабатываются, остальные лежат на Drive."""
    if not plan or plan.get("parts", 1) <= 1:
        return []
    saved = []
    for item in plan["files"]:
        local = target_dir / pathlib.Path(item["epub"]).name
        if run_quiet([colab, "download", "-s", session, item["epub"], str(local)]).returncode == 0:
            saved.append(local)
    return saved


def build_job_file(book_remote: str, fragment: str, highlight: str, drive: bool,
                   drive_dir: str, job: pathlib.Path, shrink: dict) -> pathlib.Path:
    """Задание уходит на машину вместе с аргументами: colab exec аргументы не передаёт."""
    args = [book_remote]
    if fragment:
        args += ["--seconds", fragment]
    if highlight:
        args += ["--highlight", highlight]
    args += ["--drive-dir", drive_dir] if drive else ["--no-drive"]
    if shrink.get("off"):
        args += ["--no-shrink"]
    else:
        args += ["--max-hours", str(shrink["max_hours"]), "--shrink-dir", str(shrink["dir"])]
    head = ("import sys\n"
            f"sys.argv = {json.dumps(['qwen_tts_book.py', *args])}\n"
            '__name__ = "__main__"\n')
    handle, path = tempfile.mkstemp(prefix="qwen-book-", suffix=".py")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(head + job.read_text(encoding="utf-8"))
    print("задание с параметрами:", " ".join(args), flush=True)
    return pathlib.Path(path)


# ────────────────────────────── основной ход ──────────────────────────────
def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser(description="Озвучить книгу на машине Colab через Colab CLI",
                                     usage="%(prog)s kniga.epub [--gpu T4] [--fragment 300]")
    parser.add_argument("book", nargs="?", default="", help="книга: путь, имя файла или ссылка http(s)")
    parser.add_argument("--gpu", default=os.environ.get("GPU", "T4"), help="ускоритель (T4, L4, A100)")
    parser.add_argument("--session", default=os.environ.get("SESSION", "book"), help="имя сессии CLI")
    parser.add_argument("--fragment", default=os.environ.get("FRAGMENT", ""),
                        help="озвучить только столько секунд")
    parser.add_argument("--highlight", default=os.environ.get("HIGHLIGHT", ""),
                        choices=["", "sentence", "fragment"], help="подсветка: по предложениям или абзацам")
    parser.add_argument("--keep-session", action="store_true",
                        default=os.environ.get("KEEP_SESSION", "0") == "1",
                        help="не гасить машину после работы")
    parser.add_argument("--no-drive", action="store_true", default=os.environ.get("DRIVE", "1") == "0",
                        help="не подключать Google Drive")
    parser.add_argument("--drive-dir", default=os.environ.get("DRIVE_DIR", "Audiobooks-colab"),
                        help="папка внутри MyDrive")
    parser.add_argument("--drive-wait", type=int, default=int(os.environ.get("DRIVE_WAIT", "60")),
                        help="сколько секунд ждать подтверждения Drive (по умолчанию 60)")
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("SHRINK_HOURS", "7")),
                        help="если книга длиннее — задание разрежет её на части (0 — не резать)")
    parser.add_argument("--shrink-dir", default=os.environ.get("SHRINK_DIR", "Audiobooks-shrink"),
                        help="папка на Drive для остальных частей")
    parser.add_argument("--no-shrink", action="store_true", default=os.environ.get("SHRINK", "1") == "0",
                        help="не резать книгу на части")
    parser.add_argument("--open", action="store_true", default=os.environ.get("AUTO_OPEN", "0") == "1",
                        help="открывать браузер со ссылкой авторизации Drive "
                             "(по умолчанию выключено: ссылка просто подсвечивается)")
    parser.add_argument("--no-open", action="store_true", help="оставлено для совместимости: браузер и так не открываем")
    parser.add_argument("--no-caffeinate", action="store_true",
                        default=os.environ.get("CAFFEINATE", "1") == "0",
                        help="macOS: не мешать маку засыпать (по умолчанию мешаем)")
    args, extra = parser.parse_known_args()
    if extra:
        print("лишние аргументы проигнорированы:", extra, flush=True)
    if not args.book:
        parser.print_help()
        print("\nпример: run-book.command kniga.epub   |   run-book.cmd kniga.epub")
        return 1

    here = pathlib.Path(__file__).resolve().parent
    job = here / "qwen_tts_book.py"
    if not job.is_file():
        print(f"рядом с этим файлом нет qwen_tts_book.py ({here})")
        return 1
    colab = find_colab()
    if not args.no_caffeinate:
        keep_awake()          # macOS: иначе сон может оборвать сессию Colab

    if args.book.startswith(("http://", "https://")):
        book = fetch_book(args.book, downloads_dir())
    else:
        book = pathlib.Path(args.book).expanduser()
        if not book.is_file():
            print(f"файла нет: {book}")
            return 1
    book = book.resolve()
    stem = book.stem or "book"
    remote_book = f"/content/{stem}.epub"
    shrink = {"max_hours": args.max_hours, "dir": args.shrink_dir, "off": args.no_shrink}

    def fail() -> int:
        print("\n== не получилось, освобождаю машину ==")
        if not args.keep_session:
            run_quiet([colab, "stop", "-s", args.session])
        return 1

    remaining = stop_stale_sessions(colab)
    if re.search(r"^\[\?\]", remaining, re.M):
        print("  ВНИМАНИЕ: сессия «?» всё ещё висит. Значит освободить её из терминала не вышло —")
        print("  завершите вручную: colab.research.google.com → Manage sessions → Terminate.")

    print(f"\n== поднимаю сессию {args.session} ({args.gpu}) ==")
    if run([colab, "new", "-s", args.session, "--gpu", args.gpu]).returncode != 0:
        return fail()

    drive_ok = False
    if not args.no_drive:
        drive_ok = mount_drive(colab, args.session, args.drive_wait,
                               open_links=args.open and not args.no_open)
        if drive_ok:
            print(f"  Drive подключён: book.m4b и read-along epub поедут в MyDrive/{args.drive_dir}/<имя книги>")

    print(f"\n== книга на машину: {stem} ==")
    if run([colab, "upload", "-s", args.session, str(book), remote_book]).returncode != 0:
        return fail()

    generated = build_job_file(remote_book, args.fragment, args.highlight, drive_ok, args.drive_dir,
                               job, shrink)
    print("\n== задание ==")
    try:
        # --timeout у colab exec по умолчанию 30 секунд: на книгу этого мало, ждём до суток.
        code = run([colab, "exec", "-s", args.session, "-f", str(generated), "--timeout", "86400"]).returncode
    finally:
        generated.unlink(missing_ok=True)
    if code != 0:
        print(f"== CLI вернул код {code} — это может быть таймаут ожидания кернела, а не ошибка задания")

    print("\n== забираю архив ==")
    # Толстая книга режется на части ещё на машине: узнаём по плану, как называется архив.
    plan = fetch_plan(colab, args.session, stem, downloads_dir())
    if plan and plan.get("parts", 1) <= 1:
        print(f"оценка: книга ≈{plan['estimated_hours']:.2f} ч звука (~столько же часов GPU)")
    effective = f"{stem}-part1" if plan and plan.get("parts", 1) > 1 else stem
    remote_zip = f"/content/out/{effective}.zip"
    local_zip = downloads_dir() / f"{effective}.zip"

    # Машину при неудаче НЕ гасим: готовый архив может уже лежать на ней, а после stop он пропадёт.
    if run([colab, "download", "-s", args.session, remote_zip, str(local_zip)]).returncode != 0:
        print("архива на машине нет (или он ещё собирается). Машину не гашу, чтобы не потерять работу.")
        print("дальше вручную:")
        print(f"  colab download -s {args.session} {remote_zip} {local_zip}")
        print(f"  colab stop -s {args.session}")
        return 1
    if local_zip.is_file():
        print(f"{sha256_of(local_zip)}  {local_zip.name}")
    print(f"готово: {local_zip}")

    if plan and plan.get("parts", 1) > 1:
        saved = fetch_parts(colab, args.session, plan, downloads_dir())
        print()
        print("=" * 74)
        print(f"КНИГА БЫЛА БОЛЬШОЙ: ≈{plan['estimated_hours']:.1f} ч звука, разрезана на {plan['parts']} части.")
        print("Сейчас готова только первая. Остальные части обрабатывайте ОТДЕЛЬНО — каждую своим запуском;")
        print(f"они уже лежат на Google Drive в папке «{plan.get('shrink_dir', 'Audiobooks-shrink')}» и локально.")
        for index, item in enumerate(plan["files"], 1):
            state = "готова (этот запуск)" if index == 1 else "обработать отдельно"
            local_note = ""
            if index > 1 and saved:
                local_note = f" | локально: {pathlib.Path(item['epub']).name}"
            print(f"  часть {index}: ≈{item['estimated_hours']:.1f} ч — {state}{local_note}")
        for index, item in enumerate(plan["files"][1:], 2):
            print(f"    ./run-book.command {pathlib.Path(item['epub']).name}")
        print("=" * 74)

    if args.keep_session:
        print(f"машина оставлена (--keep-session): следующая книга пойдёт быстрее.")
        print(f"освободить потом: colab stop -s {args.session}")
    else:
        print("== освобождаю машину ==")
        run_quiet([colab, "stop", "-s", args.session])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
