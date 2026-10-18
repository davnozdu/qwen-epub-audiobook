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
  OUT_DIR=…           куда складывать готовый архив (по умолчанию ~/Downloads)
  STOP_ON_FAILURE=1   гасить машину Colab и при неудаче (для контейнера и фоновых запусков)
  DRIVE_WAIT=60       через сколько секунд нажать Enter и проверить доступ к Drive
  DRIVE_AUTO_ENTER=0  не нажимать Enter автоматически (подтверждать вручную)
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
import signal
import subprocess
import sys
import tempfile

try:
    import pty                      # только Unix: нужен, чтобы нажать Enter за пользователя
except ImportError:                 # pragma: no cover - Windows
    pty = None
import time
import urllib.request
import zipfile

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
def _repo_root() -> pathlib.Path:
    """Корень репозитория: telegram_notify.py лежит там, а не рядом с обёрткой."""
    return pathlib.Path(__file__).resolve().parent.parent.parent


def notify_telegram(message: str) -> None:
    """Уведомление в Telegram, если задан TELEGRAM_BOT_TOKEN. Без ключа — тишина."""
    try:
        import telegram_notify
    except ModuleNotFoundError:
        sys.path.insert(0, str(_repo_root()))   # запускаемся из colab/cli — модуль в корне
        import telegram_notify
    try:
        telegram_notify.notify(message)
    except Exception:
        pass


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


LAST_OUTPUT = ""          # хвост последнего вывода: нужен, чтобы понять причину сбоя


def new_view(collect: bool) -> dict:
    """Состояние вывода шага: хвост строки, найденные ссылки, когда показано приглашение."""
    return {"collect": collect, "tail": "", "recent": "", "links": [], "link_at": None,
            "prompt_at": None}


PROPAGATE_NOTE = ("   Google отклонил авторизацию Google Drive (обычно это значит, что доступ подтвердили "
                  "не под тем аккаунтом, под которым работает Colab CLI). Продолжаю без Google Drive: "
                  "архив заберём локально.")


def squelch_propagate_error(text: str, view: dict) -> str:
    """CLI печатает «[colab] Error propagating: 400 <!DOCTYPE html>…</html>» — прячем HTML-портянку.

    Google отвечает страницей 400 с разметкой; она может прийти и в нескольких кусках, поэтому
    состояние «мы внутри html» держим в view. Человеку оставляем одну строку с сутью, в Telegram
    уходит один раз подробное объяснение.
    """
    if view.get("in_html"):
        end = text.find("</html>")
        if end < 0:
            return ""
        view["in_html"] = False
        text = text[end + len("</html>"):]
    marker = "[colab] Error propagating"
    if marker in text:
        head, _, tail = text.partition(marker)
        pos = tail.find("<!DOCTYPE html>")
        if pos >= 0:
            end = tail.find("</html>", pos)
            if end < 0:
                view["in_html"] = True
                rest = ""
            else:
                rest = tail[end + len("</html>"):]
            text = head.rstrip() + "\n" + PROPAGATE_NOTE + "\n" + rest
        if not view.get("propagate_notified"):
            view["propagate_notified"] = True
            notify_telegram("⚠️ Google отклонил авторизацию Google Drive — скорее всего, доступ "
                            "подтвердили не под тем аккаунтом Google. Сборка продолжается без выгрузки "
                            "на Drive. В следующий раз подтверждайте по ссылке под тем же аккаунтом, "
                            "под которым работает CLI (если в браузере несколько аккаунтов — откройте "
                            "ссылку в окне инкогнито), либо сбросьте вход и войдите заново.")
        return text
    if "<!DOCTYPE html>" in text:                    # страница пришла отдельным куском после маркера
        pos = text.find("<!DOCTYPE html>")
        end = text.find("</html>", pos)
        if end < 0:
            view["in_html"] = True
            return text[:pos]
        text = text[:pos] + text[end + len("</html>"):]
    return text


def emit_output(text: str, view: dict, final: bool = False) -> None:
    """Печатаем кусок вывода шага; ссылку авторизации Drive подсвечиваем прямо в строке CLI."""
    global LAST_OUTPUT
    text = squelch_propagate_error(text, view)
    LAST_OUTPUT = (LAST_OUTPUT + text)[-4000:]
    view["recent"] = (view["recent"] + text)[-400:]
    if re.search(r"press\s+enter", view["recent"], re.IGNORECASE) and not view["prompt_at"]:
        view["prompt_at"] = time.monotonic()        # дальше CLI ждёт нажатия Enter
    if not view["collect"]:
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    view["tail"] += text
    pieces = re.split(r"(\s+)", view["tail"])
    if not final and not view["tail"][-1:].isspace():
        view["tail"] = pieces.pop() if pieces else ""
    else:
        view["tail"] = ""
    for piece in pieces:
        if piece.startswith(("http://", "https://")) and is_drive_auth(piece):
            sys.stdout.write(clickable(piece))      # подсвечиваем только ссылку авторизации Drive
            if piece not in view["links"]:
                view["links"].append(piece)
                view["link_at"] = time.monotonic()
                notify_telegram("🔗 Подтвердите доступ к Google Drive ПОД ТЕМ ЖЕ аккаунтом Google, "
                                "под которым работает CLI (при нескольких аккаунтах в браузере "
                                "откройте ссылку в окне инкогнито):\n" + piece)
        else:
            sys.stdout.write(piece)                 # прочие ссылки не трогаем
    sys.stdout.flush()


def drive_mount_step(cmd, wait_seconds: float, silent_cap: float, auto_enter: bool = True,
                     open_links: bool = False):
    """Монтирование Drive в терминале, которым управляем мы.

    Это даёт две вещи: вывод идёт как есть (со подсвеченной ссылкой), и Enter можно нажать
    за пользователя — он подтверждает доступ с телефона, а через wait_seconds обёртка сама
    отправляет пустую строку. Дальше CLI либо доводит монтирование, либо завершается ошибкой:
    и в том и в другом случае решает проверка на самой машине.

    Возвращает (состояние, нажали_ли_enter): состояние — 'ok', 'failed' или 'timeout'.
    """
    import select

    if pty is None:                                  # Windows: терминала не дадим, работаем как раньше
        state = stream_step(cmd, timeout=wait_seconds, silent_timeout=silent_cap, open_links=open_links)
        return state, False

    # Ссылки ловим всегда: в контейнере бота stdout — труба (лог tmux), иначе ссылка авторизации
    # Drive не найдётся и не уйдёт в Telegram. Подсветка по-прежнему только в живом терминале.
    view = new_view(True)
    pid, master = pty.fork()
    if pid == 0:                                     # ребёнок: сразу подменяем себя на CLI
        try:
            os.execvpe(str(cmd[0]), [str(part) for part in cmd], dict(os.environ))
        except Exception:
            os._exit(127)

    started = time.monotonic()
    outcome, entered, after_enter = "ok", False, None
    try:
        while True:
            now = time.monotonic()
            base = view["prompt_at"] or view["link_at"]
            if after_enter:                              # после нашего Enter даём CLI доработать
                if now > after_enter + 120:
                    outcome = "timeout"
                    break
                limit = after_enter + 120
            elif base:
                if now - base >= max(1.0, wait_seconds):
                    if auto_enter and not entered:       # время вышло — нажимаем Enter за пользователя
                        os.write(master, b"\n")
                        entered, after_enter = True, time.monotonic()
                        print("\n  ⏎ Enter нажат автоматически — проверяю, дан ли доступ к Google Drive…",
                              flush=True)
                        notify_telegram("⏎ Enter нажат автоматически — проверяю доступ к Google Drive")
                        continue
                    outcome = "timeout"
                    break
                limit = base + max(1.0, wait_seconds)
            else:                                        # ссылки нет: идёт обычное монтирование
                if now > started + max(1.0, silent_cap):
                    outcome = "silent-timeout"
                    break
                limit = started + max(1.0, silent_cap)
            ready, _, _ = select.select([master], [], [], min(0.25, max(0.05, limit - now)))
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    emit_output(chunk.decode("utf-8", "replace").replace("\r\n", "\n"), view)
            done, status = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                while True:                          # добираем то, что осталось в буфере
                    ready, _, _ = select.select([master], [], [], 0.1)
                    if not ready:
                        break
                    try:
                        rest = os.read(master, 65536)
                    except OSError:
                        break
                    if not rest:
                        break
                    emit_output(rest.decode("utf-8", "replace").replace("\r\n", "\n"), view)
                emit_output("", view, final=True)
                outcome = "ok" if os.waitstatus_to_exitcode(status) == 0 else "failed"
                break
        if open_links:
            for url in view["links"]:
                print(f"\n   открываю браузер: {url}", flush=True)
                open_browser(url)
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if outcome in ("timeout", "silent-timeout"):
            for killer in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pid, killer)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    os.waitpid(pid, 0)
                    break
                except ChildProcessError:
                    break
                except OSError:
                    time.sleep(0.5)
    return outcome, entered


def stream_step(cmd, timeout: float, silent_timeout: float | None = None, open_links: bool = False):
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
    # Ссылку авторизации ловим и для Telegram, даже когда вывод уходит в трубу (бот, логи).
    collect = decorate or open_links or bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
    process = subprocess.Popen([str(part) for part in cmd], env=dict(os.environ),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    LIVE_CHILDREN.append(process)
    view = new_view(collect)
    opened, state = set(), "ok"

    def emit(chunk, final=False):
        """Печатаем поток; ссылки подсвечиваем на месте — токен может прийти по частям."""
        nonlocal deadline
        emit_output(chunk, view, final=final)
        if view["link_at"] and view["links"]:
            deadline = max(deadline, view["link_at"] + max(1.0, timeout))

    started = time.monotonic()
    deadline = started + max(1.0, timeout)          # ожидание подтверждения (ссылки и Enter)
    limit = started + max(1.0, silent_timeout or timeout)
    try:
        while True:
            if process.poll() is not None:
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if not ready:
                    break
            now = time.monotonic()
            # до появления ссылки подтверждать нечего: это просто медленное монтирование
            left = max(deadline, limit) - now if not view["links"] else deadline - now
            if left <= 0:
                state = "timeout" if view["links"] else "silent-timeout"
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
            for url in view["links"] or []:
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


def probe_session(colab: str, session: str, timeout: float = 90) -> bool:
    """Жива ли машина: короткий запуск в кернеле. «Connection was lost» = кернел умер."""
    global LAST_OUTPUT
    probe = pathlib.Path(tempfile.gettempdir()) / "colab-probe.py"
    probe.write_text('print("жив")\n', encoding="utf-8")
    LAST_OUTPUT = ""
    try:
        process = subprocess.run([colab, "exec", "-s", session, "-f", str(probe), "--timeout", "60"],
                                 capture_output=True, text=True, timeout=timeout,
                                 env=dict(os.environ))
    except subprocess.TimeoutExpired:
        return False
    finally:
        probe.unlink(missing_ok=True)
    output = (process.stdout or "") + (process.stderr or "")
    return process.returncode == 0 and "жив" in output


def short_reason(output: str) -> str:
    """Короткая причина из вывода CLI: без рамок и трассировки."""
    lines = [re.sub(r"[│╭╰─╮╯]+", " ", line).strip() for line in (output or "").splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines):
        if any(word in line for word in ("Error", "error", "refused", "Unavailable", "unavailable",
                                         "limit", "Limit", "precondition")):
            return line[:120]
    return lines[-1][:120] if lines else "причина неясна"


def start_session(colab: str, session: str, gpu: str) -> bool:
    """Поднимает машину и убеждается, что кернел отвечает.

    Машина может не достаться по внешним причинам: Colab отвечает «Service Unavailable», когда
    свободных T4 нет, или «too many assignments», когда у бесплатного тарифа закончились места.
    Поэтому пробуем несколько раз с паузой, а не падаем после первой попытки.
    """
    attempts = int(os.environ.get("START_ATTEMPTS", "4"))
    delays = [float(part) for part in os.environ.get("START_RETRY_DELAYS", "15,30,60").split(",")]
    for attempt in range(1, attempts + 1):
        done = run_quiet([colab, "new", "-s", session, "--gpu", gpu])
        output = (done.stdout or "") + (done.stderr or "")
        if done.returncode == 0:
            if probe_session(colab, session):
                return True
            print("  машина поднялась, но кернел не отвечает — проверяю ещё раз")
        else:
            print(f"  машина не выдана: {short_reason(output)}")
        if attempt == attempts:
            break
        delay = delays[min(attempt - 1, len(delays) - 1)]
        if "TooManyAssignments" in output or "too many" in output.lower():
            delay = max(delay, 120)                  # лимит бесплатного тарифа: ждём дольше
            print("  похоже, у бесплатного тарифа заняты все машины — подожду подольше")
        print(f"  повторяю попытку через {delay:.0f} с (это {attempt + 1}-я из {attempts})")
        notify_telegram(f"⏳ Colab не выдал машину ({short_reason(output)}). "
                        f"Повторю через {delay:.0f} с — попытка {attempt + 1} из {attempts}.")
        time.sleep(delay)
    notify_telegram(f"⚠️ Машину {gpu} получить не удалось после {attempts} попыток. "
                    f"Обычно это значит, что свободных ускорителей нет: попробуйте позже "
                    f"или другой ускоритель (--gpu A100 / L4).")
    return False


def mount_drive(colab: str, session: str, wait_seconds: int, open_links: bool = True,
                gpu: str = "T4", attempt: int = 0) -> bool:
    """Подключает Drive. Возвращает True только если монтирование действительно есть.

    wait_seconds — сколько ждём ПОДТВЕРЖДЕНИЯ (появления ссылки и нажатия Enter). Само монтирование
    может идти дольше: если ссылки не было, CLI просто монтирует уже разрешённый Drive, и обрывать
    его на этом нельзя. Поэтому молчаливое ожидание ограничено отдельным, более щедрым потолком;
    а после любого исхода спрашиваем машину, смонтирован ли Drive на самом деле.
    """
    silent_cap = float(os.environ.get("DRIVE_MOUNT_WAIT", "180"))
    auto_enter = os.environ.get("DRIVE_AUTO_ENTER", "1") not in ("0", "no", "false")
    print("\n== подключаю Google Drive ==")
    if auto_enter:
        print("   если Google попросит подтверждение, покажу ссылку подсвеченной (она придёт и в Telegram):")
        print("   откройте её и разрешите доступ — нажимать ничего не нужно,")
        print(f"   через {wait_seconds} с Enter будет нажат автоматически, и обёртка сама проверит,")
        print("   дан ли доступ: дан — файлы поедут на Drive, нет — задание поедет без него")
    else:
        print("   если Google попросит подтверждение, покажу ссылку подсвеченной;")
        print(f"   подтвердите доступ и нажмите Enter здесь (ждём {wait_seconds} с после ссылки),")
        print("   иначе задание поедет без Drive и архив заберём локально")
    state, pressed = drive_mount_step([colab, "drivemount", "-s", session], wait_seconds=wait_seconds,
                                      silent_cap=max(float(wait_seconds), silent_cap),
                                      auto_enter=auto_enter, open_links=open_links)
    if state == "timeout":
        if pressed:
            print("  после нажатия Enter монтирование не завершилось — снимаю попытку.")
        else:
            print(f"  подтверждения не было за {wait_seconds} с после ссылки — снимаю попытку.")
    elif state == "silent-timeout":
        print(f"  монтирование не ответило за {silent_cap:.0f} с — снимаю попытку.")
    elif state == "failed":
        print("  CLI сообщил об ошибке монтирования.")
        if "Connection was lost" in LAST_OUTPUT and attempt < 2:
            # Обычно это значит, что машина умерла: без неё и сборка не пойдёт — лечим сразу.
            print("  связь с машиной потеряна — проверяю, отвечает ли она…")
            notify_telegram("⚠️ Связь с машиной Colab потеряна на шаге подключения Google Drive — "
                            "проверяю машину и продолжаю.")
            if probe_session(colab, session):
                print("  машина отвечает — повторяю монтирование.")
            else:
                print("  машина не отвечает — поднимаю заново.")
                notify_telegram("🔁 Машина не отвечает, поднимаю новую сессию.")
                if start_session(colab, session, gpu):
                    print("  новая машина готова — повторяю монтирование.")
                else:
                    print("  поднять машину не удалось.")
            return mount_drive(colab, session, wait_seconds, open_links=open_links, gpu=gpu,
                               attempt=attempt + 1)

    # Спрашиваем машину: CLI мог быть медленным, а Drive при этом уже подключён.
    for attempt in range(6):
        if drive_is_mounted(colab, session):
            if state != "ok":
                print("  но монтирование на машине есть — работаем с Drive.")
            notify_telegram("✅ Google Drive подключён")
            return True
        if state == "ok":
            time.sleep(2)          # CLI завершился, но файловая система могла появиться не сразу
    if state == "ok":
        print("  монтирования на машине не видно (доступ не подтверждён?).")
        notify_telegram("⚠️ Google Drive не подключился (подтверждение не дошло) — "
                        "продолжаю сборку, архив заберём локально.")
    print("  едем без Drive: архив всё равно заберём в загрузки.")
    print("  подключить позже можно вручную: colab drivemount -s " + session
          + " (готовый архив уедет на Drive при завершении задания)")
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
    """Куда складывать готовый архив: OUT_DIR, иначе папка загрузок текущей системы."""
    override = os.environ.get("OUT_DIR", "").strip()
    if override:
        path = pathlib.Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
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
    # urlretrieve не умеет таймаут: мёртвая ссылка вешала бы обёртку навсегда.
    with urllib.request.urlopen(url, timeout=120) as source, target.open("wb") as sink:
        shutil.copyfileobj(source, sink, 1 << 16)
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
                   drive_dir: str, job: pathlib.Path, shrink: dict, raw: dict = None) -> pathlib.Path:
    """Задание уходит на машину вместе с аргументами: colab exec аргументы не передаёт."""
    args = [book_remote]
    if fragment:
        args += ["--seconds", fragment]
    if highlight:
        args += ["--highlight", highlight]
    if raw:
        # Режим частей: только синтез, Drive не участвует, книга собирается в образе.
        args += ["--no-drive", "--no-shrink", "--raw", "--upto", "synth",
                 "--part-index", str(raw.get("part", ""))]
    else:
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


def fetch_raw(colab, args, stem) -> int:
    """Режим частей: забираем СЫРЬЁ части в том бота. Ни Drive, ни готового архива тут нет.

    Часть сразу собирается как самостоятельная книга — но В ОБРАЗЕ, а не на машине: иначе её
    пришлось бы кодировать в облаке, а потом раскодировать для финала, и книга получила бы
    вторую генерацию AAC-потерь.
    """
    parts_dir = pathlib.Path(args.parts_dir or downloads_dir())
    parts_dir.mkdir(parents=True, exist_ok=True)
    remote_zip = f"/content/out/{stem}-raw.zip"
    local_zip = parts_dir / f"{stem}-raw.zip"
    downloaded = False
    for attempt in (1, 2):
        if run([colab, "download", "-s", args.session, remote_zip, str(local_zip)],
               timeout=3600).returncode == 0:
            downloaded = True
            break
        print(f"  сырьё не забралось с попытки {attempt} — пробую ещё раз", flush=True)
    if not downloaded:
        print(f"  сырья на машине нет: {remote_zip}")
        notify_telegram(f"⚠️ Сырьё части {args.part_index} для «{stem}» не забралось.\n"
                        f"Повторить: colab download -s {args.session} {remote_zip} {local_zip}")
        if args.stop_on_failure:
            run_quiet([colab, "stop", "-s", args.session])
        return 1
    print(f"сырьё части {args.part_index}: {local_zip.name} ({local_zip.stat().st_size / 1e6:.1f} МБ)")
    notify_telegram(f"📦 Часть {args.part_index} озвучена, сырьё забрано в том ({stem}-raw.zip)")

    part = parts_dir / f"part-{args.part_index}"
    if part.exists():                            # распаковываем только сырьё этой части
        shutil.rmtree(part)
    part.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(part)
    local_zip.unlink(missing_ok=True)            # сырьё уже распаковано: архив в томе не нужен
    print(f"сырьё части {args.part_index} распаковано: {part}")

    # Часть становится самостоятельной книгой: без громкости, фильтра и музыки — как есть.
    glue = pathlib.Path(__file__).resolve().parent.parent.parent / "glue_book.py"
    if not glue.is_file():
        print(f"нет сборщика частей {glue} — сырьё лежит в {part}, часть не собрана")
        return 1
    command = [sys.executable, str(glue), "--mode", "part", "--part", str(args.part_index),
               "--parts-dir", str(parts_dir), "--out", str(args.out_dir or downloads_dir())]
    print("== собираю часть в образе ==\n  " + " ".join(command), flush=True)
    code = subprocess.run(command, cwd=str(glue.parent)).returncode
    if code != 0:
        notify_telegram(f"⚠️ Часть {args.part_index} озвучена, но собрать её не вышло (код {code}). "
                        f"Сырьё на месте: {part}")
        return 1
    print(f"часть {args.part_index} готова: {args.out_dir or downloads_dir()}")
    notify_telegram(f"✅ Часть {args.part_index} «{stem}» готова (файлами в архивах)")
    if not args.keep_session:
        print("== освобождаю машину ==")
        run_quiet([colab, "stop", "-s", args.session])
    return 0


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
                        help="через сколько секунд нажать Enter и проверить доступ к Drive (60)")
    parser.add_argument("--stop-on-failure", action="store_true",
                        default=os.environ.get("STOP_ON_FAILURE", "0") not in ("0", "", "no", "false"),
                        help="гасить машину и при неудаче: ничего не оставляем запущенным")
    parser.add_argument("--no-auto-enter", action="store_true",
                        help="не нажимать Enter за меня: подтверждение придётся давать вручную")
    parser.add_argument("--parts-raw", action="store_true",
                        default=os.environ.get("PARTS_RAW", "0") == "1",
                        help="режим частей: часть идёт только на синтез, сырьё забирается в том, "
                             "часть и финал собираются в образе; Drive не участвует")
    parser.add_argument("--part-index", default=os.environ.get("PART_INDEX", ""),
                        help="номер части в режиме частей")
    parser.add_argument("--parts-dir", default=os.environ.get("PARTS_DIR", ""),
                        help="куда складывать сырьё частей (том бота)")
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("SHRINK_HOURS", "7")),
                        help="если книга длиннее — задание разрежет её на части (0 — не резать)")
    parser.add_argument("--shrink-dir", default=os.environ.get("SHRINK_DIR", "Audiobooks-shrink"),
                        help="папка на Drive для остальных частей")
    parser.add_argument("--no-shrink", action="store_true", default=os.environ.get("SHRINK", "1") == "0",
                        help="не резать книгу на части")
    parser.add_argument("--open", action="store_true", default=os.environ.get("AUTO_OPEN", "0") == "1",
                        help="открывать браузер со ссылкой авторизации Drive "
                             "(по умолчанию выключено: ссылка просто подсвечивается)")
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR"),
                        help="куда складывать готовый архив (по умолчанию ~/Downloads)")
    parser.add_argument("--telegram-token", default=None,
                        help="ключ бота от @BotFather: шлём стадии и итог в Telegram")
    parser.add_argument("--telegram-chat", default=None,
                        help="куда слать (не обязательно: найдём чат по последнему сообщению боту)")
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

    if args.telegram_token:                      # ключи можно задать и ключом, и переменной
        os.environ["TELEGRAM_BOT_TOKEN"] = args.telegram_token
    if args.telegram_chat:
        os.environ["TELEGRAM_CHAT_ID"] = args.telegram_chat
    sys.path.insert(0, str(_repo_root()))        # telegram_notify.py лежит в корне репозитория
    try:
        import telegram_notify                                   # чат из прошлых запусков
        if not os.environ.get("TELEGRAM_CHAT_ID") and telegram_notify.saved_chat():
            os.environ["TELEGRAM_CHAT_ID"] = telegram_notify.saved_chat()
    except Exception:
        pass
    if os.environ.get("TELEGRAM_BOT_TOKEN") and not os.environ.get("TELEGRAM_CHAT_ID"):
        # Первый запуск с ключом: чат узнаём, как только боту напишут (бот не может писать первым).
        try:
            import telegram_notify
            chat = telegram_notify.wait_for_chat(
                os.environ["TELEGRAM_BOT_TOKEN"],
                float(os.environ.get("TELEGRAM_WAIT_CHAT", "45")))
        except Exception:
            chat = ""
        if chat:
            os.environ["TELEGRAM_CHAT_ID"] = chat

    if args.out_dir:                            # бот в контейнере складывает архив в свой том
        os.environ["OUT_DIR"] = str(args.out_dir)

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
    shrink = {"max_hours": args.max_hours, "dir": args.shrink_dir,
              "off": args.no_shrink or args.parts_raw}      # в режиме частей книгу режет бот, не машина

    def fail() -> int:
        print("\n== не получилось, освобождаю машину ==")
        notify_telegram(f"⚠️ Не получилось поднять машину или залить книгу «{stem}» — смотрите вывод.")
        if not args.keep_session:
            run_quiet([colab, "stop", "-s", args.session])
        return 1

    remaining = stop_stale_sessions(colab)
    if re.search(r"^\[\?\]", remaining, re.M):
        print("  ВНИМАНИЕ: сессия «?» всё ещё висит. Значит освободить её из терминала не вышло —")
        print("  завершите вручную: colab.research.google.com → Manage sessions → Terminate.")

    print(f"\n== поднимаю сессию {args.session} ({args.gpu}) ==")
    notify_telegram(f"🚀 Поднимаю машину Colab ({args.gpu}) для «{stem}»")
    if not start_session(colab, args.session, args.gpu):
        print("  машина не поднялась или кернел не отвечает.")
        return fail()

    drive_ok = False
    if args.parts_raw:
        print("  режим частей: Google Drive не участвует — сырьё вернётся в том бота")
    elif not args.no_drive:
        if args.no_auto_enter:
            os.environ["DRIVE_AUTO_ENTER"] = "0"
        drive_ok = mount_drive(colab, args.session, args.drive_wait, gpu=args.gpu,
                               open_links=args.open and not args.no_open)
        if drive_ok:
            print(f"  Drive подключён: book.m4b и read-along epub поедут в MyDrive/{args.drive_dir}/<имя книги>")

    print(f"\n== книга на машину: {stem} ==")
    # Таймаут обязателен: зависшая сеть без него держит обёртку вечно.
    if run([colab, "upload", "-s", args.session, str(book), remote_book], timeout=3600).returncode != 0:
        return fail()

    generated = build_job_file(remote_book, args.fragment, args.highlight, drive_ok, args.drive_dir,
                               job, shrink, {"part": args.part_index} if args.parts_raw else None)
    print("\n== задание ==")
    try:
        # --timeout у colab exec по умолчанию 30 секунд: на книгу этого мало, ждём до суток.
        command = [colab, "exec", "-s", args.session, "-f", str(generated), "--timeout", "86400"]
        for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):     # уведомления работают и с машины
            if os.environ.get(key):
                command += ["--env", f"{key}={os.environ[key]}"]
        try:
            code = run(command, timeout=90000).returncode     # сутки задания + запас
        except subprocess.TimeoutExpired:
            print("== CLI не вернулся после задания — снимаю попытку (на машине оно могло ещё идти) ==")
            code = 2
    finally:
        generated.unlink(missing_ok=True)
    if code != 0:
        print(f"== CLI вернул код {code} — это может быть таймаут ожидания кернела, а не ошибка задания")

    print("\n== забираю архив ==")
    if args.parts_raw:
        return fetch_raw(colab, args, stem)
    # Толстая книга режется на части ещё на машине: узнаём по плану, как называется архив.
    plan = fetch_plan(colab, args.session, stem, downloads_dir())
    if plan and plan.get("parts", 1) <= 1:
        print(f"оценка: книга ≈{plan['estimated_hours']:.2f} ч звука (~столько же часов GPU)")
    effective = f"{stem}-part1" if plan and plan.get("parts", 1) > 1 else stem
    remote_zip = f"/content/out/{effective}.zip"
    local_zip = downloads_dir() / f"{effective}.zip"

    # Машину при неудаче НЕ гасим: готовый архив может уже лежать на ней, а после stop он пропадёт.
    downloaded = run([colab, "download", "-s", args.session, remote_zip, str(local_zip)],
                     timeout=3600).returncode == 0
    if not downloaded:
        print("архив не забрался — пробую ещё раз (бывает сетевая осечка)")
        downloaded = run([colab, "download", "-s", args.session, remote_zip, str(local_zip)],
                         timeout=3600).returncode == 0
    if not downloaded:
        print("архива на машине нет (или он ещё собирается).")
        if args.stop_on_failure:
            print("== освобождаю машину (--stop-on-failure: ничего не оставляем запущенным) ==")
            run_quiet([colab, "stop", "-s", args.session])
            notify_telegram(f"⚠️ Архив для «{stem}» не забрался; машина освобождена "
                            f"(сессия {args.session} остановлена).")
        else:
            print("машину не гашу, чтобы не потерять работу. Дальше вручную:")
            print(f"  colab download -s {args.session} {remote_zip} {local_zip}")
            print(f"  colab stop -s {args.session}")
            notify_telegram(f"⚠️ Архив для «{stem}» не забрался, машина оставлена запущенной "
                            f"(сессия {args.session}).\nОсвободить: colab stop -s {args.session}")
        return 1
    if local_zip.is_file():
        print(f"{sha256_of(local_zip)}  {local_zip.name}")
    print(f"готово: {local_zip}")
    notify_telegram(f"✅ Архив готов: {local_zip.name}\n{sha256_of(local_zip)[:16]}… · {local_zip}")

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
