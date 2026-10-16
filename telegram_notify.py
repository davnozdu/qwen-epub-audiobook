#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Уведомления о сборке книги в Telegram — необязательные.

Ключ бота и чат берутся из окружения (или передаются аргументами):

    TELEGRAM_BOT_TOKEN=123456:AA…      ключ от @BotFather
    TELEGRAM_CHAT_ID=123456789         куда слать (не обязателен)

Если ключа нет — все функции молча ничего не делают, конвейер работает как обычно.
Если чат не задан, но боту уже писали, чат находится сам по последнему сообщению.

Секреты нигде не сохраняются: этот файл только читает окружение.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"
API = "https://api.telegram.org/bot{token}/{method}"
WARNED = set()          # чтобы не повторять одну и ту же подсказку на каждой стадии


def token_from_env() -> str:
    return (os.environ.get(TOKEN_ENV) or "").strip()


def chat_from_env() -> str:
    return (os.environ.get(CHAT_ENV) or "").strip()


def _call(token: str, method: str, payload: dict | None = None, timeout: int = 25) -> dict:
    data = urllib.parse.urlencode(payload or {}).encode()
    request = urllib.request.Request(API.format(token=token, method=method), data=data)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_chat(token: str) -> str:
    """Чат по последнему сообщению боту: напишите боту что угодно — он найдётся сам."""
    try:
        updates = _call(token, "getUpdates").get("result", [])
    except Exception:
        return ""
    chats = []
    for update in updates:
        message = update.get("message") or update.get("channel_post") or {}
        chat_id = (message.get("chat") or {}).get("id")
        if chat_id is not None:
            chats.append(str(chat_id))
    return chats[-1] if chats else ""


def notify(text: str, token: str | None = None, chat: str | None = None, log=print) -> bool:
    """Отправляет сообщение. Без ключа — тихо ничего не делает (и это нормально)."""
    token = (token or token_from_env()).strip()
    if not token:
        return False
    chat = (chat or chat_from_env()).strip()
    if not chat:
        chat = discover_chat(token)
        if not chat:
            if "nochat" not in WARNED:
                WARNED.add("nochat")
                log("  (уведомления в Telegram: чат не найден — напишите боту любое сообщение)")
            return False
    try:
        result = _call(token, "sendMessage", {"chat_id": chat, "text": text[:4000],
                                              "disable_web_page_preview": "true"})
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200] if hasattr(exc, "read") else str(exc)
        if detail not in WARNED:
            WARNED.add(detail)
            log(f"  (Telegram ответил ошибкой: {detail})")
        return False
    except Exception as exc:
        message = str(exc)[:120]
        if message not in WARNED:
            WARNED.add(message)
            log(f"  (Telegram недоступен: {message})")
        return False
    if not result.get("ok"):
        log(f"  (Telegram не принял сообщение: {result.get('description')})")
        return False
    return True


def progress(text: str, log=print) -> bool:
    """Короткая обёртка для стадий конвейера."""
    return notify(text, log=log)


if __name__ == "__main__":
    import sys

    if not token_from_env():
        print(f"нет ключа: задайте {TOKEN_ENV} (и при желании {CHAT_ENV})")
        raise SystemExit(0)
    message = " ".join(sys.argv[1:]) or "Проверка связи: уведомления работают."
    print("отправлено:", notify(message))
