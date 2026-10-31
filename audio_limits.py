#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Предел обычного WAV: 4 ГиБ.

RIFF хранит длину данных в 32 битах. Дорожка книги длиннее ≈24.8 ч при 24 кГц/16 бит в
обычный WAV не влезает: звук пишется целиком, а в заголовок попадает максимум 0xFFFFFFFF
байт, и читатель (soundfile, ffmpeg, ffprobe) видит обрезанную длительность.

Так «Идиот» на 25.61 ч терял последние 45 минут: 2 212 706 712 кадров = 4 425 413 424 байта,
а из заголовка выходило 89 478.485 с вместо 92 196.113 с — сборка честно падала с
«дорожка книги вышла … с вместо … с», и по этой строке причина не читалась.

Поэтому большую дорожку пишем как RF64 (тот же WAV, но с 64-битным размером): soundfile и
ffmpeg его читают, ffprobe тоже. Внимание: стандартный модуль `wave` RF64 НЕ читает
(«file does not start with RIFF id»), поэтому длину и форму файла проверять через
`shape()` ниже, а не через `wave`.
"""

# Запас на заголовки и выравнивание: файл не должен упираться в предел вплотную.
LIMIT = 0xFFFFFFFF - 4096


def wav_format(frames: int, bytes_per_frame: int) -> str:
    """Формат для soundfile: «RF64», если обычный WAV столько не удержит, иначе «WAV»."""
    return "RF64" if int(frames) * int(bytes_per_frame) > LIMIT else "WAV"


def too_big(frames: int, bytes_per_frame: int) -> bool:
    return int(frames) * int(bytes_per_frame) > LIMIT


def shape(path):
    """Кадры, частота и каналы из заголовка: верно и для WAV, и для RF64 (в отличие от `wave`)."""
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames, info.samplerate, info.channels
