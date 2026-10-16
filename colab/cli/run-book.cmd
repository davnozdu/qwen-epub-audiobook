@echo off
rem run-book.cmd — обёртка для Windows. Вся работа в run-book.py (он же для macOS и Linux).
rem
rem   run-book.cmd kniga.epub
rem   run-book.cmd C:\Books\kniga.epub
rem   run-book.cmd https://site.ru/kniga.epub
rem
rem Переменные: set GPU=L4, set SESSION=book, set FRAGMENT=300, set HIGHLIGHT=fragment,
rem             set KEEP_SESSION=1, set DRIVE=0, set DRIVE_DIR=Audiobooks-cli, set DRIVE_WAIT=120
setlocal
set "HERE=%~dp0"
if not exist "%HERE%run-book.py" (
    echo рядом с этим файлом нет run-book.py
    exit /b 1
)

rem Сначала пробуем лаунчер py (он есть при установке с python.org), потом обычный python
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%HERE%run-book.py" %*
) else (
    python "%HERE%run-book.py" %*
)
set "CODE=%errorlevel%"
endlocal & exit /b %CODE%
