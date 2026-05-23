@echo off
title PolyBTC Trader
color 0B

:: Run from the folder this file lives in (critical for relative paths)
cd /d "%~dp0"

echo.
echo  =============================================
echo   PolyBTC Trader
echo  =============================================
echo.

:: ── Find Python ───────────────────────────────────────────────────────────────
set PY=

for %%C in (python python3 py) do (
    if "!PY!"=="" (
        %%C --version >nul 2>&1
        if !errorlevel! equ 0 (
            %%C -c "import sys;exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set PY=%%C
        )
    )
)

:: Enable delayed expansion for the loop above
setlocal enabledelayedexpansion
set PY=

for %%C in (python python3 py) do (
    if "!PY!"=="" (
        %%C --version >nul 2>&1
        if !errorlevel! equ 0 (
            %%C -c "import sys;exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set PY=%%C
        )
    )
)

if "!PY!"=="" (
    echo  [ERROR] Python 3.10 or newer not found.
    echo.
    echo  Fix:
    echo    1. Go to https://python.org/downloads
    echo    2. Download Python 3.11 (or newer)
    echo    3. Run the installer
    echo    4. Tick the box "Add Python to PATH"   ^<-- important
    echo    5. Restart this script
    echo.
    pause
    exit /b 1
)

echo  Found: !PY!
!PY! --version
echo.

:: ── Create .env on first run ──────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    ) else (
        (
            echo PAPER_TRADING=true
            echo PAPER_BALANCE=1000
            echo MIN_CONFIDENCE_THRESHOLD=65
            echo MAX_RISK_PER_TRADE_PCT=0.02
            echo MAX_DAILY_DRAWDOWN_PCT=0.05
            echo MAX_CONSECUTIVE_LOSSES=3
        ) > ".env"
    )
    echo  Opening .env — add your Polymarket private key, then save and close Notepad.
    echo.
    notepad ".env"
    echo.
)

:: ── Create data folder ────────────────────────────────────────────────────────
if not exist "data" mkdir data

:: ── Launch (web_app.py uses only Python built-ins — no pip install needed) ───
echo  =============================================
echo   Dashboard:  http://localhost:8080
echo   Browser opens automatically in 2 seconds.
echo   Close this window to stop everything.
echo  =============================================
echo.

!PY! web_app.py

echo.
echo  Stopped. Press any key to close.
pause >nul
