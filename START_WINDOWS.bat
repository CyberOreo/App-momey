@echo off
setlocal enabledelayedexpansion
title PolyBTC Trader
color 0B

:: Run from the folder this file lives in
cd /d "%~dp0"

echo.
echo  =============================================
echo   PolyBTC Trader
echo  =============================================
echo.

:: ── Find Python 3.10+ ──────────────────────────────────────────────────────────
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
    echo    2. Download Python 3.11 or newer
    echo    3. Run the installer
    echo    4. Tick "Add Python to PATH"   ^<-- important
    echo    5. Restart this script
    echo.
    pause
    exit /b 1
)

echo  Found: !PY!
echo.

:: ── Create .env on first run ───────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo  Created .env from .env.example
    ) else (
        (
            echo PAPER_TRADING=true
            echo PAPER_BALANCE=1000
            echo MIN_CONFIDENCE_THRESHOLD=65
            echo MAX_RISK_PER_TRADE_PCT=0.02
            echo MAX_DAILY_DRAWDOWN_PCT=0.05
            echo MAX_CONSECUTIVE_LOSSES=3
        ) > ".env"
        echo  Created default .env
    )
    echo  Opening .env - add your Polymarket private key, save and close Notepad.
    echo.
    notepad ".env"
    echo.
)

:: ── Create data folder ────────────────────────────────────────────────────────
if not exist "data" mkdir data

:: ── Create virtual environment ────────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo  Setting up virtual environment (first time only^)...
    !PY! -m venv venv
    if !errorlevel! neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        echo  Try: !PY! -m pip install virtualenv
        pause
        exit /b 1
    )
    echo  Installing dependencies...
    venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    venv\Scripts\pip.exe install -r requirements_core.txt --quiet
    if !errorlevel! neq 0 (
        echo  [ERROR] Failed to install dependencies.
        echo  Check your internet connection and try again.
        pause
        exit /b 1
    )
    echo  Dependencies installed successfully.
    echo.
)

:: ── Launch ────────────────────────────────────────────────────────────────────
echo  =============================================
echo   Dashboard:  http://localhost:8080
echo   Browser opens automatically in 2 seconds.
echo   Close this window to stop everything.
echo  =============================================
echo.

venv\Scripts\python.exe web_app.py

echo.
echo  Stopped. Press any key to close.
pause >nul
