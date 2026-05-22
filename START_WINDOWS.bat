@echo off
title PolyBTC Trader
color 0B

:: ── Critical: run from the folder this file is in ────────────────────────────
cd /d "%~dp0"

echo.
echo  PolyBTC Trader
echo  ==========================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
set PYTHON_CMD=

where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if %errorlevel% equ 0 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        python3 -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if %errorlevel% equ 0 set PYTHON_CMD=python3
    )
)

if "%PYTHON_CMD%"=="" (
    echo  [ERROR] Python 3.10 or newer not found.
    echo.
    echo  How to fix:
    echo    1. Go to  https://python.org/downloads
    echo    2. Download Python 3.11 or newer
    echo    3. Run the installer
    echo    4. CHECK the box that says "Add Python to PATH"
    echo    5. Restart your PC, then double-click this file again
    echo.
    pause
    exit /b 1
)

echo  Python OK:
%PYTHON_CMD% --version
echo.

:: ── Create .env on first run ──────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    ) else (
        echo PAPER_TRADING=true>  ".env"
        echo PAPER_BALANCE=1000>> ".env"
        echo MIN_CONFIDENCE_THRESHOLD=65>> ".env"
        echo MAX_RISK_PER_TRADE_PCT=0.02>> ".env"
        echo MAX_DAILY_DRAWDOWN_PCT=0.05>> ".env"
    )
    echo  First run: opening .env so you can add your Polymarket key.
    echo  Save the file, then close Notepad and the bot will continue.
    echo.
    notepad ".env"
    echo.
)

:: ── Create virtual environment on first run ───────────────────────────────────
if not exist "venv\" (
    echo  One-time setup: creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Could not create virtual environment.
        echo  Make sure Python was installed with the standard options.
        pause
        exit /b 1
    )

    echo  Installing packages (takes 1-2 minutes, only needed once)...
    venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    venv\Scripts\pip.exe install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Failed to install packages.
        echo  Check your internet connection and try again.
        echo  If the problem persists, delete the "venv" folder and retry.
        pause
        exit /b 1
    )
    echo.
    echo  Setup complete!
    echo.
)

:: ── Create data folder ────────────────────────────────────────────────────────
if not exist "data\" mkdir data

:: ── Launch ────────────────────────────────────────────────────────────────────
echo  ============================================
echo   Dashboard:  http://localhost:8080
echo   Browser will open in a few seconds.
echo   Close this window to stop the bot.
echo  ============================================
echo.

venv\Scripts\python.exe web_app.py

echo.
echo  Bot has stopped.
pause
