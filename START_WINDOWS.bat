@echo off
title PolyBTC Trader
color 0B

echo.
echo  PolyBTC Trader - Starting...
echo  ==========================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found.
    echo  Please install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Create .env if missing
if not exist ".env" (
    echo  First run detected - copying .env.example to .env
    copy .env.example .env >nul
    echo  Please edit .env with your settings, then restart.
    echo.
    start notepad .env
    pause
    exit /b 0
)

:: Create venv if missing
if not exist "venv\" (
    echo  Setting up virtual environment (first time only)...
    python -m venv venv
    echo  Installing dependencies...
    call venv\Scripts\pip install -r requirements.txt -q
    echo  Done!
    echo.
)

:: Create data dir
if not exist "data\" mkdir data

echo  Launching dashboard at http://localhost:8080
echo  A browser window will open automatically.
echo  Close this window to stop the bot.
echo.

call venv\Scripts\python web_app.py
pause
