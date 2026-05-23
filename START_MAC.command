#!/bin/bash
# Double-click this file on Mac to start PolyBTC Trader

cd "$(dirname "$0")"

echo ""
echo "  PolyBTC Trader - Starting..."
echo "  =============================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "  ERROR: Python 3 not found."
    echo "  Install from https://python.org or run: brew install python3"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Create .env from example if missing
if [ ! -f ".env" ]; then
    echo "  First run — copying .env.example to .env"
    cp .env.example .env
    echo "  Please edit .env with your settings, then restart."
    echo ""
    open -e .env 2>/dev/null || nano .env
    exit 0
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "  Setting up virtual environment (first time only)..."
    python3 -m venv venv
    echo "  Installing dependencies..."
    venv/bin/pip install -r requirements_core.txt -q
    echo "  Done!"
    echo ""
fi

mkdir -p data

echo "  Launching dashboard at http://localhost:8080"
echo "  Your browser will open automatically."
echo "  Press Ctrl+C to stop."
echo ""

venv/bin/python web_app.py
