#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo "  PolyBTC Trader - Starting..."
echo "  =============================="
echo ""

if ! command -v python3 &>/dev/null; then
    echo "  ERROR: Python 3 not found. Install with:"
    echo "  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env — edit it with your API keys, then restart."
    nano .env
    exit 0
fi

if [ ! -d "venv" ]; then
    echo "  First run — creating virtual environment..."
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt -q
    echo "  Dependencies installed."
fi

mkdir -p data

echo "  Dashboard → http://localhost:8080"
echo "  Ctrl+C to stop."
echo ""

venv/bin/python web_app.py
