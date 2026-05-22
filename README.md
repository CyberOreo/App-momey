# PolyBTC Trader

A systematic, quantitative trading bot for Polymarket BTC binary prediction markets.
Combines multi-timeframe technical analysis with statistical edge detection to
find mispricings in BTC price-level markets, then executes at appropriate risk-
adjusted sizes.

> **Disclaimer:** This software is experimental. Prediction market trading carries
> substantial risk of loss. Past backtested performance does not predict future
> results. Only trade with funds you can afford to lose entirely. Start in paper
> trading mode and observe the system for weeks before committing real capital.

---

## System Overview

PolyBTC Trader monitors active Polymarket markets of the form "Will BTC be above
$X by [date]?" and generates YES or NO signals when:

1. BTC technical indicators align with the market's directional question.
2. Polymarket's implied probability diverges from our model's fair-value estimate.
3. All risk filters pass (volatility regime, time-to-resolution, liquidity).

The system is modular, fully paper-tradeable, and produces detailed performance
analytics including Sharpe ratio, max drawdown, and trade-by-trade journals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PolyBTC Trader                              │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │ Binance Feed │   │  Polymarket  │   │   Telegram Alerter   │   │
│  │  WebSocket   │   │ CLOB Client  │   │  (alerts & summary)  │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────────────────┘   │
│         │ candles           │ markets                               │
│         ▼                   ▼                                       │
│  ┌──────────────────────────────────────┐                          │
│  │         IndicatorEngine              │                          │
│  │  EMA(20/50/200) · RSI · MACD · ATR  │                          │
│  │  Volume · Momentum · Bollinger       │                          │
│  └──────────────────┬───────────────────┘                          │
│                     ▼                                               │
│  ┌──────────────────────────────────────┐                          │
│  │     MultiTimeframeAnalyzer           │                          │
│  │  Scores 1h · 4h · 15m · 5m          │                          │
│  │  Consensus direction · Fair value    │                          │
│  └──────────────────┬───────────────────┘                          │
│                     ▼                                               │
│  ┌──────────────────────────────────────┐                          │
│  │         SignalGenerator              │                          │
│  │  Long (YES) · Short (NO) filters     │                          │
│  │  Edge gate · Veto filters            │                          │
│  └──────────────────┬───────────────────┘                          │
│                     ▼                                               │
│  ┌──────────────────────────────────────┐                          │
│  │         ConfidenceScorer             │                          │
│  │  Trend · Momentum · Volume · Edge   │                          │
│  │  Timeframe agreement · Penalties    │                          │
│  └──────────────────┬───────────────────┘                          │
│                     ▼                                               │
│  ┌──────────────────────────────────────┐                          │
│  │           RiskManager                │                          │
│  │  Kelly sizing · Drawdown limits      │                          │
│  │  Kill switch · Cooldown · Exposure   │                          │
│  └──────────────────┬───────────────────┘                          │
│                     ▼                                               │
│  ┌──────────────────────────────────────┐                          │
│  │    PaperTrader / TradeExecutor        │                          │
│  │  SQLite/Postgres persistence          │                          │
│  │  Stop-loss · Take-profit tracking    │                          │
│  └──────────────────────────────────────┘                          │
│                                                                     │
│  Analytics: PerformanceAnalyzer · BacktestEngine · TradeJournal     │
│  Intel:     RegimeDetector · SentimentAnalyzer · MLScorer           │
│  Monitoring: Dashboard (Rich TUI) · MetricsCollector                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (5 Steps)

### Prerequisites
- Python 3.11+
- pip or uv

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd App-momey
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set PAPER_TRADING=true (already default)
# No API keys needed for paper trading with SQLite
```

### 3. Create runtime directories

```bash
mkdir -p data logs
```

### 4. Run the backtest to verify the setup

```bash
python scripts/run_backtest.py --days 30 --balance 1000
```

### 5. Start the paper trading bot

```bash
python scripts/run_bot.py
```

---

## Configuration Reference

All configuration is via environment variables (see `.env.example` for the full
list). Key settings:

| Variable | Default | Description |
|---|---|---|
| `PAPER_TRADING` | `true` | Always start in paper mode |
| `PAPER_BALANCE` | `1000.0` | Starting paper balance (USDC) |
| `MIN_CONFIDENCE_THRESHOLD` | `65.0` | Signal score required (0–100) |
| `MAX_DAILY_DRAWDOWN_PCT` | `0.05` | Halt after 5% daily loss |
| `MAX_CONSECUTIVE_LOSSES` | `3` | Losses before cooldown |
| `KELLY_FRACTION` | `0.25` | Quarter-Kelly position sizing |
| `MAX_POSITION_USDC` | `200.0` | Max size per trade |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.20` | Max 20% of balance deployed |

See `config/trading_params.yaml` for a fully annotated parameter reference.

---

## Paper Trading

Paper trading simulates the full execution pipeline — signals, position sizing,
stop-loss monitoring, and P&L calculation — but no real orders are sent to
Polymarket.

```bash
# Start paper trading (default mode)
python scripts/run_bot.py

# Run with debug logging
python scripts/run_bot.py --log-level DEBUG

# Run the terminal dashboard in a separate terminal
python scripts/run_dashboard.py
```

Paper trading results are stored in `data/trading.db` (SQLite). Export a CSV
journal at any time:

```python
# From Python REPL or a script:
import asyncio
from src.analytics.journal import TradeJournal
from src.core.database import init_db

async def export():
    db = await init_db("sqlite+aiosqlite:///data/trading.db")
    journal = TradeJournal(db=db)
    path = await journal.export_csv()
    print(f"Exported: {path}")

asyncio.run(export())
```

---

## Live Trading Setup

> **Warning:** Only proceed after running in paper mode for at least 2–4 weeks
> and observing stable, profitable performance.

### Step 1: Get Polymarket API credentials

1. Create a Polygon wallet and fund it with USDC
2. Log into polymarket.com and navigate to Profile → API Keys
3. Generate an API key, secret, and passphrase
4. Note your private key (the Ethereum key for your Polygon wallet)

### Step 2: Update .env

```bash
PAPER_TRADING=false
POLYMARKET_PRIVATE_KEY=0xYourPrivateKey
POLYMARKET_API_KEY=your-key
POLYMARKET_API_SECRET=your-secret
POLYMARKET_PASSPHRASE=your-passphrase
```

### Step 3: Start with a small balance

```bash
python scripts/run_bot.py --live
```

Monitor via Telegram alerts (configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`).

---

## Backtesting

```bash
# 30-day backtest, $1000 starting balance
python scripts/run_backtest.py --days 30 --balance 1000

# 90-day backtest with verbose logging
python scripts/run_backtest.py --days 90 --balance 5000 --verbose

# Custom output path and multiple timeframes
python scripts/run_backtest.py \
    --days 60 \
    --balance 2000 \
    --output data/my_backtest.csv \
    --timeframes 1h,4h,15m
```

The backtest runner:
1. Fetches real historical candles from Binance REST API
2. Falls back to synthetic data if Binance is unavailable
3. Generates realistic mock Polymarket markets
4. Replays the full signal pipeline chronologically
5. Prints a Rich-formatted performance summary
6. Exports a CSV trade journal

---

## Docker Deployment

### Single container (SQLite, simplest)

```bash
# Build
docker build -t polybtc-trader .

# Run (paper mode, environment from .env)
docker run -d \
    --name polybtc-bot \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    polybtc-trader

# View logs
docker logs -f polybtc-bot
```

### Docker Compose (recommended)

```bash
# Create required directories
mkdir -p data logs

# Start the bot
docker compose up -d bot

# View logs
docker compose logs -f bot

# Run a one-shot backtest
docker compose --profile tools run --rm backtest

# Stop
docker compose down
```

---

## VPS Deployment Guide

Recommended: Ubuntu 22.04 LTS, 1 vCPU, 1 GB RAM (e.g. DigitalOcean $6/month).

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv git screen

# 2. Clone and install
git clone <your-repo> ~/polybtc
cd ~/polybtc
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
nano .env   # edit your settings

# 4. Create directories
mkdir -p data logs

# 5. Run in a screen session (persists after SSH disconnect)
screen -S polybtc
python scripts/run_bot.py
# Ctrl+A then D to detach

# 6. Reattach later
screen -r polybtc
```

For production, consider using `systemd` for automatic restart on VPS reboot.
A sample service file:

```ini
# /etc/systemd/system/polybtc.service
[Unit]
Description=PolyBTC Trader
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/polybtc
ExecStart=/home/ubuntu/polybtc/.venv/bin/python scripts/run_bot.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

---

## Risk Disclaimer

**This software is provided for educational and research purposes only.**

- Prediction market trading is speculative and carries high risk of loss
- The system may generate incorrect signals; no trading system is infallible
- Past backtested performance does not guarantee future results
- Markets can become illiquid, manipulated, or resolve unexpectedly
- Smart contract bugs or API failures could result in total loss of funds
- Only allocate funds you are fully prepared to lose

**Recommended starting position:** $100–$500 in paper trading for at least
30 days before any live deployment.

---

## Performance Expectations

Based on the conservative parameter defaults:

- **Win rate:** 45–60% (binary markets are hard to beat consistently)
- **Profit factor:** 1.2–1.8 (target; depends heavily on edge quality)
- **Monthly return:** -5% to +15% (high variance, especially early on)
- **Max drawdown:** 10–25% during adverse periods

The system is designed for capital preservation, not maximising returns. The
quarter-Kelly sizing, daily drawdown halt, and kill switch are more important
than the signal logic for long-term survival.

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific test file
pytest tests/test_indicators.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Project Structure

```
App-momey/
├── src/
│   ├── core/           # Models, config, database, logging
│   ├── connectors/     # Binance WebSocket + REST, Polymarket CLOB
│   ├── market/         # IndicatorEngine, MultiTimeframeAnalyzer
│   ├── trading/        # SignalGenerator, ConfidenceScorer, TradeExecutor
│   ├── risk/           # RiskManager, position sizing
│   ├── analytics/      # PerformanceAnalyzer, BacktestEngine, TradeJournal
│   ├── intelligence/   # RegimeDetector, SentimentAnalyzer, MLScorer
│   └── monitoring/     # Dashboard (Rich), TelegramAlerter, MetricsCollector
├── tests/              # pytest test suite
├── scripts/            # run_bot.py, run_backtest.py, run_dashboard.py
├── config/             # trading_params.yaml
├── data/               # SQLite DB, ML model, journal CSVs (gitignored)
├── logs/               # Log files (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
