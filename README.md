# PolyBTC Trader

Automated paper/live trading bot for Polymarket's **5-minute BTC Up/Down** markets.

Every second the engine reads live Binance price data and Polymarket CLOB bid/ask prices, computes a directional edge score (0–100), and enters a position when confidence is high enough.

> **Warning:** Prediction market trading carries substantial risk of loss. Start in paper mode and run for at least a week before considering real funds.

---

## How it works

The engine combines six independent signals:

| Signal | Source |
|---|---|
| Window delta | BTC price move since 5-min window opened |
| Price velocity | Speed of BTC move in last 30–90 seconds |
| VPIN | Volume-imbalance (buy vs. sell pressure) |
| Chainlink oracle | On-chain BTC/USD confirmation |
| Multi-exchange consensus | Binance + Coinbase + Bybit agreement |
| Funding rate | Perpetual futures market bias |

A trade fires only when the combined score exceeds the confidence threshold (default 65).

---

## Quick start (Windows)

```
Double-click START_WINDOWS.bat
```

Or manually:
```bash
pip install -r requirements_core.txt
python web_app.py
```

Then open `http://localhost:8080` in your browser.

---

## Quick start (Mac / Linux)

```bash
./START_MAC.command      # Mac
./START_LINUX.sh         # Linux
```

---

## iPhone / Mobile

Open `http://YOUR_PC_IP:8080` in Safari and tap **Add to Home Screen**.

For 24/7 access from anywhere, deploy to Railway (see below).

---

## Cloud deployment (Railway)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables in the Railway dashboard:
   ```
   PAPER_TRADING=true
   PAPER_BALANCE=1000
   ```
4. Generate a public domain under Settings → Networking

The `railway.toml` and `Dockerfile` handle everything automatically.

**Note:** Polymarket's CLOB API may block some cloud IPs. Binance price feeds always work; Polymarket live bid/ask prices might show 0.000 from cloud.

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `PAPER_TRADING` | `true` | Set to `false` for real orders |
| `PAPER_BALANCE` | `1000` | Starting paper balance in USDC |
| `MIN_CONFIDENCE_THRESHOLD` | `65` | Minimum score to enter a trade (0–100) |
| `MAX_RISK_PER_TRADE_PCT` | `0.02` | Max 2% of balance per trade |
| `MAX_DAILY_DRAWDOWN_PCT` | `0.05` | Stop trading if daily loss exceeds 5% |
| `TELEGRAM_BOT_TOKEN` | — | Optional Telegram alerts |

---

## Project structure

```
web_app.py              Web dashboard (serves http://localhost:8080)
scripts/run_bot.py      Bot entry point (launched by dashboard)
static/index.html       Dashboard frontend
src/
  core/                 Config, database, models
  market/               BTC price feeds + Polymarket CLOB feed
    btc5min.py          Polymarket live YES/NO prices every 1s
    chainlink.py        On-chain oracle via Polygon RPC
    multi_exchange.py   Binance + Coinbase + Bybit consensus
    funding_rate.py     Perpetual futures funding signal
    scanner.py          Market discovery
  trading/
    five_min_engine.py  Main decision engine (1-second loop)
  connectors/
    polymarket.py       Polymarket CLOB API client
  monitoring/
    telegram_alerts.py  Optional Telegram notifications
```

---

## Requirements

Core (no C compiler needed — all pre-built wheels):
```
aiohttp, aiosqlite, loguru, numpy, pydantic, pydantic-settings,
python-dotenv, rich, SQLAlchemy, websockets
```

Install: `pip install -r requirements_core.txt`
