"""
PolyBTC Trader — Web Dashboard
Run: python web_app.py
Then open: http://localhost:8080
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PolyBTC Trader")

# ── Serve static files ────────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Bot process handle ────────────────────────────────────────────────────────
_bot_process: Optional[subprocess.Popen] = None
_log_lines: list[str] = []
_log_lock = threading.Lock()
MAX_LOG_LINES = 200


def _stream_logs(proc: subprocess.Popen) -> None:
    """Read bot stdout/stderr into the in-memory log buffer."""
    for raw in proc.stdout:  # type: ignore[union-attr]
        line = raw.rstrip("\n")
        with _log_lock:
            _log_lines.append(line)
            if len(_log_lines) > MAX_LOG_LINES:
                _log_lines.pop(0)


def _is_running() -> bool:
    return _bot_process is not None and _bot_process.poll() is None


# ── DB helpers (read-only, safe for SQLite WAL) ───────────────────────────────

def _db_path() -> Optional[str]:
    url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/trading.db")
    if url.startswith("sqlite"):
        p = url.split("///")[-1]
        return str(Path(p).resolve())
    return None


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    try:
        import sqlite3
        db = _db_path()
        if not db or not Path(db).exists():
            return []
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _get_stats() -> dict:
    trades = _query_db(
        "SELECT outcome, realized_pnl, entry_time, exit_time, direction, "
        "entry_price, exit_price, size, market_id FROM trades ORDER BY entry_time DESC LIMIT 100"
    )
    open_pos = _query_db(
        "SELECT * FROM positions WHERE open=1"
    )
    today = date.today().isoformat()
    today_closed = [
        t for t in trades
        if t.get("exit_time", "") and t["exit_time"][:10] == today
    ]
    wins = [t for t in today_closed if t.get("outcome") == "win"]
    losses = [t for t in today_closed if t.get("outcome") == "loss"]
    daily_pnl = sum(t.get("realized_pnl") or 0 for t in today_closed)
    total_pnl = sum(t.get("realized_pnl") or 0 for t in trades if t.get("outcome") != "open")
    all_closed = [t for t in trades if t.get("outcome") and t["outcome"] != "open"]
    all_wins = [t for t in all_closed if t.get("outcome") == "win"]
    win_rate = round(len(all_wins) / len(all_closed) * 100, 1) if all_closed else 0.0

    return {
        "running": _is_running(),
        "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "daily_pnl": round(daily_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "win_rate": win_rate,
        "total_trades": len(all_closed),
        "open_positions": len(open_pos),
        "trades_today": len(today_closed),
        "wins_today": len(wins),
        "losses_today": len(losses),
        "recent_trades": trades[:15],
        "open_pos_list": open_pos,
    }


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>index.html not found in static/</h1>", status_code=500)


@app.get("/api/status")
async def api_status():
    return JSONResponse(_get_stats())


@app.get("/api/logs")
async def api_logs():
    with _log_lock:
        return JSONResponse({"lines": list(_log_lines[-100:])})


@app.post("/api/start")
async def api_start(request: Request):
    global _bot_process, _log_lines
    if _is_running():
        return JSONResponse({"ok": False, "msg": "Bot is already running"})
    body = await request.json()
    live = body.get("live", False)
    args = [sys.executable, "scripts/run_bot.py"]
    if live:
        args.append("--live")
    with _log_lock:
        _log_lines.clear()
    try:
        _bot_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).parent),
        )
        t = threading.Thread(target=_stream_logs, args=(_bot_process,), daemon=True)
        t.start()
        return JSONResponse({"ok": True, "msg": "Bot started", "pid": _bot_process.pid})
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)})


@app.post("/api/stop")
async def api_stop():
    global _bot_process
    if not _is_running():
        return JSONResponse({"ok": False, "msg": "Bot is not running"})
    _bot_process.terminate()  # type: ignore[union-attr]
    try:
        _bot_process.wait(timeout=5)  # type: ignore[union-attr]
    except subprocess.TimeoutExpired:
        _bot_process.kill()  # type: ignore[union-attr]
    return JSONResponse({"ok": True, "msg": "Bot stopped"})


@app.post("/api/settings")
async def api_settings(request: Request):
    data = await request.json()
    env_path = Path(__file__).parent / ".env"
    # Read existing .env
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
    # Merge allowed keys
    allowed = {
        "POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY",
        "POLYMARKET_API_SECRET", "POLYMARKET_PASSPHRASE",
        "PAPER_TRADING", "PAPER_BALANCE",
        "MAX_RISK_PER_TRADE_PCT", "MAX_DAILY_DRAWDOWN_PCT",
        "MIN_CONFIDENCE_THRESHOLD", "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID", "TELEGRAM_ENABLED",
        "MAX_CONSECUTIVE_LOSSES",
    }
    for k, v in data.items():
        if k.upper() in allowed and v != "":
            existing[k.upper()] = str(v)
    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n")
    # Reload into current process env
    for k, v in existing.items():
        os.environ[k] = v
    return JSONResponse({"ok": True})


@app.get("/api/config")
async def api_config():
    """Return current (non-secret) config values for the settings form."""
    return JSONResponse({
        "paper_trading": os.getenv("PAPER_TRADING", "true"),
        "paper_balance": os.getenv("PAPER_BALANCE", "1000"),
        "max_risk_per_trade_pct": os.getenv("MAX_RISK_PER_TRADE_PCT", "0.02"),
        "max_daily_drawdown_pct": os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05"),
        "min_confidence_threshold": os.getenv("MIN_CONFIDENCE_THRESHOLD", "65"),
        "max_consecutive_losses": os.getenv("MAX_CONSECUTIVE_LOSSES", "3"),
        "telegram_enabled": os.getenv("TELEGRAM_ENABLED", "false"),
        "has_private_key": bool(os.getenv("POLYMARKET_PRIVATE_KEY")),
        "has_telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                k = k.strip()
                if k and k not in os.environ:
                    os.environ[k] = v.strip()


if __name__ == "__main__":
    _load_dotenv()
    import webbrowser, time

    def _open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8080")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("\n  PolyBTC Trader dashboard → http://localhost:8080\n")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
