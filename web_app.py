"""
PolyBTC Trader — Web Dashboard
Uses ONLY Python standard library — no pip install needed to launch.
Run: python web_app.py
Then open: http://localhost:8080
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from datetime import date

# ── State ─────────────────────────────────────────────────────────────────────
_bot_process: Optional[subprocess.Popen] = None
_log_lines: list[str] = []
_log_lock = threading.Lock()
MAX_LOGS = 200

ROOT = Path(__file__).parent


def _is_running() -> bool:
    return _bot_process is not None and _bot_process.poll() is None


def _stream_logs(proc: subprocess.Popen) -> None:
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\n")
            with _log_lock:
                _log_lines.append(line)
                if len(_log_lines) > MAX_LOGS:
                    _log_lines.pop(0)
    except Exception:
        pass


# ── .env helpers ──────────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    result: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    return result


def _save_env(updates: dict[str, str]) -> None:
    env_path = ROOT / ".env"
    existing = _load_env()
    existing.update({k.upper(): v for k, v in updates.items() if v != ""})
    env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n",
        encoding="utf-8",
    )
    for k, v in existing.items():
        os.environ[k] = v


def _getenv(key: str, default: str = "") -> str:
    env = _load_env()
    return env.get(key, os.environ.get(key, default))


# ── DB helpers (read-only SQLite via stdlib) ──────────────────────────────────

def _db_path() -> Optional[str]:
    url = _getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/trading.db")
    if "sqlite" in url:
        raw = url.split("///")[-1]
        p = ROOT / raw
        return str(p) if p.exists() else None
    return None


def _query(sql: str, params: tuple = ()) -> list[dict]:
    try:
        import sqlite3
        db = _db_path()
        if not db:
            return []
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
        con.close()
        return rows
    except Exception:
        return []


def _get_stats() -> dict:
    trades = _query(
        "SELECT outcome, realized_pnl, entry_time, exit_time, direction, "
        "entry_price, exit_price, size, market_id FROM trades "
        "ORDER BY entry_time DESC LIMIT 100"
    )
    open_pos = _query("SELECT * FROM positions WHERE open=1")
    today = date.today().isoformat()
    today_closed = [t for t in trades if (t.get("exit_time") or "")[:10] == today]
    wins = [t for t in today_closed if t.get("outcome") == "win"]
    losses = [t for t in today_closed if t.get("outcome") == "loss"]
    daily_pnl = sum(t.get("realized_pnl") or 0 for t in today_closed)
    all_closed = [t for t in trades if t.get("outcome") not in (None, "open")]
    total_pnl = sum(t.get("realized_pnl") or 0 for t in all_closed)
    all_wins = [t for t in all_closed if t.get("outcome") == "win"]
    win_rate = round(len(all_wins) / len(all_closed) * 100, 1) if all_closed else 0.0
    paper = _getenv("PAPER_TRADING", "true").lower() == "true"

    # 5-min engine status (written every 2s by run_bot.py --five-min)
    engine: dict = {}
    engine_path = ROOT / "data" / "engine_status.json"
    try:
        if engine_path.exists() and (time.time() - engine_path.stat().st_mtime) < 10:
            engine = json.loads(engine_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    return {
        "running": _is_running(),
        "paper_trading": paper,
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
        "engine": engine,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            html = (ROOT / "static" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")

        elif path == "/api/status":
            self._json(_get_stats())

        elif path == "/api/logs":
            with _log_lock:
                self._json({"lines": list(_log_lines[-100:])})

        elif path == "/api/config":
            self._json({
                "paper_trading": _getenv("PAPER_TRADING", "true"),
                "paper_balance": _getenv("PAPER_BALANCE", "1000"),
                "min_confidence_threshold": _getenv("MIN_CONFIDENCE_THRESHOLD", "65"),
                "max_risk_per_trade_pct": _getenv("MAX_RISK_PER_TRADE_PCT", "0.02"),
                "max_daily_drawdown_pct": _getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05"),
                "max_consecutive_losses": _getenv("MAX_CONSECUTIVE_LOSSES", "3"),
                "telegram_enabled": _getenv("TELEGRAM_ENABLED", "false"),
                "has_private_key": bool(_getenv("POLYMARKET_PRIVATE_KEY")),
                "has_telegram": bool(_getenv("TELEGRAM_BOT_TOKEN")),
            })

        else:
            self._send(404, b"Not found")

    def do_POST(self):
        global _bot_process, _log_lines
        path = urlparse(self.path).path

        if path == "/api/start":
            if _is_running():
                self._json({"ok": False, "msg": "Already running"})
                return
            data = self._read_body()
            live = data.get("live", False)
            five_min = data.get("five_min", False)
            args = [sys.executable, str(ROOT / "scripts" / "run_bot.py")]
            if live:
                args.append("--live")
            if five_min:
                args.append("--five-min")
            with _log_lock:
                _log_lines.clear()
            try:
                _bot_process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(ROOT),
                )
                threading.Thread(
                    target=_stream_logs, args=(_bot_process,), daemon=True
                ).start()
                self._json({"ok": True, "msg": "Bot started", "pid": _bot_process.pid})
            except Exception as e:
                self._json({"ok": False, "msg": str(e)})

        elif path == "/api/stop":
            if not _is_running():
                self._json({"ok": False, "msg": "Not running"})
                return
            _bot_process.terminate()  # type: ignore[union-attr]
            try:
                _bot_process.wait(timeout=5)  # type: ignore[union-attr]
            except subprocess.TimeoutExpired:
                _bot_process.kill()  # type: ignore[union-attr]
            self._json({"ok": True, "msg": "Bot stopped"})

        elif path == "/api/settings":
            data = self._read_body()
            allowed = {
                "POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY",
                "POLYMARKET_API_SECRET", "POLYMARKET_PASSPHRASE",
                "PAPER_TRADING", "PAPER_BALANCE",
                "MAX_RISK_PER_TRADE_PCT", "MAX_DAILY_DRAWDOWN_PCT",
                "MIN_CONFIDENCE_THRESHOLD", "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID", "TELEGRAM_ENABLED",
                "MAX_CONSECUTIVE_LOSSES",
            }
            updates = {k.upper(): v for k, v in data.items() if k.upper() in allowed}
            _save_env(updates)
            self._json({"ok": True})

        else:
            self._send(404, b"Not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    def _open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  PolyBTC Trader  →  http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.shutdown()
