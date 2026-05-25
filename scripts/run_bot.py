#!/usr/bin/env python3
"""
PolyBTC Trader — 5-Minute BTC Up/Down Engine

Usage:
    python scripts/run_bot.py [--live] [--config PATH] [--log-level LEVEL]

Flags:
    --live          Disable paper trading and send real orders (requires API keys).
    --config PATH   Path to an alternate .env file (default: .env in project root).
    --log-level LVL Logging verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO).
"""
from __future__ import annotations

import argparse
import asyncio
import json as _json
import os
import signal
import sys
from datetime import datetime as _dt
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PolyBTC Trader — Polymarket BTC 5-min binary engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", default=False,
                        help="Disable paper trading and place real orders.")
    parser.add_argument("--config", type=str, default=".env", metavar="PATH",
                        help="Path to .env configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], metavar="LEVEL")
    # kept for backwards-compat with web_app.py /api/start calls
    parser.add_argument("--five-min", action="store_true", default=False,
                        help="(no-op: 5-min engine is always used)")
    return parser.parse_args()


def _setup_logging(log_level: str, log_file: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )
    os.makedirs(os.path.dirname(log_file) if "/" in log_file else "logs", exist_ok=True)
    logger.add(log_file, level=log_level, rotation="50 MB", retention="14 days",
               compression="gz")


# ── Simple in-memory paper trader ─────────────────────────────────────────────

class _Paper:
    def __init__(self, bal: float):
        self.initial = bal
        self.balance = bal
        self.open: dict = {}
        self.closed: list = []
        self.wins = self.losses = 0

    def enter(self, decision) -> None:
        if self.open:
            return
        cost = decision.size_usdc
        fee = cost * 0.001
        if cost + fee > self.balance:
            return
        self.balance -= cost + fee
        self.open = {
            "direction": decision.direction,
            "entry_price": decision.entry_price,
            "size_usdc": cost,
            "tokens": cost / max(decision.entry_price, 0.01),
            "entry_time": _dt.utcnow().isoformat(),
            "confidence": decision.confidence,
            "window_delta": decision.window_delta,
        }
        logger.success(
            f"[PAPER] ENTER {decision.direction} | "
            f"size=${cost:.0f} | price={decision.entry_price:.3f} | "
            f"balance=${self.balance:.2f}"
        )

    def close(self, exit_price: float, result: str) -> None:
        if not self.open:
            return
        t = self.open.copy()
        exit_val = t["tokens"] * exit_price
        fee = exit_val * 0.001
        pnl = exit_val - t["size_usdc"] - fee
        self.balance += t["size_usdc"] + pnl
        t.update({"exit_price": exit_price, "pnl": round(pnl, 4),
                  "outcome": result, "exit_time": _dt.utcnow().isoformat()})
        self.closed.append(t)
        if result == "win":
            self.wins += 1
        else:
            self.losses += 1
        self.open = {}
        logger.success(
            f"[PAPER] CLOSE {t['direction']} | "
            f"pnl=${pnl:+.2f} | {result.upper()} | "
            f"balance=${self.balance:.2f}"
        )

    def stats(self) -> dict:
        total = len(self.closed)
        pnl = sum(t["pnl"] for t in self.closed)
        wr = round(self.wins / total * 100, 1) if total else 0.0
        today = _dt.utcnow().date().isoformat()
        today_t = [t for t in self.closed if (t.get("exit_time") or "")[:10] == today]
        return {
            "balance": round(self.balance, 2),
            "initial": self.initial,
            "total_pnl": round(pnl, 2),
            "daily_pnl": round(sum(t["pnl"] for t in today_t), 2),
            "total_trades": total,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": wr,
            "open": self.open or None,
            "recent": self.closed[-5:][::-1],
        }


# ── Main ───────────────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    import websockets as _ws
    from src.trading.five_min_engine import FiveMinEngine, BookSnapshot, current_window_open_ts
    from src.core.database import init_db, MarketRepository
    from src.market.chainlink import ChainlinkOracle
    from src.market.multi_exchange import MultiExchangeFeed
    from src.market.funding_rate import FundingRateTracker
    from src.market.btc5min import BTC5MinFeed, MarketSnap

    if args.config and args.config != ".env":
        os.environ["ENV_FILE"] = args.config
    from dotenv import load_dotenv
    load_dotenv(args.config, override=True)

    from src.core.config import get_settings, reload_settings
    settings = reload_settings() if args.config != ".env" else get_settings()

    if args.live:
        settings.paper_trading = False
        logger.warning("LIVE TRADING MODE ENABLED — real funds at risk")

    _setup_logging(args.log_level, settings.log_file)

    logger.info(
        "PolyBTC 5-Min Engine starting",
        paper=settings.paper_trading,
        balance=settings.paper_balance,
    )

    _stop = asyncio.Event()
    signal.signal(signal.SIGINT, lambda s, f: _stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: _stop.set())

    paper = _Paper(settings.paper_balance)
    _trade_count = [0]

    # Telegram controller — active only if token + chat_id are configured
    _tg = None
    _tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    _tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if _tg_token and _tg_chat:
        from src.monitoring.telegram_ctrl import TelegramController
        _tg = TelegramController(
            token=_tg_token,
            chat_id=_tg_chat,
            engine=None,  # set after engine is created
            paper=paper,
            stop_event=_stop,
        )

    async def _handle_decision(decision) -> None:
        _trade_count[0] += 1
        logger.success(
            f"[ENGINE] #{_trade_count[0]} {decision.action} | "
            f"conf={decision.confidence:.0f} | T-{decision.seconds_to_close:.0f}s | "
            f"delta={decision.window_delta*100:+.3f}% | "
            f"entry={decision.entry_price:.3f} | ${decision.size_usdc:.0f}"
            + (" | ARBITRAGE" if decision.is_arbitrage else "")
        )
        for r in decision.reasons:
            logger.debug(f"  → {r}")
        if settings.paper_trading:
            paper.enter(decision)
        if _tg:
            asyncio.create_task(_tg.alert_trade(decision))

    engine = FiveMinEngine(
        balance=settings.paper_balance,
        max_risk_pct=settings.max_risk_per_trade_pct,
        on_decision=lambda d: asyncio.create_task(_handle_decision(d)),
    )
    if _tg:
        _tg._engine = engine

    # ── Chainlink oracle ───────────────────────────────────────────────────────
    def _oracle_cb(oracle_price) -> None:
        window_open = engine._price_buffer.window_open_price
        if window_open > 0 and oracle_price.is_fresh:
            confirms = "YES" if oracle_price.price > window_open else "NO"
            engine.on_oracle(confirms, oracle_price.age_seconds)
        else:
            engine.on_oracle(None, oracle_price.age_seconds)

    oracle = ChainlinkOracle(poll_interval=5.0, on_update=_oracle_cb)

    # ── Multi-exchange consensus ───────────────────────────────────────────────
    def _consensus_cb(result) -> None:
        engine.on_consensus_result(result.direction, result.signal_boost)

    multi_feed = MultiExchangeFeed(on_consensus=_consensus_cb)

    # ── Funding rate ───────────────────────────────────────────────────────────
    def _funding_cb(snap) -> None:
        engine.on_funding_signal(snap.signal)

    funding = FundingRateTracker(on_update=_funding_cb)

    # ── Polymarket live prices ─────────────────────────────────────────────────
    _current_market: dict = {}
    _last_window_ts = [0]
    _last_close_window = [0]

    def _on_market_snap(snap: MarketSnap) -> None:
        nonlocal _current_market
        book = BookSnapshot(
            yes_bid=snap.yes_bid, yes_ask=snap.yes_ask,
            no_bid=snap.no_bid, no_ask=snap.no_ask,
            yes_bid_size=1000.0, yes_ask_size=1000.0,
            no_bid_size=1000.0, no_ask_size=1000.0,
        )
        asyncio.create_task(engine.update_orderbook(book))
        _current_market = {
            "slug": snap.slug,
            "question": snap.question,
            "yes_bid": round(snap.yes_bid, 4),
            "yes_ask": round(snap.yes_ask, 4),
            "no_bid": round(snap.no_bid, 4),
            "no_ask": round(snap.no_ask, 4),
            "yes_mid": round(snap.yes_mid, 4),
            "no_mid": round(snap.no_mid, 4),
            "sum_ask": round(snap.sum_ask, 4),
            "seconds_to_close": round(snap.seconds_to_close, 1),
            "spread_pct": round(snap.spread_pct, 4),
        }
        engine._active_markets = [_current_market] if _current_market else []

    poly_feed = BTC5MinFeed(on_update=_on_market_snap)

    # ── Binance combined stream ────────────────────────────────────────────────
    async def _binance_feed() -> None:
        url = "wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/btcusdt@aggTrade"
        delay = 1.0
        while not _stop.is_set():
            try:
                async with _ws.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("[ENGINE] Binance stream connected")
                    delay = 1.0
                    async for raw in ws:
                        if _stop.is_set():
                            break
                        try:
                            msg = _json.loads(raw)
                            data = msg.get("data", msg)
                            stream = msg.get("stream", "")
                            if "ticker" in stream:
                                engine.on_price(float(data["c"]))
                                wts = current_window_open_ts()
                                if wts != _last_window_ts[0]:
                                    _last_window_ts[0] = wts
                                    multi_feed.set_window_baseline()
                            elif "aggTrade" in stream:
                                engine.on_trade(float(data["p"]), float(data["q"]), bool(data["m"]))
                        except Exception as e:
                            logger.debug(f"[ENGINE] Stream parse: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not _stop.is_set():
                    logger.warning(f"[ENGINE] Reconnecting in {delay:.0f}s: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

    # ── Status writer (every 0.5s) ─────────────────────────────────────────────
    _status_path = _ROOT / "data" / "engine_status.json"

    async def _write_status() -> None:
        os.makedirs(str(_ROOT / "data"), exist_ok=True)
        while not _stop.is_set():
            try:
                wts = current_window_open_ts()
                prev_wts = _last_close_window[0]
                if settings.paper_trading and paper.open:
                    if prev_wts and wts != prev_wts:
                        wd = engine._price_buffer.window_delta
                        direction = paper.open.get("direction", "YES")
                        won = (wd > 0) if direction == "YES" else (wd < 0)
                        exit_price = 1.0 if won else 0.0
                        outcome = "win" if won else "loss"
                        paper.close(exit_price, outcome)
                        if _tg:
                            ps = paper.stats()
                            closed = ps.get("recent", [{}])[0]
                            asyncio.create_task(_tg.alert_close(
                                direction, closed.get("pnl", 0),
                                outcome, ps["balance"],
                            ))
                _last_close_window[0] = wts
                st = engine.status()
                st.update({
                    "mode": "five_min_engine",
                    "paper": settings.paper_trading,
                    "trades": _trade_count[0],
                    "paper_stats": paper.stats() if settings.paper_trading else None,
                    "current_market": _current_market,
                })
                _status_path.write_text(_json.dumps(st), encoding="utf-8")
            except Exception:
                pass
            await asyncio.sleep(0.5)

    os.makedirs("data", exist_ok=True)
    db_engine = await init_db(settings.database_url)

    logger.info(
        "[ENGINE] All subsystems ready",
        balance=settings.paper_balance,
        paper=settings.paper_trading,
        min_confidence=engine.MIN_CONFIDENCE,
    )

    await engine.start()
    await oracle.start()
    await multi_feed.start()
    await funding.start()
    await poly_feed.start()
    if _tg:
        await _tg.start()

    tasks = [
        asyncio.create_task(_binance_feed()),
        asyncio.create_task(_write_status()),
    ]

    try:
        await _stop.wait()
    finally:
        logger.info("[ENGINE] Shutting down")
        if _tg:
            await _tg.stop()
        await engine.stop()
        await oracle.stop()
        await multi_feed.stop()
        await funding.stop()
        await poly_feed.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("[ENGINE] Stopped cleanly")


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
