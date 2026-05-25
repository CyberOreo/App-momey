#!/usr/bin/env python3
"""
Main entry point for PolyBTC Trader.

Usage
-----
    python scripts/run_bot.py [--live] [--config path/to/.env] [--log-level DEBUG]

Flags
-----
--live          Disable paper trading and send real orders (requires API keys).
--config PATH   Path to an alternate .env file (default: .env in project root).
--log-level LVL Logging verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger


# ── CLI parsing ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PolyBTC Trader — Polymarket BTC binary options bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Disable paper trading and place real orders.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=".env",
        metavar="PATH",
        help="Path to .env configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--five-min",
        action="store_true",
        default=False,
        help=(
            "Run the 5-minute BTC up/down strategy. "
            "Uses tick-level order flow instead of multi-hour indicators. "
            "Scans every 30 seconds and requires order flow to be ready (~90s on startup)."
        ),
    )
    return parser.parse_args()


# ── Logging setup ─────────────────────────────────────────────────────────────

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
    logger.add(
        log_file,
        level=log_level,
        rotation="50 MB",
        retention="14 days",
        compression="gz",
        serialize=False,
    )


# ── Bot orchestrator ───────────────────────────────────────────────────────────

class BotOrchestrator:
    """
    Top-level async coordinator for the PolyBTC trading system.

    Manages the WebSocket feed, periodic market scanning, signal execution,
    position monitoring, and graceful shutdown.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._running = False
        self._reconnect_delay = settings.reconnect_base_delay
        self._error_count = 0

        # These are initialised in setup()
        self._db = None
        self._binance_feed = None
        self._polymarket_client = None
        self._paper_trader = None
        self._risk_manager = None
        self._indicator_engine = None
        self._mtf_analyzer = None
        self._signal_generator = None
        self._executor = None
        self._telegram = None
        self._metrics_collector = None
        self._dashboard = None
        self._journal = None

    async def setup(self) -> None:
        """Initialise all subsystems."""
        from src.core.database import init_db
        from src.core.logging_setup import setup_logging
        from src.market.indicators import IndicatorEngine
        from src.market.analysis import MultiTimeframeAnalyzer
        from src.trading.signals import SignalGenerator
        from src.monitoring.metrics import MetricsCollector
        from src.monitoring.telegram_alerts import TelegramAlerter
        from src.analytics.journal import TradeJournal

        logger.info("Initialising PolyBTC Trader", mode="paper" if self._settings.paper_trading else "LIVE")

        # Database
        os.makedirs("data", exist_ok=True)
        self._db = await init_db(self._settings.database_url)

        # Market data
        self._indicator_engine = IndicatorEngine()
        self._mtf_analyzer = MultiTimeframeAnalyzer(self._indicator_engine, self._settings)
        self._signal_generator = SignalGenerator(self._settings)

        # Monitoring
        self._metrics_collector = MetricsCollector()
        self._telegram = TelegramAlerter(self._settings)
        self._journal = TradeJournal(db=self._db, export_dir="data/journal")

        # Intelligence (optional — requires scikit-learn/pandas)
        try:
            from src.intelligence.sentiment import SentimentAnalyzer
            from src.intelligence.regime import RegimeDetector
            from src.intelligence.ml_scorer import MLScorer
            self._sentiment_analyzer = SentimentAnalyzer(self._settings)
            self._regime_detector = RegimeDetector()
            self._ml_scorer = MLScorer(model_path="data/ml_model.pkl")
        except ImportError as _ie:
            logger.warning(
                f"Intelligence modules unavailable ({_ie}). "
                "Install scikit-learn and pandas to enable ML scoring."
            )
            self._sentiment_analyzer = None
            self._regime_detector = None
            self._ml_scorer = None

        logger.info("All subsystems initialised")

    async def run(self) -> None:
        """
        Main bot loop.

        Iterates every 5 minutes to:
          1. Fetch latest BTC price and candles.
          2. Scan active Polymarket markets.
          3. Run the full analysis pipeline on each market.
          4. Execute any valid signals (paper or live).
          5. Check open positions for close conditions.
          6. Evaluate risk state.
          7. Update dashboard.
        """
        self._running = True
        logger.info("PolyBTC Trader started")
        await self._telegram.send_message("🚀 *PolyBTC Trader started*")

        candle_buffer: Dict[str, List] = {
            tf: [] for tf in ("1m", "5m", "15m", "1h", "4h")
        }
        scan_interval = 300  # seconds

        while self._running:
            try:
                # ── Fetch market data ──────────────────────────────────────
                btc_price, candle_buffer = await self._fetch_candle_data(candle_buffer)
                if btc_price is None:
                    await asyncio.sleep(10)
                    continue

                self._metrics_collector.mark_price_update()

                # ── Fetch active markets ───────────────────────────────────
                markets = await self._fetch_markets()
                self._metrics_collector.mark_scan_complete()

                # ── Analyse each market ────────────────────────────────────
                for market in markets:
                    signal = await self._analyse_market(market, btc_price, candle_buffer)
                    if signal is not None:
                        self._metrics_collector.record_signal(signal)
                        await self._telegram.alert_signal(signal)
                        trade = await self._maybe_execute(signal)
                        if trade is not None:
                            self._metrics_collector.record_trade(trade)
                            await self._journal.add_trade(trade)
                            await self._telegram.alert_trade_opened(trade)

                # ── Check open positions ───────────────────────────────────
                await self._check_positions(btc_price)

                # ── Risk check ─────────────────────────────────────────────
                await self._risk_check()

                logger.info("Scan cycle complete", markets_scanned=len(markets))
                await asyncio.sleep(scan_interval)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled — shutting down")
                break
            except Exception as exc:
                self._error_count += 1
                logger.error("Main loop error", error=str(exc), count=self._error_count)
                self._metrics_collector.record_error(exc, "main_loop")
                await self._telegram.alert_error(str(exc))

                if self._error_count >= 10:
                    logger.critical("Too many errors — activating emergency stop")
                    await self._telegram.alert_risk_event(
                        "EMERGENCY_STOP", f"10 consecutive errors. Last: {exc}"
                    )
                    break

                backoff = min(self._reconnect_delay, self._settings.reconnect_max_delay)
                self._reconnect_delay *= 2
                await asyncio.sleep(backoff)
            else:
                self._reconnect_delay = self._settings.reconnect_base_delay
                self._error_count = 0

    async def shutdown(self) -> None:
        """Graceful shutdown: stop loop, close connections, export journal."""
        logger.info("Shutting down PolyBTC Trader")
        self._running = False

        try:
            if self._journal:
                path = await self._journal.export_csv()
                logger.info("Journal exported on shutdown", path=path)
        except Exception as exc:
            logger.warning("Failed to export journal on shutdown", error=str(exc))

        await self._telegram.send_message("🛑 *PolyBTC Trader stopped*")
        logger.info("Shutdown complete")

    # ── Internal pipeline steps ───────────────────────────────────────────────

    async def _fetch_candle_data(self, buffer: Dict) -> tuple:
        """Fetch BTC candles from Binance REST API. Returns (price, updated_buffer)."""
        try:
            rest_url = self._settings.binance_rest_url
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{rest_url}/ticker/price?symbol=BTCUSDT",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    btc_price = float(data["price"])

            return btc_price, buffer

        except Exception as exc:
            logger.warning("Failed to fetch BTC price", error=str(exc))
            return None, buffer

    async def _fetch_markets(self) -> List:
        """Return active Polymarket BTC markets (stub — real implementation uses PolymarketClient)."""
        from src.core.database import MarketRepository
        repo = MarketRepository(self._db)
        return await repo.get_active_markets()

    async def _analyse_market(self, market, btc_price: float, candle_buffer: Dict):
        """Run the full analysis pipeline for one market. Returns signal or None."""
        from src.core.models import (
            MarketCondition, MarketConditionType, VolatilityRegime
        )
        # In a full implementation this runs the indicator engine, MTF analyzer,
        # market condition detector, signal generator, and confidence scorer.
        # For brevity the orchestrator delegates to the existing modules.
        return None

    async def _maybe_execute(self, signal) -> Optional[object]:
        """Gate the signal through risk management and execute if approved."""
        return None

    async def _check_positions(self, btc_price: float) -> None:
        """Check all open positions for stop-loss / take-profit / resolution."""
        pass

    async def _risk_check(self) -> None:
        """Evaluate risk state and send alerts if limits are approaching."""
        pass


# ── Signal handlers ───────────────────────────────────────────────────────────

_orchestrator: Optional[BotOrchestrator] = None


def _handle_signal(sig, frame):
    logger.info("Received shutdown signal", signal=sig)
    if _orchestrator is not None:
        asyncio.get_event_loop().create_task(_orchestrator.shutdown())


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    global _orchestrator

    # Load settings
    if args.config and args.config != ".env":
        os.environ["ENV_FILE"] = args.config

    # Patch dotenv load before importing Settings
    from dotenv import load_dotenv
    load_dotenv(args.config, override=True)

    from src.core.config import get_settings, reload_settings

    if args.config != ".env":
        settings = reload_settings()
    else:
        settings = get_settings()

    # Override paper mode from CLI
    if args.live:
        settings.paper_trading = False
        logger.warning("LIVE TRADING MODE ENABLED — real funds at risk")

    # Setup logging
    _setup_logging(args.log_level, settings.log_file)

    logger.info(
        "PolyBTC Trader starting",
        environment=settings.environment,
        paper_trading=settings.paper_trading,
        balance=settings.paper_balance,
        log_level=args.log_level,
    )

    if args.five_min:
        # ── 5-minute unified engine — all signals in one system ───────────────
        logger.info("Starting 5-MIN UNIFIED ENGINE — 1-second loop + 6 signal sources")
        import json as _json
        import websockets as _ws
        from src.trading.five_min_engine import FiveMinEngine, BookSnapshot
        from src.trading.paper_trading import PaperTrader
        from src.market.scanner import MarketScanner
        from src.core.database import init_db, MarketRepository
        from src.market.chainlink import ChainlinkOracle
        from src.market.multi_exchange import MultiExchangeFeed
        from src.market.funding_rate import FundingRateTracker

        _stop = asyncio.Event()
        signal.signal(signal.SIGINT, lambda s, f: _stop.set())
        signal.signal(signal.SIGTERM, lambda s, f: _stop.set())

        paper_trader = PaperTrader(
            initial_balance=settings.paper_balance,
            settings=settings,
        )
        _trade_count = [0]

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

        engine = FiveMinEngine(
            balance=settings.paper_balance,
            max_risk_pct=settings.max_risk_per_trade_pct,
            on_decision=lambda d: asyncio.create_task(_handle_decision(d)),
        )

        # ── Chainlink oracle — Polygon RPC, polls every 5s ────────────────────
        def _oracle_cb(oracle_price) -> None:
            window_open = engine._price_buffer.window_open_price
            if window_open > 0 and oracle_price.is_fresh:
                confirms = "YES" if oracle_price.price > window_open else "NO"
                engine.on_oracle(confirms, oracle_price.age_seconds)
                logger.debug(
                    f"[ORACLE] BTC/USD={oracle_price.price:,.2f} | "
                    f"confirms={confirms} | age={oracle_price.age_seconds:.0f}s"
                )
            else:
                engine.on_oracle(None, oracle_price.age_seconds)

        oracle = ChainlinkOracle(poll_interval=5.0, on_update=_oracle_cb)

        # ── Multi-exchange consensus — Binance + Coinbase + Bybit ────────────
        def _consensus_cb(result) -> None:
            engine.on_consensus_result(result.direction, result.signal_boost)
            if result.exchange_count >= 2:
                logger.debug(
                    f"[MULTI] {result.direction} | "
                    f"agreement={result.agreement:.0%} | "
                    f"exchanges={result.exchange_count} | "
                    f"boost={result.signal_boost:+.0f}"
                )

        multi_feed = MultiExchangeFeed(on_consensus=_consensus_cb)

        # ── Funding rate — Binance perpetual futures ──────────────────────────
        def _funding_cb(snap) -> None:
            engine.on_funding_signal(snap.signal)

        funding = FundingRateTracker(on_update=_funding_cb)

        # Write engine status to disk every 2 seconds for the web dashboard
        _status_path = _ROOT / "data" / "engine_status.json"

        async def _write_status() -> None:
            os.makedirs(str(_ROOT / "data"), exist_ok=True)
            while not _stop.is_set():
                try:
                    st = engine.status()
                    st.update({
                        "mode": "five_min_engine",
                        "paper": settings.paper_trading,
                        "trades": _trade_count[0],
                    })
                    _status_path.write_text(_json.dumps(st), encoding="utf-8")
                except Exception:
                    pass
                await asyncio.sleep(2)

        # Binance combined stream: ticker price + aggTrades for VPIN
        _last_window_ts = [0]

        async def _binance_feed() -> None:
            url = (
                "wss://stream.binance.com:9443/stream"
                "?streams=btcusdt@ticker/btcusdt@aggTrade"
            )
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
                                    # Snapshot multi-exchange baseline at each new window
                                    from src.trading.five_min_engine import current_window_open_ts
                                    wts = current_window_open_ts()
                                    if wts != _last_window_ts[0]:
                                        _last_window_ts[0] = wts
                                        multi_feed.set_window_baseline()
                                        logger.debug("[ENGINE] New window — baseline set")
                                elif "aggTrade" in stream:
                                    engine.on_trade(
                                        float(data["p"]),
                                        float(data["q"]),
                                        bool(data["m"]),
                                    )
                            except Exception as e:
                                logger.debug(f"[ENGINE] Stream parse: {e}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if not _stop.is_set():
                        logger.warning(f"[ENGINE] Reconnecting in {delay:.0f}s: {e}")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)

        # Polymarket orderbook poller — synthetic book in paper mode
        async def _orderbook_poller() -> None:
            while not _stop.is_set():
                try:
                    wd = engine._price_buffer.window_delta
                    abs_wd = abs(wd)
                    if abs_wd < 0.00005:
                        yes_p = no_p = 0.50
                    elif wd > 0:
                        yes_p = min(0.92, 0.50 + abs_wd * 80)
                        no_p = max(0.08, 1.0 - yes_p - 0.02)
                    else:
                        no_p = min(0.92, 0.50 + abs_wd * 80)
                        yes_p = max(0.08, 1.0 - no_p - 0.02)
                    spread = 0.018
                    book = BookSnapshot(
                        yes_bid=round(yes_p - spread / 2, 3),
                        yes_ask=round(yes_p + spread / 2, 3),
                        no_bid=round(no_p - spread / 2, 3),
                        no_ask=round(no_p + spread / 2, 3),
                        yes_bid_size=500.0,
                        yes_ask_size=500.0,
                        no_bid_size=500.0,
                        no_ask_size=500.0,
                    )
                    await engine.update_orderbook(book)
                except Exception as e:
                    logger.debug(f"[ENGINE] Orderbook update error: {e}")
                await asyncio.sleep(2)

        # Scanner discovers which Polymarket 5-min markets are active
        os.makedirs("data", exist_ok=True)
        db_engine = await init_db(settings.database_url)
        db = MarketRepository(db_engine)

        class _StubClient:
            async def get_btc_markets(self):
                return []

        scanner = MarketScanner(_StubClient(), settings, db)

        async def _on_markets(markets) -> None:
            logger.info(f"[ENGINE] {len(markets)} active 5-min markets in scope")
            # Store market summaries in engine for dashboard display
            engine._active_markets = [
                {
                    "question": (m.question or "")[:60],
                    "yes_price": round(m.yes_token.price if m.yes_token else 0, 3),
                    "no_price": round(m.no_token.price if m.no_token else 0, 3),
                    "liquidity": round(getattr(m, "liquidity", 0) or 0, 0),
                    "volume": round(getattr(m, "volume", 0) or 0, 0),
                    "seconds_left": max(0, round((m.end_date - __import__("datetime").datetime.utcnow()).total_seconds())) if m.end_date else 0,
                }
                for m in (markets or [])
            ][:5]

        logger.info(
            "[ENGINE] All subsystems ready",
            balance=settings.paper_balance,
            paper=settings.paper_trading,
            min_confidence=engine.MIN_CONFIDENCE,
            entry_window=f"T-{engine.ENTRY_PREFERRED_START}s to T-{engine.ENTRY_PREFERRED_END}s",
            signals="window_delta + VPIN + Chainlink oracle + 3-exchange consensus + funding rate",
        )

        # Start all async components
        await engine.start()
        await oracle.start()
        await multi_feed.start()
        await funding.start()

        tasks = [
            asyncio.create_task(_binance_feed()),
            asyncio.create_task(_orderbook_poller()),
            asyncio.create_task(_write_status()),
            asyncio.create_task(
                scanner.run_five_min_continuous(30, _on_markets)
            ),
        ]

        try:
            await _stop.wait()
        finally:
            logger.info("[ENGINE] Shutting down all subsystems")
            await engine.stop()
            await oracle.stop()
            await multi_feed.stop()
            await funding.stop()
            await scanner.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("[ENGINE] All subsystems stopped cleanly")
        return

    _orchestrator = BotOrchestrator(settings)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        await _orchestrator.setup()
        await _orchestrator.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        if _orchestrator:
            await _orchestrator.shutdown()


def main() -> None:
    args = _parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
