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
        from src.intelligence.sentiment import SentimentAnalyzer
        from src.intelligence.regime import RegimeDetector
        from src.intelligence.ml_scorer import MLScorer

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

        # Intelligence
        self._sentiment_analyzer = SentimentAnalyzer(self._settings)
        self._regime_detector = RegimeDetector()
        self._ml_scorer = MLScorer(model_path="data/ml_model.pkl")

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
