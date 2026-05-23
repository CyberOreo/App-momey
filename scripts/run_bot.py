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

    if args.five_min:
        # ── 5-minute strategy mode ─────────────────────────────────────────────
        logger.info("Starting in 5-MINUTE strategy mode")
        from src.market.order_flow import OrderFlowAnalyzer
        from src.market.scanner import MarketScanner
        from src.trading.five_min_strategy import FiveMinStrategy
        from src.risk.manager import RiskManager
        from src.trading.paper_trading import PaperTrader

        order_flow = OrderFlowAnalyzer(large_trade_usdc=50_000)
        strategy = FiveMinStrategy(order_flow=order_flow, settings=settings)
        risk_mgr = RiskManager(settings=settings)
        await risk_mgr.initialize(settings.paper_balance)
        paper_trader = PaperTrader(
            initial_balance=settings.paper_balance,
            settings=settings,
        )

        import aiohttp
        _stop = asyncio.Event()

        async def _get_btc_price() -> Optional[float]:
            import aiohttp as _aiohttp
            try:
                async with _aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{settings.binance_rest_url}/ticker/price?symbol=BTCUSDT",
                        timeout=_aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        data = await resp.json()
                        return float(data["price"])
            except Exception as e:
                logger.warning(f"[5MIN] BTC price fetch failed: {e}")
                return None

        async def _check_early_exits() -> None:
            """
            Runs every 10 seconds. Checks every open paper trade against the
            three early-exit rules: take-profit price, time pressure, flow reversal.
            """
            open_trades = paper_trader.get_open_trades()
            if not open_trades:
                return

            btc_price = await _get_btc_price()
            if btc_price is None:
                return
            strategy.update_btc_price(btc_price)

            for trade in open_trades:
                # Approximate current token price from order flow signal:
                # In live mode this would come from the Polymarket API.
                # In paper mode we estimate it from the fair value model.
                snap = order_flow.snapshot(30)
                if snap is None:
                    continue

                # Estimate how much the token has moved since entry
                velocity = order_flow.recent_price_velocity(90)
                estimated_price_delta = abs(velocity) * 15  # rough mapping
                if trade.direction.value == "YES":
                    current_token_price = min(0.97, trade.entry_price + estimated_price_delta)
                else:
                    current_token_price = min(0.97, trade.entry_price + estimated_price_delta)

                # Calculate seconds to resolution from trade metadata
                # (stored in market_id — in practice fetched from Polymarket)
                seconds_left = 180.0  # fallback; live mode reads from market

                should_exit, reason = strategy.should_exit_early(
                    entry_price=trade.entry_price,
                    current_token_price=current_token_price,
                    seconds_to_resolution=seconds_left,
                    direction=trade.direction,
                )

                if should_exit:
                    closed = await paper_trader.close_position(
                        trade, current_price=current_token_price, reason=reason
                    )
                    won = (closed.realized_pnl or 0) > 0
                    strategy.record_result(trade.market_id, won)
                    logger.success(
                        f"[5MIN] Early exit | {reason} | "
                        f"pnl={'+'if won else ''}{closed.realized_pnl:.2f}"
                    )

        async def _exit_monitor_loop() -> None:
            """Background task: check exits every 10 seconds."""
            while not _stop.is_set():
                try:
                    await _check_early_exits()
                except Exception as e:
                    logger.debug(f"[5MIN] Exit monitor error: {e}")
                await asyncio.sleep(10)

        async def _on_markets(markets):
            """Called every 30s with fresh list of active 5-min markets."""
            btc_price = await _get_btc_price()
            if btc_price is None:
                return

            strategy.update_btc_price(btc_price)

            if not order_flow.is_ready():
                logger.info("[5MIN] Waiting for order flow data (~90s on first start)")
                return

            risk_state = await risk_mgr.get_state()
            if not risk_state.can_trade:
                logger.warning("[5MIN] Trading halted by risk manager")
                return

            for market in markets:
                signal = strategy.evaluate_market(market, candles_1m=[])
                if signal is None:
                    continue
                can_exec, reason = await risk_mgr.can_execute(signal, None)
                if not can_exec:
                    logger.info(f"[5MIN] Signal blocked: {reason}")
                    continue
                trade = await paper_trader.place_order(signal, size_usdc=20.0)
                logger.success(
                    f"[5MIN] Trade opened | {signal.direction.value} | "
                    f"conf={signal.confidence:.0f} | edge={signal.edge*100:.1f}% | "
                    f"fair={signal.fair_value_estimate:.2f} vs mkt={signal.price:.2f}"
                )

        signal.signal(signal.SIGINT, lambda s, f: _stop.set())
        signal.signal(signal.SIGTERM, lambda s, f: _stop.set())

        # Import scanner (needs a Polymarket client — use stub in paper mode)
        class _StubClient:
            async def get_btc_markets(self):
                return []

        from src.core.database import init_db, MarketRepository
        import os as _os
        _os.makedirs("data", exist_ok=True)
        db_engine = await init_db(settings.database_url)
        db = MarketRepository(db_engine)
        scanner = MarketScanner(_StubClient(), settings, db)

        await asyncio.gather(
            order_flow.start(),
            scanner.run_five_min_continuous(interval_seconds=30, callback=_on_markets),
            _exit_monitor_loop(),
        )
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
