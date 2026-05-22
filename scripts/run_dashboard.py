#!/usr/bin/env python3
"""
Launch the Rich terminal dashboard.

Reads live system state from the database every 2 seconds and renders
a full-screen trading dashboard using Rich Live.

Usage
-----
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --config path/to/.env
    python scripts/run_dashboard.py --refresh 5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PolyBTC Trader — Terminal dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=".env",
        metavar="PATH",
        help="Path to .env configuration file.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Dashboard refresh interval in seconds.",
    )
    return parser.parse_args()


# ── State provider ────────────────────────────────────────────────────────────

class DBStateProvider:
    """
    Polls the SQLite / PostgreSQL database for current system state and
    formats it for the dashboard renderer.
    """

    def __init__(self, db, settings) -> None:
        self._db = db
        self._settings = settings
        self._last_price: float = 0.0
        self._prev_price: float = 0.0

    async def __call__(self) -> Dict[str, Any]:
        """
        Fetch and aggregate state from the database.

        Returns a dict with all keys expected by Dashboard._build_layout():
            btc_price, btc_change_24h, risk_state, open_positions,
            recent_signals, recent_trades, system_status.
        """
        from src.core.database import TradeRepository, MarketRepository
        from src.core.models import (
            Direction, RiskState, Position, TradeOutcome
        )

        state: Dict[str, Any] = {
            "btc_price": self._last_price or None,
            "btc_change_24h": 0.0,
            "risk_state": None,
            "open_positions": [],
            "recent_signals": [],
            "recent_trades": [],
            "system_status": {
                "mode": "PAPER" if self._settings.paper_trading else "LIVE",
                "environment": self._settings.environment,
                "api_ok": False,
                "binance_ok": False,
                "polymarket_ok": False,
                "last_scan": "Never",
                "error_count": 0,
            },
        }

        try:
            # ── BTC price from Binance ─────────────────────────────────────
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._settings.binance_rest_url}/ticker/24hr?symbol=BTCUSDT",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._prev_price = self._last_price
                        self._last_price = float(data["lastPrice"])
                        change_pct = float(data["priceChangePercent"]) / 100.0
                        state["btc_price"] = self._last_price
                        state["btc_change_24h"] = change_pct
                        state["system_status"]["binance_ok"] = True
                        state["system_status"]["api_ok"] = True

        except Exception:
            pass

        try:
            # ── Trades from DB ─────────────────────────────────────────────
            repo = TradeRepository(self._db)
            all_trades = await repo.get_all_trades()

            open_trades = [t for t in all_trades if t.outcome == TradeOutcome.OPEN]
            closed_trades = [t for t in all_trades if t.outcome != TradeOutcome.OPEN]

            # Convert open trades to minimal Position objects for rendering
            from src.core.models import Position
            import uuid
            positions: List[Position] = []
            for trade in open_trades[-10:]:
                positions.append(
                    Position(
                        position_id=str(uuid.uuid4()),
                        market_id=trade.market_id,
                        condition_id=trade.condition_id,
                        token_id=trade.token_id,
                        direction=trade.direction,
                        size=trade.size,
                        entry_price=trade.entry_price,
                        current_price=trade.entry_price,  # real impl would poll current price
                        entry_time=trade.entry_time,
                        confidence=trade.confidence,
                    )
                )

            state["open_positions"] = positions
            state["recent_trades"] = sorted(
                closed_trades, key=lambda t: t.exit_time or t.entry_time, reverse=True
            )[:10]

            # ── Build risk state from trade history ────────────────────────
            total_pnl = sum(
                t.realized_pnl for t in closed_trades
                if t.realized_pnl is not None
                and t.entry_time.date() == datetime.utcnow().date()
            )
            state["risk_state"] = RiskState(
                balance=self._settings.paper_balance + total_pnl,
                start_of_day_balance=self._settings.paper_balance,
                daily_pnl=total_pnl,
                consecutive_losses=0,
                open_positions=len(open_trades),
                total_exposure_usdc=sum(
                    t.size * t.entry_price for t in open_trades
                ),
                in_cooldown=False,
                cooldown_until=None,
                kill_switch_active=False,
                kill_switch_reason="",
            )

            state["system_status"]["last_scan"] = datetime.utcnow().strftime("%H:%M:%S")
            state["system_status"]["polymarket_ok"] = True

        except Exception as exc:
            logger.debug("State provider DB error", error=str(exc))

        return state


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv(args.config, override=False)

    from src.core.config import get_settings
    from src.core.database import init_db
    from src.monitoring.dashboard import Dashboard

    settings = get_settings()

    # Suppress loguru noise while dashboard is running
    logger.remove()
    logger.add("logs/dashboard.log", level="DEBUG", rotation="10 MB")

    # Init DB
    os.makedirs("data", exist_ok=True)
    db = await init_db(settings.database_url)

    state_provider = DBStateProvider(db, settings)
    dashboard = Dashboard()

    try:
        await dashboard.run(state_provider)
    except KeyboardInterrupt:
        pass


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
