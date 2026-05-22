#!/usr/bin/env python3
"""
Backtest runner — fetches historical BTC data from Binance and runs BacktestEngine.

Usage
-----
    python scripts/run_backtest.py --days 30 --balance 1000
    python scripts/run_backtest.py --days 90 --balance 5000 --output data/backtest.csv --verbose

Flags
-----
--days INT      Number of days of historical data to fetch (default: 30).
--balance FLOAT Starting paper balance in USDC (default: 1000.0).
--output PATH   CSV export path for the trade journal (default: data/backtest_TIMESTAMP.csv).
--verbose       Enable DEBUG logging for detailed pipeline output.
--timeframes    Comma-separated list of timeframes to fetch (default: 1h,4h,15m,5m,1m).
--markets INT   Number of mock Polymarket markets to generate (default: 5).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger


# ── CLI parsing ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PolyBTC Trader — Backtest runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical candle data to fetch.",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Starting paper balance in USDC.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV export filepath (default: auto-generated with timestamp).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1h,4h,15m,5m",
        help="Comma-separated timeframe list to fetch from Binance.",
    )
    parser.add_argument(
        "--markets",
        type=int,
        default=5,
        help="Number of mock Polymarket markets to generate.",
    )
    return parser.parse_args()


# ── Binance REST data fetcher ─────────────────────────────────────────────────

_BINANCE_REST = "https://api.binance.com/api/v3"

_TF_TO_BINANCE_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

_TF_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


async def fetch_candles(
    timeframe: str,
    days: int,
    symbol: str = "BTCUSDT",
) -> List:
    """
    Fetch historical OHLCV candles from Binance REST API.

    Parameters
    ----------
    timeframe:
        Timeframe label (e.g. "1h", "4h").
    days:
        Number of calendar days to fetch (counting backwards from now).
    symbol:
        Trading pair (default: BTCUSDT).

    Returns
    -------
    List[Candle] ordered oldest-first.
    """
    from src.core.models import Candle

    interval = _TF_TO_BINANCE_INTERVAL.get(timeframe, "1h")
    minutes_per_bar = _TF_TO_MINUTES.get(timeframe, 60)
    bars_needed = (days * 24 * 60) // minutes_per_bar

    # Binance max per request = 1000 klines
    max_per_req = 1000
    end_time_ms = int(datetime.utcnow().timestamp() * 1000)

    all_candles: List[Candle] = []
    remaining = bars_needed

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            while remaining > 0:
                limit = min(remaining, max_per_req)
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                    "endTime": end_time_ms,
                }

                async with session.get(
                    f"{_BINANCE_REST}/klines",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            "Binance REST non-200", status=resp.status, body=text[:200]
                        )
                        break

                    data = await resp.json()

                if not data:
                    break

                for bar in data:
                    open_time_ms = int(bar[0])
                    ts = datetime.utcfromtimestamp(open_time_ms / 1000)
                    all_candles.append(
                        Candle(
                            timestamp=ts,
                            open=float(bar[1]),
                            high=float(bar[2]),
                            low=float(bar[3]),
                            close=float(bar[4]),
                            volume=float(bar[5]),
                            timeframe=timeframe,
                        )
                    )

                # Move end_time back to fetch earlier data
                end_time_ms = int(data[0][0]) - 1  # exclusive
                remaining -= len(data)

                if len(data) < limit:
                    break

    except Exception as exc:
        logger.error("Failed to fetch Binance candles", timeframe=timeframe, error=str(exc))

    # Sort oldest first
    all_candles.sort(key=lambda c: c.timestamp)
    logger.info(
        "Candles fetched from Binance",
        timeframe=timeframe,
        count=len(all_candles),
        symbol=symbol,
    )
    return all_candles


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _generate_synthetic_candles(timeframe: str, days: int, start_price: float = 95_000.0) -> List:
    """
    Generate synthetic candles when Binance is unavailable.

    Uses a geometric random walk with mild upward drift so backtests produce
    a realistic mix of signals.
    """
    import math
    import numpy as np
    from src.core.models import Candle

    rng = np.random.default_rng(42)
    minutes_per_bar = _TF_TO_MINUTES.get(timeframe, 60)
    n_bars = (days * 24 * 60) // minutes_per_bar

    start = datetime.utcnow() - timedelta(days=days)
    candles = []
    price = start_price

    for i in range(n_bars):
        ts = start + timedelta(minutes=i * minutes_per_bar)
        log_ret = 0.0001 + 0.008 * rng.standard_normal()
        close = price * math.exp(log_ret)
        high = max(price, close) * (1.0 + abs(rng.normal(0, 0.002)))
        low = min(price, close) * (1.0 - abs(rng.normal(0, 0.002)))
        volume = abs(rng.normal(500, 100))
        candles.append(
            Candle(
                timestamp=ts,
                open=round(price, 2),
                high=round(high, 2),
                low=round(max(1.0, low), 2),
                close=round(close, 2),
                volume=round(volume, 4),
                timeframe=timeframe,
            )
        )
        price = close

    logger.info("Synthetic candles generated", timeframe=timeframe, count=len(candles))
    return candles


# ── Main backtest runner ──────────────────────────────────────────────────────

async def run_backtest(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv(".env", override=False)

    from src.core.config import get_settings
    from src.analytics.backtest import BacktestEngine
    from src.analytics.performance import PerformanceAnalyzer

    settings = get_settings()
    # Force paper trading for backtest
    settings.paper_trading = True

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    logger.info(
        "Backtest configuration",
        days=args.days,
        balance=args.balance,
        timeframes=timeframes,
        markets=args.markets,
    )

    # ── Fetch candle data ──────────────────────────────────────────────────
    print(f"\n  Fetching {args.days} days of BTC candle data from Binance...")
    candles_by_tf: Dict[str, List] = {}

    for tf in timeframes:
        print(f"  → Fetching {tf} candles... ", end="", flush=True)
        try:
            candles = await fetch_candles(tf, args.days)
            if len(candles) < 200:
                raise ValueError(f"Only {len(candles)} candles, need 200+")
            candles_by_tf[tf] = candles
            print(f"OK ({len(candles)} bars)")
        except Exception as exc:
            print(f"FAILED ({exc}) — using synthetic data")
            candles_by_tf[tf] = _generate_synthetic_candles(tf, args.days)

    if not candles_by_tf:
        print("  ERROR: No candle data available. Cannot run backtest.")
        sys.exit(1)

    # Get current BTC price for market generation
    current_price = candles_by_tf[timeframes[0]][-1].close if candles_by_tf else 95_000.0

    # ── Generate mock markets ──────────────────────────────────────────────
    print(f"\n  Generating {args.markets} mock Polymarket markets (BTC ≈ ${current_price:,.0f})...")
    markets = BacktestEngine.generate_mock_markets(n=args.markets, current_btc_price=current_price)
    for m in markets:
        print(f"  → {m['question']}")

    # ── Run backtest ───────────────────────────────────────────────────────
    print(f"\n  Running backtest on {sum(len(v) for v in candles_by_tf.values())} total candles...")
    engine = BacktestEngine(settings, initial_balance=args.balance)
    metrics = await engine.run(candles_by_tf, markets)

    # ── Print summary ──────────────────────────────────────────────────────
    analyzer = PerformanceAnalyzer()
    print()
    analyzer.print_summary(metrics)

    # ── Export CSV ─────────────────────────────────────────────────────────
    results = engine.get_results()
    trades = results["trades"]

    os.makedirs("data", exist_ok=True)
    output_path = args.output or (
        f"data/backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    if trades:
        analyzer.export_to_csv(trades, output_path)
        print(f"\n  Trade journal exported → {output_path}")
    else:
        print("\n  No trades to export.")

    # ── Quick equity summary ───────────────────────────────────────────────
    equity_curve = results["equity_curve"]
    if len(equity_curve) > 1:
        initial = equity_curve[0][1]
        final = equity_curve[-1][1]
        total_return = (final - initial) / initial * 100
        print(f"\n  Equity: ${initial:,.2f} → ${final:,.2f} ({total_return:+.2f}%)")

    print(f"\n  Backtest complete. {metrics.total_trades} trades analysed.\n")


def main() -> None:
    args = _parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    import sys as _sys
    logger.add(_sys.stderr, level=log_level, colorize=True)

    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
