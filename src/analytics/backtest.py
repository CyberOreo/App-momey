"""
Backtesting engine that replays historical candle data through the full signal pipeline.

Chronologically feeds candles through every module — indicators, market condition,
multi-timeframe analysis, signal generation, confidence scoring, position sizing,
and simulated paper execution — then computes PerformanceMetrics on the result.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.analytics.performance import PerformanceAnalyzer
from src.core.models import (
    Candle,
    Direction,
    IndicatorSet,
    Market,
    MarketCondition,
    MarketConditionType,
    PerformanceMetrics,
    PolymarketToken,
    PositionSizing,
    TimeframeScore,
    Trade,
    TradeOutcome,
    TradeSignal,
    VolatilityRegime,
)
from src.market.analysis import MultiTimeframeAnalyzer
from src.market.indicators import IndicatorEngine
from src.trading.scoring import ConfidenceScorer
from src.trading.signals import SignalGenerator


# ── Simple paper trader for backtesting ──────────────────────────────────────

class _BacktestPaperTrader:
    """Minimal paper trader that records trades without DB persistence."""

    def __init__(self, balance: float) -> None:
        self._balance = balance
        self._trades: List[Trade] = []

    def get_balance(self) -> float:
        return self._balance

    def place_order(self, signal: TradeSignal, size_usdc: float) -> Trade:
        tokens = size_usdc / signal.price if signal.price > 0 else 0.0
        fee = size_usdc * 0.001
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            market_id=signal.market_id,
            condition_id=signal.market_id,
            token_id=signal.token_id,
            direction=signal.direction,
            size=tokens,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            confidence=signal.confidence,
            signal_reasons=signal.reasons,
            paper_trade=True,
            fees_paid=fee,
        )
        self._balance -= size_usdc + fee
        self._trades.append(trade)
        return trade

    def close_trade(self, trade: Trade, exit_price: float, exit_time: datetime) -> None:
        pnl = trade.size * (exit_price - trade.entry_price)
        fee = trade.size * exit_price * 0.001
        pnl -= fee
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.realized_pnl = pnl
        trade.fees_paid += fee
        if pnl > 0.01:
            trade.outcome = TradeOutcome.WIN
        elif pnl < -0.01:
            trade.outcome = TradeOutcome.LOSS
        else:
            trade.outcome = TradeOutcome.BREAK_EVEN
        self._balance += trade.size * exit_price - fee

    @property
    def trades(self) -> List[Trade]:
        return self._trades


# ── Market condition detector ─────────────────────────────────────────────────

class _MarketConditionDetector:
    """Simple market condition detector used in backtesting."""

    def detect(self, indicators: IndicatorSet) -> MarketCondition:
        close = indicators.close
        ema20 = indicators.ema_20
        ema50 = indicators.ema_50
        atr_pct = indicators.atr_pct
        rsi = indicators.rsi

        # Volatility regime
        if atr_pct > 0.04:
            vol_regime = VolatilityRegime.EXTREME
        elif atr_pct > 0.025:
            vol_regime = VolatilityRegime.HIGH
        elif atr_pct > 0.012:
            vol_regime = VolatilityRegime.MEDIUM
        else:
            vol_regime = VolatilityRegime.LOW

        # Trend
        ema_spread = abs(ema20 - ema50) / max(ema50, 1e-8)
        if ema_spread < 0.003 or (40.0 < rsi < 60.0 and ema_spread < 0.008):
            condition = MarketConditionType.CHOPPY
            trend_dir = None
            strength = 0.1
        elif close > ema50 and ema20 > ema50:
            condition = MarketConditionType.TRENDING_UP
            trend_dir = "up"
            strength = min(1.0, ema_spread * 50)
        elif close < ema50 and ema20 < ema50:
            condition = MarketConditionType.TRENDING_DOWN
            trend_dir = "down"
            strength = min(1.0, ema_spread * 50)
        elif atr_pct > 0.025:
            condition = MarketConditionType.HIGH_VOLATILITY
            trend_dir = None
            strength = 0.3
        else:
            condition = MarketConditionType.CHOPPY
            trend_dir = None
            strength = 0.2

        return MarketCondition(
            condition=condition,
            trend_direction=trend_dir,
            volatility_regime=vol_regime,
            trend_strength=strength,
            confidence=strength,
            timestamp=indicators.timestamp,
        )


# ── Position sizer ────────────────────────────────────────────────────────────

class _BacktestPositionSizer:
    """Kelly-based position sizer for backtesting."""

    def __init__(self, settings) -> None:
        self._settings = settings

    def compute(self, signal: TradeSignal, balance: float) -> PositionSizing:
        confidence_frac = signal.confidence / 100.0
        edge = signal.edge
        implied = signal.implied_probability

        # Kelly: f = (p * b - q) / b  where b = (1-implied)/implied
        p_win = min(0.95, confidence_frac)
        q_lose = 1.0 - p_win
        if implied > 0.0 and implied < 1.0:
            b = (1.0 - implied) / implied
        else:
            b = 1.0

        kelly = (p_win * b - q_lose) / b if b > 0 else 0.0
        kelly = max(0.0, kelly) * self._settings.kelly_fraction

        size_usdc = kelly * balance
        size_usdc = max(self._settings.min_position_usdc, min(self._settings.max_position_usdc, size_usdc))
        size_usdc = min(size_usdc, balance * self._settings.max_total_exposure_pct)

        return PositionSizing(
            recommended_size_usdc=size_usdc,
            max_size_usdc=self._settings.max_position_usdc,
            risk_amount_usdc=size_usdc * self._settings.max_risk_per_trade_pct,
            kelly_fraction=kelly,
            method="kelly" if self._settings.use_kelly else "fixed_pct",
        )


# ── BacktestEngine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Replay historical candle data through the full trading pipeline.

    Usage
    -----
    engine = BacktestEngine(settings, initial_balance=1000.0)
    metrics = await engine.run(candles_by_tf, markets)
    report  = engine.get_results()
    """

    def __init__(self, settings, initial_balance: float = 1000.0) -> None:
        self._settings = settings
        self._initial_balance = initial_balance

        self._indicator_engine = IndicatorEngine()
        self._mtf_analyzer = MultiTimeframeAnalyzer(self._indicator_engine, settings)
        self._signal_generator = SignalGenerator(settings)
        self._scorer = ConfidenceScorer()
        self._condition_detector = _MarketConditionDetector()
        self._position_sizer = _BacktestPositionSizer(settings)
        self._paper_trader = _BacktestPaperTrader(initial_balance)
        self._analyzer = PerformanceAnalyzer()

        self._equity_curve: List[Tuple[datetime, float]] = []
        self._metrics: Optional[PerformanceMetrics] = None
        self._open_positions: Dict[str, Trade] = {}  # trade_id -> Trade

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        markets: List[dict],
    ) -> PerformanceMetrics:
        """
        Replay candles chronologically through the full pipeline.

        Parameters
        ----------
        candles_by_tf:
            Dict mapping timeframe labels to ordered candle lists (oldest first).
        markets:
            List of market dicts (from generate_mock_markets or real data).

        Returns
        -------
        PerformanceMetrics computed from all simulated trades.
        """
        logger.info(
            "Backtest starting",
            initial_balance=self._initial_balance,
            market_count=len(markets),
            timeframes=list(candles_by_tf.keys()),
        )

        # Convert markets dicts to Market objects
        market_objects = [self._dict_to_market(m) for m in markets]

        # Determine the primary timeframe for stepping
        primary_tf = "1h" if "1h" in candles_by_tf else next(iter(candles_by_tf))
        primary_candles = candles_by_tf[primary_tf]

        # Minimum candles needed before we can compute indicators
        min_candles = 200
        if len(primary_candles) < min_candles:
            logger.warning(
                "Insufficient candles for backtest",
                count=len(primary_candles),
                minimum=min_candles,
            )
            now = datetime.utcnow()
            self._metrics = PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, total_pnl_pct=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                max_drawdown_pct=0.0, calmar_ratio=0.0,
                avg_holding_time_hours=0.0, start_date=now, end_date=now,
            )
            return self._metrics

        self._equity_curve.append((primary_candles[0].timestamp, self._initial_balance))

        # Step through each candle bar
        for i in range(min_candles, len(primary_candles)):
            current_candle = primary_candles[i]
            current_time = current_candle.timestamp
            btc_price = current_candle.close

            # Build window candles for each timeframe
            window_by_tf: Dict[str, List[Candle]] = {}
            for tf, tf_candles in candles_by_tf.items():
                tf_window = [c for c in tf_candles if c.timestamp <= current_time]
                if len(tf_window) >= min_candles:
                    window_by_tf[tf] = tf_window[-500:]  # keep last 500

            if not window_by_tf:
                continue

            # Resolve open positions whose markets have expired
            expired_positions = []
            for trade_id, trade in list(self._open_positions.items()):
                market = next(
                    (m for m in market_objects if m.condition_id == trade.market_id),
                    None,
                )
                if market and current_time >= market.end_date:
                    exit_price = self.simulate_resolution(
                        trade, btc_price, market.question
                    )
                    self._paper_trader.close_trade(trade, exit_price, current_time)
                    expired_positions.append(trade_id)
                    logger.debug(
                        "Position resolved",
                        trade_id=trade_id,
                        exit_price=exit_price,
                        outcome=trade.outcome.value,
                    )

            for tid in expired_positions:
                del self._open_positions[tid]

            # Compute indicators and market condition
            try:
                indicator_sets = self._indicator_engine.compute_all_timeframes(window_by_tf)
            except Exception:
                continue

            if primary_tf not in indicator_sets:
                continue

            primary_indicators = indicator_sets[primary_tf]
            market_condition = self._condition_detector.detect(primary_indicators)
            timeframe_scores = self._mtf_analyzer.analyze(window_by_tf)

            # Only scan markets every 4 bars (reduce compute in backtest)
            if i % 4 != 0:
                self._update_equity(current_time)
                continue

            # Skip if max positions reached
            if len(self._open_positions) >= self._settings.max_open_positions:
                self._update_equity(current_time)
                continue

            # Try to generate a signal for each market
            for market in market_objects:
                if market.condition_id in {t.market_id for t in self._open_positions.values()}:
                    continue

                # Skip expired or nearly-expired markets
                hours_left = (market.end_date - current_time).total_seconds() / 3600
                if hours_left < self._settings.min_time_to_resolution_hours:
                    continue
                if hours_left > self._settings.max_time_to_resolution_hours:
                    continue

                # Compute fair value
                fair_value = self._mtf_analyzer.compute_fair_value(
                    btc_price=btc_price,
                    market_question=market.question,
                    indicators=primary_indicators,
                    market_condition=market_condition,
                )

                # Generate signal
                signal = self._signal_generator.generate(
                    market=market,
                    indicators_1h=primary_indicators,
                    timeframe_scores=timeframe_scores,
                    market_condition=market_condition,
                    btc_price=btc_price,
                    fair_value=fair_value,
                )

                if signal is None:
                    continue

                # Size the position
                sizing = self._position_sizer.compute(signal, self._paper_trader.get_balance())
                if sizing.recommended_size_usdc < self._settings.min_position_usdc:
                    continue

                if self._paper_trader.get_balance() < sizing.recommended_size_usdc:
                    continue

                # Execute the trade
                trade = self._paper_trader.place_order(signal, sizing.recommended_size_usdc)
                self._open_positions[trade.trade_id] = trade
                logger.debug(
                    "Backtest trade opened",
                    market=market.condition_id,
                    direction=signal.direction.value,
                    size=sizing.recommended_size_usdc,
                    price=signal.price,
                )
                break  # one trade per bar

            self._update_equity(current_time)

        # Force-close any remaining open positions at last price
        last_candle = primary_candles[-1]
        for trade in list(self._open_positions.values()):
            market = next(
                (m for m in market_objects if m.condition_id == trade.market_id),
                None,
            )
            question = market.question if market else "Will BTC be above $95000?"
            exit_price = self.simulate_resolution(trade, last_candle.close, question)
            self._paper_trader.close_trade(trade, exit_price, last_candle.timestamp)

        self._open_positions.clear()

        # Compute metrics
        all_trades = self._paper_trader.trades
        self._metrics = self._analyzer.compute_metrics(all_trades, self._initial_balance)

        logger.info(
            "Backtest complete",
            trades=self._metrics.total_trades,
            win_rate=round(self._metrics.win_rate, 3),
            total_pnl=round(self._metrics.total_pnl, 2),
        )
        return self._metrics

    # ── Simulation helpers ────────────────────────────────────────────────────

    def simulate_resolution(
        self,
        trade: Trade,
        btc_price_at_resolution: float,
        market_question: str,
    ) -> float:
        """
        Simulate market resolution for a trade.

        Parses the question for a target price, then determines whether
        the YES condition was met. Returns the token price at resolution:
            - 0.97 if the position's direction won
            - 0.03 if the position's direction lost
        """
        target_price, direction_keyword = self._parse_question(market_question)

        if target_price is None or target_price <= 0:
            # Can't parse — assume 50/50 coin flip
            yes_wins = btc_price_at_resolution > 90000.0
        elif direction_keyword == "above":
            yes_wins = btc_price_at_resolution > target_price
        else:  # "below"
            yes_wins = btc_price_at_resolution < target_price

        # Return resolution price from our position's perspective
        if trade.direction == Direction.YES:
            return 0.97 if yes_wins else 0.03
        else:  # Direction.NO
            return 0.97 if not yes_wins else 0.03

    @staticmethod
    def _parse_question(question: str) -> Tuple[Optional[float], str]:
        """Extract target price and direction from a market question."""
        text = question.lower()
        direction = "above"
        if "below" in text or "under" in text or "less than" in text:
            direction = "below"

        pattern = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
        matches = pattern.findall(question)
        if not matches:
            return None, direction

        raw = matches[-1].replace(",", "").strip()
        try:
            return float(raw), direction
        except ValueError:
            return None, direction

    # ── Mock market generator ─────────────────────────────────────────────────

    @staticmethod
    def generate_mock_markets(
        n: int = 5,
        current_btc_price: float = 95000.0,
    ) -> List[dict]:
        """
        Generate realistic synthetic BTC Polymarket markets for testing.

        Produces a mix of "above" and "below" threshold questions at varying
        strike prices (±5% to ±20% from current price) and time horizons.

        Parameters
        ----------
        n:
            Number of markets to generate.
        current_btc_price:
            Reference BTC price to anchor strike levels.

        Returns
        -------
        List[dict] — each dict can be passed to BacktestEngine.run() or
        converted via _dict_to_market().
        """
        now = datetime.utcnow()
        rng = np.random.default_rng(42)

        scenarios = []
        price_multipliers = [0.95, 0.97, 1.00, 1.03, 1.05, 1.08, 1.10, 1.15, 0.92, 1.20]
        directions = ["above", "above", "above", "above", "below", "above", "above", "below", "below", "above"]
        horizons_days = [7, 14, 3, 30, 7, 14, 21, 7, 14, 30]

        for idx in range(n):
            mult = price_multipliers[idx % len(price_multipliers)]
            direction = directions[idx % len(directions)]
            horizon = horizons_days[idx % len(horizons_days)]
            strike = round(current_btc_price * mult, -3)  # round to nearest 1000

            end_date = now + timedelta(days=horizon)
            condition_id = f"btc-{direction}-{int(strike)}-{end_date.strftime('%Y%m%d')}"

            if direction == "above":
                question = f"Will BTC be above ${strike:,.0f} by {end_date.strftime('%B %d, %Y')}?"
            else:
                question = f"Will BTC be below ${strike:,.0f} by {end_date.strftime('%B %d, %Y')}?"

            # Implied probability based on distance from current price
            distance = (current_btc_price - strike) / strike
            if direction == "above":
                raw_prob = 1.0 / (1.0 + np.exp(-20.0 * distance))
            else:
                raw_prob = 1.0 / (1.0 + np.exp(20.0 * distance))

            yes_price = float(np.clip(raw_prob + rng.normal(0, 0.03), 0.05, 0.95))
            no_price = round(1.0 - yes_price, 4)

            scenarios.append({
                "condition_id": condition_id,
                "question": question,
                "end_date": end_date.isoformat(),
                "yes_token_id": f"{condition_id}-yes",
                "no_token_id": f"{condition_id}-no",
                "yes_price": round(yes_price, 4),
                "no_price": round(no_price, 4),
                "volume": float(rng.uniform(10000, 500000)),
                "liquidity": float(rng.uniform(1000, 50000)),
                "active": True,
            })

        return scenarios[:n]

    # ── Results ───────────────────────────────────────────────────────────────

    def get_results(self) -> dict:
        """
        Return a complete backtest report dictionary.

        Keys:
            metrics         — PerformanceMetrics dataclass
            equity_curve    — List[(datetime, float)]
            trades          — List[Trade]
            parameters      — dict of key settings used
        """
        return {
            "metrics": self._metrics,
            "equity_curve": self._equity_curve,
            "trades": self._paper_trader.trades,
            "parameters": {
                "initial_balance": self._initial_balance,
                "min_confidence_threshold": self._settings.min_confidence_threshold,
                "kelly_fraction": self._settings.kelly_fraction,
                "max_position_usdc": self._settings.max_position_usdc,
                "min_position_usdc": self._settings.min_position_usdc,
                "max_risk_per_trade_pct": self._settings.max_risk_per_trade_pct,
                "max_daily_drawdown_pct": self._settings.max_daily_drawdown_pct,
                "paper_trading": self._settings.paper_trading,
            },
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_equity(self, timestamp: datetime) -> None:
        """Append current balance to the equity curve."""
        balance = self._paper_trader.get_balance()
        # Subtract unrealized PnL of open positions at cost basis
        self._equity_curve.append((timestamp, balance))

    @staticmethod
    def _dict_to_market(m: dict) -> Market:
        """Convert a market dict (from generate_mock_markets) to a Market object."""
        end_date_raw = m.get("end_date", "")
        if isinstance(end_date_raw, str):
            try:
                end_date = datetime.fromisoformat(end_date_raw)
            except ValueError:
                end_date = datetime.utcnow() + timedelta(days=7)
        elif isinstance(end_date_raw, datetime):
            end_date = end_date_raw
        else:
            end_date = datetime.utcnow() + timedelta(days=7)

        yes_token = PolymarketToken(
            token_id=m.get("yes_token_id", f"{m.get('condition_id', 'unknown')}-yes"),
            outcome="Yes",
            price=m.get("yes_price", 0.5),
        )
        no_token = PolymarketToken(
            token_id=m.get("no_token_id", f"{m.get('condition_id', 'unknown')}-no"),
            outcome="No",
            price=m.get("no_price", 0.5),
        )

        return Market(
            condition_id=m.get("condition_id", str(uuid.uuid4())),
            question=m.get("question", "Will BTC be above $95,000?"),
            tokens=[yes_token, no_token],
            end_date=end_date,
            active=m.get("active", True),
            volume=m.get("volume", 10000.0),
            liquidity=m.get("liquidity", 1000.0),
        )
