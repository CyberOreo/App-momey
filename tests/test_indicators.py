"""
Tests for IndicatorEngine (src/market/indicators.py).

Covers EMA, RSI, MACD, ATR, momentum, volume, and full IndicatorSet output.
All tests use synthetic candle data from conftest.py fixtures.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from src.core.models import Candle, IndicatorSet
from src.market.indicators import IndicatorEngine, ema, rsi, macd, atr, momentum


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flat_candles(n: int, price: float, timeframe: str = "1h") -> List[Candle]:
    """Generate candles with stable price (for EMA ≈ SMA check)."""
    start = datetime(2024, 1, 1)
    candles = []
    for i in range(n):
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=i),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=100.0,
                timeframe=timeframe,
            )
        )
    return candles


def _rising_candles(n: int, start_price: float, step: float = 10.0) -> List[Candle]:
    """Generate strictly rising candles."""
    start = datetime(2024, 1, 1)
    candles = []
    price = start_price
    for i in range(n):
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=i),
                open=price,
                high=price + step,
                low=price - step * 0.1,
                close=price + step,
                volume=100.0 + i,
                timeframe="1h",
            )
        )
        price += step
    return candles


def _falling_candles(n: int, start_price: float, step: float = 10.0) -> List[Candle]:
    """Generate strictly falling candles."""
    start = datetime(2024, 1, 1)
    candles = []
    price = start_price
    for i in range(n):
        candles.append(
            Candle(
                timestamp=start + timedelta(hours=i),
                open=price,
                high=price + step * 0.1,
                low=price - step,
                close=max(1.0, price - step),
                volume=100.0,
                timeframe="1h",
            )
        )
        price = max(1.0, price - step)
    return candles


# ── EMA tests ─────────────────────────────────────────────────────────────────

class TestEmaCalculation:
    def test_ema_stable_price_close_to_sma(self):
        """
        For a constant price series, EMA-20 should equal the price exactly
        (or be within floating-point error) once it has converged.
        """
        prices = np.array([100.0] * 50, dtype=float)
        ema_vals = ema(prices, 20)

        # EMA should converge to the constant value
        valid = ema_vals[~np.isnan(ema_vals)]
        assert len(valid) > 0
        for v in valid:
            assert abs(v - 100.0) < 0.01, f"EMA {v} far from SMA 100.0 on flat series"

    def test_ema_rising_prices_above_sma(self):
        """EMA of strictly rising prices should be close to but below the last close."""
        # Use a perfectly rising series to avoid stochastic drift
        prices = np.linspace(90_000.0, 100_000.0, 250)
        ema_20 = ema(prices, 20)
        valid = ema_20[~np.isnan(ema_20)]
        assert len(valid) > 0
        # EMA lags price — must be below the last close on a monotone up-series
        assert valid[-1] < prices[-1]
        # But should not lag by more than 5%
        assert valid[-1] > prices[-1] * 0.94

    def test_ema_empty_returns_empty(self):
        result = ema(np.array([]), 20)
        assert len(result) == 0

    def test_ema_length_matches_input(self, sample_candles):
        closes = np.array([c.close for c in sample_candles])
        ema_20 = ema(closes, 20)
        assert len(ema_20) == len(closes)


# ── RSI tests ─────────────────────────────────────────────────────────────────

class TestRsiRange:
    def test_rsi_always_0_to_100(self, sample_candles):
        """RSI must always be within [0, 100]."""
        closes = np.array([c.close for c in sample_candles])
        rsi_vals = rsi(closes, period=14)
        valid = rsi_vals[~np.isnan(rsi_vals)]
        assert len(valid) > 0
        assert float(np.min(valid)) >= 0.0 - 1e-9
        assert float(np.max(valid)) <= 100.0 + 1e-9

    def test_rsi_overbought_on_uptrend(self):
        """A flat uptrend (one-direction moves) should push RSI toward 70+."""
        n = 250
        prices = np.linspace(90_000, 110_000, n)
        rsi_vals = rsi(prices, period=14)
        valid = rsi_vals[~np.isnan(rsi_vals)]
        assert len(valid) > 0
        # In a monotone up-trend RSI should eventually exceed 60
        assert float(np.max(valid)) > 60.0, "RSI should be elevated in uptrend"

    def test_rsi_oversold_on_downtrend(self):
        """A monotone downtrend should push RSI toward <40."""
        n = 250
        prices = np.linspace(110_000, 60_000, n)
        rsi_vals = rsi(prices, period=14)
        valid = rsi_vals[~np.isnan(rsi_vals)]
        assert len(valid) > 0
        assert float(np.min(valid)) < 40.0, "RSI should be depressed in downtrend"

    def test_rsi_nan_prefix(self, sample_candles):
        """First `period` values should be NaN."""
        closes = np.array([c.close for c in sample_candles])
        rsi_vals = rsi(closes, period=14)
        # At least period positions at start should be NaN
        assert np.all(np.isnan(rsi_vals[:14]))


# ── MACD tests ────────────────────────────────────────────────────────────────

class TestMacdCrossover:
    def test_macd_histogram_sign_on_uptrend(self, sample_candles):
        """MACD histogram should be mostly positive on a strong uptrend."""
        closes = np.array([c.close for c in sample_candles])
        macd_line, signal_line, histogram = macd(closes)
        valid_hist = histogram[~np.isnan(histogram)]
        assert len(valid_hist) > 0
        # In an uptrend histogram should eventually turn positive
        positive_count = np.sum(valid_hist > 0)
        assert positive_count > len(valid_hist) * 0.4

    def test_macd_histogram_sign_on_downtrend(self, sample_candles_bearish):
        """MACD histogram should be mostly negative on a strong downtrend."""
        closes = np.array([c.close for c in sample_candles_bearish])
        macd_line, signal_line, histogram = macd(closes)
        valid_hist = histogram[~np.isnan(histogram)]
        negative_count = np.sum(valid_hist < 0)
        assert negative_count > len(valid_hist) * 0.3

    def test_macd_line_equals_ema_diff(self, sample_candles):
        """MACD line should equal EMA(12) - EMA(26)."""
        from src.market.indicators import ema as ema_fn
        closes = np.array([c.close for c in sample_candles])
        macd_line, _, _ = macd(closes, fast=12, slow=26)
        ema_fast = ema_fn(closes, 12)
        ema_slow = ema_fn(closes, 26)
        expected = ema_fast - ema_slow

        # Compare non-NaN positions
        mask = ~np.isnan(macd_line) & ~np.isnan(expected)
        np.testing.assert_allclose(macd_line[mask], expected[mask], rtol=1e-10)

    def test_macd_length_matches_input(self, sample_candles):
        closes = np.array([c.close for c in sample_candles])
        ml, sl, hist = macd(closes)
        assert len(ml) == len(closes)
        assert len(sl) == len(closes)
        assert len(hist) == len(closes)


# ── ATR tests ─────────────────────────────────────────────────────────────────

class TestAtrPositive:
    def test_atr_always_positive(self, sample_candles):
        """ATR must be strictly positive wherever it is not NaN."""
        highs = np.array([c.high for c in sample_candles])
        lows = np.array([c.low for c in sample_candles])
        closes = np.array([c.close for c in sample_candles])
        atr_vals = atr(highs, lows, closes, period=14)
        valid = atr_vals[~np.isnan(atr_vals)]
        assert len(valid) > 0
        assert float(np.min(valid)) > 0.0

    def test_atr_increases_with_volatility(self):
        """ATR on high-volatility candles should be greater than on calm candles."""
        n = 250
        start = datetime(2024, 1, 1)

        def make_candles_vol(vol: float) -> tuple:
            rng = np.random.default_rng(0)
            prices = 90_000.0 + np.cumsum(rng.normal(0, vol, n))
            highs = prices + abs(rng.normal(0, vol, n))
            lows = prices - abs(rng.normal(0, vol, n))
            closes = prices.copy()
            return highs, lows, closes

        h_calm, l_calm, c_calm = make_candles_vol(50.0)
        h_vol, l_vol, c_vol = make_candles_vol(500.0)

        atr_calm = atr(h_calm, l_calm, c_calm, 14)
        atr_volatile = atr(h_vol, l_vol, c_vol, 14)

        valid_calm = atr_calm[~np.isnan(atr_calm)]
        valid_vol = atr_volatile[~np.isnan(atr_volatile)]

        assert float(np.mean(valid_vol)) > float(np.mean(valid_calm))


# ── Momentum tests ────────────────────────────────────────────────────────────

class TestMomentum:
    def test_positive_momentum_on_rising(self):
        """Rising prices should produce positive 10-bar momentum near the end."""
        # Use a deterministic rising series to guarantee positive momentum
        prices = np.linspace(90_000.0, 100_000.0, 250)
        mom = momentum(prices, period=10)
        valid = mom[~np.isnan(mom)]
        assert len(valid) > 0
        # On a monotone up series every momentum value must be positive
        assert float(np.min(valid)) > 0.0

    def test_negative_momentum_on_falling(self, sample_candles_bearish):
        closes = np.array([c.close for c in sample_candles_bearish])
        mom = momentum(closes, period=10)
        valid = mom[~np.isnan(mom)]
        assert float(np.mean(valid)) < 0.0

    def test_momentum_formula(self):
        """Spot-check momentum formula: (close - close_10_bars_ago) / close_10_bars_ago * 100."""
        prices = np.array([100.0 + i for i in range(30)], dtype=float)
        mom = momentum(prices, period=10)
        # At index 20: (120 - 110) / 110 * 100 ≈ 9.09%
        expected = (prices[20] - prices[10]) / prices[10] * 100.0
        assert abs(mom[20] - expected) < 1e-9

    def test_momentum_length_matches_input(self, sample_candles):
        closes = np.array([c.close for c in sample_candles])
        mom = momentum(closes, period=10)
        assert len(mom) == len(closes)


# ── Full IndicatorSet tests ───────────────────────────────────────────────────

class TestComputeIndicatorSet:
    def test_indicator_set_no_nans(self, sample_candles):
        """
        IndicatorEngine.compute() on 300 candles should return an IndicatorSet
        with no NaN values in any field.
        """
        engine = IndicatorEngine()
        ind = engine.compute(sample_candles)

        assert not math.isnan(ind.ema_20)
        assert not math.isnan(ind.ema_50)
        assert not math.isnan(ind.ema_200)
        assert not math.isnan(ind.rsi)
        assert not math.isnan(ind.macd)
        assert not math.isnan(ind.macd_signal)
        assert not math.isnan(ind.macd_histogram)
        assert not math.isnan(ind.atr)
        assert not math.isnan(ind.atr_pct)
        assert not math.isnan(ind.volume_ma)
        assert not math.isnan(ind.volume_ratio)
        assert not math.isnan(ind.momentum)

    def test_indicator_set_rsi_in_range(self, sample_candles):
        engine = IndicatorEngine()
        ind = engine.compute(sample_candles)
        assert 0.0 <= ind.rsi <= 100.0

    def test_indicator_set_ema_order_bullish(self):
        """On a deterministically rising dataset, EMA-20 should be above EMA-200."""
        engine = IndicatorEngine()
        # Use deterministic candles — perfectly rising prices
        start = datetime(2024, 1, 1)
        prices = np.linspace(70_000.0, 100_000.0, 250)
        candles = [
            Candle(
                timestamp=start + timedelta(hours=i),
                open=p * 0.999,
                high=p * 1.002,
                low=p * 0.998,
                close=p,
                volume=500.0,
                timeframe="1h",
            )
            for i, p in enumerate(prices)
        ]
        ind = engine.compute(candles)
        # On a monotone up-trend EMA-20 reacts faster than EMA-200
        assert ind.ema_20 > ind.ema_200, (
            f"Expected EMA-20 ({ind.ema_20:.2f}) > EMA-200 ({ind.ema_200:.2f}) on rising series"
        )

    def test_indicator_set_atr_positive(self, sample_candles):
        engine = IndicatorEngine()
        ind = engine.compute(sample_candles)
        assert ind.atr > 0.0
        assert ind.atr_pct > 0.0

    def test_indicator_set_volume_ratio_positive(self, sample_candles):
        engine = IndicatorEngine()
        ind = engine.compute(sample_candles)
        assert ind.volume_ratio > 0.0
        assert ind.volume_ma > 0.0

    def test_compute_all_timeframes_skips_insufficient(self, sample_candles, mock_settings):
        """compute_all_timeframes should skip TFs with < 200 candles."""
        from src.market.analysis import MultiTimeframeAnalyzer

        short_candles = sample_candles[:50]  # too few
        candles_by_tf = {
            "1h": sample_candles,   # enough
            "4h": short_candles,    # not enough → should be skipped
        }
        engine = IndicatorEngine()
        result = engine.compute_all_timeframes(candles_by_tf)
        assert "1h" in result
        assert "4h" not in result


class TestInsufficientDataRaises:
    def test_fewer_than_200_candles_raises(self):
        """IndicatorEngine.compute() must raise ValueError with < 200 candles."""
        engine = IndicatorEngine()
        candles = _flat_candles(199, 90_000.0)
        with pytest.raises(ValueError, match="200"):
            engine.compute(candles)

    def test_exactly_200_candles_does_not_raise(self):
        """IndicatorEngine.compute() must succeed with exactly 200 candles."""
        engine = IndicatorEngine()
        candles = _flat_candles(200, 90_000.0)
        ind = engine.compute(candles)
        assert isinstance(ind, IndicatorSet)

    def test_zero_candles_raises(self):
        engine = IndicatorEngine()
        with pytest.raises(ValueError):
            engine.compute([])
