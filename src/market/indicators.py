"""
Pure numpy/pandas technical indicator calculations.

No TA-Lib dependency. All algorithms use standard Wilder smoothing or
exponential weighting as documented in each function's docstring.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.core.models import Candle, IndicatorSet

# Minimum candle counts for each indicator to be meaningful
_MIN_CANDLES_EMA200 = 200
_MIN_CANDLES_MACD = 35   # slow(26) + signal(9) warm-up
_MIN_CANDLES_RSI = 15    # period(14) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Standalone indicator functions
# ─────────────────────────────────────────────────────────────────────────────

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential moving average using pandas ewm (span definition).

    alpha = 2 / (period + 1)
    The first ``period`` elements are seeded from the simple mean of the
    initial window so the series converges faster.

    Parameters
    ----------
    prices:
        1-D array of closing prices (must be finite, no NaN).
    period:
        Lookback window (number of bars).

    Returns
    -------
    np.ndarray
        Same length as ``prices``; the first ``period - 1`` elements are NaN.
    """
    if len(prices) == 0:
        return np.array([], dtype=float)

    s = pd.Series(prices, dtype=float)
    result = s.ewm(span=period, min_periods=period, adjust=False).mean()
    return result.to_numpy(dtype=float)


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index using Wilder smoothing (EWMA with alpha=1/period).

    Classic Wilder formula:
        delta  = price[i] - price[i-1]
        gain   = delta if delta > 0 else 0
        loss   = -delta if delta < 0 else 0
        avg_gain / avg_loss smoothed with alpha = 1/period (Wilder EMA)
        RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    Parameters
    ----------
    prices:
        1-D closing price array.
    period:
        RSI period (default 14).

    Returns
    -------
    np.ndarray
        RSI values in [0, 100]; first ``period`` elements are NaN.
    """
    if len(prices) < 2:
        return np.full(len(prices), np.nan)

    prices_f = np.asarray(prices, dtype=float)
    deltas = np.diff(prices_f)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha = 1.0 / period
    result = np.full(len(prices_f), np.nan)

    # Seed with simple mean over the first 'period' deltas
    if len(deltas) < period:
        return result

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss

    # Back-fill from the first complete period onward
    # Re-run from the beginning to populate all valid positions
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        idx = i + 1   # offset because deltas has len-1 elements
        if avg_loss == 0.0:
            result[idx] = 100.0
        else:
            result[idx] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result


def macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator.

    macd_line   = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram   = macd_line - signal_line

    Parameters
    ----------
    prices:
        1-D closing price array.
    fast, slow, signal:
        Standard MACD periods.

    Returns
    -------
    Tuple of (macd_line, signal_line, histogram) — all same length as prices.
    NaN where insufficient data.
    """
    prices_f = np.asarray(prices, dtype=float)
    ema_fast = ema(prices_f, fast)
    ema_slow = ema(prices_f, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Average True Range using Wilder smoothing (same alpha = 1/period).

    True Range = max(high - low,
                     |high - prev_close|,
                     |low  - prev_close|)

    Parameters
    ----------
    highs, lows, closes:
        Arrays of equal length.
    period:
        ATR period (default 14).

    Returns
    -------
    np.ndarray
        ATR values; first element is NaN (no prev_close for TR).
    """
    highs_f = np.asarray(highs, dtype=float)
    lows_f = np.asarray(lows, dtype=float)
    closes_f = np.asarray(closes, dtype=float)

    n = len(closes_f)
    tr = np.full(n, np.nan)

    if n < 2:
        return tr

    prev_closes = closes_f[:-1]
    hl = highs_f[1:] - lows_f[1:]
    hc = np.abs(highs_f[1:] - prev_closes)
    lc = np.abs(lows_f[1:] - prev_closes)
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    result = np.full(n, np.nan)

    if n - 1 < period:
        return result

    # Seed ATR with simple mean of first 'period' TR values
    first_valid_tr = tr[1: period + 1]
    if np.any(np.isnan(first_valid_tr)):
        return result

    atr_val = float(np.mean(first_valid_tr))
    result[period] = atr_val

    alpha = 1.0 / period
    for i in range(period + 1, n):
        if np.isnan(tr[i]):
            result[i] = result[i - 1]
        else:
            atr_val = alpha * tr[i] + (1.0 - alpha) * atr_val
            result[i] = atr_val

    return result


def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Rate of change: (price - price_n_bars_ago) / price_n_bars_ago * 100.

    Parameters
    ----------
    prices:
        1-D closing price array.
    period:
        Number of bars to look back.

    Returns
    -------
    np.ndarray
        Momentum values in percent; first ``period`` elements are NaN.
    """
    prices_f = np.asarray(prices, dtype=float)
    n = len(prices_f)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    prior = prices_f[:-period]
    current = prices_f[period:]
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = np.where(prior != 0.0, (current - prior) / prior * 100.0, np.nan)
    result[period:] = roc
    return result


def volume_profile(
    volumes: np.ndarray,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling volume moving average and ratio.

    volume_ma    = simple moving average of ``volumes`` over ``period`` bars.
    volume_ratio = current_volume / volume_ma   (element-wise)

    Parameters
    ----------
    volumes:
        1-D volume array.
    period:
        Rolling window (default 20).

    Returns
    -------
    Tuple of (volume_ma, volume_ratio) — same length as ``volumes``.
    NaN where insufficient data.
    """
    volumes_f = np.asarray(volumes, dtype=float)
    s = pd.Series(volumes_f)
    vol_ma = s.rolling(window=period, min_periods=period).mean().to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_ma > 0.0, volumes_f / vol_ma, np.nan)

    return vol_ma, vol_ratio


# ─────────────────────────────────────────────────────────────────────────────
# IndicatorEngine
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorEngine:
    """
    Computes the full indicator suite from raw OHLCV candles and packages
    results into an IndicatorSet dataclass.
    """

    def compute(self, candles: List[Candle]) -> IndicatorSet:
        """
        Compute all indicators from a list of Candle objects.

        Requires a minimum of 200 candles so that EMA-200 is meaningful.
        Raises ValueError if fewer candles are supplied.

        Parameters
        ----------
        candles:
            List of Candle objects ordered chronologically (oldest first).

        Returns
        -------
        IndicatorSet
            Latest values for all indicators.

        Raises
        ------
        ValueError
            If ``len(candles) < 200``.
        """
        if len(candles) < _MIN_CANDLES_EMA200:
            raise ValueError(
                f"IndicatorEngine.compute requires at least {_MIN_CANDLES_EMA200} "
                f"candles; received {len(candles)}."
            )

        closes = np.array([c.close for c in candles], dtype=float)
        highs = np.array([c.high for c in candles], dtype=float)
        lows = np.array([c.low for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)
        timeframe = candles[-1].timeframe
        timestamp = candles[-1].timestamp

        logger.debug(
            "Computing indicators",
            timeframe=timeframe,
            n_candles=len(candles),
            latest_close=closes[-1],
        )

        # ── EMAs ──────────────────────────────────────────────────────────────
        ema_20_arr = ema(closes, 20)
        ema_50_arr = ema(closes, 50)
        ema_200_arr = ema(closes, 200)

        ema_20_val = self._last_valid(ema_20_arr)
        ema_50_val = self._last_valid(ema_50_arr)
        ema_200_val = self._last_valid(ema_200_arr)

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_arr = rsi(closes, period=14)
        rsi_val = self._last_valid(rsi_arr, fallback=50.0)

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_line, signal_line, histogram = macd(closes)
        macd_val = self._last_valid(macd_line, fallback=0.0)
        signal_val = self._last_valid(signal_line, fallback=0.0)
        histogram_val = self._last_valid(histogram, fallback=0.0)

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_arr = atr(highs, lows, closes, period=14)
        atr_val = self._last_valid(atr_arr, fallback=closes[-1] * 0.01)
        close_val = float(closes[-1])
        atr_pct_val = atr_val / close_val if close_val > 0 else 0.0

        # ── Volume ────────────────────────────────────────────────────────────
        vol_ma_arr, vol_ratio_arr = volume_profile(volumes, period=20)
        vol_ma_val = self._last_valid(vol_ma_arr, fallback=float(np.mean(volumes[-20:])))
        vol_ratio_val = self._last_valid(vol_ratio_arr, fallback=1.0)

        # ── Momentum ──────────────────────────────────────────────────────────
        mom_arr = momentum(closes, period=10)
        mom_val = self._last_valid(mom_arr, fallback=0.0)

        return IndicatorSet(
            timestamp=timestamp,
            timeframe=timeframe,
            close=close_val,
            ema_20=ema_20_val,
            ema_50=ema_50_val,
            ema_200=ema_200_val,
            rsi=rsi_val,
            macd=macd_val,
            macd_signal=signal_val,
            macd_histogram=histogram_val,
            atr=atr_val,
            atr_pct=atr_pct_val,
            volume_ma=vol_ma_val,
            volume_ratio=vol_ratio_val,
            momentum=mom_val,
        )

    def compute_all_timeframes(
        self,
        candles_by_tf: Dict[str, List[Candle]],
    ) -> Dict[str, IndicatorSet]:
        """
        Run ``compute()`` for every timeframe in ``candles_by_tf``.

        Timeframes with insufficient data are logged and skipped.

        Parameters
        ----------
        candles_by_tf:
            Mapping of timeframe label (e.g. ``"1h"``) to ordered candle list.

        Returns
        -------
        Dict[str, IndicatorSet]
            Keyed by timeframe label; only contains successful computations.
        """
        results: Dict[str, IndicatorSet] = {}

        for tf, candles in candles_by_tf.items():
            try:
                ind = self.compute(candles)
                results[tf] = ind
                logger.debug(
                    "Indicators computed",
                    timeframe=tf,
                    rsi=round(ind.rsi, 2),
                    ema_20=round(ind.ema_20, 2),
                    atr_pct=round(ind.atr_pct * 100, 3),
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping timeframe — insufficient candles",
                    timeframe=tf,
                    n_candles=len(candles),
                    error=str(exc),
                )
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "Unexpected error computing indicators",
                    timeframe=tf,
                    error=str(exc),
                )

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _last_valid(arr: np.ndarray, fallback: float = 0.0) -> float:
        """Return the last non-NaN value in an array, or ``fallback``."""
        valid = arr[~np.isnan(arr)]
        return float(valid[-1]) if len(valid) > 0 else fallback
