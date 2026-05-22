"""
Volatility regime detection using statistical methods.

Provides:
    VolatilityRegime   — enum (imported from models)
    RegimeDetector     — realized vol, Bollinger breakout, Hurst exponent
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from loguru import logger

from src.core.models import Candle, VolatilityRegime


class RegimeDetector:
    """
    Detect the current BTC volatility regime using realized volatility,
    historical percentiles, Bollinger Band breakouts, and Hurst exponent.

    All methods are synchronous and stateless — they can be called in an
    async context without blocking concerns.
    """

    # ── Regime classification thresholds (annualized %) ──────────────────────
    _LOW_VOL_THRESHOLD: float = 0.45        # < 45 % annualized → LOW
    _MEDIUM_VOL_THRESHOLD: float = 0.75     # 45–75 % → MEDIUM
    _HIGH_VOL_THRESHOLD: float = 1.20       # 75–120 % → HIGH
    # ≥ 120 % → EXTREME

    def detect_regime(
        self,
        candles: List[Candle],
        lookback: int = 50,
    ) -> dict:
        """
        Classify the current volatility regime.

        Parameters
        ----------
        candles:
            Ordered list of Candle objects (oldest first); at least ``lookback``
            candles are required.
        lookback:
            Number of recent candles used for rolling realized volatility.

        Returns
        -------
        dict with keys:
            regime               — VolatilityRegime
            realized_vol_annual  — float (annualized volatility fraction)
            percentile           — float 0–100 (vol percentile vs history)
            description          — str human-readable summary
        """
        if len(candles) < max(lookback, 2):
            logger.warning(
                "Insufficient candles for regime detection",
                n=len(candles),
                required=lookback,
            )
            return {
                "regime": VolatilityRegime.MEDIUM,
                "realized_vol_annual": 0.60,
                "percentile": 50.0,
                "description": "Insufficient data — defaulting to MEDIUM regime.",
            }

        closes = [c.close for c in candles]

        # Current realized vol (last `lookback` bars)
        recent_closes = closes[-lookback:]
        current_vol = self.compute_realized_volatility(recent_closes, window=min(20, lookback))

        # Historical percentile (using full history)
        hist_vol = self._rolling_vols(closes, window=20)
        percentile = self._percentile_rank(hist_vol, current_vol)

        # Classify
        if current_vol < self._LOW_VOL_THRESHOLD:
            regime = VolatilityRegime.LOW
            desc = f"Low volatility ({current_vol * 100:.1f}% ann.) — range-bound, tight spreads."
        elif current_vol < self._MEDIUM_VOL_THRESHOLD:
            regime = VolatilityRegime.MEDIUM
            desc = f"Medium volatility ({current_vol * 100:.1f}% ann.) — normal trending conditions."
        elif current_vol < self._HIGH_VOL_THRESHOLD:
            regime = VolatilityRegime.HIGH
            desc = f"High volatility ({current_vol * 100:.1f}% ann.) — elevated risk, reduce sizing."
        else:
            regime = VolatilityRegime.EXTREME
            desc = f"Extreme volatility ({current_vol * 100:.1f}% ann.) — avoid new positions."

        logger.debug(
            "Regime detected",
            regime=regime.value,
            vol_annual=round(current_vol, 4),
            percentile=round(percentile, 1),
        )

        return {
            "regime": regime,
            "realized_vol_annual": round(current_vol, 6),
            "percentile": round(percentile, 2),
            "description": desc,
        }

    def compute_realized_volatility(
        self,
        closes: List[float],
        window: int = 20,
    ) -> float:
        """
        Compute annualized realized volatility from log returns.

        Formula:
            log_returns  = ln(price[i] / price[i-1])
            sigma_period = std(log_returns, ddof=1)
            sigma_annual = sigma_period * sqrt(252)   [for daily candles]

        If fewer than ``window + 1`` prices are provided, computes over the
        full available history.

        Parameters
        ----------
        closes:
            List of closing prices (oldest first).
        window:
            Rolling window size; the last ``window`` returns are used.

        Returns
        -------
        float — annualized volatility (e.g. 0.80 = 80 %).
        """
        arr = np.asarray(closes, dtype=float)
        if len(arr) < 2:
            return 0.0

        log_returns = np.diff(np.log(arr))

        # Use last `window` returns, capped at available data
        tail = log_returns[-window:] if len(log_returns) >= window else log_returns

        if len(tail) < 2:
            return 0.0

        sigma = float(np.std(tail, ddof=1))
        annualized = sigma * math.sqrt(252)
        return annualized

    def detect_breakout(
        self,
        candles: List[Candle],
        lookback: int = 20,
    ) -> Optional[str]:
        """
        Detect a Bollinger Band breakout on the most recent candle.

        Uses the standard 20-bar Bollinger Bands (middle band = SMA-20,
        upper/lower = SMA-20 ± 2*std).

        Parameters
        ----------
        candles:
            Ordered candle list (oldest first); needs at least ``lookback + 1``.
        lookback:
            Bollinger Band period (default 20).

        Returns
        -------
        'up'   — close broke above the upper band
        'down' — close broke below the lower band
        None   — no breakout
        """
        if len(candles) < lookback + 1:
            return None

        recent = candles[-(lookback + 1):]
        closes = np.array([c.close for c in recent], dtype=float)

        window_closes = closes[:-1]  # 20 bars for band calculation
        current_close = closes[-1]

        sma = float(np.mean(window_closes))
        std = float(np.std(window_closes, ddof=1))

        upper_band = sma + 2.0 * std
        lower_band = sma - 2.0 * std

        if current_close > upper_band:
            logger.debug(
                "Bollinger Band breakout UP",
                close=current_close,
                upper=round(upper_band, 2),
            )
            return "up"
        elif current_close < lower_band:
            logger.debug(
                "Bollinger Band breakout DOWN",
                close=current_close,
                lower=round(lower_band, 2),
            )
            return "down"

        return None

    def compute_hurst_exponent(self, prices: List[float]) -> float:
        """
        Estimate the Hurst exponent via R/S (Rescaled Range) analysis.

        Interpretation:
            H < 0.5   — mean-reverting (anti-persistent)
            H ≈ 0.5   — random walk (efficient market)
            H > 0.5   — trending (persistent)

        The R/S method divides the return series into sub-periods of
        increasing length, computes the R/S statistic for each, then
        regresses log(R/S) ~ H * log(n) via OLS.

        Parameters
        ----------
        prices:
            List of closing prices (at least 20 recommended).

        Returns
        -------
        float — Hurst exponent estimate in (0, 1).  Returns 0.5 if the
        series is too short or the regression is degenerate.
        """
        arr = np.asarray(prices, dtype=float)
        n = len(arr)

        if n < 20:
            logger.debug("Hurst: insufficient data, returning 0.5")
            return 0.5

        log_returns = np.diff(np.log(arr))
        n_returns = len(log_returns)

        # Choose sub-period sizes as powers of 2 (or linear)
        min_size = 8
        sizes: List[int] = []
        s = min_size
        while s <= n_returns // 2:
            sizes.append(s)
            s = int(s * 1.5)

        if len(sizes) < 3:
            return 0.5

        rs_values: List[float] = []
        valid_sizes: List[int] = []

        for size in sizes:
            sub_rs: List[float] = []
            # Split into non-overlapping windows of `size`
            for start in range(0, n_returns - size + 1, size):
                chunk = log_returns[start : start + size]
                mean = np.mean(chunk)
                deviations = np.cumsum(chunk - mean)
                r_range = float(np.max(deviations) - np.min(deviations))
                s_std = float(np.std(chunk, ddof=1))
                if s_std > 0:
                    sub_rs.append(r_range / s_std)

            if sub_rs:
                rs_values.append(float(np.mean(sub_rs)))
                valid_sizes.append(size)

        if len(valid_sizes) < 3:
            return 0.5

        # OLS regression: log(RS) = H * log(n) + const
        log_n = np.log(np.array(valid_sizes, dtype=float))
        log_rs = np.log(np.array(rs_values, dtype=float))

        # Simple linear regression
        n_pts = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_xx = np.sum(log_n ** 2)

        denom = n_pts * sum_xx - sum_x ** 2
        if abs(denom) < 1e-12:
            return 0.5

        hurst = (n_pts * sum_xy - sum_x * sum_y) / denom
        hurst = float(np.clip(hurst, 0.0, 1.0))

        logger.debug("Hurst exponent computed", H=round(hurst, 4), n_prices=n)
        return hurst

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rolling_vols(closes: List[float], window: int = 20) -> List[float]:
        """
        Compute a list of rolling realized volatilities across the full history.

        Used for percentile ranking the current vol level.
        """
        arr = np.asarray(closes, dtype=float)
        log_returns = np.diff(np.log(arr))
        vols: List[float] = []

        for i in range(window, len(log_returns) + 1):
            chunk = log_returns[i - window : i]
            if len(chunk) >= 2:
                vols.append(float(np.std(chunk, ddof=1)) * math.sqrt(252))

        return vols

    @staticmethod
    def _percentile_rank(series: List[float], value: float) -> float:
        """
        Return the percentile rank of *value* within *series* (0–100).

        Example: 75.0 means the value exceeds 75% of historical readings.
        """
        if not series:
            return 50.0
        below = sum(1 for v in series if v < value)
        return 100.0 * below / len(series)
