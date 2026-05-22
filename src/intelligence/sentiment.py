"""
News and social sentiment analysis module.

Gracefully disables itself when no API key is configured — returns neutral
scores so the rest of the pipeline continues uninterrupted.

Data sources:
    CryptoPanic API — curated crypto news with vote-based sentiment signals.
    Alternative.me Fear & Greed Index — aggregated market sentiment score (0–100).
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


# ── URL constants ─────────────────────────────────────────────────────────────

_CRYPTOPANIC_URL = (
    "https://cryptopanic.com/api/free/v1/posts/"
    "?auth_token={key}&currencies=BTC&kind=news"
)
_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

# Window for "recent" articles
_RECENT_HOURS = 4


class SentimentAnalyzer:
    """
    Combine CryptoPanic news sentiment and Fear & Greed index into a single
    [-1, +1] score that can be used as a confidence modifier.

    Parameters
    ----------
    settings:
        Application Settings object.  The following attributes are used:

        ``cryptopanic_api_key``  — CryptoPanic auth token (empty = disabled)
        ``sentiment_enabled``   — Master flag; when False, always returns neutral.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._api_key: str = getattr(settings, "cryptopanic_api_key", "")
        self._enabled: bool = bool(getattr(settings, "sentiment_enabled", False))

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_btc_sentiment(self) -> Dict[str, Any]:
        """
        Fetch and parse BTC news sentiment from CryptoPanic.

        Returns a dict with keys:
            score          — float in [-1, +1]; 0.0 when disabled/unavailable
            bullish_count  — int
            bearish_count  — int
            neutral_count  — int
            articles       — List[dict] (title, url, published_at, votes)
            timestamp      — ISO 8601 string
            source         — 'cryptopanic' | 'disabled' | 'error'
            note           — human-readable status
        """
        ts = datetime.utcnow().isoformat()

        if not self._enabled or not self._api_key:
            return {
                "score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "articles": [],
                "timestamp": ts,
                "source": "disabled",
                "note": "Sentiment disabled or CryptoPanic API key not configured.",
            }

        url = _CRYPTOPANIC_URL.format(key=self._api_key)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "CryptoPanic API returned non-200",
                            status=resp.status,
                        )
                        return self._neutral_sentiment("error", f"HTTP {resp.status}")

                    data = await resp.json()

            parsed = self._parse_cryptopanic(data)
            logger.debug(
                "CryptoPanic sentiment fetched",
                bullish=parsed["bullish_count"],
                bearish=parsed["bearish_count"],
                score=round(parsed["score"], 3),
            )
            return parsed

        except aiohttp.ClientError as exc:
            logger.warning("CryptoPanic request failed", error=str(exc))
            return self._neutral_sentiment("error", str(exc))
        except Exception as exc:
            logger.error("Unexpected error in get_btc_sentiment", error=str(exc))
            return self._neutral_sentiment("error", str(exc))

    def _parse_cryptopanic(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a raw CryptoPanic API response.

        Only articles published within the last ``_RECENT_HOURS`` hours are
        counted.  Vote counts determine sentiment classification:
            liked > disliked and liked > 2  → bullish
            disliked > liked and disliked > 2 → bearish
            otherwise                         → neutral
        """
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(hours=_RECENT_HOURS)

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        articles: List[Dict[str, Any]] = []

        for post in response.get("results", []):
            published_raw = post.get("published_at", "")
            try:
                pub_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if pub_dt < cutoff:
                continue

            votes = post.get("votes", {})
            liked = int(votes.get("liked", 0))
            disliked = int(votes.get("disliked", 0))
            important = int(votes.get("important", 0))
            saved = int(votes.get("saved", 0))

            # Classify by votes
            if liked > disliked and liked > 2:
                sentiment_label = "bullish"
                bullish_count += 1
            elif disliked > liked and disliked > 2:
                sentiment_label = "bearish"
                bearish_count += 1
            else:
                sentiment_label = "neutral"
                neutral_count += 1

            articles.append({
                "title": post.get("title", ""),
                "url": post.get("url", ""),
                "published_at": published_raw,
                "sentiment": sentiment_label,
                "liked": liked,
                "disliked": disliked,
                "important": important,
                "saved": saved,
            })

        total = bullish_count + bearish_count + neutral_count
        if total > 0:
            # Weighted score: bullish = +1, bearish = -1, neutral = 0
            score = float((bullish_count - bearish_count) / total)
        else:
            score = 0.0

        return {
            "score": round(score, 4),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "articles": articles,
            "timestamp": now.isoformat(),
            "source": "cryptopanic",
            "note": f"Analysed {total} articles from last {_RECENT_HOURS}h.",
        }

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Fetch the latest Crypto Fear & Greed Index from alternative.me.

        Returns a dict with keys:
            value          — int 0-100
            classification — str (e.g. "Extreme Fear", "Greed")
            timestamp      — ISO 8601 string
            source         — 'alternative.me' | 'error'
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return self._default_fear_greed(f"HTTP {resp.status}")

                    data = await resp.json(content_type=None)

            fng_data = data.get("data", [{}])[0]
            value = int(fng_data.get("value", 50))
            classification = fng_data.get("value_classification", "Neutral")
            ts = fng_data.get("timestamp", str(int(datetime.utcnow().timestamp())))

            # Convert Unix ts to ISO
            try:
                dt = datetime.utcfromtimestamp(int(ts))
                iso_ts = dt.isoformat()
            except (ValueError, TypeError):
                iso_ts = datetime.utcnow().isoformat()

            logger.debug(
                "Fear & Greed index fetched",
                value=value,
                classification=classification,
            )
            return {
                "value": value,
                "classification": classification,
                "timestamp": iso_ts,
                "source": "alternative.me",
            }

        except aiohttp.ClientError as exc:
            logger.warning("Fear & Greed request failed", error=str(exc))
            return self._default_fear_greed(str(exc))
        except Exception as exc:
            logger.error("Unexpected error in get_fear_greed_index", error=str(exc))
            return self._default_fear_greed(str(exc))

    def compute_sentiment_score(
        self,
        sentiment_data: Dict[str, Any],
        fear_greed: Dict[str, Any],
    ) -> float:
        """
        Blend CryptoPanic news sentiment and Fear & Greed into a [-1, +1] score.

        Weights:
            CryptoPanic news sentiment : 40 %
            Fear & Greed index         : 60 %

        Fear & Greed normalisation:
            0   → -1.0 (extreme fear)
            50  →  0.0 (neutral)
            100 → +1.0 (extreme greed)

        Parameters
        ----------
        sentiment_data:
            Output of :meth:`get_btc_sentiment`.
        fear_greed:
            Output of :meth:`get_fear_greed_index`.

        Returns
        -------
        float in [-1.0, +1.0].
        """
        news_score = float(sentiment_data.get("score", 0.0))

        fg_value = float(fear_greed.get("value", 50))
        fg_norm = (fg_value - 50.0) / 50.0  # maps 0→-1, 50→0, 100→+1

        blended = 0.40 * news_score + 0.60 * fg_norm
        return float(max(-1.0, min(1.0, blended)))

    def sentiment_to_signal_modifier(self, score: float) -> float:
        """
        Map a blended sentiment score to a confidence modifier in [-10, +10] points.

        Extreme readings push the modifier toward ±10; neutral readings return 0.

        |score| < 0.2  → modifier = 0       (too noisy to act on)
        |score| 0.2–0.5 → linear ramp to ±5
        |score| 0.5–1.0 → linear ramp to ±10

        Parameters
        ----------
        score:
            Blended sentiment score in [-1.0, +1.0].

        Returns
        -------
        float — confidence modifier in [-10, +10].
        """
        abs_score = abs(score)

        if abs_score < 0.2:
            modifier = 0.0
        elif abs_score < 0.5:
            # 0.2 → 0.0,  0.5 → 5.0  (linear)
            modifier = (abs_score - 0.2) / 0.3 * 5.0
        else:
            # 0.5 → 5.0,  1.0 → 10.0  (linear)
            modifier = 5.0 + (abs_score - 0.5) / 0.5 * 5.0

        # Apply sign from original score
        modifier = modifier * (1.0 if score >= 0 else -1.0)
        return float(max(-10.0, min(10.0, modifier)))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _neutral_sentiment(source: str, note: str) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "articles": [],
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "note": note,
        }

    @staticmethod
    def _default_fear_greed(note: str) -> Dict[str, Any]:
        return {
            "value": 50,
            "classification": "Neutral",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "error",
            "note": note,
        }
