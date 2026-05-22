"""
Machine learning confidence booster using scikit-learn GradientBoostingClassifier.

Trains on historical trade outcomes and indicator feature vectors to produce a
probability-based confidence adjustment that supplements the rule-based scorer.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.core.models import Trade, TradeOutcome

# Graceful import — sklearn is optional
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; MLScorer will return 0.5 for all predictions.")


# ── Feature schema ─────────────────────────────────────────────────────────────
# Exactly 15 features in this order; callers must supply all of them.
_FEATURE_NAMES = [
    "rsi",
    "ema_spread_20_50",
    "ema_spread_50_200",
    "macd_histogram",
    "volume_ratio",
    "atr_pct",
    "momentum",
    "score_1h",
    "score_4h",
    "score_15m",
    "score_5m",
    "hours_to_resolution",
    "edge",
    "implied_prob",
    "sentiment_score",
]

_N_FEATURES = len(_FEATURE_NAMES)  # 15


class MLScorer:
    """
    Gradient Boosting binary classifier that predicts the probability a trade
    will be a WIN given a 15-dimensional feature vector.

    The model can be trained offline from historical trade data or updated
    incrementally as new results arrive.

    Parameters
    ----------
    model_path:
        File path for saving/loading the pickled model bundle.
    """

    def __init__(self, model_path: str = "data/ml_model.pkl") -> None:
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._trained = False

        # Attempt to load a pre-existing model
        if os.path.exists(model_path):
            try:
                self.load_model()
            except Exception as exc:
                logger.warning("MLScorer: failed to load existing model", error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        trades: List[Trade],
        indicator_history: List[Dict[str, Any]],
    ) -> None:
        """
        Train the GradientBoostingClassifier on historical trade outcomes.

        Parameters
        ----------
        trades:
            Closed Trade objects (WIN / LOSS / BREAK_EVEN).
        indicator_history:
            Parallel list of indicator feature dicts (see _FEATURE_NAMES).
            Must be the same length as ``trades``.

        Training is skipped when:
            - sklearn is unavailable
            - fewer than 10 closed trades are available
            - X/y dimensions do not match
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("MLScorer.train: sklearn not available, skipping.")
            return

        closed = [
            (t, ind) for t, ind in zip(trades, indicator_history)
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAK_EVEN)
        ]

        if len(closed) < 10:
            logger.info(
                "MLScorer: not enough closed trades to train",
                count=len(closed),
                required=10,
            )
            return

        X_rows, y_labels = [], []
        for trade, ind in closed:
            features = self._extract_features(ind)
            if features is not None:
                X_rows.append(features)
                y_labels.append(1 if trade.outcome == TradeOutcome.WIN else 0)

        if len(X_rows) < 10:
            logger.warning("MLScorer: insufficient valid feature rows after extraction.")
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_labels, dtype=int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            random_state=42,
        )
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler
        self._trained = True

        win_pct = sum(y) / len(y) * 100
        logger.info(
            "MLScorer trained",
            samples=len(X_rows),
            win_rate=f"{win_pct:.1f}%",
            features=_N_FEATURES,
        )
        self.save_model()

    def update_model(self, new_trades: List[Trade]) -> None:
        """
        Perform incremental warm-start training on new trades.

        When the model is not yet trained this method is a no-op (train first).
        When sklearn is unavailable this method is a no-op.

        Parameters
        ----------
        new_trades:
            New closed Trade objects to incorporate.  Feature extraction uses
            only data embedded in the Trade (confidence, entry/exit price, etc.)
            so no separate indicator_history is required.
        """
        if not _SKLEARN_AVAILABLE or not self._trained:
            return

        closed = [
            t for t in new_trades
            if t.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS)
            and t.realized_pnl is not None
        ]

        if not closed:
            return

        # Build minimal feature vectors from embedded trade data
        X_rows, y_labels = [], []
        for t in closed:
            # Construct a minimal feature dict from trade fields
            ind = {
                "rsi": 55.0,               # unknown at update time
                "ema_spread_20_50": 0.0,
                "ema_spread_50_200": 0.0,
                "macd_histogram": 0.0,
                "volume_ratio": 1.0,
                "atr_pct": 0.02,
                "momentum": 0.0,
                "score_1h": 0.0,
                "score_4h": 0.0,
                "score_15m": 0.0,
                "score_5m": 0.0,
                "hours_to_resolution": 24.0,
                "edge": max(0.0, t.entry_price - 0.5) * 2,
                "implied_prob": t.entry_price,
                "sentiment_score": 0.0,
            }
            features = self._extract_features(ind)
            if features is not None:
                X_rows.append(features)
                y_labels.append(1 if t.outcome == TradeOutcome.WIN else 0)

        if not X_rows:
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_labels, dtype=int)

        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        self._model.n_estimators += 10  # type: ignore[union-attr]
        self._model.fit(X_scaled, y)  # type: ignore[union-attr]

        logger.debug("MLScorer model updated", new_samples=len(X_rows))
        self.save_model()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_confidence(self, features: Dict[str, Any]) -> float:
        """
        Predict the win probability for a given feature vector.

        Parameters
        ----------
        features:
            Dict with keys matching ``_FEATURE_NAMES``.

        Returns
        -------
        float in [0.0, 1.0].  Returns 0.5 when the model is not trained or
        sklearn is unavailable.
        """
        if not _SKLEARN_AVAILABLE or not self._trained:
            return 0.5

        feat_arr = self._extract_features(features)
        if feat_arr is None:
            return 0.5

        X = feat_arr.reshape(1, -1)
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]

        try:
            prob = self._model.predict_proba(X_scaled)[0][1]  # type: ignore[union-attr]
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as exc:
            logger.warning("MLScorer predict error", error=str(exc))
            return 0.5

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self) -> None:
        """Persist the model bundle (classifier + scaler) to disk as pickle."""
        if not self._trained:
            return

        os.makedirs(os.path.dirname(self._model_path) or ".", exist_ok=True)
        bundle = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": _FEATURE_NAMES,
        }
        with open(self._model_path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("MLScorer model saved", path=self._model_path)

    def load_model(self) -> None:
        """Load a previously saved model bundle from disk."""
        with open(self._model_path, "rb") as f:
            bundle = pickle.load(f)

        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._trained = True
        logger.info("MLScorer model loaded", path=self._model_path)

    def is_trained(self) -> bool:
        """Return True if the model has been trained (or loaded from disk)."""
        return self._trained

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(ind: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert a feature dict to a fixed-length numpy array.

        Returns None if any feature value cannot be cast to float.
        """
        try:
            row = np.array(
                [float(ind.get(k, 0.0)) for k in _FEATURE_NAMES],
                dtype=float,
            )
            if np.any(np.isnan(row)) or np.any(np.isinf(row)):
                # Replace nan/inf with neutral defaults
                row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0)
            return row
        except (TypeError, ValueError):
            return None
