from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "PolyBTC Trader"
    environment: str = "development"
    paper_trading: bool = True
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"

    # ── Polymarket CLOB ────────────────────────────────────────────────────
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_passphrase: str = ""
    polymarket_base_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_chain_id: int = 137  # Polygon mainnet

    # ── Binance (public endpoints need no auth) ───────────────────────────
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    binance_rest_url: str = "https://api.binance.com/api/v3"
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/trading.db"

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False

    # ── Capital ───────────────────────────────────────────────────────────
    paper_balance: float = 1000.0

    # ── Risk limits ───────────────────────────────────────────────────────
    max_risk_per_trade_pct: float = 0.02        # 2 % of balance per trade
    max_daily_drawdown_pct: float = 0.05        # 5 % daily loss → halt
    max_consecutive_losses: int = 3             # streak → cooldown
    cooldown_minutes: int = 60                  # cooldown duration
    max_open_positions: int = 3
    max_total_exposure_pct: float = 0.20        # 20 % of balance max deployed

    # ── Execution quality ─────────────────────────────────────────────────
    min_confidence_threshold: float = 65.0      # 0-100 score required
    min_liquidity_usdc: float = 500.0           # min order-book depth
    max_spread_pct: float = 0.08                # 8 % max bid/ask spread
    max_slippage_pct: float = 0.01              # 1 % max expected slippage
    min_time_to_resolution_hours: float = 2.0
    max_time_to_resolution_hours: float = 168.0 # 1 week

    # ── Position sizing ───────────────────────────────────────────────────
    use_kelly: bool = True
    kelly_fraction: float = 0.25               # quarter-Kelly
    min_position_usdc: float = 10.0
    max_position_usdc: float = 200.0

    # ── Strategy parameters ───────────────────────────────────────────────
    rsi_long_min: float = 50.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 50.0
    min_volume_ratio: float = 1.2              # volume vs 20-bar MA
    require_volume_confirmation: bool = True

    # ── Sentiment (Phase 6) ───────────────────────────────────────────────
    cryptopanic_api_key: str = ""
    sentiment_enabled: bool = False

    # ── Reconnect settings ────────────────────────────────────────────────
    max_reconnect_attempts: int = 10
    reconnect_base_delay: float = 1.0          # seconds, doubles each retry
    reconnect_max_delay: float = 60.0


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
