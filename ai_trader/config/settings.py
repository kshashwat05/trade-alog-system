from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Trading instruments
    index_symbol: str = Field("NIFTY 50", description="Underlying index symbol")
    option_symbol: str = Field("NIFTY", description="Base symbol for NIFTY options")

    # Risk parameters
    max_daily_loss: float = 20000.0
    max_trades_per_day: int = 3
    max_position_lots: int = 2
    max_open_trades: int = 1
    signal_cooldown_minutes: int = 15
    position_monitor_interval_seconds: int = 20
    orchestrator_min_score: int = 7

    # Market hours (IST)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30
    market_timezone: str = "Asia/Kolkata"

    # External APIs
    kite_api_key: Optional[str] = Field(default=None, repr=False)
    kite_api_secret: Optional[str] = Field(default=None, repr=False)
    kite_access_token: Optional[str] = Field(default=None, repr=False)
    kite_last_login: Optional[str] = None
    kite_redirect_url: str = "http://localhost:8000/callback"
    kite_auth_timeout_seconds: int = 300
    kite_instrument_token: Optional[int] = None

    news_api_key: Optional[str] = Field(default=None, repr=False)
    newsdata_api_key: Optional[str] = Field(default=None, repr=False)
    marketaux_api_key: Optional[str] = Field(default=None, repr=False)
    financial_modeling_prep_api_key: Optional[str] = Field(default=None, repr=False)
    alpha_vantage_api_key: Optional[str] = Field(default=None, repr=False)
    tradingeconomics_api_key: Optional[str] = Field(default=None, repr=False)
    openai_api_key: Optional[str] = Field(default=None, repr=False)
    gemini_api_key: Optional[str] = Field(default=None, repr=False)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    llm_validation_enabled: bool = False
    market_data_cache_seconds: int = 20
    news_feed_cache_seconds: int = 120
    global_market_cache_seconds: int = 120
    macro_calendar_cache_seconds: int = 300
    max_price_candle_age_seconds: int = 900
    max_option_chain_age_seconds: int = 180
    max_vix_age_seconds: int = 300
    max_news_age_seconds: int = 1800

    twilio_account_sid: Optional[str] = Field(default=None, repr=False)
    twilio_auth_token: Optional[str] = Field(default=None, repr=False)
    twilio_whatsapp_from: Optional[str] = Field(default=None, repr=False)
    whatsapp_to: Optional[str] = Field(default=None, repr=False)
    twilio_content_sid: Optional[str] = Field(default=None, repr=False)
    twilio_trade_content_sid: Optional[str] = Field(default=None, repr=False)

    # Backtesting
    backtest_data_path: str = "ai_trader/tests/mock_data/nifty_intraday.csv"
    trade_journal_path: str = "ai_trader/data/trade_journal.db"
    live_state_path: str = "ai_trader/data/live_state.json"
    runtime_health_path: str = "ai_trader/data/runtime_health.json"
    replay_reports_path: str = "ai_trader/data/replay_reports"
    strict_startup_checks: bool = False

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
