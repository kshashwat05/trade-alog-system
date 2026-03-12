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
    max_trades_per_day: int = 5
    max_position_lots: int = 2

    # Market hours (IST)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30
    market_timezone: str = "Asia/Kolkata"

    # External APIs
    kite_api_key: Optional[str] = None
    kite_api_secret: Optional[str] = None
    kite_access_token: Optional[str] = None
    kite_instrument_token: Optional[int] = None

    news_api_key: Optional[str] = None

    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_from: Optional[str] = None
    whatsapp_to: Optional[str] = None

    # Backtesting
    backtest_data_path: str = "ai_trader/tests/mock_data/nifty_intraday.csv"

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
