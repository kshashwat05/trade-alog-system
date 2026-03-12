from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Callable, Any
from urllib.parse import urlparse
import webbrowser
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from kiteconnect import KiteConnect
from loguru import logger

from ai_trader.auth.callback_server import RequestTokenCallbackServer
from ai_trader.config.settings import settings


@dataclass
class SessionValidationResult:
    valid: bool
    reason: str


class TokenManager:
    def __init__(
        self,
        *,
        env_path: str | Path = ".env",
        now_provider: Callable[[], datetime] | None = None,
        kite_factory: Callable[..., KiteConnect] = KiteConnect,
    ) -> None:
        self.env_path = Path(env_path)
        self.now_provider = now_provider or (lambda: datetime.now(ZoneInfo(settings.market_timezone)))
        self.kite_factory = kite_factory
        load_dotenv(self.env_path, override=True)

    @property
    def api_key(self) -> str | None:
        return settings.kite_api_key or os.getenv("KITE_API_KEY")

    @property
    def api_secret(self) -> str | None:
        return settings.kite_api_secret or os.getenv("KITE_API_SECRET")

    @property
    def access_token(self) -> str | None:
        return settings.kite_access_token or os.getenv("KITE_ACCESS_TOKEN")

    @property
    def last_login(self) -> str | None:
        return settings.kite_last_login or os.getenv("KITE_LAST_LOGIN")

    @property
    def redirect_url(self) -> str:
        return os.getenv("KITE_REDIRECT_URL") or settings.kite_redirect_url or "http://localhost:8000/callback"

    def _build_kite(self, access_token: str | None = None) -> KiteConnect:
        if not self.api_key:
            raise RuntimeError("KITE_API_KEY is not configured.")
        kite = self.kite_factory(api_key=self.api_key)
        if access_token:
            kite.set_access_token(access_token)
        return kite

    def get_login_url(self) -> str:
        if not self.api_key:
            raise RuntimeError("KITE_API_KEY is not configured.")
        return self._build_kite().login_url()

    def _update_env(self, updates: dict[str, str]) -> None:
        existing: dict[str, str] = {}
        if self.env_path.exists():
            for line in self.env_path.read_text().splitlines():
                if "=" not in line or line.strip().startswith("#"):
                    continue
                key, value = line.split("=", 1)
                existing[key] = value
        existing.update(updates)
        lines = [f"{key}={value}" for key, value in existing.items()]
        self.env_path.write_text("\n".join(lines) + "\n")
        for key, value in updates.items():
            os.environ[key] = value
        if "KITE_ACCESS_TOKEN" in updates:
            settings.kite_access_token = updates["KITE_ACCESS_TOKEN"]
        if "KITE_LAST_LOGIN" in updates:
            settings.kite_last_login = updates["KITE_LAST_LOGIN"]

    def exchange_request_token(self, request_token: str) -> KiteConnect:
        if not request_token:
            raise ValueError("request_token is required.")
        if not self.api_secret:
            raise RuntimeError("KITE_API_SECRET is not configured.")
        kite = self._build_kite()
        logger.info("Exchanging Kite request token for access token.")
        session = kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = session.get("access_token")
        if not access_token:
            raise RuntimeError("Kite session response did not include access_token.")
        login_time = self.now_provider().isoformat()
        self._update_env(
            {
                "KITE_ACCESS_TOKEN": str(access_token),
                "KITE_LAST_LOGIN": login_time,
            }
        )
        kite.set_access_token(str(access_token))
        logger.info("Kite access token stored successfully for current trading session.")
        return kite

    def validate_session(self) -> SessionValidationResult:
        access_token = self.access_token
        last_login = self.last_login
        if not access_token:
            return SessionValidationResult(False, "KITE_ACCESS_TOKEN is missing.")
        if not last_login:
            return SessionValidationResult(False, "KITE_LAST_LOGIN is missing.")
        try:
            login_dt = datetime.fromisoformat(last_login)
        except ValueError:
            return SessionValidationResult(False, "KITE_LAST_LOGIN is not a valid ISO timestamp.")
        market_tz = ZoneInfo(settings.market_timezone)
        if login_dt.tzinfo is None:
            login_dt = login_dt.replace(tzinfo=market_tz)
        now = self.now_provider()
        if now.tzinfo is None:
            now = now.replace(tzinfo=market_tz)
        if login_dt.astimezone(market_tz).date() != now.astimezone(market_tz).date():
            return SessionValidationResult(False, "Kite access token is not from the current trading day.")

        try:
            kite = self._build_kite(access_token)
            kite.profile()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Kite session validation failed: {exc}")
            return SessionValidationResult(False, "Kite access token is invalid or expired.")
        return SessionValidationResult(True, "Kite session is valid.")

    def authenticate_interactively(
        self,
        timeout_seconds: int | None = None,
        *,
        open_browser: bool = True,
    ) -> KiteConnect:
        parsed = urlparse(self.redirect_url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise RuntimeError("Only localhost redirect URLs are supported for local auth flow.")
        callback_server = RequestTokenCallbackServer(self.redirect_url)
        login_url = self.get_login_url()
        timeout = timeout_seconds or settings.kite_auth_timeout_seconds
        print("\nOpen this Zerodha login URL in your browser:\n")
        print(login_url)
        print(f"\nWaiting for callback on {self.redirect_url} ...\n")
        callback_server.start()
        if open_browser:
            opened = webbrowser.open(login_url)
            logger.info(f"Attempted to open Zerodha login URL in browser. opened={opened}")
        try:
            request_token = callback_server.wait_for_request_token(timeout)
        finally:
            callback_server.stop()
        return self.exchange_request_token(request_token)

    def get_authenticated_kite_client(
        self,
        *,
        auto_login: bool = True,
        open_browser: bool = True,
    ) -> KiteConnect:
        validation = self.validate_session()
        if validation.valid:
            logger.info("Using existing valid Kite access token.")
            return self._build_kite(self.access_token)
        logger.warning(f"Kite session invalid: {validation.reason}")
        if not auto_login:
            raise RuntimeError(validation.reason)
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Kite API credentials are incomplete. Configure KITE_API_KEY and KITE_API_SECRET.")
        return self.authenticate_interactively(open_browser=open_browser)


def validate_session() -> SessionValidationResult:
    return TokenManager().validate_session()


def get_authenticated_kite_client(*, auto_login: bool = True, open_browser: bool = True) -> KiteConnect:
    return TokenManager().get_authenticated_kite_client(auto_login=auto_login, open_browser=open_browser)
