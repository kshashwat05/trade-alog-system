from __future__ import annotations

from datetime import datetime
from threading import Event
import types
from zoneinfo import ZoneInfo

import pytest

from ai_trader.auth import token_manager as token_manager_module
from ai_trader.auth.callback_server import CallbackCapture, create_callback_app
from ai_trader.auth.token_manager import TokenManager
from ai_trader.config.settings import settings
from ai_trader.macos_launchd import build_launchd_plist


def test_callback_server_captures_request_token():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    capture = CallbackCapture()
    event = Event()
    app = create_callback_app(capture, event)
    client = TestClient(app)
    response = client.get("/callback?request_token=test-token")
    assert response.status_code == 200
    assert capture.request_token == "test-token"
    assert event.is_set() is True


def test_callback_server_handles_missing_request_token():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    capture = CallbackCapture()
    event = Event()
    app = create_callback_app(capture, event)
    client = TestClient(app)
    response = client.get("/callback")
    assert response.status_code == 400
    assert capture.error is not None


def test_token_manager_exchanges_request_token_and_updates_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "KITE_API_KEY=test-key",
                "KITE_API_SECRET=test-secret",
                "KITE_ACCESS_TOKEN=",
                "KITE_LAST_LOGIN=",
                "KITE_REDIRECT_URL=http://localhost:8000/callback",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(settings, "kite_api_key", "test-key")
    monkeypatch.setattr(settings, "kite_api_secret", "test-secret")
    monkeypatch.setattr(settings, "kite_access_token", None)
    monkeypatch.setattr(settings, "kite_last_login", None)

    class DummyKite:
        def __init__(self, api_key):
            self.api_key = api_key
            self.access_token = None

        def set_access_token(self, token):
            self.access_token = token

        def login_url(self):
            return "http://kite/login"

        def generate_session(self, request_token, api_secret):
            assert request_token == "request-token"
            assert api_secret == "test-secret"
            return {"access_token": "access-token"}

        def profile(self):
            return {"user_id": "AB1234"}

    now = datetime(2026, 3, 13, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    manager = TokenManager(env_path=env_path, now_provider=lambda: now, kite_factory=DummyKite)
    kite = manager.exchange_request_token("request-token")
    assert kite.access_token == "access-token"
    env_text = env_path.read_text()
    assert "KITE_ACCESS_TOKEN=access-token" in env_text
    assert "KITE_LAST_LOGIN=2026-03-13T09:00:00+05:30" in env_text


def test_token_manager_validates_current_day_session(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "KITE_API_KEY=test-key",
                "KITE_API_SECRET=test-secret",
                "KITE_ACCESS_TOKEN=access-token",
                "KITE_LAST_LOGIN=2026-03-13T09:00:00+05:30",
                "KITE_REDIRECT_URL=http://localhost:8000/callback",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(settings, "kite_api_key", "test-key")
    monkeypatch.setattr(settings, "kite_api_secret", "test-secret")
    monkeypatch.setattr(settings, "kite_access_token", "access-token")
    monkeypatch.setattr(settings, "kite_last_login", "2026-03-13T09:00:00+05:30")

    class DummyKite:
        def __init__(self, api_key):
            self.api_key = api_key

        def set_access_token(self, token):
            self.access_token = token

        def profile(self):
            return {"user_id": "AB1234"}

    manager = TokenManager(
        env_path=env_path,
        now_provider=lambda: datetime(2026, 3, 13, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        kite_factory=DummyKite,
    )
    result = manager.validate_session()
    assert result.valid is True


def test_token_manager_rejects_stale_session(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "KITE_API_KEY=test-key",
                "KITE_API_SECRET=test-secret",
                "KITE_ACCESS_TOKEN=access-token",
                "KITE_LAST_LOGIN=2026-03-12T09:00:00+05:30",
                "KITE_REDIRECT_URL=http://localhost:8000/callback",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(settings, "kite_api_key", "test-key")
    monkeypatch.setattr(settings, "kite_api_secret", "test-secret")
    monkeypatch.setattr(settings, "kite_access_token", "access-token")
    monkeypatch.setattr(settings, "kite_last_login", "2026-03-12T09:00:00+05:30")

    class DummyKite:
        def __init__(self, api_key):
            self.api_key = api_key

        def set_access_token(self, token):
            self.access_token = token

        def profile(self):
            return {"user_id": "AB1234"}

    manager = TokenManager(
        env_path=env_path,
        now_provider=lambda: datetime(2026, 3, 13, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        kite_factory=DummyKite,
    )
    result = manager.validate_session()
    assert result.valid is False
    assert "current trading day" in result.reason


def test_token_manager_auto_opens_browser(monkeypatch):
    opened_urls: list[str] = []

    class DummyServer:
        def __init__(self, redirect_url):
            self.redirect_url = redirect_url

        def start(self):
            return None

        def wait_for_request_token(self, timeout_seconds):
            return "request-token"

        def stop(self):
            return None

    monkeypatch.setattr(token_manager_module, "RequestTokenCallbackServer", DummyServer)
    monkeypatch.setattr(token_manager_module.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    class DummyKite:
        def __init__(self, api_key):
            self.api_key = api_key

        def login_url(self):
            return "http://kite/login"

        def generate_session(self, request_token, api_secret):
            return {"access_token": "token"}

        def set_access_token(self, token):
            self.access_token = token

    manager = TokenManager(
        env_path=".env.example",
        kite_factory=DummyKite,
    )
    monkeypatch.setattr(settings, "kite_api_key", "test-key")
    monkeypatch.setattr(settings, "kite_api_secret", "test-secret")
    monkeypatch.setattr(settings, "kite_redirect_url", "http://localhost:8000/callback")
    manager.authenticate_interactively(open_browser=True, timeout_seconds=1)
    assert opened_urls == ["http://kite/login"]


def test_start_trading_day_authenticates_then_starts_runtime(monkeypatch):
    from ai_trader import start_trading_day

    calls = {"auth": 0, "dashboard": 0, "main": 0}

    class DummyManager:
        def get_authenticated_kite_client(self, auto_login=True, open_browser=True):
            calls["auth"] += 1
            return object()

    monkeypatch.setattr(start_trading_day, "TokenManager", lambda: DummyManager())
    monkeypatch.setattr(start_trading_day.subprocess, "Popen", lambda args: calls.__setitem__("dashboard", calls["dashboard"] + 1))
    monkeypatch.setitem(
        __import__("sys").modules,
        "ai_trader.main",
        types.SimpleNamespace(main=lambda: calls.__setitem__("main", calls["main"] + 1)),
    )
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["start_trading_day", "--with-dashboard", "--no-open-browser"],
    )
    start_trading_day.main()
    assert calls == {"auth": 1, "dashboard": 1, "main": 1}


def test_build_launchd_plist_contains_startup_command(tmp_path):
    plist = build_launchd_plist(
        label="com.ai_trader.daily",
        repo_root=tmp_path,
        python_path=tmp_path / ".venv/bin/python",
        log_path=tmp_path / "logs/start.log",
        start_hour=8,
        start_minute=55,
        with_dashboard=True,
    )
    assert "ai_trader.start_trading_day" in plist
    assert "--with-dashboard" in plist
    assert "<integer>8</integer>" in plist
