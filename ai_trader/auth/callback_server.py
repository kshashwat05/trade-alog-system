from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import argparse
from threading import Event, Thread
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from loguru import logger
import uvicorn


@dataclass
class CallbackCapture:
    request_token: str | None = None
    error: str | None = None
    received_at: str | None = None


def create_callback_app(capture: CallbackCapture, event: Event, callback_path: str = "/callback") -> FastAPI:
    app = FastAPI(title="Kite Callback Server")

    @app.get(callback_path, response_class=HTMLResponse)
    def callback(
        request_token: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
    ) -> HTMLResponse:
        capture.received_at = datetime.utcnow().isoformat()
        if request_token is None:
            capture.error = f"Missing request_token. status={status!r} action={action!r}"
            event.set()
            return HTMLResponse(
                "<html><body><h2>Kite login failed</h2><p>Missing request token.</p></body></html>",
                status_code=400,
            )
        capture.request_token = request_token
        event.set()
        return HTMLResponse(
            "<html><body><h2>Kite login successful</h2><p>You can return to the terminal.</p></body></html>"
        )

    return app


class RequestTokenCallbackServer:
    def __init__(self, redirect_url: str) -> None:
        parsed = urlparse(redirect_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("redirect_url must be an http(s) URL.")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.callback_path = parsed.path or "/callback"
        self.capture = CallbackCapture()
        self._event = Event()
        self._server: uvicorn.Server | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        app = create_callback_app(self.capture, self._event, self.callback_path)
        config = uvicorn.Config(app=app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config=config)
        self._thread = Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(f"Started Kite callback listener at http://{self.host}:{self.port}{self.callback_path}.")

    def wait_for_request_token(self, timeout_seconds: int) -> str:
        if not self._event.wait(timeout_seconds):
            raise TimeoutError("Timed out waiting for Zerodha callback.")
        if self.capture.request_token is None:
            raise RuntimeError(self.capture.error or "Kite callback failed without request_token.")
        return self.capture.request_token

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Stopped Kite callback listener.")
        self._server = None
        self._thread = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local Zerodha callback listener.")
    parser.add_argument("--redirect-url", required=True, help="Registered redirect URL, e.g. http://localhost:8000/callback")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for the callback.")
    args = parser.parse_args()

    server = RequestTokenCallbackServer(args.redirect_url)
    server.start()
    try:
        server.wait_for_request_token(args.timeout)
        print("Received request token successfully.")
    finally:
        server.stop()
