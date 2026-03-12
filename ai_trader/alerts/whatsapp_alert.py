from __future__ import annotations

import json

from loguru import logger
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from ai_trader.config.settings import settings
from ai_trader.agents.trigger_agent import TradeSignal


class WhatsAppAlerter:
    """Sends trade alerts via Twilio WhatsApp API."""

    def __init__(self) -> None:
        if not (
            settings.twilio_account_sid
            and settings.twilio_auth_token
            and settings.twilio_whatsapp_from
            and settings.whatsapp_to
        ):
            logger.warning("Twilio settings incomplete; WhatsApp alerts will be disabled.")
            self.client: Client | None = None
        else:
            self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    @staticmethod
    def _trade_body(
        signal: TradeSignal,
        *,
        institutional_bias: str | None = None,
        gamma_regime: str | None = None,
    ) -> str:
        return (
            "NIFTY TRADE SIGNAL\n\n"
            f"{signal.signal.replace('_', ' ')}\n\n"
            f"Entry: {signal.entry:.2f}\n"
            f"Stop Loss: {signal.stop_loss:.2f}\n"
            f"Target: {signal.target:.2f}\n\n"
            f"Confidence: {signal.confidence * 100:.0f}%\n"
            f"Institutional Bias: {institutional_bias or 'neutral'}\n"
            f"Gamma Regime: {gamma_regime or 'positive_gamma'}\n\n"
            f"Reason: {signal.rationale}"
        )

    @staticmethod
    def _trade_content_variables(
        signal: TradeSignal,
        *,
        institutional_bias: str | None = None,
        gamma_regime: str | None = None,
    ) -> str:
        payload = {
            "1": signal.signal.replace("_", " "),
            "2": f"{signal.entry:.2f}",
            "3": f"{signal.stop_loss:.2f}",
            "4": f"{signal.target:.2f}",
            "5": f"{signal.confidence * 100:.0f}%",
            "6": institutional_bias or "neutral",
            "7": gamma_regime or "positive_gamma",
            "8": signal.rationale[:180],
        }
        return json.dumps(payload)

    def send_template_message(
        self,
        *,
        content_sid: str | None = None,
        content_variables: dict[str, str] | None = None,
        to: str | None = None,
    ) -> str | None:
        if self.client is None:
            logger.info("WhatsApp template alert skipped because Twilio client is unavailable.")
            return None
        sid = content_sid or settings.twilio_content_sid
        if not sid:
            raise ValueError("content_sid is required for template messages.")
        try:
            message = self.client.messages.create(
                from_=settings.twilio_whatsapp_from,
                to=to or settings.whatsapp_to,
                content_sid=sid,
                content_variables=json.dumps(content_variables or {}),
            )
            logger.info("WhatsApp template alert sent successfully.")
            return getattr(message, "sid", None)
        except TwilioRestException as exc:
            logger.error(f"Failed to send WhatsApp template alert: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected WhatsApp template alert failure: {exc}")
            raise

    def send_trade_signal(
        self,
        signal: TradeSignal,
        *,
        institutional_bias: str | None = None,
        gamma_regime: str | None = None,
    ) -> None:
        if self.client is None:
            logger.info(f"WhatsApp alert (mock): {signal}")
            return

        if signal.signal == "NONE":
            logger.info("No trade signal, not sending WhatsApp alert.")
            return

        try:
            if settings.twilio_trade_content_sid:
                self.client.messages.create(
                    from_=settings.twilio_whatsapp_from,
                    to=settings.whatsapp_to,
                    content_sid=settings.twilio_trade_content_sid,
                    content_variables=self._trade_content_variables(
                        signal,
                        institutional_bias=institutional_bias,
                        gamma_regime=gamma_regime,
                    ),
                )
            else:
                self.client.messages.create(
                    body=self._trade_body(
                        signal,
                        institutional_bias=institutional_bias,
                        gamma_regime=gamma_regime,
                    ),
                    from_=settings.twilio_whatsapp_from,
                    to=settings.whatsapp_to,
                )
            logger.info("WhatsApp alert sent successfully.")
        except TwilioRestException as exc:
            logger.error(f"Failed to send WhatsApp alert: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected WhatsApp alert failure: {exc}")

    def send_exit_alert(self, message: str) -> None:
        if self.client is None:
            logger.info(f"WhatsApp exit alert (mock): {message}")
            return

        try:
            self.client.messages.create(
                body=message,
                from_=settings.twilio_whatsapp_from,
                to=settings.whatsapp_to,
            )
            logger.info("WhatsApp exit alert sent successfully.")
        except TwilioRestException as exc:
            logger.error(f"Failed to send WhatsApp exit alert: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected WhatsApp exit alert failure: {exc}")
