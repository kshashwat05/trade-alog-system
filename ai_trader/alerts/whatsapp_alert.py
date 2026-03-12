from __future__ import annotations

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

    def send_trade_signal(self, signal: TradeSignal) -> None:
        if self.client is None:
            logger.info(f"WhatsApp alert (mock): {signal}")
            return

        if signal.signal == "NONE":
            logger.info("No trade signal, not sending WhatsApp alert.")
            return

        body = (
            "NIFTY TRADE SIGNAL\n\n"
            f"{signal.signal.replace('_', ' ')}\n\n"
            f"Entry: {signal.entry:.2f}\n"
            f"Stop Loss: {signal.stop_loss:.2f}\n"
            f"Target: {signal.target:.2f}\n\n"
            f"Confidence: {signal.confidence * 100:.0f}%\n"
            f"Reason: {signal.rationale}"
        )

        try:
            self.client.messages.create(
                body=body,
                from_=settings.twilio_whatsapp_from,
                to=settings.whatsapp_to,
            )
            logger.info("WhatsApp alert sent successfully.")
        except TwilioRestException as exc:
            logger.error(f"Failed to send WhatsApp alert: {exc}")
