from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from openai import OpenAI
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore[assignment]

from ai_trader.config.settings import settings

ValidationDecision = Literal["approved", "rejected"]


@dataclass
class LlmValidationResult:
    validation: ValidationDecision
    confidence_adjustment: float
    reasoning: str
    source: str = "deterministic"
    fallback_used: bool = False


class _LlmStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    validation: ValidationDecision
    confidence_adjustment: float = Field(ge=-0.2, le=0.1)
    reasoning: str = Field(min_length=1, max_length=500)


class LlmValidatorAgent:
    RESPONSE_SCHEMA: dict[str, Any] = {
        "name": "llm_signal_validation",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "validation": {
                    "type": "string",
                    "enum": ["approved", "rejected"],
                },
                "confidence_adjustment": {
                    "type": "number",
                    "minimum": -0.2,
                    "maximum": 0.1,
                },
                "reasoning": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                },
            },
            "required": ["validation", "confidence_adjustment", "reasoning"],
        },
        "strict": True,
    }

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        validation_enabled: Optional[bool] = None,
        client: Any | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.validation_enabled = (
            settings.llm_validation_enabled if validation_enabled is None else validation_enabled
        )
        self.timeout_seconds = timeout_seconds
        self.client = client
        if self.client is None and self.validation_enabled and self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key, timeout=timeout_seconds)

    @staticmethod
    def _deterministic_passthrough(reason: str, *, fallback_used: bool = False) -> LlmValidationResult:
        return LlmValidationResult(
            validation="approved",
            confidence_adjustment=0.0,
            reasoning=reason,
            source="deterministic_fallback",
            fallback_used=fallback_used,
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        raise ValueError("LLM response did not include structured text content.")

    @classmethod
    def _normalize_model_response(cls, raw_text: str) -> LlmValidationResult:
        parsed = _LlmStructuredResponse.model_validate_json(raw_text)
        return LlmValidationResult(
            validation=parsed.validation,
            confidence_adjustment=float(parsed.confidence_adjustment),
            reasoning=parsed.reasoning.strip(),
            source="openai",
            fallback_used=False,
        )

    def _build_instructions(self) -> str:
        return (
            "You are validating an already-generated trade from a deterministic signal engine. "
            "You must never invent a new trade, change the signal side, or modify entry, stop loss, or target. "
            "You may only return approved or rejected, plus a small confidence adjustment and concise reasoning. "
            "Reject weak or inconsistent setups. If the deterministic payload is incomplete, reject it."
        )

    def _call_openai(self, payload: dict[str, Any]) -> LlmValidationResult:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured.")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_completion_tokens=180,
            response_format={
                "type": "json_schema",
                "json_schema": self.RESPONSE_SCHEMA,
            },
            messages=[
                {"role": "system", "content": self._build_instructions()},
                {
                    "role": "user",
                    "content": json.dumps(payload, default=str),
                },
            ],
        )
        raw_text = self._extract_response_text(response)
        return self._normalize_model_response(raw_text)

    def validate(self, payload: dict) -> LlmValidationResult:
        signal = payload.get("signal")
        if signal in (None, "NONE"):
            return LlmValidationResult(
                "rejected",
                0.0,
                "No deterministic trade to validate.",
                source="deterministic",
                fallback_used=False,
            )

        if not self.validation_enabled:
            result = self._deterministic_passthrough(
                "LLM validation disabled; using deterministic signal.",
                fallback_used=False,
            )
            logger.info(f"LlmValidatorAgent validation: {result}")
            return result

        if not self.api_key or self.client is None:
            result = self._deterministic_passthrough(
                "LLM client unavailable; using deterministic signal.",
                fallback_used=True,
            )
            logger.warning(f"LlmValidatorAgent validation fallback: {result}")
            return result

        try:
            result = self._call_openai(payload)
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            result = self._deterministic_passthrough(
                f"LLM response invalid; using deterministic signal. {exc}",
                fallback_used=True,
            )
            logger.error(f"LlmValidatorAgent invalid response fallback: {exc}")
        except Exception as exc:  # noqa: BLE001
            result = self._deterministic_passthrough(
                f"LLM unavailable; using deterministic signal. {exc}",
                fallback_used=True,
            )
            logger.error(f"LlmValidatorAgent API failure fallback: {exc}")

        logger.info(f"LlmValidatorAgent validation: {result}")
        return result
