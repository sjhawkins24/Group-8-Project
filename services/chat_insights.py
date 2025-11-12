"""Wrapper around the OpenRouter API for narrative summaries."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

try:
    from openai import OpenAI  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ChatSettings:
    model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    temperature: float = 0.4
    max_tokens: int = 350


class ChatInsightGenerator:
    """Formats model output into user-friendly narratives via OpenRouter."""

    def __init__(self, settings: ChatSettings | None = None) -> None:
        self.settings = settings or ChatSettings()

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Populate the environment variable to enable chat insights.")

        if OpenAI is None:
            raise RuntimeError("The 'openai' package is not installed. Add it to requirements.txt and reinstall dependencies.")

        referer = os.getenv("OPENROUTER_REFERER")
        app_title = os.getenv("OPENROUTER_TITLE")

        default_headers = {}
        if referer:
            default_headers["HTTP-Referer"] = referer
        if app_title:
            default_headers["X-Title"] = app_title

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers=default_headers or None,
        )

    def explain_prediction(self, prediction: Dict[str, object]) -> str:
        """Return a concise natural-language summary for a prediction payload."""
        system_prompt = (
            "You are an analyst explaining college football ranking movements to casual fans. "
            "Keep the answer to two short paragraphs. Highlight what the model expects to happen, "
            "why the Elo/feature inputs matter, and provide one actionable insight for the team."
        )

        user_prompt = (
            f"Prediction details:\n{prediction}\n\n"
            "Explain the likely ranking movement and provide one actionable insight."
        )

        response = self.client.chat.completions.create(
            model=self.settings.model,
            messages=self._format_messages(system_prompt, user_prompt),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

        message = response.choices[0].message
        content = None
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

        return (content or "").strip()

    @staticmethod
    def _format_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
