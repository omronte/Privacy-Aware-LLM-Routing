"""Minimal OpenAI-compatible chat client used by external providers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import requests


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout=60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, max_tokens=512, temperature=0.0, **kwargs):
        request_body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model.startswith("openai/gpt-oss"):
            request_body["reasoning_effort"] = "low"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Provider request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        response.raise_for_status()
        payload = response.json()
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=self._message_content(choice),
                    reasoning_content=choice.get("message", {}).get(
                        "reasoning_content"
                    ),
                    reasoning=choice.get("message", {}).get("reasoning"),
                    raw=choice,
                ),
                finish_reason=choice.get("finish_reason"),
            )
            for choice in payload.get("choices", [])
        ]
        return SimpleNamespace(choices=choices)

    @staticmethod
    def _message_content(choice):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        if content:
            return content
        return (
            choice.get("text")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )


def get_external_client(model: str):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        return None
    return OpenAICompatibleClient(api_key, base_url, model)