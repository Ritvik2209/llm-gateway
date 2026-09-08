"""Groq provider integration."""

import json
import logging
from typing import AsyncGenerator
from uuid import uuid4

import httpx

from app.config import GROQ_API_KEY
from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider


logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Provider implementation for Groq's OpenAI-compatible chat API."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        self.api_key = api_key or GROQ_API_KEY
        self.base_url = base_url.rstrip("/")
        self.last_stream_usage = {"input_tokens": 0, "output_tokens": 0}

    def _build_messages(self, request: UnifiedChatRequest) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

    def _build_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Groq chat request failed: GROQ_API_KEY is not set")

        return {"Authorization": f"Bearer {self.api_key}"}

    def _build_usage(self, usage: dict) -> dict[str, int]:
        input_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)

        if reasoning_tokens:
            logger.info(
                "Groq response included %s hidden reasoning tokens",
                reasoning_tokens,
            )

        return {
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens + reasoning_tokens,
        }

    async def chat(self, request: UnifiedChatRequest) -> UnifiedChatResponse:
        payload = {
            "model": request.model,
            "messages": self._build_messages(request),
            "stream": False,
        }

        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Groq chat request failed with "
                f"status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Groq chat request failed: {exc}") from exc

        response_data = response.json()
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message") or {}
        usage = self._build_usage(response_data.get("usage") or {})

        return UnifiedChatResponse(
            id=str(uuid4()),
            model=response_data["model"],
            content=message.get("content", ""),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason", ""),
        )

    async def chat_stream(
        self, request: UnifiedChatRequest
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": request.model,
            "messages": self._build_messages(request),
            "stream": True,
        }
        self.last_stream_usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=10.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        data = line.removeprefix("data: ")
                        if data == "[DONE]":
                            break

                        try:
                            response_data = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"Groq stream returned invalid JSON: {data}"
                            ) from exc

                        usage_data = response_data.get("usage")
                        if usage_data:
                            self.last_stream_usage = self._build_usage(usage_data)

                        choices = response_data.get("choices") or []
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or {}
                        content = delta.get("content", "")
                        if content:
                            yield content
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Groq streaming chat request failed with "
                f"status {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Groq streaming chat request failed: {exc}"
            ) from exc
