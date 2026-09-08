"""Ollama provider integration."""

import json
import os
from typing import AsyncGenerator
from uuid import uuid4

import httpx

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Provider implementation for Ollama's local chat API."""

    provider_name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")
        self.last_stream_usage = {"input_tokens": 0, "output_tokens": 0}

    def _build_messages(self, request: UnifiedChatRequest) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

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
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Ollama chat request failed with "
                f"status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

        response_data = response.json()
        message = response_data.get("message") or {}

        return UnifiedChatResponse(
            id=str(uuid4()),
            model=response_data["model"],
            content=message.get("content", ""),
            input_tokens=response_data.get("prompt_eval_count", 0),
            output_tokens=response_data.get("eval_count", 0),
            provider=self.provider_name,
            finish_reason=response_data.get("done_reason", ""),
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
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            response_data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"Ollama stream returned invalid JSON: {line}"
                            ) from exc

                        message = response_data.get("message") or {}
                        content = message.get("content", "")
                        if content:
                            yield content

                        if response_data.get("done") is True:
                            self.last_stream_usage = {
                                "input_tokens": response_data.get(
                                    "prompt_eval_count",
                                    0,
                                ),
                                "output_tokens": response_data.get("eval_count", 0),
                            }
                            break
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Ollama streaming chat request failed with "
                f"status {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Ollama streaming chat request failed: {exc}"
            ) from exc
