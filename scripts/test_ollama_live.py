"""Live smoke test for the Ollama provider.

Run from the repository root:
    python scripts/test_ollama_live.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

__test__ = False

from app.models.schemas import ChatMessage, UnifiedChatRequest
from app.providers.ollama_provider import OllamaProvider


async def main() -> None:
    request = UnifiedChatRequest(
        model="llama3.2",
        messages=[
            ChatMessage(
                role="user",
                content="What is the Prime Minsiter of India ? Answer in one sentence.",
            )
        ],
    )

    provider = OllamaProvider()
    response = await provider.chat(request)

    print("UnifiedChatResponse:")
    print(f"id: {response.id}")
    print(f"model: {response.model}")
    print(f"content: {response.content}")
    print(f"input_tokens: {response.input_tokens}")
    print(f"output_tokens: {response.output_tokens}")
    print(f"provider: {response.provider}")
    print(f"finish_reason: {response.finish_reason}")


if __name__ == "__main__":
    asyncio.run(main())
