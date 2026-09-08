"""Pydantic request and response model definitions for the gateway."""

from typing import Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class UnifiedChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


class UnifiedChatResponse(BaseModel):
    id: str
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    provider: str
    finish_reason: str
