"""
agent/harmony.py — Streaming wrapper around openai_harmony for OSS-120B.
Emits typed events: thinking, text, tool_call, tool_result, error, done.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

# ── Event types streamed to the frontend via SSE ──────────────────────────────

EventType = Literal["thinking", "text", "tool_call", "tool_result", "error", "done", "status"]


@dataclass
class AgentEvent:
    type: EventType
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = {
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "metadata": self.metadata,
        }
        return f"data: {json.dumps(payload)}\n\n"


# ── Harmony client ─────────────────────────────────────────────────────────────

def _get_harmony_client():
    """Lazily import + create the openai_harmony client pointed at the local model."""
    try:
        from openai_harmony import Client
        base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
        api_key = os.environ.get("HARMONY_API_KEY", "kaggleclaw-local")
        return Client(base_url=base_url, api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to create harmony client: {e}") from e


# ── Conversation message builders ──────────────────────────────────────────────

def user_message(text: str) -> Message:
    return Message(
        author=Author(role=Role.USER),
        content=[TextContent(text=text)],
    ).with_recipient("assistant")


def system_message(text: str) -> Message:
    return Message(
        author=Author(role=Role.SYSTEM),
        content=[TextContent(text=text)],
    ).with_recipient("assistant")


def assistant_message(text: str) -> Message:
    return Message(
        author=Author(role=Role.ASSISTANT),
        content=[TextContent(text=text)],
    ).with_recipient("assistant")


# ── Streaming completion ───────────────────────────────────────────────────────

async def stream_completion(
    messages: list[Message],
    model: str,
    tool_config: list | None = None,
    event_queue: asyncio.Queue | None = None,
) -> AsyncIterator[Message]:
    """
    Stream a completion from the harmony model.
    Yields complete messages for tool routing.
    Pushes AgentEvents to event_queue for SSE streaming.
    """
    client = _get_harmony_client()

    async def _emit(event: AgentEvent):
        if event_queue:
            await event_queue.put(event)

    await _emit(AgentEvent(type="status", content="Calling model..."))

    try:
        # Use harmony's streaming completion
        full_text = ""
        thinking_text = ""
        in_thinking = False

        async for chunk in client.stream_completion(
            messages=messages,
            model=model,
            tool_configs=tool_config or [],
        ):
            # Handle thinking tokens
            if hasattr(chunk, "thinking") and chunk.thinking:
                thinking_text += chunk.thinking
                in_thinking = True
                await _emit(AgentEvent(type="thinking", content=chunk.thinking))

            # Handle text tokens
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
                await _emit(AgentEvent(type="text", content=chunk.text))

            # Handle tool call chunks
            if hasattr(chunk, "tool_call") and chunk.tool_call:
                tool_name = getattr(chunk.tool_call, "name", "")
                tool_args = getattr(chunk.tool_call, "arguments", "")
                await _emit(AgentEvent(
                    type="tool_call",
                    tool_name=tool_name,
                    content=tool_args,
                ))

        # Build and yield the final assistant message
        if full_text or thinking_text:
            content_text = full_text or thinking_text
            msg = assistant_message(content_text)
            yield msg

    except Exception as e:
        err_msg = f"Model error: {e}"
        await _emit(AgentEvent(type="error", content=err_msg))
        raise


async def emit_tool_result(
    tool_name: str,
    output: str,
    event_queue: asyncio.Queue | None = None,
):
    """Emit a tool result event to the SSE stream."""
    if event_queue:
        await event_queue.put(AgentEvent(
            type="tool_result",
            tool_name=tool_name,
            content=output,
        ))
