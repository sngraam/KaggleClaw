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
    """Lazily import + create the openai client pointed at the local model."""
    try:
        from openai import AsyncOpenAI
        base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
        # kaggleclaw-local or actual empty depending on vLLM, standard is just any string or EMPTY
        api_key = os.environ.get("HARMONY_API_KEY", "EMPTY")
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to create openai client for vLLM: {e}") from e


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
    Stream a completion from the harmony model using standard chat completions.
    vLLM will handle harmony formatting.
    """
    client = _get_harmony_client()

    async def _emit(event: AgentEvent):
        if event_queue:
            await event_queue.put(event)

    await _emit(AgentEvent(type="status", content="Calling model..."))

    try:
        full_text = ""
        # Need to convert Harmony Messages to standard dicts
        standard_msgs = []
        for msg in messages:
            role = str(msg.author.role.value).lower()
            if role == "developer": role = "system" # developer is system
            
            # extract text
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent) and hasattr(c, "text")]
            standard_msgs.append({"role": role, "content": "".join(text_parts)})

        stream = await client.chat.completions.create(
            model=model,
            messages=standard_msgs,
            stream=True,
            # tools=tool_config if tool_config else None  # Assuming vllm supports tools or we handle manually
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            if hasattr(delta, "content") and delta.content:
                full_text += delta.content
                await _emit(AgentEvent(type="text", content=delta.content))
                
            # If tool calls are supported via standard vllm streaming:
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tcall in delta.tool_calls:
                    if hasattr(tcall, "function") and tcall.function:
                        # stream the tool call args
                        args = tcall.function.arguments or ""
                        await _emit(AgentEvent(
                            type="tool_call",
                            tool_name=tcall.function.name if hasattr(tcall.function, "name") and tcall.function.name else "",
                            content=args
                        ))

        if full_text:
            yield assistant_message(full_text)

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
