"""
agent/harmony.py — Harmony token-based streaming wrapper for gpt-oss / OSS-120B.

Architecture:
  1. Encode conversation messages into harmony tokens via openai_harmony
  2. Call vLLM /v1/completions with prompt_token_ids (NOT chat/completions)
  3. Stream token IDs back, buffer them, and parse harmony messages
  4. Route by channel: analysis→thinking, final→text, tool call→tool_call event

This is the correct approach for gpt-oss models. Using /v1/chat/completions
causes "Unexpected token" errors because the model expects harmony-formatted input.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

# ── Imports with graceful fallback ─────────────────────────────────────────────

try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Conversation,
        Message,
        Role,
        Author,
        TextContent,
        DeveloperContent,
        SystemContent,
    )
    _HARMONY_AVAILABLE = True
except ImportError:
    _HARMONY_AVAILABLE = False
    # Minimal stubs so file can import without openai_harmony installed
    class Role:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        DEVELOPER = "developer"

    class Author:
        def __init__(self, role): self.role = role

    class TextContent:
        def __init__(self, text): self.text = text

    class Message:
        def __init__(self, author, content, channel=None, recipient=None):
            self.author = author
            self.content = content
            self.channel = channel
            self.recipient = recipient
        def with_recipient(self, r): self.recipient = r; return self

    class Conversation:
        def __init__(self, messages=None): self.messages = messages or []


# ── Event types ────────────────────────────────────────────────────────────────

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


# ── Message builders ────────────────────────────────────────────────────────────

def user_message(text: str) -> Message:
    if not _HARMONY_AVAILABLE:
        return Message(Author(Role.USER), [TextContent(text)], recipient="assistant")
    return Message(
        author=Author(role=Role.USER),
        content=[TextContent(text=text)],
    ).with_recipient("assistant")


def system_message(text: str) -> Message:
    if not _HARMONY_AVAILABLE:
        return Message(Author(Role.SYSTEM), [TextContent(text)], recipient="assistant")
    return Message(
        author=Author(role=Role.SYSTEM),
        content=[SystemContent(model_identity=text)],
    ).with_recipient("assistant")


def assistant_message(text: str, channel: str = "final") -> Message:
    if not _HARMONY_AVAILABLE:
        return Message(Author(Role.ASSISTANT), [TextContent(text)], channel=channel, recipient="assistant")
    return Message(
        author=Author(role=Role.ASSISTANT),
        content=[TextContent(text=text)],
        channel=channel,
    ).with_recipient("assistant")


# ── OpenAI async client ────────────────────────────────────────────────────────

def _get_openai_client():
    """Create AsyncOpenAI client pointed at the vLLM /v1 endpoint."""
    from openai import AsyncOpenAI
    base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("HARMONY_API_KEY", "EMPTY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


# ── Harmony encoding ───────────────────────────────────────────────────────────

_enc_cache = None

def _get_encoding():
    """Load the harmony encoding (cached)."""
    global _enc_cache
    if _enc_cache is None:
        _enc_cache = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _enc_cache


def _encode_messages(messages: list[Message]) -> list[int]:
    """Render harmony messages to token IDs for the completions endpoint."""
    enc = _get_encoding()
    conv = Conversation(messages=list(messages))
    # render_conversation_for_completion appends the assistant header to trigger generation
    token_ids = enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    return token_ids


def _parse_output_tokens(token_ids: list[int]) -> list[Message]:
    """Parse harmony output token IDs back to Message objects."""
    enc = _get_encoding()
    return enc.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)


def _extract_text(msg: Message) -> str:
    """Extract all text content from a message."""
    parts = []
    if hasattr(msg, "content") and msg.content:
        for c in msg.content:
            if hasattr(c, "text") and c.text:
                parts.append(c.text)
    return "".join(parts)


# ── Streaming completion ───────────────────────────────────────────────────────

async def stream_completion(
    messages: list[Message],
    model: str,
    tool_config: list | None = None,
    event_queue: asyncio.Queue | None = None,
) -> AsyncIterator[Message]:
    """
    Stream a completion from the gpt-oss model using harmony token format.

    Flow:
      1. Encode messages → token IDs via openai_harmony
      2. Call vLLM /v1/completions with prompt_token_ids
      3. Accumulate streamed tokens
      4. Parse harmony messages from output tokens
      5. Emit AgentEvents per channel: analysis→thinking, final→text, recipient→tool_call
      6. Yield completed assistant messages for history
    """

    async def _emit(event: AgentEvent):
        if event_queue:
            await event_queue.put(event)

    await _emit(AgentEvent(type="status", content="Calling model..."))

    # ── Check harmony availability ──
    if not _HARMONY_AVAILABLE:
        await _emit(AgentEvent(type="error", content="openai_harmony not installed. Install with: pip install openai-harmony"))
        return

    try:
        client = _get_openai_client()

        # 1. Encode messages to token IDs
        prompt_token_ids = _encode_messages(messages)

        # 2. Call vLLM /v1/completions with token IDs
        stream = await client.completions.create(
            model=model,
            prompt=prompt_token_ids,   # vLLM accepts list of token IDs as prompt
            stream=True,
            max_tokens=4096,
            stop=None,                 # harmony uses its own stop tokens within the model
            temperature=1.0,
        )

        # 3. Accumulate tokens from stream
        token_buffer: list[int] = []
        full_text_for_events = ""
        thinking_buffer = ""
        final_buffer = ""

        async for chunk in stream:
            choice = chunk.choices[0]

            # vLLM returns logprobs with token IDs when using completions endpoint
            if hasattr(choice, "logprobs") and choice.logprobs and hasattr(choice.logprobs, "token_ids"):
                new_ids = choice.logprobs.token_ids
                token_buffer.extend(new_ids)

            # Also track text for streaming live feedback
            if choice.text:
                # Stream the raw text as-is for immediate feedback while tokens accumulate
                # We'll parse properly at the end, but this gives live streaming feel
                raw_chunk = choice.text
                # Strip harmony special tokens from display text
                display = _strip_harmony_tokens(raw_chunk)
                if display:
                    full_text_for_events += display
                    await _emit(AgentEvent(type="text", content=display))

            if choice.finish_reason:
                break

        # 4. If we have token IDs, parse harmony messages for proper channel routing
        if token_buffer:
            try:
                parsed_msgs = _parse_output_tokens(token_buffer)
                # Re-emit as proper typed events
                # First clear previous text events by emitting a "done" marker
                for pmsg in parsed_msgs:
                    channel = getattr(pmsg, "channel", "final")
                    recipient = getattr(pmsg, "recipient", "assistant")
                    text = _extract_text(pmsg)

                    if channel == "analysis":
                        # Chain-of-thought thinking
                        if text:
                            await _emit(AgentEvent(type="thinking", content=text))
                    elif channel == "final":
                        # This is the actual response — yield for history
                        if text:
                            yield pmsg
                    elif recipient and recipient not in ("assistant", None):
                        # Tool call — recipient is the tool name
                        await _emit(AgentEvent(
                            type="tool_call",
                            tool_name=str(recipient),
                            content=text,
                        ))
            except Exception as parse_err:
                # Fallback: we already streamed text events above, just yield assistant msg
                if full_text_for_events:
                    yield assistant_message(full_text_for_events)
        elif full_text_for_events:
            # Fallback: no token IDs in stream, use accumulated text
            yield assistant_message(full_text_for_events)

    except Exception as e:
        err_msg = f"Model error: {e}"
        await _emit(AgentEvent(type="error", content=err_msg))
        raise


def _strip_harmony_tokens(text: str) -> str:
    """Remove harmony special tokens from display text."""
    import re
    # Remove harmony special tokens like <|start|>, <|end|>, <|channel|>analysis, etc.
    cleaned = re.sub(r'<\|[a-z_]+\|>', '', text)
    # Remove channel names that appear after <|channel|>
    cleaned = re.sub(r'\b(analysis|final|commentary)\b\s*', '', cleaned)
    return cleaned.strip()


# ── Tool result emitter ───────────────────────────────────────────────────────

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
