"""
agent/harmony.py — Simple, correct Harmony streaming wrapper for OSS-120B.

Architecture:
  1. Build standard chat messages (system/user/assistant dicts)
  2. Call vLLM /v1/chat/completions — the model outputs harmony tokens in text
  3. Accumulate the full streamed text
  4. Parse harmony special tokens to route by channel:
       <|channel|>analysis  → thinking event
       <|channel|>final     → text event (user-facing answer)
       <|call|>tool_name    → tool_call event (stop token, model wants to call a tool)

This avoids broken token-ID plumbing and uses the standard OpenAI-compatible
chat endpoint that vLLM serves out of the box.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal


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
# Messages are plain dicts compatible with OpenAI chat API.

def system_message(text: str) -> dict:
    return {"role": "system", "content": text}

def user_message(text: str) -> dict:
    return {"role": "user", "content": text}

def assistant_message(text: str) -> dict:
    return {"role": "assistant", "content": text}


# ── OpenAI async client ────────────────────────────────────────────────────────

def _get_openai_client():
    """Create AsyncOpenAI client pointed at the vLLM /v1 endpoint."""
    from openai import AsyncOpenAI
    base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("HARMONY_API_KEY", "EMPTY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


# ── Harmony token parsing ──────────────────────────────────────────────────────
#
# Model output format (from harmony docs):
#
#   <|start|>assistant<|channel|>analysis<|message|>...thinking...<|end|>
#   <|start|>assistant<|channel|>final<|message|>...answer...<|return|>
#   <|start|>assistant<|channel|>final<|message|>...partial...<|call|>python
#
# Special tokens:
#   <|start|>    200006  — beginning of a message
#   <|end|>      200007  — end of a message
#   <|message|>  200008  — header → content transition
#   <|channel|>  200005  — transition to channel info
#   <|return|>   200002  — stop token: model done
#   <|call|>     200012  — stop token: model wants to call a tool
#   <|constrain|> 200003 — tool call data type definition

HARMONY_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def parse_harmony_output(raw: str) -> list[dict]:
    """
    Parse a full harmony model output into a list of structured segments.

    Returns a list of dicts:
        {"channel": "analysis"|"final", "content": str, "tool": str|None}

    where `tool` is set when the segment ends with <|call|>tool_name.
    """
    segments = []

    # Split on <|start|> to get individual message blocks
    blocks = re.split(r"<\|start\|>", raw)

    for block in blocks:
        if not block.strip():
            continue

        # Extract channel
        channel_match = re.search(r"<\|channel\|>(\w+)", block)
        channel = channel_match.group(1) if channel_match else "final"

        # Extract content after <|message|>
        msg_match = re.search(r"<\|message\|>(.*?)(?=<\|(?:end|return|call)\|>|$)", block, re.DOTALL)
        content = msg_match.group(1).strip() if msg_match else ""

        # Extract tool name if present (stop by <|call|>tool_name)
        tool_match = re.search(r"<\|call\|>(\w+)", block)
        tool = tool_match.group(1) if tool_match else None

        if content or tool:
            segments.append({
                "channel": channel,
                "content": content,
                "tool": tool,
            })

    return segments


def _clean_for_display(text: str) -> str:
    """Strip harmony special tokens from text for clean display."""
    return HARMONY_SPECIAL.sub("", text).strip()


# ── Streaming completion ───────────────────────────────────────────────────────

async def stream_completion(
    messages: list[dict],
    model: str,
    tool_config: list | None = None,
    event_queue: asyncio.Queue | None = None,
) -> AsyncIterator[dict]:
    """
    Stream a completion from the OSS-120B model via vLLM chat/completions.

    Flow:
      1. Call /v1/chat/completions, stream text chunks
      2. Emit live status/text events as chunks arrive (for responsiveness)
      3. After stream ends, parse harmony tokens from full text
      4. Re-emit properly typed events: thinking, text, tool_call
      5. Yield the final assistant message dict for conversation history

    Yields: assistant message dicts for appending to conversation history.
    """

    async def _emit(event: AgentEvent):
        if event_queue:
            await event_queue.put(event)

    await _emit(AgentEvent(type="status", content="Calling model..."))

    try:
        client = _get_openai_client()

        # Call the standard chat completions endpoint
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=16384,
            temperature=1.0,
            stop=None,
        )

        # Accumulate the full response text
        full_text = ""
        live_buffer = ""   # for live partial display before parse

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                chunk_text = delta.content
                full_text += chunk_text
                live_buffer += chunk_text

                # Emit live text for responsiveness — but don't emit raw harmony tokens
                # Wait until we have a natural break (space/newline) to avoid partial tokens
                if "\n" in live_buffer or len(live_buffer) > 120:
                    display = _clean_for_display(live_buffer)
                    if display:
                        await _emit(AgentEvent(type="text", content=display))
                    live_buffer = ""

            if choice.finish_reason:
                break

        # Flush any remaining live buffer
        if live_buffer:
            display = _clean_for_display(live_buffer)
            if display:
                await _emit(AgentEvent(type="text", content=display))

        # Now parse the full text for proper channel routing
        if full_text:
            segments = parse_harmony_output(full_text)

            if segments:
                # Clear the live text events and re-emit as structured events
                # (The frontend should handle overwrite when it sees structured segments)
                await _emit(AgentEvent(type="status", content="Parsing response..."))

                final_text_parts = []

                for seg in segments:
                    channel = seg["channel"]
                    content = seg["content"]
                    tool = seg["tool"]

                    if channel == "analysis" and content:
                        # Chain-of-thought reasoning → thinking block
                        await _emit(AgentEvent(type="thinking", content=content))

                    elif channel == "final":
                        if tool:
                            # Tool call — content before <|call|> is the tool input
                            await _emit(AgentEvent(
                                type="tool_call",
                                tool_name=tool,
                                content=content,
                            ))
                        elif content:
                            final_text_parts.append(content)
                            await _emit(AgentEvent(type="text", content=content))

                # Yield the final assistant message for conversation history
                final_content = "\n".join(final_text_parts) if final_text_parts else _clean_for_display(full_text)
                if final_content:
                    yield assistant_message(final_content)

                # Check for any tool calls and yield tool call info
                for seg in segments:
                    if seg.get("tool"):
                        yield {
                            "role": "assistant",
                            "content": seg["content"],
                            "__tool__": seg["tool"],
                        }

            else:
                # No harmony tokens found — model output is plain text (fallback)
                plain = _clean_for_display(full_text)
                if plain:
                    await _emit(AgentEvent(type="text", content=plain))
                    yield assistant_message(plain)
        else:
            await _emit(AgentEvent(type="error", content="Model returned empty response."))

    except Exception as e:
        err_msg = f"Model error: {e}"
        await _emit(AgentEvent(type="error", content=err_msg))
        raise


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