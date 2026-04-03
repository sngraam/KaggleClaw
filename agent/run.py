"""
agent/run.py — KaggleClaw AgentRunner

The core agentic loop: builds messages, streams from vLLM, parses harmony
tokens, dispatches tools, emits SSE events, loops until done or cancelled.

Key design decisions:
  - 512-token buffer before calling parser → avoids per-chunk parse overhead
  - 5-min generation timeout (asyncio shield) → model can never silently hang
  - 2-min tool dispatch timeout → tools can never deadlock the loop
  - Ping emitted every 15s → SSE connection stays alive through proxies
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("kaggleclaw.runner")

# ── Harmony imports (guarded) ──────────────────────────────────────────────────

try:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        StreamableParser,
        load_harmony_encoding,
    )
    _HARMONY_OK = True
except ImportError:
    _HARMONY_OK = False
    logger.warning("openai_harmony not installed — AgentRunner will be limited.")

# ── Constants ──────────────────────────────────────────────────────────────────

# How many tokens to accumulate before parsing. Reduces parser call frequency.
TOKEN_BUFFER_FLUSH_SIZE = 512

# Seconds before we forcibly timeout a single model generation call
GENERATION_TIMEOUT = 300   # 5 minutes

# Seconds before we forcibly timeout a single tool execution
TOOL_TIMEOUT = 120          # 2 minutes

# Seconds between SSE keep-alive pings
PING_INTERVAL = 15

# ── AgentEvent ─────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    """A single event pushed to the SSE stream."""
    type: str                               # thinking | text | tool_call | tool_result | error | status | done | ping
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return (
            f"data: {json.dumps({'type': self.type, 'content': self.content, 'tool_name': self.tool_name, 'metadata': self.metadata})}\n\n"
        )


# ── Harmony special tokens ─────────────────────────────────────────────────────

HARMONY_TOKENS = {
    200002: "<|return|>",
    200003: "<|constrain|>",
    200005: "<|channel|>",
    200006: "<|start|>",
    200007: "<|end|>",
    200008: "<|message|>",
    200012: "<|call|>",
}

# Stop tokens that indicate a tool call boundary
_TOOL_STOP_IDS = {200002, 200012}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _extract_text(msg) -> str:
    """Safely extract text content from a harmony Message."""
    try:
        content = msg.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "text"):
                    parts.append(item.text or "")
                else:
                    parts.append(str(item))
            return "".join(parts)
        if hasattr(content, "text"):
            return content.text or ""
        return str(content)
    except Exception:
        return ""


def _has_stop_token(token_ids: list[int]) -> bool:
    """Return True if any token in the list is a stop/call boundary token."""
    return any(tid in _TOOL_STOP_IDS for tid in token_ids)


# ── AgentRunner ────────────────────────────────────────────────────────────────

class AgentRunner:

    def __init__(
        self,
        event_queue: asyncio.Queue,
        tools: list,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str = "sk-local",
        max_turns: int = 50,
        context_tokens: int = 32_000,
        temperature: float = 1.0,
    ):
        self.event_queue    = event_queue
        self.tools          = {t.name: t for t in tools}
        self.model          = model or os.environ.get("VLLM_MODEL", "open-scorer-120b")
        self.base_url       = base_url or os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8080/v1")
        self.api_key        = api_key
        self.max_turns      = max_turns
        self.context_tokens = context_tokens
        self.temperature    = temperature

        self.messages: list[Message] = []
        self._running     = False
        self._cancel_flag = False
        self._client      = None

        # Ping task handle
        self._ping_task: asyncio.Task | None = None

        if _HARMONY_OK:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        else:
            self.encoding       = None
            self.stop_token_ids = [200002, 200012]

    # ── Client ──────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    # ── Event helpers ────────────────────────────────────────────────────────────

    async def _emit(self, event: AgentEvent):
        await self.event_queue.put(event)

    async def _emit_status(self, msg: str, **meta):
        await self._emit(AgentEvent(type="status", content=msg, metadata=meta))

    async def _emit_error(self, msg: str, exc: Exception | None = None):
        detail = f"{msg}"
        if exc:
            detail += f"\n\n{traceback.format_exc()}"
        logger.error(detail)
        await self._emit(AgentEvent(type="error", content=detail))

    async def _emit_ping(self):
        """Send SSE keep-alive pings at PING_INTERVAL seconds."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                await self._emit(AgentEvent(type="ping", content=""))
        except asyncio.CancelledError:
            pass

    # ── Public interface ─────────────────────────────────────────────────────────

    async def run(self, initial_message: str = ""):
        """Start a fresh agent session and run the agentic loop."""
        if self._running:
            await self._emit_error("Agent is already running.")
            return

        try:
            from .harmo import build_messages
            self.messages = build_messages()
        except Exception as exc:
            await self._emit_error("Failed to build initial messages", exc)
            return

        if initial_message:
            try:
                from openai_harmony import Message, Role
                user_msg = Message.from_role_and_content(Role.USER, initial_message)
                self.messages.append(user_msg)
            except Exception:
                self.messages.append({"role": "user", "content": initial_message})

        await self._agent_loop()

    async def send_user_message(self, text: str):
        """Inject a user message and continue the loop."""
        if not self.messages:
            await self.run(initial_message=text)
            return

        try:
            from openai_harmony import Message, Role
            user_msg = Message.from_role_and_content(Role.USER, text)
            self.messages.append(user_msg)
        except Exception:
            self.messages.append({"role": "user", "content": text})

        if not self._running:
            await self._agent_loop()

    def cancel(self):
        """Request cancellation of the current loop."""
        self._cancel_flag = True

    def reset(self):
        """Clear conversation history."""
        self.messages = []
        self._running = False
        self._cancel_flag = False
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        self._ping_task = None

    # ── Agentic loop ─────────────────────────────────────────────────────────────

    async def _agent_loop(self):
        self._running     = True
        self._cancel_flag = False
        turn = 0
        consecutive_no_tool = 0

        # Start keep-alive ping task
        self._ping_task = asyncio.create_task(self._emit_ping())

        try:
            await self._emit_status(f"Starting agent — {len(self.messages)} initial messages")

            for turn in range(self.max_turns):
                if self._cancel_flag:
                    await self._emit_status("Agent cancelled", done=True)
                    break

                await self._emit_status(f"Turn {turn + 1}/{self.max_turns} — calling model...")

                # ── Model call with outer timeout guard ──────────────────────
                try:
                    tool_message = await asyncio.wait_for(
                        self._call_model_and_stream(),
                        timeout=GENERATION_TIMEOUT + 60,   # slightly above inner timeout
                    )
                except asyncio.TimeoutError:
                    await self._emit_error(
                        f"Generation hard-timeout on turn {turn + 1} "
                        f"({GENERATION_TIMEOUT + 60}s). Stopping agent to prevent infinite hang."
                    )
                    break
                except Exception as exc:
                    await self._emit_error(f"Model call failed on turn {turn + 1}", exc)
                    break

                # ── Detect "done" / no-tool scenarios ───────────────────────
                if tool_message is None:
                    final_text = ""
                    if self.messages:
                        final_text = _extract_text(self.messages[-1])

                    if "SUBMISSION READY" in final_text:
                        await self._emit_status("Mission Accomplished: SUBMISSION READY found.", done=True)
                        await self._emit(AgentEvent(type="done", content="Task Completed Successfully!"))
                        break

                    consecutive_no_tool += 1
                    if consecutive_no_tool >= 3:
                        await self._emit_status(
                            "Agent stopped making progress (3 consecutive turns without a tool call).",
                            done=True,
                        )
                        await self._emit(AgentEvent(type="done", content="Agent halted — no progress."))
                        break

                    # Nudge model to keep going
                    try:
                        from openai_harmony import Message, Author, Role, TextContent
                        from uuid import uuid4
                        continuation = Message(
                            id=uuid4(),
                            author=Author(role=Role.USER),
                            content=[TextContent(
                                text=(
                                    "You stopped without outputting 'SUBMISSION READY' or calling a tool. "
                                    "Please continue analyzing, call necessary tools, and when fully complete "
                                    "output 'SUBMISSION READY'."
                                )
                            )],
                        ).with_recipient("assistant")
                        self.messages.append(continuation)
                        continue
                    except Exception:
                        pass
                else:
                    consecutive_no_tool = 0

                # ── Dispatch tool with timeout ───────────────────────────────
                try:
                    tool_responses = await asyncio.wait_for(
                        self._dispatch_tool(tool_message),
                        timeout=TOOL_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    tool_name = getattr(tool_message, "recipient", "unknown")
                    err = f"[TOOL TIMEOUT] Tool '{tool_name}' exceeded {TOOL_TIMEOUT}s limit."
                    logger.error(err)
                    await self._emit(AgentEvent(type="tool_result", tool_name=str(tool_name), content=err))
                    tool_responses = []
                except Exception as exc:
                    await self._emit_error(f"Tool dispatch failed: {tool_message}", exc)
                    tool_responses = []

                for resp in tool_responses:
                    self.messages.append(resp)

            else:
                await self._emit_status(f"Reached max turns ({self.max_turns})", done=True)
                await self._emit(AgentEvent(type="done", content=f"Max turns ({self.max_turns}) reached."))

        except Exception as exc:
            await self._emit_error("Unexpected error in agent loop", exc)
        finally:
            self._running = False
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
                self._ping_task = None

    # ── vLLM streaming ───────────────────────────────────────────────────────────

    async def _call_model_and_stream(self) -> Message | None:
        """
        Stream completion tokens from vLLM, parse harmony messages, emit SSE events.

        Token buffering strategy:
          - Accumulate tokens in `token_buffer` (list[int]).
          - Only invoke the harmony parser when buffer >= TOKEN_BUFFER_FLUSH_SIZE
            OR when a stop/call boundary token is detected.
          - This prevents the parser from being called on every tiny chunk.
        """
        if not _HARMONY_OK:
            await self._emit_error("openai_harmony not available — cannot stream.")
            return None

        encoding = self.encoding
        conversation = Conversation.from_messages(self.messages)

        try:
            prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        except Exception as exc:
            await self._emit_error("Failed to render conversation for completion", exc)
            return None

        # Dynamic max-token budget
        max_tokens = self.context_tokens - len(prompt_ids)
        if max_tokens < 500:
            await self._emit_error(
                f"Context length exceeded. Prompt: {len(prompt_ids)} tokens, "
                f"window: {self.context_tokens}."
            )
            return None

        client = self._get_client()

        # Auto-detect model name
        try:
            model_info = await client.models.list()
            if model_info.data:
                self.model = model_info.data[0].id
                logger.info(f"Auto-detected vLLM model: {self.model}")
        except Exception:
            pass

        # ── Start streaming request ──────────────────────────────────────────
        try:
            stream = await client.completions.create(
                model=self.model,
                prompt=prompt_ids,
                max_tokens=min(max_tokens, 32256),
                temperature=self.temperature,
                stream=True,
                logprobs=5,
                stop=None,
                extra_body={
                    "stop_token_ids": self.stop_token_ids,
                    "return_token_ids": True,
                    "min_p": 0.02,
                },
            )
        except Exception as exc:
            await self._emit_error(f"vLLM request failed (url={self.base_url})", exc)
            return None

        token_buffer: list[int] = []
        tool_message: Message | None = None
        current_channel = None
        emitted_len = 0
        new_messages: list = []

        # Track the last time we flushed the buffer so we always flush within
        # a reasonable wall-clock window even on slow generation.
        last_flush_time = time.monotonic()
        FLUSH_INTERVAL_S = 2.0   # flush at least every 2 seconds

        def _token_meta(ids: list[int]) -> dict:
            for tid, tname in HARMONY_TOKENS.items():
                if tid in ids:
                    return {"token": tname}
            return {}

        async def _try_parse_and_emit() -> None:
            nonlocal current_channel, emitted_len, new_messages, tool_message, last_flush_time
            if not token_buffer:
                return
            try:
                new_messages = encoding.parse_messages_from_completion_tokens(
                    token_buffer, Role.ASSISTANT
                )
            except Exception:
                return
            last_flush_time = time.monotonic()

            if not new_messages:
                return

            msg = new_messages[-1]
            channel   = getattr(msg, "channel", None)
            recipient = getattr(msg, "recipient", None)
            full_text = _extract_text(msg)

            if channel != current_channel:
                current_channel = channel
                emitted_len = 0

            delta = full_text[emitted_len:]
            if not delta:
                return
            emitted_len = len(full_text)

            meta = _token_meta(token_buffer[-8:])  # check last few tokens for boundary

            if channel in ("thinking", "analysis"):
                await self._emit(AgentEvent(type="thinking", content=delta, metadata=meta))

            elif channel == "final":
                await self._emit(AgentEvent(type="text", content=delta, metadata=meta))

            elif recipient == "python":
                await self._emit(AgentEvent(
                    type="tool_call", tool_name="python",
                    content=delta, metadata={**meta, "channel": channel},
                ))
                tool_message = msg

            elif recipient in ("file", "apply_patch", "web_search", "plan_follow", "browser"):
                await self._emit(AgentEvent(
                    type="tool_call", tool_name=recipient,
                    content=delta, metadata={**meta, "channel": channel},
                ))
                tool_message = msg

            elif channel == "functions" or (recipient and recipient.startswith("functions.")):
                tool_name_inner = (recipient or "").replace("functions.", "")
                await self._emit(AgentEvent(
                    type="tool_call", tool_name=tool_name_inner or "function",
                    content=delta, metadata={**meta, "channel": channel},
                ))
                tool_message = msg

        # ── Main streaming loop ──────────────────────────────────────────────
        try:
            # Wrap with our inner generation timeout
            async def _stream_with_timeout():
                nonlocal token_buffer
                async for chunk in stream:
                    if self._cancel_flag:
                        return

                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        continue

                    text_delta = choice.text or ""

                    # Extract token IDs from wherever vLLM puts them
                    token_ids: list[int] = []
                    if hasattr(choice, "token_ids"):
                        token_ids = choice.token_ids or []
                    elif hasattr(chunk, "token_ids"):
                        token_ids = chunk.token_ids or []
                    elif getattr(choice, "model_extra", None):
                        token_ids = choice.model_extra.get("token_ids", [])
                    elif getattr(chunk, "model_extra", None):
                        token_ids = chunk.model_extra.get("token_ids", [])

                    if not token_ids and text_delta and self.encoding:
                        # Fallback: encode the text delta to keep parser fed
                        try:
                            token_ids = self.encoding.encode(text_delta)
                        except Exception:
                            pass

                    if not token_ids:
                        # Pure text fallback (no harmony parsing possible)
                        if text_delta:
                            await self._emit(AgentEvent(type="text", content=text_delta))
                        continue

                    token_buffer.extend(token_ids)

                    # Decide whether to flush the buffer now
                    should_flush = (
                        len(token_buffer) >= TOKEN_BUFFER_FLUSH_SIZE
                        or _has_stop_token(token_ids)
                        or (time.monotonic() - last_flush_time) >= FLUSH_INTERVAL_S
                    )

                    if should_flush:
                        await _try_parse_and_emit()

                    # Check finish reason — ALWAYS evaluate this regardless of token flush
                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason in ("stop", "length"):
                        if finish_reason == "length":
                            await self._emit_status(
                                "Model stopped: reached max token limit.",
                                metadata={"reason": "length"},
                            )
                        # Final flush before exit
                        await _try_parse_and_emit()
                        return

            await asyncio.wait_for(_stream_with_timeout(), timeout=GENERATION_TIMEOUT)

        except asyncio.TimeoutError:
            logger.warning(f"Generation soft-timeout ({GENERATION_TIMEOUT}s) — partial output returned.")
            await self._emit_status(
                f"⚠️ Generation timeout after {GENERATION_TIMEOUT}s — partial output used.",
                metadata={"reason": "timeout"},
            )
            # Still try to parse whatever we got
            await _try_parse_and_emit()

        except Exception as exc:
            await self._emit_error("Error during token streaming", exc)
            return None

        # Final parse of any remaining buffered tokens
        if token_buffer:
            await _try_parse_and_emit()

        if new_messages:
            self.messages.extend(new_messages)

        return tool_message if tool_message else None

    # ── Tool dispatch ─────────────────────────────────────────────────────────────

    async def _dispatch_tool(self, message: Message) -> list[Message]:
        recipient = getattr(message, "recipient", None) or ""
        channel   = getattr(message, "channel", None) or ""

        tool_name = recipient
        if not tool_name and channel.startswith("functions."):
            tool_name = channel.removeprefix("functions.")
        if tool_name.startswith("functions."):
            tool_name = tool_name.removeprefix("functions.")

        tool = self.tools.get(tool_name)

        if tool is None:
            error_text = (
                f"[ERROR] Unknown tool '{tool_name}'. "
                f"Available: {', '.join(self.tools.keys())}"
            )
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name or "unknown", content=error_text))
            try:
                from openai_harmony import Author, TextContent
                from uuid import uuid4
                return [Message(
                    id=uuid4(),
                    author=Author(role=Role.TOOL, name=tool_name or "unknown"),
                    content=[TextContent(text=error_text)],
                    channel=channel,
                ).with_recipient("assistant")]
            except Exception:
                return []

        responses: list[Message] = []
        result_parts: list[str] = []

        try:
            async for resp_msg in tool.process(message):
                # Truncate large tool output to prevent context overflow
                if getattr(resp_msg, "content", None):
                    for c in resp_msg.content:
                        if hasattr(c, "text") and len(c.text) > 15_000:
                            c.text = (
                                c.text[:15_000]
                                + f"\n...[TRUNCATED from {len(c.text)} bytes to prevent context overflow]..."
                            )
                responses.append(resp_msg)
                result_parts.append(_extract_text(resp_msg))

            combined_result = "\n".join(result_parts)
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name, content=combined_result))

        except Exception as exc:
            error_text = f"[TOOL ERROR] {tool_name}: {exc}\n{traceback.format_exc()}"
            logger.error(error_text)
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name, content=error_text))
            try:
                from openai_harmony import Author, TextContent
                from uuid import uuid4
                responses = [Message(
                    id=uuid4(),
                    author=Author(role=Role.TOOL, name=tool_name),
                    content=[TextContent(text=error_text)],
                    channel=channel,
                ).with_recipient("assistant")]
            except Exception:
                responses = []

        return responses
