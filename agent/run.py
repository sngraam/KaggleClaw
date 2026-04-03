"""
agent/run.py — KaggleClaw AgentRunner

The core agentic loop: builds messages, streams from vLLM, parses harmony
tokens, dispatches tools, emits SSE events, loops until done or cancelled.
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

# ── AgentEvent ─────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    """A single event pushed to the SSE stream."""
    type: str                               # thinking | text | tool_call | tool_result | error | status | done
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


# ── AgentRunner ────────────────────────────────────────────────────────────────

def _extract_text(msg) -> str:
    """Safely extract text content from a harmony Message."""
    try:
        content = msg.content
        if isinstance(content, list):
            parts =[]
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
        self.event_queue   = event_queue
        self.tools         = {t.name: t for t in tools}
        self.model         = model or os.environ.get("VLLM_MODEL", "open-scorer-120b")
        self.base_url      = base_url or os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8080/v1")
        self.api_key       = api_key
        self.max_turns     = max_turns
        self.context_tokens = context_tokens
        self.temperature   = temperature

        self.messages: list[Message] =[]
        self._running     = False
        self._cancel_flag = False

        self._client = None

        if _HARMONY_OK:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        else:
            self.encoding       = None
            self.stop_token_ids = [200002, 200012]

    # ── Client ──────────────────────────────────────────────────────────────

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

    # ── Event helpers ────────────────────────────────────────────────────────

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

    # ── Public interface ─────────────────────────────────────────────────────

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
        self.messages =[]
        self._running = False
        self._cancel_flag = False

    # ── Agentic loop ─────────────────────────────────────────────────────────

    async def _agent_loop(self):
        self._running     = True
        self._cancel_flag = False
        turn = 0
        consecutive_no_tool = 0

        try:
            await self._emit_status(f"Starting agent — {len(self.messages)} initial messages")

            for turn in range(self.max_turns):
                if self._cancel_flag:
                    await self._emit_status("Agent cancelled", done=True)
                    break

                await self._emit_status(f"Turn {turn + 1}/{self.max_turns} — calling model...")

                try:
                    tool_message = await self._call_model_and_stream()
                except Exception as exc:
                    await self._emit_error(f"Model call failed on turn {turn + 1}", exc)
                    break

                # ── Fix: Stop model loop when done ──
                if tool_message is None:
                    final_text = ""
                    if self.messages:
                        final_text = _extract_text(self.messages[-1])
                    
                    if "SUBMISSION READY" in final_text:
                        await self._emit_status("Mission Accomplished: SUBMISSION READY string found.", done=True)
                        await self._emit(AgentEvent(type="done", content="Task Completed Successfully!"))
                        break

                    consecutive_no_tool += 1
                    if consecutive_no_tool >= 3:
                        await self._emit_status("Agent stopped making progress. Stopping to prevent infinite loops.", done=True)
                        await self._emit(AgentEvent(type="done", content="Agent halted due to lack of progress."))
                        break

                    # Instruct it to keep going, but clarify final objective.
                    try:
                        from openai_harmony import Message, Author, Role, TextContent
                        from uuid import uuid4
                        continuation = Message(
                            id=uuid4(),
                            author=Author(role=Role.USER),
                            content=[TextContent(text="You stopped without outputting 'SUBMISSION READY' or calling a tool. Please continue analyzing, call necessary tools, and when you are fully complete, output 'SUBMISSION READY'.")]
                        ).with_recipient("assistant")
                        self.messages.append(continuation)
                        continue
                    except Exception:
                        pass
                else:
                    consecutive_no_tool = 0

                # ── Dispatch Tool ──
                try:
                    tool_responses = await self._dispatch_tool(tool_message)
                except Exception as exc:
                    await self._emit_error(f"Tool dispatch failed: {tool_message}", exc)
                    tool_responses =[]

                for resp in tool_responses:
                    self.messages.append(resp)

            else:
                await self._emit_status(f"Reached max turns ({self.max_turns})", done=True)
                await self._emit(AgentEvent(type="done", content=f"Max turns ({self.max_turns}) reached."))

        except Exception as exc:
            await self._emit_error("Unexpected error in agent loop", exc)
        finally:
            self._running = False

    # ── vLLM streaming ───────────────────────────────────────────────────────

    async def _call_model_and_stream(self) -> Message | None:

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

        # ── Fix: Dynamic Maximum tokens bounding ──
        max_tokens = self.context_tokens - len(prompt_ids)
        if max_tokens < 500:
            await self._emit_error(f"Context length limits reached. Prompt tokens: {len(prompt_ids)}. Context window: {self.context_tokens}")
            return None

        client = self._get_client()
        token_buffer: list[int] =[]

        try:
            stream = await client.completions.create(
                model=self.model,
                prompt=prompt_ids,
                max_tokens=min(max_tokens, 32256), # vLLM outputs usually capped
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

        tool_message: Message | None = None
        current_channel = None
        emitted_len = 0
        new_messages =[]

        try:
            async for chunk in stream:
                if self._cancel_flag:
                    break

                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                text_delta = choice.text or ""

                # ── Fix: Robust extraction of token_ids ensuring the token_buffer fills ──
                token_ids: list[int] =[]
                if hasattr(choice, "token_ids"):
                    token_ids = choice.token_ids or[]
                elif hasattr(chunk, "token_ids"):
                    token_ids = chunk.token_ids or[]
                elif getattr(choice, "model_extra", None):
                    token_ids = choice.model_extra.get("token_ids",[])
                elif getattr(chunk, "model_extra", None):
                    token_ids = chunk.model_extra.get("token_ids",[])

                if not token_ids:
                    # Fallback to text encoding to keep parser alive if vLLM fails to send IDs
                    if text_delta and self.encoding:
                        try:
                            token_ids = self.encoding.encode(text_delta)
                        except Exception:
                            pass

                if not token_ids:
                    if text_delta:
                        await self._emit(AgentEvent(type="text", content=text_delta))
                    continue

                token_buffer.extend(token_ids)

                try:
                    new_messages = encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                except Exception:
                    continue

                if not new_messages:
                    continue

                msg = new_messages[-1]
                channel   = getattr(msg, "channel", None)
                recipient = getattr(msg, "recipient", None)
                full_text = _extract_text(msg)

                if channel != current_channel:
                    current_channel = channel
                    emitted_len = 0

                content = full_text[emitted_len:]
                if not content:
                    continue
                emitted_len = len(full_text)

                meta = {}
                for tid, tname in HARMONY_TOKENS.items():
                    if str(tid) in str(token_ids):
                        meta["token"] = tname
                        break

                if channel == "thinking" or channel == "analysis":
                    await self._emit(AgentEvent(type="thinking", content=content, metadata=meta))

                elif channel == "final":
                    await self._emit(AgentEvent(type="text", content=content, metadata=meta))

                elif recipient == "python":
                    await self._emit(AgentEvent(
                        type="tool_call",
                        tool_name="python",
                        content=content,
                        metadata={**meta, "channel": channel},
                    ))
                    tool_message = msg

                elif recipient in ("file", "apply_patch", "web_search", "plan_follow", "browser"):
                    await self._emit(AgentEvent(
                        type="tool_call",
                        tool_name=recipient,
                        content=content,
                        metadata={**meta, "channel": channel},
                    ))
                    tool_message = msg

                elif channel == "functions" or (recipient and recipient.startswith("functions.")):
                    tool_name = (recipient or "").replace("functions.", "")
                    await self._emit(AgentEvent(
                        type="tool_call",
                        tool_name=tool_name or "function",
                        content=content,
                        metadata={**meta, "channel": channel},
                    ))
                    tool_message = msg

                # Handle limit hit
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason in ("stop", "length"):
                    if finish_reason == "length":
                        await self._emit_status("Model stopped due to reaching token limit.", metadata={"reason": "length"})
                    break

        except Exception as exc:
            await self._emit_error("Error during token streaming", exc)
            return None

        if new_messages:
            self.messages.extend(new_messages)

        if tool_message is None:
            return None

        return tool_message

    # ── Tool dispatch ─────────────────────────────────────────────────────────

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
            error_text = f"[ERROR] Unknown tool '{tool_name}'. Available: {', '.join(self.tools.keys())}"
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name or "unknown", content=error_text))
            try:
                from openai_harmony import Author, TextContent
                from uuid import uuid4
                return[Message(
                    id=uuid4(),
                    author=Author(role=Role.TOOL, name=tool_name or "unknown"),
                    content=[TextContent(text=error_text)],
                    channel=channel,
                ).with_recipient("assistant")]
            except Exception:
                return []

        responses: list[Message] = []
        result_parts: list[str] =[]

        try:
            async for resp_msg in tool.process(message):
                # TRUNCATE TOOL OUTPUT TO PREVENT CONTEXT OVERFLOW
                if getattr(resp_msg, 'content', None):
                    for i, c in enumerate(resp_msg.content):
                        if hasattr(c, 'text') and len(c.text) > 15000:
                            c.text = c.text[:15000] + f"\n...[TRUNCATED from {len(c.text)} bytes to prevent context overflow] ..."

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
                responses =[Message(
                    id=uuid4(),
                    author=Author(role=Role.TOOL, name=tool_name),
                    content=[TextContent(text=error_text)],
                    channel=channel,
                ).with_recipient("assistant")]
            except Exception:
                responses =[]

        return responses
