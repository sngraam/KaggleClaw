"""
agent/harmony.py — Correct harmony implementation for KaggleClaw.

Key rules from official docs (https://cookbook.openai.com/articles/openai-harmony):

1. PROMPT STRUCTURE:
   system    → SystemContent (identity, dates, reasoning level, built-in tools)
   developer → DeveloperContent (instructions + custom function tools)
   user      → plain text
   assistant → channel=analysis (CoT, NEVER shown to user)
               channel=commentary (tool calls go here for functions.*)
               channel=final (user-facing answer)
   tool      → Author(TOOL, "functions.xxx") + channel=commentary + recipient=assistant

2. CONVERSATION HISTORY (critical):
   - After a turn that ends in <|return|> (final answer):
     DROP analysis messages, keep only the final message.
   - After a turn that ends in <|call|> (tool call):
     KEEP the full chain (analysis + commentary call + tool result) so the
     model can continue its reasoning.
   - Stored messages use <|end|> not <|return|> (library handles this).

3. STREAMING via StreamableParser:
   - Feed token IDs one at a time.
   - stream.current_channel → "analysis" / "commentary" / "final"
   - stream.last_content_delta → incremental text
   - stream.current_recipient → tool name when it's a tool call
   - Stop on token 200002 (<|return|>) or 200012 (<|call|>).

4. TOOL FORMAT:
   - Function tools: channel=commentary, recipient=functions.{name}
   - Built-in python/browser: channel=analysis, recipient=python / browser.search etc.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

# ── openai_harmony imports ─────────────────────────────────────────────────────

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
        ReasoningEffort,
        ToolNamespaceConfig,
        ToolDescription,
        StreamableParser,
    )
    _HARMONY_AVAILABLE = True
    _harmony_import_error = None
except ImportError as _err:
    _HARMONY_AVAILABLE = False
    _harmony_import_error = str(_err)

    class Role:
        USER = "user"; ASSISTANT = "assistant"; SYSTEM = "system"
        DEVELOPER = "developer"; TOOL = "tool"

    class Author:
        def __init__(self, role, name=None): self.role = role; self.name = name
        @classmethod
        def new(cls, role, name=None): return cls(role, name)

    class TextContent:
        def __init__(self, text): self.text = text

    class Message:
        def __init__(self, author, content, channel=None, recipient=None, **_kw):
            self.author = author; self.content = content
            self.channel = channel; self.recipient = recipient
        def with_channel(self, c): self.channel = c; return self
        def with_recipient(self, r): self.recipient = r; return self
        @classmethod
        def from_role_and_content(cls, role, content):
            t = content if isinstance(content, str) else str(content)
            return cls(Author(role), [TextContent(t)])
        @classmethod
        def from_author_and_content(cls, author, content):
            t = content if isinstance(content, str) else str(content)
            return cls(author, [TextContent(t)])

    class Conversation:
        def __init__(self, messages=None): self.messages = messages or []
        @classmethod
        def from_messages(cls, msgs): return cls(list(msgs))


# ── SSE event ──────────────────────────────────────────────────────────────────

EventType = Literal["thinking", "text", "tool_call", "tool_result", "error", "done", "status"]


@dataclass
class AgentEvent:
    type: EventType
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, 'content': self.content, 'tool_name': self.tool_name, 'metadata': self.metadata})}\n\n"


# ── Encoding singleton ─────────────────────────────────────────────────────────

_enc_cache = None

def _get_encoding():
    global _enc_cache
    if _enc_cache is None:
        _enc_cache = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _enc_cache


# ── Message helpers ────────────────────────────────────────────────────────────

def user_message(text: str) -> "Message":
    return Message.from_role_and_content(Role.USER, text)


def tool_result_message(tool_name: str, output: str) -> "Message":
    """
    Per docs:
        <|start|>{toolname} to=assistant<|channel|>commentary<|message|>{output}<|end|>
    """
    author = Author.new(Role.TOOL, tool_name)
    msg = Message.from_author_and_content(author, output)
    return msg.with_channel("commentary").with_recipient("assistant")


def extract_text(msg: "Message") -> str:
    if not getattr(msg, "content", None):
        return ""
    return "".join(c.text for c in msg.content if hasattr(c, "text") and c.text)


# ── Conversation rendering ─────────────────────────────────────────────────────

def render_conversation(messages: list) -> list[int]:
    enc = _get_encoding()
    convo = Conversation.from_messages(messages)
    return enc.render_conversation_for_completion(convo, Role.ASSISTANT)


def parse_completion_tokens(token_ids: list[int]) -> list:
    enc = _get_encoding()
    return enc.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)


# ── vLLM client ────────────────────────────────────────────────────────────────

def _get_client():
    from openai import AsyncOpenAI
    base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
    api_key  = os.environ.get("HARMONY_API_KEY",  "EMPTY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


# ── Streaming completion ───────────────────────────────────────────────────────

async def stream_completion(
    messages: list,
    model: str,
    event_queue: asyncio.Queue | None = None,
) -> AsyncIterator:
    """
    Stream one agent turn from vLLM using the Harmony format.

    Steps:
      1. render_conversation(messages) → token IDs
      2. POST to vLLM /v1/completions with logprobs=1 to get token IDs back
      3. Feed each token ID through StreamableParser
      4. Emit AgentEvents based on current_channel:
            analysis   → "thinking"  (CoT — never shown to end users)
            final      → "text"      (user-facing answer)
            commentary → "tool_call" when current_recipient is set
      5. After stream ends, parse full token_buffer → Message objects
      6. Yield those Messages to run.py for history management
    """

    async def _emit(evt: AgentEvent):
        if event_queue:
            await event_queue.put(evt)

    if not _HARMONY_AVAILABLE:
        await _emit(AgentEvent(type="error", content=f"openai_harmony not installed: {_harmony_import_error}"))
        return

    await _emit(AgentEvent(type="status", content="Encoding prompt..."))

    try:
        prompt_ids = render_conversation(messages)
    except Exception as e:
        await _emit(AgentEvent(type="error", content=f"Prompt render error: {e}"))
        return

    await _emit(AgentEvent(type="status", content="Calling model..."))

    try:
        client = _get_client()
        enc    = _get_encoding()
        parser = StreamableParser(enc, role=Role.ASSISTANT)

        stream = await client.completions.create(
            model=model,
            prompt=prompt_ids,
            stream=True,
            max_tokens=16384,
            logprobs=1,       # so vLLM echoes token IDs back
            temperature=1.0,
            stop=None,
        )

        token_buffer: list[int] = []

        # Accumulated text per channel so we can send deltas
        _acc: dict[str, str] = {"analysis": "", "final": "", "commentary": ""}
        _sent_len: dict[str, int] = {"analysis": 0, "final": 0, "commentary": 0}

        _announced_recipients: set[str] = set()

        async for chunk in stream:
            choice = chunk.choices[0]

            # Extract token IDs from logprobs (vLLM puts them here)
            new_ids: list[int] = []
            lp = getattr(choice, "logprobs", None)
            if lp is not None:
                if hasattr(lp, "model_extra") and lp.model_extra:
                    new_ids = lp.model_extra.get("token_ids", [])
                elif hasattr(lp, "token_ids") and lp.token_ids:
                    new_ids = list(lp.token_ids)

            for tid in new_ids:
                token_buffer.append(tid)
                try:
                    parser.process(tid)
                except Exception:
                    pass  # non-fatal — keep streaming

                channel   = getattr(parser, "current_channel",  None) or ""
                recipient = getattr(parser, "current_recipient", None)
                delta     = getattr(parser, "last_content_delta", None) or ""

                # Announce new tool calls once
                if recipient and recipient not in ("assistant", "", None):
                    if recipient not in _announced_recipients:
                        _announced_recipients.add(recipient)
                        await _emit(AgentEvent(
                            type="tool_call",
                            tool_name=str(recipient),
                            content="",
                        ))

                if not delta:
                    continue

                # Accumulate and emit incremental text by channel
                if channel in _acc:
                    _acc[channel] += delta

                if channel == "analysis":
                    new_chunk = _acc["analysis"][_sent_len["analysis"]:]
                    if new_chunk:
                        _sent_len["analysis"] = len(_acc["analysis"])
                        await _emit(AgentEvent(type="thinking", content=new_chunk))

                elif channel == "final":
                    new_chunk = _acc["final"][_sent_len["final"]:]
                    if new_chunk:
                        _sent_len["final"] = len(_acc["final"])
                        await _emit(AgentEvent(type="text", content=new_chunk))

                elif channel == "commentary" and recipient and recipient not in ("assistant", "", None):
                    # Tool call argument tokens
                    await _emit(AgentEvent(
                        type="tool_call",
                        tool_name=str(recipient),
                        content=delta,
                    ))
                # commentary without recipient = preamble; silently skip

            if choice.finish_reason:
                break

        # ── Parse token buffer → Messages ──────────────────────────────────────
        parsed: list = []
        if token_buffer:
            try:
                parsed = parse_completion_tokens(token_buffer)
            except Exception as e:
                await _emit(AgentEvent(type="error", content=f"Token parse warning: {e}"))

        # Fallback: synthesize a final message from accumulated text
        if not parsed:
            text = _acc.get("final", "") or _acc.get("analysis", "")
            if text:
                msg = Message.from_role_and_content(Role.ASSISTANT, text)
                msg.channel = "final" if _acc.get("final") else "analysis"
                parsed = [msg]

        for msg in parsed:
            yield msg

    except Exception as e:
        await _emit(AgentEvent(type="error", content=f"Model error: {e}"))
        raise


# ── Tool result SSE emitter ────────────────────────────────────────────────────

async def emit_tool_result(
    tool_name: str,
    output: str,
    event_queue: asyncio.Queue | None = None,
):
    if event_queue:
        await event_queue.put(AgentEvent(
            type="tool_result",
            tool_name=tool_name,
            content=output,
        ))