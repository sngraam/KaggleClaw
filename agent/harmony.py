import asyncio
import json
import os
import re
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
        ReasoningEffort,
        ToolNamespaceConfig,
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
        @classmethod
        def from_role_and_content(cls, role, content):
            return cls(Author(role), [TextContent(content)])

    class Conversation:
        def __init__(self, messages=None): self.messages = messages or []

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
    return Message.from_role_and_content(Role.USER, text)

def system_message(text: str) -> Message:
    if _HARMONY_AVAILABLE:
        system = (
            SystemContent.new()
            .with_model_identity(text)
            .with_reasoning_effort(ReasoningEffort.HIGH)
            .with_tools(ToolNamespaceConfig.python())
            .with_tools(ToolNamespaceConfig.browser())
        )
        return Message.from_role_and_content(Role.SYSTEM, system)
    return Message.from_role_and_content(Role.SYSTEM, text)

def assistant_message(text: str, channel: str = "final") -> Message:
    msg = Message.from_role_and_content(Role.ASSISTANT, text)
    msg.channel = channel
    return msg

# ── OpenAI async client ────────────────────────────────────────────────────────

def _get_openai_client():
    from openai import AsyncOpenAI
    base_url = os.environ.get("HARMONY_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("HARMONY_API_KEY", "EMPTY")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

# ── Harmony encoding ───────────────────────────────────────────────────────────

_enc_cache = None

def _get_encoding():
    global _enc_cache
    if _enc_cache is None:
        _enc_cache = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _enc_cache

def _encode_messages(messages: list[Message]) -> list[int]:
    enc = _get_encoding()
    conv = Conversation(messages=list(messages))
    token_ids = enc.render_conversation_for_completion(conv, Role.ASSISTANT)
    return token_ids

def _parse_output_tokens(token_ids: list[int]) -> list[Message]:
    enc = _get_encoding()
    return enc.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)

def _extract_text(msg: Message) -> str:
    parts =[]
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

    async def _emit(event: AgentEvent):
        if event_queue:
            await event_queue.put(event)

    await _emit(AgentEvent(type="status", content="Calling model..."))

    if not _HARMONY_AVAILABLE:
        await _emit(AgentEvent(type="error", content="openai_harmony not installed."))
        return

    try:
        client = _get_openai_client()
        prompt_token_ids = _encode_messages(messages)

        # 1. Include logprobs=1 to force vLLM to return token details!
        stream = await client.completions.create(
            model=model,
            prompt=prompt_token_ids,
            stream=True,
            max_tokens=16256,
            logprobs=5, 
            stop=None,
            temperature=1.0,
        )

        token_buffer: list[int] =[]
        full_raw_text = ""
        emitted_lengths = {"thinking": 0, "text": 0}

        def _parse_live_text(text: str):
            """Dynamically separate thinking vs final text, hiding JSON tool calls entirely."""
            thinking_text, final_text = "", ""
            blocks = text.split("<|start|>")
            for block in blocks:
                if not block: continue
                block = block.replace("<|end|>", "")

                # Suppress Tool Calls from frontend stream
                rec_match = re.search(r'<\|recipient\|>([a-zA-Z0-9_]+)', block)
                if rec_match and rec_match.group(1) not in ("assistant", "None", ""):
                    continue 

                chan_match = re.search(r'<\|channel\|>([a-zA-Z0-9_]+)', block)
                is_thinking = (chan_match and chan_match.group(1) == "analysis")

                text_idx = block.find("<|text|>")
                if text_idx != -1:
                    content = block[text_idx + 8:]
                    content = re.sub(r'<\|[a-zA-Z0-9_]+\|>', '', content)
                    content = re.sub(r'<\|[a-zA-Z0-9_]*$', '', content) # strip partial trailing tags
                    if is_thinking:
                        thinking_text += content
                    else:
                        final_text += content
            return thinking_text, final_text

        async for chunk in stream:
            choice = chunk.choices[0]

            # 2. Extract token IDs via standard fields or pydantic extra 
            if getattr(choice, "logprobs", None) is not None:
                lp = choice.logprobs
                if hasattr(lp, "model_extra") and lp.model_extra and "token_ids" in lp.model_extra:
                    token_buffer.extend(lp.model_extra["token_ids"])
                elif hasattr(lp, "token_ids") and lp.token_ids is not None:
                    token_buffer.extend(lp.token_ids)

            if choice.text:
                full_raw_text += choice.text

                # Clean streaming parser handles deltas flawlessly
                thinking_total, final_total = _parse_live_text(full_raw_text)

                if len(thinking_total) > emitted_lengths["thinking"]:
                    delta = thinking_total[emitted_lengths["thinking"]:]
                    emitted_lengths["thinking"] = len(thinking_total)
                    await _emit(AgentEvent(type="thinking", content=delta))

                if len(final_total) > emitted_lengths["text"]:
                    delta = final_total[emitted_lengths["text"]:]
                    emitted_lengths["text"] = len(final_total)
                    await _emit(AgentEvent(type="text", content=delta))

            if choice.finish_reason:
                break

        # 3. Handle Parsing Tokens securely
        parsed_msgs =[]
        if token_buffer:
            try:
                parsed_msgs = _parse_output_tokens(token_buffer)
            except Exception:
                parsed_msgs =[]

        # Fallback Parser: If vLLM stripped token_ids, use string parsing to build properly shaped tool_call Messages
        if not parsed_msgs and full_raw_text:
            blocks = full_raw_text.split("<|start|>")
            for block in blocks:
                if not block.strip(): continue
                block = block.replace("<|end|>", "")

                rec_match = re.search(r'<\|recipient\|>([a-zA-Z0-9_]+)', block)
                recipient = rec_match.group(1) if rec_match else "assistant"

                chan_match = re.search(r'<\|channel\|>([a-zA-Z0-9_]+)', block)
                channel = chan_match.group(1) if chan_match else "final"

                text_idx = block.find("<|text|>")
                if text_idx != -1:
                    content = block[text_idx + 8:]
                    content = re.sub(r'<\|[a-zA-Z0-9_]+\|>', '', content).strip()
                else:
                    content = re.sub(r'<\|[a-zA-Z0-9_]+\|>', '', block).strip()

                if content:
                    try:
                        msg = Message.from_role_and_content(Role.ASSISTANT, content)
                    except Exception:
                        msg = assistant_message(content)
                    msg.channel = channel
                    msg.recipient = recipient
                    msg._recipient = recipient
                    parsed_msgs.append(msg)

        # 4. Route accurately mapped events and yield ALL chunks back to agent logic
        for pmsg in parsed_msgs:
            recipient = getattr(pmsg, "recipient", getattr(pmsg, "_recipient", "assistant"))
            text = _extract_text(pmsg)

            if recipient and str(recipient) not in ("assistant", "None", "NoneType"):
                await _emit(AgentEvent(
                    type="tool_call",
                    tool_name=str(recipient),
                    content=text,
                ))
            
            # NEVER suppress yielding. The `run.py` history requires the tool calls to parse and map correctly
            if text:
                yield pmsg

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
    if event_queue:
        await event_queue.put(AgentEvent(
            type="tool_result",
            tool_name=tool_name,
            content=output,
        ))