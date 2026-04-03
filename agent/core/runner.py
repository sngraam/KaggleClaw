from __future__ import annotations
import asyncio
import logging
import traceback
from typing import Any
from .client import LLMClient
from .events import AgentEvent
from ..harmo import build_messages
from config.settings import settings

logger = logging.getLogger("kaggleclaw.runner")

class AgentRunner:
    def __init__(
        self,
        event_queue: asyncio.Queue,
        tools: list,
        client: LLMClient | None = None,
        max_turns: int = settings.MAX_TURNS,
    ):
        self.event_queue = event_queue
        self.tools = {t.name: t for t in tools}
        self.client = client or LLMClient(
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY,
            model=settings.VLLM_MODEL,
            context_tokens=settings.CONTEXT_TOKENS
        )
        self.max_turns = max_turns
        self.messages = []
        self._running = False
        self._cancel_flag = False

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

    async def run(self, initial_message: str = ""):
        if self._running:
            await self._emit_error("Agent is already running.")
            return

        try:
            self.messages = build_messages()
        except Exception as exc:
            await self._emit_error("Failed to build initial messages", exc)
            return

        if initial_message:
            from openai_harmony import Message, Role
            self.messages.append(Message.from_role_and_content(Role.USER, initial_message))

        await self._agent_loop()

    async def _agent_loop(self):
        self._running = True
        self._cancel_flag = False
        consecutive_no_tool = 0

        try:
            await self._emit_status(f"Starting agent — {len(self.messages)} initial messages")

            for turn in range(self.max_turns):
                if self._cancel_flag:
                    await self._emit_status("Agent cancelled")
                    break

                await self._emit_status(f"Turn {turn + 1}/{self.max_turns} — calling model...")

                new_messages = []
                tool_message = None
                current_channel = None
                emitted_len = 0

                try:
                    async for event in self.client.completion_stream(self.messages):
                        if self._cancel_flag: break
                        
                        if event["type"] == "text":
                            await self._emit(AgentEvent(type="text", content=event["content"]))
                        
                        elif event["type"] == "harmony_messages":
                            new_messages = event["messages"]
                            msg = new_messages[-1]
                            recipient = getattr(msg, "recipient", None)
                            channel = getattr(msg, "channel", None)
                            full_text = self._extract_text(msg)

                            if channel != current_channel:
                                current_channel = channel
                                emitted_len = 0

                            content = full_text[emitted_len:]
                            if content:
                                emitted_len = len(full_text)
                                if channel in ("thinking", "analysis"):
                                    await self._emit(AgentEvent(type="thinking", content=content))
                                elif channel == "final":
                                    await self._emit(AgentEvent(type="text", content=content))
                                elif recipient:
                                    # Identify tool name from recipient or channel
                                    tool_name = recipient
                                    if not tool_name and channel and channel.startswith("functions."):
                                        tool_name = channel.removeprefix("functions.")
                                    
                                    await self._emit(AgentEvent(type="tool_call", tool_name=tool_name, content=content))
                                    tool_message = msg

                    if new_messages:
                        self.messages.extend(new_messages)
                        
                except Exception as exc:
                    await self._emit_error(f"Model call failed on turn {turn + 1}", exc)
                    break

                if tool_message is None:
                    final_text = self._extract_text(self.messages[-1]) if self.messages else ""
                    if "SUBMISSION READY" in final_text:
                        await self._emit_status("Mission Accomplished: SUBMISSION READY string found.", done=True)
                        await self._emit(AgentEvent(type="done", content="Task Completed Successfully!"))
                        break
                    
                    consecutive_no_tool += 1
                    if consecutive_no_tool >= 3:
                        await self._emit_status("Agent stopped making progress.", done=True)
                        await self._emit(AgentEvent(type="done", content="Agent halted due to lack of progress."))
                        break
                    continue
                else:
                    consecutive_no_tool = 0

                # ── Dispatch Tool ──
                try:
                    responses = await self._dispatch_tool(tool_message)
                    self.messages.extend(responses)
                except Exception as exc:
                    await self._emit_error(f"Tool dispatch failed", exc)

            else:
                await self._emit_status(f"Reached max turns ({self.max_turns})")
                await self._emit(AgentEvent(type="done", content="Max turns reached."))

        except Exception as exc:
            await self._emit_error("Unexpected error in agent loop", exc)
        finally:
            self._running = False

    async def _dispatch_tool(self, message: Any) -> list[Any]:
        from openai_harmony import Role, Author, TextContent
        from uuid import uuid4
        
        recipient = getattr(message, "recipient", None) or ""
        channel = getattr(message, "channel", None) or ""
        tool_name = recipient or (channel.removeprefix("functions.") if channel.startswith("functions.") else "")
        if tool_name.startswith("functions."): tool_name = tool_name.removeprefix("functions.")

        tool = self.tools.get(tool_name)
        if tool is None:
            err = f"[ERROR] Unknown tool '{tool_name}'"
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name or "unknown", content=err))
            return [Message(id=uuid4(), author=Author(role=Role.TOOL, name=tool_name or "unknown"), content=[TextContent(text=err)], channel=channel).with_recipient("assistant")]

        responses = []
        result_parts = []
        try:
            async for resp_msg in tool.process(message):
                # Truncate large outputs
                if hasattr(resp_msg, 'content') and isinstance(resp_msg.content, list):
                    for c in resp_msg.content:
                        if hasattr(c, 'text') and len(c.text) > 15000:
                            c.text = c.text[:15000] + "... [TRUNCATED] ..."
                
                responses.append(resp_msg)
                result_parts.append(self._extract_text(resp_msg))

            combined_result = "\n".join(result_parts)
            await self._emit(AgentEvent(type="tool_result", tool_name=tool_name, content=combined_result))
        except Exception as exc:
            err = f"[TOOL ERROR] {tool_name}: {exc}"
            await self._emit_error(err)
            responses = [Message(id=uuid4(), author=Author(role=Role.TOOL, name=tool_name), content=[TextContent(text=err)], channel=channel).with_recipient("assistant")]

        return responses
        finally:
            self._running = False

    def _extract_text(self, msg) -> str:
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
