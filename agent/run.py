"""
agent/run.py — Main KaggleClaw agent loop.

Orchestrates the multi-turn conversation between:
  - The OSS-120B model (via openai_harmony / harmony.py)
  - Available tools (browser, python, file, apply_patch)
  - The SSE event queue (consumed by FastAPI /stream endpoint)

Usage:
    from agent.run import AgentRunner
    runner = AgentRunner(event_queue=queue)
    await runner.run(initial_message="Start solving the competition.")
"""

import asyncio
import os
from typing import Any

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from .harmony import (
    AgentEvent,
    assistant_message,
    stream_completion,
    system_message,
    user_message,
    emit_tool_result,
)
from .prompt import build_system_prompt


# Model to use — overridable via env var
MODEL_NAME = os.environ.get("MODEL_NAME", "oss-120b")

# Maximum turns before forcing a stop (safety net)
MAX_TURNS = 50


class AgentRunner:
    """
    Manages the full agent loop for one Kaggle competition session.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        tools: list | None = None,
        model: str = MODEL_NAME,
    ):
        self.event_queue = event_queue
        self.model = model
        self.tools = tools or []
        self._tool_map: dict[str, Any] = {t.name: t for t in self.tools}
        self._messages: list[Message] = []
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def reset(self):
        self._messages = []
        self._running = False

    async def add_user_message(self, text: str):
        """Inject a user message during an ongoing session."""
        msg = user_message(text)
        self._messages.append(msg)

    async def run(self, initial_message: str | None = None):
        """
        Start (or continue) the agent loop.
        If initial_message is provided, it's appended before the loop begins.
        """
        if not self._messages:
            # First run: inject system prompt
            sys_msg = system_message(build_system_prompt())
            self._messages.append(sys_msg)

        if initial_message:
            msg = user_message(initial_message)
            self._messages.append(msg)
            await self._emit(AgentEvent(type="status", content=f"Starting agent with: {initial_message[:100]}"))

        self._running = True
        turn = 0

        try:
            while self._running and turn < MAX_TURNS:
                turn += 1
                await self._emit(AgentEvent(type="status", content=f"Turn {turn}"))

                # — Generate model response —
                assistant_response = None
                tool_calls_in_turn: list[tuple[str, str]] = []  # (tool_name, args)

                async for msg in stream_completion(
                    messages=self._messages,
                    model=self.model,
                    tool_config=[t.tool_config if hasattr(t, "tool_config") else None
                                 for t in self.tools],
                    event_queue=self.event_queue,
                ):
                    assistant_response = msg

                if assistant_response is None:
                    await self._emit(AgentEvent(type="error", content="Model returned no response."))
                    break

                self._messages.append(assistant_response)

                # — Check for terminal signals —
                response_text = self._extract_text(assistant_response)

                if "SUBMISSION READY" in response_text:
                    await self._emit(AgentEvent(
                        type="status",
                        content="✅ Agent completed! Submission ready.",
                        metadata={"done": True},
                    ))
                    self._running = False
                    break

                # — Route tool calls if any —
                routed = await self._route_tool_calls(self._messages)
                if not routed:
                    # No tool calls → agent is done or waiting for user
                    if turn >= 2:
                        await self._emit(AgentEvent(type="done", content="Agent turn complete."))
                    self._running = False
                    break

        except asyncio.CancelledError:
            self._running = False
            await self._emit(AgentEvent(type="status", content="Agent cancelled."))
        except Exception as e:
            self._running = False
            await self._emit(AgentEvent(type="error", content=f"Agent loop error: {e}"))
            raise

    async def send_user_message(self, text: str):
        """
        Send a message from the user and resume the agent loop.
        Call this while the agent is idle (not running).
        """
        msg = user_message(text)
        self._messages.append(msg)
        await self._emit(AgentEvent(type="text", content=f"**User:** {text}", tool_name="user"))
        await self.run()

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _emit(self, event: AgentEvent):
        await self.event_queue.put(event)

    def _extract_text(self, msg: Message) -> str:
        if not msg.content:
            return ""
        parts = []
        for item in msg.content:
            if hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)

    async def _route_tool_calls(self, messages: list[Message]) -> bool:
        """
        Look at the last assistant message, find any tool-directed sub-messages,
        route them to the right tool, collect responses, append to history.
        Returns True if at least one tool was called.
        """
        last_msg = messages[-1]
        if not last_msg.content:
            return False

        # In openai_harmony, tool calls are sub-messages routed to tool names via .recipient
        # For simplicity, we scan the text for tool invocation markers
        # The model produces messages with recipients set to tool names
        # Here we handle the case where the model produces a message for each tool call

        tool_called = False

        # Check if last message has a recipient that maps to a tool
        recipient = getattr(last_msg, "recipient", None) or getattr(last_msg, "_recipient", None)

        if recipient and recipient in self._tool_map:
            tool = self._tool_map[recipient]
            tool_name = tool.name
            args_text = self._extract_text(last_msg)

            await self._emit(AgentEvent(
                type="tool_call",
                tool_name=tool_name,
                content=args_text,
            ))

            try:
                tool_responses = []
                async for response_msg in tool.process(last_msg):
                    tool_responses.append(response_msg)
                    output = self._extract_text(response_msg)
                    await emit_tool_result(tool_name, output, self.event_queue)
                    self._messages.append(response_msg)
                tool_called = bool(tool_responses)
            except Exception as e:
                err = f"[TOOL ERROR] {tool_name}: {e}"
                await self._emit(AgentEvent(type="error", content=err, tool_name=tool_name))
                # Inject error as tool response so model can retry
                err_msg = tool.error_message(err)
                self._messages.append(err_msg)
                tool_called = True

        return tool_called
