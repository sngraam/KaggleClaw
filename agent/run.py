"""
agent/run.py — Main KaggleClaw agent loop.

Orchestrates the multi-turn conversation between:
  - The OSS-120B model (via harmony.py)
  - Available tools (browser, python, file, apply_patch)
  - The SSE event queue (consumed by FastAPI /stream endpoint)

Messages are plain dicts: {"role": "system"|"user"|"assistant", "content": str}
Tool call messages carry an extra "__tool__" key for routing.

Usage:
    from agent.run import AgentRunner
    runner = AgentRunner(event_queue=queue)
    await runner.run(initial_message="Start solving the competition.")
"""

import asyncio
import os
from typing import Any

from .harmony import (
    AgentEvent,
    assistant_message,
    emit_tool_result,
    stream_completion,
    system_message,
    user_message,
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
        # Messages are plain dicts: {"role": ..., "content": ...}
        self._messages: list[dict] = []
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def reset(self):
        self._messages = []
        self._running = False

    async def add_user_message(self, text: str):
        """Inject a user message during an ongoing session."""
        self._messages.append(user_message(text))

    async def run(self, initial_message: str | None = None):
        """
        Start (or continue) the agent loop.
        If initial_message is provided, it's appended before the loop begins.
        """
        if not self._messages:
            # First run: inject system prompt
            self._messages.append(system_message(build_system_prompt()))

        if initial_message:
            self._messages.append(user_message(initial_message))
            await self._emit(AgentEvent(
                type="status",
                content=f"Starting agent with: {initial_message[:100]}"
            ))

        self._running = True
        turn = 0

        try:
            while self._running and turn < MAX_TURNS:
                turn += 1
                await self._emit(AgentEvent(type="status", content=f"Turn {turn}"))

                # — Generate model response —
                assistant_response: dict | None = None
                tool_calls_in_turn: list[dict] = []

                async for msg in stream_completion(
                    messages=self._messages,
                    model=self.model,
                    tool_config=[getattr(t, "tool_config", None) for t in self.tools],
                    event_queue=self.event_queue,
                ):
                    # Messages with "__tool__" are tool call requests — collect separately
                    if msg.get("__tool__"):
                        tool_calls_in_turn.append(msg)
                    else:
                        # Normal assistant message — keep last one as the "response"
                        assistant_response = msg

                if assistant_response is None and not tool_calls_in_turn:
                    await self._emit(AgentEvent(
                        type="error",
                        content="Model returned no response."
                    ))
                    break

                # Append the final assistant response to history
                if assistant_response:
                    self._messages.append(assistant_response)

                    # — Check for terminal signal —
                    response_text = assistant_response.get("content", "")
                    if "SUBMISSION READY" in response_text:
                        await self._emit(AgentEvent(
                            type="status",
                            content="✅ Agent completed! Submission ready.",
                            metadata={"done": True},
                        ))
                        self._running = False
                        break

                # — Route tool calls —
                if tool_calls_in_turn:
                    for tool_msg in tool_calls_in_turn:
                        routed = await self._route_tool_call(tool_msg)
                        if not routed:
                            break
                else:
                    # No tool calls → agent is done with this turn
                    if turn >= 1:
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
        self._messages.append(user_message(text))
        await self._emit(AgentEvent(
            type="text",
            content=f"**User:** {text}",
            tool_name="user",
        ))
        await self.run()

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _emit(self, event: AgentEvent):
        await self.event_queue.put(event)

    async def _route_tool_call(self, tool_msg: dict) -> bool:
        """
        Execute a tool call identified in a harmony message.

        tool_msg has:
            role: "assistant"
            content: str  (the tool input / code to run)
            __tool__: str (tool name, e.g. "python", "browser", "file")

        Returns True if the tool was called successfully.
        """
        tool_name = tool_msg.get("__tool__", "")
        args_text = tool_msg.get("content", "")

        if not tool_name or tool_name not in self._tool_map:
            # Unknown tool — log and continue
            await self._emit(AgentEvent(
                type="error",
                content=f"Unknown tool requested: {tool_name!r}",
            ))
            return False

        tool = self._tool_map[tool_name]

        await self._emit(AgentEvent(
            type="tool_call",
            tool_name=tool_name,
            content=args_text,
        ))

        # Append the tool call to history as an assistant message
        self._messages.append(tool_msg)

        try:
            tool_responses = []
            async for response_msg in tool.process(tool_msg):
                tool_responses.append(response_msg)
                output = response_msg.get("content", "") if isinstance(response_msg, dict) else str(response_msg)
                await emit_tool_result(tool_name, output, self.event_queue)
                # Append tool result to history as a "tool" role message
                if isinstance(response_msg, dict):
                    self._messages.append(response_msg)
                else:
                    self._messages.append({"role": "tool", "content": output, "tool_name": tool_name})
            return bool(tool_responses)
        except Exception as e:
            err = f"[TOOL ERROR] {tool_name}: {e}"
            await self._emit(AgentEvent(type="error", content=err, tool_name=tool_name))
            # Inject error as a tool result so the model can react
            self._messages.append({
                "role": "tool",
                "content": err,
                "tool_name": tool_name,
            })
            return True  # Still counts as a tool turn — let model react to the error
