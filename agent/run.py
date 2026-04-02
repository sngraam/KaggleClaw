"""
agent/run.py — KaggleClaw agent loop.

Orchestrates the multi-turn conversation following Harmony history rules:

  • After a turn ending in a FINAL answer:
      → Drop all analysis messages from that turn.
      → Keep only the single final message in history.

  • After a turn ending in a TOOL CALL:
      → Keep the full chain: analysis + commentary call + tool result.
      → This is necessary so the model can continue its chain-of-thought.

  • Tool result messages are built with tool_result_message() which sets
    Author(TOOL, tool_name), channel="commentary", recipient="assistant".
"""

import asyncio
import os
from typing import Any

from .harmony import (
    AgentEvent,
    emit_tool_result,
    extract_text,
    stream_completion,
    tool_result_message,
    user_message,
)
from .prompt import build_messages

MODEL_NAME = os.environ.get("MODEL_NAME", "oss-120b")
MAX_TURNS  = 50


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
        self.model       = model
        self.tools       = tools or []
        self._tool_map: dict[str, Any] = {t.name: t for t in self.tools}
        self._messages: list = []
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list:
        return list(self._messages)

    def reset(self):
        self._messages = []
        self._running  = False

    async def run(self, initial_message: str | None = None):
        """
        Start (or continue) the agent loop.
        Injects system+developer messages on first run.
        """
        if not self._messages:
            # First run: inject properly structured system + developer messages
            self._messages.extend(build_messages())

        if initial_message:
            self._messages.append(user_message(initial_message))
            await self._emit(AgentEvent(
                type="status",
                content=f"Starting: {initial_message[:120]}"
            ))

        self._running = True
        turn = 0

        try:
            while self._running and turn < MAX_TURNS:
                turn += 1
                await self._emit(AgentEvent(type="status", content=f"Turn {turn}"))

                # ── Stream one completion ──────────────────────────────────────
                turn_messages: list = []

                async for msg in stream_completion(
                    messages=self._messages,
                    model=self.model,
                    event_queue=self.event_queue,
                ):
                    turn_messages.append(msg)

                if not turn_messages:
                    await self._emit(AgentEvent(type="error", content="Model returned nothing."))
                    break

                # ── Determine how this turn ended ──────────────────────────────
                # Find the last non-analysis message
                last_msg = turn_messages[-1]
                last_channel = getattr(last_msg, "channel", None) or "final"
                last_recipient = (
                    getattr(last_msg, "recipient", None) or
                    getattr(last_msg, "_recipient", None)
                )

                # Detect tool call: last message has a non-assistant recipient
                is_tool_call = (
                    last_recipient is not None
                    and str(last_recipient) not in ("assistant", "None", "", "NoneType")
                )

                # ── Update conversation history ────────────────────────────────
                if is_tool_call:
                    # TOOL CALL: keep all messages (analysis + call)
                    # The model needs the full chain to continue reasoning
                    self._messages.extend(turn_messages)
                else:
                    # FINAL ANSWER: drop analysis, keep only the final message
                    # Per docs: "drop any previous CoT content on subsequent sampling
                    # if the responses ended in a message to the final channel"
                    final_msgs = [
                        m for m in turn_messages
                        if getattr(m, "channel", None) == "final"
                    ]
                    if final_msgs:
                        self._messages.extend(final_msgs)
                    else:
                        # No final message found — keep last message anyway
                        self._messages.append(last_msg)

                # ── Check for completion signal ────────────────────────────────
                all_text = "\n".join(extract_text(m) for m in turn_messages)
                if "SUBMISSION READY" in all_text:
                    await self._emit(AgentEvent(
                        type="status",
                        content="✅ Competition solved! Submission ready.",
                        metadata={"done": True},
                    ))
                    self._running = False
                    break

                # ── Route tool calls ───────────────────────────────────────────
                if is_tool_call:
                    routed = await self._route_tool_call(last_msg)
                    if not routed:
                        await self._emit(AgentEvent(type="done", content="Turn complete."))
                        self._running = False
                else:
                    await self._emit(AgentEvent(type="done", content="Turn complete."))
                    self._running = False

        except asyncio.CancelledError:
            self._running = False
            await self._emit(AgentEvent(type="status", content="Agent cancelled."))
        except Exception as e:
            self._running = False
            await self._emit(AgentEvent(type="error", content=f"Agent loop error: {e}"))
            raise

    async def send_user_message(self, text: str):
        """Send a user message while agent is idle, then resume the loop."""
        msg = user_message(text)
        self._messages.append(msg)
        await self._emit(AgentEvent(type="text", content=text, tool_name="user"))
        await self.run()

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _emit(self, event: AgentEvent):
        await self.event_queue.put(event)

    async def _route_tool_call(self, msg) -> bool:
        """
        Execute the tool call encoded in msg and append the result to history.

        Returns True if a tool was called, False if the recipient was unknown.
        """
        recipient = (
            getattr(msg, "recipient", None) or
            getattr(msg, "_recipient", None)
        )
        if not recipient:
            return False

        recipient_str = str(recipient)

        # Built-in tools: python / browser.* — handled by the tool objects
        # Custom function tools: file, apply_patch (via functions.file etc.)
        # Strip "functions." prefix if present
        tool_key = recipient_str.removeprefix("functions.")

        # Also handle "python" and "browser.*" as built-in names
        if tool_key.startswith("browser."):
            tool_key = "browser"

        tool = self._tool_map.get(tool_key)
        if not tool:
            await self._emit(AgentEvent(
                type="error",
                content=f"Unknown tool: {recipient_str}",
                tool_name=recipient_str,
            ))
            return False

        args_text = extract_text(msg)

        await self._emit(AgentEvent(
            type="tool_call",
            tool_name=recipient_str,
            content=args_text,
        ))

        try:
            tool_responses = []
            async for response_msg in tool.process(msg):
                tool_responses.append(response_msg)
                output = extract_text(response_msg)
                await emit_tool_result(recipient_str, output, self.event_queue)

                # Build correct harmony tool result message and add to history
                result_msg = tool_result_message(recipient_str, output)
                self._messages.append(result_msg)

            return bool(tool_responses)

        except Exception as e:
            err = f"[TOOL ERROR] {recipient_str}: {e}"
            await self._emit(AgentEvent(type="error", content=err, tool_name=recipient_str))
            # Inject error as tool result so model knows what happened
            result_msg = tool_result_message(recipient_str, err)
            self._messages.append(result_msg)
            return True  # We attempted the call; let the model continue