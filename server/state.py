"""
server/state.py — Global conversation state for KaggleClaw.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    """Singleton state shared across all FastAPI routes."""

    # SSE event queue — AgentEvent objects
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Serializable conversation history for the frontend
    conversation: list[dict[str, Any]] = field(default_factory=list)

    # Agent runner instance (set on startup)
    runner: Any = None

    # Background task handle for the running agent
    agent_task: asyncio.Task | None = None

    # ngrok public URL
    public_url: str = ""

    # vLLM local model server reference
    vllm_server: Any = None

    # vllm process handle for clean shutdowns + vllm starting state
    vllm_process: Any = None
    vllm_starting: bool = False

    def reset(self):
        self.conversation = []
        self.event_queue = asyncio.Queue()
        if self.runner:
            self.runner.reset()
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        self.agent_task = None


# Global singleton
state = AppState()
