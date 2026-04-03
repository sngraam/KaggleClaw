import json
from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal["thinking", "text", "tool_call", "tool_result", "error", "done", "status"]

@dataclass
class AgentEvent:
    """A single event pushed to the SSE stream."""
    type: EventType
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return (
            f"data: {json.dumps({'type': self.type, 'content': self.content, 'tool_name': self.tool_name, 'metadata': self.metadata})}\n\n"
        )
