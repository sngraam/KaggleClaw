"""
agent/tools/plan_follow/tool.py — Plan follower for KaggleClaw.

Reads plan.md, reports current step, marks steps done.
Keeps the agent on track through the competition workflow.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from openai_harmony import Author, Message, Role, TextContent

from agent.tool import Tool

# ── Path resolution ────────────────────────────────────────────────────────────

def _plan_path() -> Path:
    for p in ["/kaggle/working/KaggleClaw/plan.md", "./plan.md"]:
        path = Path(p)
        if path.exists():
            return path
    # Return default path for writing even if it doesn't exist yet
    return Path("./plan.md")


# ── Plan helpers ───────────────────────────────────────────────────────────────

def _read_plan() -> str:
    """Read the plan file or return a helpful message if not found."""
    path = _plan_path()
    if not path.exists():
        return (
            "[PLAN NOT FOUND] No plan.md exists yet.\n"
            "Create one with the 'file' tool: write plan.md\\n<your plan>\n"
            "Or ask the agent to generate a plan."
        )
    return path.read_text(encoding="utf-8")


def _find_current_step(plan_text: str) -> str:
    """Find the first unchecked step in the plan (lines with '- [ ]')."""
    lines = plan_text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*-\s*\[\s*\]\s+", line):
            return f"Line {i+1}: {line.strip()}"
    return "[ALL DONE] All steps in plan are marked complete."


def _mark_step_done(plan_text: str, step_text: str) -> tuple[str, str]:
    """
    Mark the first line matching step_text as done ([ ] -> [x]).
    Returns (new_plan_text, status_message).
    """
    lines = plan_text.splitlines()
    for i, line in enumerate(lines):
        if step_text.strip().lower() in line.lower() and "[ ]" in line:
            lines[i] = line.replace("[ ]", "[x]", 1)
            return "\n".join(lines), f"[OK] Marked done: {lines[i].strip()}"
    return plan_text, f"[NOT FOUND] Could not find step matching: {step_text}"


def _dispatch(raw: str) -> str:
    """Dispatch plan commands."""
    raw = raw.strip()

    if raw == "read" or raw == "":
        return _read_plan()

    if raw == "status":
        plan = _read_plan()
        if plan.startswith("[PLAN NOT FOUND]"):
            return plan
        current = _find_current_step(plan)
        total    = len(re.findall(r"\[\s*[ x]\s*\]", plan))
        done     = len(re.findall(r"\[\s*x\s*\]", plan))
        return f"Progress: {done}/{total} steps done\nNext step: {current}"

    if raw.startswith("mark_done "):
        step_text = raw[len("mark_done "):].strip()
        plan = _read_plan()
        if plan.startswith("[PLAN NOT FOUND]"):
            return plan
        new_plan, msg = _mark_step_done(plan, step_text)
        path = _plan_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_plan, encoding="utf-8")
        return msg

    return (
        "[ERROR] Unknown command.\n"
        "Available: read, status, mark_done <step description>\n"
        f"Got: {raw[:80]}"
    )


# ── Tool class ─────────────────────────────────────────────────────────────────

class PlanFollowTool(Tool):
    """Read and follow the plan.md competition workflow."""

    @property
    def name(self) -> str:
        return "plan_follow"

    def instruction(self) -> str:
        return """\
Use this tool to read and follow the current competition plan.

Commands:
  read                        — Read the full plan.md
  status                      — Show progress: how many steps done, what is next
  mark_done <step description> — Mark a step as complete in plan.md

The plan uses markdown checkboxes:
  - [ ] Uncompleted step
  - [x] Completed step

Always call `status` at the start of a new turn to know your next action.
After completing a step, call `mark_done <step text>` to update progress.
"""

    def _make_response(self, text: str, channel: str | None = None) -> Message:
        return Message(
            id=uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=[TextContent(text=text)],
            channel=channel,
        ).with_recipient("assistant")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        raw = ""
        if message.content:
            c = message.content
            if isinstance(c, list):
                raw = c[0].text.strip() if hasattr(c[0], "text") else str(c[0])
            elif hasattr(c, "text"):
                raw = c.text.strip()
            else:
                raw = str(c).strip()

        try:
            import json
            if raw.startswith("{"):
                parsed = json.loads(raw)
                if "command" in parsed:
                    raw = parsed["command"].strip()
        except Exception:
            pass

        try:
            result = _dispatch(raw)
        except Exception as exc:
            result = f"[ERROR] plan_follow failed: {type(exc).__name__}: {exc}"

        yield self._make_response(result, channel=message.channel)
