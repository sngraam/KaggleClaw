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


EventType = Literal["thinking", "text", "tool_call", "tool_result", "error", "done", "status"]


@dataclass
class AgentEvent:
    type: EventType
    content: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, 'content': self.content, 'tool_name': self.tool_name, 'metadata': self.metadata})}\n\n"


class HarmoTemplate:

    def __init__(self):

        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:

        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]
    
from datetime import date
from pathlib import Path

WORKING_DIR = "/kaggle/working/KaggleClaw"
RUN_DIR     = "/kaggle/working/KaggleClaw/run"
INPUT_DIR   = "/kaggle/input/notebooks/sangrampatil5150/notebook38b92fed30/"


def _read_safe(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return "[file not found]"


def _load_competition_context() -> str:
    for p in [f"{WORKING_DIR}/competition.md", "competition.md"]:
        content = _read_safe(p)
        if content != "[file not found]":
            # Prevent context length overflows if user pastes massive text
            if len(content) > 5000:
                return content[:5000] + "\n\n... [TRUNCATED] ... \nWARNING: competition.md was too long. Consider using file or web tools to read specifics."
            return content
    return "[competition.md not found — please create it]"


def build_messages() -> list:
    """
    Build the initial [system, developer] message list for a new agent session.

    Returns a list of Message objects ready to be passed to render_conversation().
    """
    try:
        from openai_harmony import (
            Message, Role, SystemContent, DeveloperContent,
            ReasoningEffort, ToolNamespaceConfig, ToolDescription,
        )
    except ImportError:
        # Fallback: plain text messages if library not available
        return _build_plain_messages()

    competition_ctx = _load_competition_context()
    today = date.today().isoformat()

    # ── System message ─────────────────────────────────────────────────────────
    # Must contain: identity, dates, reasoning, channels, built-in tools
    system_content = (
        SystemContent.new()
        .with_model_identity(
            "You are KaggleClaw, an elite autonomous ML agent built to win Kaggle competitions."
        )
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date(today)
        .with_python_tool()
        .with_browser_tool()
    )

    system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)

    # ── Developer message ──────────────────────────────────────────────────────
    # This is the actual "system prompt" — instructions + file/patch tools
    instructions = _build_instructions(competition_ctx)

    # Custom tools: file and apply_patch go here as function tools
    file_tool = ToolDescription.new(
        "file",
        "Read, write, list, delete, move files in /kaggle/working/. "
        "Commands: read <path>, write <path>\\n<content>, list [path], "
        "delete <path>, mkdir <path>, exists <path>, move <src> <dst>",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Full command string, e.g. 'read /kaggle/working/KaggleClaw/submission.csv' "
                        "or 'write /kaggle/working/model.py\\nimport pandas...'"
                    ),
                }
            },
            "required": ["command"],
        },
    )

    patch_tool = ToolDescription.new(
        "apply_patch",
        "Apply a unified diff patch to an existing file in /kaggle/working/KaggleClaw/. "
        "Use for surgical edits without rewriting the whole file.",
        parameters={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "A valid unified diff patch string",
                }
            },
            "required": ["patch"],
        },
    )

    web_search_tool = ToolDescription.new(
        "web_search",
        "Search the web for information. Send a plain text query, get back top-5 results with titles, snippets, and URLs. "
        "No API key required. Use for researching competition domain, top notebooks, ML techniques, and error messages.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The search query, e.g. 'Kaggle fraud detection top solutions XGBoost'",
                }
            },
            "required": ["command"],
        },
    )

    plan_follow_tool = ToolDescription.new(
        "plan_follow",
        "Read and follow the competition plan. Commands: (if plan.md exists) "
        "'read' — show full plan.md, "
        "'status' — show progress and next step, "
        "'mark_done <step description>' — mark a completed step as done.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "One of: read | status | mark_done <step text>",
                }
            },
            "required": ["command"],
        },
    )

    developer_content = (
        DeveloperContent.new()
        .with_instructions(instructions)
        .with_function_tools([file_tool, patch_tool, web_search_tool, plan_follow_tool])
    )

    developer_msg = Message.from_role_and_content(Role.DEVELOPER, developer_content)

    return [system_msg, developer_msg]


def _build_instructions(competition_ctx: str) -> str:
    return f"""You are KaggleClaw — an elite, autonomous machine learning agent built to win Kaggle competitions.
You have full authority to use all available tools without asking for permission.

═══════════════════════════════════════
COMPETITION CONTEXT
═══════════════════════════════════════
{competition_ctx}

═══════════════════════════════════════
WORKING ENVIRONMENT
═══════════════════════════════════════
- Working directory : {WORKING_DIR}/
- Experiments dir   : {RUN_DIR}/    (disposable experiments)
- Dataset dir       : {INPUT_DIR}/  (read-only — competition data) (train.parquet and test.parquet)
- Metrics evaluator : {WORKING_DIR}/metrics.py
  Call evaluate(y_true, y_pred) or score_submission(...) to self-assess.

═══════════════════════════════════════
YOUR TOOLS
═══════════════════════════════════════
1. python       — Execute Python in a persistent Jupyter kernel.
                  Variables and state persist across calls.
                  Use for EDA, training, evaluation, submission generation.

2. browser      — Browse the web. Research techniques, papers, Kaggle discussions,
                  top public notebooks for this competition.

3. file         — Read/write/list/delete files in /kaggle/working/.
                  (Custom function tool — calls go to commentary channel)

4. apply_patch  — Apply unified diffs to files. Surgical edits without rewrite.
                  (Custom function tool — calls go to commentary channel)

═══════════════════════════════════════
HOW TO WIN — AGENTIC LOOP
═══════════════════════════════════════
STEP 1 — UNDERSTAND
  • Read competition.md. Understand the metric (higher vs lower is better).
  • Use python to inspect the dataset: shape, dtypes, head, missing values.
  • Use browser to research the competition, top notebooks, discussions.

STEP 2 — BASELINE
  • Build a quick baseline (logistic regression, mean predictor, decision tree).
  • Save submission to /kaggle/working/submission_baseline.csv.
  • Score it with metrics.py. Know your starting point.

STEP 3 — ITERATE
  • Try better features and models (XGBoost, LightGBM, CatBoost, neural nets).
  • Cross-validate properly (StratifiedKFold, time-based split as appropriate).
  • Score every iteration. Only keep changes that improve the metric.
  • Name scripts clearly: lgbm_v1.py, feature_eng_v2.py, etc.

STEP 4 — OPTIMIZE
  • Hyperparameter tuning (Optuna preferred).
  • Domain-specific features — use browser to research domain knowledge.
  • Ensemble/stack top models.

STEP 5 — FINALIZE
  • Write final submission to /kaggle/working/submission_final.csv.
  • Verify the format matches sample_submission.csv exactly.
  • Print your best CV score.
  • End your response with:
    SUBMISSION READY: /kaggle/working/submission_final.csv — CV Score: X.XXXX

═══════════════════════════════════════
REASONING & BEHAVIOUR
═══════════════════════════════════════
- Think carefully in your analysis channel before acting.
- After every tool result, interpret what you learned and plan the next step.
- Debug failures and retry. Be iterative.
- Never hallucinate data — always verify with actual tool calls.
- Prefer correctness over speed, but keep moving.
"""


def _build_plain_messages() -> list:
    """Fallback plain-text messages when openai_harmony is not installed."""
    try:
        from openai_harmony import Message, Role
    except ImportError:
        # Use stubs from harmony.py
        from .harmony import user_message
        # Just return a single user-style context injection
        ctx = _load_competition_context()
        return [user_message(f"[SYSTEM] {_build_instructions(ctx)}")]

    competition_ctx = _load_competition_context()
    return [
        Message.from_role_and_content(Role.SYSTEM, _build_instructions(competition_ctx))
    ]


# ── Frontend sidebar helper ────────────────────────────────────────────────────

def build_competition_summary() -> dict:
    """Return structured competition info for the frontend sidebar."""
    ctx = _load_competition_context()
    lines = ctx.splitlines()

    def extract_field(label: str) -> str:
        for i, line in enumerate(lines):
            if line.strip().startswith(f"## {label}"):
                for j in range(i + 1, min(i + 6, len(lines))):
                    l = lines[j].strip()
                    if l and not l.startswith("<!--") and not l.startswith("```"):
                        return l
        return "—"

    return {
        "name":     extract_field("Name"),
        "metric":   extract_field("Evaluation Metric"),
        "task":     extract_field("Task Type"),
        "target":   extract_field("Target Column"),
        "deadline": extract_field("Deadline"),
        "url":      extract_field("URL"),
    }