"""
agent/tools/web_search/tool.py — Free web search for KaggleClaw.

Uses DuckDuckGo Instant Answer API (no key needed) with duckduckgo_search
library as a richer fallback. Gracefully handles network failures.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import AsyncIterator
from uuid import uuid4

from openai_harmony import Author, Message, Role, TextContent

from ..tool import Tool

# ── DuckDuckGo helpers ─────────────────────────────────────────────────────────

_DDG_INSTANT = "https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
_MAX_RESULTS  = 5
_TIMEOUT      = 10  # seconds


def _ddg_instant(query: str) -> list[dict]:
    """Hit the DDG Instant Answer API — returns structured results."""
    url = _DDG_INSTANT.format(q=urllib.parse.quote_plus(query))
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []

    results: list[dict] = []

    # Abstract (top summary)
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query),
            "snippet": data["AbstractText"],
            "url": data.get("AbstractURL", ""),
        })

    # Related topics
    for topic in data.get("RelatedTopics", []):
        if len(results) >= _MAX_RESULTS:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })

    return results


def _ddg_library(query: str) -> list[dict]:
    """Use duckduckgo_search library for richer results (optional dep)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=_MAX_RESULTS):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
        return results
    except Exception:
        return []


def _search(query: str) -> str:
    """Run search and format top results as readable text."""
    results = _ddg_library(query)
    if not results:
        results = _ddg_instant(query)

    if not results:
        return f"[NO RESULTS] Could not retrieve results for: {query}"

    lines = [f"🔍 Search results for: {query}\n"]
    for i, r in enumerate(results[:_MAX_RESULTS], 1):
        title   = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        url     = r.get("url", "").strip()
        lines.append(f"{i}. **{title}**")
        if snippet:
            lines.append(f"   {snippet[:300]}")
        if url:
            lines.append(f"   {url}")
        lines.append("")

    return "\n".join(lines).strip()


# ── Tool class ─────────────────────────────────────────────────────────────────

class WebSearchTool(Tool):
    """Free DuckDuckGo web search — no API key required."""

    @property
    def name(self) -> str:
        return "web_search"

    def instruction(self) -> str:
        return """\
Use this tool to search the web for information relevant to the Kaggle competition.

Send a plain text search query. The tool will return up to 5 result snippets with titles and URLs.

Examples:
  XGBoost hyperparameter tuning tips
  Kaggle titanic top solutions feature engineering
  LightGBM vs CatBoost binary classification benchmark

Use this to:
- Research domain knowledge for the competition
- Find top public notebooks and discussions
- Look up ML techniques, papers, and implementations
"""

    def _make_response(self, text: str, channel: str | None = None) -> Message:
        return Message(
            id=uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=[TextContent(text=text)],
            channel=channel,
        ).with_recipient("assistant")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        query = ""
        if message.content:
            raw = message.content
            if isinstance(raw, list):
                query = raw[0].text.strip() if hasattr(raw[0], "text") else str(raw[0])
            elif hasattr(raw, "text"):
                query = raw.text.strip()
            else:
                query = str(raw).strip()

        if not query:
            yield self._make_response("[ERROR] Empty search query.", channel=message.channel)
            return

        try:
            result = await _run_in_thread(_search, query)
        except Exception as exc:
            result = f"[ERROR] Search failed: {exc}"

        yield self._make_response(result, channel=message.channel)


# ── Thread helper ──────────────────────────────────────────────────────────────

import asyncio
import functools


async def _run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args))
