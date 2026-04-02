"""
server/app.py — KaggleClaw FastAPI server.

Endpoints:
  GET  /            — serves frontend/index.html
  GET  /stream      — SSE stream of agent events
  POST /start       — start the agent on the competition
  POST /chat        — send a message to the agent
  POST /reset       — reset conversation
  GET  /health      — status + public URL
  GET  /competition — competition info for sidebar
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .state import state
from agent.prompt import build_competition_summary

app = FastAPI(title="KaggleClaw", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static frontend files
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize the agent runner on startup."""
    _init_runner()


def _init_runner():
    """Create agent runner with all tools."""
    from agent.run import AgentRunner
    from agent.tools import get_all_tools
    from agent.browser import ExaBackend

    # Browser backend (optional — needs EXA_API_KEY or YDC_API_KEY)
    browser_backend = None
    if os.environ.get("EXA_API_KEY"):
        browser_backend = ExaBackend(source="Exa Search")

    # Jupyter connection file (set by Kaggle kernel environment)
    jupyter_conn = os.environ.get("PYTHON_LOCAL_JUPYTER_CONNECTION_FILE")

    tools = get_all_tools(
        browser_backend=browser_backend,
        jupyter_connection_file=jupyter_conn,
    )

    state.runner = AgentRunner(
        event_queue=state.event_queue,
        tools=tools,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend SPA."""
    index_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>KaggleClaw</h1><p>Frontend not found. Check frontend/ directory.</p>")


@app.get("/health")
async def health():
    return {
        "status": "running",
        "public_url": state.public_url,
        "agent_running": state.runner._running if state.runner else False,
        "turn_count": len(state.runner.messages) if state.runner else 0,
    }


@app.get("/competition")
async def competition_info():
    """Return structured competition info for the frontend sidebar."""
    return build_competition_summary()


@app.get("/stream")
async def stream(request: Request):
    """SSE endpoint — streams AgentEvents to the frontend."""

    async def generator():
        # Re-send conversation history on connect
        for event_dict in state.conversation:
            yield {"data": json.dumps(event_dict)}
            await asyncio.sleep(0)

        # Then stream live events
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(state.event_queue.get(), timeout=30.0)
                payload = {
                    "type": event.type,
                    "content": event.content,
                    "tool_name": event.tool_name,
                    "metadata": event.metadata,
                }
                # Persist to conversation history
                state.conversation.append(payload)
                yield {"data": json.dumps(payload)}
            except asyncio.TimeoutError:
                # keepalive ping
                yield {"data": json.dumps({"type": "ping"})}

    return EventSourceResponse(generator())


@app.post("/start")
async def start_agent(request: Request):
    """Start the agent on the current competition."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    message = body.get("message", "Begin solving the competition described in competition.md. Take full ownership.")

    if not state.runner:
        _init_runner()

    if state.agent_task and not state.agent_task.done():
        return JSONResponse({"error": "Agent already running"}, status_code=409)

    state.agent_task = asyncio.create_task(
        state.runner.run(initial_message=message)
    )
    return {"status": "started", "message": message}


@app.post("/chat")
async def chat(request: Request):
    """Send a user message to the agent."""
    body = await request.json()
    message = body.get("message", "").strip()

    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    if not state.runner:
        _init_runner()

    # If agent is idle, start a new turn
    state.agent_task = asyncio.create_task(
        state.runner.send_user_message(message)
    )
    return {"status": "sent", "message": message}


@app.post("/reset")
async def reset():
    """Reset the conversation and agent state."""
    state.reset()
    _init_runner()
    return {"status": "reset"}
