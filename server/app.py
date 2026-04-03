"""
server/app.py — KaggleClaw FastAPI server.

Endpoints:
  GET  /            — serves frontend/index.html
  GET  /stream      — SSE stream of agent events (with keep-alive pings)
  POST /start       — start the agent on the competition
  POST /chat        — send a message to the agent
  POST /reset       — reset conversation
  GET  /health      — status + public URL
  GET  /logs        — last N terminal log lines (JSON)
  POST /host_model  — triggers local vLLM hosting
  POST /stop_hosting — gracefully terminates local vLLM server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from collections import deque

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .state import state
from agent.harmo import build_competition_summary

# ── In-memory log handler ──────────────────────────────────────────────────────

class _RingLogHandler(logging.Handler):
    """Stores up to `maxlen` formatted log records in a ring-buffer."""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self._buf: deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        try:
            self._buf.append(self.format(record))
        except Exception:
            pass

    def get_lines(self, n: int = 200) -> list[str]:
        lines = list(self._buf)
        return lines[-n:] if n < len(lines) else lines


_ring_handler = _RingLogHandler(maxlen=500)
_ring_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

# Attach to root logger so we capture everything
_root_logger = logging.getLogger()
_root_logger.addHandler(_ring_handler)
_root_logger.setLevel(logging.INFO)


# ── App setup ──────────────────────────────────────────────────────────────────

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


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse({"error": "Invalid request", "detail": str(exc)}, status_code=422)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger("kaggleclaw.server").exception("Unhandled server error")
    return JSONResponse({"error": "Internal server error", "detail": str(exc)}, status_code=500)


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize the agent runner on startup."""
    _init_runner()


def _init_runner():
    """Create agent runner with all tools."""
    from agent.run import AgentRunner
    from agent.tools import get_all_tools

    tools = get_all_tools()

    model    = os.environ.get("VLLM_MODEL", "open-scorer-120b")
    base_url = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8080/v1")

    state.runner = AgentRunner(
        event_queue=state.event_queue,
        tools=tools,
        model=model,
        base_url=base_url,
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


@app.get("/files")
async def list_files():
    """Return a JSON directory tree of the Kaggle working directory for the file sidebar."""
    def _build_tree(path: Path, max_depth: int = 3, depth: int = 0):
        if depth > max_depth:
            return []
        nodes = []
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return []
        for entry in entries:
            if entry.name.startswith('.') or entry.name == '__pycache__':
                continue
            if entry.is_dir():
                nodes.append({
                    "type": "dir",
                    "name": entry.name,
                    "children": _build_tree(entry, max_depth, depth + 1),
                })
            else:
                nodes.append({"type": "file", "name": entry.name})
        return nodes

    working_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    if not working_dir.exists():
        working_dir = Path(__file__).parent.parent

    return {"base": str(working_dir), "tree": _build_tree(working_dir)}


@app.get("/logs")
async def get_logs(n: int = 200):
    """Return last N terminal log lines from the in-memory ring buffer."""
    lines = _ring_handler.get_lines(n)
    return {"lines": lines, "count": len(lines)}


# ── SSE Stream ─────────────────────────────────────────────────────────────────

@app.get("/stream")
async def stream_agent(request: Request):
    """
    SSE endpoint. Yields agent events from the shared queue.
    Handles:
      - Normal event flow (thinking, text, tool_call, tool_result, error, status, done)
      - Keep-alive pings every PING_INTERVAL seconds (handled by AgentRunner._emit_ping)
      - Clean disconnect when client drops
    """
    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait up to 20s for a new event; if nothing arrives, loop back
                    event = await asyncio.wait_for(state.event_queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # Queue empty — yield a heartbeat comment to keep TCP alive
                    yield ": heartbeat\n\n"
                    continue

                if hasattr(event, "to_sse"):
                    yield event.to_sse()
                else:
                    yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming once the agent signals completion
                if getattr(event, "type", None) == "done" or (
                    isinstance(event, dict) and event.get("type") == "done"
                ):
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return EventSourceResponse(event_generator())


# ── Agent control routes ───────────────────────────────────────────────────────

@app.post("/start")
async def start_agent(request: Request):
    """Start the agent on the current competition."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = body.get(
        "message",
        "Begin solving the competition described in competition.md. Take full ownership.",
    )

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

    state.agent_task = asyncio.create_task(
        state.runner.send_user_message(message)
    )
    return {"status": "sent", "message": message}


@app.post("/reset")
async def reset():
    """Reset the conversation and agent state."""
    # Cancel running agent task cleanly before reinitializing
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(state.agent_task), timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    state.reset()
    _init_runner()
    return {"status": "reset"}


# ── Model hosting ──────────────────────────────────────────────────────────────

@app.post("/host_model")
async def host_model():
    """Start local vLLM server via models/host.py."""
    if state.vllm_server:
        return {"status": "already_hosting"}

    if state.vllm_starting:
        return {"status": "starting"}

    state.vllm_starting = True

    from fastapi.concurrency import run_in_threadpool
    from models.host import Server, ServerConfig

    async def _start():
        try:
            cfg = ServerConfig()
            state.vllm_server = Server(cfg=cfg, port=8080)
        except Exception as e:
            logging.getLogger("kaggleclaw.server").error(f"vLLM start failed: {e}")
        finally:
            state.vllm_starting = False

    asyncio.create_task(run_in_threadpool(lambda: asyncio.run(_start())))
    return {"status": "hosting_started"}


@app.post("/stop_hosting")
async def stop_hosting():
    """Stop the local vLLM server if it is running."""
    import subprocess
    if state.vllm_server:
        try:
            if hasattr(state.vllm_server, "server_process") and state.vllm_server.server_process:
                proc = state.vllm_server.server_process
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        except Exception as e:
            logging.getLogger("kaggleclaw.server").error(f"Error stopping vLLM: {e}")
        finally:
            state.vllm_server = None
            state.vllm_starting = False
            return {"status": "hosting_stopped"}
    return {"status": "not_hosting"}
